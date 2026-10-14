"""
Minimal FastAPI service exposing the pipeline outputs to the dashboard.

Endpoints:
  GET  /api/health          -> liveness check (+ thresholds shown in the UI)
  GET  /api/hotspots        -> ranked, geolocated review priorities (GeoJSON):
                                historical CSV + demo pipeline events +
                                evidence from videos analysed at a location
  GET  /api/report          -> the recommendation report as Markdown
  POST /api/pipeline/run    -> re-runs the full pipeline on the sample/demo
                                video + accident CSV and regenerates outputs
  POST /api/analyze-video   -> upload ANY video, get a direct near-miss
                                verdict for just that clip (synchronous;
                                optional location fields as below)
  POST /api/analyses        -> same analysis as a background job with real
                                progress; form fields: file, and optionally
                                site_id OR lat + lon (+ location_name)
  GET  /api/analyses/{id}/status      -> job progress / result
  GET  /api/analyses/{id}             -> stored analysis (analysis.json)
  GET  /api/analyses/{id}/report.pdf  -> evidence report PDF (?event=N)
  POST /api/analyses/{id}/email       -> e-mail that same PDF {to, subject, event}
  GET  /api/hotspots/{site_id}/report.pdf -> site review brief PDF
  POST /api/hotspots/{site_id}/email      -> e-mail that same PDF {to, subject}
  GET  /api/email/status    -> whether e-mail delivery is configured (no secrets)
  POST /api/assistant       -> Road Safety Assistant: grounded answer {question, site_id?, analysis_id?}
  DELETE /api/observations  -> clear uploaded-video evidence and map pins
  GET  /outputs/...         -> static files (evidence frames, annotated clips)

This is intentionally unauthenticated and single-user — see
docs/architecture.md "Production deployment notes" for what a real
deployment would add (auth, a job queue for pipeline runs, file-size /
content-type limits on uploads, etc.).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, envfile

envfile.load_env(config.ROOT_DIR / ".env")  # server-side settings (e.g. SMTP); never sent to the browser

from . import analysis_service, assistant, detection_yolo, hotspot_insight, mailer, report, site_evidence  # noqa: E402
from .evidence_report import build_event_pdf  # noqa: E402
from .pipeline import run_full_pipeline  # noqa: E402
from .site_report import build_site_pdf  # noqa: E402

app = FastAPI(
    title="Verge — Road-Risk API",
    description="AI road-risk and near-miss intelligence — decision support, not automated fault assignment.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo scaffold only — restrict this in production
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Needed before mount(): StaticFiles requires the directory to exist, and
# this may be the very first thing the app does (e.g. someone hits
# /api/analyze-video before ever running the demo pipeline).
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
config.ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")

# Videos we'll actually accept via upload — keeps someone from pointing
# this at, say, a .exe. Not a security boundary on its own (see
# docs/architecture.md), just a sane guardrail for a demo endpoint.
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "thresholds": {
            "distance_m": config.DIST_THRESH_M,
            "ttc_s": config.TTC_THRESH_S,
            "pixels_per_meter": config.PIXELS_PER_METER,
        },
        "historical_data_label": config.HISTORICAL_DATA_LABEL,
        "object_detector_available": detection_yolo.available(),
        # review-priority bands (lower bound of each tier on the 0-1 score)
        "risk_tiers": [{"tier": label, "min_score": t} for t, label in config.RISK_TIERS],
    }


@app.get("/api/hotspots")
def get_hotspots() -> JSONResponse:
    try:
        data = site_evidence.current_geojson()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Accident data not found: {exc}") from exc
    return JSONResponse(content=data)


@app.get("/api/report", response_class=PlainTextResponse)
def get_report(download: bool = False):
    text = report.generate_report(site_evidence.current_hotspots())
    if download:
        return Response(
            content=text.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="verge-review-priorities.md"'},
        )
    return text


@app.post("/api/pipeline/run")
def trigger_pipeline() -> dict:
    """
    Re-runs the pipeline against the demo/sample video and accident data.
    In a real deployment this would accept an uploaded video / camera feed
    reference and run as a background job, not synchronously.
    """
    if not config.SAMPLE_VIDEO_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "No demo video found. Run `python scripts/generate_sample_video.py` "
                "first, or wire in a real video path."
            ),
        )
    summary = run_full_pipeline()
    return {"status": "ok", **summary}


async def _save_upload(file: UploadFile) -> Path:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No video file was attached.")
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}",
        )

    dest = config.UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    too_large = False
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                too_large = True
                break
            out.write(chunk)

    if too_large:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Video too large (max 200 MB for this demo).")
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return dest


def _location(site_id: Optional[str], lat: Optional[str], lon: Optional[str], name: Optional[str]) -> Optional[dict]:
    try:
        lat_f = float(lat) if lat not in (None, "") else None
        lon_f = float(lon) if lon not in (None, "") else None
        return analysis_service.resolve_location(site_id or None, lat_f, lon_f, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _detector(value: Optional[str]) -> str:
    value = (value or "auto").strip().lower()
    if value not in ("auto", "object", "motion"):
        raise HTTPException(status_code=400, detail="detector must be 'auto', 'object' or 'motion'.")
    return value


@app.post("/api/analyze-video")
async def analyze_video_upload(
    file: UploadFile = File(...),
    site_id: Optional[str] = Form(None),
    lat: Optional[str] = Form(None),
    lon: Optional[str] = Form(None),
    location_name: Optional[str] = Form(None),
    detector: Optional[str] = Form(None),
) -> dict:
    """
    Upload a single video and get back a direct near-miss verdict for just
    that clip — runs detection -> tracking -> conflict analysis with no
    accident CSV or site catalogue required. This is separate from POST
    /api/pipeline/run, which is the multi-site historical-hotspot workflow.
    Optionally link it to a location (site_id, or lat/lon) to add the
    observed evidence to that site's review priority.
    """
    det = _detector(detector)
    location = _location(site_id, lat, lon, location_name)
    dest = await _save_upload(file)
    try:
        return analysis_service.run_analysis(dest, Path(file.filename).name, location, detector=det)
    except Exception as exc:  # noqa: BLE001 — surface as a clean 400, not a 500 traceback
        raise HTTPException(status_code=400, detail=analysis_service.friendly_error(exc)) from exc
    finally:
        dest.unlink(missing_ok=True)  # raw footage is never retained


@app.post("/api/analyses")
async def start_analysis(
    file: UploadFile = File(...),
    site_id: Optional[str] = Form(None),
    lat: Optional[str] = Form(None),
    lon: Optional[str] = Form(None),
    location_name: Optional[str] = Form(None),
    detector: Optional[str] = Form(None),
) -> dict:
    det = _detector(detector)
    location = _location(site_id, lat, lon, location_name)
    dest = await _save_upload(file)
    analysis_id = analysis_service.start_job(dest, Path(file.filename).name, location, detector=det)
    return {"analysis_id": analysis_id, "status_url": f"/api/analyses/{analysis_id}/status"}


@app.get("/api/analyses/{analysis_id}/status")
def analysis_status(analysis_id: str) -> dict:
    job = analysis_service.job_status(analysis_id)
    if job is not None:
        return job
    try:  # e.g. the API restarted after the analysis finished
        result = analysis_service.load_analysis(analysis_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Unknown analysis.") from exc
    return {"analysis_id": analysis_id, "state": "done", "stages": [], "error": None, "result": result}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict:
    try:
        return analysis_service.load_analysis(analysis_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Unknown analysis.") from exc


@app.get("/api/analyses/{analysis_id}/report.pdf")
def get_evidence_pdf(analysis_id: str, event: int = Query(0, ge=0)) -> Response:
    try:
        analysis = analysis_service.load_analysis(analysis_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Unknown analysis.") from exc
    try:
        pdf = build_event_pdf(analysis, event)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not generate the evidence report: {exc}") from exc
    filename = f"verge-evidence-{analysis_id}-event{event + 1}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class EmailRequest(BaseModel):
    to: str
    subject: str = ""
    event: int = 0


class AssistantRequest(BaseModel):
    question: str
    site_id: Optional[str] = None
    analysis_id: Optional[str] = None


def _pdf_response(pdf: bytes, filename: str) -> Response:
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def _paragraphs(parts: list) -> str:
    """Plain-text e-mail body: one paragraph per part."""
    return "\n\n".join(p for p in parts if p) + "\n"


def _hotspot(site_id: str) -> tuple[dict, int]:
    sites = site_evidence.current_hotspots()
    h = next((x for x in sites if x["site_id"] == site_id), None)
    if h is None:
        raise HTTPException(status_code=404, detail="Unknown monitored location.")
    return h, len(sites)


def _send(to: str, subject: str, body: str, filename: str, pdf: bytes) -> dict:
    try:
        return mailer.send_report(to, subject, body, filename, pdf)
    except mailer.MailError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@app.get("/api/hotspots/{site_id}/report.pdf")
def get_site_pdf(site_id: str) -> Response:
    h, n = _hotspot(site_id)
    return _pdf_response(build_site_pdf(h, n), hotspot_insight.site_report_filename(h))


@app.post("/api/hotspots/{site_id}/email")
def email_site_pdf(site_id: str, req: EmailRequest) -> dict:
    h, n = _hotspot(site_id)
    pdf = build_site_pdf(h, n)  # the same bytes the download returns
    body = _paragraphs([
        f"Verge road-safety site review brief: {h['name']}",
        h["summary"],
        "The full brief is attached as a PDF. Decision support for human road-safety review; "
        "not a finding of fault or a crash prediction.",
    ])
    subject = req.subject or f"Verge Road Safety Report — {h['name']}"
    return _send(req.to, subject, body, hotspot_insight.site_report_filename(h), pdf)


@app.post("/api/analyses/{analysis_id}/email")
def email_evidence_pdf(analysis_id: str, req: EmailRequest) -> dict:
    try:
        analysis = analysis_service.load_analysis(analysis_id)
        pdf = build_event_pdf(analysis, req.event)  # the same PDF as /report.pdf
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Unknown analysis or event.") from exc
    loc = (analysis.get("location") or {}).get("name")
    label = loc or analysis.get("source_name") or f"analysis {analysis_id}"
    filename = f"verge-road-safety-report-{hotspot_insight.slug(label)}-event{req.event + 1}.pdf"
    body = _paragraphs([
        "Verge AI near-miss evidence report",
        analysis.get("verdict", ""),
        f"Location: {loc or 'not linked to a location'}",
        f"Source clip: {analysis.get('source_name', '-')}",
        "The evidence report is attached as a PDF. Decision support for human road-safety review; "
        "not a finding of fault.",
    ])
    subject = req.subject or f"Verge Road Safety Report — {label}"
    return _send(req.to, subject, body, filename, pdf)


@app.get("/api/email/status")
def email_status() -> dict:
    return mailer.status()


@app.post("/api/assistant")
def ask_assistant(req: AssistantRequest) -> dict:
    return assistant.answer(req.question[:500], req.site_id, req.analysis_id)


@app.delete("/api/observations")
def clear_observations() -> dict:
    return {"status": "ok", **site_evidence.clear_observations()}
