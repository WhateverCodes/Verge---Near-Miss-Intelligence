"""
Minimal FastAPI service exposing the pipeline outputs to the dashboard.

Endpoints:
  GET  /api/health          -> simple liveness check
  GET  /api/hotspots        -> hotspots.geojson (ranked, geolocated risk scores)
  GET  /api/report          -> report.md as plain text/markdown
  POST /api/pipeline/run    -> re-runs the full pipeline on the sample/demo
                                video + accident CSV and regenerates outputs
  POST /api/analyze-video   -> upload ANY video, get a direct near-miss /
                                "was a crash about to happen" verdict for
                                just that clip (no accident CSV needed)
  GET  /outputs/...         -> static files (annotated videos, reports)

This is intentionally unauthenticated and single-user — see
docs/architecture.md "Production deployment notes" for what a real
deployment would add (auth, a job queue for pipeline runs, file-size /
content-type limits on uploads, etc.).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .pipeline import run_full_pipeline
from .video_analysis import analyze_video

app = FastAPI(
    title="Verge — Road-Risk API",
    description="AI road-risk and near-miss intelligence — decision support, not automated fault assignment.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo scaffold only — restrict this in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Needed before mount(): StaticFiles requires the directory to exist, and
# this may be the very first thing the app does (e.g. someone hits
# /api/analyze-video before ever running the demo pipeline).
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")

# Videos we'll actually accept via upload — keeps someone from pointing
# this at, say, a .exe. Not a security boundary on its own (see
# docs/architecture.md), just a sane guardrail for a demo endpoint.
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/hotspots")
def get_hotspots() -> JSONResponse:
    if not config.HOTSPOTS_GEOJSON.exists():
        raise HTTPException(
            status_code=404,
            detail="No hotspots yet — run POST /api/pipeline/run or `python scripts/run_pipeline.py` first.",
        )
    data = json.loads(config.HOTSPOTS_GEOJSON.read_text())
    return JSONResponse(content=data)


@app.get("/api/report", response_class=PlainTextResponse)
def get_report() -> str:
    if not config.REPORT_MD.exists():
        raise HTTPException(
            status_code=404,
            detail="No report yet — run POST /api/pipeline/run or `python scripts/run_pipeline.py` first.",
        )
    return config.REPORT_MD.read_text()


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


@app.post("/api/analyze-video")
async def analyze_video_upload(file: UploadFile = File(...)) -> dict:
    """
    Upload a single video and get back a direct near-miss / "was a crash
    about to happen" verdict for just that clip — runs detection ->
    tracking -> conflict analysis with no accident CSV or site catalogue
    required. This is separate from POST /api/pipeline/run, which is the
    multi-site historical-hotspot workflow.
    """
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_SUFFIXES)}",
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

    try:
        result = analyze_video(dest, out_dir=config.UPLOADS_DIR)
    except Exception as exc:  # noqa: BLE001 — surface as a clean 400, not a 500 traceback
        raise HTTPException(status_code=400, detail=f"Could not analyze video: {exc}") from exc

    if "annotated_video" in result:
        rel = Path(result["annotated_video"]).relative_to(config.OUTPUT_DIR)
        result["annotated_video_url"] = f"/outputs/{rel.as_posix()}"

    return result
