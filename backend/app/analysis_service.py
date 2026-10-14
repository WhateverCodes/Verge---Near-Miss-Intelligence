"""
Location-linked video analysis: the glue between the existing single-clip
analysis (video_analysis.py) and the site risk picture (site_evidence.py).

  run_analysis()   analyse one saved upload, optionally at a location:
                   evidence replay + snapshot, observation recorded for the
                   site, before/after review priority, analysis.json saved.
  start_job()      run_analysis() on a background thread with real,
                   stage-by-stage progress the dashboard can poll.

The raw uploaded video is deleted as soon as analysis finishes (success or
failure): only the annotated, face-blurred evidence and anonymized
trajectories are kept — see README "Ethics & privacy by design".
"""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import config, site_evidence
from .video_analysis import analyze_video

STAGES = [
    ("detect", "Detecting road users"),
    ("track", "Tracking trajectories"),
    ("conflict", "Computing conflict indicators"),
    ("render", "Rendering evidence replay"),
    ("risk", "Updating site risk"),
]


def new_analysis_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def _url(path: str | Path) -> str:
    rel = Path(path).resolve().relative_to(config.OUTPUT_DIR.resolve())
    return f"/outputs/{rel.as_posix()}"


def resolve_location(site_id: Optional[str], lat: Optional[float], lon: Optional[float], name: Optional[str]) -> Optional[dict]:
    """An existing site (CSV or earlier pin), a new map pin, or None."""
    if site_id:
        site = site_evidence.known_site(site_id)
        if site is None:
            raise ValueError(f"Unknown location '{site_id}'. Refresh the map and select it again.")
        return site
    if lat is not None and lon is not None:
        return site_evidence.add_custom_site(name or "", float(lat), float(lon))
    return None


def load_analysis(analysis_id: str) -> dict:
    if not analysis_id.replace("-", "").isalnum():
        raise FileNotFoundError(analysis_id)
    path = config.ANALYSES_DIR / analysis_id / "analysis.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_analysis(
    video_path: Path,
    source_name: str,
    location: Optional[dict] = None,
    analysis_id: Optional[str] = None,
    progress: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
    detector: str = "auto",
) -> dict:
    analysis_id = analysis_id or new_analysis_id()
    adir = config.ANALYSES_DIR / analysis_id
    adir.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda *a: None)

    before = site_evidence.current_hotspots() if location else None

    result = analyze_video(
        video_path,
        site_id=location["site_id"] if location else None,
        out_dir=adir,
        evidence_dir=adir,
        progress=report,
        detector=detector,
    )

    # Public URLs instead of absolute server paths.
    result["source_video"] = source_name
    result["source_name"] = source_name
    result["annotated_video_url"] = _url(result["annotated_video"])
    result["annotated_video"] = Path(result["annotated_video"]).name
    preview = result.pop("preview", None)
    if preview and preview.get("frames"):
        result["preview"] = {"frame_urls": [_url(p) for p in preview["frames"]], "fps": preview["fps"]}
    for e in result["events"]:
        ev = e.get("evidence")
        if ev:
            ev["frame_urls"] = [_url(p) for p in ev.pop("frames")]
            ev["snapshot_url"] = _url(ev["snapshot"]) if ev.get("snapshot") else None
            ev.pop("dir", None)
            e["report_pdf_url"] = f"/api/analyses/{analysis_id}/report.pdf?event={e['event_index']}"

    result["analysis_id"] = analysis_id
    result["created_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    result["location"] = None
    result["site_update"] = None

    if location:
        report("risk", 0, 1)
        site_evidence.record_observation(
            analysis_id, location["site_id"], result["events"],
            meta={"source_name": source_name, "duration_s": result["duration_s"]},
        )
        after = site_evidence.current_hotspots()
        result["site_update"] = site_evidence.site_update(before, after, location["site_id"], result["events"])
        site_after = next((h for h in after if h["site_id"] == location["site_id"]), None)
        result["site_after"] = site_after
        result["location"] = {
            "site_id": location["site_id"],
            "name": location["name"],
            "lat": location["lat"],
            "lon": location["lon"],
            "has_historical_record": bool(site_after and site_after.get("has_historical_record")),
        }
        report("risk", 1, 1)

    # Keep evidence on disk (for the PDF), but not the absolute snapshot path in the API.
    stored = json.loads(json.dumps(result))
    (adir / "analysis.json").write_text(json.dumps(stored, indent=2), encoding="utf-8")
    for e in result["events"]:
        if e.get("evidence"):
            e["evidence"].pop("snapshot", None)
    return result


# ---- background jobs --------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job(analysis_id: str, with_location: bool) -> dict:
    stages = [
        {"key": k, "label": label, "state": "pending", "done": None, "total": None}
        for k, label in STAGES
        if k != "risk" or with_location
    ]
    return {
        "analysis_id": analysis_id,
        "state": "running",
        "stages": stages,
        "error": None,
        "result": None,
    }


def _on_progress(job: dict, stage: str, done: Optional[int], total: Optional[int]) -> None:
    with _jobs_lock:
        keys = [s["key"] for s in job["stages"]]
        if stage not in keys:
            return
        idx = keys.index(stage)
        # A later stage has started: never move backwards.
        if any(s["state"] != "pending" for s in job["stages"][idx + 1:]):
            return
        for s in job["stages"][:idx]:
            s["state"] = "done"
        cur = job["stages"][idx]
        cur["done"], cur["total"] = done, total
        if done is not None and total and done >= total:
            cur["state"] = "done"
            if idx + 1 < len(job["stages"]):
                job["stages"][idx + 1]["state"] = "active"
        else:
            cur["state"] = "active"


def start_job(upload_path: Path, source_name: str, location: Optional[dict], detector: str = "auto") -> str:
    analysis_id = new_analysis_id()
    job = _new_job(analysis_id, with_location=location is not None)
    job["stages"][0]["state"] = "active"
    with _jobs_lock:
        _jobs[analysis_id] = job

    def work() -> None:
        try:
            result = run_analysis(upload_path, source_name, location, analysis_id,
                                  progress=lambda s, d, t: _on_progress(job, s, d, t), detector=detector)
            with _jobs_lock:
                for s in job["stages"]:
                    s["state"] = "done"
                job["result"] = result
                job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 — reported to the UI, never a crash
            traceback.print_exc()
            with _jobs_lock:
                for s in job["stages"]:
                    if s["state"] == "active":
                        s["state"] = "error"
                job["error"] = friendly_error(exc)
                job["state"] = "error"
        finally:
            Path(upload_path).unlink(missing_ok=True)

    threading.Thread(target=work, name=f"analysis-{analysis_id}", daemon=True).start()
    return analysis_id


def job_status(analysis_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(analysis_id)
        return json.loads(json.dumps(job)) if job else None


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, FileNotFoundError) and "Could not open video" in msg:
        return "This file could not be opened as a video. Try an .mp4 (H.264) or .avi export."
    if isinstance(exc, ValueError):
        return msg
    return f"Analysis failed: {msg or exc.__class__.__name__}"
