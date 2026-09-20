"""
Single-clip "was a crash about to happen?" analysis.

pipeline.py answers a *multi-site* question: given a video plus a
historical accident CSV, how do known locations rank against each other?
That requires a site catalogue and accident records.

This module answers a narrower, more direct question about ONE video, with
no other inputs required: did the detect -> track -> conflict stack find a
moment where a vehicle and a pedestrian (or two vehicles) came close
enough, closing fast enough, that a crash was plausibly seconds away?

Entry point: analyze_video(video_path) -> JSON-serializable verdict dict.
Optionally writes an annotated copy of the clip with tracked boxes and the
near-miss moment(s) highlighted, so a reviewer can see exactly what the
system saw (never an automated finding of fault -- see README).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import cv2

from . import config, detection, tracking, conflict, privacy

# How long (seconds) the "NEAR-MISS" highlight stays on screen around each
# flagged frame, so a viewer scrubbing the annotated clip can actually see
# it rather than a single-frame flash.
HIGHLIGHT_WINDOW_S = 0.4

BOX_COLORS = {
    "vehicle": (60, 200, 60),      # green, BGR
    "pedestrian": (230, 200, 40),  # cyan-ish
}
ALERT_COLOR = (0, 0, 255)  # red, BGR


def _severity_label(severity: float) -> str:
    if severity >= 0.66:
        return "severe"
    if severity >= 0.33:
        return "moderate"
    return "mild"


def _verdict_sentence(events: list[dict]) -> str:
    if not events:
        return (
            "No near-miss detected. Tracked road users kept a safe distance "
            "and/or closing speed throughout this clip."
        )
    worst = max(events, key=lambda e: e["severity"])
    who = "a vehicle and a pedestrian" if worst["kind"] == "vehicle_pedestrian" else "two vehicles"
    ttc_txt = f"{worst['ttc_s']:.1f}s" if worst["ttc_s"] is not None else "a very short, unmeasured interval"
    extra = f" ({len(events) - 1} more event(s) also flagged.)" if len(events) > 1 else ""
    return (
        f"Yes \u2014 a crash looks like it was almost about to happen. At {worst['t']:.1f}s into the "
        f"clip, {who} closed to {worst['distance_m']:.1f}m apart with a time-to-collision of "
        f"{ttc_txt} ({_severity_label(worst['severity'])} severity).{extra}"
    )


def analyze_video(
    video_path: str | Path,
    site_id: Optional[str] = None,
    annotate: bool = True,
    out_dir: Optional[Path] = None,
) -> dict:
    """
    Run the full detect -> track -> conflict stack on ONE video and return a
    near-miss verdict for it. Does not touch outputs/tracks.json etc. from
    the multi-site pipeline (pipeline.py) -- this is a self-contained call.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not find video: {video_path}")

    site_id = site_id or f"upload-{uuid.uuid4().hex[:8]}"
    out_dir = Path(out_dir) if out_dir else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = detection.process_video(video_path)
    tracks = tracking.build_tracks(frames)
    events = conflict.find_all_conflicts(tracks, site_id=site_id)

    n_vehicles = len({t["id"] for t in tracks if t["cls"] == "vehicle"})
    n_pedestrians = len({t["id"] for t in tracks if t["cls"] == "pedestrian"})
    duration_s = frames[-1]["t"] if frames else 0.0

    result = {
        "site_id": site_id,
        "source_video": str(video_path),
        "frames_processed": len(frames),
        "duration_s": round(duration_s, 2),
        "vehicles_tracked": n_vehicles,
        "pedestrians_tracked": n_pedestrians,
        "near_miss_detected": len(events) > 0,
        "event_count": len(events),
        "events": events,
        "verdict": _verdict_sentence(events),
    }

    if annotate:
        annotated_path = out_dir / f"{video_path.stem}_annotated.mp4"
        _write_annotated_video(video_path, tracks, events, annotated_path)
        result["annotated_video"] = str(annotated_path)

    return result


def _write_annotated_video(video_path: Path, tracks: list[dict], events: list[dict], out_path: Path) -> None:
    """
    Re-reads the source video and writes a copy with: every tracked box
    drawn (green = vehicle, yellow = pedestrian, each labeled with its
    track id), a red connecting line + "NEAR-MISS" banner during any
    flagged conflict window, and faces blurred (privacy.py) before the
    frame is ever written to disk -- matches the no-raw-imagery-retained
    rule described in README "Ethics & privacy by design".
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not re-open video for annotation: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    by_frame: dict[int, list[dict]] = {}
    for row in tracks:
        by_frame.setdefault(row["frame"], []).append(row)

    half_window = max(1, int(fps * HIGHLIGHT_WINDOW_S))
    flagged_frames: dict[int, list[dict]] = {}
    for e in events:
        for f in range(e["frame"] - half_window, e["frame"] + half_window + 1):
            flagged_frames.setdefault(f, []).append(e)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = privacy.blur_faces(frame)

        for row in by_frame.get(frame_idx, []):
            color = BOX_COLORS.get(row["cls"], (200, 200, 200))
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, f'{row["cls"]}#{row["id"]}', (x, max(12, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

        rows_here = by_frame.get(frame_idx, [])
        for e in flagged_frames.get(frame_idx, []):
            a = next((r for r in rows_here if r["id"] == e["a_id"]), None)
            b = next((r for r in rows_here if r["id"] == e["b_id"]), None)
            if a and b:
                ac = (a["x"] + a["w"] // 2, a["y"] + a["h"] // 2)
                bc = (b["x"] + b["w"] // 2, b["y"] + b["h"] // 2)
                cv2.line(frame, ac, bc, ALERT_COLOR, 2)
            cv2.putText(
                frame, "NEAR-MISS", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ALERT_COLOR, 2, cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
