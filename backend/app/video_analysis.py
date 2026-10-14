"""
Single-clip "was a crash about to happen?" analysis.

pipeline.py answers a *multi-site* question: given a video plus a
historical accident CSV, how do known locations rank against each other?
That requires a site catalogue and accident records.

This module answers a narrower, more direct question about ONE video, with
no other inputs required: did the detect -> track -> conflict stack find a
moment where road users came close enough, closing fast enough, to count as
an observed near-miss?

Two detector paths:
  "object"  (default for real footage) pretrained object detector on sampled
            frames (detection_yolo.py) -> IoU tracker (tracking.ObjectTracker)
            -> evidence-based conflict engine (conflict_engine.py)
  "motion"  the original background-subtraction detector (detection.py) ->
            centroid tracker -> conflict.py. Kept for the synthetic demo clips,
            which are drawn shapes an object detector cannot recognize.
  "auto"    motion detector for clips recognised as synthetic (drawn
            shapes, see looks_synthetic), object detector for everything
            else. Real footage never falls back to the motion detector.

Entry point: analyze_video(video_path) -> JSON-serializable verdict dict.
Optionally writes an annotated copy of the clip, plus (when `evidence_dir`
is given) a short evidence replay per event built from that event's own
start / peak / end, a peak-moment snapshot and a lightweight full-clip
preview, so a reviewer can see exactly what the system saw (never an
automated finding of fault -- see README).
"""
from __future__ import annotations

import bisect
import math
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import config, conflict, conflict_engine, detection, detection_yolo, privacy, tracking

# How long (seconds) the "NEAR-MISS" highlight stays on screen around each
# flagged frame, so a viewer scrubbing the annotated clip can actually see
# it rather than a single-frame flash.
HIGHLIGHT_WINDOW_S = 0.4

BOX_COLORS = {
    "vehicle": (60, 200, 60),      # green, BGR
    "pedestrian": (230, 200, 40),  # cyan-ish
}
ALERT_COLOR = (0, 0, 255)  # red, BGR
# Second road user of a conflict pair, so the two stay distinguishable.
ALERT_COLOR_B = (0, 150, 255)  # amber, BGR
HUD_BG = (27, 35, 16)  # dark green-black, matches the dashboard ink colour

TRAIL_S = 1.5  # seconds of trajectory drawn behind each conflict road user

OUTPUT_MAX_FPS = 15.0     # annotated clip / replay frame rate cap
OUTPUT_MAX_WIDTH = 960    # annotated clip width cap
PREVIEW_FPS = 5.0         # full-clip preview (JPEG sequence) frame rate
PREVIEW_WIDTH = 640
REPLAY_MAX_S = 8.0

ProgressFn = Callable[[str, Optional[int], Optional[int]], None]

DISPLAY_LABEL = {"person": "Pedestrian", "two-wheeler": "Two-wheeler"}


def _severity_label(severity: float) -> str:
    return config.severity_band(severity).lower()


def fmt_time(t: float | None) -> str:
    """Clip timestamp as MM:SS.s (e.g. 00:07.4)."""
    if t is None:
        return "--:--"
    minutes, seconds = divmod(max(0.0, float(t)), 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"


def _label_of(e: dict, side: str) -> str:
    lab = e.get(f"{side}_label") or ""
    if lab and lab != "vehicle":
        return DISPLAY_LABEL.get(lab, lab.capitalize())
    return e[f"{side}_cls"].capitalize()


def pair_label(e: dict) -> str:
    return f"{_label_of(e, 'a')} ↔ {_label_of(e, 'b')}"


def road_user_label(cls: str, track_id: int, label: str = "") -> str:
    name = DISPLAY_LABEL.get(label, label.capitalize()) if label and label != "vehicle" else cls.capitalize()
    return f"{name} #{track_id}"


def _verdict_sentence(events: list[dict], thresholds: dict | None = None) -> str:
    th = thresholds or {"distance_m": config.DIST_THRESH_M, "ttc_s": config.TTC_THRESH_S}
    if not events:
        return (
            "No confirmed near-miss. No tracked pair of road users met the conflict criteria "
            f"(within {th['distance_m']:g} m, time-to-collision at or below {th['ttc_s']:g} s, "
            "sustained over consecutive frames) in this clip."
        )
    crashes = [e for e in events if e.get("event_type") == "crash"]
    if crashes:
        c = crashes[0]
        who = pair_label(c).lower().replace(" ↔ ", " and ")
        others = len(events) - 1
        extra = f" {others} further event(s) confirmed." if others else ""
        return (f"Crash detected at {fmt_time(c['peak_t'])}: {who} collided (contact after a fast approach, "
                f"followed by an impact).{extra}")
    worst = max(events, key=lambda e: e.get("peak_severity", e["severity"]))
    who = pair_label(worst).lower().replace(" ↔ ", " and ")
    dist = worst.get("min_distance_m", worst["distance_m"])
    ttc = worst.get("min_ttc_s", worst["ttc_s"])
    t = worst.get("peak_t", worst["t"])
    sev = worst.get("peak_severity", worst["severity"])
    if worst.get("indicator") == "close_pass":
        ttc_txt = "a close pass while moving (proximity at speed)"
    else:
        ttc_txt = f"a minimum time-to-collision of {ttc:.2f} s" if ttc is not None else "no measurable time-to-collision"
    extra = f" {len(events) - 1} further event(s) confirmed." if len(events) > 1 else ""
    return (
        f"Near-miss detected at {fmt_time(t)}: {who} came within {dist:.1f} m of each other, "
        f"with {ttc_txt} ({_severity_label(sev)} severity).{extra}"
    )


def explain_event(e: dict, thresholds: dict | None = None) -> dict:
    """
    Short, evidence-based bullets for one event: what was observed, and which
    configured indicator(s) caused it to be flagged. Used by the dashboard and
    the PDF evidence report so both say exactly the same thing.
    """
    th = thresholds or {"distance_m": config.DIST_THRESH_M, "ttc_s": config.TTC_THRESH_S}
    a = road_user_label(e["a_cls"], e["a_id"], e.get("a_label", ""))
    b = road_user_label(e["b_cls"], e["b_id"], e.get("b_label", ""))
    zone = f"{th['distance_m']:g} m"
    what, why = [], []

    if e.get("event_type") == "crash":
        what.append(f"{a} and {b} approached each other fast (gap closing at about {e.get('approach_closing_ms', 0):.0f} m/s).")
        what.append(f"Their footprints met at {fmt_time(e.get('contact_t'))} and stayed together for about {e.get('zone_duration_s', 0):.1f} s.")
        if e.get("impact_dv_ms", 0) >= config.CRASH_DV_MS:
            what.append(f"Straight after contact, one of them changed velocity by about {e['impact_dv_ms']:.0f} m/s.")
        if e.get("impact_shape_change", 1) >= config.CRASH_SHAPE_CHANGE:
            what.append("A vehicle's outline changed sharply after contact (spin or rollover).")
        why.append("Observed collision: footprints in contact after a fast approach.")
        why.append(f"Impact signature: velocity change of {e.get('impact_dv_ms', 0):.1f} m/s across the contact "
                   f"(threshold {config.CRASH_DV_MS:g} m/s, beyond hard braking) or a sudden change of shape.")
        why.append("Both tracks stayed on the same objects through the impact (not a tracker switch).")
        if e["kind"] == "vehicle_pedestrian":
            why.append("Involves a pedestrian (vulnerable road user).")
        return {"what_happened": what, "why_flagged": why}

    if e.get("closing_observed"):
        what.append(f"{a} and {b} trajectories converged.")
    else:
        what.append(f"{a} and {b} entered the {zone} encounter zone.")
    if e.get("contact_phase"):
        what.append(f"Their footprints met on screen at {fmt_time(e.get('contact_t'))}: clearance at the measurement limit (about 0 m).")
    elif e.get("min_distance_m") is not None:
        what.append(f"Minimum separation of {e['min_distance_m']:.1f} m at {fmt_time(e.get('min_distance_t'))}.")
    if e.get("zone_duration_s"):
        what.append(f"The conflict indicator held for about {e['zone_duration_s']:.1f} s.")
    if e.get("pet_s") is not None:
        what.append(f"Both paths passed the same point {e['pet_s']:.2f} s apart (post-encroachment time).")

    ttc = e.get("min_ttc_s")
    ttc_limit = th.get("ttc_vv_s", th["ttc_s"]) if e["kind"] == "vehicle_vehicle" else th["ttc_s"]
    if e.get("indicator") == "close_pass":
        lim = config.CONFLICT_CLOSE_PASS_VV_M if e["kind"] == "vehicle_vehicle" else config.CONFLICT_CLOSE_PASS_VRU_M
        why.append(f"Close pass: clearance of {e['min_distance_m']:.2f} m while moving (threshold {lim:g} m).")
        why.append("The two were seen closing in from further apart before passing this close.")
    else:
        if ttc is not None and ttc <= ttc_limit:
            why.append(f"Time-to-collision fell to {ttc:.2f} s (threshold {ttc_limit:g} s).")
        elif ttc is None:
            why.append("Separation fell below the encounter threshold; time-to-collision was not measurable.")
        why.append(f"Separation below the {zone} threshold.")
        if th.get("collision_course"):
            why.append("The two were on a collision course, not just passing side by side.")
    if e["kind"] == "vehicle_pedestrian":
        why.append("Involves a pedestrian (vulnerable road user).")
    if e.get("pet_s") is not None and e["pet_s"] <= 1.5:
        why.append(f"Short post-encroachment time ({e['pet_s']:.2f} s).")
    return {"what_happened": what, "why_flagged": why}


def _legacy_thresholds() -> dict:
    return {
        "distance_m": config.DIST_THRESH_M,
        "distance_kind": "centre distance on screen at a fixed demo scale",
        "ttc_s": config.TTC_THRESH_S,
        "pixels_per_meter": config.PIXELS_PER_METER,
        "pet_radius_m": config.PET_RADIUS_M,
    }


def _object_thresholds() -> dict:
    return {
        "distance_m": config.CONFLICT_MAX_GAP_M,
        "distance_kind": "estimated edge-to-edge ground clearance",
        "ttc_s": config.CONFLICT_TTC_S,
        "ttc_vv_s": config.CONFLICT_TTC_VV_S,
        "min_duration_s": config.CONFLICT_MIN_DURATION_S,
        "collision_course": True,
        "pet_radius_m": config.PET_RADIUS_M,
    }


def analyze_video(
    video_path: str | Path,
    site_id: Optional[str] = None,
    annotate: bool = True,
    out_dir: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
    detector: str = "auto",
) -> dict:
    """
    Run the detect -> track -> conflict stack on ONE video and return a
    near-miss verdict for it. Does not touch outputs/tracks.json etc. from
    the multi-site pipeline (pipeline.py) -- this is a self-contained call.

    detector: "auto" | "object" | "motion" (see module docstring).
    evidence_dir: if given, also writes per-event evidence replays (JPEG
      frame sequences), peak snapshots and a full-clip preview there.
    progress: optional callback progress(stage, done, total) with stage in
      {"detect", "track", "conflict", "render"}.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not find video: {video_path}")
    if detector not in ("auto", "object", "motion"):
        raise ValueError(f"Unknown detector '{detector}'.")

    def report(stage: str, done: Optional[int] = None, total: Optional[int] = None) -> None:
        if progress is not None:
            progress(stage, done, total)

    site_id = site_id or f"upload-{uuid.uuid4().hex[:8]}"
    out_dir = Path(out_dir) if out_dir else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    t0 = time.perf_counter()

    want_object = detector in ("auto", "object")
    if want_object and not detection_yolo.available():
        if detector == "object":
            raise ValueError("The object-detection model file is missing (models/yolo11x.onnx, yolo11l.onnx or yolov8n.onnx).")
        want_object = False

    note = None
    if detector == "auto" and want_object and looks_synthetic(video_path):
        # Drawn demo clips: an object detector cannot recognise flat shapes,
        # and the motion detector is exactly what they were made for. Real
        # footage never takes this path (see looks_synthetic).
        want_object = False
        note = "Synthetic (drawn) clip recognised, so the motion-based detector was used."
    if want_object:
        analysis = _analyze_objects(video_path, site_id, report, timings)
    else:
        analysis = _analyze_motion(video_path, site_id, report, timings)
    if note:
        analysis["diagnostics"]["detector_note"] = note

    tracks, events, th = analysis["tracks"], analysis["events"], analysis["thresholds"]
    for i, e in enumerate(events):
        e["event_index"] = i
        e["pair_label"] = pair_label(e)
        e.update(explain_event(e, th))
    report("conflict", 1, 1)

    n_vehicles = len({t["id"] for t in tracks if t["cls"] == "vehicle"})
    n_pedestrians = len({t["id"] for t in tracks if t["cls"] == "pedestrian"})
    diag = analysis["diagnostics"]
    diag["warnings"] = _sanity_warnings(analysis, n_vehicles + n_pedestrians, len(events))

    result = {
        "site_id": site_id,
        "source_video": str(video_path),
        "frames_processed": diag["frames_analyzed"],
        "frames_total": diag["frames_total"],
        "duration_s": round(analysis["duration_s"], 2),
        "vehicles_tracked": n_vehicles,
        "pedestrians_tracked": n_pedestrians,
        "road_users_by_class": _count_by_label(tracks),
        "road_users_detected": bool(tracks),
        "near_miss_detected": len(events) > 0,
        "event_count": len(events),
        "events": events,
        "set_aside_events": analysis.get("set_aside", []),
        "verdict": _verdict_sentence(events, th),
        "thresholds": th,
        "detector": analysis["detector"],
        "diagnostics": diag,
    }

    if annotate:
        t_r = time.perf_counter()
        annotated_path = out_dir / f"{video_path.stem}_annotated.mp4"
        extras = _write_annotated_video(
            video_path, tracks, events, annotated_path,
            evidence_dir=Path(evidence_dir) if evidence_dir else None,
            progress=lambda d, t: report("render", d, t),
            total_frames=diag["frames_total"],
            privacy_records=analysis.get("privacy"),
        )
        result["annotated_video"] = str(annotated_path)
        if extras.get("preview"):
            result["preview"] = extras["preview"]
        timings["render_s"] = round(time.perf_counter() - t_r, 2)
    report("render", 1, 1)
    timings["total_s"] = round(time.perf_counter() - t0, 2)
    diag["timings_s"] = timings
    return result


def looks_synthetic(video_path: Path) -> bool:
    """
    True for computer-drawn clips (flat colour regions, very few distinct
    colours), such as the demo clips in data/demo_clips. Measured on the
    bundled synthetic clips vs. real traffic footage: synthetic >= 64% flat
    pixels and <= ~200 distinct (5-bit) colours; real footage <= 12% flat
    and >= ~800 colours. The thresholds sit well between the two.
    """
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    votes = []
    for k in range(3):
        if n > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / 3))
        ok, f = cap.read()
        if not ok:
            continue
        sm = cv2.resize(f, (320, max(1, int(320 * f.shape[0] / f.shape[1]))), interpolation=cv2.INTER_NEAREST)
        q = (sm // 8).reshape(-1, 3).astype(np.int32)
        unique = len(np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]))
        flat = float((np.abs(cv2.Laplacian(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), cv2.CV_64F)) < 1).mean())
        votes.append(flat > 0.4 and unique < 500)
    cap.release()
    return bool(votes) and all(votes)


def _analyze_objects(video_path: Path, site_id: str, report, timings: dict) -> dict:
    t = time.perf_counter()
    records, meta = detection_yolo.process_video_objects(video_path, progress=lambda d, n: report("detect", d, n))
    timings["detect_s"] = round(time.perf_counter() - t, 2)
    if not records:
        raise ValueError("The file could not be decoded as video (no readable frames).")
    duration = meta["frames_total"] / meta["src_fps"]
    if duration < config.MIN_CLIP_S:
        raise ValueError(f"Clip is too short ({duration:.1f} s). At least {config.MIN_CLIP_S:g} s of footage is needed.")

    t = time.perf_counter()
    rows, tstats = tracking.build_object_tracks(records)
    timings["track_s"] = round(time.perf_counter() - t, 2)
    report("track", 1, 1)

    t = time.perf_counter()
    events, cdiag = conflict_engine.find_conflict_events(rows, (meta["width"], meta["height"]), site_id)
    timings["conflict_s"] = round(time.perf_counter() - t, 2)
    reclass = set(cdiag.pop("reclassified_ids", []))
    for r in rows:  # keep counts and on-screen labels consistent with the engine
        if r["id"] in reclass:
            r["cls"], r["label"] = "vehicle", "two-wheeler"

    diag = {
        "detector": "object",
        "detector_name": f"{meta.get('model', 'YOLO')} (COCO, ONNX) on {meta.get('inference_backend', 'OpenCV DNN (CPU)')}",
        "source_fps": meta["src_fps"],
        "resolution": f"{meta['width']}x{meta['height']}",
        "frames_total": meta["frames_total"],
        "frames_analyzed": meta["frames_analyzed"],
        "detect_fps": meta["detect_fps"],
        "camera": meta["camera"],
        "tracks": tstats,
        "conflicts": cdiag,
    }
    return {
        "detector": "object",
        "tracks": rows,
        "events": events,
        "thresholds": _object_thresholds(),
        "duration_s": duration,
        "track_stats": tstats,
        "diagnostics": diag,
        "privacy": [(r["frame"], r["privacy"]) for r in records],
    }


def _analyze_motion(video_path: Path, site_id: str, report, timings: dict) -> dict:
    t = time.perf_counter()
    frames = detection.process_video(video_path, progress=lambda d, n: report("detect", d, n))
    timings["detect_s"] = round(time.perf_counter() - t, 2)
    if not frames:
        raise ValueError("The file could not be decoded as video (no readable frames).")
    duration_s = frames[-1]["t"] if frames else 0.0
    if duration_s < config.MIN_CLIP_S:
        raise ValueError(
            f"Clip is too short ({duration_s:.1f} s). At least {config.MIN_CLIP_S:g} s of footage "
            "is needed to model the background and track motion."
        )
    t = time.perf_counter()
    tracks = tracking.build_tracks(frames)
    timings["track_s"] = round(time.perf_counter() - t, 2)
    report("track", 1, 1)

    t = time.perf_counter()
    size = _frame_size(video_path)
    events = conflict.find_all_conflicts(tracks, site_id=site_id)
    conflict.enrich_events(tracks, events, frame_size=size)
    set_aside = []
    if config.REQUIRE_CLOSING_MOTION:
        set_aside = [
            {"t": e["t"], "kind": e["kind"], "a_id": e["a_id"], "b_id": e["b_id"],
             "reason": "No closing motion observed between the pair (likely a tracking fragment)."}
            for e in events if not e.get("closing_observed")
        ]
        events = [e for e in events if e.get("closing_observed")]
    timings["conflict_s"] = round(time.perf_counter() - t, 2)
    fps = len(frames) / duration_s if duration_s else config.FRAME_RATE_FPS
    ids = {r["id"] for r in tracks}
    diag = {
        "detector": "motion",
        "detector_name": "Background subtraction (MOG2) + contour heuristic",
        "source_fps": round(fps, 2),
        "resolution": f"{size[0]}x{size[1]}" if size else "?",
        "frames_total": len(frames),
        "frames_analyzed": len(frames),
        "detect_fps": round(fps, 2),
        "tracks": {"confirmed_tracks": len(ids)},
    }
    return {
        "detector": "motion",
        "tracks": tracks,
        "events": events,
        "set_aside": set_aside,
        "thresholds": _legacy_thresholds(),
        "duration_s": duration_s,
        "track_stats": {"confirmed_tracks": len(ids)},
        "diagnostics": diag,
    }


def _count_by_label(tracks: list[dict]) -> dict:
    seen = {}
    for r in tracks:
        seen[r["id"]] = r.get("label") or r["cls"]
    out: dict = {}
    for lab in seen.values():
        out[lab] = out.get(lab, 0) + 1
    return out


def _sanity_warnings(analysis: dict, n_tracks: int, n_events: int) -> list[str]:
    """Plain-language flags for results that should be treated with caution.
    Nothing is capped or hidden; the warnings just say why to be careful."""
    w = []
    diag = analysis["diagnostics"]
    dur = max(1e-6, analysis["duration_s"])
    if analysis["detector"] == "motion" and n_tracks > 40:
        w.append("The motion-based detector found an unusually large number of objects. It is meant for "
                 "synthetic clips with a fixed camera; use the object detector for real footage.")
    if analysis["detector"] == "object":
        ts = diag["tracks"]
        peak = max(1, ts.get("peak_simultaneous", 1))
        # rough turnover check: how many distinct road users per visible slot per 10 s
        if n_tracks / peak / max(1.0, dur / 10) > 6:
            w.append(f"High track turnover ({n_tracks} tracks, at most {peak} visible at once): some road "
                     "users may have been split into several tracks (occlusion or missed detections).")
        if diag.get("conflicts", {}).get("camera_too_unstable"):
            w.append("The camera moves or zooms too much in this clip (handheld or moving footage) to measure "
                     "distances between road users, so no near-misses were assessed. Use footage from a fixed camera.")
        elif diag.get("camera", {}).get("moving_camera"):
            w.append("The camera moves in this clip. Camera motion was estimated and compensated, but "
                     "distances and speeds are less reliable than with a fixed camera.")
    if n_tracks == 0:
        w.append("No road users were detected. Check that the clip shows road traffic in reasonable light.")
    if n_events > 10 * max(1.0, dur / 30):
        w.append("Unusually many conflict events for the clip length: review the replays before relying on the count.")
    return w


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _frame_size(video_path: Path) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (w, h) if w > 0 and h > 0 else None


def _evidence_window(e: dict, fps: float, n_frames: int) -> tuple[int, int]:
    """
    Frames [start, end] for an event's replay, built from the event's OWN
    timestamps: from ~1 s before the conflict indicator started (at least
    1.5 s before the peak) to ~1 s after it ended (at least 1 s after the
    peak). Long events are centred on the peak; the window is clamped to the
    clip.
    """
    peak_t = e.get("peak_t", e["t"])
    start_t = e.get("zone_start_t") if e.get("zone_start_t") is not None else e["t"]
    end_t = e.get("zone_end_t") if e.get("zone_end_t") is not None else peak_t
    lo = min(start_t - 1.0, peak_t - config.EVIDENCE_PRE_S)
    hi = max(end_t + 1.0, peak_t + config.EVIDENCE_POST_S)
    if hi - lo > REPLAY_MAX_S:
        lo, hi = peak_t - REPLAY_MAX_S / 2, peak_t + REPLAY_MAX_S / 2
    start = max(0, int(math.floor(lo * fps)))
    end = min(n_frames - 1, int(math.ceil(hi * fps)))
    return start, max(start, end)


class _TrackIndex:
    """Boxes of every track at any source frame: exact where the track was
    observed, linearly interpolated between observations (the object detector
    runs on sampled frames), nothing outside the track's lifetime or across
    gaps longer than max_gap frames."""

    def __init__(self, rows: list[dict], max_gap: int):
        by: dict[int, list[dict]] = {}
        for r in rows:
            by.setdefault(r["id"], []).append(r)
        self.tracks = []
        for tid, rs in by.items():
            rs.sort(key=lambda r: r["frame"])
            self.tracks.append((rs[0]["frame"], rs[-1]["frame"], [r["frame"] for r in rs], rs))
        self.max_gap = max_gap

    def at(self, f: int) -> list[dict]:
        out = []
        for first, last, frames, rs in self.tracks:
            if f < first or f > last:
                continue
            i = bisect.bisect_left(frames, f)
            if frames[i] == f:
                out.append(rs[i])
                continue
            a, b = rs[i - 1], rs[i]
            if b["frame"] - a["frame"] > self.max_gap:
                continue
            k = (f - a["frame"]) / (b["frame"] - a["frame"])
            r = dict(a)
            for key in ("x", "y", "w", "h"):
                r[key] = int(round(a[key] + (b[key] - a[key]) * k))
            r["frame"] = f
            out.append(r)
        return out


def _put_label(img, text, org, color, scale, thickness=1, bg=HUD_BG):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - tw - 6))
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 6, y + base), bg, -1)
    cv2.putText(img, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _hud(img, lines, scale):
    """Compact translucent panel in the top-left corner."""
    pad = int(8 * scale / 0.45)
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, th)[0] for t, s, _, th in lines]
    w = max(sz[0] for sz in sizes) + 2 * pad
    h = sum(sz[1] for sz in sizes) + pad * (len(lines) + 1)
    x1, y1 = min(img.shape[1] - 1, 8 + w), min(img.shape[0] - 1, 8 + h)
    roi = img[8:y1, 8:x1]
    img[8:y1, 8:x1] = cv2.addWeighted(roi, 0.22, np.full_like(roi, HUD_BG), 0.78, 0)
    y = 8 + pad
    for (text, s, color, th), sz in zip(lines, sizes):
        y += sz[1]
        cv2.putText(img, text, (8 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, th, cv2.LINE_AA)
        y += pad


def _row_text(row: dict) -> str:
    lab = row.get("label") or ""
    if lab and lab != "vehicle":
        return f"{DISPLAY_LABEL.get(lab, lab).upper()} #{row['id']}"
    return f"{row['cls'].upper()} #{row['id']}"


def _draw_frame(img, frame_idx, t, rows_here, histories, focus: list[dict], fps, show_hud: bool, peak_marker: bool):
    """
    Draw every tracked road user (thin, class-coloured) and, for each event in
    `focus`, emphasize its two road users: thick boxes, trails, the line
    between them, and the conflict point.
    """
    h_img, w_img = img.shape[:2]
    scale = max(0.4, min(1.2, w_img / 640 * 0.45))
    thick = max(1, int(round(w_img / 640)))

    focus_ids: dict[int, tuple] = {}
    for e in focus:
        focus_ids.setdefault(e["a_id"], ALERT_COLOR)
        focus_ids.setdefault(e["b_id"], ALERT_COLOR_B)

    by_id = {r["id"]: r for r in rows_here}

    trail_frames = int(TRAIL_S * fps)
    for tid, color in focus_ids.items():
        pts = [
            (int(r["x"] + r["w"] / 2), int(r["y"] + r["h"] / 2))
            for r in histories.get(tid, [])
            if frame_idx - trail_frames <= r["frame"] <= frame_idx
        ]
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, thick, cv2.LINE_AA)

    for row in rows_here:
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        label = _row_text(row)
        if row["id"] in focus_ids:
            color = focus_ids[row["id"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thick + 2)
            # First road user of a pair is labelled below its box, the second
            # above, so the two labels don't collide when the pair is close.
            below = color == ALERT_COLOR and y + h + 22 < h_img
            org = (x, y + h + int(18 * scale / 0.45)) if below else (x, y - 4)
            _put_label(img, label, org, (255, 255, 255), scale, max(1, thick), bg=color)
        else:
            color = BOX_COLORS.get(row["cls"], (200, 200, 200))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
            if not focus:  # keep evidence frames uncluttered: only the pair is labelled
                _put_label(img, label, (x, y - 4), color, scale * 0.8, 1)

    for e in focus:
        a, b = by_id.get(e["a_id"]), by_id.get(e["b_id"])
        if a and b:
            ac = (int(a["x"] + a["w"] / 2), int(a["y"] + a["h"] / 2))
            bc = (int(b["x"] + b["w"] / 2), int(b["y"] + b["h"] / 2))
            cv2.line(img, ac, bc, (255, 255, 255), thick, cv2.LINE_AA)
        # Predicted path: where each road user is heading (its motion over the
        # last ~0.5 s, projected ~1 s ahead) while the conflict builds up.
        if frame_idx <= int(e.get("peak_frame", e["frame"])) + int(0.3 * fps):
            for tid, color in ((e["a_id"], ALERT_COLOR), (e["b_id"], ALERT_COLOR_B)):
                cur = by_id.get(tid)
                past = [r for r in histories.get(tid, []) if frame_idx - int(0.5 * fps) <= r["frame"] < frame_idx - 1]
                if cur is None or not past:
                    continue
                p0 = past[0]
                dt = max(1, frame_idx - p0["frame"]) / fps
                c0 = (p0["x"] + p0["w"] / 2, p0["y"] + p0["h"])
                c1 = (cur["x"] + cur["w"] / 2, cur["y"] + cur["h"])
                vx, vy = (c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt
                if math.hypot(vx, vy) * 1.0 < 0.15 * max(cur["w"], cur["h"]):
                    continue  # (almost) stationary: no arrow
                tip = (int(c1[0] + vx * 1.0), int(c1[1] + vy * 1.0))
                cv2.arrowedLine(img, (int(c1[0]), int(c1[1])), tip, color, thick + 1, cv2.LINE_AA, tipLength=0.18)
        cp = e.get("conflict_point_px")
        if cp and e["zone_start_t"] - 0.2 <= t <= e["zone_end_t"] + 0.5:
            c = (int(cp[0]), int(cp[1]))
            r = max(6, int(10 * w_img / 640))
            cv2.circle(img, c, r, ALERT_COLOR, thick + 1, cv2.LINE_AA)
            cv2.line(img, (c[0] - r, c[1]), (c[0] + r, c[1]), ALERT_COLOR, thick, cv2.LINE_AA)
            cv2.line(img, (c[0], c[1] - r), (c[0], c[1] + r), ALERT_COLOR, thick, cv2.LINE_AA)

    if show_hud and focus:
        e = focus[0]
        ttc = ("collision" if e.get("event_type") == "crash" else
               f'{e["min_ttc_s"]:.2f} s' if e.get("min_ttc_s") is not None else
               ("close pass" if e.get("indicator") == "close_pass" else "n/a"))
        dist = f'{e["min_distance_m"]:.1f} m' if e.get("min_distance_m") is not None else "n/a"
        a_lbl = road_user_label(e["a_cls"], e["a_id"], e.get("a_label", "")).upper()
        b_lbl = road_user_label(e["b_cls"], e["b_id"], e.get("b_label", "")).upper()
        # phase of the replay relative to the measured conflict
        crash = e.get("event_type") == "crash"
        if t < e["zone_start_t"] - 0.05:
            title, tcol = "APPROACHING", (0, 190, 255)
        elif t <= e["zone_end_t"] + 0.05:
            title, tcol = ("CRASH DETECTED" if crash else "CONFLICT DETECTED"), (80, 80, 255)
        else:
            title, tcol = ("AFTER THE CRASH" if crash else "AFTER THE CONFLICT"), (200, 200, 200)
        _hud(img, [
            (title, scale * 1.05, tcol, max(1, thick + 1)),
            (f"{a_lbl}  <->  {b_lbl}", scale, (255, 255, 255), 1),
            (f"TTC {ttc}   MIN DIST {dist}   SEVERITY {e.get('severity_label', '').upper()}", scale * 0.9, (220, 230, 220), 1),
        ], scale)

    stamp = fmt_time(t) + ("   CONFLICT PEAK" if peak_marker else "")
    _put_label(img, stamp, (8, h_img - 8), (80, 80, 255) if peak_marker else (255, 255, 255), scale, max(1, thick))


def _resize_max_width(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


def _write_annotated_video(
    video_path: Path,
    tracks: list[dict],
    events: list[dict],
    out_path: Path,
    evidence_dir: Optional[Path] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    total_frames: int = 0,
    privacy_records: Optional[list] = None,
) -> dict:
    """
    Reads the source video once, decoding only the frames that are output:
      - out_path (.mp4): the annotated clip at <= OUTPUT_MAX_FPS and
        <= OUTPUT_MAX_WIDTH, every tracked road user labelled and each
        conflict pair emphasized while its conflict is active;
      - if evidence_dir is given: for the most severe events, a JPEG frame
        sequence (the "evidence replay") built from the event's own start /
        peak / end, a snapshot of the exact peak frame, and a small full-clip
        preview sequence.
    Faces are blurred (privacy.py) before any frame is written: Haar face
    search on every output frame plus, when the object detector ran, the
    head region of every detected person (including riders).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not re-open video for annotation: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE_FPS
    if fps <= 0 or fps > 240:
        fps = config.FRAME_RATE_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = total_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1

    out_step = max(1, int(round(fps / OUTPUT_MAX_FPS)))
    out_fps = fps / out_step
    out_w = min(width, OUTPUT_MAX_WIDTH)
    out_size = (out_w, int(round(height * out_w / width)))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, out_size)

    index = _TrackIndex(tracks, max_gap=int(round(fps * 1.0)))
    histories: dict[int, list[dict]] = {}
    for row in tracks:
        histories.setdefault(row["id"], []).append(row)
    for h in histories.values():
        h.sort(key=lambda r: r["frame"])

    half_window = max(1, int(fps * HIGHLIGHT_WINDOW_S))

    def zone_frames(e):
        start = int(round((e.get("zone_start_t") if e.get("zone_start_t") is not None else e["t"]) * fps))
        end = int(round((e.get("zone_end_t") if e.get("zone_end_t") is not None else e["t"]) * fps))
        return start - half_window, end + half_window

    zones = [(zone_frames(e), e) for e in events]

    privacy_frames = [f for f, _ in privacy_records] if privacy_records else []
    privacy_boxes = [b for _, b in privacy_records] if privacy_records else []

    def persons_at(f: int):
        if not privacy_frames:
            return []
        i = bisect.bisect_left(privacy_frames, f)
        best = [j for j in (i - 1, i) if 0 <= j < len(privacy_frames)]
        j = min(best, key=lambda j: abs(privacy_frames[j] - f))
        return privacy_boxes[j] if abs(privacy_frames[j] - f) <= out_step * 3 else []

    extras: dict = {}
    replays = []
    preview_dir = None
    preview_step = max(1, int(round(fps / PREVIEW_FPS)))
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = evidence_dir / "preview"
        preview_dir.mkdir(exist_ok=True)
        extras["preview"] = {"frames": [], "fps": round(fps / preview_step, 3)}
        ranked = sorted(events, key=lambda e: e.get("peak_severity", e["severity"]), reverse=True)
        for e in ranked[: config.EVIDENCE_MAX_EVENTS]:
            start, end = _evidence_window(e, fps, n_frames)
            ev_dir = evidence_dir / f"event_{e['event_index']}"
            ev_dir.mkdir(parents=True, exist_ok=True)
            e["evidence"] = {
                "dir": str(ev_dir),
                "frames": [],
                "frame_times": [],
                "fps": round(out_fps, 3),
                "start_t": round(start / fps, 2),
                "end_t": round(end / fps, 2),
                "peak_t": e.get("peak_t", e["t"]),
                "peak_index": 0,
                "snapshot": None,
            }
            replays.append((start, end, e, ev_dir))
    peak_frames = {int(e.get("peak_frame", e["frame"])) for _, _, e, _ in replays}

    frame_idx = 0
    while True:
        need_out = frame_idx % out_step == 0
        need_replay = any(s <= frame_idx <= en and ((frame_idx - s) % out_step == 0 or frame_idx in peak_frames)
                          for s, en, _, _ in replays)
        need_preview = preview_dir is not None and frame_idx % preview_step == 0
        if not (need_out or need_replay or need_preview):
            if not cap.grab():
                break
            frame_idx += 1
            continue
        ok, raw = cap.read()
        if not ok:
            break

        raw = privacy.blur_faces(raw, detect_width=640)
        people = persons_at(frame_idx)
        if people:
            privacy.blur_person_heads(raw, people)
        rows_here = index.at(frame_idx)
        t = frame_idx / fps

        if need_out or need_preview:
            active = [e for (s, en), e in zones if s <= frame_idx <= en]
            frame = raw.copy()
            is_peak = any(abs(frame_idx - int(e.get("peak_frame", e["frame"]))) <= out_step for e in active)
            _draw_frame(frame, frame_idx, t, rows_here, histories, active, fps, show_hud=bool(active), peak_marker=is_peak)
            if need_out:
                writer.write(cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA) if out_size != (width, height) else frame)
            if need_preview:
                name = preview_dir / f"p_{len(extras['preview']['frames']):04d}.jpg"
                cv2.imwrite(str(name), _resize_max_width(frame, PREVIEW_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, 72])
                extras["preview"]["frames"].append(str(name))

        for start, end, e, ev_dir in replays:
            if not (start <= frame_idx <= end):
                continue
            peak_frame = int(e.get("peak_frame", e["frame"]))
            is_ev_peak = frame_idx == peak_frame
            if (frame_idx - start) % out_step != 0 and not is_ev_peak:
                continue
            img = raw.copy()
            _draw_frame(img, frame_idx, t, rows_here, histories, [e], fps, show_hud=True, peak_marker=is_ev_peak)
            ev = e["evidence"]
            if is_ev_peak:
                ev["peak_index"] = len(ev["frames"])
                snap = ev_dir / "snapshot.jpg"
                cv2.imwrite(str(snap), _resize_max_width(img, 1280), [cv2.IMWRITE_JPEG_QUALITY, 88])
                ev["snapshot"] = str(snap)
            name = ev_dir / f"f_{len(ev['frames']):04d}.jpg"
            cv2.imwrite(str(name), _resize_max_width(img, config.EVIDENCE_MAX_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, 82])
            ev["frames"].append(str(name))
            ev["frame_times"].append(round(t, 3))

        frame_idx += 1
        if progress is not None and frame_idx % 15 == 0:
            progress(frame_idx, max(n_frames, frame_idx + 1))

    cap.release()
    writer.release()
    return extras
