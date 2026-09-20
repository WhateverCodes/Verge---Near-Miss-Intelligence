"""
Orchestrates the full pipeline: detection -> tracking -> conflict indicators
-> risk scoring -> report. Shared by scripts/run_pipeline.py (CLI) and
main.py (POST /api/pipeline/run) so there's exactly one place this logic
lives.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config, detection, tracking, conflict, risk_scoring, report


def run_full_pipeline(
    video_path: Path = config.SAMPLE_VIDEO_PATH,
    accidents_csv: Path = config.SAMPLE_ACCIDENTS_CSV,
    site_id: str = config.DEMO_SITE["site_id"],
) -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) detect objects frame-by-frame
    frames = detection.process_video(video_path)

    # 2) stitch detections into anonymized trajectories
    tracks = tracking.build_tracks(frames)
    config.TRACKS_JSON.write_text(json.dumps(tracks, indent=2))

    # 3) derive near-miss / conflict events from the trajectories
    events = conflict.find_near_miss_events(tracks, site_id=site_id)
    event_dicts = conflict.events_to_dicts(events)
    config.NEAR_MISS_JSON.write_text(json.dumps(event_dicts, indent=2))

    # 4) merge with historical accident data into ranked hotspots
    hotspots = risk_scoring.run(
        accidents_csv=accidents_csv,
        near_miss_json=config.NEAR_MISS_JSON,
        out_geojson=config.HOTSPOTS_GEOJSON,
    )

    # 5) generate the transparent, human-readable recommendation report
    report.run(hotspots, out_path=config.REPORT_MD)

    return {
        "frames_processed": len(frames),
        "tracks_found": len({t["id"] for t in tracks}),
        "near_miss_events": len(event_dicts),
        "hotspots_ranked": len(hotspots),
        "top_hotspot": hotspots[0]["name"] if hotspots else None,
        "outputs": {
            "tracks": str(config.TRACKS_JSON),
            "near_miss_events": str(config.NEAR_MISS_JSON),
            "hotspots_geojson": str(config.HOTSPOTS_GEOJSON),
            "report": str(config.REPORT_MD),
        },
    }
