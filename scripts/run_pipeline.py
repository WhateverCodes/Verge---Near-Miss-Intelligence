#!/usr/bin/env python3
"""
Runs the full pipeline end-to-end on the sample/demo data:

    detection -> tracking -> conflict indicators -> risk scoring -> report

Run:
    python scripts/generate_sample_video.py   # once, if not already done
    python scripts/run_pipeline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import config
from backend.app.pipeline import run_full_pipeline


def main() -> None:
    if not config.SAMPLE_VIDEO_PATH.exists():
        print(
            f"No demo video found at {config.SAMPLE_VIDEO_PATH}.\n"
            "Run `python scripts/generate_sample_video.py` first."
        )
        sys.exit(1)

    summary = run_full_pipeline()
    print(json.dumps(summary, indent=2))
    print("\nDone. Outputs written to ./outputs/")
    print("Serve the API with: uvicorn backend.app.main:app --reload --port 8000")
    print("Then open frontend/index.html in a browser.")


if __name__ == "__main__":
    main()
