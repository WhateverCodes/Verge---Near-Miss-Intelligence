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

# Windows consoles/pipes default to a legacy code page; print UTF-8 so
# dashes and arrows in the output don't turn into "?" or mojibake.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    print("Then serve the dashboard: python -m http.server 5500 --directory frontend")


if __name__ == "__main__":
    main()
