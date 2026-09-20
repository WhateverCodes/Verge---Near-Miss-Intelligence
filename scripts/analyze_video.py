#!/usr/bin/env python3
"""
Analyze ONE traffic video and answer: did a crash almost happen in it?

Runs the same detect -> track -> conflict stack as the main pipeline, but
on a single clip with no accident CSV / site catalogue required -- just a
direct near-miss verdict for that footage.

Run:
    python scripts/analyze_video.py --video path/to/clip.mp4
    python scripts/analyze_video.py --video path/to/clip.mp4 --no-annotate
    python scripts/analyze_video.py                       # uses the synthetic demo clip

Outputs (written to --out-dir, default ./outputs/):
    <clip-name>_verdict.json     full machine-readable result (all events)
    <clip-name>_annotated.mp4    same clip with tracked boxes + near-miss
                                  highlight burned in (unless --no-annotate)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import config
from backend.app.video_analysis import analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--video", type=Path, default=config.SAMPLE_VIDEO_PATH,
        help="Path to the video to analyze (default: the synthetic demo clip).",
    )
    parser.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument(
        "--no-annotate", action="store_true",
        help="Skip writing the annotated output video (faster; verdict/JSON still written).",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}")
        if args.video == config.SAMPLE_VIDEO_PATH:
            print("Run `python scripts/generate_sample_video.py` first, or pass --video <path>.")
        sys.exit(1)

    print(f"Analyzing {args.video} ...")
    result = analyze_video(args.video, annotate=not args.no_annotate, out_dir=args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = args.out_dir / f"{args.video.stem}_verdict.json"
    verdict_path.write_text(json.dumps(result, indent=2))

    print("=" * 72)
    print(result["verdict"])
    print("=" * 72)
    summary = {k: v for k, v in result.items() if k != "events"}
    print(json.dumps(summary, indent=2))

    if result["events"]:
        print(f"\n{len(result['events'])} event(s) total \u2014 full detail in {verdict_path}")
    if "annotated_video" in result:
        print(f"Annotated video: {result['annotated_video']}")


if __name__ == "__main__":
    main()
