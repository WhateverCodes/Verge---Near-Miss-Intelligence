#!/usr/bin/env python3
"""
Generates a short synthetic "traffic" video: a car-shaped rectangle moving
left-to-right and a pedestrian-shaped rectangle moving top-to-bottom, timed
so their paths come close near the middle of the clip (a near-miss). This
lets the whole pipeline (detection -> tracking -> conflict -> risk scoring
-> report) run end-to-end with zero real footage, per the problem
statement's "recorded or simulated traffic data" requirement.

Run:
    python scripts/generate_sample_video.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from backend.app import config

WIDTH, HEIGHT = 640, 480
FPS = config.FRAME_RATE_FPS
DURATION_S = 6
N_FRAMES = int(FPS * DURATION_S)

ROAD_COLOR = (60, 60, 60)
LANE_COLOR = (200, 200, 200)
CAR_COLOR = (40, 90, 200)     # BGR
PED_COLOR = (30, 160, 60)


def draw_background(frame: np.ndarray) -> np.ndarray:
    frame[:] = (25, 100, 25)  # grass green background
    cv2.rectangle(frame, (0, HEIGHT // 2 - 60), (WIDTH, HEIGHT // 2 + 60), ROAD_COLOR, -1)
    for x in range(0, WIDTH, 40):
        cv2.line(frame, (x, HEIGHT // 2), (x + 20, HEIGHT // 2), LANE_COLOR, 2)
    # a "crosswalk" band where pedestrian crosses
    cv2.rectangle(frame, (WIDTH // 2 - 30, HEIGHT // 2 - 60), (WIDTH // 2 + 30, HEIGHT // 2 + 60), (230, 230, 230), 1)
    return frame


def main() -> None:
    out_path = config.SAMPLE_VIDEO_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (WIDTH, HEIGHT))

    car_w, car_h = 70, 34
    ped_w, ped_h = 18, 30

    for i in range(N_FRAMES):
        t = i / N_FRAMES
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_background(frame)

        # Car drives left -> right across the full clip.
        car_x = int(-car_w + t * (WIDTH + 2 * car_w))
        car_y = HEIGHT // 2 - car_h // 2
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), CAR_COLOR, -1)

        # Pedestrian starts crossing partway through the clip, timed so the
        # two paths come close near the middle of the road (a near-miss),
        # then continues off-screen.
        ped_t = max(0.0, min(1.0, (t - 0.25) / 0.5))
        ped_x = WIDTH // 2 - ped_w // 2
        ped_y = int(-ped_h + ped_t * (HEIGHT + 2 * ped_h))
        if 0.0 < ped_t < 1.0:
            cv2.rectangle(frame, (ped_x, ped_y), (ped_x + ped_w, ped_y + ped_h), PED_COLOR, -1)

        writer.write(frame)

    writer.release()
    print(f"Wrote synthetic demo clip: {out_path} ({N_FRAMES} frames @ {FPS:.0f}fps)")


if __name__ == "__main__":
    main()
