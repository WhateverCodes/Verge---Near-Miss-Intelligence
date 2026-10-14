#!/usr/bin/env python3
"""
Generates a small set of SYNTHETIC demo clips for the "Analyze a video" flow.
They are clearly synthetic (drawn shapes), not real footage — they exist so
the full demo (upload -> conflict -> evidence replay -> site update -> PDF)
can be shown reliably offline, including the "no near-miss" outcome.

  data/demo_clips/crossing_near_miss.mp4   pedestrian steps into the road,
                                           hesitates at the lane line while
                                           a car passes, then hurries across
  data/demo_clips/vehicle_rear_end.mp4     lead car brakes, following car
                                           closes in fast and brakes late
  data/demo_clips/safe_crossing.mp4        pedestrian crosses well before the
                                           car arrives (expected: no near-miss)

The original scripts/generate_sample_video.py (used by the multi-site
pipeline) is unchanged.

Run:
    python scripts/generate_demo_clips.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from backend.app import config

OUT_DIR = config.DATA_DIR / "demo_clips"
W, H = 800, 450
FPS = config.FRAME_RATE_FPS

ROAD_TOP, ROAD_BOT = 190, 340
LANE_W_Y = 228          # westbound lane centre
LANE_E_Y = 300          # eastbound lane centre
CROSS_X = 430           # zebra crossing centre

CAR_W, CAR_H = 76, 36


def background() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (72, 96, 70)                                            # verge / grass
    cv2.rectangle(img, (0, 0), (W, 120), (92, 104, 112), -1)         # building block
    for x in range(0, W, 110):
        cv2.rectangle(img, (x + 8, 20), (x + 92, 112), (110, 124, 134), -1)
        for wx in range(x + 16, x + 86, 22):
            cv2.rectangle(img, (wx, 34), (wx + 12, 50), (150, 170, 176), -1)
            cv2.rectangle(img, (wx, 70), (wx + 12, 86), (150, 170, 176), -1)
    cv2.rectangle(img, (0, 150), (W, ROAD_TOP), (170, 170, 164), -1)  # north footpath
    cv2.rectangle(img, (0, ROAD_BOT), (W, 380), (170, 170, 164), -1)  # south footpath
    cv2.rectangle(img, (0, ROAD_TOP), (W, ROAD_BOT), (62, 62, 62), -1)
    for x in range(0, W, 48):                                        # centre line
        cv2.line(img, (x, 264), (x + 26, 264), (210, 210, 210), 2)
    for y in range(ROAD_TOP + 6, ROAD_BOT - 6, 16):                  # zebra
        cv2.rectangle(img, (CROSS_X - 30, y), (CROSS_X + 30, y + 8), (224, 224, 224), -1)
    for tx in (90, 260, 620, 740):                                   # trees
        cv2.circle(img, (tx, 136), 14, (50, 110, 55), -1)
        cv2.circle(img, (tx - 40, 396), 14, (50, 110, 55), -1)
    return img


def draw_car(img, cx, cy, body, facing_right=True):
    x, y = int(cx - CAR_W / 2), int(cy - CAR_H / 2)
    cv2.rectangle(img, (x, y), (x + CAR_W, y + CAR_H), body, -1)
    shade = tuple(int(c * 0.55) for c in body)
    cab = (x + 18, y + 6, x + 52, y + CAR_H - 6) if facing_right else (x + 24, y + 6, x + 58, y + CAR_H - 6)
    cv2.rectangle(img, cab[:2], cab[2:], shade, -1)
    wind_x = cab[2] - 8 if facing_right else cab[0]
    cv2.rectangle(img, (wind_x, cab[1] + 2), (wind_x + 8, cab[3] - 2), (200, 210, 215), -1)
    light = (x + CAR_W - 4, y + 4) if facing_right else (x, y + 4)
    cv2.rectangle(img, light, (light[0] + 4, light[1] + 6), (180, 240, 255), -1)


def draw_ped(img, cx, cy, shirt=(40, 150, 235)):
    cx, cy = int(cx), int(cy)
    cv2.circle(img, (cx, cy - 12), 5, (60, 90, 140), -1)             # head
    cv2.rectangle(img, (cx - 7, cy - 7), (cx + 7, cy + 7), shirt, -1)
    cv2.rectangle(img, (cx - 6, cy + 7), (cx - 1, cy + 17), (70, 50, 40), -1)
    cv2.rectangle(img, (cx + 1, cy + 7), (cx + 6, cy + 17), (70, 50, 40), -1)


LEAD_IN_FRAMES = 40  # 2 s of empty road first: MOG2 adapts very fast in its
                     # first frames and would otherwise split early objects


def write(name: str, frames) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = 0
    bg = background()
    for _ in range(LEAD_IN_FRAMES):
        wr.write(bg)
        n += 1
    for f in frames:
        wr.write(f)
        n += 1
    wr.release()
    print(f"Wrote {path} ({n} frames, {n / FPS:.1f} s, synthetic)")
    return path


def crossing_near_miss():
    bg = background()
    ped_y, ped_state = 150.0, "wait"
    stop_y = 243.0                          # hesitates on the lane line
    car_x = None
    for i in range(190):
        img = bg.copy()
        if 10 <= i <= 56:                   # early westbound car, before anyone crosses
            draw_car(img, W + 40 - (i - 10) * 19, LANE_W_Y, (170, 120, 40), facing_right=False)
        if i >= 70:
            if ped_state == "wait":
                ped_state = "walk"
            if ped_state == "walk":
                ped_y = min(stop_y, ped_y + 1.45)
                if ped_y >= stop_y:
                    ped_state = "hesitate"
            elif ped_state == "hesitate" and car_x is not None and car_x - CAR_W / 2 > CROSS_X + 14:
                ped_state = "hurry"
            elif ped_state == "hurry":
                ped_y += 2.7
            if ped_y < H + 20:
                draw_ped(img, CROSS_X, ped_y)
        if i >= 72:
            car_x = -50 + (i - 72) * 8.2
            if car_x < W + 60:
                draw_car(img, car_x, LANE_E_Y, (40, 40, 200))
        yield img


def vehicle_rear_end():
    # Speeds stay >= ~4.5 px/frame: MOG2 background subtraction slowly absorbs
    # near-stationary, uniformly coloured objects (a known limitation).
    bg = background()
    lead_x, lead_v = -45.0, 8.0
    foll_x, foll_v = None, 12.0
    for i in range(150):
        img = bg.copy()
        if i >= 10:
            if 300 < lead_x < 420:          # lead car slows for the crossing
                lead_v = max(4.5, lead_v - 0.45)
            elif lead_x >= 420:
                lead_v = min(9.0, lead_v + 0.3)
            lead_x += lead_v
            if lead_x < W + 60:
                draw_car(img, lead_x, LANE_E_Y, (200, 200, 205))
        if i >= 44:
            if foll_x is None:
                foll_x = -45.0
            gap = (lead_x - CAR_W / 2) - (foll_x + CAR_W / 2)
            if gap < 22:                    # late, hard braking
                foll_v = max(lead_v, foll_v - 4.0)
            elif gap > 60 and foll_v < 12:
                foll_v = min(12.0, foll_v + 0.5)
            foll_x += foll_v
            if foll_x < W + 60:
                draw_car(img, foll_x, LANE_E_Y, (40, 40, 190))
        yield img


def safe_crossing():
    bg = background()
    ped_y = 150.0
    for i in range(230):
        img = bg.copy()
        if 12 <= i <= 60:
            draw_car(img, W + 40 - (i - 12) * 18, LANE_W_Y, (170, 120, 40), facing_right=False)
        if i >= 66:
            ped_y += 1.8
            if ped_y < H + 20:
                draw_ped(img, CROSS_X, ped_y)
        if i >= 175:
            draw_car(img, -50 + (i - 175) * 8.5, LANE_E_Y, (40, 40, 200))
        yield img


def main() -> None:
    write("crossing_near_miss.mp4", crossing_near_miss())
    write("vehicle_rear_end.mp4", vehicle_rear_end())
    write("safe_crossing.mp4", safe_crossing())


if __name__ == "__main__":
    main()
