"""
Frame-level object detection.

This module deliberately uses classical computer vision (background
subtraction + contour analysis) instead of a trained neural network, so the
whole repository runs anywhere with `pip install -r requirements.txt` and no
GPU or model-weight download. It exists behind one clean function —
`ObjectDetector.detect(frame)` — so it can be swapped for a real model
(YOLOv8, etc.) without touching any other module. See docs/architecture.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from . import config


@dataclass
class Detection:
    """A single detected object in one frame."""
    x: int
    y: int
    w: int
    h: int
    cls: str          # "vehicle" | "pedestrian"
    confidence: float  # heuristic confidence in [0, 1], not a calibrated probability

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class ObjectDetector:
    """
    Background-subtraction based moving-object detector.

    Classifies each detected blob as "vehicle" or "pedestrian" using a crude
    size/aspect-ratio heuristic. This is a prototype-grade stand-in for a
    trained detector (see docs/architecture.md for the YOLOv8 drop-in).
    """

    def __init__(self) -> None:
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=True
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        fg_mask = self._bg_subtractor.apply(frame)
        # Drop shadow pixels (MOG2 marks them as 127) and clean up noise.
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < config.MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append(self._classify(x, y, w, h, area))
        return detections

    @staticmethod
    def _classify(x: int, y: int, w: int, h: int, area: float) -> Detection:
        aspect = w / max(h, 1)
        is_vehicle = area >= config.VEHICLE_MIN_AREA and aspect >= config.VEHICLE_MIN_ASPECT
        cls = "vehicle" if is_vehicle else "pedestrian"
        # crude, unitless "confidence" purely for demo/report purposes
        confidence = min(1.0, area / (config.VEHICLE_MIN_AREA * 2))
        return Detection(x=x, y=y, w=w, h=h, cls=cls, confidence=round(confidence, 2))


def process_video(video_path: str) -> list[dict]:
    """
    Run the detector over every frame of a video and return raw per-frame
    detections. Does NOT track identity across frames — see tracking.py.

    Returns a list of {"frame": int, "t": float, "detections": [...]}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE_FPS
    detector = ObjectDetector()
    frames_out = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        frames_out.append(
            {
                "frame": frame_idx,
                "t": round(frame_idx / fps, 3),
                "detections": [d.__dict__ for d in detections],
            }
        )
        frame_idx += 1

    cap.release()
    return frames_out
