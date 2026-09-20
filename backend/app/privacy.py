"""
Privacy-preserving helpers.

The problem statement explicitly requires the system to "protect privacy
[and] avoid face identification". Concretely, that means: never persist a
frame — to disk, to an API response, or to a debug snapshot — without first
blurring any detectable face region, and never carry raw imagery past the
detection/tracking stage. Everything downstream of this module only ever
sees anonymized (x, y, t, class, id) trajectories, never pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def blur_faces(frame: np.ndarray, blur_strength: int = 35) -> np.ndarray:
    """
    Detect faces in a frame (Haar cascade) and Gaussian-blur those regions
    in place. Returns the modified frame.

    This is a lightweight, dependency-free baseline. For production use,
    prefer a proper face/plate detector (e.g. a small YOLO-face model) with
    a wider margin around each box, since Haar cascades can miss faces at
    odd angles or low resolution.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    out = frame.copy()
    for (x, y, w, h) in faces:
        roi = out[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        out[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def strip_to_trajectory(detection_record: dict) -> dict:
    """
    Given a per-frame detection record that may (in a fuller implementation)
    carry image crops or other identifying payload, return only the
    anonymized geometric/temporal fields that are allowed to persist.

    This is a defensive no-op in this prototype (detection.py already never
    attaches imagery), but it documents and enforces the boundary: nothing
    but position, size, class, id and time should ever reach tracks.json.
    """
    allowed_keys = {"frame", "t", "id", "cls", "x", "y", "w", "h", "confidence"}
    return {k: v for k, v in detection_record.items() if k in allowed_keys}
