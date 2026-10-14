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


def blur_faces(frame: np.ndarray, blur_strength: int = 35, detect_width: int | None = None) -> np.ndarray:
    """
    Detect faces in a frame (Haar cascade) and Gaussian-blur those regions
    in place. Returns the modified frame.

    detect_width: if given and smaller than the frame, faces are searched on
    a copy downscaled to this width (much faster on HD footage; the blur is
    still applied at full resolution).

    This is a lightweight, dependency-free baseline. For production use,
    prefer a proper face/plate detector (e.g. a small YOLO-face model) with
    a wider margin around each box, since Haar cascades can miss faces at
    odd angles or low resolution.
    """
    scale = 1.0
    src = frame
    if detect_width and frame.shape[1] > detect_width:
        scale = detect_width / frame.shape[1]
        src = cv2.resize(frame, (detect_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20) if scale == 1.0 else (12, 12)
    )
    if scale != 1.0:
        faces = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces]
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


def blur_person_heads(frame: np.ndarray, person_boxes, blur_strength: int = 31) -> np.ndarray:
    """
    Blur the head region (top ~30%, slightly widened) of every detected
    person box, in place. Complements the Haar face search: it also covers
    faces in profile, from behind a helmet visor, or too small for Haar.
    """
    h_img, w_img = frame.shape[:2]
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    for (x, y, w, h) in person_boxes:
        hx0 = max(0, int(x - 0.1 * w))
        hx1 = min(w_img, int(x + 1.1 * w))
        hy0 = max(0, int(y - 0.05 * h))
        hy1 = min(h_img, int(y + 0.3 * h))
        roi = frame[hy0:hy1, hx0:hx1]
        if roi.size:
            kk = max(3, min(k, (min(roi.shape[:2]) // 2) * 2 + 1))
            frame[hy0:hy1, hx0:hx1] = cv2.GaussianBlur(roi, (kk, kk), 0)
    return frame
