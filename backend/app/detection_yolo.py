"""
Object detection for real traffic footage: a pretrained YOLO11 (COCO)
network exported to ONNX, run with ONNX Runtime DirectML on a GPU when
available, otherwise with OpenCV's DNN module (YOLO11x on GPU, YOLO11l on
CPU, YOLOv8n as a last-resort fallback; see _best_detector).

Why this exists next to detection.py: the motion-based detector
(background subtraction + contours) cannot tell objects apart. On real
footage one moving truck becomes a dozen "pedestrian" blobs, every blob gets
its own track, and every pair of blobs becomes a "conflict". A trained
detector recognizes actual road users, so this is the default for uploaded
real-world clips. detection.py is kept, unchanged, for the synthetic demo
clips and the multi-site pipeline.

Only classes the model genuinely detects are used:
    person -> pedestrian
    bicycle, car, motorcycle, bus, truck -> vehicle (specific label kept)

Riders and occupants: COCO "person" also fires on motorcycle riders and on
people visible inside cars/buses. Those are not pedestrians; treating them
as such would put every rider in a permanent "conflict" with their own bike.
They are removed from the road-user list but kept as privacy boxes, so their
faces are still blurred.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from . import config
from .detection import Detection

COCO_ROAD_USERS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TWO_WHEELERS = {"bicycle", "motorcycle"}
ENCLOSED = {"car", "bus", "truck"}


def category(label: str) -> str:
    return "pedestrian" if label == "person" else "vehicle"


def _inter(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b) -> float:
    i = _inter(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


class YoloDetector:
    def __init__(self, model_path: Path = config.YOLO_MODEL_PATH, input_size: int = config.YOLO_INPUT_SIZE):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Object-detection model not found: {model_path}")
        self.size = input_size
        self.model_path = Path(model_path)
        self._lock = threading.Lock()  # cv2.dnn.Net is not thread-safe
        self.ort = None
        self.backend = "OpenCV DNN (CPU)"
        if config.YOLO_BACKEND == "auto":
            self._try_directml(model_path)
        self.net = None if self.ort else cv2.dnn.readNetFromONNX(str(model_path))

    def _try_directml(self, model_path: Path) -> None:
        """Optional GPU path. Any failure (package missing, no DX12 GPU, driver
        problem) silently keeps the CPU path: reliability over speed."""
        try:
            import onnxruntime as ort  # optional dependency (requirements-gpu.txt)
            if "DmlExecutionProvider" not in ort.get_available_providers():
                return
            sess = ort.InferenceSession(str(model_path), providers=["DmlExecutionProvider"])
            name = sess.get_inputs()[0].name
            sess.run(None, {name: np.zeros((1, 3, self.size, self.size), np.float32)})  # smoke test
            if sess.get_providers()[0] != "DmlExecutionProvider":
                return
            self.ort, self.ort_input = sess, name
            self.backend = "ONNX Runtime DirectML (GPU)"
        except Exception:  # noqa: BLE001
            self.ort = None

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        if self.ort is not None:
            try:
                return self.ort.run(None, {self.ort_input: blob})[0]
            except Exception:  # noqa: BLE001 — GPU hiccup: fall back for good
                self.ort = None
                self.backend = "OpenCV DNN (CPU)"
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.net.setInput(blob)
        return self.net.forward()

    def _infer(self, frame: np.ndarray):
        S = self.size
        h, w = frame.shape[:2]
        r = min(S / w, S / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((S, S, 3), 114, np.uint8)
        px, py = (S - nw) // 2, (S - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (S, S), swapRB=True)
        with self._lock:
            out = self._forward(blob)
        pred = out[0].T  # (8400, 4 + 80)
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(scores)), cls]
        min_conf = min(config.YOLO_CONF.values())
        keep = (conf >= min_conf) & np.isin(cls, list(COCO_ROAD_USERS))
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        labels = [COCO_ROAD_USERS[int(c)] for c in cls]
        ok = np.array([conf[i] >= config.YOLO_CONF[labels[i]] for i in range(len(labels))], bool)
        pred, cls, conf = pred[ok], cls[ok], conf[ok]
        labels = [lb for lb, k in zip(labels, ok) if k]
        if not len(pred):
            return []
        b = pred[:, :4]
        x = (b[:, 0] - b[:, 2] / 2 - px) / r
        y = (b[:, 1] - b[:, 3] / 2 - py) / r
        boxes = np.stack([x, y, b[:, 2] / r, b[:, 3] / r], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(), cls.tolist(), 0.0, config.YOLO_NMS_IOU)
        idx = np.array(idx).flatten().tolist()
        raw = []
        for i in idx:
            bx, by, bw, bh = boxes[i]
            x0, y0 = max(0.0, bx), max(0.0, by)
            x1, y1 = min(float(w), bx + bw), min(float(h), by + bh)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            raw.append({"box": (x0, y0, x1 - x0, y1 - y0), "label": labels[i], "conf": float(conf[i])})
        return raw

    def detect(self, frame: np.ndarray) -> tuple[List[Detection], List[tuple]]:
        """Returns (road users, privacy boxes). Privacy boxes = every person
        box, including riders/occupants that are not road users on their own."""
        h_img = frame.shape[0]
        raw = self._infer(frame)

        # Class-agnostic merge across vehicle classes (car vs truck vs bus on
        # the same object): keep the most confident box.
        vehicles = sorted([d for d in raw if d["label"] != "person"], key=lambda d: -d["conf"])
        kept_v: list[dict] = []
        for d in vehicles:
            if all(_iou(d["box"], k["box"]) < config.YOLO_VEHICLE_MERGE_IOU for k in kept_v):
                kept_v.append(d)
        # Part-of-vehicle boxes (e.g. a truck's cab detected on its own next to
        # the whole truck): mostly inside a larger vehicle box AND sharing two of
        # its edges. A separate car in front of a bus doesn't share edges that way.
        kept_v = [d for d in kept_v if not any(_is_part_of(d["box"], k["box"]) for k in kept_v if k is not d)]
        persons = [d for d in raw if d["label"] == "person"]

        privacy = [tuple(int(v) for v in p["box"]) for p in persons]
        min_h = config.YOLO_MIN_BOX_H_FRAC * h_img
        out: List[Detection] = []
        for p in persons:
            if p["box"][3] < min_h or _is_rider_or_occupant(p["box"], kept_v):
                continue
            out.append(_to_det(p))
        for v in kept_v:
            if v["box"][3] < min_h * 0.6:
                continue
            out.append(_to_det(v))
        return out, privacy


def _is_part_of(small, big) -> bool:
    if small[2] * small[3] >= big[2] * big[3]:
        return False
    if _inter(small, big) / max(1e-6, small[2] * small[3]) < 0.8:
        return False
    tol_x, tol_y = 0.1 * big[2], 0.1 * big[3]
    shared = sum([
        abs(small[0] - big[0]) <= tol_x, abs(small[0] + small[2] - big[0] - big[2]) <= tol_x,
        abs(small[1] - big[1]) <= tol_y, abs(small[1] + small[3] - big[1] - big[3]) <= tol_y,
    ])
    return shared >= 2


def _is_rider_or_occupant(pbox, vehicles) -> bool:
    px, py, pw, ph = pbox
    p_area = pw * ph
    p_bottom = py + ph
    pcx = px + pw / 2
    for v in vehicles:
        vx, vy, vw, vh = v["box"]
        overlap = _inter(pbox, v["box"]) / p_area if p_area else 0
        if v["label"] in TWO_WHEELERS:
            # rider: overlaps the bike, centred over it, feet not below the wheels
            if overlap > 0.25 and vx <= pcx <= vx + vw and p_bottom <= vy + vh + 0.25 * vh:
                return True
        elif v["label"] in ENCLOSED:
            # occupant: almost entirely inside the vehicle box and feet clearly
            # above its bottom edge (a pedestrian in front of a bus has their
            # feet at or below the bus's bottom edge, so is kept)
            if overlap > 0.8 and p_bottom < vy + vh - 0.15 * vh:
                return True
    return False


def _to_det(d: dict) -> Detection:
    x, y, w, h = d["box"]
    return Detection(x=int(round(x)), y=int(round(y)), w=int(round(w)), h=int(round(h)),
                     cls=category(d["label"]), confidence=round(d["conf"], 3), label=d["label"])


_detector: Optional[YoloDetector] = None
_detector_lock = threading.Lock()


def get_detector() -> YoloDetector:
    """One network instance for the whole process (loading it is ~0.1 s,
    but there is no reason to repeat it per video)."""
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = _best_detector()
        return _detector


def _best_detector() -> YoloDetector:
    """Largest model the hardware runs at a practical speed: YOLO11x on the
    GPU backend, otherwise YOLO11l on the CPU, otherwise the small fallback."""
    if Path(config.YOLO_MODEL_GPU_PATH).exists() and config.YOLO_BACKEND == "auto":
        det = YoloDetector(config.YOLO_MODEL_GPU_PATH)
        if det.ort is not None:
            return det
    for path in (config.YOLO_MODEL_PATH, config.YOLO_FALLBACK_MODEL_PATH):
        if Path(path).exists():
            return YoloDetector(path)
    raise FileNotFoundError("No object-detection model found in models/.")


def available() -> bool:
    return Path(config.YOLO_MODEL_PATH).exists() or Path(config.YOLO_FALLBACK_MODEL_PATH).exists()


class CameraMotion:
    """
    Estimates the camera's own motion between sampled frames (pan / tilt /
    zoom / shake) from background feature points, with detected road users
    masked out, and accumulates it into a transform that maps each frame's
    pixel coordinates into the FIRST frame's coordinates. Positions measured
    in that stabilized frame are not inflated by camera motion.
    """

    WORK_W = 640

    def __init__(self) -> None:
        self.prev = None
        self.to_ref = np.eye(3)
        self.s = 1.0
        self.shifts: list[float] = []
        self.failures = 0

    def update(self, frame: np.ndarray, boxes: list) -> list:
        h, w = frame.shape[:2]
        self.s = min(1.0, self.WORK_W / w)
        gray = cv2.cvtColor(cv2.resize(frame, (int(w * self.s), int(h * self.s))), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for (x, y, bw, bh) in boxes:
            x0, y0 = int(x * self.s) - 4, int(y * self.s) - 4
            x1, y1 = int((x + bw) * self.s) + 4, int((y + bh) * self.s) + 4
            cv2.rectangle(mask, (max(0, x0), max(0, y0)), (x1, y1), 0, -1)
        if self.prev is not None:
            prev_gray, prev_mask = self.prev
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=prev_mask)
            M = None
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                if good.sum() >= 10:
                    # current -> previous, in working-resolution pixels
                    M, inl = cv2.estimateAffinePartial2D(nxt[good], pts[good], method=cv2.RANSAC,
                                                         ransacReprojThreshold=2.0)
                    if M is not None and (inl is None or inl.sum() < 8):
                        M = None
            if M is None:
                self.failures += 1
                M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            # to full-resolution pixels: same linear part, translation / s
            Mf = np.vstack([M, [0, 0, 1]])
            Mf[0, 2] /= self.s
            Mf[1, 2] /= self.s
            self.shifts.append(float(np.hypot(Mf[0, 2], Mf[1, 2])))
            self.to_ref = self.to_ref @ Mf
        self.prev = (gray, mask)
        return self.to_ref[:2].tolist()

    def summary(self, frame_w: int, fps: float) -> dict:
        if not self.shifts:
            return {"moving_camera": False, "median_shift_px_per_s": 0.0, "estimation_failures": 0}
        med = float(np.median(self.shifts)) * fps
        return {
            "moving_camera": med > 0.01 * frame_w,  # more than 1% of the width per second
            "median_shift_px_per_s": round(med, 1),
            "estimation_failures": self.failures,
        }


def process_video_objects(
    video_path: str | Path,
    detect_fps: Optional[float] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[dict], dict]:
    """
    Decode the whole video but run the detector only on frames sampled at
    ~detect_fps (skipped frames are grabbed, not decoded to images). Returns
    (records, meta); each record is {"frame": source frame index, "t": seconds
    from source fps, "detections": [...], "privacy": [...]}.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE_FPS
    if src_fps <= 0 or src_fps > 240:
        src_fps = config.FRAME_RATE_FPS
    total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = get_detector()
    if detect_fps is None:
        detect_fps = config.DETECT_FPS if det.ort is not None else config.DETECT_FPS_CPU

    step = max(1, int(round(src_fps / detect_fps)))
    records: list[dict] = []
    motion = CameraMotion()
    idx = 0
    while True:
        if idx % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            dets, privacy_boxes = det.detect(frame)
            to_ref = motion.update(frame, [(d.x, d.y, d.w, d.h) for d in dets] + list(privacy_boxes))
            records.append({
                "frame": idx,
                "t": round(idx / src_fps, 4),
                "detections": [d.__dict__ for d in dets],
                "privacy": privacy_boxes,
                "to_ref": to_ref,
            })
        else:
            if not cap.grab():
                break
        idx += 1
        if progress is not None and idx % 15 == 0:
            progress(idx, max(total, idx + 1))
    cap.release()
    if progress is not None:
        progress(idx, idx)
    meta = {
        "src_fps": round(src_fps, 3),
        "frames_total": idx,
        "frames_analyzed": len(records),
        "detect_fps": round(src_fps / step, 2),
        "width": width,
        "height": height,
        "camera": motion.summary(width, src_fps / step),
        "inference_backend": det.backend,
        "model": Path(det.model_path).name,
    }
    return records, meta
