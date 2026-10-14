"""
A minimal centroid tracker.

Assigns a persistent integer ID to each detected object across frames by
greedily matching new detections to existing tracks based on nearest
centroid distance. This is intentionally simple (no motion model, no
re-identification) — good enough to stitch together a short demo clip.
Swap in ByteTrack/DeepSORT/Norfair for production use; this module exposes
a single `CentroidTracker.update(detections)` entry point so that's a
localized change (see docs/architecture.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from . import config
from .detection import Detection


@dataclass
class Track:
    track_id: int
    cls: str
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    frames_disappeared: int = 0
    history: List[dict] = field(default_factory=list)
    # per-frame class votes; the track's class is the majority (see config)
    class_votes: Dict[str, int] = field(default_factory=dict)

    def vote(self, cls: str) -> None:
        self.class_votes[cls] = self.class_votes.get(cls, 0) + 1
        best = max(self.class_votes.values())
        if self.class_votes.get(self.cls, 0) < best:
            self.cls = max(self.class_votes, key=self.class_votes.get)


class CentroidTracker:
    def __init__(self) -> None:
        self._next_id = 0
        self._tracks: Dict[int, Track] = {}
        # Every track ever created, including ones later dropped for being
        # stale. build_tracks() reads from here so a track that left the
        # frame (or was lost) before the video ended is not silently
        # discarded — only its *live* matching stops.
        self.all_tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection], frame_idx: int, t: float) -> Dict[int, Track]:
        if not detections:
            for track in self._tracks.values():
                track.frames_disappeared += 1
            self._drop_stale_tracks()
            return dict(self._tracks)

        unmatched_detections = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        # Greedy nearest-neighbor matching, cheapest-first.
        candidate_pairs = []
        for track_id, track in self._tracks.items():
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                penalty = 0.0
                if det.cls != track.cls:
                    if not config.TRACK_ALLOW_CLASS_SWITCH:
                        continue
                    penalty = config.TRACK_CLASS_SWITCH_PENALTY_PX
                dist = _euclidean(track.centroid, det.centroid)
                if dist <= config.MAX_TRACK_DISTANCE_PX:
                    candidate_pairs.append((dist + penalty, track_id, det_idx))
        candidate_pairs.sort(key=lambda p: p[0])

        used_dets: set[int] = set()
        for dist, track_id, det_idx in candidate_pairs:
            if track_id in matched_track_ids or det_idx in used_dets:
                continue
            det = detections[det_idx]
            track = self._tracks[track_id]
            track.centroid = det.centroid
            track.bbox = (det.x, det.y, det.w, det.h)
            track.frames_disappeared = 0
            track.vote(det.cls)
            track.history.append(
                {"frame": frame_idx, "t": t, "x": det.x, "y": det.y, "w": det.w, "h": det.h}
            )
            matched_track_ids.add(track_id)
            used_dets.add(det_idx)

        # Anything left over is either a lost track or a brand-new object.
        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.frames_disappeared += 1

        for det_idx, det in enumerate(detections):
            if det_idx in used_dets:
                continue
            new_id = self._next_id
            self._next_id += 1
            new_track = Track(
                track_id=new_id,
                cls=det.cls,
                centroid=det.centroid,
                bbox=(det.x, det.y, det.w, det.h),
                history=[{"frame": frame_idx, "t": t, "x": det.x, "y": det.y, "w": det.w, "h": det.h}],
                class_votes={det.cls: 1},
            )
            self._tracks[new_id] = new_track
            self.all_tracks[new_id] = new_track

        self._drop_stale_tracks()
        return dict(self._tracks)

    def _drop_stale_tracks(self) -> None:
        stale = [
            tid
            for tid, tr in self._tracks.items()
            if tr.frames_disappeared > config.MAX_FRAMES_DISAPPEARED
        ]
        for tid in stale:
            del self._tracks[tid]


def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def build_tracks(frames: List[dict]) -> List[dict]:
    """
    Run the tracker over a full sequence of per-frame detections (as
    produced by detection.process_video) and flatten the result into a
    single anonymized trajectory list suitable for JSON export:
    [{id, cls, frame, t, x, y, w, h}, ...]
    """
    tracker = CentroidTracker()
    for frame_record in frames:
        dets = [Detection(**d) for d in frame_record["detections"]]
        tracker.update(dets, frame_record["frame"], frame_record["t"])

    flat: List[dict] = []
    for track in tracker.all_tracks.values():
        for point in track.history:
            flat.append({"id": track.track_id, "cls": track.cls, **point})
    flat.sort(key=lambda r: (r["frame"], r["id"]))
    return flat


# ---------------------------------------------------------------------------
# Tracker for the object detector (detection_yolo.py).
#
# The centroid tracker above assumes one detection per object per frame at
# the full frame rate. The object detector runs on sampled frames (~10 fps)
# with real bounding boxes, so this tracker instead:
#   - predicts each track forward with its own (smoothed) velocity,
#   - matches by IoU between predicted and detected boxes (with a
#     size-normalized centre-distance fallback for small, fast objects),
#   - only confirms a track after OBJ_TRACK_MIN_HITS matches (one-off false
#     detections never become road users),
#   - drops a track after OBJ_TRACK_MAX_AGE_S without a match,
#   - suppresses duplicate tracks that sit on the same object.
# Pedestrian and vehicle detections never swap tracks; within vehicles the
# specific label (car/truck/bus/...) is decided by majority vote.
# ---------------------------------------------------------------------------

def _box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ObjTrack:
    track_id: int
    cls: str
    box: Tuple[float, float, float, float]
    last_t: float
    vel: Tuple[float, float] = (0.0, 0.0)   # px/s of the box centre
    hits: int = 1
    history: List[dict] = field(default_factory=list)
    label_votes: Dict[str, int] = field(default_factory=dict)
    alive: bool = True

    def predicted(self, t: float):
        # Extrapolate at most 0.5 s: a coasting track's velocity is a guess,
        # and a long extrapolation drifts away from where the object is.
        dt = min(0.5, max(0.0, t - self.last_t))
        x, y, w, h = self.box
        return (x + self.vel[0] * dt, y + self.vel[1] * dt, w, h)


class ObjectTracker:
    def __init__(self) -> None:
        self._next = 0
        self.tracks: Dict[int, ObjTrack] = {}
        self.all_tracks: Dict[int, ObjTrack] = {}

    def update(self, detections: List[Detection], frame_idx: int, t: float) -> None:
        live = list(self.tracks.values())
        pairs = []
        for tr in live:
            pb = tr.predicted(t)
            pcx, pcy = pb[0] + pb[2] / 2, pb[1] + pb[3] / 2
            for j, d in enumerate(detections):
                if d.cls != tr.cls:
                    continue
                db = (d.x, d.y, d.w, d.h)
                # best of predicted and last-seen box: robust to both fast
                # motion and velocity noise from jittery boxes
                iou = max(_box_iou(pb, db), _box_iou(tr.box, db))
                if iou >= config.OBJ_TRACK_MIN_IOU:
                    score = iou
                else:
                    # small/fast objects: accept a close centre if sizes agree
                    dcx, dcy = d.x + d.w / 2, d.y + d.h / 2
                    size = max(pb[2], pb[3], d.w, d.h)
                    lcx, lcy = tr.box[0] + tr.box[2] / 2, tr.box[1] + tr.box[3] / 2
                    dist = min(((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5,
                               ((lcx - dcx) ** 2 + (lcy - dcy) ** 2) ** 0.5) / size
                    size_ratio = min(pb[2] * pb[3], d.w * d.h) / max(pb[2] * pb[3], d.w * d.h, 1)
                    if dist > 1.0 or size_ratio < 0.35:
                        continue
                    score = 0.1 * (1 - dist)
                if d.label and tr.label_votes and d.label not in tr.label_votes:
                    score *= 0.9  # label disagreement (car vs truck) allowed but penalised
                pairs.append((score, tr.track_id, j))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for score, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            self._match(self.tracks[tid], detections[j], frame_idx, t)

        for j, d in enumerate(detections):
            if j in used_d:
                continue
            # don't start a second track on an object that is already tracked
            db = (d.x, d.y, d.w, d.h)
            if any(tr.cls == d.cls and _box_iou(tr.predicted(t), db) > 0.6 for tr in live):
                continue
            tid = self._next
            self._next += 1
            tr = ObjTrack(track_id=tid, cls=d.cls, box=db, last_t=t,
                          label_votes={d.label: 1} if d.label else {})
            tr.history.append(self._row(d, frame_idx, t))
            self.tracks[tid] = tr
            self.all_tracks[tid] = tr

        for tr in list(self.tracks.values()):
            if t - tr.last_t > config.OBJ_TRACK_MAX_AGE_S:
                tr.alive = False
                del self.tracks[tr.track_id]

    @staticmethod
    def _row(d: Detection, frame_idx: int, t: float) -> dict:
        return {"frame": frame_idx, "t": t, "x": d.x, "y": d.y, "w": d.w, "h": d.h,
                "label": d.label, "confidence": d.confidence}

    def _match(self, tr: ObjTrack, d: Detection, frame_idx: int, t: float) -> None:
        dt = t - tr.last_t
        if dt > 0:
            ocx, ocy = tr.box[0] + tr.box[2] / 2, tr.box[1] + tr.box[3] / 2
            ncx, ncy = d.x + d.w / 2, d.y + d.h / 2
            v = ((ncx - ocx) / dt, (ncy - ocy) / dt)
            a = 0.3 if tr.hits > 1 else 1.0  # heavy smoothing: box centres jitter
            tr.vel = ((1 - a) * tr.vel[0] + a * v[0], (1 - a) * tr.vel[1] + a * v[1])
        tr.box = (d.x, d.y, d.w, d.h)
        tr.last_t = t
        tr.hits += 1
        if d.label:
            tr.label_votes[d.label] = tr.label_votes.get(d.label, 0) + 1
        tr.history.append(self._row(d, frame_idx, t))


def _suppress_duplicates(tracks: List[ObjTrack]) -> List[ObjTrack]:
    """Two confirmed tracks whose boxes overlap strongly (IoU > 0.6) on most
    of their shared frames are the same object: keep the longer one."""
    by_len = sorted(tracks, key=lambda tr: -len(tr.history))
    kept: List[ObjTrack] = []
    for tr in by_len:
        rows = {r["frame"]: r for r in tr.history}
        dup = False
        for k in kept:
            if k.cls != tr.cls:
                continue
            shared = [(rows[r["frame"]], r) for r in k.history if r["frame"] in rows]
            if len(shared) >= 3:
                same = sum(
                    1 for a, b in shared
                    if _box_iou((a["x"], a["y"], a["w"], a["h"]), (b["x"], b["y"], b["w"], b["h"])) > 0.6
                )
                if same >= 0.6 * len(shared):
                    dup = True
                    break
        if not dup:
            kept.append(tr)
    return kept


def build_object_tracks(frames: List[dict]) -> tuple[List[dict], dict]:
    """
    Track object-detector records (detection_yolo.process_video_objects) and
    return (rows, stats). Rows use the same flat format as build_tracks()
    plus "label"; only confirmed tracks are returned, renumbered 1..N in
    order of first appearance so on-screen IDs stay small and readable.
    """
    tracker = ObjectTracker()
    to_ref = {rec["frame"]: rec.get("to_ref") for rec in frames}
    for rec in frames:
        tracker.update([Detection(**d) for d in rec["detections"]], rec["frame"], rec["t"])

    all_tracks = list(tracker.all_tracks.values())
    confirmed_all = [tr for tr in all_tracks if tr.hits >= config.OBJ_TRACK_MIN_HITS]
    confirmed = _suppress_duplicates(confirmed_all)
    confirmed.sort(key=lambda tr: (tr.history[0]["frame"], tr.track_id))

    flat: List[dict] = []
    per_label: Dict[str, int] = {}
    for new_id, tr in enumerate(confirmed, start=1):
        if tr.label_votes:
            label = max(tr.label_votes, key=tr.label_votes.get)
        else:
            label = "person" if tr.cls == "pedestrian" else "vehicle"
        per_label[label] = per_label.get(label, 0) + 1
        for r in tr.history:
            row = {"id": new_id, "cls": tr.cls, **r, "label": label}
            if to_ref.get(r["frame"]) is not None:
                row["to_ref"] = to_ref[r["frame"]]  # camera-motion compensation (conflict_engine)
            flat.append(row)
    flat.sort(key=lambda r: (r["frame"], r["id"]))

    # Peak number of confirmed road users visible at once: a sanity reference
    # for the total count (fragmented IDs would make total >> peak).
    per_frame: Dict[int, int] = {}
    for r in flat:
        per_frame[r["frame"]] = per_frame.get(r["frame"], 0) + 1
    stats = {
        "raw_tracks": len(all_tracks),
        "confirmed_tracks": len(confirmed),
        "unconfirmed_dropped": len(all_tracks) - len(confirmed_all),
        "duplicates_merged": len(confirmed_all) - len(confirmed),
        "per_label": per_label,
        "peak_simultaneous": max(per_frame.values()) if per_frame else 0,
        "mean_detections_per_frame": round(
            sum(len(r["detections"]) for r in frames) / max(1, len(frames)), 1
        ),
    }
    return flat, stats
