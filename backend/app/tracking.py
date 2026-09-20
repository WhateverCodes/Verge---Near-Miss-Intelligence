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
                if det.cls != track.cls:
                    continue
                dist = _euclidean(track.centroid, det.centroid)
                if dist <= config.MAX_TRACK_DISTANCE_PX:
                    candidate_pairs.append((dist, track_id, det_idx))
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
