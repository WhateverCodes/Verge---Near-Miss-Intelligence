from backend.app import config
from backend.app.detection import Detection
from backend.app.tracking import build_tracks


def _frames(dets_per_frame):
    return [
        {"frame": i, "t": i / config.FRAME_RATE_FPS, "detections": [d.__dict__ for d in dets]}
        for i, dets in enumerate(dets_per_frame)
    ]


def _entering_car():
    # A car entering from the left edge: narrow ("pedestrian"-shaped) for the
    # first frames, then full width.
    out = []
    for i in range(12):
        w = min(80, 15 + 15 * i)
        out.append([Detection(x=max(0, 10 * i - 20), y=100, w=w, h=40, cls="vehicle" if w / 40 >= 1.3 and w * 40 >= 2200 else "pedestrian", confidence=1.0)])
    return out


def test_class_flip_does_not_split_the_track():
    tracks = build_tracks(_frames(_entering_car()))
    ids = {r["id"] for r in tracks}
    assert len(ids) == 1
    assert {r["cls"] for r in tracks} == {"vehicle"}  # majority vote


def test_strict_same_class_matching_still_available(monkeypatch):
    monkeypatch.setattr(config, "TRACK_ALLOW_CLASS_SWITCH", False)
    tracks = build_tracks(_frames(_entering_car()))
    assert len({r["id"] for r in tracks}) == 2


def test_same_class_track_is_preferred_over_other_class():
    # A pedestrian stands still next to a passing vehicle: each detection must
    # stay on its own track.
    frames = []
    for i in range(10):
        frames.append([
            Detection(x=10 * i, y=100, w=80, h=40, cls="vehicle", confidence=1.0),
            Detection(x=60, y=150, w=18, h=34, cls="pedestrian", confidence=1.0),
        ])
    tracks = build_tracks(_frames(frames))
    by_id = {}
    for r in tracks:
        by_id.setdefault(r["id"], set()).add(r["cls"])
    assert sorted(len(v) for v in by_id.values()) == [1, 1]
    assert len(by_id) == 2
