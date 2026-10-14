import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.conflict import find_near_miss_events, _severity
from backend.app import config


def _make_track(track_id: int, cls: str, positions: list[tuple[int, int, int]]) -> list[dict]:
    """positions: list of (frame, x, y). Returns rows for one track."""
    rows = []
    for frame, x, y in positions:
        rows.append(
            {
                "id": track_id,
                "cls": cls,
                "frame": frame,
                "t": frame / config.FRAME_RATE_FPS,
                "x": x,
                "y": y,
                "w": 20,
                "h": 20,
            }
        )
    return rows


def test_close_crossing_paths_flagged_as_near_miss():
    # Vehicle moves right along y=100; pedestrian moves down through x=100,
    # crossing paths near frame 5 — should trigger a near-miss.
    vehicle = _make_track(0, "vehicle", [(f, 20 * f, 100) for f in range(10)])
    pedestrian = _make_track(1, "pedestrian", [(f, 100, 15 * f) for f in range(10)])

    events = find_near_miss_events(vehicle + pedestrian, site_id="TEST_SITE")

    assert len(events) >= 1
    event = events[0]
    assert event.vehicle_id == 0
    assert event.pedestrian_id == 1
    assert event.distance_m <= config.DIST_THRESH_M
    assert 0.0 <= event.severity <= 1.0


def test_far_apart_paths_not_flagged():
    # Vehicle and pedestrian never come close.
    vehicle = _make_track(0, "vehicle", [(f, 20 * f, 50) for f in range(10)])
    pedestrian = _make_track(1, "pedestrian", [(f, 20 * f, 400) for f in range(10)])

    events = find_near_miss_events(vehicle + pedestrian, site_id="TEST_SITE")
    assert events == []


def test_severity_increases_as_distance_and_ttc_shrink():
    close_severe = _severity(distance_m=0.1, ttc_s=0.1)
    far_mild = _severity(distance_m=2.9, ttc_s=1.9)
    assert close_severe > far_mild
    assert 0.0 <= close_severe <= 1.0
    assert 0.0 <= far_mild <= 1.0
