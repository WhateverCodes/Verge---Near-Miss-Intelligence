from backend.app import config
from backend.app.conflict import (
    encounter_metrics,
    enrich_events,
    find_all_conflicts,
    find_near_miss_events,
)

from .conftest import make_track


def test_peak_and_min_distance_come_from_the_whole_encounter():
    # Same crossing as test_conflict.py: the trigger frame is the first one
    # inside the zone; the closest approach happens later.
    vehicle = make_track(0, "vehicle", [(f, 20 * f, 100) for f in range(10)])
    ped = make_track(1, "pedestrian", [(f, 100, 15 * f) for f in range(10)])
    tracks = vehicle + ped
    trigger = find_near_miss_events(tracks, site_id="T")[0]

    m = encounter_metrics(tracks, 0, 1)
    assert m["min_distance_m"] <= trigger.distance_m
    assert m["peak_severity"] >= trigger.severity
    assert m["closing_observed"] is True
    assert m["min_ttc_s"] is not None and m["min_ttc_s"] <= config.TTC_THRESH_S
    assert m["zone_duration_s"] > 0


def test_separating_pair_reports_no_ttc():
    # Start close together and move apart: never closing -> TTC not reported.
    a = make_track(0, "vehicle", [(f, 100 + 10 * f, 100) for f in range(8)])
    b = make_track(1, "pedestrian", [(f, 90 - 5 * f, 100) for f in range(8)])
    m = encounter_metrics(a + b, 0, 1)
    assert m["closing_observed"] is False
    assert m["min_ttc_s"] is None


def test_pet_only_when_paths_share_a_point():
    # Vehicle passes x=200 at frame 10; pedestrian reaches the same point at
    # frame 20 while the vehicle is still tracked -> PET ~ 10 frames.
    vehicle = make_track(0, "vehicle", [(f, 20 * f, 200) for f in range(30)])
    ped = make_track(1, "pedestrian", [(f, 200, 200 - 60 + 3 * f) for f in range(30)])
    m = encounter_metrics(vehicle + ped, 0, 1)
    assert m["pet_s"] is not None
    assert abs(m["pet_s"] - 10 / config.FRAME_RATE_FPS) < 0.2

    # Parallel paths 100 px apart never share a point -> PET not applicable.
    far = make_track(2, "pedestrian", [(f, 20 * f, 300) for f in range(30)])
    assert encounter_metrics(vehicle + far, 0, 2)["pet_s"] is None


def test_truncated_boxes_are_ignored_when_frame_size_known():
    # Pedestrian box touches the left frame edge for the whole encounter.
    vehicle = make_track(0, "vehicle", [(f, 5 + 5 * f, 100) for f in range(10)])
    ped = make_track(1, "pedestrian", [(f, 0, 110) for f in range(10)])
    assert encounter_metrics(vehicle + ped, 0, 1)["min_distance_m"] is not None
    m = encounter_metrics(vehicle + ped, 0, 1, frame_size=(640, 480))
    assert m["min_distance_m"] is None and m["closing_observed"] is False


def test_enrich_keeps_original_keys():
    vehicle = make_track(0, "vehicle", [(f, 20 * f, 100) for f in range(10)])
    ped = make_track(1, "pedestrian", [(f, 100, 15 * f) for f in range(10)])
    events = find_all_conflicts(vehicle + ped, site_id="T")
    before = {k: events[0][k] for k in ("frame", "t", "distance_m", "ttc_s", "severity", "kind")}
    enrich_events(vehicle + ped, events)
    for k, v in before.items():
        assert events[0][k] == v
    assert events[0]["severity_label"] in {"Low", "Moderate", "High"}
