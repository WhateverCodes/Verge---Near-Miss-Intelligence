"""Object-detector path: tracker, conflict engine, detector-selection rules."""
import cv2
import numpy as np
import pytest

from backend.app import config, conflict_engine, detection_yolo, tracking, video_analysis
from backend.app.detection import Detection

FPS = 10.0  # detection sample rate used by the object path
FW, FH = 1280, 720


def _records(boxes_per_frame):
    """boxes_per_frame: list (per sample) of lists of (x, y, w, h, label)."""
    recs = []
    for i, boxes in enumerate(boxes_per_frame):
        dets = [Detection(x=int(x), y=int(y), w=int(w), h=int(h),
                          cls="pedestrian" if lab == "person" else "vehicle", confidence=0.9, label=lab).__dict__
                for x, y, w, h, lab in boxes]
        recs.append({"frame": i * 3, "t": i / FPS, "detections": dets, "privacy": []})
    return recs


def _events(boxes_per_frame):
    rows, stats = tracking.build_object_tracks(_records(boxes_per_frame))
    events, diag = conflict_engine.find_conflict_events(rows, (FW, FH), "T")
    return rows, stats, events, diag


# ---- tracker ------------------------------------------------------------------

def test_tracker_keeps_one_id_through_missed_detections():
    frames = []
    for i in range(30):
        # a car moving right; detector misses it on every 5th sample
        frames.append([] if i % 5 == 4 else [(100 + 12 * i, 400, 160, 90, "car")])
    rows, stats = tracking.build_object_tracks(_records(frames))
    assert stats["confirmed_tracks"] == 1
    assert {r["label"] for r in rows} == {"car"}


def test_one_off_false_detection_is_never_confirmed():
    frames = [[(500, 300, 60, 150, "person")] if i == 7 else [] for i in range(20)]
    rows, stats = tracking.build_object_tracks(_records(frames))
    assert stats["confirmed_tracks"] == 0 and rows == []


def test_label_flicker_is_resolved_by_majority_vote():
    frames = [[(100 + 10 * i, 400, 160, 90, "truck" if i % 4 == 0 else "car")] for i in range(20)]
    rows, stats = tracking.build_object_tracks(_records(frames))
    assert stats["confirmed_tracks"] == 1
    assert {r["label"] for r in rows} == {"car"}


# ---- conflict engine ------------------------------------------------------------

def _crossing(ped_speed_px=6, car_speed_px=25, n=40):
    """A car driving right along the bottom of the frame and a pedestrian
    walking down across its path; the car reaches the crossing point just
    after the pedestrian, i.e. a genuine collision course."""
    frames = []
    for i in range(n):
        car = (40 + car_speed_px * i, 560, 200, 110, "car")
        ped = (600, 380 + ped_speed_px * i, 60, 170, "person")
        frames.append([car, ped])
    return frames


def test_collision_course_crossing_is_one_confirmed_event():
    rows, stats, events, diag = _events(_crossing())
    assert len(events) == 1
    e = events[0]
    assert e["kind"] == "vehicle_pedestrian"
    assert e["zone_start_t"] <= e["peak_t"] <= e["zone_end_t"]
    assert e["min_ttc_s"] <= config.CONFLICT_TTC_S
    assert e["confirmed_samples"] >= 2


def test_side_by_side_traffic_is_not_a_conflict():
    # two cars in adjacent lanes, same speed and direction, 3 m apart laterally
    frames = [[(100 + 20 * i, 500, 200, 110, "car"), (100 + 20 * i, 640, 200, 110, "car")] for i in range(40)]
    assert _events(frames)[2] == []


def test_pedestrians_walking_together_are_not_a_conflict():
    frames = [[(300 + 8 * i, 400, 60, 170, "person"), (380 + 8 * i, 400, 60, 170, "person")] for i in range(40)]
    assert _events(frames)[2] == []


def test_parked_vehicles_with_box_jitter_are_not_a_conflict():
    rng = np.random.default_rng(0)
    frames = [[(400 + rng.integers(-3, 4), 500 + rng.integers(-3, 4), 200, 110, "car"),
               (620 + rng.integers(-3, 4), 500 + rng.integers(-3, 4), 200, 110, "car")] for _ in range(50)]
    assert _events(frames)[2] == []


def test_distant_small_objects_are_not_measured():
    # the same crossing, scaled down to 4% of the frame height: too far to trust
    frames = [[(40 + 5 * i, 300, 40, 22, "car"), (160, 260 + 1 * i, 12, 28, "person")] for i in range(40)]
    assert _events(frames)[2] == []


# ---- detector selection -------------------------------------------------------

def test_synthetic_clip_is_recognised_and_real_texture_is_not(demo_clip, tmp_path):
    assert video_analysis.looks_synthetic(demo_clip) is True
    noisy = tmp_path / "noise.mp4"
    rng = np.random.default_rng(1)
    wr = cv2.VideoWriter(str(noisy), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    base = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for i in range(20):
        wr.write(np.clip(base.astype(int) + rng.integers(-10, 10, base.shape), 0, 255).astype(np.uint8))
    wr.release()
    assert video_analysis.looks_synthetic(noisy) is False


@pytest.mark.skipif(not detection_yolo.available(), reason="model file not present")
def test_object_detector_loads_and_finds_nothing_in_a_blank_frame():
    dets, privacy = detection_yolo.get_detector().detect(np.full((720, 1280, 3), 90, np.uint8))
    assert dets == [] and privacy == []


def test_replay_window_is_built_from_the_event_timestamps():
    e = {"t": 7.1, "zone_start_t": 7.1, "zone_end_t": 8.4, "peak_t": 7.8}
    start, end = video_analysis._evidence_window(e, fps=30, n_frames=30 * 20)
    assert start / 30 <= 6.2 and end / 30 >= 9.3        # context before/after the event
    assert start / 30 <= 7.1 and end / 30 >= 8.4         # the whole event is inside
    e_early = {"t": 0.2, "zone_start_t": 0.2, "zone_end_t": 0.6, "peak_t": 0.4}
    s2, e2 = video_analysis._evidence_window(e_early, fps=30, n_frames=300)
    assert s2 == 0 and e2 / 30 >= 1.4


# ---- following / queuing, track consistency, backend parity ---------------------

def _following(follower_px_per_sample, n=40):
    """Two small cars in the same lane, same direction; the lead moves 10 px
    per sample, the follower faster (car height 66 px ~ 44 px/m)."""
    return [[(400 + 10 * i, 520, 120, 66, "car"), (10 + follower_px_per_sample * i, 520, 120, 66, "car")]
            for i in range(n)]


def test_normal_following_is_not_a_conflict():
    # follower gains ~2.5 m/s on the lead: an ordinary close follow, not a hard approach
    assert _events(_following(21))[2] == []


def test_hard_approach_from_behind_is_a_conflict():
    # follower gains ~6.8 m/s on the lead, then brakes hard just short of it:
    # a genuine rear-end near-miss
    frames, fx = [], 10
    for i in range(40):
        lead = 400 + 10 * i
        fx = fx + 40 if lead - (fx + 120) > 45 else fx + 10
        frames.append([(lead, 520, 120, 66, "car"), (fx, 520, 120, 66, "car")])
    events = _events(frames)[2]
    assert len(events) == 1 and events[0]["kind"] == "vehicle_vehicle"


def test_erratic_tracks_are_set_aside():
    # the same crossing as above, but the car's box size jumps around (the
    # tracker hopping between objects): not usable as conflict evidence
    frames = []
    for i, (car, ped) in enumerate(_crossing()):
        x, y, w, h, lab = car
        k = 1.6 if i % 2 else 1.0
        frames.append([(x, y, w * k, h * k, lab), ped])
    rows, stats, events, diag = _events(frames)
    assert events == []


@pytest.mark.skipif(not detection_yolo.available(), reason="model file not present")
def test_gpu_and_cpu_backends_agree(demo_clip, monkeypatch):
    det = detection_yolo.get_detector()
    if det.ort is None:
        pytest.skip("DirectML backend not active on this machine")
    cap = cv2.VideoCapture(str(demo_clip))
    ok, frame = cap.read()
    cap.release()
    frame = cv2.resize(frame, (1280, 720))
    gpu = det._infer(frame)
    monkeypatch.setattr(config, "YOLO_BACKEND", "opencv")
    cpu = detection_yolo.YoloDetector()._infer(frame)
    assert [d["label"] for d in gpu] == [d["label"] for d in cpu]


# ---- view geometry: perspective, overhead, camera motion, ID jumps ----------------

def _perspective_box(u, v_bottom, label="car", y_h=300.0, a=0.5):
    """A box for a road user whose ground contact is at image point (u, v_bottom)
    in a street-level view with the horizon at row y_h (scale a*(v - y_h) px/m)."""
    s = a * (v_bottom - y_h)
    w, h = (1.8 * s, 1.5 * s) if label == "car" else (0.55 * s, 1.7 * s)
    return (u - w / 2, v_bottom - h, w, h, label)


def test_street_level_cars_in_different_lanes_are_not_a_conflict():
    # Two cars pass in opposite directions, one ~4 m further from the camera.
    # On screen their boxes almost touch (the far one sits just above the near
    # one), but on the road they are a lane apart.
    frames = []
    for i in range(40):
        near = _perspective_box(200 + 20 * i, 520)
        far = _perspective_box(1000 - 16 * i, 470)
        frames.append([near, far])
    rows, stats, events, diag = _events(frames)
    assert diag["ground_model"].get("bird_eye") is True
    assert events == []


def test_moving_camera_reports_no_conflicts():
    recs = _records(_crossing())
    for i, r in enumerate(recs):  # the camera pans ~40 px per sample
        r["to_ref"] = [[1.0, 0.0, 40.0 * i], [0.0, 1.0, 0.0]]
    rows, _ = tracking.build_object_tracks(recs)
    events, diag = conflict_engine.find_conflict_events(rows, (FW, FH), "T")
    assert events == [] and diag["camera_too_unstable"] is True


def test_track_jump_is_not_read_as_speed():
    # a parked car and a pedestrian standing beside it; mid-clip the tracker's
    # box for the "car" jumps to a different car (ID switch). The jump must not
    # be read as motion, so there is no collision course.
    frames = []
    for i in range(40):
        car = (300, 500, 200, 110, "car") if i < 20 else (900, 500, 200, 110, "car")
        frames.append([car, (1150, 440, 60, 170, "person")])
    rows, stats, events, diag = _events(frames)
    assert events == []


# ---- crash (observed collision) ------------------------------------------------

def _rear_end(stop_x):
    """A car at ~9 m/s (80 px per 0.1 s sample; box 160 px ~ 1.8 m) drives into
    a stopped car and stops dead at stop_x; the stopped car is shunted a little."""
    frames, x = [], 0
    for i in range(40):
        x = min(x + 80, stop_x)
        hit = x >= stop_x
        lead_x = 700 + (min(i - 13, 3) * 12 if hit and i >= 13 else 0)
        frames.append([(x, 520, 160, 90, "car"), (lead_x, 520, 160, 90, "car")])
    return frames


def test_high_speed_impact_is_reported_as_a_crash():
    events = _events(_rear_end(stop_x=640))[2]   # ends overlapping the stopped car by 100 px
    crashes = [e for e in events if e.get("event_type") == "crash"]
    assert len(crashes) == 1
    c = crashes[0]
    assert c["severity"] == 1.0 and c["indicator"] == "collision"
    assert c["impact_dv_ms"] >= config.CRASH_DV_MS
    assert not [e for e in events if e.get("event_type") != "crash"]  # the crash supersedes the near-miss


def test_hard_stop_short_of_the_other_car_is_not_a_crash():
    events = _events(_rear_end(stop_x=520))[2]   # stops 20 px short: no contact
    assert not [e for e in events if e.get("event_type") == "crash"]


def test_tracker_hopping_between_vehicles_is_not_a_crash():
    # the "moving car" box jumps onto the parked car and stays there: an ID
    # switch looks like a sudden stop in contact, but the box never overlaps
    # its previous position, so identity is not kept
    frames = []
    for i in range(40):
        x = 80 * i if i < 8 else 700
        frames.append([(x, 520, 160, 90, "car"), (700, 520, 160, 90, "car")])
    events = _events(frames)[2]
    assert not [e for e in events if e.get("event_type") == "crash"]
