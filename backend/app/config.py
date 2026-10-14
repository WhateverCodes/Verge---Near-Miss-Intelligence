"""
Central, tunable constants for the pipeline.

Nothing here is "correct" out of the box — these are reasonable demo
defaults. Real deployments should calibrate DIST_THRESH_M / TTC_THRESH_S
against local speed limits and camera calibration (pixels-per-meter).
"""
from pathlib import Path

# ---- paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

SAMPLE_VIDEO_PATH = DATA_DIR / "simulated_traffic.mp4"
SAMPLE_ACCIDENTS_CSV = DATA_DIR / "sample_accidents.csv"

# Where videos uploaded via POST /api/analyze-video are saved, and where
# their annotated output clips are written (see video_analysis.py).
UPLOADS_DIR = OUTPUT_DIR / "uploads"

TRACKS_JSON = OUTPUT_DIR / "tracks.json"
NEAR_MISS_JSON = OUTPUT_DIR / "near_miss_events.json"
HOTSPOTS_GEOJSON = OUTPUT_DIR / "hotspots.geojson"
REPORT_MD = OUTPUT_DIR / "report.md"

# Location-linked video analyses (see site_evidence.py / analysis_service.py).
# Each analysis gets its own folder of evidence (replay frames, snapshot,
# annotated clip, analysis.json). OBSERVATIONS_JSON is the index of
# AI-observed evidence per site; it is kept separate from the historical
# accident CSV, which is never modified.
ANALYSES_DIR = OUTPUT_DIR / "analyses"
OBSERVATIONS_JSON = OUTPUT_DIR / "observations.json"
CUSTOM_SITES_JSON = OUTPUT_DIR / "custom_sites.json"

# Label shown wherever the bundled historical figures are displayed.
HISTORICAL_DATA_LABEL = "synthetic MoRTH-style sample data (data/sample_accidents.csv)"

# ---- detection ---------------------------------------------------------
MIN_CONTOUR_AREA = 350          # px^2, ignore smaller noise blobs
# crude vehicle-vs-pedestrian heuristic thresholds (bbox area, px^2)
VEHICLE_MIN_AREA = 2200
# aspect ratio (width/height) above this looks "car-shaped" (wide) rather
# than "person-shaped" (tall/narrow)
VEHICLE_MIN_ASPECT = 1.3

# ---- object detector (real footage) -----------------------------------------
# Pretrained YOLOv8n (COCO) exported to ONNX and run with OpenCV's DNN module,
# so no extra Python dependency and no download at run time (models/README.md).
# YOLO11-large (COCO). Chosen over YOLOv8n after a side-by-side test on real
# clips: v8n found ~17% of the vehicles in a steep overhead CCTV clip, YOLO11l
# ~83% (docs/cv-benchmark.md). yolov8n.onnx is kept as a fast fallback.
# Model per hardware: YOLO11x with the GPU backend (36 ms/frame on an RTX 4060),
# YOLO11l on a CPU-only machine (~250 ms/frame). yolov8n is a last-resort fallback.
YOLO_MODEL_GPU_PATH = ROOT_DIR / "models" / "yolo11x.onnx"
YOLO_MODEL_PATH = ROOT_DIR / "models" / "yolo11l.onnx"
YOLO_FALLBACK_MODEL_PATH = ROOT_DIR / "models" / "yolov8n.onnx"
YOLO_INPUT_SIZE = 640
# Inference backend: "auto" uses ONNX Runtime + DirectML (any DirectX 12 GPU on
# Windows, no CUDA install) when the optional package onnxruntime-directml is
# installed and works, otherwise OpenCV DNN on the CPU. "opencv" forces the CPU
# path. Both run the same ONNX file; outputs were checked to match.
YOLO_BACKEND = "auto"
# Per-class confidence thresholds (tuned on real traffic clips, see
# docs/cv-benchmark.md). NMS is class-wise, then class-agnostic across the
# vehicle classes so one truck can't also be reported as a car/bus.
YOLO_CONF = {"person": 0.35, "bicycle": 0.30, "car": 0.30, "motorcycle": 0.30, "bus": 0.25, "truck": 0.25}
YOLO_NMS_IOU = 0.45
YOLO_VEHICLE_MERGE_IOU = 0.70
# Ignore boxes smaller than this fraction of the frame height (too far away
# for a meaningful size-based distance estimate).
YOLO_MIN_BOX_H_FRAC = 0.025
# Detection runs on sampled frames at about this rate; tracks are
# interpolated in between for rendering. Source fps is used for all timing.
DETECT_FPS = 15.0       # with the GPU backend
DETECT_FPS_CPU = 10.0   # CPU-only: fewer frames so analysis stays practical

# ---- tracking ------------------------------------------------------------
MAX_TRACK_DISTANCE_PX = 80      # max centroid jump between frames to match
MAX_FRAMES_DISAPPEARED = 15     # frames a track can go unmatched before drop
# A detection may continue a track of the OTHER class (e.g. a car that is
# only partly in view when it enters the frame looks narrow, i.e.
# "pedestrian"-shaped, for a few frames). Such matches pay this extra cost,
# so a same-class track is always preferred, and a track's final class is the
# majority of its per-frame classifications. Set False for the original
# strict same-class matching.
TRACK_ALLOW_CLASS_SWITCH = True
TRACK_CLASS_SWITCH_PENALTY_PX = 30

# Object-detector tracker (tracking.ObjectTracker): IoU matching against a
# constant-velocity prediction, confirmed only after MIN_HITS matches,
# dropped after MAX_AGE_S without a match.
OBJ_TRACK_MIN_IOU = 0.15
OBJ_TRACK_MIN_HITS = 4
OBJ_TRACK_MAX_AGE_S = 1.5

# ---- conflict indicators --------------------------------------------------
# Pixels-per-meter is camera-calibration dependent; this is a placeholder
# for the synthetic demo clip.
PIXELS_PER_METER = 12.0
FRAME_RATE_FPS = 20.0

DIST_THRESH_M = 8.0              # objects closer than this = "encounter zone"
TTC_THRESH_S = 3.0               # TTC below this = flagged near-miss

# PET (post-encroachment time) is only reported when the two centroid paths
# actually pass within this distance of each other (i.e. they share a
# conflict point). Otherwise PET is left as "not applicable" — never guessed.
PET_RADIUS_M = 1.5

# A near-miss requires the two road users to be observed CLOSING on each
# other (separation shrinking over consecutive frames). Pairs that only ever
# appear inside the encounter zone without any measurable approach (typically
# a short-lived tracking fragment next to its own vehicle) are set aside and
# reported separately instead of being counted. False = original behaviour.
REQUIRE_CLOSING_MOTION = True

# ---- conflict engine for the object detector (conflict_engine.py) ------------
# Distances are ground-plane estimates (bottom-centre of each box, scaled by
# the objects' typical real-world heights) minus each road user's half width,
# i.e. approximate edge-to-edge clearance, not centre distance.
CONFLICT_MAX_GAP_M = 3.0         # clearance at or below this can be a conflict
CONFLICT_TTC_S = 1.5             # TTC at or below this (on a collision course) can be a conflict
CONFLICT_TTC_VV_S = 0.8          # stricter for vehicle-vehicle: dense traffic is close all the time
CONFLICT_MIN_DURATION_VV_S = 0.4
# Both road users must appear at least this tall (fraction of frame height):
# for smaller, distant objects a few pixels of box jitter are metres of depth.
CONFLICT_MIN_BOX_H_FRAC = 0.09
CONFLICT_MAX_OCCLUDED_FRAC = 0.3  # set aside if boxes overlap on screen in more of the run
# Following / queuing: two road users heading the same way (within this
# angle) are only a conflict if the gap closes at least this fast. Ordinary
# following closes slowly; a hard approach (or a lead vehicle braking) doesn't.
CONFLICT_SAME_DIRECTION_DEG = 30.0
CONFLICT_SAME_DIRECTION_MIN_CLOSING_MS = 3.0
# Track consistency around an event (+-1 s): box height may change by at most
# this ratio between consecutive samples in 80% of steps, and at most one
# step may imply a speed above the plausibility limit. Jumps beyond that mean
# the tracker switched objects or the box was cut off, not real motion.
TRACK_MAX_HEIGHT_STEP = 1.25
CONFLICT_MIN_REL_SPEED_MS = 1.0  # m/s; slower relative motion is within estimate noise
# The gap between the two must actually be shrinking this fast (m/s, measured
# from the gap itself). A car rolling to a stop behind another closes slowly;
# a near-miss is a fast approach.
CONFLICT_MIN_CLOSING_VV_MS = 3.0
# Second indicator, "close pass" (proximity at speed): road users moving side
# by side / past each other are never on a head-on collision course, so TTC
# can't flag them. A close pass is a clearance at or below this while at least
# one of them is moving, sustained for CONFLICT_CLOSE_PASS_MIN_S.
CONFLICT_CLOSE_PASS_VV_M = 0.35
CONFLICT_CLOSE_PASS_VRU_M = 0.5
CONFLICT_CLOSE_PASS_MIN_SPEED_MS = 1.5
CONFLICT_CLOSE_PASS_MIN_S = 0.4
CONFLICT_CLOSE_PASS_MIN_REL_MS = 2.5   # a real pass: one sweeps past the other
# Allow a fast collision-course approach to continue into footprint contact.
CONFLICT_ALLOW_CONTACT_PHASE = "overhead"  # True / False / "overhead" (only near-overhead views)
# Evasive action (the hallmark of a traffic conflict): within +-1 s of the event
# one of the two road users must brake or swerve, i.e. change its velocity
# vector by at least this much within 0.8 s. 0 disables the requirement.
CONFLICT_EVASIVE_DV_MS = 0.0
CONFLICT_MIN_CLOSING_VRU_MS = 1.0
CONFLICT_MISS_MARGIN_M = 0.5     # collision course: predicted closest approach within this
CONFLICT_MIN_APPROACH_M = 0.5    # pair must have closed in by at least this before the run
# Observed closeness: a predicted contact only counts if the pair really came
# this close during the event (noisy box motion can predict contacts that
# never come anywhere near happening).
CONFLICT_MAX_OBSERVED_GAP_VV_M = 1.0
CONFLICT_MAX_OBSERVED_GAP_VRU_M = 2.0
CONFLICT_MIN_DURATION_S = 0.3    # indicator must hold this long to be confirmed
CONFLICT_MERGE_GAP_S = 2.0       # same pair within this gap = the same event
CONFLICT_MIN_TRACK_S = 0.5       # both tracks must have existed this long

# Severity bands used for labels in the UI / reports (severity is 0-1).
SEVERITY_BANDS = [(0.66, "High"), (0.33, "Moderate"), (0.0, "Low")]

# ---- evidence replay ---------------------------------------------------------
# Seconds of footage kept before / after each conflict's peak moment. The
# window is clamped to the clip, and shifted (not shrunk) near clip edges.
EVIDENCE_PRE_S = 1.5
EVIDENCE_POST_S = 1.0
EVIDENCE_MAX_EVENTS = 15         # replays are rendered for the N most severe events
EVIDENCE_MAX_WIDTH = 960         # px, replay frames are downscaled to this width
MIN_CLIP_S = 1.0                 # shorter clips can't establish a background model

# The synthetic demo clip stands in for one physical camera site.
# In a real deployment each camera/intersection has its own site_id + coords.
DEMO_SITE = {
    "site_id": "SITE_DEMO_01",
    "name": "Demo Intersection (synthetic clip)",
    "lat": 18.5204,
    "lon": 73.8567,  # Pune, as an example coordinate
}

# ---- risk scoring ----------------------------------------------------------
# Composite score weights — tune against real outcome data when available.
W_ACCIDENTS = 0.35
W_FATALITIES = 0.30
W_NEAR_MISS = 0.25
W_SEVERITY = 0.10

# Review-priority bands on the existing 0-1 score (same score, same weights).
# Why not quarters (0.25/0.5/0.75): crash history carries W_ACCIDENTS +
# W_FATALITIES = 0.65 of the score, so a site WITHOUT AI-observed evidence can
# score at most 0.65. With a 0.75 "Critical" line, even the site with the worst
# crash and fatality record in the dataset (Swargate: 25 crashes, 6 deaths,
# score 0.65) could never be Critical. The bands below keep equal 0.2 steps up
# to Critical at 0.60, which a site reaches only when it is at or near the
# network's worst on BOTH crashes and fatalities (>= ~90% of the maximum), or
# combines a strong crash history with observed near-misses.
RISK_TIERS = [
    (0.6, "Critical"),
    (0.4, "High"),
    (0.2, "Moderate"),
    (0.0, "Low"),
]


def risk_tier(score: float) -> str:
    for threshold, label in RISK_TIERS:
        if score >= threshold:
            return label
    return "Low"


def severity_band(severity: float) -> str:
    for threshold, label in SEVERITY_BANDS:
        if severity >= threshold:
            return label
    return "Low"

# Perspective (ground-plane) correction of along-road gaps, fitted per clip
# from the vehicles' own sizes (conflict_engine._ground_model).
CONFLICT_GROUND_CORRECTION = True
# Near-overhead views: fraction of the box (from the bottom) used as the ground footprint.
CONFLICT_OVERHEAD_FOOTPRINT_FRAC = 0.75
# Vehicle-vehicle closing speed needed in near-overhead views, where footprint
# gaps are measured directly (no depth ambiguity).
CONFLICT_MIN_CLOSING_VV_OVERHEAD_MS = 1.0
# Oblique views: measure gaps/TTC in a bird's-eye (road-plane) view built from the ground model.
CONFLICT_BIRD_EYE = True
# Conflicts are not measured when the camera travels more than this fraction of
# the frame size, or zooms by more than this, over the clip.
CONFLICT_MAX_CAMERA_TRAVEL_FRAC = 0.5
CONFLICT_MAX_CAMERA_ZOOM = 0.5
# Minimum road-user size is measured against the shorter frame side (portrait video).
CONFLICT_MIN_SIZE_SHORT_SIDE = True

# ---- crash (observed collision) detection -----------------------------------
# A crash is reported only with all three: footprints meet (and boxes overlap),
# after a fast approach, followed by an impact signature. See
# conflict_engine._find_crashes. Hard braking is at most ~0.8-1 g, i.e. a speed
# change of about 5-6 m/s in 0.6 s, so a larger change there is an impact.
CRASH_DETECTION = True
CRASH_MIN_APPROACH_M = 1.5     # the gap closed by at least this much ...
CRASH_MIN_CLOSING_MS = 4.0     # ... at at least this speed, right before contact
CRASH_DV_MS = 6.0              # impact: sustained velocity change across the contact ...
CRASH_SHAPE_CHANGE = 1.6       # ... or box aspect ratio changing by this factor (spin / rollover)
CRASH_MIN_SPEED_MS = 5.0       # one of the pair was moving at least this fast before contact
CRASH_MAX_DEPTH_TOL_M = 1.0    # contact only counted where depth is measurable to within ~1 m
CRASH_MIN_CONTACT_S = 0.3      # the footprints stay together at least this long ...
CRASH_MIN_BOX_OVERLAP = 0.3    # ... with the boxes overlapping (median, share of the smaller box)
CRASH_MAX_SPEED_MS = 30.0      # a faster measured speed before contact is measurement error
