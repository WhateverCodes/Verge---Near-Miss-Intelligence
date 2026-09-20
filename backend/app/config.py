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

# ---- detection ---------------------------------------------------------
MIN_CONTOUR_AREA = 350          # px^2, ignore smaller noise blobs
# crude vehicle-vs-pedestrian heuristic thresholds (bbox area, px^2)
VEHICLE_MIN_AREA = 2200
# aspect ratio (width/height) above this looks "car-shaped" (wide) rather
# than "person-shaped" (tall/narrow)
VEHICLE_MIN_ASPECT = 1.3

# ---- tracking ------------------------------------------------------------
MAX_TRACK_DISTANCE_PX = 80      # max centroid jump between frames to match
MAX_FRAMES_DISAPPEARED = 15     # frames a track can go unmatched before drop

# ---- conflict indicators --------------------------------------------------
# Pixels-per-meter is camera-calibration dependent; this is a placeholder
# for the synthetic demo clip.
PIXELS_PER_METER = 12.0
FRAME_RATE_FPS = 20.0

DIST_THRESH_M = 8.0              # objects closer than this = "encounter zone"
TTC_THRESH_S = 3.0               # TTC below this = flagged near-miss

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

RISK_TIERS = [
    (0.75, "Critical"),
    (0.5, "High"),
    (0.25, "Moderate"),
    (0.0, "Low"),
]


def risk_tier(score: float) -> str:
    for threshold, label in RISK_TIERS:
        if score >= threshold:
            return label
    return "Low"
