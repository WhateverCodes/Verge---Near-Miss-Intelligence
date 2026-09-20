# Architecture & Upgrade Path

## Pipeline overview

```
video/CSV input
      │
      ▼
detection.py  ──► per-frame bounding boxes + rough class (vehicle/pedestrian)
      │
      ▼
tracking.py   ──► persistent object IDs across frames (centroid tracker)
      │
      ▼
privacy.py    ──► face-region blurring before anything is persisted
      │
      ▼
conflict.py   ──► pairwise TTC (time-to-collision) & minimum-distance checks
      │             → near_miss_events.json
      ▼
risk_scoring.py ─► merge with historical accident CSV, weighted composite
      │             score per location → hotspots.geojson
      ▼
report.py     ──► plain-language, factor-by-factor recommendation report
      │             → report.md
      ▼
main.py (FastAPI) ─► serves hotspots.geojson / report.md to the dashboard
      │
      ▼
frontend/     ──► Leaflet map + report viewer
```

### Single-clip path (video_analysis.py)

`video_analysis.py` reuses the same `detection.py` → `tracking.py` →
`conflict.py` stack but skips `risk_scoring.py` (no historical accident CSV
or site catalogue needed) and returns a verdict for just that one video:

```
video (upload or path)
      │
      ▼
detection.py + tracking.py  ──► anonymized trajectories (same code as above)
      │
      ▼
conflict.py: find_all_conflicts()  ──► veh↔ped AND veh↔veh near-miss events
      │
      ▼
video_analysis.analyze_video()  ──► verdict dict + annotated .mp4
      │
      ▼
main.py POST /api/analyze-video  ──► JSON verdict + annotated_video_url
      │
      ▼
frontend/ "Analyze a video" tab  ──► verdict banner + event table + <video>
```

`conflict.find_all_conflicts()` is additive: it calls the existing,
unit-tested `find_near_miss_events()` (vehicle↔pedestrian) unchanged, and a
new `find_vehicle_vehicle_conflicts()` using the same TTC/distance
methodology, then merges both into one time-sorted list. Nothing about the
multi-site pipeline's behavior or tests changed.

## Why classical CV for detection, not a neural net, in this scaffold?

This repo is meant to run anywhere, instantly, with `pip install -r
requirements.txt` and no GPU, no model weights to download, and no internet
access at run time. Background subtraction (`cv2.createBackgroundSubtractorMOG2`)
plus contour-based bounding boxes gets a demonstrable end-to-end pipeline
working on a static camera view with zero external dependencies.

**This is the first thing to swap out for real use.** `detection.py` isolates
the detector behind a single function, `detect_objects(frame) -> List[Detection]`,
so replacing it does not require touching any other module:

```python
# drop-in replacement sketch
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model(frame, classes=[0, 2, 3, 5, 7])  # person, car, motorcycle, bus, truck
    return [Detection(bbox=..., cls=..., confidence=...) for r in results ...]
```

## Why a centroid tracker, not DeepSORT/ByteTrack?

Same reasoning — zero extra dependencies, easy to read, good enough to keep
IDs stable across a short demo clip. `tracking.py` exposes a single
`CentroidTracker.update(detections) -> Dict[id, TrackedObject]` interface, so
a production tracker (ByteTrack, DeepSORT, Norfair) can be substituted
without changing `conflict.py`.

## Near-miss / conflict indicators

`conflict.py` implements two standard traffic-conflict-technique (TCT)
measures:

- **TTC (Time-to-Collision):** distance between two objects on a closing
  trajectory, divided by their closing speed. Lower TTC = more dangerous.
- **PET (Post-Encroachment Time):** the time gap between when one object
  leaves a shared point/area and the other arrives at it — approximated here
  via minimum pairwise distance over the track overlap window.

Both are standard, peer-reviewed proxies used in real road-safety research
(Surrogate Safety Measures) precisely so that risk can be estimated *without
waiting for a crash to happen*.

## Extending risk scoring

`risk_scoring.py`'s weights (`W_ACCIDENTS`, `W_FATALITIES`, `W_NEAR_MISS`,
`W_SEVERITY`) are simple constants at the top of the file — tune them, or
replace the weighted sum with a proper model (e.g., a Poisson regression
against real historical crash counts) once real data is available.

## Swapping in the real dataset

See `data/README.md` for how to plug in the actual **Road Accidents in
India — MoRTH** dataset from data.gov.in in place of the bundled synthetic
sample.

## Production deployment notes (not implemented here, intentionally)

- Authentication/authorization on the FastAPI endpoints.
- Persistent storage (Postgres + PostGIS instead of flat JSON/GeoJSON files).
- A real video ingestion pipeline (RTSP camera streams, batched processing
  queue) instead of a single local file.
- Model monitoring / drift checks on the detector.
- A formal review workflow so a human traffic engineer signs off before any
  recommendation is acted on.
