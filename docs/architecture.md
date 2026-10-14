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

### Location-linked analysis and evidence (offline edition)

```
dashboard: location (hotspot / map pin) + video
      |  POST /api/analyses  (background job, real per-stage progress)
      v
analysis_service.run_analysis()
      |- video_analysis.analyze_video()        same detect -> track -> conflict stack
      |     |- conflict.enrich_events()        peak moment, min distance, min TTC
      |     |                                   while closing, time in zone, PET
      |     '- _write_annotated_video()        one pass: annotated .mp4 (+ .webm),
      |                                         evidence replay frames around each
      |                                         conflict peak, peak snapshot
      |- site_evidence.record_observation()    AI-observed evidence for the site
      |- site_evidence.current_hotspots()      CSV (read-only) + pipeline events +
      |                                         observations -> compute_hotspots()
      '- site_evidence.site_update()           before/after: score, rank, components
      v
outputs/analyses/<id>/analysis.json  -> GET /api/analyses/<id>/report.pdf
                                        (evidence_report.py + pdf.py)
```

Review priority keeps the original weighted model. Because it is a weighted
sum, each score splits exactly into `historical_component` (crashes +
fatalities) and `observed_component` (AI-detected conflicts + severity).
Map pins have no crash history and are excluded from the historical
normalization, so adding one does not shift other sites' historical scores.

`tracking.py` now lets a detection continue a track of the other class at a
small cost (`TRACK_CLASS_SWITCH_PENALTY_PX`), with the track's class decided
by majority vote. That stops a car which is only partly in view as it enters
the frame (narrow, so "pedestrian"-shaped) from becoming two tracks. Set
`TRACK_ALLOW_CLASS_SWITCH = False` for the original strict behaviour.

## Real-footage pipeline (object detector)

```
video -> detection_yolo.process_video_objects()   YOLO11x (GPU, DirectML) / YOLO11l (CPU, OpenCV DNN),
                                                   15 / 10 fps samples, rider/occupant removal,
                                                   camera-motion estimate
      -> tracking.build_object_tracks()            IoU + motion prediction, confirm after 4 hits,
                                                   1.5 s max age, duplicate suppression
      -> conflict_engine.find_conflict_events()    per-clip perspective fit (scale vs image row):
                                                   oblique view -> bird's-eye road-plane footprints,
                                                   near-overhead -> box footprints (+ contact phase),
                                                   moving camera -> not assessed; motion never read
                                                   across track jumps / scene cuts; collision-course
                                                   TTC or close pass, persistence, approach +
                                                   occlusion checks, one event per interaction
      -> video_analysis._write_annotated_video()   decodes only output frames; replay per event from
                                                   its start/peak/end; peak snapshot; preview frames
```

`video_analysis.analyze_video(detector="auto")` picks the motion-based
pipeline below only for computer-drawn clips (`looks_synthetic`); real
footage never falls back to it. See `docs/cv-benchmark.md` for measured
before/after results and the root-cause analysis.

## Why classical CV for detection, not a neural net, in this scaffold?

*(Historical note: this section describes the original motion-based
detector, which is still used for the synthetic demo clips and the
multi-site demo pipeline.)*

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
  leaves a shared point and the other arrives at it.
  `conflict.post_encroachment_time()` finds where the two centroid paths come
  closest; PET is reported only if that is within `PET_RADIUS_M` and both
  road users were separately tracked at both instants. Otherwise it is "not
  applicable", never estimated.

Both are standard, peer-reviewed proxies used in real road-safety research
(Surrogate Safety Measures) precisely so that risk can be estimated *without
waiting for a crash to happen*.

## Extending risk scoring

`risk_scoring.py`'s weights (`W_ACCIDENTS`, `W_FATALITIES`, `W_NEAR_MISS`,
`W_SEVERITY`) are simple constants at the top of the file — tune them, or
replace the weighted sum with a proper model (e.g., a Poisson regression
against real historical crash counts) once real data is available.

The Low / Moderate / High / Critical bands are `config.RISK_TIERS` (0.20 /
0.40 / 0.60), applied to the same score. They are exposed at `/api/health`
(`risk_tiers`) so the map legend shows the exact ranges. If the weights
change, re-check the bands: crash history alone can score at most
`W_ACCIDENTS + W_FATALITIES`, and the Critical line must stay reachable for
the sites with the worst crash records.

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
