# Verge — AI Road-Risk & Near-Miss Intelligence for Safer Streets

**PS-17 · Track: Safe & Smart Communities · SDG 3 (Good Health) & SDG 11 (Sustainable Cities)**

Most road-safety programs only react after a crash is reported. This prototype
flips that around: it looks for **near misses and conflict patterns** —
vehicles and pedestrians that came dangerously close — *before* they turn
into a crash statistic, and ranks locations for engineering or enforcement
review.

It is a **working scaffold**, not a finished product. Every module runs
end-to-end on synthetic/sample data out of the box so you can see the full
pipeline, and every "real-world" integration point (live camera feed,
production object detector, the real MoRTH dataset) is clearly marked so it
can be swapped in.

> ⚠️ **This system produces decision-support, not verdicts.** It never
> assigns fault, never identifies individuals, and every score it outputs is
> shown with the factors behind it so a human reviewer can question it.

---

## What it does

```
 recorded / simulated        object          near-miss /            risk hotspot          transparent
 traffic video          →    detection   →    conflict indicator →  scoring/ranking   →    recommendation
 (or accident CSV)           & tracking       (TTC / PET)            (+ historical          report (.md)
                                                                       accident data)
```

1. **`scripts/generate_sample_video.py`** — synthesizes a short traffic clip
   (a "vehicle" and a "pedestrian" on a collision-adjacent path) so the
   pipeline can be demoed without any real footage.
2. **`backend/app/detection.py`** — classical computer-vision detector
   (background subtraction + contour heuristics) that finds moving objects
   in a frame and roughly classifies them as *vehicle* or *pedestrian* by
   size/shape. This is intentionally lightweight (no GPU / model download
   needed) — swap in a YOLOv8 / other detector for production use (see
   `docs/architecture.md`).
3. **`backend/app/tracking.py`** — a simple centroid tracker that assigns a
   persistent ID to each detected object across frames.
4. **`backend/app/privacy.py`** — blurs faces (Haar-cascade based) before any
   frame is ever written to disk. **No raw imagery or personally
   identifiable data is retained** — only anonymized (x, y, t, class, id)
   trajectories continue through the pipeline.
5. **`backend/app/conflict.py`** — computes **Time-to-Collision (TTC)** and
   minimum encounter distance between vehicle/pedestrian track pairs and
   flags **near-miss events** when they cross safety thresholds.
6. **`backend/app/risk_scoring.py`** — merges near-miss counts/severity with
   historical accident data (sample MoRTH-style CSV included in `data/`) into
   a single, weighted, per-location **risk score**, and writes a GeoJSON
   hotspot map.
7. **`backend/app/report.py`** — generates a plain-language, factor-by-factor
   **recommendation report** (Markdown) per hotspot — e.g. "3 near-misses
   with sub-2s TTC + 2 historical accidents → recommend pedestrian refuge
   island / signal-timing review" — always phrased as a suggestion for human
   review, never as an automated conclusion.
8. **`backend/app/video_analysis.py`** — the direct, single-clip version of
   the question: *"did a crash almost happen in this video?"* Runs the same
   detect → track → conflict stack against ONE uploaded clip (no accident
   CSV or site catalogue required) and returns a plain-language verdict plus
   an annotated copy of the clip with tracked boxes and the flagged
   near-miss moment highlighted. See "Analyze your own video" below.
9. **`backend/app/main.py`** — a small FastAPI service that exposes the
   pipeline outputs (`/api/hotspots`, `/api/report`), can re-run the pipeline
   (`/api/pipeline/run`), and accepts a video upload for direct near-miss
   analysis (`/api/analyze-video`) so the frontend dashboard has something to
   call.
10. **`frontend/`** — a static Leaflet-based dashboard: a map of ranked risk
    hotspots, colored by score, with a click-through breakdown of *why* each
    location ranked where it did, the rendered recommendation report, and an
    "Analyze a video" tab to upload and check a clip directly.

---

## Quick start

**Easiest: one-click start.** From the project folder:

- **Windows:** double-click `start_dashboard.bat`
- **Mac/Linux:** run `./start_dashboard.sh`

This installs dependencies, generates the demo video and runs the pipeline
if needed, starts both the API and the dashboard, and opens your browser to
the dashboard automatically. Leave the terminal window(s) it opens running;
closing them stops the servers.

**Manual steps** (same thing, spelled out):

```bash
# 1. clone & install
git clone <this-repo>
cd verge-ai-road-risk
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 2. generate a synthetic demo clip (no real footage needed)
python scripts/generate_sample_video.py

# 3. run the full pipeline end-to-end (detection → tracking → conflict →
#    risk scoring → report) on the sample video + sample accident data
python scripts/run_pipeline.py

# outputs land in ./outputs/:
#   tracks.json            anonymized object trajectories
#   near_miss_events.json  detected near-miss / conflict events
#   hotspots.geojson       ranked, geolocated risk hotspots
#   report.md              transparent, per-hotspot recommendation report

# 4. serve the API (so the dashboard has something to fetch)
uvicorn backend.app.main:app --reload --port 8000

# 5. serve the dashboard — IMPORTANT: don't just double-click
#    frontend/index.html. OpenStreetMap's tile servers require pages to be
#    served over HTTP (they reject requests from a page opened as a local
#    file), so use a tiny local server instead:
python -m http.server 5500 --directory frontend
# then open http://localhost:5500 in your browser
```

> **"Couldn't load hotspots (Failed to fetch)" in the dashboard?** That
> means the API server (step 4 above, `uvicorn ...`) isn't running. Start
> it and click "Refresh".
>
> **Map tiles showing "Access blocked" or a watermark instead of a map?**
> You're most likely opening `frontend/index.html` directly from disk
> instead of through `http://localhost:5500`. See step 5 above — this is
> a restriction from OpenStreetMap's tile servers, not a bug in this repo.

Run the tests with:

```bash
pytest backend/tests
```

---

## Analyze your own video

Everything above runs the *multi-site hotspot* workflow (a video plus
historical accident data, ranked across locations). If you just want a
direct answer for **one clip** — *"did a crash almost happen in this
footage?"* — use this instead. No accident CSV or site catalogue needed.

**Command line:**

```bash
python scripts/analyze_video.py --video path/to/your_clip.mp4
```

This prints a plain-language verdict (e.g. *"Yes — a crash looks like it was
almost about to happen. At 2.5s into the clip, a vehicle and a pedestrian
closed to 7.4m apart with a time-to-collision of 0.5s..."* or *"No near-miss
detected."*), and writes to `./outputs/`:

- `<clip-name>_verdict.json` — every detected event (timestamp, the two
  objects involved, distance, TTC, severity), machine-readable
- `<clip-name>_annotated.mp4` — the same clip with tracked boxes drawn
  (green = vehicle, yellow = pedestrian) and a red "NEAR-MISS" highlight
  burned in at the flagged moment(s), so you can see exactly what the
  detector saw

Add `--no-annotate` to skip writing the video (faster, verdict/JSON only).

**Via the API / dashboard:** once `uvicorn backend.app.main:app --reload
--port 8000` is running, `POST /api/analyze-video` (multipart form field
`file`) accepts any `.mp4` / `.mov` / `.avi` / `.mkv` / `.webm` up to 200MB
and returns the same verdict as JSON, plus a URL to the annotated clip. The
dashboard's **"Analyze a video"** tab wraps this in a simple upload form —
pick a file, click *Analyze video*, and the verdict banner, event table, and
annotated clip appear inline.

```bash
curl -F "file=@path/to/your_clip.mp4" http://localhost:8000/api/analyze-video
```

Detection is classical computer vision (background subtraction), so it works
best on a **fixed, relatively steady camera angle** with moving
foreground objects against a mostly-static background — a real traffic/CCTV
framing, not a handheld clip. See "Honest limitations of this prototype"
below and `docs/architecture.md` for the trained-detector upgrade path.

---

## Project structure

```
verge-ai-road-risk/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI app — hotspots/report, pipeline trigger, video upload
│   │   ├── detection.py     Frame-level vehicle/pedestrian detection (CV heuristic)
│   │   ├── tracking.py      Centroid tracker → persistent object IDs
│   │   ├── privacy.py       Face blurring / anonymization utilities
│   │   ├── conflict.py      TTC near-miss & conflict indicator logic (veh↔ped, veh↔veh)
│   │   ├── video_analysis.py Single-clip "did a crash almost happen?" verdict + annotation
│   │   ├── risk_scoring.py  Weighted hotspot scoring + GeoJSON export
│   │   └── report.py        Human-readable recommendation report generator
│   └── tests/
│       └── test_conflict.py Unit tests for the conflict-indicator math
├── data/
│   ├── sample_accidents.csv Small synthetic MoRTH-style accident dataset
│   └── README.md            Notes on swapping in the real data.gov.in dataset
├── scripts/
│   ├── generate_sample_video.py  Builds a synthetic demo clip
│   ├── run_pipeline.py           Orchestrates the full multi-site pipeline
│   └── analyze_video.py          Analyze one video: near-miss verdict + annotated clip
├── frontend/
│   ├── index.html / app.js / style.css   Leaflet hotspot dashboard + video-upload tab
├── docs/
│   └── architecture.md      Design notes + how to swap in production components
├── outputs/                  Pipeline output artifacts (gitignored)
├── requirements.txt
├── start_dashboard.bat       One-click start (Windows)
├── start_dashboard.sh        One-click start (Mac/Linux)
└── LICENSE
```

---

## Ethics & privacy by design

- **No face identification.** `privacy.py` blurs faces before any frame
  touches disk; only anonymized trajectory data (position/time/class/ID)
  flows past the detection stage.
- **No automatic fault assignment.** Every output is a *ranked suggestion for
  human review*, with the contributing factors shown alongside the score —
  never a claim about who was at fault or a definitive risk verdict.
- **Aggregation over surveillance.** The system is designed to answer "which
  intersections deserve engineering attention?", not "who did what?".
- **Transparent scoring.** `risk_scoring.py` and `report.py` always retain
  and expose the individual factors (near-miss count, severity, historical
  accidents) behind every score — nothing is a black box.

## Honest limitations of this prototype

- The detector is classical CV (background subtraction), not a trained
  neural network — good enough to demo the pipeline on a synthetic clip, but
  not accurate enough for production. See `docs/architecture.md` for the
  drop-in upgrade path (YOLOv8 + DeepSORT/ByteTrack).
- `data/sample_accidents.csv` is a small **synthetic** stand-in for the real
  [Road Accidents in India (MoRTH)](https://data.gov.in) dataset — swap it in
  once downloaded (see `data/README.md`).
- Near-miss detection thresholds (TTC/distance) are simple, tunable
  constants, not calibrated against real-world crash data.

## License

MIT — see `LICENSE`.
