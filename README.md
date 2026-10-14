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
2. **`backend/app/detection_yolo.py`** — object detection for real
   footage with a pretrained **YOLO11** (COCO) network bundled in `models/`
   (no internet needed at run time):
   - YOLO11x on a DirectX 12 GPU through ONNX Runtime DirectML;
   - otherwise YOLO11l on the CPU with OpenCV's DNN module;
   - YOLOv8n as a last-resort fallback (see `models/README.md`).

   It detects pedestrians, cars, motorcycles, buses, trucks and bicycles on
   frames sampled at 15 fps (10 fps on CPU), leaves out riders and vehicle
   occupants, and estimates camera motion so a moving camera doesn't inflate
   speeds. **`backend/app/detection.py`** (background subtraction +
   contour heuristics) is kept for the synthetic demo clips and the
   multi-site demo pipeline; it is not reliable on real footage (see
   `docs/cv-benchmark.md`).
3. **`backend/app/tracking.py`** — the original centroid tracker (synthetic
   clips) and an IoU tracker for the object detector: motion prediction,
   confirmation after several matches, lost-track expiry, duplicate
   suppression, majority-vote class labels.
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
   hotspot map. Score (0–1) = 0.35 × crashes + 0.30 × fatalities + 0.25 ×
   AI near-misses + 0.10 × near-miss severity, each min-max normalized across
   sites. Risk levels on that same score: **Low** < 0.20 ≤ **Moderate** <
   0.40 ≤ **High** < 0.60 ≤ **Critical**. Crash history alone can reach at
   most 0.65, so the Critical line sits where a site is at or near the
   network's worst on both crashes and fatalities (with the sample data:
   Swargate Bus Depot Junction, 25 crashes and 6 deaths, score 0.65).
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
11. **`backend/app/site_evidence.py`** — keeps **historical evidence** (the
    accident CSV, never modified) and **AI-observed evidence** (conflict
    events from videos analysed at a location) separate, and feeds both into
    the *existing* scoring model. Every score splits exactly into a
    historical part and an AI-observed part, so the dashboard can show *why*
    a location's review priority changed.
12. **`backend/app/analysis_service.py`** + **`evidence_report.py`** — the
    location-linked analysis flow (real stage-by-stage progress, evidence
    replay, site update) and the one-page **PDF evidence report** for a
    detected conflict, built with a tiny dependency-free PDF writer
    (`pdf.py`).

### Offline edition: the demo flow

```
risk hotspot map -> select / pin a location -> upload traffic video
  -> detection + tracking -> conflict analysis -> near-miss detected
  -> automatic evidence replay (short annotated clip around the conflict)
  -> TTC / minimum distance / severity (+ PET when the paths share a point)
  -> location's review priority updated (historical + AI-observed, shown separately)
  -> hotspot ranking and map updated -> suggested human review -> PDF evidence report
```

---

## Quick start

**Easiest: one-click start.** From the project folder:

- **Windows:** double-click `start_dashboard.bat`
- **Mac/Linux:** run `./start_dashboard.sh`

This installs dependencies, generates the demo video (and the synthetic
clips in `data/demo_clips/`) and runs the pipeline if needed, starts both the API and the dashboard, and opens your browser to
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

The suite covers the original conflict math plus encounter metrics / PET,
tracking, the site-evidence scoring (including that the historical CSV is
never modified), report text, and the full API flow (upload -> evidence ->
site update -> PDF) on a generated synthetic clip.

---

## Judge walkthrough (offline round)

Everything below runs on one laptop with no internet (Leaflet is bundled;
only the background map tiles need a connection. Without them the hotspots
still render on a blank map).

1. `start_dashboard.bat` (or `./start_dashboard.sh`). For a clean slate,
   click **Clear uploaded-video evidence** at the bottom of the hotspot list.
2. **Ranked hotspots**: point out a low-ranked site (e.g. *Nagar Road
   Signal*, #7). Each card shows its priority split into historical and
   AI-observed evidence.
3. **Analyze a video**: search *Nagar Road Signal* (or click any hotspot, or
   click the map to pin a new location), choose
   `data/demo_clips/crossing_near_miss.mp4`, click **Analyze video**.
4. The progress list advances as each backend stage actually completes.
5. Result: **Near-miss detected**. The evidence replay plays the conflict
   window, with TTC / minimum distance / severity / PET below it, then
   *What happened* and *Why flagged*.
6. **Risk priority updated** shows historical vs AI-observed evidence and the
   rank change; the map marker is labelled with the change and the hotspot
   list re-ranks.
7. **Download PDF evidence report**: one page, generated from this analysis.

Other synthetic clips: `vehicle_rear_end.mp4` (vehicle-vehicle conflict) and
`safe_crossing.mp4` (expected result: *No near-miss detected*). They are drawn
shapes, clearly synthetic; regenerate with `python scripts/generate_demo_clips.py`.

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

This prints a plain-language verdict (e.g. *"Near-miss detected at 00:02.6:
a vehicle and a pedestrian came within 4.0 m of each other, with a minimum
time-to-collision of 0.23 s (high severity)."* or *"No near-miss
detected."*), and writes to `./outputs/`:

- `<clip-name>_verdict.json` — every detected event (trigger time, peak
  moment, the two objects involved, minimum distance, minimum TTC, time in
  the encounter zone, PET when applicable, severity), machine-readable
- `<clip-name>_annotated.mp4` — the same clip with every tracked road user
  labelled (e.g. `VEHICLE #2`, `PEDESTRIAN #1`) and, during each conflict,
  the pair emphasized with trails, the separation line, the conflict point
  and a compact TTC / distance / severity panel

Add `--no-annotate` to skip writing the video (faster, verdict/JSON only).

**Via the API / dashboard:** once `uvicorn backend.app.main:app --reload
--port 8000` is running, `POST /api/analyze-video` (multipart form field
`file`) accepts any `.mp4` / `.mov` / `.avi` / `.mkv` / `.webm` up to 200MB
and returns the same verdict as JSON, plus a URL to the annotated clip.
Optional form fields `site_id` (an existing location) or `lat` + `lon`
(+ `location_name`, a new map pin) link the clip to a location so its
evidence is added to that site's review priority. The dashboard uses the
background-job version, `POST /api/analyses` + `GET
/api/analyses/{id}/status`, for real progress; the evidence PDF is at `GET
/api/analyses/{id}/report.pdf?event=N`, and `DELETE /api/observations`
clears uploaded-video evidence (historical data is untouched). The
dashboard's **"Analyze a video"** tab wraps this in a simple upload form —
pick a file, click *Analyze video*, and the verdict banner, event table, and
annotated clip appear inline.

```bash
curl -F "file=@path/to/your_clip.mp4" http://localhost:8000/api/analyze-video
```

The **detector** is chosen with the form field `detector`: `auto` (the
default) uses the object detector for real footage and the motion-based
detector only for clips it recognises as computer-drawn (the synthetic demo
clips). `object` and `motion` force one or the other. Real-footage conflicts
come from `backend/app/conflict_engine.py`. It first fits a **perspective
(ground-plane) model** to the clip's own vehicles (their size grows linearly
with their image row), then measures each view accordingly:
- **Street-level (oblique) views:** road users are mapped to a bird's-eye view
  of the road, so two cars that touch on screen but are a lane apart are not
  a conflict.
- **Near-overhead views:** CCTV looking down; box footprints are used directly.
- **Moving cameras:** clips where the camera moves or zooms too much are
  reported as "near-misses not assessed" instead of guessed.

A conflict then needs all of the following:
- a collision course with TTC ≤ 1.5 s (≤ 0.8 s for vehicle–vehicle pairs),
  or a close pass (≤ 0.35 m at speed);
- a clearance ≤ 3 m and a clearly closing gap;
- persistence over consecutive frames;
- an observed approach.

Normal same-direction following/queuing is excluded, and tracks that jump
between objects (or across scene cuts) are never read as motion. Each
interaction becomes one event with a start, peak and end, and the replay is cut
around those times. [`docs/cv-benchmark.md`](docs/cv-benchmark.md) has:
- the results on 14 real clips (36 events before this engine, 6 after);
- a frame-by-frame review of every event;
- timings: an 8 s CCTV clip takes about 11 s end to end with the GPU backend.

A fixed, elevated camera still gives the most reliable distances.

---

## Road Safety Assistant and emailing reports

**Verge Road Safety Assistant** (header button) answers questions about the
monitored hotspots, the latest video analysis, and how Verge works. It is
deterministic and grounded: `backend/app/assistant.py` matches the question to
a known intent and builds the answer only from Verge's own data (the same
hotspot records, brief and summary as the map, the report and the PDFs). There
is no language model and no cloud call. When the data can't answer a question
(e.g. "the safest road in Pune", crash causes, weather), it says so. The
assistant's context follows what you select (map search, a marker, a ranked
card, or **Ask Verge About This** in View Analysis) and the latest video
analysed in the session.

**Email Report** (in View Analysis for a site brief, and next to each video
evidence PDF) sends the same PDF as the download, from the backend. SMTP
credentials are read from environment variables or a local `.env` file and
never reach the browser:

```bash
cp .env.example .env      # then fill in SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM
# Gmail example: SMTP_HOST=smtp.gmail.com, SMTP_PORT=587, SMTP_SECURITY=starttls,
# SMTP_PASSWORD = an App Password (Google account > Security > 2-step verification > App passwords)
```

Restart the API after editing `.env`. Without credentials the dialog says
"Email delivery is not configured" and does not send. Set `VERGE_EMAIL_OUTBOX=1`
for a test mode that saves the complete message (PDF attached) to
`outputs/outbox/*.eml` instead of sending it; the UI then says "Saved in test
mode (not sent)". "Report sent successfully" appears only after the SMTP server
has accepted the message.

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
│   │   ├── report.py        Recommendation report + per-site brief (evidence, why, review)
│   │   ├── site_evidence.py Historical + AI-observed evidence -> current review priorities
│   │   ├── analysis_service.py Location-linked analysis jobs with real progress
│   │   ├── evidence_report.py  One-page PDF evidence report for a conflict event
│   │   └── pdf.py           Minimal dependency-free PDF writer (text, shapes, JPEG)
│   └── tests/               Conflict math, encounter metrics, tracking, scoring, API flow
├── data/
│   ├── sample_accidents.csv Small synthetic MoRTH-style accident dataset
│   └── README.md            Notes on swapping in the real data.gov.in dataset
├── scripts/
│   ├── generate_sample_video.py  Builds a synthetic demo clip (multi-site pipeline)
│   ├── generate_demo_clips.py    Synthetic clips for the Analyze tab (near-miss, rear-end, safe)
│   ├── run_pipeline.py           Orchestrates the full multi-site pipeline
│   └── analyze_video.py          Analyze one video: near-miss verdict + annotated clip
├── frontend/
│   ├── index.html / app.js / style.css   Leaflet hotspot dashboard + video-upload tab
│   └── vendor/leaflet/                   Bundled Leaflet 1.9.4 (BSD-2) for offline use
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
- **No raw footage retained.** A video uploaded through the API is deleted
  as soon as its analysis finishes (success or failure). What remains is the
  face-blurred annotated evidence and the anonymized trajectories.
- **No automatic fault assignment.** Every output is a *ranked suggestion for
  human review*, with the contributing factors shown alongside the score —
  never a claim about who was at fault or a definitive risk verdict.
- **Aggregation over surveillance.** The system is designed to answer "which
  intersections deserve engineering attention?", not "who did what?".
- **Transparent scoring.** `risk_scoring.py` and `report.py` always retain
  and expose the individual factors (near-miss count, severity, historical
  accidents) behind every score — nothing is a black box.

## Honest limitations of this prototype

- Real footage uses a pretrained COCO detector (YOLO11 at 640 px). It can miss
  small or distant road users and struggles at night. Distances and speeds
  come from one uncalibrated camera. They are estimated from typical road-user
  sizes and a perspective model fitted to the clip, so they are approximate;
  the evidence replay is what a reviewer should check. No ground-truth labels
  exist for the test clips, so no accuracy figure is claimed (see
  `docs/cv-benchmark.md`). In very dense traffic, events can involve tracker
  fragments and need human review.
- The motion-based detector (background subtraction) is only suitable for
  the synthetic demo clips; on real footage it breaks vehicles into many
  false "pedestrians".
- `data/sample_accidents.csv` is a small **synthetic** stand-in for the real
  [Road Accidents in India (MoRTH)](https://data.gov.in) dataset — swap it in
  once downloaded (see `data/README.md`).
- Near-miss detection thresholds (TTC/distance) are simple, tunable
  constants, not calibrated against real-world crash data, and distances use
  an uncalibrated demo scale (`PIXELS_PER_METER`).
- Background subtraction needs a fixed camera, merges road users that touch
  into one blob, and slowly absorbs objects that stop. A near-miss is only
  counted when the pair is observed *closing*; candidate pairs without a
  measurable approach (typically tracking fragments) are listed as "set
  aside" in the result instead of being counted.
- PET is reported only when two centroid paths genuinely pass through the
  same point while both road users are separately tracked; otherwise it is
  shown as not applicable rather than estimated.
- Review priority is **relative** (min-max normalized across monitored
  sites), not a probability of a crash.

## License

MIT — see `LICENSE`.
