# CV pipeline benchmark and validation (real traffic clips)

Prototype measurements from 25 Sep 2026 on the development laptop (Windows 11,
Python 3.13, OpenCV 4.14, RTX 4060 through ONNX Runtime DirectML). There are
no ground-truth labels for these clips, so this document reports counts,
timings and a **manual frame-by-frame review of every reported event**. It is
not a statistical accuracy evaluation: 14 clips are far too few for that.

Detector: YOLO11x (COCO, ONNX) on the GPU at 15 samples/s. On a CPU-only
machine, YOLO11l runs at 10 samples/s instead (see `models/README.md`).

## Final results: 14 real clips

"Before" is the conflict engine at the start of this validation pass. "After" is
the current engine. Both use the same detections and tracks.

| Clip | Length, resolution | Tracks | View handling | Events before | Events after |
|---|---|---|---|---|---|
| WhatsApp 2.42 (truck squeezes car, CCTV) | 8.0 s, 478×750 | 8 | near-overhead | 1 (**wrong pair**) | **1, correct pair** |
| WhatsApp 2.39 (same incident, longer cut) | 14.6 s, 478×750 | 10 | near-overhead | **0 (missed)** | **1, correct pair** |
| A: Indian junction, elevated camera | 39.0 s, 1280×720 | 447 | oblique → bird's-eye | 13 | 0 |
| B: intersection, 1440p | 11.6 s, 2560×1440 | 49 | oblique → bird's-eye | 0 | 0 |
| C: dense scooter traffic, panning | 37.1 s, 2048×932 | 185 | oblique → bird's-eye | 5 | 2 |
| Pexels 13239132 (compilation with cuts) | 112.4 s, 1920×1080 | 147 | oblique → bird's-eye | 0 | 0 |
| Pexels 32266696 (moving camera) | 14.3 s, 2560×1440 | 59 | camera too unstable | 4 | 0 (not assessed) |
| Pexels 32625861 (crossing from a footbridge) | 27.1 s, 1080×1920 | 86 | near-overhead | 0 | 1 |
| Pexels 34405273 (cyclists and pedestrians) | 49.3 s, 2160×3840 | 81 | oblique → bird's-eye | 1 | 1 |
| Pexels 35239545 (drone) | 25.1 s, 2160×3840 | 16 | camera too unstable | 0 | 0 (not assessed) |
| Pexels 35559503 (handheld, walking) | 13.2 s, 2560×1440 | 68 | camera too unstable | 3 | 0 (not assessed) |
| Pexels 36330861 (4K junction) | 39.5 s, 3840×2160 | 142 | oblique → bird's-eye | 2 | 0 |
| Pexels 37247221 | 24.0 s, 1920×1080 | 53 | oblique → bird's-eye | 0 | 0 |
| Pexels 4791734 (street-level, two-way) | 64.1 s, 1920×1080 | 153 | oblique → bird's-eye | 7 | 0 |
| **Total** | | | | **36** | **6** |

The synthetic demo clips still give the same results (1 / 0 / 1 events; they
use the motion detector, which this pass didn't change).

End-to-end times through the real service path (detect, track, conflicts,
render replay and preview), GPU:

| Clip | Total time |
|---|---|
| WhatsApp 2.42 | 11.4 s |
| WhatsApp 2.39 | 16.8 s |
| Clip C | 38.1 s |
| Pexels 35559503 | 24 s |

## Manual review of every event after the change

| Clip / time | Pair | What the frames show | Verdict |
|---|---|---|---|
| WhatsApp 2.42, 3.5–4.8 s | car #3 ↔ truck #4 | The red truck swings right into the white SUV's lane. The SUV is squeezed against the container truck; the footprints meet at 4.1 s. | **correct**: this is the incident in the clip |
| WhatsApp 2.39, 3.6–5.3 s | car #2 ↔ truck #4 | The same incident in the longer recording. | **correct** |
| Pexels 32625861, 11.5 s | car ↔ pedestrian | A car advances onto the zebra crossing while a pedestrian crosses in front of it (TTC 1.3 s). | credible vehicle–pedestrian conflict |
| Pexels 34405273, 34.2 s | bicycle ↔ pedestrian | A cyclist rides through people on the crossing and passes one of them very closely. | credible |
| C, 13.0 s | motorcycle ↔ motorcycle | Dense scooter flow with the camera panning; the pair weave close, but the frames are too crowded to confirm. | unverified: needs human review |
| C, 35.0 s | motorcycle ↔ motorcycle | Two scooters cross paths in the intersection. | plausible, needs human review |

Replay, snapshot, metrics and PDF were checked programmatically for every
event (`problems: none`). In each case:
- the replay covers the event's start–end span;
- the replay's peak frame sits at the reported peak time;
- the snapshot, the metrics and the PDF describe the same event.

The WhatsApp replay was also checked visually. The heads-up display moves
through three phases:
1. "APPROACHING";
2. "CONFLICT DETECTED" (while the event lasts);
3. "AFTER THE CONFLICT".

The conflict marker sits where the truck meets the car.

## What was wrong, and the fix for each

1. **Street-level views: depth ambiguity** (most of the 36 old events). Two
   cars a lane apart, or one several metres behind the other, touch on screen
   in a low camera view.

   *Fix: a per-clip ground-plane model.* On a flat road, a vehicle's apparent
   scale grows linearly with the image row of its ground contact:
   `s(y) = a·(y − y_horizon)`. The fit uses the clip's own vehicles, so no
   calibration is needed. With it, every road user's ground point is mapped to
   road-plane metres (inverse perspective mapping: `Z = f/s`,
   `X = (u − c_x)/s`). Gaps and time-to-collision are then measured between
   vehicle-sized footprints oriented along each road user's direction of
   travel, i.e. in a bird's-eye view.
2. **Overhead views: the wrong footprint.** Seen from above, a truck's box
   covers its whole ground footprint, but the old engine used only the bottom
   25% of the box. For a long truck beside a car, that strip is metres away
   from the car. A truck cut off at the frame edge also got its scale from its
   box height, which from above is its *length*. Together these caused the
   missed event (2.39) and the wrong pair (2.42).

   *Fix:* the view is classified from the same fit. A horizon far above the
   frame means the camera looks steeply down. In that case:
   - cars, buses and trucks use most of their box (75%) as the footprint;
   - people and riders keep only their feet area;
   - a cut-off vehicle's scale comes from the ground model at its row.
3. **Low-speed squeezes were rejected.** In a near-overhead view, footprints
   that meet straight after a measured collision-course approach are direct
   evidence (the "contact phase"). The contact phase:
   - is used only for two four-wheel vehicles in near-overhead views;
   - needs no extra "evasive manoeuvre" evidence;
   - uses a lower closing-speed threshold (1 m/s instead of 3 m/s), because
     these views have no depth ambiguity.
4. **Scene cuts and ID switches read as speed.** A box that "teleports"
   (a step over 1.5 m at more than 30 m/s) is a different object, so motion is
   never estimated across such a jump. This removed a false close pass in a
   clip compilation.
5. **Moving cameras.** Some clips have a camera that travels more than half
   the frame size, or zooms by more than 50% (handheld walking, a drone). For
   these, gaps can't be measured with a single uncalibrated camera, so the app
   says **"Near-misses not assessed"** and explains why. It no longer reports
   conflicts that can't be trusted.
6. **Evasive-action requirement removed.** It had been added to hide the
   geometry errors above, and it also hid real conflicts where nobody braked
   (e.g. a pedestrian crossing in front of a car that keeps rolling). With the
   geometry fixed it is no longer needed. It remains in `config.py`
   (`CONFLICT_EVASIVE_DV_MS`, default 0).

## Conflict rules, in plain words

A pair of road users becomes a **confirmed conflict** only if all of these hold:

1. **Meaningful pair:** vehicle–pedestrian or vehicle–vehicle, never two
   pedestrians. Presumed riders whose bike was never detected are excluded.
2. **Measurable:**
   - the camera is fixed enough (see fix 5);
   - both road users are large enough on screen (≥ 9% of the frame height)
     and not cut off at the top or bottom;
   - both are tracked for ≥ 0.5 s since their last track discontinuity.
3. **Similar depth:** their apparent scales agree within 1.6×.
4. **Collision course or close pass:**
   - *Collision course:* at the current relative motion, the footprints would
     touch within TTC ≤ 1.5 s (≤ 0.8 s for two vehicles), with a gap ≤ 3 m and
     a clearly closing gap (≥ 1 m/s with a pedestrian or two-wheeler; ≥ 3 m/s
     between vehicles in oblique views and ≥ 1 m/s in near-overhead views).
   - *Close pass:* a clearance of ≤ 0.35 m (≤ 0.5 m with a pedestrian) while
     moving, sustained for ≥ 0.4 s.
5. **Not normal following:** two road users heading the same way (within 30°)
   must be closing at ≥ 3 m/s.
6. **Sustained:** ≥ 0.3 s (≥ 0.4 s for two vehicles).
7. **A real approach:** the pair was seen further apart before. They were not
   mostly overlapping (occlusion) during the approach, and they came within
   1 m (2 m with a pedestrian).
8. **Consistent tracks:** steady box size and no more than one implausible
   jump around the event.
9. **One event per interaction:** repeats of the same pair less than 2 s apart
   are merged.

## Honest limitations

- Distances come from a single uncalibrated camera. They use typical
  road-user sizes and a perspective model fitted to the clip. They are
  approximate (the focal length is assumed to be about the image's long
  side). The replay is the evidence a reviewer should check.
- The view classification is automatic and can be wrong for unusual
  cameras, e.g. strongly tilted views or very wide-angle lenses.
- In very dense traffic (clip C), events may involve tracker fragments. They
  are labelled for human review, not as confirmed facts.
- Night and aerial footage are harder for the detector.
- Six events over 14 clips were each checked by eye. That isn't a precision or
  recall figure. A proper evaluation needs a labelled dataset.
