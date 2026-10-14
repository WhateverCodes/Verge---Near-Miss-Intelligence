"""
Near-miss / conflict indicators.

Implements two standard Surrogate Safety Measures used in real traffic-
conflict-technique (TCT) research, so risk can be estimated from close calls
instead of waiting for an actual crash:

- Minimum encounter distance between a vehicle and a pedestrian track.
- TTC (Time-to-Collision): distance / closing speed, at the moment of
  closest approach. A low TTC means the two objects were on a trajectory
  that would collide very soon if nothing changed.

A "near-miss event" is flagged when a vehicle/pedestrian pair comes within
DIST_THRESH_M of each other AND the TTC at that moment is below
TTC_THRESH_S. Thresholds live in config.py and should be tuned per site.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

from . import config


@dataclass
class NearMissEvent:
    site_id: str
    frame: int
    t: float
    vehicle_id: int
    pedestrian_id: int
    distance_m: float
    ttc_s: float | None
    severity: float  # 0-1, higher = more severe (lower TTC / distance)


def _px_to_m(px: float) -> float:
    return px / config.PIXELS_PER_METER


def _severity(distance_m: float, ttc_s: float | None) -> float:
    """
    Simple, transparent severity heuristic in [0, 1]: closer distance and
    lower TTC both push severity up. Intentionally simple and documented so
    a reviewer can see exactly how it was computed (no black box).
    """
    dist_component = max(0.0, 1 - distance_m / config.DIST_THRESH_M)
    ttc_component = 0.0
    if ttc_s is not None:
        ttc_component = max(0.0, 1 - ttc_s / config.TTC_THRESH_S)
    return round(min(1.0, 0.5 * dist_component + 0.5 * ttc_component), 3)


def find_near_miss_events(tracks: List[dict], site_id: str = config.DEMO_SITE["site_id"]) -> List[NearMissEvent]:
    """
    tracks: flat list of {id, cls, frame, t, x, y, w, h} as produced by
    tracking.build_tracks().
    """
    by_frame: dict[int, list[dict]] = {}
    for row in tracks:
        by_frame.setdefault(row["frame"], []).append(row)

    # Index track history by id for closing-speed calculation.
    by_id: dict[int, list[dict]] = {}
    for row in tracks:
        by_id.setdefault(row["id"], []).append(row)
    for rows in by_id.values():
        rows.sort(key=lambda r: r["frame"])

    events: List[NearMissEvent] = []
    seen_pairs: set[tuple[int, int]] = set()

    for frame_idx, rows in sorted(by_frame.items()):
        vehicles = [r for r in rows if r["cls"] == "vehicle"]
        peds = [r for r in rows if r["cls"] == "pedestrian"]

        for v in vehicles:
            for p in peds:
                vc = (v["x"] + v["w"] / 2, v["y"] + v["h"] / 2)
                pc = (p["x"] + p["w"] / 2, p["y"] + p["h"] / 2)
                dist_px = ((vc[0] - pc[0]) ** 2 + (vc[1] - pc[1]) ** 2) ** 0.5
                distance_m = _px_to_m(dist_px)

                if distance_m > config.DIST_THRESH_M:
                    continue

                ttc = _time_to_collision(by_id[v["id"]], by_id[p["id"]], frame_idx)
                if ttc is not None and ttc > config.TTC_THRESH_S:
                    continue

                pair_key = (v["id"], p["id"])
                if pair_key in seen_pairs:
                    continue  # only record the first/closest moment of an encounter
                seen_pairs.add(pair_key)

                events.append(
                    NearMissEvent(
                        site_id=site_id,
                        frame=frame_idx,
                        t=v["t"],
                        vehicle_id=v["id"],
                        pedestrian_id=p["id"],
                        distance_m=round(distance_m, 2),
                        ttc_s=round(ttc, 2) if ttc is not None else None,
                        severity=_severity(distance_m, ttc),
                    )
                )
    return events


def _time_to_collision(vehicle_history: list[dict], ped_history: list[dict], frame_idx: int) -> float | None:
    """
    Approximate TTC using each object's velocity over the last two frames
    (finite difference) and the current separation distance, projected onto
    the closing (relative-velocity) direction.
    """
    v_now = _row_at_or_before(vehicle_history, frame_idx)
    p_now = _row_at_or_before(ped_history, frame_idx)
    v_prev = _row_before(vehicle_history, v_now)
    p_prev = _row_before(ped_history, p_now)
    if not (v_now and p_now and v_prev and p_prev):
        return None

    dt = max(v_now["t"] - v_prev["t"], 1e-6)
    v_vel = (
        ((v_now["x"] - v_prev["x"]) / dt),
        ((v_now["y"] - v_prev["y"]) / dt),
    )
    p_vel = (
        ((p_now["x"] - p_prev["x"]) / dt),
        ((p_now["y"] - p_prev["y"]) / dt),
    )
    rel_vel = (v_vel[0] - p_vel[0], v_vel[1] - p_vel[1])
    rel_speed_px_s = (rel_vel[0] ** 2 + rel_vel[1] ** 2) ** 0.5
    if rel_speed_px_s < 1e-3:
        return None  # not closing at all

    vc = (v_now["x"] + v_now["w"] / 2, v_now["y"] + v_now["h"] / 2)
    pc = (p_now["x"] + p_now["w"] / 2, p_now["y"] + p_now["h"] / 2)
    dist_px = ((vc[0] - pc[0]) ** 2 + (vc[1] - pc[1]) ** 2) ** 0.5

    ttc_s = dist_px / rel_speed_px_s
    return ttc_s


def _row_at_or_before(history: list[dict], frame_idx: int) -> dict | None:
    candidates = [r for r in history if r["frame"] <= frame_idx]
    return candidates[-1] if candidates else None


def _row_before(history: list[dict], row: dict | None) -> dict | None:
    if row is None:
        return None
    candidates = [r for r in history if r["frame"] < row["frame"]]
    return candidates[-1] if candidates else None


def events_to_dicts(events: List[NearMissEvent]) -> List[dict]:
    return [asdict(e) for e in events]


@dataclass
class ConflictEvent:
    """
    Generic version of NearMissEvent that isn't tied to one object being a
    vehicle and the other a pedestrian — used for vehicle/vehicle
    near-misses (see find_vehicle_vehicle_conflicts). NearMissEvent itself
    is left untouched above so existing callers/tests keep working.
    """
    site_id: str
    frame: int
    t: float
    kind: str  # "vehicle_pedestrian" | "vehicle_vehicle"
    a_id: int
    a_cls: str
    b_id: int
    b_cls: str
    distance_m: float
    ttc_s: float | None
    severity: float


def find_vehicle_vehicle_conflicts(
    tracks: List[dict], site_id: str = config.DEMO_SITE["site_id"]
) -> List[ConflictEvent]:
    """
    Same Surrogate-Safety-Measure approach as find_near_miss_events (minimum
    encounter distance + TTC), applied to vehicle/vehicle pairs instead of
    vehicle/pedestrian pairs -- e.g. two cars that nearly collided at a
    junction. A "crash almost happening" isn't only ever a vehicle vs a
    pedestrian, so video_analysis.py checks both.
    """
    by_frame: dict[int, list[dict]] = {}
    for row in tracks:
        by_frame.setdefault(row["frame"], []).append(row)

    by_id: dict[int, list[dict]] = {}
    for row in tracks:
        by_id.setdefault(row["id"], []).append(row)
    for rows in by_id.values():
        rows.sort(key=lambda r: r["frame"])

    events: List[ConflictEvent] = []
    seen_pairs: set[tuple[int, int]] = set()

    for frame_idx, rows in sorted(by_frame.items()):
        vehicles = [r for r in rows if r["cls"] == "vehicle"]

        for i, a in enumerate(vehicles):
            for b in vehicles[i + 1 :]:
                pair_key = tuple(sorted((a["id"], b["id"])))
                if pair_key in seen_pairs:
                    continue

                ac = (a["x"] + a["w"] / 2, a["y"] + a["h"] / 2)
                bc = (b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
                dist_px = ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5
                distance_m = _px_to_m(dist_px)
                if distance_m > config.DIST_THRESH_M:
                    continue

                ttc = _time_to_collision(by_id[a["id"]], by_id[b["id"]], frame_idx)
                if ttc is not None and ttc > config.TTC_THRESH_S:
                    continue

                seen_pairs.add(pair_key)
                events.append(
                    ConflictEvent(
                        site_id=site_id,
                        frame=frame_idx,
                        t=a["t"],
                        kind="vehicle_vehicle",
                        a_id=a["id"],
                        a_cls="vehicle",
                        b_id=b["id"],
                        b_cls="vehicle",
                        distance_m=round(distance_m, 2),
                        ttc_s=round(ttc, 2) if ttc is not None else None,
                        severity=_severity(distance_m, ttc),
                    )
                )
    return events


def find_all_conflicts(tracks: List[dict], site_id: str = config.DEMO_SITE["site_id"]) -> List[dict]:
    """
    Convenience wrapper combining vehicle/pedestrian and vehicle/vehicle
    conflict detection into one uniformly-shaped, time-sorted list of
    dicts: {site_id, frame, t, kind, a_id, a_cls, b_id, b_cls, distance_m,
    ttc_s, severity}.

    Used by video_analysis.py to answer "did a crash almost happen in this
    clip?" for a single uploaded video. find_near_miss_events() above is
    unchanged and still what the multi-site hotspot pipeline
    (pipeline.py / risk_scoring.py) uses.
    """
    unified: List[dict] = []
    for e in find_near_miss_events(tracks, site_id=site_id):
        d = asdict(e)
        d["kind"] = "vehicle_pedestrian"
        d["a_id"] = d.pop("vehicle_id")
        d["a_cls"] = "vehicle"
        d["b_id"] = d.pop("pedestrian_id")
        d["b_cls"] = "pedestrian"
        unified.append(d)
    for e in find_vehicle_vehicle_conflicts(tracks, site_id=site_id):
        unified.append(asdict(e))
    unified.sort(key=lambda e: e["frame"])
    return unified


# ---------------------------------------------------------------------------
# Encounter metrics (additive — nothing above is changed).
#
# find_near_miss_events() / find_vehicle_vehicle_conflicts() record the FIRST
# frame at which a pair entered the encounter zone. That is the right trigger,
# but not the most informative moment to show a reviewer. encounter_metrics()
# looks at the whole encounter between two tracks and reports:
#   - the peak moment (highest per-frame severity),
#   - minimum separation, and when it occurred,
#   - minimum TTC over frames where the pair was actually closing
#     (separation shrinking) — so objects moving apart never report a TTC,
#   - how long the pair stayed inside the encounter zone,
#   - PET, only when the two centroid paths genuinely share a conflict point.
# ---------------------------------------------------------------------------

def _centroid(row: dict) -> tuple[float, float]:
    return (row["x"] + row["w"] / 2, row["y"] + row["h"] / 2)


def _dist_px(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def post_encroachment_time(hist_a: list[dict], hist_b: list[dict]) -> dict | None:
    """
    Approximate PET from two centroid paths: find the pair of points (one
    from each path, any time) that are spatially closest. If they lie within
    PET_RADIUS_M the paths share a conflict point, and PET is the time gap
    between the two objects occupying it. Returns None when the paths never
    cross — PET is then not applicable rather than estimated.

    Both objects must be individually tracked at BOTH instants. Background
    subtraction merges touching blobs into one box carrying a single track
    id; without this check that merged box would masquerade as a near-zero
    PET.
    """
    radius_px = config.PET_RADIUS_M * config.PIXELS_PER_METER
    frames_a = {r["frame"] for r in hist_a}
    frames_b = {r["frame"] for r in hist_b}
    best = None
    for ra in hist_a:
        if ra["frame"] not in frames_b:
            continue
        ca = _centroid(ra)
        for rb in hist_b:
            if rb["frame"] not in frames_a:
                continue
            d = _dist_px(ca, _centroid(rb))
            if best is None or d < best[0]:
                best = (d, ra, rb)
    if best is None or best[0] > radius_px:
        return None
    _, ra, rb = best
    ca, cb = _centroid(ra), _centroid(rb)
    first, second = (ra, rb) if ra["t"] <= rb["t"] else (rb, ra)
    return {
        "pet_s": round(abs(ra["t"] - rb["t"]), 2),
        "conflict_point_px": [round((ca[0] + cb[0]) / 2, 1), round((ca[1] + cb[1]) / 2, 1)],
        "first_id": first.get("id"),
        "first_t": first["t"],
        "second_t": second["t"],
    }


def _truncated(row: dict, frame_size: tuple[int, int] | None) -> bool:
    """Box cut off by the frame edge: its centroid is not the object's centre,
    so it would fake motion (e.g. a car leaving the frame appears to slow)."""
    if frame_size is None:
        return False
    fw, fh = frame_size
    return row["x"] <= 1 or row["y"] <= 1 or row["x"] + row["w"] >= fw - 1 or row["y"] + row["h"] >= fh - 1


def _nearly_parallel(hist_a: list[dict], hist_b: list[dict]) -> bool:
    """True when both road users travel in (almost) the same or opposite
    direction, e.g. one following the other in a lane."""
    def heading(h):
        (x0, y0), (x1, y1) = _centroid(h[0]), _centroid(h[-1])
        n = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return None if n < 1e-6 else ((x1 - x0) / n, (y1 - y0) / n)
    ha, hb = heading(hist_a), heading(hist_b)
    return bool(ha and hb and abs(ha[0] * hb[0] + ha[1] * hb[1]) > 0.9)


def encounter_metrics(
    tracks: List[dict], a_id: int, b_id: int, frame_size: tuple[int, int] | None = None
) -> dict:
    """
    Summarize the full encounter between tracks a_id and b_id (flat track
    rows as produced by tracking.build_tracks). All values are measured from
    the observed trajectories; nothing is extrapolated beyond the TTC
    definition already used by the flagging logic above.

    frame_size (w, h), if given, excludes frames where either box is cut off
    by the frame edge from the peak / minimum-distance / TTC measurements.
    """
    hist_a = sorted((r for r in tracks if r["id"] == a_id), key=lambda r: r["frame"])
    hist_b = sorted((r for r in tracks if r["id"] == b_id), key=lambda r: r["frame"])
    b_by_frame = {r["frame"]: r for r in hist_b}

    samples = []  # (frame, t, dist_m, ttc_or_None, closing, a_row, b_row)
    prev_dist = None
    prev_frame = None
    for ra in hist_a:
        rb = b_by_frame.get(ra["frame"])
        if rb is None or _truncated(ra, frame_size) or _truncated(rb, frame_size):
            prev_dist = None
            continue
        dist_m = _px_to_m(_dist_px(_centroid(ra), _centroid(rb)))
        # "Closing" = separation shrank since the previous co-visible frame
        # (only trusted across a short gap, so a long occlusion doesn't count).
        closing = (
            prev_dist is not None
            and ra["frame"] - prev_frame <= 3
            and dist_m < prev_dist
        )
        ttc = _time_to_collision(hist_a, hist_b, ra["frame"]) if closing else None
        samples.append((ra["frame"], ra["t"], dist_m, ttc, closing, ra, rb))
        prev_dist, prev_frame = dist_m, ra["frame"]

    zone = [s for s in samples if s[2] <= config.DIST_THRESH_M]
    out: dict = {
        "peak_frame": None, "peak_t": None, "peak_severity": 0.0,
        "min_distance_m": None, "min_distance_t": None,
        "min_ttc_s": None, "min_ttc_t": None,
        "zone_start_t": None, "zone_end_t": None, "zone_duration_s": 0.0,
        "closing_observed": False,
        "pet_s": None, "conflict_point_px": None,
    }
    if zone:
        per_frame = [(_severity(s[2], s[3]), s) for s in zone]
        peak_sev, peak = max(per_frame, key=lambda p: p[0])  # max() keeps the earliest tie
        nearest = min(zone, key=lambda s: s[2])
        ttc_samples = [s for s in zone if s[3] is not None]
        frame_dt = _frame_dt(hist_a)
        out.update(
            peak_frame=peak[0],
            peak_t=round(peak[1], 2),
            peak_severity=peak_sev,
            min_distance_m=round(nearest[2], 2),
            min_distance_t=round(nearest[1], 2),
            zone_start_t=round(zone[0][1], 2),
            zone_end_t=round(zone[-1][1], 2),
            zone_duration_s=round(zone[-1][1] - zone[0][1] + frame_dt, 2),
            closing_observed=any(s[4] for s in zone),
        )
        if ttc_samples:
            fastest = min(ttc_samples, key=lambda s: s[3])
            out.update(min_ttc_s=round(fastest[3], 2), min_ttc_t=round(fastest[1], 2))
        # Conflict point for display: midpoint of the pair at closest approach.
        ca, cb = _centroid(nearest[5]), _centroid(nearest[6])
        out["conflict_point_px"] = [round((ca[0] + cb[0]) / 2, 1), round((ca[1] + cb[1]) / 2, 1)]

    pet = post_encroachment_time(hist_a, hist_b)
    if pet is not None:
        out["pet_s"] = pet["pet_s"]
        # Crossing paths: the shared point IS the conflict point. For
        # (near-)parallel paths any point of the shared lane would match, so
        # keep the closest-approach midpoint for display instead.
        if not _nearly_parallel(hist_a, hist_b) or out["conflict_point_px"] is None:
            out["conflict_point_px"] = pet["conflict_point_px"]
    return out


def _frame_dt(history: list[dict]) -> float:
    if len(history) < 2:
        return 1.0 / config.FRAME_RATE_FPS
    span_f = history[-1]["frame"] - history[0]["frame"]
    span_t = history[-1]["t"] - history[0]["t"]
    return span_t / span_f if span_f > 0 and span_t > 0 else 1.0 / config.FRAME_RATE_FPS


def enrich_events(tracks: List[dict], events: List[dict], frame_size: tuple[int, int] | None = None) -> List[dict]:
    """
    Add encounter_metrics() fields to unified event dicts (as returned by
    find_all_conflicts). Original keys are left exactly as they were; the new
    keys are purely additive.
    """
    for e in events:
        a_id = e.get("a_id", e.get("vehicle_id"))
        b_id = e.get("b_id", e.get("pedestrian_id"))
        m = encounter_metrics(tracks, a_id, b_id, frame_size=frame_size)
        if m["peak_frame"] is None:  # defensive: fall back to the trigger frame
            m.update(peak_frame=e["frame"], peak_t=e["t"], peak_severity=e["severity"],
                     min_distance_m=e["distance_m"], min_distance_t=e["t"])
        if m["min_ttc_s"] is None and e.get("ttc_s") is not None and m["closing_observed"]:
            m.update(min_ttc_s=e["ttc_s"], min_ttc_t=e["t"])
        m["severity_label"] = config.severity_band(m["peak_severity"])
        e.update(m)
    return events
