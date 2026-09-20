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
