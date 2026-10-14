"""
Historical evidence + AI-observed evidence -> current review priorities.

    HISTORICAL EVIDENCE      data/sample_accidents.csv        (read-only, never modified)
  + AI-OBSERVED EVIDENCE     conflict events from the demo pipeline run
                             (outputs/near_miss_events.json) and from videos
                             analysed at a location (outputs/observations.json)
  + SITE CONTEXT             CSV sites, plus locations pinned on the map
                             (outputs/custom_sites.json, no crash history)
  = UPDATED REVIEW PRIORITY  risk_scoring.compute_hotspots(), unchanged model

Nothing here invents a score. The existing weighted model already has two
AI-evidence factors (near-miss count and severity), so a new observation is
simply more input to the same model. Because the model is a weighted sum,
each score splits exactly into:

    historical_component = W_ACCIDENTS * accidents_norm + W_FATALITIES * fatalities_norm
    observed_component   = W_NEAR_MISS * near_miss_norm + W_SEVERITY  * avg_severity_norm

Scores are min-max normalized across sites, i.e. RELATIVE review priority —
not a probability of a crash.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import config, hotspot_insight, report, risk_scoring

_lock = threading.RLock()

# Only these fields of an analysed event are kept as site evidence.
_OBS_EVENT_KEYS = (
    "event_index", "kind", "event_type", "severity", "peak_severity", "peak_t",
    "min_ttc_s", "min_distance_m", "pet_s",
)


def _read_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        # A corrupt evidence file must never take the dashboard down.
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---- site context ------------------------------------------------------------

def load_custom_sites() -> dict:
    return _read_json(config.CUSTOM_SITES_JSON, {})


def add_custom_site(name: str, lat: float, lon: float) -> dict:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Location coordinates are out of range.")
    name = (name or "").strip()[:80] or f"Pinned location ({lat:.4f}, {lon:.4f})"
    with _lock:
        sites = load_custom_sites()
        # Re-use an existing pin at (almost) the same spot instead of duplicating it.
        for s in sites.values():
            if abs(s["lat"] - lat) < 1e-4 and abs(s["lon"] - lon) < 1e-4:
                return s
        n = 1 + max([int(k.split("_")[-1]) for k in sites if k.split("_")[-1].isdigit()] or [0])
        site = {
            "site_id": f"PIN_{n:03d}",
            "name": name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "accidents_count": 0,
            "fatalities": 0,
            "injuries": 0,
            "years": "n/a",
        }
        sites[site["site_id"]] = site
        _write_json(config.CUSTOM_SITES_JSON, sites)
        return site


def known_site(site_id: str) -> Optional[dict]:
    sites = risk_scoring.load_accident_data(config.SAMPLE_ACCIDENTS_CSV)
    sites.update(load_custom_sites())
    return sites.get(site_id)


# ---- AI-observed evidence ----------------------------------------------------

def load_observations() -> list:
    return _read_json(config.OBSERVATIONS_JSON, [])


def record_observation(analysis_id: str, site_id: str, events: List[dict], meta: dict) -> dict:
    obs = {
        "analysis_id": analysis_id,
        "site_id": site_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "video_upload",
        **meta,
        "events": [{k: e.get(k) for k in _OBS_EVENT_KEYS} for e in events],
    }
    with _lock:
        data = load_observations()
        data.append(obs)
        _write_json(config.OBSERVATIONS_JSON, data)
    return obs


def clear_observations() -> dict:
    """Remove uploaded-video evidence and map pins (demo reset). Historical
    data and the demo pipeline outputs are untouched."""
    with _lock:
        n_obs = len(load_observations())
        n_sites = len(load_custom_sites())
        for p in (config.OBSERVATIONS_JSON, config.CUSTOM_SITES_JSON):
            Path(p).unlink(missing_ok=True)
    return {"observations_removed": n_obs, "pinned_sites_removed": n_sites}


def _pipeline_events() -> list:
    events = _read_json(config.NEAR_MISS_JSON, [])
    if config.REQUIRE_CLOSING_MOTION:  # same rule as single-clip analysis
        events = [e for e in events if e.get("closing_observed", True)]
    for e in events:
        e.setdefault("kind", "vehicle_pedestrian")
        e["source"] = "demo_pipeline"
    return events


# ---- current picture -----------------------------------------------------------

def current_hotspots() -> List[dict]:
    with _lock:
        accident_sites = risk_scoring.load_accident_data(config.SAMPLE_ACCIDENTS_CSV)
        historical_ids = list(accident_sites)
        for sid, s in load_custom_sites().items():
            accident_sites.setdefault(sid, dict(s))
        observations = load_observations()

    events = _pipeline_events()
    analyses_per_site: dict[str, int] = {}
    latest: dict[str, str] = {}
    for o in observations:
        analyses_per_site[o["site_id"]] = analyses_per_site.get(o["site_id"], 0) + 1
        latest[o["site_id"]] = o["analysis_id"]
        for e in o["events"]:
            events.append({**e, "site_id": o["site_id"], "source": "video_upload"})

    near_miss_by_site = risk_scoring.aggregate_near_miss_by_site(events)
    hotspots = risk_scoring.compute_hotspots(accident_sites, near_miss_by_site, historical_site_ids=historical_ids)

    per_site: dict[str, dict] = {}
    for e in events:
        s = per_site.setdefault(e["site_id"], {
            "vehicle_pedestrian": 0, "vehicle_vehicle": 0,
            "min_ttc_s": None, "min_distance_m": None, "crashes": 0,
            "sources": {"demo_pipeline": 0, "video_upload": 0},
        })
        s[e.get("kind", "vehicle_pedestrian")] = s.get(e.get("kind", "vehicle_pedestrian"), 0) + 1
        if e.get("event_type") == "crash":
            s["crashes"] += 1
        s["sources"][e["source"]] += 1
        ttc = e.get("min_ttc_s", e.get("ttc_s"))
        if ttc is not None and (s["min_ttc_s"] is None or ttc < s["min_ttc_s"]):
            s["min_ttc_s"] = ttc
        d = e.get("min_distance_m", e.get("distance_m"))
        if d is not None and (s["min_distance_m"] is None or d < s["min_distance_m"]):
            s["min_distance_m"] = d

    pinned_ids = set(load_custom_sites())
    for h in hotspots:
        f = h["factors"]
        sid = h["site_id"]
        h["has_historical_record"] = sid in historical_ids
        h["historical_component"] = round(config.W_ACCIDENTS * f["accidents_norm"] + config.W_FATALITIES * f["fatalities_norm"], 3)
        h["observed_component"] = round(config.W_NEAR_MISS * f["near_miss_norm"] + config.W_SEVERITY * f["avg_severity_norm"], 3)
        h["evidence"] = {
            **per_site.get(sid, {"vehicle_pedestrian": 0, "vehicle_vehicle": 0, "min_ttc_s": None,
                                 "min_distance_m": None, "crashes": 0,
                                 "sources": {"demo_pipeline": 0, "video_upload": 0}}),
            "analyses": analyses_per_site.get(sid, 0),
            "latest_analysis_id": latest.get(sid),
        }
        h["pinned"] = sid in pinned_ids
        h["brief"] = report.site_brief(h)
        h["summary"] = hotspot_insight.hotspot_summary(h)
    return hotspots


def current_geojson() -> dict:
    return risk_scoring.hotspots_to_geojson(current_hotspots())


# ---- before / after --------------------------------------------------------------

def _snapshot(hotspots: List[dict], site_id: str) -> Optional[dict]:
    for h in hotspots:
        if h["site_id"] == site_id:
            return {
                "rank": h["rank"],
                "of": len(hotspots),
                "risk_score": h["risk_score"],
                "risk_tier": h["risk_tier"],
                "historical_component": h["historical_component"],
                "observed_component": h["observed_component"],
                "near_miss_count": h["near_miss_count"],
            }
    return None


def site_update(before: List[dict], after: List[dict], site_id: str, events: List[dict]) -> dict:
    b, a = _snapshot(before, site_id), _snapshot(after, site_id)
    name = next((h["name"] for h in after if h["site_id"] == site_id), site_id)
    n = len(events)
    vp = sum(1 for e in events if e["kind"] == "vehicle_pedestrian")
    vv = n - vp
    if n:
        kinds = ", ".join(x for x in (f"{vp} vehicle–pedestrian" if vp else "", f"{vv} vehicle–vehicle" if vv else "") if x)
        peak = max(e.get("peak_severity", e["severity"]) for e in events)
        nc = sum(1 for e in events if e.get("event_type") == "crash")
        crash_txt = f", including {nc} crash{'es' if nc != 1 else ''} (severity 1.00)" if nc else ""
        reason = (f"{n} conflict event{'s' if n != 1 else ''} ({kinds}){crash_txt} observed in the submitted video; "
                  f"peak severity {peak:.2f}.")
    else:
        reason = "Video analysed; no conflict events detected, so AI-observed evidence is unchanged."
    notes = []
    if b and a and n and a["observed_component"] <= b["observed_component"]:
        notes.append(
            "This site already had the most observed conflicts, so its normalized conflict "
            "factor was already at the maximum; the new evidence mainly lowers other sites' relative scores."
        )
    rank_changes = []
    before_rank = {h["site_id"]: h["rank"] for h in before}
    for h in after:
        r0 = before_rank.get(h["site_id"])
        if r0 is not None and r0 != h["rank"]:
            rank_changes.append({"site_id": h["site_id"], "name": h["name"], "before": r0, "after": h["rank"]})
    return {
        "site_id": site_id,
        "name": name,
        "before": b,
        "after": a,
        "delta": round(a["risk_score"] - b["risk_score"], 3) if (a and b) else None,
        "reason": reason,
        "notes": notes,
        "rank_changes": rank_changes,
        "scoring_note": "Relative review priority across monitored sites (0–1), not a crash probability.",
    }
