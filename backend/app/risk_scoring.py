"""
Merge historical accident data with detected near-miss events into a single,
transparent, per-location risk score — and export it as GeoJSON so it can be
plotted directly on a map.

The score is a plain weighted sum of normalized factors (see config.py for
weights). It is deliberately NOT a black-box model: `hotspots.geojson`
carries every contributing factor alongside the final score, and
`report.py` turns that into a human-readable explanation. This is
decision-support, not a verdict — see README "Ethics & privacy by design".
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import config


def load_accident_data(csv_path: Path = config.SAMPLE_ACCIDENTS_CSV) -> Dict[str, dict]:
    sites: Dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sites[row["site_id"]] = {
                "site_id": row["site_id"],
                "name": row["name"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "accidents_count": int(row["accidents_count"]),
                "fatalities": int(row["fatalities"]),
                "injuries": int(row["injuries"]),
                "years": row["years"],
            }
    return sites


def aggregate_near_miss_by_site(events: List[dict]) -> Dict[str, dict]:
    agg: Dict[str, dict] = {}
    for e in events:
        site = agg.setdefault(e["site_id"], {"near_miss_count": 0, "severity_sum": 0.0})
        site["near_miss_count"] += 1
        # Events enriched by conflict.encounter_metrics() carry the severity at
        # the conflict's peak moment; older/plain events only have the
        # severity at first entry into the encounter zone.
        site["severity_sum"] += e.get("peak_severity", e["severity"])
    for site_id, s in agg.items():
        s["avg_severity"] = round(s["severity_sum"] / s["near_miss_count"], 3) if s["near_miss_count"] else 0.0
    return agg


def _normalize(values: List[float]) -> Dict[int, float]:
    """Min-max normalize a list of values to [0, 1] by position index."""
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return {i: (v - lo) / span for i, v in enumerate(values)}


def compute_hotspots(
    accident_sites: Dict[str, dict],
    near_miss_by_site: Dict[str, dict],
    historical_site_ids: Optional[Iterable[str]] = None,
) -> List[dict]:
    """
    historical_site_ids: optional subset of sites that actually have a
    historical accident record. When given, the accident/fatality factors are
    normalized over those sites only, and every other site (e.g. a location
    pinned on the map with no crash history) gets 0 for those factors instead
    of shifting everyone else's normalization. Default (None) = original
    behaviour: all sites are normalized together.
    """
    site_ids = list(accident_sites.keys())
    # Ensure every site referenced by a near-miss event is represented even
    # if it wasn't in the accident CSV (falls back to zero historical data).
    for sid in near_miss_by_site:
        if sid not in accident_sites and sid == config.DEMO_SITE["site_id"]:
            accident_sites[sid] = {
                "site_id": sid,
                "name": config.DEMO_SITE["name"],
                "lat": config.DEMO_SITE["lat"],
                "lon": config.DEMO_SITE["lon"],
                "accidents_count": 0,
                "fatalities": 0,
                "injuries": 0,
                "years": "n/a",
            }
            site_ids.append(sid)

    accidents = [accident_sites[s]["accidents_count"] for s in site_ids]
    fatalities = [accident_sites[s]["fatalities"] for s in site_ids]
    near_miss_counts = [near_miss_by_site.get(s, {}).get("near_miss_count", 0) for s in site_ids]
    avg_severities = [near_miss_by_site.get(s, {}).get("avg_severity", 0.0) for s in site_ids]

    if historical_site_ids is None:
        norm_accidents = _normalize(accidents)
        norm_fatalities = _normalize(fatalities)
    else:
        hist = set(historical_site_ids)
        idx = [i for i, s in enumerate(site_ids) if s in hist]
        na = _normalize([accidents[i] for i in idx])
        nf = _normalize([fatalities[i] for i in idx])
        norm_accidents = {i: na[k] for k, i in enumerate(idx)}
        norm_fatalities = {i: nf[k] for k, i in enumerate(idx)}
    norm_near_miss = _normalize(near_miss_counts)
    norm_severity = _normalize(avg_severities)

    hotspots = []
    for i, sid in enumerate(site_ids):
        site = accident_sites[sid]
        factors = {
            "accidents_norm": round(norm_accidents.get(i, 0.0), 3),
            "fatalities_norm": round(norm_fatalities.get(i, 0.0), 3),
            "near_miss_norm": round(norm_near_miss.get(i, 0.0), 3),
            "avg_severity_norm": round(norm_severity.get(i, 0.0), 3),
        }
        score = round(
            config.W_ACCIDENTS * factors["accidents_norm"]
            + config.W_FATALITIES * factors["fatalities_norm"]
            + config.W_NEAR_MISS * factors["near_miss_norm"]
            + config.W_SEVERITY * factors["avg_severity_norm"],
            3,
        )
        hotspots.append(
            {
                **site,
                "near_miss_count": near_miss_by_site.get(sid, {}).get("near_miss_count", 0),
                "avg_near_miss_severity": near_miss_by_site.get(sid, {}).get("avg_severity", 0.0),
                "risk_score": score,
                "risk_tier": config.risk_tier(score),
                "factors": factors,
            }
        )

    hotspots.sort(key=lambda h: h["risk_score"], reverse=True)
    for rank, h in enumerate(hotspots, start=1):
        h["rank"] = rank
    return hotspots


def hotspots_to_geojson(hotspots: List[dict]) -> dict:
    features = []
    for h in hotspots:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
                "properties": {k: v for k, v in h.items() if k not in ("lat", "lon")},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def run(
    accidents_csv: Path = config.SAMPLE_ACCIDENTS_CSV,
    near_miss_json: Path = config.NEAR_MISS_JSON,
    out_geojson: Path = config.HOTSPOTS_GEOJSON,
) -> List[dict]:
    accident_sites = load_accident_data(accidents_csv)
    events = json.loads(Path(near_miss_json).read_text(encoding="utf-8")) if Path(near_miss_json).exists() else []
    near_miss_by_site = aggregate_near_miss_by_site(events)
    hotspots = compute_hotspots(accident_sites, near_miss_by_site)

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(hotspots_to_geojson(hotspots), indent=2), encoding="utf-8")
    return hotspots
