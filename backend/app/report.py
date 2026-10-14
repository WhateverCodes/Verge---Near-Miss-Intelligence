"""
Turns ranked hotspots into a plain-language Markdown report.

Every recommendation is phrased as a suggestion for a human traffic
engineer / enforcement planner to review — never as an automated finding of
fault or a mandate. Every number shown traces back to a named factor
(accident history, near-miss count/severity) so nothing here is a black
box. See README "Ethics & privacy by design".

site_brief() is the single source of the per-location wording (evidence ->
why flagged -> recommended review). The Markdown report, the dashboard and
the PDF evidence report all use it, so they never disagree.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

from . import config

_DISCLAIMER = (
    "> **Decision support for human review.** Priorities combine historical "
    "crash records with AI-observed conflict evidence. They are not a finding "
    "of fault and not a prediction that a crash will occur. No individuals "
    "are identified: conflict detection works only on anonymized object "
    "trajectories, and faces are blurred before any frame is written to disk."
)


def _recommendation_for(hotspot: dict) -> str:
    f = hotspot["factors"]
    # Simple, transparent rule-based suggestion — pick the dominant factor
    # and map it to a plausible engineering/enforcement lever. A real
    # deployment would route this to a traffic engineer, not act on it
    # automatically.
    dominant = max(f, key=f.get)
    suggestions = {
        "accidents_norm": "Historical crash count is the main driver — prioritize "
        "for a formal road-safety audit (sight lines, signage, road geometry).",
        "fatalities_norm": "Fatality history is elevated — high-priority candidate "
        "for engineering countermeasures (speed reduction, protected crossings) "
        "and review by the road-safety authority.",
        "near_miss_norm": "A high volume of detected near-misses relative to other "
        "sites suggests a recurring conflict pattern — consider signal-timing "
        "review or added pedestrian crossing infrastructure, and validate with "
        "on-site observation.",
        "avg_severity_norm": "Near-misses at this site tend to be severe (very "
        "low time-to-collision) — consider targeted enforcement during peak "
        "hours and a sight-distance/visibility check.",
    }
    return suggestions.get(dominant, "Recommend inclusion in the next road-safety review cycle.")


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def site_brief(h: dict) -> dict:
    """
    Structured, evidence-first explanation of one hotspot:
      headline      one line: priority + what mainly drives it
      evidence      the facts (counts, measured values)
      key_factors   why it ranks where it does
      review        suggested human-review actions (max 3)
    Works on plain compute_hotspots() output; uses the extra fields added by
    site_evidence.py (per-kind conflict counts, lowest TTC, ...) when present.
    """
    f = h["factors"]
    ev = h.get("evidence", {})
    has_hist = h.get("has_historical_record", True)
    n_conf = int(h.get("near_miss_count", 0))
    vp = int(ev.get("vehicle_pedestrian", n_conf if not ev else 0))
    vv = int(ev.get("vehicle_vehicle", 0))
    min_ttc = ev.get("min_ttc_s")
    analyses = int(ev.get("analyses", 0))

    evidence: list[str] = []
    if has_hist:
        evidence.append(f"{_plural(h['accidents_count'], 'recorded crash', 'recorded crashes')} ({h['years']})")
        evidence.append(f"{_plural(h['fatalities'], 'fatality', 'fatalities')}, {_plural(h['injuries'], 'injury', 'injuries')}")
    else:
        evidence.append("No historical crash record for this location")
    if n_conf:
        parts = []
        if vp:
            parts.append(f"{vp} vehicle–pedestrian")
        if vv:
            parts.append(f"{vv} vehicle–vehicle")
        detail = f" ({', '.join(parts)})" if parts else ""
        evidence.append(f"{_plural(n_conf, 'AI-detected conflict event')}{detail}")
        if int(ev.get("crashes", 0)):
            evidence.append(f"{_plural(int(ev['crashes']), 'AI-observed crash', 'AI-observed crashes')} (counted at maximum severity)")
        evidence.append(f"Average conflict severity {h['avg_near_miss_severity']:.2f} (0–1 scale)")
        if min_ttc is not None:
            evidence.append(f"Lowest observed time-to-collision {min_ttc:.2f} s")
    elif analyses:
        evidence.append(f"{_plural(analyses, 'video')} analysed here — no conflict events detected")
    else:
        evidence.append("No AI-observed conflict evidence yet")

    key: list[str] = []
    review: list[str] = []
    if has_hist and f["accidents_norm"] >= 0.6:
        key.append("High historical crash frequency relative to other sites")
        review.append("Schedule a formal road-safety audit (geometry, signage, lighting)")
    elif has_hist and f["accidents_norm"] >= 0.3:
        key.append("Moderate historical crash frequency")
    if has_hist and f["fatalities_norm"] >= 0.5:
        key.append("Elevated fatality history")
        review.append("Assess speed management on the approaches")
    if vp:
        key.append("Vehicle–pedestrian conflict observed")
        review.insert(0, "Inspect crossing visibility and pedestrian sight lines")
        review.append("Review signal timing and pedestrian protection at the crossing")
    if vv:
        key.append("Vehicle–vehicle conflict observed")
        review.append("Review junction priority and turning movements")
    if min_ttc is not None and min_ttc < 1.0:
        key.append("Very low time-to-collision during an observed interaction")
        review.append("Check approach speeds and stopping sight distance")
    elif n_conf and f["avg_severity_norm"] >= 0.66:
        key.append("Observed conflicts tend to be severe")
    if not key:
        key.append("No single factor stands out; ranked on combined evidence")
    if not review:
        review.append("Include in the next routine road-safety review cycle")

    hist_c = h.get("historical_component")
    obs_c = h.get("observed_component")
    if hist_c is not None and obs_c is not None and (hist_c or obs_c):
        driver = "AI-observed conflict evidence" if obs_c > hist_c else "historical crash record"
        headline = f"{h['risk_tier']} review priority — driven mainly by {driver}"
    else:
        headline = f"{h['risk_tier']} review priority"

    return {"headline": headline, "evidence": evidence, "key_factors": key, "review": review[:3]}


def generate_report(hotspots: List[dict]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_events = sum(int(h.get("near_miss_count", 0)) for h in hotspots)
    lines = [
        "# Verge — Road-Risk Review Priorities",
        "",
        f"_Generated {generated_at} · {_plural(len(hotspots), 'location')} · "
        f"{_plural(n_events, 'AI-detected conflict event')} · historical figures: "
        f"{config.HISTORICAL_DATA_LABEL}_",
        "",
        _DISCLAIMER,
        "",
        "## Ranked locations",
        "",
        "| # | Location | Priority | Crashes | Fatalities | AI conflicts |",
        "|---|---|---|---|---|---|",
    ]
    for h in hotspots:
        crashes = h["accidents_count"] if h.get("has_historical_record", True) else "n/a"
        deaths = h["fatalities"] if h.get("has_historical_record", True) else "n/a"
        lines.append(
            f"| {h['rank']} | {h['name']} | {h['risk_tier']} {h['risk_score']:.2f} "
            f"| {crashes} | {deaths} | {h['near_miss_count']} |"
        )

    lines += ["", "## Location briefs", ""]
    for h in hotspots:
        b = h.get("brief") or site_brief(h)
        score_line = f"**{b['headline']}** · score {h['risk_score']:.2f}"
        if h.get("historical_component") is not None:
            score_line += (
                f" (historical {h['historical_component']:.2f} + "
                f"AI-observed {h['observed_component']:.2f})"
            )
        lines += [f"### {h['rank']}. {h['name']}", "", score_line, "", "**Evidence**"]
        lines += [f"- {x}" for x in b["evidence"]]
        lines += ["", "**Why it was flagged**"]
        lines += [f"- {x}" for x in b["key_factors"]]
        lines += ["", "**Recommended human review**"]
        lines += [f"- {x}" for x in b["review"]]
        tech = " · ".join(f"{k.replace('_norm', '')} {v:.2f}" for k, v in h["factors"].items())
        lines += ["", f"_Normalized factors (0–1): {tech} · location {h['lat']:.4f}, {h['lon']:.4f}_", ""]

    lines += [
        "---",
        "",
        "_Method: score = 0.35 × crashes + 0.30 × fatalities + 0.25 × AI-detected "
        "conflicts + 0.10 × conflict severity, each min-max normalized across "
        "locations, so priorities are relative to the other monitored sites. Historical "
        "figures are " + config.HISTORICAL_DATA_LABEL + "; replace them and calibrate "
        "the camera before using these rankings operationally (see docs/architecture.md)._",
    ]
    return "\n".join(lines)


def run(hotspots: List[dict], out_path: Path = config.REPORT_MD) -> str:
    report_text = generate_report(hotspots)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    return report_text
