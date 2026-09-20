"""
Turns ranked hotspots into a plain-language Markdown report.

Every recommendation is phrased as a suggestion for a human traffic
engineer / enforcement planner to review — never as an automated finding of
fault or a mandate. Every number shown traces back to a named factor
(accident history, near-miss count/severity) so nothing here is a black
box. See README "Ethics & privacy by design".
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

from . import config

_DISCLAIMER = (
    "> **How to read this report.** Scores below combine historical "
    "accident/fatality records with AI-detected near-miss patterns into a "
    "single ranking. This is **decision support for human review, not an "
    "automated finding of fault or a guarantee of danger.** No individuals "
    "are identified anywhere in this pipeline; near-miss detection works "
    "only on anonymized object trajectories, and any face regions in source "
    "footage are blurred before frames are ever written to disk."
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


def generate_report(hotspots: List[dict]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Verge — Road-Risk Hotspot Report",
        "",
        f"_Generated {generated_at}_",
        "",
        _DISCLAIMER,
        "",
        "## Ranked locations",
        "",
        "| Rank | Location | Risk score | Tier | Accidents | Fatalities | Near-misses | Avg. near-miss severity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in hotspots:
        lines.append(
            f"| {h['rank']} | {h['name']} | {h['risk_score']:.2f} | {h['risk_tier']} "
            f"| {h['accidents_count']} | {h['fatalities']} | {h['near_miss_count']} "
            f"| {h['avg_near_miss_severity']:.2f} |"
        )

    lines += ["", "## Site-by-site detail", ""]
    for h in hotspots:
        lines += [
            f"### {h['rank']}. {h['name']} — {h['risk_tier']} risk (score {h['risk_score']:.2f})",
            "",
            f"- Coordinates: `{h['lat']}, {h['lon']}`",
            f"- Historical accidents: **{h['accidents_count']}** ({h['years']}), fatalities: **{h['fatalities']}**, injuries: **{h['injuries']}**",
            f"- AI-detected near-miss events: **{h['near_miss_count']}**, average severity: **{h['avg_near_miss_severity']:.2f}** (0–1 scale)",
            "- Contributing factors (normalized 0–1, higher = more concerning): "
            + ", ".join(f"`{k}` = {v}" for k, v in h["factors"].items()),
            f"- **Suggested next step for human review:** {_recommendation_for(h)}",
            "",
        ]

    lines += [
        "---",
        "",
        "_This report was generated automatically from sample/synthetic data "
        "for demonstration purposes. Replace `data/sample_accidents.csv` and "
        "the input video with real, locally-calibrated data before using "
        "these rankings operationally — see `docs/architecture.md`._",
    ]
    return "\n".join(lines)


def run(hotspots: List[dict], out_path: Path = config.REPORT_MD) -> str:
    report_text = generate_report(hotspots)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text)
    return report_text
