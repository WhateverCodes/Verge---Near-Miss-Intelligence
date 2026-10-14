"""
One source of truth for how a monitored hotspot is described.

Every place that explains a hotspot (the dashboard's View Analysis dialog and
its text / audio summary, the Road Safety Assistant, and the site PDF brief)
uses these functions on the same record returned by
site_evidence.current_hotspots(), so a number can never differ between them.
Nothing here computes a new score: it only reads the existing fields
(risk_scoring + site_evidence) and the existing brief (report.site_brief).
"""
from __future__ import annotations

import re
from typing import List


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def ai_counts(h: dict) -> tuple[int, int]:
    """(AI-observed conflict events, of which crashes)."""
    n = int(h.get("near_miss_count") or 0)
    crashes = int((h.get("evidence") or {}).get("crashes") or 0)
    return n, crashes


def main_driver(h: dict) -> str:
    """Which evidence mainly drives the score: 'historical' or 'ai'."""
    return "ai" if (h.get("observed_component") or 0) > (h.get("historical_component") or 0) else "historical"


def hotspot_summary(h: dict) -> str:
    """3-5 sentence, decision-support summary of one hotspot (English).
    Only facts from the record; never a crash prediction."""
    hist = h.get("has_historical_record", True)
    n, nc = ai_counts(h)
    tier, score = h["risk_tier"], f"{h['risk_score']:.2f}"
    review = (h.get("brief") or {}).get("review", [])[:2]
    s: List[str] = []
    if hist:
        driver = "AI-observed conflict evidence" if main_driver(h) == "ai" else "historical crash evidence"
        s.append(f"{h['name']} is classified as {tier} review priority with a risk score of {score}, driven mainly by {driver}.")
        years = str(h.get("years") or "").replace("-", "–")
        s.append(f"The location recorded {_plural(h['accidents_count'], 'crash', 'crashes')} and "
                 f"{_plural(h['fatalities'], 'fatality', 'fatalities')} during {years}.")
    else:
        s.append(f"{h['name']} has no historical crash record; its {tier} priority (score {score}) comes from "
                 "AI-observed conflict evidence.")
    if n:
        s.append(f"The AI observed {_plural(n, 'conflict event', 'conflict events')} here"
                 + (f", including {_plural(nc, 'crash', 'crashes')}" if nc else "") + ".")
    else:
        s.append("No AI-observed conflict evidence is currently available for this location.")
    if review:
        s.append("For human review, Verge recommends: " + "; and ".join(r[0].lower() + r[1:] for r in review) + ".")
    else:
        s.append("A review by a road-safety engineer is recommended.")
    s.append("This is decision support, not a prediction of crashes.")
    return " ".join(s[:5])


def slug(text: str, max_len: int = 60) -> str:
    """Safe file-name fragment: lowercase letters, digits and single dashes."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:max_len].rstrip("-")) or "report"


def site_report_filename(h: dict) -> str:
    return f"verge-road-safety-report-{slug(h['name'])}.pdf"
