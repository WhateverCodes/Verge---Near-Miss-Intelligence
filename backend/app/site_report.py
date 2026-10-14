"""
One-page PDF "site review brief" for a monitored hotspot.

Built from exactly the same record and text the dashboard shows (the hotspot
from site_evidence.current_hotspots(), its report.site_brief() and its
hotspot_insight.hotspot_summary()), with the same dependency-free PDF writer
as the per-event evidence report.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import config, hotspot_insight
from .evidence_report import ACCENT, ACCENT_2, INK, LINE, M, MUTED, PAPER, _bullets, _ensure, _footer
from .pdf import PdfDocument

TIER_COLORS = {"Critical": "#c1442e", "High": "#dd8a2e", "Moderate": "#b9971f", "Low": "#3f8f5c"}


def build_site_pdf(h: dict, total_sites: int) -> bytes:
    doc = PdfDocument()
    W = doc.w
    cw = W - 2 * M
    brief = h.get("brief") or {"evidence": [], "key_factors": [], "review": []}
    hist = h.get("has_historical_record", True)
    n_ai, n_crash = hotspot_insight.ai_counts(h)

    # header band
    doc.rect(0, 0, W, 70, fill=ACCENT_2)
    doc.text(M, 34, "VERGE", 20, bold=True, color="#ffffff")
    doc.text(M, 52, "ROAD-SAFETY SITE REVIEW BRIEF", 9.5, bold=True, color="#cfe3d8")
    doc.text(W - M, 34, datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"), 8.5, color="#cfe3d8", align="right")
    doc.text(W - M, 48, f"Rank #{h['rank']} of {total_sites} monitored locations", 8.5, color="#cfe3d8", align="right")

    # location + priority strip
    y = 98
    y = doc.paragraph(M, y, h["name"], 15, cw, bold=True, color=INK)
    doc.text(M, y, f"{h['lat']:.5f}, {h['lon']:.5f}" + ("" if hist else "  (map pin, no historical crash record)"),
             7.5, color=MUTED)
    y += 12
    col = TIER_COLORS.get(h["risk_tier"], ACCENT)
    doc.rect(M, y, cw, 26, fill=col)
    doc.text(M + 10, y + 17, f"{h['risk_tier'].upper()} REVIEW PRIORITY", 11, bold=True, color="#ffffff")
    doc.text(W - M - 10, y + 17, f"Risk score {h['risk_score']:.2f}  |  historical {h.get('historical_component', 0):.2f} "
             f"+ AI-observed {h.get('observed_component', 0):.2f}", 9.5, bold=True, color="#ffffff", align="right")
    y += 40

    # key figures
    figs = [
        ("Recorded crashes", str(h["accidents_count"]) if hist else "no record", str(h.get("years") or "")),
        ("Fatalities", str(h["fatalities"]) if hist else "-", ""),
        ("Injuries", str(h["injuries"]) if hist else "-", ""),
        ("AI conflict events", str(n_ai), f"incl. {n_crash} crash{'es' if n_crash != 1 else ''}" if n_crash else ""),
    ]
    doc.rect(M, y, cw, 52, fill=PAPER, stroke=LINE)
    fw = cw / len(figs)
    for i, (label, value, sub) in enumerate(figs):
        x = M + 10 + i * fw
        doc.text(x, y + 16, label, 7.5, color=MUTED)
        doc.text(x, y + 36, value, 15, bold=True, color=INK)
        if sub:
            doc.text(x, y + 47, sub, 7, color=MUTED)
    y += 68

    # summary (same text as the dashboard's text / audio summary and the assistant)
    doc.text(M, y, "SUMMARY", 8, bold=True, color=ACCENT)
    y = doc.paragraph(M, y + 14, h.get("summary") or hotspot_insight.hotspot_summary(h), 9.5, cw, color=INK, leading=1.4) + 8

    y = _ensure(doc, y, 80)
    y = _bullets(doc, M, y, "EVIDENCE", brief["evidence"], cw)
    y = _ensure(doc, y, 60)
    y = _bullets(doc, M, y, "WHY IT WAS FLAGGED", brief["key_factors"], cw)
    y = _ensure(doc, y, 60)
    y = _bullets(doc, M, y, "RECOMMENDED HUMAN REVIEW", brief["review"], cw)

    # method
    y = _ensure(doc, y, 90)
    f = h.get("factors", {})
    method = (
        "Review priority is a transparent weighted sum of normalised factors: "
        f"{config.W_ACCIDENTS:g} x crashes ({f.get('accidents_norm', 0):.2f}) + {config.W_FATALITIES:g} x fatalities "
        f"({f.get('fatalities_norm', 0):.2f}) + {config.W_NEAR_MISS:g} x AI conflict events ({f.get('near_miss_norm', 0):.2f}) "
        f"+ {config.W_SEVERITY:g} x conflict severity ({f.get('avg_severity_norm', 0):.2f}), each min-max normalised across "
        "monitored sites. It is a relative review priority for human road-safety engineers, not a crash probability, "
        "not a finding of fault, and not a prediction that a crash will occur. Historical figures: "
        f"{config.HISTORICAL_DATA_LABEL}."
    )
    doc.text(M, y, "METHOD AND LIMITS", 8, bold=True, color=ACCENT)
    doc.paragraph(M, y + 13, method, 7.5, cw, color=MUTED, leading=1.4)
    _footer(doc)
    return doc.to_bytes()
