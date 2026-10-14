"""
One-to-two page PDF "AI Near-Miss Evidence Report" for a single analysed
conflict event, built from the stored analysis (analysis.json) and the
peak-moment snapshot rendered by video_analysis.py. Everything on the page
comes from the actual analysis — nothing is templated with sample values.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from . import config
from .pdf import PdfDocument
from .video_analysis import fmt_time

INK = "#10231b"
MUTED = "#6b7568"
LINE = "#dfdccf"
ACCENT = "#1f6f4c"
ACCENT_2 = "#12463a"
PAPER = "#f6f4ee"
SEVERITY_COLORS = {"High": "#c1442e", "Moderate": "#dd8a2e", "Low": "#3f8f5c"}

M = 40.0  # page margin (pt)



def _model_name(analysis: dict) -> str:
    """Display name of the detection model, e.g. 'YOLO11x' from 'yolo11x.onnx'."""
    name = (analysis.get("diagnostics", {}).get("detector_name") or "").split(" ")[0]
    stem = name.rsplit(".", 1)[0] if name.endswith(".onnx") else name
    if not stem.lower().startswith("yolo"):
        return stem or "YOLO"
    rest = stem[4:]  # e.g. "11x", "v8n"
    return "YOLO" + rest.lower()

def _fmt_dt(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return iso or "-"


def _suggested_review(analysis: dict, e: dict) -> list[str]:
    # Location-linked analysis: use the same site brief as the dashboard.
    brief = (analysis.get("site_after") or {}).get("brief")
    if brief and brief.get("review"):
        return brief["review"]
    if e["kind"] == "vehicle_pedestrian":
        return [
            "Review pedestrian crossing visibility and vehicle-pedestrian interaction at this location.",
            "Check approach speeds and whether crossing protection is adequate.",
        ]
    return ["Review junction priority, turning movements and approach speeds at this location."]


def build_event_pdf(analysis: dict, event_index: int) -> bytes:
    events = analysis.get("events") or []
    e = next((x for x in events if x.get("event_index") == event_index), None)
    if e is None:
        raise ValueError(f"No conflict event #{event_index} in this analysis.")

    doc = PdfDocument()
    W = doc.w
    content_w = W - 2 * M
    loc = analysis.get("location") or {}
    sev = e.get("severity_label") or config.severity_band(e.get("peak_severity", e["severity"]))
    sev_color = SEVERITY_COLORS.get(sev, ACCENT)

    # ---- header band ------------------------------------------------------
    doc.rect(0, 0, W, 70, fill=ACCENT_2)
    doc.text(M, 34, "VERGE", 20, bold=True, color="#ffffff")
    doc.text(M, 52, "AI NEAR-MISS EVIDENCE REPORT", 9.5, bold=True, color="#cfe3d8")
    doc.text(W - M, 34, f"Analysis {analysis.get('analysis_id', '-')}", 8.5, color="#cfe3d8", align="right")
    doc.text(W - M, 48, f"Event {event_index + 1} of {len(events)}", 8.5, color="#cfe3d8", align="right")

    # ---- context row ----------------------------------------------------------
    y = 96
    cols = [
        ("LOCATION", loc.get("name") or "Not linked to a location"),
        ("ANALYSIS TIME", _fmt_dt(analysis.get("created_at"))),
        ("CONFLICT TYPE", e.get("pair_label", "").replace("↔", "-")),
        ("SOURCE CLIP", analysis.get("source_name") or "uploaded video"),
    ]
    col_w = content_w / 4
    max_y = y
    for i, (label, value) in enumerate(cols):
        x = M + i * col_w
        doc.text(x, y, label, 7, bold=True, color=MUTED)
        yy = doc.paragraph(x, y + 13, value, 9.5, col_w - 10, bold=True, color=INK, leading=1.25)
        max_y = max(max_y, yy)
    if loc.get("lat") is not None:
        doc.text(M, max_y + 1, f"{loc['lat']:.5f}, {loc['lon']:.5f}" + ("" if loc.get("has_historical_record", True) else "  (map pin, no historical crash record)"), 7.5, color=MUTED)
        max_y += 10
    y = max_y + 10

    # ---- verdict strip --------------------------------------------------------
    doc.rect(M, y, content_w, 26, fill=sev_color)
    crash = e.get("event_type") == "crash"
    doc.text(M + 10, y + 17, "CRASH DETECTED" if crash else "NEAR-MISS DETECTED", 11, bold=True, color="#ffffff")
    doc.text(W - M - 10, y + 17, f"{'Impact' if crash else 'Conflict peak'} at {fmt_time(e.get('peak_t', e['t']))}  |  Severity {sev.upper()}",
             9.5, bold=True, color="#ffffff", align="right")
    y += 40

    # ---- evidence image + metrics ---------------------------------------------
    img_w = 300.0
    img_h = 0.0
    snap = (e.get("evidence") or {}).get("snapshot")
    if snap and Path(snap).exists():
        data = Path(snap).read_bytes()
        im = cv2.imread(snap)
        if im is not None:
            ph, pw = im.shape[:2]
            img_h = img_w * ph / pw
            doc.jpeg(data, pw, ph, M, y, img_w, img_h)
            doc.rect(M, y, img_w, img_h, stroke=LINE, lw=0.75)
    if img_h == 0:
        img_h = 60
        doc.rect(M, y, img_w, img_h, fill=PAPER, stroke=LINE)
        doc.text(M + img_w / 2, y + 34, "Evidence frame not available", 9, color=MUTED, align="center")
    cap_y = doc.paragraph(M, y + img_h + 12,
                          f"Annotated frame at the conflict peak ({fmt_time(e.get('peak_t', e['t']))}). "
                          "Boxes and labels are the tracker's own output; faces are blurred before storage.",
                          7.5, img_w, color=MUTED, leading=1.3)

    mx = M + img_w + 18
    mw = W - M - mx
    doc.text(mx, y + 8, "CONFLICT METRICS", 8, bold=True, color=ACCENT)
    ttc = e.get("min_ttc_s")
    rows = [
        ("Time of conflict", fmt_time(e.get("peak_t", e["t"]))),
        ("Indicator", "Observed collision (crash)" if e.get("event_type") == "crash" else
         "Close pass (proximity at speed)" if e.get("indicator") == "close_pass" else "Collision course (TTC)"),
        ("Time-to-collision (min.)", f"{ttc:.2f} s" if ttc is not None else
         ("Not applicable (passing)" if e.get("indicator") == "close_pass" else "Not measurable")),
        ("Minimum distance", "about 0 m (measurement limit)" if e.get("contact_phase") else
         (f"{e['min_distance_m']:.1f} m" if e.get("min_distance_m") is not None else "-")),
        ("Severity", f"{sev} ({e.get('peak_severity', e['severity']):.2f})"),
        ("Conflict duration", f"{e.get('zone_duration_s', 0):.1f} s"),
    ]
    if e.get("pet_s") is not None:
        rows.insert(3, ("Post-encroachment time", f"{e['pet_s']:.2f} s"))
    rows += [
        ("Road users tracked", f"{analysis.get('vehicles_tracked', 0)} vehicle(s), {analysis.get('pedestrians_tracked', 0)} pedestrian(s)"),
        ("Clip analysed", f"{analysis.get('duration_s', 0):.1f} s, {analysis.get('frames_total', analysis.get('frames_processed', 0))} frames"),
        ("Detector", f"Object detector ({_model_name(analysis)})" if analysis.get("detector") == "object" else "Motion-based"),
    ]
    ry = y + 22
    for label, value in rows:
        doc.line(mx, ry - 9, mx + mw, ry - 9, color=LINE)
        doc.text(mx, ry + 3, label, 8.5, color=MUTED)
        doc.text(mx + mw, ry + 3, value, 9, bold=True, color=INK, align="right")
        ry += 21
    doc.line(mx, ry - 9, mx + mw, ry - 9, color=LINE)
    if e.get("pet_s") is None:
        ry = doc.paragraph(mx, ry + 4, "PET not reported: the two paths did not share a conflict point.",
                           7, mw, color=MUTED)
    y = max(cap_y, ry) + 12

    # ---- observation / why flagged (two columns) ---------------------------------
    half = (content_w - 20) / 2
    y0 = y
    ya = _bullets(doc, M, y0, "AI OBSERVATION", e.get("what_happened", []), half)
    yb = _bullets(doc, M + half + 20, y0, "WHY IT WAS FLAGGED", e.get("why_flagged", []), half)
    y = max(ya, yb) + 6

    # ---- site priority ---------------------------------------------------------------
    upd = analysis.get("site_update")
    if upd and upd.get("before") and upd.get("after"):
        y = _ensure(doc, y, 90)
        doc.text(M, y, "SITE REVIEW PRIORITY", 8, bold=True, color=ACCENT)
        y += 8
        doc.rect(M, y, content_w, 50, fill=PAPER, stroke=LINE)
        b, a = upd["before"], upd["after"]
        cells = [
            ("Historical evidence", f"{a['historical_component']:.2f}"),
            ("AI-observed evidence", f"{a['observed_component']:.2f}  ({a['observed_component'] - b['observed_component']:+.2f})"),
            ("Updated review priority", f"{a['risk_score']:.2f}  {a['risk_tier']}"),
            ("Rank", f"#{b['rank']} -> #{a['rank']} of {a['of']}"),
        ]
        cw = content_w / 4
        for i, (label, value) in enumerate(cells):
            doc.text(M + 10 + i * cw, y + 18, label, 7.5, color=MUTED)
            doc.text(M + 10 + i * cw, y + 36, value, 11, bold=True, color=INK)
        y += 62
        y = doc.paragraph(M, y, upd.get("reason", "") + " " + upd.get("scoring_note", ""), 8, content_w, color=MUTED)
        y += 6

    # ---- suggested review -----------------------------------------------------------
    y = _ensure(doc, y, 70)
    y = _bullets(doc, M, y, "SUGGESTED HUMAN REVIEW", _suggested_review(analysis, e), content_w)
    y += 4

    # ---- privacy / methodology ---------------------------------------------------------
    th = analysis.get("thresholds", {})
    base = (
        "Decision support for human road-safety review. This report describes an observed interaction between "
        "tracked road users; it is not a finding of fault and not a prediction that a crash would have occurred. "
    )
    if analysis.get("detector") == "object":
        detail = (
            f"Road users were detected with a pretrained object detector ({_model_name(analysis)}) on sampled frames and tracked "
            "across frames. A conflict is confirmed only when two tracked road users are on a collision course with "
            f"a time-to-collision at or below {th.get('ttc_s', config.CONFLICT_TTC_S):g} s "
            f"({th.get('ttc_vv_s', config.CONFLICT_TTC_VV_S):g} s for two vehicles), an estimated clearance of at most "
            f"{th.get('distance_m', config.CONFLICT_MAX_GAP_M):g} m, sustained over consecutive frames. Distances are "
            "estimated from the apparent size of the road users and a perspective (ground-plane) model fitted to the "
            "clip; there is no camera calibration, so they are approximate and a reviewer should confirm the event "
            "in the replay. "
        )
    else:
        detail = (
            f"A conflict is flagged when two tracked road users come within {th.get('distance_m', config.DIST_THRESH_M):g} m "
            f"while the time-to-collision is at or below {th.get('ttc_s', config.TTC_THRESH_S):g} s. Distances use an "
            f"uncalibrated demo scale of {th.get('pixels_per_meter', config.PIXELS_PER_METER):g} px/m and detection is "
            "background subtraction (intended for synthetic demo clips), so a reviewer should confirm the event in the replay. "
        )
    method = base + detail + (
        "Privacy: faces are blurred before any frame is written, the uploaded source video is deleted after "
        "analysis, and only anonymized trajectories (position, time, class, track id) are kept."
    )
    y = _ensure(doc, y, 80)
    doc.text(M, y, "PRIVACY AND METHODOLOGY", 8, bold=True, color=ACCENT)
    doc.paragraph(M, y + 13, method, 7.5, content_w, color=MUTED, leading=1.4)

    _footer(doc)
    return doc.to_bytes()


def _bullets(doc: PdfDocument, x: float, y: float, title: str, items: list[str], width: float) -> float:
    doc.text(x, y, title, 8, bold=True, color=ACCENT)
    y += 14
    for it in items:
        doc.rect(x, y - 5, 3, 3, fill=ACCENT)
        y = doc.paragraph(x + 10, y, it, 9, width - 10, color=INK, leading=1.3) + 2
    return y + 6


def _ensure(doc: PdfDocument, y: float, needed: float) -> float:
    if y + needed > doc.h - 50:
        _footer(doc)
        doc.add_page()
        return 50
    return y


def _footer(doc: PdfDocument) -> None:
    doc.line(M, doc.h - 34, doc.w - M, doc.h - 34, color=LINE)
    doc.text(M, doc.h - 22, "Verge  |  AI road-risk and near-miss intelligence  |  decision support, not fault assignment",
             7, color=MUTED)
    doc.text(doc.w - M, doc.h - 22, f"Page {doc.page_count}", 7, color=MUTED, align="right")
