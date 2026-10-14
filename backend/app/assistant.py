"""
Verge Road Safety Assistant: grounded, deterministic question answering.

No language model is involved. A question is matched to one of a fixed set
of intents, and every answer is assembled from Verge's own data:

  * monitored hotspots   site_evidence.current_hotspots() (same records, brief
                         and summary as the map, the report and the PDFs)
  * latest video         the stored analysis.json of the analysis in context
                         (or the most recent one on disk)
  * how Verge works      fixed explanations whose numbers are read from
                         config.py, so they match what the code actually does

When the data can't answer a question, the reply says so instead of guessing.
Answers are a list of blocks: {"type": "text" | "bullets" | "table", ...}.
"""
from __future__ import annotations

import difflib
import re
from typing import List, Optional

from . import analysis_service, config, hotspot_insight, site_evidence

SRC_HOTSPOTS = "Based on current Verge hotspot data"
SRC_VIDEO = "Based on the latest video analysis"
SRC_METHOD = "Based on how Verge is implemented"
SRC_SCOPE = "Verge's data covers monitored locations and analysed videos only"

# words that don't identify a location on their own
_GENERIC = {"road", "junction", "approach", "signal", "crossing", "chowk", "bus", "depot", "flyover", "phase",
            "x", "the", "intersection", "synthetic", "clip", "pinned", "location", "and", "of", "1", "2", "3"}


def _t(text: str) -> dict:
    return {"type": "text", "text": text}


def _b(items: List[str]) -> dict:
    return {"type": "bullets", "items": items}


def _table(columns: List[str], rows: List[List[str]]) -> dict:
    return {"type": "table", "columns": columns, "rows": rows}


def _fmt_time(t) -> str:
    if t is None:
        return "-"
    m, s = divmod(float(t), 60)
    return f"{int(m):02d}:{s:04.1f}"


def _words(q: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", q.lower())


# common chat shorthand -> words the intent rules understand
_SHORTHAND = {
    "wt": "what", "wat": "what", "wht": "what", "whats": "what is", "abt": "about", "abut": "about",
    "hw": "how", "y": "why", "u": "you", "r": "are", "ur": "your", "whr": "where", "wch": "which",
    "whch": "which", "cmpr": "compare", "vid": "video", "pls": "please", "plz": "please", "thx": "thanks",
    "info": "information", "tel": "tell", "abot": "about", "wich": "which", "wher": "where",
}


def normalize(question: str) -> str:
    q = (question or "").strip().lower().replace("’", "'")
    return " ".join(_SHORTHAND.get(w, w) for w in re.split(r"\s+", q) if w)


# words that belong to Verge's own vocabulary, i.e. never a place name
_VOCAB = {
    "this", "that", "it", "here", "there", "hotspot", "hotspots", "location", "locations", "site", "sites", "junction",
    "video", "clip", "footage", "analysis", "latest", "last", "conflict", "conflicts", "event", "events", "crash",
    "crashes", "near", "miss", "misses", "ttc", "pet", "risk", "score", "priority", "privacy", "verge", "yolo",
    "fault", "data", "evidence", "recommendation", "recommendations", "review", "the", "a", "an", "top", "all",
    "them", "these", "those", "method", "system", "fatalities", "injuries", "accidents", "safety", "road", "roads",
    "report", "pdf", "email", "summary", "model", "you", "your", "me", "us", "it's", "is", "are", "in", "of", "for",
    "pedestrian", "pedestrians", "vehicle", "vehicles", "car", "cars", "truck", "trucks", "bus", "buses", "bike",
    "bikes", "two", "wheeler", "wheelers", "cyclist", "cyclists", "speed", "speeding", "weather", "night", "rain",
    "authorities", "police", "budget", "cost", "people", "person", "camera", "cameras", "cctv", "detection", "tracking",
    "accuracy", "ai", "observed", "historical", "history", "and", "or", "with", "other", "others", "rest", "more",
    "severity", "severe", "danger", "dangerous", "safe", "safest", "worst", "best", "city", "pune", "india", "here",
    "now", "today", "what", "why", "how", "which", "when", "who", "please", "thanks", "help", "features", "it?",
}

_PLACE_PATTERNS = [
    r"(?:what|how)\s+about\s+(.+)",
    r"tell\s+me\s+about\s+(.+)",
    r"information\s+(?:on|about|for)\s+(.+)",
    r"(?:risk|crashes|fatalities|data|details|analysis|priority|score|evidence|status)\s+(?:of|for|at|in|near)\s+(.+)",
    r"^is\s+(.+?)\s+(?:dangerous|safe|risky|critical|high risk|a hotspot|monitored)",
    r"^(?:show|analy[sz]e|check)\s+(.+)",
]


def _unknown_place(q: str) -> Optional[str]:
    """A place the user named that is not a monitored hotspot, or None."""
    for pat in _PLACE_PATTERNS:
        m = re.search(pat, q)
        if not m:
            continue
        cand = re.sub(r"[?.!,]+$", "", m.group(1)).strip()
        cand = re.sub(r"^(?:the|a)\s+", "", cand)
        words = _words(cand)
        if not words or len(words) > 6 or all(w in _VOCAB for w in words):
            return None
        return " ".join(w.capitalize() if len(w) > 2 else w.upper() for w in words)
    return None


def _key_tokens(name: str) -> List[str]:
    return [w for w in _words(name) if w not in _GENERIC and not w.isdigit() and len(w) >= 2]


def _mentioned(q: str, sites: List[dict]) -> List[dict]:
    """Sites named in the question, in the order they appear. Small typos in a
    distinctive name word are tolerated (e.g. "swargte", "hinjewadi")."""
    ws = _words(q)
    hits = []
    for h in sites:
        pos = []
        for t in _key_tokens(h["name"]):
            if t in ws:
                pos.append(ws.index(t))
            elif len(t) >= 5:
                close = difflib.get_close_matches(t, [w for w in ws if len(w) >= 4], n=1, cutoff=0.8)
                if close:
                    pos.append(ws.index(close[0]))
        if pos:
            hits.append((min(pos), h))
    return [h for _, h in sorted(hits, key=lambda x: x[0])]


def _has(q: str, *terms: str) -> bool:
    return any(t in q for t in terms)


def _row(h: dict) -> List[str]:
    hist = h.get("has_historical_record", True)
    n, nc = hotspot_insight.ai_counts(h)
    return [f"#{h['rank']} {h['name']}", f"{h['risk_tier']} {h['risk_score']:.2f}",
            str(h["accidents_count"]) if hist else "n/a", str(h["fatalities"]) if hist else "n/a",
            f"{n}" + (f" ({nc} crash{'es' if nc != 1 else ''})" if nc else "")]


_COLS = ["Location", "Risk", "Crashes", "Fatalities", "AI conflicts"]


# ---- hotspot answers ---------------------------------------------------------------

def _site_why(h: dict) -> List[dict]:
    b = h.get("brief") or {}
    return [_t(h["summary"]), _b([f"Why it was flagged: {x}" for x in b.get("key_factors", [])[:2]])]


def _site_evidence(h: dict) -> List[dict]:
    b = h.get("brief") or {}
    items = list(b.get("evidence", []))
    return [_t(f"The system currently records this evidence for {h['name']} "
               f"({h['risk_tier']}, risk score {h['risk_score']:.2f}):"), _b(items),
            _t(f"Score split: historical {h.get('historical_component', 0):.2f} + AI-observed "
               f"{h.get('observed_component', 0):.2f}.")]


def _site_review(h: dict) -> List[dict]:
    b = h.get("brief") or {}
    return [_t(f"For {h['name']}, Verge recommends human review of:"), _b(b.get("review", [])),
            _t("These are suggestions for a road-safety engineer to confirm on site, not automated decisions.")]


def _compare(sites: List[dict], title: str) -> List[dict]:
    blocks = [_t(title), _table(_COLS, [_row(h) for h in sites])]
    top = sites[0]
    drivers = {hotspot_insight.main_driver(h) for h in sites}
    note = (f"{top['name']} has the highest review priority of these ({top['risk_score']:.2f}). "
            + ("All of them are driven mainly by historical crash evidence." if drivers == {"historical"} else
               "Some are driven mainly by AI-observed conflict evidence." if "ai" in drivers else ""))
    blocks.append(_t(note.strip()))
    return blocks


# ---- video answers ---------------------------------------------------------------------

def _latest_analysis(analysis_id: Optional[str]) -> Optional[dict]:
    if analysis_id:
        try:
            return analysis_service.load_analysis(analysis_id)
        except (FileNotFoundError, ValueError):
            pass
    dirs = sorted((d for d in config.ANALYSES_DIR.glob("*") if (d / "analysis.json").exists()),
                  key=lambda d: (d / "analysis.json").stat().st_mtime, reverse=True)
    for d in dirs:
        try:
            return analysis_service.load_analysis(d.name)
        except (FileNotFoundError, ValueError):
            continue
    return None


def _event_line(e: dict) -> str:
    kind = "Crash" if e.get("event_type") == "crash" else "Near-miss"
    ttc = e.get("min_ttc_s")
    bits = [f"{kind} at {_fmt_time(e.get('peak_t'))}: {e.get('pair_label', '')}",
            f"{e.get('severity_label', '')} severity"]
    if e.get("event_type") != "crash" and ttc is not None:
        bits.append(f"TTC {ttc:.2f} s")
    if e.get("min_distance_m") is not None:
        bits.append("clearance about 0 m" if e.get("contact_phase") else f"clearance {e['min_distance_m']:.1f} m")
    return " · ".join(bits)


def _worst(events: List[dict]) -> dict:
    return max(events, key=lambda e: (e.get("event_type") == "crash", e.get("peak_severity", e.get("severity", 0))))


def _video_answer(q: str, a: dict) -> List[dict]:
    events = a.get("events") or []
    clip = a.get("source_name") or "the uploaded clip"
    loc = (a.get("location") or {}).get("name")
    where = f" at {loc}" if loc else " (not linked to a location)"
    n_crash = sum(1 for e in events if e.get("event_type") == "crash")
    n_near = len(events) - n_crash
    if not events:
        unassessed = ((a.get("diagnostics") or {}).get("conflicts") or {}).get("camera_too_unstable")
        return [_t(f"The latest analysis ({clip}{where}) found no confirmed near-miss or crash."
                   + (" The camera moved too much to measure distances, so conflicts were not assessed."
                      if unassessed else ""))]
    w = _worst(events)
    if _has(q, "how many", "number of", "count"):
        return [_t(f"The latest analysis ({clip}{where}) confirmed {len(events)} event"
                   f"{'s' if len(events) != 1 else ''}: {n_crash} crash{'es' if n_crash != 1 else ''} and "
                   f"{n_near} near-miss{'es' if n_near != 1 else ''}."), _b([_event_line(e) for e in events])]
    if _has(q, "when", "most significant", "worst", "most severe", "biggest"):
        return [_t(f"The most significant event in {clip} is at {_fmt_time(w.get('peak_t'))}."), _b([_event_line(w)])]
    if _has(q, "why", "considered", "flagged", "criteria"):
        return [_t(f"Why the event at {_fmt_time(w.get('peak_t'))} ({w.get('pair_label', '')}) was flagged:"),
                _b(w.get("why_flagged") or [])]
    if _has(q, "what happened", "explain", "describe", "detail") and _has(q, "event", "conflict", "crash", "near"):
        return [_t(f"What the analysis measured for the event at {_fmt_time(w.get('peak_t'))} ({w.get('pair_label', '')}):"),
                _b(w.get("what_happened") or [])]
    # summary
    blocks = [_t(a.get("verdict") or f"{len(events)} event(s) confirmed."),
              _t(f"Clip: {clip}{where} · {a.get('duration_s', 0):.1f} s · {a.get('vehicles_tracked', 0)} vehicles and "
                 f"{a.get('pedestrians_tracked', 0)} pedestrians tracked."),
              _b([_event_line(e) for e in events])]
    upd = a.get("site_update") or {}
    if upd.get("before") and upd.get("after"):
        b, af = upd["before"], upd["after"]
        blocks.append(_t(f"Site update: {upd.get('name', loc)} review priority {b['risk_score']:.2f} {b['risk_tier']} → "
                         f"{af['risk_score']:.2f} {af['risk_tier']}, rank #{b['rank']} → #{af['rank']}."))
    return blocks


# ---- how Verge works -------------------------------------------------------------------

def _method(q: str) -> Optional[List[dict]]:
    tiers = sorted(config.RISK_TIERS)
    if _has(q, "fault", "blame", "who is responsible", "responsible"):
        return [_t("No. Verge does not assign fault or identify individuals. It measures how road users moved "
                   "relative to each other (anonymous tracks) and flags interactions for a human road-safety "
                   "engineer to review. Legal responsibility is outside what the system measures.")]
    if _has(q, "predict", "forecast", "will there be", "will a crash", "future crash"):
        return [_t("No. Verge does not predict that a crash will happen. The review priority is a relative ranking "
                   "of monitored locations based on recorded crashes and observed conflicts, used to decide where "
                   "human road-safety review should look first. It is not a crash probability.")]
    if _has(q, "privacy", "face", "identify", "identity", "personal data", "number plate", "licence plate"):
        return ([_t("No. Verge does not identify people.")] if _has(q, "identify", "identity", "recogni") else []) + [_b(["Faces and pedestrians' heads are blurred before any frame is saved.",
                    "The uploaded source video is deleted after analysis.",
                    "Only anonymous trajectories are kept (position, time, class, track id); no identities.",
                    "Evidence is aggregated per location for road-safety review, not used to track people."])]
    if _has(q, "police", "enforce", "deploy", "challan", "penalt"):
        return [_t("Verge does not make enforcement or deployment decisions. It ranks monitored locations and "
                   "provides evidence (replays, measurements, reports) so authorities can decide what to review.")]
    if _has(q, "yolo", "object detect", "which model", "detector"):
        return [_t("Verge uses a pretrained YOLO11 object detector (COCO classes: person, bicycle, car, motorcycle, "
                   "bus, truck), exported to ONNX so it runs offline: YOLO11x on a GPU through ONNX Runtime "
                   "DirectML, YOLO11l on a CPU with OpenCV. It was chosen because it recognises individual road "
                   "users reliably, unlike the earlier motion-blob detector, which split one truck into many "
                   "fake pedestrians.")]
    if _has(q, "what is ttc", "ttc mean", "time to collision", "time-to-collision", "explain ttc", "what's ttc") \
            or q.strip() in ("ttc", "ttc?"):
        return [_t("TTC (time-to-collision) is how many seconds remain before two road users would touch if both "
                   f"kept their current speed and direction. Verge flags a near-miss when TTC drops to "
                   f"{config.CONFLICT_TTC_S:g} s or less ({config.CONFLICT_TTC_VV_S:g} s between two vehicles), within "
                   f"{config.CONFLICT_MAX_GAP_M:g} m, sustained over consecutive frames. Lower TTC means less time to react.")]
    if _has(q, "what is pet", "pet mean", "post-encroachment", "post encroachment", "explain pet") \
            or q.strip() in ("pet", "pet?"):
        return [_t("PET (post-encroachment time) is the time between one road user leaving a point and another "
                   "arriving at the same point. A small PET means their paths crossed with little margin, even if they "
                   "were never on a direct collision course. Verge reports PET only when both paths really pass "
                   f"through the same spot (within {config.PET_RADIUS_M:g} m).")]
    if _has(q, "historical") and _has(q, "why", "use", "need", "important"):
        return [_t("Historical crash records show where people have actually been killed or injured, which is the "
                   "strongest evidence of risk. They are slow and sparse, though, so Verge adds AI-observed near-misses "
                   "from video as an earlier signal. Both are shown separately in every score, and the historical "
                   "records are never modified.")]
    if _has(q, "crash detect", "detect a crash", "detect crash", "detects crash", "collision detect"):
        return [_t("A crash is reported only when all of these are measured: the two road users' footprints meet "
                   f"for at least {config.CRASH_MIN_CONTACT_S:g} s with their boxes overlapping; the gap was closing "
                   f"at {config.CRASH_MIN_CLOSING_MS:g} m/s or more just before; and straight after contact one of them "
                   f"shows a velocity change of at least {config.CRASH_DV_MS:g} m/s (beyond hard braking) or a sudden "
                   "change of shape (spin or rollover). Both tracks must stay on the same objects through the impact.")]
    if (_has(q, "near-miss", "near miss", "conflict") and _has(q, "how does", "how do", "how is", "how are",
                                                                   "explain how", "method")
            and not _has(q, "how many", "happened", "were detected", "the detected", "latest")):
        return [_b(["Detect: YOLO11 finds each road user in sampled frames.",
                    "Track: an IoU tracker with motion prediction follows them; camera shake is compensated.",
                    "Measure: a perspective model fitted to the clip gives distances on the road "
                    "(bird's-eye view for street-level cameras).",
                    f"Flag: a near-miss needs TTC ≤ {config.CONFLICT_TTC_S:g} s ({config.CONFLICT_TTC_VV_S:g} s for two "
                    f"vehicles) or a close pass ≤ {config.CONFLICT_CLOSE_PASS_VV_M:g} m at speed, a closing gap, "
                    "persistence over frames and a seen approach.",
                    "Evidence: each event gets a replay, measurements, reasons and a PDF for human review."])]
    if _has(q, "score", "priority") and _has(q, "how ", "calculat", "formula", "computed", "work", "explain the"):
        return [_t(f"Review priority (0–1) = {config.W_ACCIDENTS:g} × crashes + {config.W_FATALITIES:g} × fatalities + "
                   f"{config.W_NEAR_MISS:g} × AI conflict events + {config.W_SEVERITY:g} × conflict severity, each "
                   "normalised across the monitored sites. An AI-observed crash counts as a conflict at maximum severity."),
                _t("Risk levels on that score: " + ", ".join(
                    f"{lab} ≥ {t:.2f}" if t > 0 else f"{lab} < {tiers[1][0]:.2f}" for t, lab in tiers) + ".")]
    if _has(q, "historical") and _has(q, "ai", "observed") and _has(q, "difference", "vs", "versus", "compare", "between"):
        return [_b(["Historical evidence: recorded crashes and fatalities from the accident dataset (read-only).",
                    "AI-observed evidence: near-misses and crashes Verge measured in analysed videos at the location.",
                    "Each score splits exactly into the two parts, and both are shown so reviewers can see what drives it."])]
    if _has(q, "near-miss", "near miss", "near-misses") and _has(q, "why", "useful", "important", "matter"):
        return [_t("Near-misses happen far more often than crashes and follow the same patterns, so they reveal risky "
                   "locations earlier (the basis of the traffic-conflict technique). Measuring them lets authorities "
                   "review a junction before serious crashes accumulate in the records.")]
    if _has(q, "prioriti", "resources", "help traffic", "help authorities", "budget", "how can this help"):
        return [_t("Verge ranks monitored locations by combined crash history and observed conflicts, shows why each "
                   "is ranked, and packages the evidence (replays, figures, PDF reports). Engineers can start with the "
                   "Critical and High locations and justify the choice with the evidence.")]
    if _has(q, "how does verge", "what is verge", "what does verge"):
        return [_t("Verge detects and tracks road users in traffic video, measures near-misses and crashes, and "
                   "combines them with historical crash records into a ranked, explainable review priority for each "
                   "monitored location. It is decision support for human road-safety review.")]
    return None


# ---- entry point -------------------------------------------------------------------------

SUGGESTIONS = [
    "Why is this hotspot high risk?",
    "Compare the top hotspots",
    "Explain the latest video analysis",
    "What should authorities review?",
    "How does Verge detect conflicts?",
    "How is the risk score calculated?",
]


def answer(question: str, site_id: Optional[str] = None, analysis_id: Optional[str] = None) -> dict:
    q = normalize(question)
    sites = site_evidence.current_hotspots()
    by_id = {h["site_id"]: h for h in sites}
    ctx = by_id.get(site_id) if site_id else None
    named = _mentioned(q, sites)
    target = named[0] if named else ctx

    def out(blocks, source, used_site=None, used_video=None):
        return {"blocks": blocks, "source": source,
                "context": {"site": used_site and {"site_id": used_site["site_id"], "name": used_site["name"]},
                            "analysis_id": used_video}}

    if not q:
        return out([_t("Ask about a monitored hotspot, the latest video analysis, or how Verge works.")], SRC_METHOD)

    # 1. how Verge works / positioning
    m = _method(q)
    if m:
        return out(m, SRC_METHOD)

    # 1b. a named place that is not a monitored hotspot: say so, never invent it
    if not named:
        place = _unknown_place(q)
        if place:
            return out([_t(f"I don't currently have {place} in Verge's monitored hotspot data. I can analyze or "
                           f"compare the {len(sites)} locations currently available in the system, for example "
                           f"{sites[0]['name']} or {sites[1]['name']}.")], SRC_SCOPE)

    # 2. out of scope: questions about places / statistics Verge doesn't hold
    if _has(q, "safest", "most dangerous road", "in pune", "in india", "in the city", "citywide", "city-wide",
            "all roads", "every road", "national") and not named:
        low = min(sites, key=lambda h: h["risk_score"])
        top = sites[0]
        return out([_t("The current Verge dataset covers monitored locations rather than every road, so I can't "
                       "determine that across the city."),
                    _t(f"Among the {len(sites)} monitored locations, the lowest review priority is {low['name']} "
                       f"({low['risk_tier']} {low['risk_score']:.2f}) and the highest is {top['name']} "
                       f"({top['risk_tier']} {top['risk_score']:.2f}).")], SRC_SCOPE)
    if _has(q, "weather", "cause of", "caused the", "drunk", "speeding driver", "who was driving"):
        return out([_t("I don't have that evidence in the current Verge dataset. Verge records crash counts, "
                       "fatalities and injuries per location and measures conflicts in analysed video; it does not "
                       "record crash causes, weather or driver details.")], SRC_SCOPE)

    # 3. latest video
    if _has(q, "video", "clip", "footage", "latest analysis", "last analysis", "recording") or \
            (not named and _has(q, "conflict", "event", "crash", "near-miss", "near miss")
             and _has(q, "detected", "when did", "happened", "explain the", "summar", "how many", "most significant",
                      "why was", "considered")):
        a = _latest_analysis(analysis_id)
        if not a:
            return out([_t("Analyze a traffic video to ask Verge about detected conflicts.")], SRC_VIDEO)
        return out(_video_answer(q, a), SRC_VIDEO, used_video=a.get("analysis_id"))

    # 4. comparisons
    if _has(q, "compare", " vs ", "versus", "difference between"):
        if len(named) >= 2:
            return out(_compare(named[:5], "Comparison of the requested monitored locations:"), SRC_HOTSPOTS)
        mt = re.search(r"top\s*(\d+)", q)
        k = max(2, min(int(mt.group(1)) if mt else 3, len(sites)))
        return out(_compare(sites[:k], f"Top {k} monitored locations by review priority:"), SRC_HOTSPOTS)

    # 5. dataset-wide lists
    if _has(q, "which") and _has(q, "ai-observed", "ai observed", "ai conflict", "conflict evidence", "near-miss evidence",
                                 "ai evidence"):
        with_ai = [h for h in sites if h.get("near_miss_count")]
        if not with_ai:
            return out([_t("No monitored location currently has AI-observed conflict evidence.")], SRC_HOTSPOTS)
        return out([_t(f"{len(with_ai)} monitored location{'s have' if len(with_ai) != 1 else ' has'} AI-observed conflict evidence:"),
                    _table(_COLS, [_row(h) for h in with_ai])], SRC_HOTSPOTS)
    if _has(q, "driven") and _has(q, "historical", "history"):
        hs = [h for h in sites if h.get("historical_component", 0) > 0 and hotspot_insight.main_driver(h) == "historical"]
        return out([_t("Locations whose score is driven mainly by historical crash evidence:"),
                    _b([f"{h['name']} — historical {h['historical_component']:.2f} vs AI-observed "
                        f"{h['observed_component']:.2f}" for h in hs])], SRC_HOTSPOTS)
    if _has(q, "driven") and _has(q, " ai", "observed"):
        hs = [h for h in sites if hotspot_insight.main_driver(h) == "ai"]
        if not hs:
            return out([_t("No monitored location is currently driven mainly by AI-observed evidence.")], SRC_HOTSPOTS)
        return out([_t("Locations driven mainly by AI-observed conflict evidence:"),
                    _b([f"{h['name']} — AI-observed {h['observed_component']:.2f} vs historical "
                        f"{h['historical_component']:.2f}" for h in hs])], SRC_HOTSPOTS)
    tier_q = next((t for t in ("critical", "high", "moderate", "low") if re.search(rf"\b{t}\b", q)), None)
    if tier_q and _has(q, "which", "list", "show", "how many") and not named and not _has(q, "this", "here"):
        hs = [h for h in sites if h["risk_tier"].lower() == tier_q]
        if not hs:
            return out([_t(f"No monitored location is currently {tier_q.capitalize()}.")], SRC_HOTSPOTS)
        return out([_t(f"{len(hs)} monitored location{'s are' if len(hs) != 1 else ' is'} {tier_q.capitalize()}:"),
                    _table(_COLS, [_row(h) for h in hs])], SRC_HOTSPOTS)
    for field, label in (("fatalities", "fatalities"), ("injuries", "injuries"), ("accidents_count", "crashes")):
        words = {"fatalities": ("fatalit", "death", "deaths", "killed"), "injuries": ("injur",),
                 "accidents_count": ("crash", "accident")}[field]
        if _has(q, *words) and not named and not _has(q, "this", "here") and \
                _has(q, "most", "highest", "maximum", "max ", "fewest", "least", "lowest", "minimum"):
            hist = [h for h in sites if h.get("has_historical_record", True)]
            low = _has(q, "fewest", "least", "lowest", "minimum")
            best = (min if low else max)(h[field] for h in hist)
            top = [h for h in hist if h[field] == best]
            names = " and ".join(h["name"] for h in top)
            return out([_t(f"Based on the historical records, {names} {'has' if len(top) == 1 else 'have'} the "
                           f"{'fewest' if low else 'most'} {label} among monitored locations: {best}."),
                        _table(_COLS, [_row(h) for h in sorted(hist, key=lambda x: x[field], reverse=not low)[:3]])],
                       SRC_HOTSPOTS, used_site=top[0])
    if _has(q, "highest", "top priority", "most critical", "riskiest", "most risky", "rank first", "number one",
            "highest review") and not named:
        h = sites[0]
        return out([_t(f"{h['name']} has the highest review priority: {h['risk_tier']} {h['risk_score']:.2f}."),
                    _t(h["summary"])], SRC_HOTSPOTS, used_site=h)
    if _has(q, "lowest", "least risk", "least risky") and not named:
        h = min(sites, key=lambda x: x["risk_score"])
        return out([_t(f"{h['name']} has the lowest review priority among monitored locations: "
                       f"{h['risk_tier']} {h['risk_score']:.2f}.")], SRC_HOTSPOTS, used_site=h)

    # 6. one location (named, or the selected context for "this")
    if target is None and _has(q, "recommend", "review", "action", "should", "authorit"):
        top = [h for h in sites if h["risk_tier"] in ("Critical", "High")] or sites[:3]
        return out([_t("Based on the current review priorities, authorities should start with these monitored "
                       "locations (select one for its full evidence):"),
                    _b([f"{h['name']} ({h['risk_tier']} {h['risk_score']:.2f}): "
                        + ((h.get("brief") or {}).get("review") or ["Routine review"])[0] for h in top]),
                    _t("These are suggestions for human road-safety review, not automated decisions.")], SRC_HOTSPOTS)
    if target is None:
        if _has(q, "this", "here", "selected", "hotspot", "location", "junction", "site"):
            return out([_t("Select a monitored hotspot to ask location-specific questions (search the map, click a "
                           "marker or a ranked card, or use Ask Verge About This in View Analysis).")], SRC_HOTSPOTS)
        return out([_t("I can answer questions about Verge's monitored hotspots, traffic-video analysis, risk "
                       "evidence, recommendations, and system methodology. For example:"), _b(SUGGESTIONS[:5])], SRC_SCOPE)
    # a selected hotspot only answers questions that are actually about a location
    if not named and not _has(q, "this", "here", " it", "hotspot", "location", "junction", "site", "risk", "why", "critical",
                              "high", "priority", "score", "crash", "fatal", "injur", "evidence", "flag", "review",
                              "recommend", "action", "should", "where", "summar", "explain", "status", "data", "danger",
                              "safe", "stat", "what about", "how about"):
        return out([_t("I can answer questions about Verge's monitored hotspots, traffic-video analysis, risk "
                       "evidence, recommendations, and system methodology."), _b(SUGGESTIONS[:4])], SRC_SCOPE)
    if _has(q, "where is", "where's", "coordinates", "located", "location of", "map position"):
        return out([_t(f"{target['name']} is at {target['lat']:.5f}, {target['lon']:.5f} (rank #{target['rank']}, "
                       f"{target['risk_tier']} {target['risk_score']:.2f}).")], SRC_HOTSPOTS, used_site=target)
    if _has(q, "recommend", "review", "action", "should", "what to do", "improve", "fix", "measure"):
        return out(_site_review(target), SRC_HOTSPOTS, used_site=target)
    if _has(q, "crash", "fatal", "injur", "evidence", "death", "record", "statistic"):
        return out(_site_evidence(target), SRC_HOTSPOTS, used_site=target)
    return out(_site_why(target), SRC_HOTSPOTS, used_site=target)
