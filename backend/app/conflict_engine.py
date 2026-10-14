"""
Conflict engine for object-detector tracks (real footage).

conflict.py flags a pair the first time it comes within 8 m on screen with a
low TTC. On real footage that is not enough evidence, and a metric 3-D
reconstruction from one uncalibrated camera turned out to be unreliable
across camera types (overhead CCTV vs. street level). This engine therefore
measures the conflict in the geometry each camera view supports:

0. View geometry. A perspective model is fitted per clip from the vehicles
   themselves (_ground_model): on a flat road a road user's scale grows
   linearly with the image row of its ground contact, s(y) = a (y - y_h).
   - Near-overhead view (horizon far above the frame): footprints are
     measured on screen (below); a car/bus/truck box is mostly footprint.
   - Oblique (street-level) view: every ground point is mapped to road-plane
     metres, Z = f/s, X = (u - cx)/s (_bird_eye), and each road user gets a
     vehicle-sized footprint oriented along its direction of travel, so cars a
     lane apart or metres apart in depth never "touch" just on screen.
   - Camera travelling/zooming far more than a traffic camera: not assessed.
   Motion is never estimated across a jump no road user can make (scene cut
   or tracker ID switch, _segment_starts).
1. Footprints. On screen, a road user's footprint is the bottom strip of its
   box (where it meets the road). Two road users at different distances from
   the camera have footprints on different image rows, so they cannot
   "touch" merely because their boxes overlap on screen (occlusion).
2. Time-to-collision. Each footprint is extrapolated along the road user's
   recent motion (velocity from a 0.8 s regression). TTC is the time until
   the two footprints would touch.
3. Scale. Metres come from each road user's own size (box width or height
   divided by the class's typical width or height, whichever is not inflated
   by the vehicle's length) or from the ground model. Distances and speeds
   are approximate.
4. A sample is a CANDIDATE only if: meaningful pair (never two pedestrians),
   both tracks established and large enough on screen, speeds plausible,
   footprints on a collision course with TTC <= CONFLICT_TTC_S (0.8 s for
   two vehicles), clearance <= CONFLICT_MAX_GAP_M, and, for two road users
   heading the same way, a hard approach (>= 3 m/s), not normal following.
5. A CONFIRMED conflict needs the candidate to persist, the pair to have been
   seen approaching from a separated state, and both tracks to be physically
   consistent (no tracker jumps). One interaction = one event with a start,
   peak and end.
"""
from __future__ import annotations

import bisect
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config

# Typical real-world size (m) of what the detector's box covers, used only to
# convert pixels to metres for display and for the plausibility/following rules.
TYPICAL_HEIGHT_M = {
    "person": 1.7, "two-wheeler": 1.7, "bicycle": 1.05, "motorcycle": 1.15,
    "car": 1.5, "bus": 3.1, "truck": 3.0, "vehicle": 1.5,
}
TYPICAL_WIDTH_M = {
    "person": 0.55, "two-wheeler": 0.7, "bicycle": 0.6, "motorcycle": 0.75,
    "car": 1.8, "bus": 2.5, "truck": 2.5, "vehicle": 1.8,
}
RIDER_SPEED_MS = 5.0           # sustained "pedestrian" speed above this = rider
VEL_WINDOW_S = 0.8
MAX_PLAUSIBLE_SPEED_MS = 20.0  # 72 km/h: faster estimates in city traffic are measurement error
FOOTPRINT_FRAC = 0.25          # footprint strip = bottom 25% of the box height
SAME_DEPTH_RATIO = 1.6         # the two road users' own scales must agree within this
TTC_HORIZON_S = 3.0
JUMP_M, JUMP_SPEED_MS = 1.5, 30.0  # a single step longer and faster than this = a different object
BRIDGE_MAX_S = 0.6             # bird's-eye: longest overlap bridged between approach and separation
BRIDGE_MIN_CLOSING_MS = 8.0    # ... and only after a fast approach (gap closing at >= 8 m/s)
CRASH_MIN_STEP_IOU = 0.3       # consecutive boxes of one object overlap at least this much
CRASH_BEFORE_S = 0.6           # impact signature: motion fitted over this long before contact ...
CRASH_AFTER_WINDOW = (0.2, 0.9)  # ... against the motion fitted over this window after contact
JUMP_NOISE_PX = 3.0                # bird's-eye view: box jitter (px) that is measurement noise, not a jump
FOUR_WHEELERS = {"car", "bus", "truck", "vehicle"}
GROUND_MIN_K = 0.2             # cap on the perspective correction (5x) for noisy fits
OVERHEAD_HORIZON_FRAC = 0.2    # horizon this far (x frame height) above the top = near-overhead view


def _truncated(r: dict, fw: int, fh: int) -> bool:
    return r["x"] <= 1 or r["y"] <= 1 or r["x"] + r["w"] >= fw - 1 or r["y"] + r["h"] >= fh - 1


def _cut_vertically(r: dict, fh: int) -> bool:
    """Box cut off at the top or bottom of the frame: its ground-contact point
    (bottom edge) or its height is unknown, so it can't be measured."""
    return r["y"] <= 1 or r["y"] + r["h"] >= fh - 1


def _label(r: dict) -> str:
    return r.get("label") or ("person" if r["cls"] == "pedestrian" else "vehicle")


def _height_m(r: dict) -> float:
    return TYPICAL_HEIGHT_M.get(_label(r), 1.5)


def _scale_ppm(r: dict, side_cut: bool) -> float:
    """Pixels per metre from the road user's own size. The smaller of the two
    estimates is used: length inflates box height in overhead views and box
    width in side views, never both. A side-cut box only has a valid height."""
    lab = _label(r)
    by_h = r["h"] / TYPICAL_HEIGHT_M.get(lab, 1.5)
    if side_cut:
        return by_h
    by_w = r["w"] / TYPICAL_WIDTH_M.get(lab, 1.8)
    return max(1e-3, min(by_h, by_w))


def _smooth_seg(vals: List[float], seg: List[int], k: int = 3) -> List[float]:
    """_smooth within each segment (never across a jump)."""
    out, i = [], 0
    while i < len(vals):
        j = i
        while j < len(vals) and seg[j] == seg[i]:
            j += 1
        out += _smooth(vals[i:j], k)
        i = j
    return out


def _smooth(vals: List[float], k: int = 3) -> List[float]:
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - k // 2), min(len(vals), i + k // 2 + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def _running_median(vals: List[float], k: int = 7) -> List[float]:
    return [float(np.median(vals[max(0, i - k // 2): i + k // 2 + 1])) for i in range(len(vals))]


def _stabilize(r: dict, u: float, v: float) -> Tuple[float, float]:
    """Map an image point into the first frame's coordinates when a
    camera-motion transform is available (detection_yolo.CameraMotion)."""
    M = r.get("to_ref")
    if M:
        return M[0][0] * u + M[0][1] * v + M[0][2], M[1][0] * u + M[1][1] * v + M[1][2]
    return u, v


def stabilized_ground_point(r: dict) -> Tuple[float, float]:
    return _stabilize(r, r["x"] + r["w"] / 2, r["y"] + r["h"])


# ---- per-track kinematics (image plane, stabilized) ---------------------------

def _smooth_rows(rows: List[dict], max_gap_s: float = 0.5) -> List[dict]:
    """Trajectory smoothing: (1) bridge short detection gaps (<= max_gap_s) by
    linear interpolation, (2) running median of each box edge over +-2
    samples, which removes single-frame flicker (e.g. a truck detected as
    just its cab for one frame) without shifting the real motion."""
    rows = sorted(rows, key=lambda r: r["frame"])
    if len(rows) < 3:
        return rows
    ts = [r["t"] for r in rows]
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.1
    filled = [rows[0]]
    for r0, r1 in zip(rows, rows[1:]):
        gap = r1["t"] - r0["t"]
        n_missing = int(round(gap / step)) - 1 if step > 0 else 0
        if 0 < n_missing and gap <= max_gap_s + 1e-6:
            for k in range(1, n_missing + 1):
                q = k / (n_missing + 1)
                r = dict(r0)
                for key in ("x", "y", "w", "h", "t"):
                    r[key] = r0[key] + (r1[key] - r0[key]) * q
                r["frame"] = int(round(r0["frame"] + (r1["frame"] - r0["frame"]) * q))
                r["interpolated"] = True
                filled.append(r)
        filled.append(r1)
    edges = np.array([[r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]] for r in filled], float)
    out = []
    for i, r in enumerate(filled):
        lo, hi = max(0, i - 2), min(len(filled), i + 3)
        x0, y0, x1, y1 = np.median(edges[lo:hi], axis=0)
        s = dict(r)
        s.update(x=float(x0), y=float(y0), w=float(max(1.0, x1 - x0)), h=float(max(1.0, y1 - y0)))
        out.append(s)
    return out


def _segment_starts(t: List[float], steps_m: List[float], tol_m: Optional[List[float]] = None) -> List[int]:
    """Index of the segment start for every sample. A step no road user can
    make (a scene cut or an ID switch in the tracker) starts a new segment:
    motion is never estimated across it. tol_m is the measurement uncertainty
    of each step (bird's-eye view: far from the camera a few pixels of box
    jitter already span metres); a step must exceed it to count as a jump."""
    seg, out = 0, []
    for i in range(len(t)):
        tol = tol_m[i] if tol_m else 0.0
        dt = max(1e-6, t[i] - t[i - 1]) if i else 1.0
        if i > 0 and steps_m[i] > JUMP_M + tol and (steps_m[i] - tol) / dt > JUMP_SPEED_MS:
            seg = i
        out.append(seg)
    return out


def _regress_vel(t, xs, ys, seg):
    vel = []
    for i in range(len(t)):
        j = max(seg[i], bisect.bisect_left(t, t[i] - VEL_WINDOW_S))
        if i - j >= 2:
            tt = np.array(t[j:i + 1]) - t[i]
            vel.append((float(np.polyfit(tt, xs[j:i + 1], 1)[0]), float(np.polyfit(tt, ys[j:i + 1], 1)[0])))
        else:
            vel.append(None)
    return vel


def _track_kinematics(rows: List[dict], fw: int, fh: int, footprint_frac: float = FOOTPRINT_FRAC,
                      gm=None) -> dict:
    rows = _smooth_rows(rows)
    side_cut = [r["x"] <= 1 or r["x"] + r["w"] >= fw - 1 for r in rows]
    scales = [_scale_ppm(r, sc) for r, sc in zip(rows, side_cut)]
    if gm is not None and gm[3]:
        # Overhead view: a box cut off at the frame side has no usable size (its
        # height is the vehicle's LENGTH), so its scale comes from the clip's
        # ground model at its row instead.
        a, y_h = gm[0], gm[1]
        scales = [max(1e-3, a * (r["y"] + r["h"] - y_h)) if sc else v
                  for r, sc, v in zip(rows, side_cut, scales)]
    ppm = _running_median(scales)
    # footprint rectangles in stabilized image coordinates
    fps_ = []
    for r in rows:
        dx, dy = (a - b for a, b in zip(stabilized_ground_point(r), (r["x"] + r["w"] / 2, r["y"] + r["h"])))
        fps_.append([r["x"] + dx, r["y"] + r["h"] * (1 - footprint_frac) + dy, r["x"] + r["w"] + dx, r["y"] + r["h"] + dy])
    # Reference point for motion: a side-cut road user's INNER visible edge is a
    # real point of the vehicle (its centre isn't while it is cut off).
    cut_l = any(r["x"] <= 1 for r in rows)
    cut_r = any(r["x"] + r["w"] >= fw - 1 for r in rows)
    ref_x = [f[2] if (cut_l and not cut_r) else f[0] if (cut_r and not cut_l) else (f[0] + f[2]) / 2 for f in fps_]
    ref_y = [f[3] for f in fps_]
    t = [r["t"] for r in rows]
    steps = [0.0] + [math.hypot(ref_x[i] - ref_x[i - 1], ref_y[i] - ref_y[i - 1]) / ppm[i] for i in range(1, len(rows))]
    seg = _segment_starts(t, steps)
    sx, sy = _smooth_seg(ref_x, seg), _smooth_seg(ref_y, seg)
    vel = _regress_vel(t, sx, sy, seg)  # px/s of the reference point
    raw_step = [0.0] + [steps[i] / max(1e-6, t[i] - t[i - 1]) for i in range(1, len(rows))]
    h_step = [1.0] + [rows[i]["h"] / max(1, rows[i - 1]["h"]) for i in range(1, len(rows))]
    return {
        "frames": [r["frame"] for r in rows],
        "t": t,
        "rows": rows,
        "fp": fps_,
        "ppm": ppm,
        "vel": vel,
        "speed": [math.hypot(*v) / p if v else 0.0 for v, p in zip(vel, ppm)],
        "raw_step_speed": raw_step,
        "h_step": h_step,
        "ok": [not _cut_vertically(r, fh) and not (cut_l and cut_r and sc) for r, sc in zip(rows, side_cut)],
        "side_cut": side_cut,
        "pos": list(zip(sx, sy)),
        "ref": list(zip(ref_x, ref_y)),
        "ref_side": "right" if (cut_l and not cut_r) else "left" if (cut_r and not cut_l) else "centre",
        "scale_px": ppm,
        "seg_t0": [t[k] for k in seg],
    }


# Typical ground footprint (length, width) in metres, for the bird's-eye view.
FOOTPRINT_LW_M = {
    "car": (4.5, 1.8), "vehicle": (4.5, 1.8), "bus": (12.0, 2.5), "truck": (8.0, 2.5),
    "motorcycle": (2.0, 0.7), "bicycle": (1.8, 0.6), "two-wheeler": (2.0, 0.7), "person": (0.5, 0.5),
}


def _to_ground(gm, u: float, v: float, cx: float) -> Optional[Tuple[float, float]]:
    """Inverse perspective mapping of an image point on the road to road-plane
    metres (X across, Z away from the camera), from the clip's ground model:
    the scale at row v is s = a (v - y_h) px/m, so Z = f / s and X = (u - cx) / s."""
    a, y_h, f, _ = gm
    sc = a * (v - y_h)
    if v - y_h < 0.05 * f or sc <= 1e-6:
        return None
    return (u - cx) / sc, f / sc


def _bird_eye(tr: dict, gm, fw: int) -> bool:
    """Re-express a track's footprint and motion in the road plane (metres):
    footprint = a vehicle-sized rectangle placed at the ground-contact point
    and oriented along the direction of travel. Samples too close to the
    horizon to place on the road (a road user still far away) are marked not
    measurable; the rest of the track is kept. Returns False only when no
    sample can be mapped."""
    cx = fw / 2
    mappable = [_to_ground(gm, u, v, cx) is not None for u, v in tr["ref"]]
    if not any(mappable):
        return False
    # far samples are clamped to the nearest mappable row (continuity only:
    # they are never measured, and motion is not estimated across them)
    v_min = gm[1] + 0.05 * gm[2] + 1.0
    pts = [_to_ground(gm, u, max(v, v_min), cx) for u, v in tr["ref"]]
    t = tr["t"]
    steps = [0.0] + [math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(t))]
    # depth uncertainty of JUMP_NOISE_PX pixels of row jitter at each sample:
    # Z = f/s with s = a (v - y_h), so dZ/dv = Z / (v - y_h)
    tol = [JUMP_NOISE_PX * pts[i][1] / max(1.0, max(v, v_min) - gm[1]) for i, (_, v) in enumerate(tr["ref"])]
    seg = _segment_starts(t, steps, tol)
    tr["depth_tol"] = tol
    cur = 0
    for i in range(len(t)):  # a new segment starts where the road user becomes measurable
        if i > 0 and mappable[i] and not mappable[i - 1]:
            cur = i
        cur = max(cur, seg[i])
        seg[i] = cur
    tr["ok"] = [o and m for o, m in zip(tr["ok"], mappable)]
    gx, gz = _smooth_seg([p_[0] for p_ in pts], seg), _smooth_seg([p_[1] for p_ in pts], seg)
    vel = _regress_vel(t, gx, gz, seg)
    L, W = FOOTPRINT_LW_M.get(tr["label"], (4.5, 1.8))
    # initial heading from the box shape: a box much wider than the road user
    # is wide is seen from the side, i.e. travelling across the image
    r0 = tr["rows"][0]
    across = r0["w"] / max(1e-6, tr["scale_px"][0]) > (L + W) / 2
    head = (1.0, 0.0) if across else (0.0, 1.0)
    fp = []
    for i, (x, z) in enumerate(zip(gx, gz)):
        if i > 0 and seg[i] == i:  # new segment: heading unknown again
            head = (1.0, 0.0) if across else (0.0, 1.0)
        if vel[i] and math.hypot(*vel[i]) >= 1.0:
            n = math.hypot(*vel[i])
            head = (vel[i][0] / n, vel[i][1] / n)
        ex = L * abs(head[0]) + W * abs(head[1])
        ez = L * abs(head[1]) + W * abs(head[0])
        x0 = x - ex if tr["ref_side"] == "right" else x if tr["ref_side"] == "left" else x - ex / 2
        fp.append([x0, z, x0 + ex, z + ez])  # the visible bottom edge is the near side
    # step speed beyond the depth measurement noise of each step
    raw = [0.0] + [max(0.0, steps[i] - tol[i]) / max(1e-6, t[i] - t[i - 1]) for i in range(1, len(t))]
    tr.update(fp=fp, vel=vel, pos=list(zip(gx, gz)), ppm=[1.0] * len(t), seg_t0=[t[k] for k in seg],
              speed=[math.hypot(*v_) if v_ else 0.0 for v_ in vel], raw_step_speed=raw)
    return True


def _reclassify_riders(tracks: Dict[int, dict]) -> List[int]:
    """A 'pedestrian' track moving at sustained vehicle speed is a rider whose
    two-wheeler was not detected (typically occluded in dense traffic)."""
    ids = []
    for tid, tr in tracks.items():
        if tr["cls"] != "pedestrian" or tr["t"][-1] - tr["t"][0] < 1.0:
            continue
        sp = [s for s, v in zip(tr["speed"], tr["vel"]) if v]
        if sp and float(np.median(sp)) > RIDER_SPEED_MS:
            tr["cls"], tr["label"] = "vehicle", "two-wheeler"
            ids.append(tid)
    return ids


def _rect_gap(a, b) -> float:
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return math.hypot(dx, dy)


def _ground_model(tracks: Dict[int, dict], fw: int, fh: int):
    """Perspective ground model, fitted per clip from the vehicles themselves.

    On a flat road seen by a pinhole camera, a road user's scale (px per metre)
    grows linearly with the image row of its ground contact:
    s(y) = a * (y - y_h), where y_h is the horizon row. Distances ALONG the
    road (image rows) are then compressed relative to distances ACROSS it by
    k(y) = (y - y_h) / f (f = focal length in px, assumed ~ image width, i.e.
    a normal lens; the long image side is used so portrait phone video works). k -> 1 for a top-down view (no correction); in a
    street-level view k is small, so two cars that touch on screen can be
    several metres apart in depth. Returns (a, y_h, f, overhead) or None;
    near-overhead views get no correction (see _is_overhead)."""
    ys, ss = [], []
    for tr in tracks.values():
        if tr["cls"] != "vehicle":
            continue
        for f_, p, sc in zip(tr["fp"], tr["ppm"], tr["side_cut"]):
            if sc:
                continue
            ys.append(f_[3])
            ss.append(p)
    if len(ys) < 30:
        return None
    ys, ss = np.array(ys), np.array(ss)
    # needs vehicles spread over the image rows and scale clearly tied to the
    # row; otherwise (a single lane, a top-down view) no correction is applied
    if np.ptp(np.quantile(ys, [0.1, 0.9])) < 0.05 * fh or np.corrcoef(ys, ss)[0, 1] < 0.5:
        return None
    a, b = np.polyfit(ys, ss, 1)
    if a <= 1e-6:
        return None
    y_h = float(-b / a)
    return float(a), y_h, float(max(fw, fh)), y_h <= -OVERHEAD_HORIZON_FRAC * fh


def _is_overhead(gm, fh: int) -> bool:
    """Near-overhead view: the fitted horizon lies well above the top of the
    frame (camera looking steeply down). Footprints on screen are then close to
    their true ground layout, and the level-camera correction does not apply,
    so none is made."""
    return gm is not None and gm[1] <= -OVERHEAD_HORIZON_FRAC * fh


def _foreshortening(gm, y: float) -> float:
    if gm is None or gm[3]:
        return 1.0
    _, y_h, f, _ = gm
    return float(min(1.0, max(GROUND_MIN_K, (y - y_h) / f)))


def _gap_m(a, b, ppm: float, k: float) -> float:
    """Ground gap (m) between two footprint rectangles; the along-road (vertical)
    component is corrected for perspective foreshortening k."""
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return math.hypot(dx, dy / k) / ppm


def _time_to_contact(a, b, w) -> Optional[float]:
    """Earliest t in [0, TTC_HORIZON_S] at which rectangle b, moving with
    relative velocity w (px/s), overlaps rectangle a. None if never."""
    t_in, t_out = 0.0, TTC_HORIZON_S
    for lo_a, hi_a, lo_b, hi_b, vw in ((a[0], a[2], b[0], b[2], w[0]), (a[1], a[3], b[1], b[3], w[1])):
        # overlap on this axis while lo_b + vw t < hi_a and hi_b + vw t > lo_a
        if abs(vw) < 1e-9:
            if hi_b <= lo_a or lo_b >= hi_a:
                return None
            continue
        t1, t2 = (lo_a - hi_b) / vw, (hi_a - lo_b) / vw
        lo, hi = min(t1, t2), max(t1, t2)
        t_in, t_out = max(t_in, lo), min(t_out, hi)
        if t_in > t_out:
            return None
    return t_in


def _same_direction(va, vb) -> bool:
    sa, sb = math.hypot(*va), math.hypot(*vb)
    if sa < 1e-6 or sb < 1e-6:
        return False
    cos = (va[0] * vb[0] + va[1] * vb[1]) / (sa * sb)
    return cos >= math.cos(math.radians(config.CONFLICT_SAME_DIRECTION_DEG))


def _consistent(tr: dict, t0: float, t1: float, max_speed: float = MAX_PLAUSIBLE_SPEED_MS) -> bool:
    """Physically consistent track between t0 and t1: steady box size and no
    more than one implausible position jump (see config.TRACK_MAX_HEIGHT_STEP)."""
    idx = [i for i, t in enumerate(tr["t"]) if t0 <= t <= t1 and i > 0]
    if len(idx) < 3:
        return False
    lim = config.TRACK_MAX_HEIGHT_STEP
    bad_h = sum(1 for i in idx if not (1 / lim <= tr["h_step"][i] <= lim))
    fast = sum(1 for i in idx if tr["raw_step_speed"][i] > max_speed)
    return bad_h <= max(1, int(0.2 * len(idx))) and fast <= 1


def _evasive_dv(tr: dict, t0: float, t1: float) -> float:
    """Largest change of the velocity vector (m/s) within 0.8 s between t0 and
    t1, i.e. braking and/or swerving. The first 0.5 s of a track is skipped
    (velocity regressions are unreliable right after a track starts)."""
    idx = [i for i, t in enumerate(tr["t"]) if t0 <= t <= t1 and t - tr["seg_t0"][i] >= 0.5 and tr["vel"][i]]
    best = 0.0
    for n, i in enumerate(idx):
        vi, pi = tr["vel"][i], tr["ppm"][i]
        for j in idx[n + 1:]:
            if tr["t"][j] - tr["t"][i] > 0.8:
                break
            vj, pj = tr["vel"][j], tr["ppm"][j]
            dv = math.hypot(vj[0] / pj - vi[0] / pi, vj[1] / pj - vi[1] / pi)
            best = max(best, dv)
    return best


def _severity(gap_m: float, ttc_s: Optional[float]) -> float:
    d = max(0.0, 1 - gap_m / config.CONFLICT_MAX_GAP_M)
    t = max(0.0, 1 - ttc_s / config.CONFLICT_TTC_S) if ttc_s is not None else 0.0
    return round(min(1.0, 0.5 * d + 0.5 * t), 3)


# ---- events ------------------------------------------------------------------------

def find_conflict_events(rows: List[dict], frame_size: Tuple[int, int], site_id: str,
                         debug: bool = False) -> Tuple[List[dict], dict]:
    fw, fh = frame_size
    # relative to the shorter frame side, so portrait phone video is not held to
    # a larger size in pixels than the same scene filmed in landscape
    min_px = config.CONFLICT_MIN_BOX_H_FRAC * (min(fw, fh) if config.CONFLICT_MIN_SIZE_SHORT_SIDE else fh)

    # Every distance here assumes a (nearly) fixed camera. When the camera
    # travels or zooms far more than a traffic camera does (a handheld walk,
    # a drone flight), the gaps can't be measured: no conflicts are reported.
    travel, zoom = 0.0, 0.0
    for r in rows:
        M = r.get("to_ref")
        if M:
            travel = max(travel, math.hypot(M[0][2], M[1][2]))
            zoom = max(zoom, abs(math.hypot(M[0][0], M[1][0]) - 1))
    if travel > config.CONFLICT_MAX_CAMERA_TRAVEL_FRAC * max(fw, fh) or zoom > config.CONFLICT_MAX_CAMERA_ZOOM:
        return [], {
            "ground_model": {"mode": "not measured: camera moves too much"},
            "camera_too_unstable": True,
            "camera_travel_px": round(travel), "camera_zoom_change": round(zoom, 2),
            "riders_reclassified": 0, "reclassified_ids": [],
            "potential_conflicts_not_confirmed": 0, "set_aside_no_clear_approach_or_overlap": 0,
            "set_aside_inconsistent_tracking": 0, "set_aside_never_actually_close": 0,
            "set_aside_no_evasive_action": 0, "pairs_evaluated": 0,
        }

    by_id: Dict[int, List[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    def build(frac, gm_):
        out: Dict[int, dict] = {}
        for tid, rs in by_id.items():
            # people and riders stand upright: only their feet area is on the ground
            k_frac = frac if _label(rs[0]) in FOUR_WHEELERS else FOOTPRINT_FRAC
            k = _track_kinematics(rs, fw, fh, k_frac, gm_)
            k["fp_frac"] = k_frac
            k["cls"] = rs[0]["cls"]
            k["label"] = _label(rs[0])
            k["idx"] = {f: i for i, f in enumerate(k["frames"])}
            out[tid] = k
        return out

    tracks = build(FOOTPRINT_FRAC, None)
    gm = _ground_model(tracks, fw, fh) if config.CONFLICT_GROUND_CORRECTION else None
    # in a near-overhead view footprints touching on screen really are touching
    # on the ground; in an oblique view they may be metres apart in depth
    overhead = _is_overhead(gm, fh)
    bird_eye, unmappable = False, set()
    if overhead:
        # Seen from above, a car/bus/truck box covers its ground footprint (not a
        # standing height), so most of it is used instead of the bottom strip.
        tracks = build(config.CONFLICT_OVERHEAD_FOOTPRINT_FRAC, gm)
    elif gm is not None and config.CONFLICT_BIRD_EYE:
        # Oblique (street-level) view: boxes that touch on screen can be metres
        # apart on the road, so gaps and motion are measured in a bird's-eye view.
        bird_eye = True
        # road users at or beyond the horizon can't be placed on the road plane
        # (they are far too distant to measure anyway)
        unmappable = {tid for tid, tr in tracks.items() if not _bird_eye(tr, gm, fw)}
    riders = _reclassify_riders(tracks)
    allow_contact = (config.CONFLICT_ALLOW_CONTACT_PHASE is True
                     or (config.CONFLICT_ALLOW_CONTACT_PHASE == "overhead" and overhead))
    # A presumed rider's vehicle was never actually seen, so its position and
    # size can't be trusted: it is counted, but not used for conflicts.
    unreliable = set(riders) | unmappable

    present: Dict[int, List[int]] = {}
    for tid, tr in tracks.items():
        for f in tr["frames"]:
            present.setdefault(f, []).append(tid)

    # pair -> samples: (t, frame, gap_m, ttc, candidate?, overlap_heavy)
    series: Dict[Tuple[int, int], List[tuple]] = {}
    for f in sorted(present):
        ids = [i for i in sorted(present[f]) if i not in unreliable]
        for n, a in enumerate(ids):
            ta = tracks[a]
            ia = ta["idx"][f]
            ra = ta["rows"][ia]
            if not ta["ok"][ia] or max(ra["h"], ra["w"]) < min_px:
                continue
            for b in ids[n + 1:]:
                tb = tracks[b]
                if ta["cls"] == "pedestrian" and tb["cls"] == "pedestrian":
                    continue  # people walking together are not a road-safety conflict
                ib = tb["idx"][f]
                rb = tb["rows"][ib]
                if not tb["ok"][ib] or max(rb["h"], rb["w"]) < min_px:
                    continue
                fa, fb = ta["fp"][ia], tb["fp"][ib]
                ppm = 0.5 * (ta["ppm"][ia] + tb["ppm"][ib])
                kf = 1.0 if bird_eye else _foreshortening(gm, 0.5 * (fa[3] + fb[3]))
                gap_m = _gap_m(fa, fb, ppm, kf)
                if gap_m > config.CONFLICT_MAX_GAP_M * 3:
                    continue
                # heavy overlap of the FULL boxes = merged / duplicate detections
                inter = (max(0, min(ra["x"] + ra["w"], rb["x"] + rb["w"]) - max(ra["x"], rb["x"]))
                         * max(0, min(ra["y"] + ra["h"], rb["y"] + rb["h"]) - max(ra["y"], rb["y"])))
                heavy = inter / max(1, min(ra["w"] * ra["h"], rb["w"] * rb["h"])) > 0.7
                ttc, ok, close = None, False, False
                va, vb = ta["vel"][ia], tb["vel"][ib]
                plausible = max(ta["speed"][ia], tb["speed"][ib]) <= MAX_PLAUSIBLE_SPEED_MS
                # independent depth cue: similar apparent scale = similar distance from
                # the camera (not checked for a box cut off at the frame side)
                if ta["side_cut"][ia] or tb["side_cut"][ib]:
                    same_depth = True
                else:
                    ratio = ta["scale_px"][ia] / max(1e-6, tb["scale_px"][ib])
                    same_depth = 1 / SAME_DEPTH_RATIO <= ratio <= SAME_DEPTH_RATIO
                # footprints already overlapping on screen = one road user behind the
                # other (occlusion), not a measurable approach
                separate = gap_m > 0.05
                if va and vb and plausible and same_depth and separate:
                    w = (vb[0] - va[0], vb[1] - va[1])
                    if math.hypot(w[0], w[1] / kf) / ppm >= config.CONFLICT_MIN_REL_SPEED_MS:
                        ttc = _time_to_contact(fa, fb, w)
                    moving = ta["speed"][ia] >= 1.0 and tb["speed"][ib] >= 1.0
                    if ttc is not None and moving and _same_direction(va, vb):
                        # closing speed of the footprint gap (m/s) along the relative motion
                        closing = math.hypot(w[0], w[1] / kf) / ppm
                        if closing < config.CONFLICT_SAME_DIRECTION_MIN_CLOSING_MS:
                            ttc = None  # normal following / queuing, not a hard approach
                    established = (ta["t"][ia] - ta["seg_t0"][ia] >= config.CONFLICT_MIN_TRACK_S
                                   and tb["t"][ib] - tb["seg_t0"][ib] >= config.CONFLICT_MIN_TRACK_S)
                    vv = ta["cls"] == "vehicle" and tb["cls"] == "vehicle"
                    ttc_max = config.CONFLICT_TTC_VV_S if vv else config.CONFLICT_TTC_S
                    ok = (established and ttc is not None and gap_m <= config.CONFLICT_MAX_GAP_M
                          and ttc <= ttc_max)
                    close_lim = config.CONFLICT_CLOSE_PASS_VV_M if vv else config.CONFLICT_CLOSE_PASS_VRU_M
                    close = (not ok and established and 0.0 < gap_m <= close_lim
                             and max(ta["speed"][ia], tb["speed"][ib]) >= config.CONFLICT_CLOSE_PASS_MIN_SPEED_MS
                             and math.hypot(w[0], w[1] / kf) / ppm >= config.CONFLICT_CLOSE_PASS_MIN_REL_MS)
                elif va and vb and plausible and same_depth and 0.0 < gap_m:
                    established = (ta["t"][ia] - ta["seg_t0"][ia] >= config.CONFLICT_MIN_TRACK_S
                                   and tb["t"][ib] - tb["seg_t0"][ib] >= config.CONFLICT_MIN_TRACK_S)
                    vv = ta["cls"] == "vehicle" and tb["cls"] == "vehicle"
                    close_lim = config.CONFLICT_CLOSE_PASS_VV_M if vv else config.CONFLICT_CLOSE_PASS_VRU_M
                    w = (vb[0] - va[0], vb[1] - va[1])
                    close = (established and gap_m <= close_lim
                             and max(ta["speed"][ia], tb["speed"][ib]) >= config.CONFLICT_CLOSE_PASS_MIN_SPEED_MS
                             and math.hypot(w[0], w[1] / kf) / ppm >= config.CONFLICT_CLOSE_PASS_MIN_REL_MS)
                series.setdefault((a, b), []).append(
                    (ta["t"][ia], f, gap_m, ttc, ok or close, heavy, "ttc" if ok else "close" if close else None))

    potential = occlusion = inconsistent = not_close = no_evasive = 0
    rejects: List[tuple] = []  # (pair, run start, run end, reason) - debug only
    events: List[dict] = []
    for (a, b), samples in series.items():
        vv = tracks[a]["cls"] == "vehicle" and tracks[b]["cls"] == "vehicle"
        # the contact phase is only read for two four-wheel vehicles, whose
        # footprints are measured directly in an overhead view
        pair_contact = allow_contact and tracks[a]["label"] in FOUR_WHEELERS and tracks[b]["label"] in FOUR_WHEELERS
        if vv:
            # oblique views need a clearer closing speed: their along-road gaps
            # carry the most depth error
            min_closing = config.CONFLICT_MIN_CLOSING_VV_OVERHEAD_MS if overhead else config.CONFLICT_MIN_CLOSING_VV_MS
        else:
            min_closing = config.CONFLICT_MIN_CLOSING_VRU_MS
        # closing speed of the measured gap itself (m/s), from a short regression
        ts = [s[0] for s in samples]
        gaps = [s[2] for s in samples]
        checked = []
        for k, s_ in enumerate(samples):
            lo = bisect.bisect_left(ts, ts[k] - 0.35)
            if s_[4] and s_[6] == "ttc":
                if k - lo >= 2:
                    closing = -float(np.polyfit(np.array(ts[lo:k + 1]) - ts[k], gaps[lo:k + 1], 1)[0])
                    keep = closing >= min_closing
                else:
                    keep = False
                s_ = s_[:4] + (keep,) + s_[5:6] + ("ttc" if keep else None,)
            checked.append(s_)
        samples = checked
        runs, cur, miss, last_cand_t = [], [], 0, None
        for s in samples:
            if s[4]:
                cur.append(s)
                miss, last_cand_t = 0, s[0]
            elif (pair_contact and s[2] <= 0.05 and not cur and runs and s[0] - last_cand_t <= 0.5
                  and any(x[6] == "ttc" for x in runs[-1])):
                # footprints meet just after an approach run closed (low-speed
                # closing samples are noisy): the contact continues that run
                cur = runs.pop()
                cur.append(s[:4] + (True,) + s[5:6] + ("contact",))
                miss = 0
            elif (pair_contact and cur and s[2] <= 0.05 and any(x[6] == "ttc" for x in cur)
                  and s[0] - last_cand_t <= 1.0):
                # Contact phase: footprints meet right after a measured, fast
                # collision-course approach. On its own an overlap is ambiguous
                # (one road user behind the other); straight after such an
                # approach it means they met or came within the measurement limit.
                cur.append(s[:4] + (True,) + s[5:6] + ("contact",))
                miss = 0
            elif cur:
                miss += 1
                if miss > 1:
                    runs.append(cur)
                    cur, miss = [], 0
        if cur:
            runs.append(cur)
        if bird_eye and tracks[a]["label"] in FOUR_WHEELERS and tracks[b]["label"] in FOUR_WHEELERS:
            runs = _bridge_contact(runs, samples)
        confirmed = []
        min_dur = config.CONFLICT_MIN_DURATION_VV_S if vv else config.CONFLICT_MIN_DURATION_S
        # one sample interval is added: a run of n samples covers n intervals
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.07
        for run in runs:
            dur = run[-1][0] - run[0][0] + dt
            has_ttc = any(s_[6] == "ttc" for s_ in run)
            if len(run) < 2 or dur < (min_dur if has_ttc else max(min_dur, config.CONFLICT_CLOSE_PASS_MIN_S)):
                potential += 1
                rejects.append(((a, b), run[0][0], run[-1][0], "too short"))
                continue
            # the pair must have been seen approaching from a separated state
            before = [s[2] for s in samples if run[0][0] - 1.5 <= s[0] <= run[0][0]]
            run_min = min(s[2] for s in run)
            # occlusion is judged over the approach and the event itself; what the
            # pair does afterwards (e.g. stopping side by side) is not evidence against it
            contact_ts = {x[0] for x in run if x[6] == "contact"}
            win = [s for s in samples if run[0][0] - 0.5 <= s[0] <= run[-1][0] + 0.2 and s[0] not in contact_ts]
            overlapping = sum(1 for s in win if s[2] <= 0.05 or s[5]) / max(1, len(win))
            if (not before or max(before) - run_min < config.CONFLICT_MIN_APPROACH_M
                    or overlapping > config.CONFLICT_MAX_OCCLUDED_FRAC):
                occlusion += 1
                rejects.append(((a, b), run[0][0], run[-1][0], "no approach / overlapping"))
                continue
            observed = min(s[2] for s in samples if run[0][0] - 0.5 <= s[0] <= run[-1][0] + 1.0)
            if observed > (config.CONFLICT_MAX_OBSERVED_GAP_VV_M if vv else config.CONFLICT_MAX_OBSERVED_GAP_VRU_M):
                not_close += 1
                rejects.append(((a, b), run[0][0], run[-1][0], "never close"))
                continue
            # footprints that actually met after a measured approach need no
            # further evidence of an evasive manoeuvre
            reached_contact = any(x[6] == "contact" for x in run)
            if config.CONFLICT_EVASIVE_DV_MS > 0 and not reached_contact:
                ev_dv = max(_evasive_dv(tracks[a], run[0][0] - 1.0, run[-1][0] + 1.0),
                            _evasive_dv(tracks[b], run[0][0] - 1.0, run[-1][0] + 1.0))
                if ev_dv < config.CONFLICT_EVASIVE_DV_MS:
                    no_evasive += 1
                    rejects.append(((a, b), run[0][0], run[-1][0], f"no evasive action ({ev_dv:.1f} m/s)"))
                    continue
            if not (_consistent(tracks[a], run[0][0] - 1.0, run[-1][0] + 1.0)
                    and _consistent(tracks[b], run[0][0] - 1.0, run[-1][0] + 1.0)):
                inconsistent += 1
                rejects.append(((a, b), run[0][0], run[-1][0], "inconsistent tracking"))
                continue
            confirmed.append(run)
        merged: List[List[tuple]] = []
        for run in confirmed:
            if merged and run[0][0] - merged[-1][-1][0] < config.CONFLICT_MERGE_GAP_S:
                merged[-1].extend(run)
            else:
                merged.append(list(run))
        for run in merged:
            events.append(_make_event(site_id, a, b, tracks[a], tracks[b], run, samples))

    crashes = _find_crashes(series, tracks, fw, fh, site_id) if config.CRASH_DETECTION else []
    for c in crashes:
        # a crash supersedes the near-miss event of the same pair and moment
        events = [e for e in events if not ({e["a_id"], e["b_id"]} == {c["a_id"], c["b_id"]}
                                            and e["zone_start_t"] <= c["zone_end_t"] + 1.0
                                            and c["zone_start_t"] <= e["zone_end_t"] + 1.0)]
    events += crashes
    events.sort(key=lambda e: e["peak_t"])
    diag = {
        "crashes_detected": len(crashes),
        "camera_too_unstable": False,
        "ground_model": ({"mode": "image-plane footprints, scale from road-user size",
                          "perspective_correction": False} if gm is None else
                         {"mode": "image-plane footprints, scale from road-user size",
                          "perspective_correction": True,
                          "view": "near-overhead" if overhead else "oblique",
                          "bird_eye": bird_eye,
                          "horizon_row_frac": round(gm[1] / fh, 3),
                          "foreshortening_at_bottom": round(_foreshortening(gm, fh), 2),
                          "foreshortening_at_middle": round(_foreshortening(gm, fh / 2), 2)}),
        "riders_reclassified": len(riders),
        "reclassified_ids": riders,
        "potential_conflicts_not_confirmed": potential,
        "set_aside_no_clear_approach_or_overlap": occlusion,
        "set_aside_inconsistent_tracking": inconsistent,
        "set_aside_never_actually_close": not_close,
        "set_aside_no_evasive_action": no_evasive,
        "pairs_evaluated": len(series),
    }
    if debug:  # per-sample evidence for offline inspection (not used by the app)
        diag["_series"], diag["_tracks"], diag["_rejects"] = series, tracks, rejects
    return events, diag


def _make_event(site_id, a, b, ta, tb, run, all_samples) -> dict:
    # order: vehicle first for vehicle-pedestrian pairs (matches conflict.py)
    if ta["cls"] == "pedestrian" and tb["cls"] == "vehicle":
        a, b, ta, tb = b, a, tb, ta
    kind = "vehicle_pedestrian" if "pedestrian" in (ta["cls"], tb["cls"]) else "vehicle_vehicle"
    ttc_samples = [s for s in run if s[6] == "ttc" and s[3] is not None]
    indicator = "collision_course" if ttc_samples else "close_pass"
    if ttc_samples:
        peak = min(ttc_samples, key=lambda s: (s[3], s[2]))
        sev = _severity(peak[2], peak[3])
    else:
        peak = min(run, key=lambda s: s[2])
        vv = ta["cls"] == "vehicle" and tb["cls"] == "vehicle"
        lim = config.CONFLICT_CLOSE_PASS_VV_M if vv else config.CONFLICT_CLOSE_PASS_VRU_M
        sev = round(0.4 + 0.6 * max(0.0, 1 - peak[2] / lim), 3)
    start, end = run[0], run[-1]
    # closest measured approach during the event (the samples that were
    # separately measurable; overlapping footprints carry no distance)
    nearest = min(run, key=lambda s: s[2])

    ia, ib = ta["idx"][peak[1]], tb["idx"][peak[1]]
    # conflict point in raw image coordinates: where the two ground footprints
    # come closest (or meet), at the moment of the smallest separation
    closest = next((x for x in run if x[6] == "contact"), nearest)
    ja, jb = ta["idx"][closest[1]], tb["idx"][closest[1]]
    cp = _closest_point(ta["rows"][ja], tb["rows"][jb], ta["fp_frac"], tb["fp_frac"])
    return {
        "site_id": site_id,
        "frame": start[1],
        "t": round(start[0], 2),
        "kind": kind,
        "a_id": a, "a_cls": ta["cls"], "a_label": ta["label"],
        "b_id": b, "b_cls": tb["cls"], "b_label": tb["label"],
        "distance_m": round(start[2], 2),
        "ttc_s": round(start[3], 2) if start[3] is not None and indicator == "collision_course" else None,
        "severity": sev,
        "peak_frame": peak[1],
        "peak_t": round(peak[0], 2),
        "peak_severity": sev,
        "min_distance_m": round(nearest[2], 2),
        "min_distance_t": round(nearest[0], 2),
        "min_ttc_s": round(peak[3], 2) if peak[3] is not None and indicator == "collision_course" else None,
        "min_ttc_t": round(peak[0], 2) if indicator == "collision_course" else None,
        "indicator": indicator,
        "event_type": "near_miss",
        "contact_phase": any(x[6] == "contact" for x in run),
        "contact_t": next((round(x[0], 2) for x in run if x[6] == "contact"), None),
        "zone_start_t": round(start[0], 2),
        "zone_end_t": round(end[0], 2),
        "zone_duration_s": round(end[0] - start[0] + (end[0] - start[0]) / max(1, len(run) - 1), 2),
        "closing_observed": True,
        "pet_s": _pet(ta, tb),
        "conflict_point_px": cp,
        "speeds_ms": [round(ta["speed"][ia], 1), round(tb["speed"][ib], 1)],
        "severity_label": config.severity_band(sev),
        "confirmed_samples": len(run),
    }


# ---- bridged contact and crash detection --------------------------------------

def _bridge_contact(runs: List[list], samples: List[tuple]) -> List[list]:
    """Bird's-eye view: join a collision-course run and the separation run that
    follows it when, in between, the two footprints overlapped on the road
    plane (clearance at the measurement limit) for at most BRIDGE_MAX_S. An
    overlap bracketed by a measured approach AND a measured separation is the
    closest moment of a near-miss (or a crash), not one road user hiding
    another; a one-sided overlap is still not used."""
    out: List[list] = []
    for run in runs:
        if out and any(x[6] == "ttc" for x in out[-1]):
            prev = out[-1]
            between = [x for x in samples if prev[-1][0] < x[0] < run[0][0]]
            # only a FAST approach into the overlap (not lanes drifting together)
            last = [x for x in samples if x[0] <= prev[-1][0]][-3:]
            closing = ((last[0][2] - between[0][2]) / max(1e-6, between[0][0] - last[0][0])
                       if last and between else 0.0)
            if (between and run[0][0] - prev[-1][0] <= BRIDGE_MAX_S and closing >= BRIDGE_MIN_CLOSING_MS
                    and all(x[2] <= 0.05 for x in between)):
                out[-1] = prev + [x[:4] + (True,) + x[5:6] + ("contact",) for x in between] + run
                continue
        out.append(run)
    return out


def _box_overlap(ra: dict, rb: dict) -> float:
    """Intersection of two boxes relative to the smaller one."""
    iw = min(ra["x"] + ra["w"], rb["x"] + rb["w"]) - max(ra["x"], rb["x"])
    ih = min(ra["y"] + ra["h"], rb["y"] + rb["h"]) - max(ra["y"], rb["y"])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1.0, min(ra["w"] * ra["h"], rb["w"] * rb["h"]))


def _box_iou(ra: dict, rb: dict) -> float:
    iw = min(ra["x"] + ra["w"], rb["x"] + rb["w"]) - max(ra["x"], rb["x"])
    ih = min(ra["y"] + ra["h"], rb["y"] + rb["h"]) - max(ra["y"], rb["y"])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (ra["w"] * ra["h"] + rb["w"] * rb["h"] - inter)


def _same_object(tr: dict, t0: float, t1: float) -> bool:
    """The track stays on ONE object between t0 and t1: every box overlaps the
    previous one (IoU >= CRASH_MIN_STEP_IOU). A vehicle that spins or rolls
    over keeps overlapping itself; a tracker that hops to another road user
    (in dense traffic) does not."""
    idx = [i for i, t in enumerate(tr["t"]) if t0 <= t <= t1]
    return all(_box_iou(tr["rows"][i - 1], tr["rows"][i]) >= CRASH_MIN_STEP_IOU for i in idx if i > 0)


def _median_vec(tr: dict, t0: float, t1: float):
    """Velocity (m/s) from a straight-line fit of the positions between t0 and
    t1 only, so the estimate on one side of an impact never mixes in motion
    from the other side (the running 0.8 s velocity would lag behind)."""
    idx = [i for i, t in enumerate(tr["t"]) if t0 <= t <= t1 and tr["ok"][i]]
    if len(idx) < 3:
        return None
    tt = np.array([tr["t"][i] for i in idx]) - tr["t"][idx[0]]
    px = [tr["pos"][i][0] for i in idx]
    py = [tr["pos"][i][1] for i in idx]
    scale = float(np.median([tr["ppm"][i] for i in idx]))
    return float(np.polyfit(tt, px, 1)[0]) / scale, float(np.polyfit(tt, py, 1)[0]) / scale


def _median_aspect(tr: dict, t0: float, t1: float, fw: int, fh: int) -> Optional[float]:
    asp = [r["w"] / max(1.0, r["h"]) for t, r, ok in zip(tr["t"], tr["rows"], tr["ok"])
           if t0 <= t <= t1 and ok and not _truncated(r, fw, fh)]
    return float(np.median(asp)) if len(asp) >= 2 else None


def _impact_signature(tr: dict, c0: float, fw: int, fh: int) -> Tuple[float, float, float]:
    """Sustained change across the contact moment c0, robust to single-sample
    noise: median velocity over the CRASH_BEFORE_S before contact vs. the
    median over CRASH_AFTER_WINDOW after it (m/s), and the same for the box's
    aspect ratio (a vehicle that spins or rolls over changes shape and keeps
    it). Returns (velocity change, shape change factor, speed before)."""
    t_a0, t_a1 = c0 + CRASH_AFTER_WINDOW[0], c0 + CRASH_AFTER_WINDOW[1]
    vb = _median_vec(tr, c0 - CRASH_BEFORE_S, c0)
    va = _median_vec(tr, t_a0, t_a1)
    dv = math.hypot(va[0] - vb[0], va[1] - vb[1]) if vb and va else 0.0
    ab = _median_aspect(tr, c0 - CRASH_BEFORE_S, c0, fw, fh)
    aa = _median_aspect(tr, t_a0, t_a1, fw, fh)
    shape = max(ab, aa) / max(1e-6, min(ab, aa)) if ab and aa else 1.0
    return dv, shape, (math.hypot(*vb) if vb else 0.0)


def _find_crashes(series: Dict[Tuple[int, int], List[tuple]], tracks: Dict[int, dict],
                  fw: int, fh: int, site_id: str) -> List[dict]:
    """A CRASH (observed collision) needs all of these, measured on the pair's
    own evidence:
      1. contact: the two ground footprints meet (clearance at the measurement
         limit) for >= 2 consecutive samples, and their boxes overlap on screen;
      2. a fast approach into that contact: the gap closed by
         >= CRASH_MIN_APPROACH_M at >= CRASH_MIN_CLOSING_MS just before;
      3. an impact signature on at least one of them: a SUSTAINED velocity
         change >= CRASH_DV_MS across the contact (median before vs. median
         after, beyond what braking does in that time) or a sustained box-shape
         change >= CRASH_SHAPE_CHANGE (spin or rollover);
      4. at least one of them moving at >= CRASH_MIN_SPEED_MS before contact,
         and the contact close enough to the camera to measure."""
    out = []
    for (a, b), samples in series.items():
        ta, tb = tracks[a], tracks[b]
        if ta["cls"] == "pedestrian" and tb["cls"] == "pedestrian":
            continue
        k = 0
        while k < len(samples):
            if samples[k][2] > 0.05:
                k += 1
                continue
            j = k
            while j + 1 < len(samples) and samples[j + 1][2] <= 0.05 and samples[j + 1][0] - samples[j][0] <= 0.2:
                j += 1
            contact = samples[k:j + 1]
            k = j + 1
            if len(contact) < 2 or contact[-1][0] - contact[0][0] < config.CRASH_MIN_CONTACT_S:
                continue  # a crash keeps the two together; a brief touch on screen is not one
            c0 = contact[0][0]
            overlap = float(np.median([_box_overlap(ta["rows"][ta["idx"][x[1]]], tb["rows"][tb["idx"][x[1]]])
                                       for x in contact]))
            if overlap < config.CRASH_MIN_BOX_OVERLAP:
                continue
            # the approach itself must be cleanly tracked (no ID switch into the
            # contact); what happens after an impact is expected to be violent
            t_pre = c0  # up to and including the first contact sample
            if not (_consistent(ta, c0 - 1.0, t_pre, config.CRASH_MAX_SPEED_MS)
                    and _consistent(tb, c0 - 1.0, t_pre, config.CRASH_MAX_SPEED_MS)):
                continue
            before = [x for x in samples if c0 - 1.0 <= x[0] < c0]
            if len(before) < 2:
                continue
            approach = max(x[2] for x in before) - contact[0][2]
            recent = [x for x in before if x[0] >= c0 - 0.5] or before
            if len(recent) >= 2:
                tt = np.array([x[0] for x in recent] + [c0]) - c0
                closing = -float(np.polyfit(tt, [x[2] for x in recent] + [contact[0][2]], 1)[0])
            else:
                closing = (recent[0][2] - contact[0][2]) / max(1e-6, c0 - recent[0][0])
            if approach < config.CRASH_MIN_APPROACH_M or closing < config.CRASH_MIN_CLOSING_MS:
                continue
            # contact must be where positions are measurable (bird's-eye: depth
            # uncertainty at that image row below CRASH_MAX_DEPTH_TOL_M)
            ia0, ib0 = ta["idx"][contact[0][1]], tb["idx"][contact[0][1]]
            if max(ta.get("depth_tol", [0.0] * (ia0 + 1))[ia0],
                   tb.get("depth_tol", [0.0] * (ib0 + 1))[ib0]) > config.CRASH_MAX_DEPTH_TOL_M:
                continue
            # identity must hold through the impact, or the "impact" is a tracker hop
            if not (_same_object(ta, c0 - 0.5, c0 + CRASH_AFTER_WINDOW[1])
                    and _same_object(tb, c0 - 0.5, c0 + CRASH_AFTER_WINDOW[1])):
                continue
            sig = [_impact_signature(tr, c0, fw, fh) for tr in (ta, tb)]
            if max(x[2] for x in sig) < config.CRASH_MIN_SPEED_MS:
                continue  # nobody was moving at a speed that makes an impact
            if max(x[2] for x in sig) > config.CRASH_MAX_SPEED_MS:
                continue  # implausible measured speed = measurement error, not evidence
            dv = max(x[0] for x in sig)
            shape = max(x[1] for x in sig)
            if dv < config.CRASH_DV_MS and shape < config.CRASH_SHAPE_CHANGE:
                continue
            run = before[-3:] + [x[:4] + (True,) + x[5:6] + ("contact",) for x in contact]
            e = _make_event(site_id, a, b, ta, tb, run, samples)
            ttcs = [x[3] for x in before if x[3] is not None]
            e.update({
                "event_type": "crash",
                "indicator": "collision",
                "t": round(c0, 2),
                "zone_start_t": round(c0, 2),
                "zone_end_t": round(contact[-1][0], 2),
                "zone_duration_s": round(contact[-1][0] - c0, 2),
                "peak_t": round(c0, 2),
                "peak_frame": contact[0][1],
                "frame": contact[0][1],
                "min_distance_m": 0.0,
                "min_distance_t": round(c0, 2),
                "min_ttc_s": round(min(ttcs), 2) if ttcs else None,
                "min_ttc_t": None,
                "ttc_s": None,
                "severity": 1.0,
                "peak_severity": 1.0,
                "severity_label": config.severity_band(1.0),
                "contact_phase": True,
                "contact_t": round(c0, 2),
                "approach_closing_ms": round(closing, 1),
                "impact_dv_ms": round(dv, 1),
                "impact_shape_change": round(shape, 2),
            })
            out.append(e)
            break  # one crash per pair
    return out


def _closest_point(ra: dict, rb: dict, fa: float, fb: float) -> List[float]:
    """Midpoint between the closest points of two footprint strips (the bottom
    fa / fb of each box), in image pixels."""
    def span(lo_a, hi_a, lo_b, hi_b):
        lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
        if lo <= hi:            # overlapping on this axis: middle of the overlap
            return (lo + hi) / 2
        return (hi + lo) / 2    # apart: midway between the facing edges
    x = span(ra["x"], ra["x"] + ra["w"], rb["x"], rb["x"] + rb["w"])
    y = span(ra["y"] + ra["h"] * (1 - fa), ra["y"] + ra["h"], rb["y"] + rb["h"] * (1 - fb), rb["y"] + rb["h"])
    return [round(x, 1), round(y, 1)]


def _pet(ta: dict, tb: dict) -> Optional[float]:
    """PET from the two footprint paths: the closest points in space (each
    taken while BOTH are separately tracked). Reported only when that is
    within PET_RADIUS_M, i.e. the paths genuinely share a point."""
    fa, fb = set(ta["frames"]), set(tb["frames"])
    pa = [(t, p, s) for t, p, s, f in zip(ta["t"], ta["pos"], ta["ppm"], ta["frames"]) if f in fb]
    pb = [(t, p, s) for t, p, s, f in zip(tb["t"], tb["pos"], tb["ppm"], tb["frames"]) if f in fa]
    if not pa or not pb:
        return None
    A = np.array([p for _, p, _ in pa])
    B = np.array([p for _, p, _ in pb])
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    scale = 0.5 * (pa[i][2] + pb[j][2])
    if d[i, j] / scale > config.PET_RADIUS_M:
        return None
    return round(abs(pa[i][0] - pb[j][0]), 2)
