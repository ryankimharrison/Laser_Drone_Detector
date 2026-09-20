"""Is optical flow a better velocity source than differencing box centres?

    python tools/flow_velocity_study.py                 # both runs
    python tools/flow_velocity_study.py --run <dir> --limit 400

STAGED; belongs at tools/. Offline, read-only, no hardware.

THE QUESTION
------------
`FEEDFORWARD_GAIN` is 0.0 because the feedforward flipped sign at +/-150-300
px/s frame to frame on a target moving ~175 px/s. The velocity feeding it comes
from differencing box centres, and a box centre moves when the BOX changes shape
-- an arm occludes a rotor, the detector's idea of the extent shifts a few
pixels -- with no motion at all. That is the noise. Optical flow measures the
motion of the TEXTURE inside the box, which does not care where the detector
thinks the edges are.

This scores four velocity sources on the only thing the feedforward needs them
for: predicting where the target will be one loop-latency from now.

  naive        v = 0. The honest baseline, and the one to beat -- a previous
               study found the Kalman velocity losing to it at every horizon.
  boxdiff      v = (centre[t] - centre[t-1]) / dt. What the system uses now.
  filter       est.du / est.dv straight from the log.
  flow         Lucas-Kanade on features inside the box, previous raw frame to
               this one.

Scored as: predict centre(t + h) = centre(t) + v * h, compare against the
detector's actual box centre at t + h. Error in pixels. Lower is better.

RAW FRAMES, NOT THE ANNOTATED ONES
----------------------------------
`frames/` has the green box and the aim cross burned in; tracking those would
measure the overlay, and the overlay moves with the ESTIMATE, which is the very
thing under test. `raw/` carries one unannotated frame per control row.

DEGENERATE FLOW IS REJECTED, NOT AVERAGED
-----------------------------------------
A blurred drone against a flat wall gives few corners and LK will happily return
a confident answer built from nothing. Three guards, and a frame that fails any
of them is reported as NO MEASUREMENT rather than folded in as a zero:

  * at least MIN_POINTS features survive the forward-backward check
  * forward-backward reprojection error under FB_MAX_PX
  * the surviving vectors agree: median absolute deviation under COHERENCE_PX

The availability rate is a headline result in its own right. A velocity source
that is right when it answers and silent a third of the time is not usable as a
feedforward without a fallback, and that has to be known before any gain is
raised off zero.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import cv2
import numpy as np

FLIGHT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "diag", "flight")
RUNS = ("run_2026-09-20_144709", "run_2026-09-20_145109")

HORIZONS_MS = (33, 66, 100, 200)
MIN_POINTS = 6
FB_MAX_PX = 1.0
COHERENCE_PX = 3.0
#: A pair of frames further apart than this is a gap, not a frame interval.
MAX_DT_S = 0.25

LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def load(run):
    path = run if os.path.isdir(run) else os.path.join(FLIGHT, run)
    with open(os.path.join(path, "control.jsonl"), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh], path


def centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def flow_in_box(prev_gray, gray, box):
    """Median LK flow of features inside `box`, or None if degenerate."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 255
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=120, qualityLevel=0.01,
                                 minDistance=4, mask=mask, blockSize=7)
    if p0 is None or len(p0) < MIN_POINTS:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK)
    if p1 is None:
        return None
    # Forward-backward: track them home again and keep only the ones that agree.
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK)
    if p0r is None:
        return None
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    if good.sum() < MIN_POINTS:
        return None
    a = p0.reshape(-1, 2)[good]
    b = p1.reshape(-1, 2)[good]
    r = p0r.reshape(-1, 2)[good]
    fb = np.linalg.norm(a - r, axis=1)
    keep = fb < FB_MAX_PX
    if keep.sum() < MIN_POINTS:
        return None
    d = b[keep] - a[keep]
    med = np.median(d, axis=0)
    # Coherence: do the survivors agree with each other, or is this a spray?
    mad = float(np.median(np.linalg.norm(d - med, axis=1)))
    if mad > COHERENCE_PX:
        return None
    return float(med[0]), float(med[1]), int(keep.sum()), mad


def build(rows, path, limit=None):
    """Per-row velocity estimates. Skips duplicate frames by box_t."""
    out = []
    prev_gray = None
    prev_c = None
    prev_t = None
    seen_box_t = None
    n = 0
    for r in rows:
        e = r.get("est") or {}
        box = e.get("box")
        bt = e.get("box_t")
        t = r.get("frame_t")
        if not box or bt is None or t is None:
            prev_gray = None          # a gap breaks the flow chain
            continue
        if bt == seen_box_t:
            continue                  # same detection re-stamped; not a new sample
        seen_box_t = bt
        p = os.path.join(path, "raw", "%06d.jpg" % r["frame_index"])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            prev_gray = None
            continue
        c = centre(box)
        rec = {"t": t, "frame": r["frame_index"], "c": c,
               "filter": (e.get("du"), e.get("dv")),
               "boxdiff": None, "flow": None, "flow_pts": 0}
        if prev_gray is not None and prev_t is not None and 0 < t - prev_t <= MAX_DT_S:
            dt = t - prev_t
            rec["boxdiff"] = ((c[0] - prev_c[0]) / dt, (c[1] - prev_c[1]) / dt)
            f = flow_in_box(prev_gray, img, box)
            if f is not None:
                rec["flow"] = (f[0] / dt, f[1] / dt)
                rec["flow_pts"] = f[2]
        out.append(rec)
        prev_gray, prev_c, prev_t = img, c, t
        n += 1
        if limit and n >= limit:
            break
    return out


def score(recs):
    """Prediction error per source per horizon, against the later real box."""
    times = [r["t"] for r in recs]
    res = {h: {k: [] for k in ("naive", "boxdiff", "filter", "flow")}
           for h in HORIZONS_MS}
    import bisect
    for i, r in enumerate(recs):
        for hms in HORIZONS_MS:
            h = hms / 1000.0
            j = bisect.bisect_left(times, r["t"] + h)
            if j >= len(recs):
                continue
            # Nearest sample to the target time, and only if it is close.
            best = j if abs(times[j] - (r["t"] + h)) <= 0.020 else None
            if best is None and j > 0 and abs(times[j - 1] - (r["t"] + h)) <= 0.020:
                best = j - 1
            if best is None:
                continue
            truth = recs[best]["c"]
            for k in ("naive", "boxdiff", "filter", "flow"):
                if k == "naive":
                    v = (0.0, 0.0)
                else:
                    v = r.get(k)
                    if v is None or v[0] is None or v[1] is None:
                        continue
                px = r["c"][0] + v[0] * h
                py = r["c"][1] + v[1] * h
                res[hms][k].append(math.hypot(px - truth[0], py - truth[1]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    for run in (a.run or list(RUNS)):
        rows, path = load(run)
        recs = build(rows, path, a.limit)
        avail = sum(1 for r in recs if r["flow"] is not None)
        pairs = sum(1 for r in recs if r["boxdiff"] is not None)
        print("\n=== %s ===" % os.path.basename(path))
        print("  samples with a box and a distinct detection: %d" % len(recs))
        print("  consecutive pairs available:                 %d" % pairs)
        print("  OPTICAL FLOW available:                      %d (%.0f%% of pairs)"
              % (avail, 100.0 * avail / max(pairs, 1)))
        pts = [r["flow_pts"] for r in recs if r["flow"] is not None]
        if pts:
            print("  surviving LK points: median %d, min %d" % (statistics.median(pts), min(pts)))
        res = score(recs)
        print("\n  median prediction error, px (n in brackets)")
        print("  %-9s %-16s %-16s %-16s %-16s" % ("horizon", "naive", "boxdiff", "filter", "flow"))
        for hms in HORIZONS_MS:
            cells = []
            for k in ("naive", "boxdiff", "filter", "flow"):
                v = res[hms][k]
                cells.append("%7.1f (%4d)" % (statistics.median(v), len(v)) if v else "      -     ")
            print("  %-9s %-16s %-16s %-16s %-16s" % ("%d ms" % hms, *cells))
        print("\n  p90 prediction error, px")
        for hms in HORIZONS_MS:
            cells = []
            for k in ("naive", "boxdiff", "filter", "flow"):
                v = sorted(res[hms][k])
                cells.append("%7.1f      " % v[int(0.9 * (len(v) - 1))] if len(v) > 9 else "      -     ")
            print("  %-9s %-16s %-16s %-16s %-16s" % ("%d ms" % hms, *cells))
        # Frame-to-frame jitter: the thing that made the feedforward flip sign.
        print("\n  frame-to-frame velocity JUMP, px/s (median |v[i] - v[i-1]|)")
        for k in ("boxdiff", "filter", "flow"):
            j = []
            prev = None
            for r in recs:
                v = r.get(k)
                if v is None or v[0] is None:
                    prev = None
                    continue
                if prev is not None:
                    j.append(math.hypot(v[0] - prev[0], v[1] - prev[1]))
                prev = v
            if j:
                print("    %-8s median %6.1f   p90 %6.1f   n=%d"
                      % (k, statistics.median(j), sorted(j)[int(0.9 * (len(j) - 1))], len(j)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
