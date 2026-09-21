"""Score a flight run the way the feedforward ramp is being judged.

    python staged/score_run.py diag/flight/run_2026-09-20_203933
    python staged/score_run.py <new_run> --vs diag/flight/run_2026-09-20_203933

STAGED, NOT APPLIED. Offline only -- reads a run folder, opens no port, starts
nothing. Belongs at tools/score_run.py once the ramp is settled.

WHAT IT SCORES, AND WHY THESE AND NOTHING ELSE
----------------------------------------------
The standing objective is the beam on the drone's CENTRE regardless of speed or
range, so every number here is px-from-centre against how fast the platform is
moving. Beam time, burst counts and row counts appear only where they explain
that curve.

The headline is LAG_S PER FEEDFORWARD TIER. The lag law from run_184448 is

    standing error ~= LAG_S * sustained platform rate

with LAG_S 0.49-0.58 s under pure P AS ORIGINALLY PUBLISHED -- but that figure
used a different px-per-degree than this file does, so it is not comparable to
the seconds printed here; see px_per_deg(). Re-scored through this file,
run_184448 is 0.78 s overall and 0.53 s on its flow rows. The feedforward's whole claim is that it
cuts LAG_S, so that is what it is scored on, split by whether the feedforward
actually had a measurement that row -- because on run_2026-09-20_203933 it did
not on 47% of rows, and pooling those with the rest scored the availability
instead of the term. Comparing runs without splitting by tier will do that
again.

THE X AXIS IS GYRO RATE, NOT IMAGE SPEED
----------------------------------------
Image speed is the wrong axis and produced a wrong answer once already: it is
an OUTPUT of the loop, so binning by it conditions on the thing being measured.
The gyro reads body rotation directly and is never fed back. `gyro_at` takes
the MEDIAN over the 0.5 s window ENDING at each row, because a standing error
is a response to a sustained rate, not to one sample.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not fit a Jacobian from the rows (closed-loop identification gave the
wrong SIGN at n=1268 -- see closed-loop-jacobian-is-invalid); it reads the same
calibration/jacobian.json the loop used. It does not derive velocity from box
centres (they are noise). It does not score a run without looking at frames --
`--frames` renders the worst rows so the standing rule is cheap to follow.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics as st
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AXIS_STEP_DEG = 0.04875
#: Bins wide enough that each holds tens of rows on a 60-90 s run, and narrow
#: enough that the error is roughly linear inside one.
GYRO_BINS = [(0, 10), (10, 20), (20, 35), (35, 55), (55, 90), (90, 200)]
#: A bin with fewer than this contributes to no fit: a median of five rows is
#: not a measurement of anything.
MIN_BIN = 8


def load(run: str):
    rows = [json.loads(l) for l in
            open(os.path.join(run, "control.jsonl"), encoding="utf-8")]
    imu_path = os.path.join(run, "imu.jsonl")
    imu = ([json.loads(l) for l in open(imu_path, encoding="utf-8")]
           if os.path.exists(imu_path) else [])
    return rows, imu


def gyro_series(imu):
    """(times, |omega|) on the control log's own rel_t base."""
    t, m = [], []
    for s in imu:
        g = s.get("gyro_dps")
        if g is None or s.get("rel_t") is None:
            continue
        t.append(float(s["rel_t"]))
        m.append(math.sqrt(sum(float(x) ** 2 for x in g[:3])))
    return t, m


def px_per_deg(run: str):
    """Pixels of image motion per DEGREE OF PAYLOAD ROTATION.

    This is a pure scale factor on every LAG_S below, so it is worth being
    exact about. Three values are defensible-looking and only one is right:

      11.57  mean singular value of J / AXIS_STEP_DEG.  WRONG, and it is what
             an earlier version of this file used. J's singular values are the
             gains of its own eigen-directions in MOTOR space; they do not
             decompose onto the payload's yaw and pitch axes, so dividing by
             AXIS_STEP_DEG does not produce px per payload degree.
      24.43  NARROW_F_PX (1400) x pi/180. The baseline "LAG_S 0.49-0.58 s" for
             run_184448 was computed with this. Physically it is the right KIND
             of quantity -- a body rotation of theta moves a world-fixed point
             by f*theta -- but NARROW_F_PX is a nominal constant with a known
             disagreement against measurement (focal-constants-are-inconsistent),
             and it disagrees with the value below by 48%.
      16.54  THIS ONE. Derived from the measured J through the differential's
             own kinematics: yaw = (a-b)/2 * AXIS_STEP_DEG and
             pitch = -(a+b)/2 * AXIS_STEP_DEG, so |J @ (1,-1)| / AXIS_STEP_DEG
             is px per degree of yaw (15.50) and |J @ (1,1)| / AXIS_STEP_DEG is
             px per degree of pitch (17.58). J was measured by stepping the
             motors and watching features move, so this is the rig answering
             rather than a constant asserting.

    CONSEQUENCE, and it is the important part: LAG_S values computed with
    different choices here are NOT comparable, and mixing them once already
    made a real improvement look like a different-sized one. Compare runs
    through this function or compare RATIOS, which are scale-free.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "turret_host", "calibration", "jacobian.json")
    J = np.array(json.load(open(p, encoding="utf-8"))["J"], float)
    yaw = float(np.linalg.norm(J @ np.array([1.0, -1.0]))) / AXIS_STEP_DEG
    pitch = float(np.linalg.norm(J @ np.array([1.0, 1.0]))) / AXIS_STEP_DEG
    return 0.5 * (yaw + pitch)


def collect(run: str):
    rows, imu = load(run)
    gt, gm = gyro_series(imu)

    def gyro_at(t, win=0.5):
        lo, hi = bisect.bisect_left(gt, t - win), bisect.bisect_right(gt, t)
        return st.median(gm[lo:hi]) if hi - lo >= 3 else None

    out = []
    for r in rows:
        c = r.get("cmd")
        if not c or r.get("track") != "TRACK":
            continue
        g = gyro_at(float(r["rel_t"])) if gt else None
        out.append({
            "rel_t": float(r["rel_t"]), "frame_index": r.get("frame_index"),
            "tier": c.get("ff_source", "?"), "err": float(c["error_px"]),
            "gyro": g,
            "ff": math.hypot(c.get("ff_inertial_u") or 0.0,
                             c.get("ff_inertial_v") or 0.0),
            "rate": max(abs(float(c.get("rate_a") or 0.0)),
                        abs(float(c.get("rate_b") or 0.0))),
            "saturated": bool(c.get("saturated")),
            "box": (r.get("est") or {}).get("box"),
        })
    return rows, out


def fit_lag(sel, ppd):
    """Weighted fit of median error against median gyro, per bin -> LAG_S.

    Fitting BIN MEDIANS rather than raw rows on purpose: the raw cloud is
    heavy-tailed (a few hundred-px excursions) and a least-squares line through
    it follows the tail. It also keeps the x-noise out of the slope -- fitting
    raw rows here is what produced a 0.63 delivery figure once, by regression
    dilution.
    """
    pts = []
    for lo, hi in GYRO_BINS:
        b = [x for x in sel if x["gyro"] is not None and lo <= x["gyro"] < hi]
        if len(b) < MIN_BIN:
            continue
        pts.append((st.median([x["gyro"] for x in b]),
                    st.median([x["err"] for x in b]), len(b), lo, hi))
    if len(pts) < 3:
        return None, pts
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    w = np.diag(np.array([p[2] for p in pts], float))
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, icept = np.linalg.lstsq(w @ A, w @ ys, rcond=None)[0]
    return (float(slope), float(icept), float(slope) / ppd), pts


def report(run, ppd, show_bins=True):
    rows, recs = collect(run)
    print("=" * 78)
    print(os.path.basename(run.rstrip("/\\")))
    print("=" * 78)
    allc = [r for r in rows if r.get("cmd")]
    print("rows %d | with a command %d | TRACK %d" % (len(rows), len(allc), len(recs)))
    print("states: %s" % dict(Counter(r.get("track") for r in rows)))
    if not recs:
        print("no TRACK rows -- nothing to score.")
        return None

    tiers = Counter(x["tier"] for x in recs)
    n = len(recs)
    print("tiers (TRACK): %s" % ", ".join(
        "%s %d (%.0f%%)" % (k, v, 100 * v / n) for k, v in tiers.most_common()))

    print()
    print("THE HEADLINE -- px from centre, per tier")
    print("  %-6s %6s %10s %10s %11s" % ("tier", "n", "med err", "med gyro", "med |ff|"))
    for t in ("flow", "held", "none"):
        s = [x for x in recs if x["tier"] == t]
        if not s:
            continue
        g = [x["gyro"] for x in s if x["gyro"] is not None]
        print("  %-6s %6d %10.0f %10s %11.0f"
              % (t, len(s), st.median([x["err"] for x in s]),
                 "%.1f" % st.median(g) if g else "-",
                 st.median([x["ff"] for x in s])))
    print("  %-6s %6d %10.0f" % ("ALL", n, st.median([x["err"] for x in recs])))

    print()
    print("LAG_S per tier   (run_184448 re-scored through THIS file: 0.78 s")
    print("                  overall, 0.53 s on its flow rows. The published")
    print("                  0.49-0.58 used a different px/deg -- see px_per_deg.)")
    res = {}
    for t in ("flow", "held", "none", "ALL"):
        sel = recs if t == "ALL" else [x for x in recs if x["tier"] == t]
        if len(sel) < 20:
            continue
        fit, pts = fit_lag(sel, ppd)
        if fit is None:
            print("  %-5s n=%d -- fewer than 3 usable gyro bins" % (t, len(sel)))
            continue
        slope, icept, lag = fit
        res[t] = lag
        print("  %-5s n=%4d   err = %.1f*gyro + %-5.0f  ->  LAG_S %.2f s"
              % (t, len(sel), slope, icept, lag))
        print("          intercept %.0f px is the error that is NOT lag." % icept)
        if show_bins:
            for mg, me, cnt, lo, hi in pts:
                print("            %3.0f-%-3.0f  n=%4d  med err %5.0f px" % (lo, hi, cnt, me))

    print()
    print("SATURATION (is the rate cap binding?)   -- TRACK rows only")
    rt = sorted(x["rate"] for x in recs)
    q = lambda p: rt[int(p * (len(rt) - 1))]                     # noqa: E731
    print("  max|rate| per row: median %.0f  p75 %.0f  p90 %.0f  p99 %.0f  max %.0f"
          % (q(.5), q(.75), q(.9), q(.99), rt[-1]))
    sat = sum(1 for x in recs if x["saturated"])
    print("  cmd.saturated on %d/%d rows (%.1f%%)" % (sat, len(recs), 100 * sat / len(recs)))
    print("  NOTE: a per-bin MAXIMUM equal to the cap is an order statistic, not")
    print("  a pin. Read the median and p99 before concluding the cap binds.")
    return {"tiers": tiers, "n": n, "lag": res,
            "med_err": st.median([x["err"] for x in recs]), "recs": recs}


def dump_frames(run, recs, k=6):
    """Render the worst TRACK rows. Never score a run without looking."""
    import cv2
    src = os.path.join(run, "raw")
    if not os.path.isdir(src):
        src = os.path.join(run, "frames")
    if not os.path.isdir(src):
        print("no frames on disk.")
        return
    have = set(int(x[:6]) for x in os.listdir(src) if x.endswith(".jpg"))
    worst = sorted((x for x in recs if x["frame_index"] in have),
                   key=lambda x: -x["err"])[:k]
    tiles = []
    for x in worst:
        im = cv2.imread(os.path.join(src, "%06d.jpg" % x["frame_index"]))
        if im is None:
            continue
        if x["box"]:
            a, b, c, d = [int(v) for v in x["box"]]
            cv2.rectangle(im, (a, b), (c, d), (0, 255, 0), 3)
        lab = "%06d t=%.1f %s err=%.0f" % (x["frame_index"], x["rel_t"],
                                           x["tier"], x["err"])
        for col, th in (((0, 0, 0), 5), ((0, 255, 255), 2)):
            cv2.putText(im, lab, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th)
        tiles.append(cv2.resize(im, (640, 360)))
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    out = os.path.join(run, "worst_rows.jpg")
    cv2.imwrite(out, grid)
    print("\nwrote %s -- OPEN IT. Confirm the box is the drone before believing" % out)
    print("any number above (look-at-the-frames-rule).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--vs", default=None, help="a baseline run to compare against")
    ap.add_argument("--frames", action="store_true",
                    help="render the worst TRACK rows to <run>/worst_rows.jpg")
    ap.add_argument("--no-bins", action="store_true")
    a = ap.parse_args(argv)

    ppd = px_per_deg(a.run)
    print("px per deg, from the loop's own J: %.1f\n" % ppd)
    cur = report(a.run, ppd, not a.no_bins)
    if a.frames and cur:
        dump_frames(a.run, cur["recs"])
    if a.vs:
        print()
        base = report(a.vs, ppd, not a.no_bins)
        if cur and base:
            print()
            print("=" * 78)
            print("COMPARISON  (this run minus the baseline)")
            print("=" * 78)
            print("  median TRACK error: %.0f -> %.0f px  (%+.0f)"
                  % (base["med_err"], cur["med_err"], cur["med_err"] - base["med_err"]))
            for t in ("flow", "held", "none", "ALL"):
                if t in cur["lag"] and t in base["lag"]:
                    print("  LAG_S %-5s      : %.2f -> %.2f s  (%+.2f)"
                          % (t, base["lag"][t], cur["lag"][t],
                             cur["lag"][t] - base["lag"][t]))
            for t in ("flow", "held", "none"):
                cb = 100 * base["tiers"].get(t, 0) / max(1, base["n"])
                cc = 100 * cur["tiers"].get(t, 0) / max(1, cur["n"])
                print("  tier %-5s share : %.0f%% -> %.0f%%  (%+.0f)" % (t, cb, cc, cc - cb))
            print()
            print("  A tier share that moved is a change in AVAILABILITY; a LAG_S")
            print("  that moved within a tier is a change in the CONTROL LAW. They")
            print("  are different results and only the second scores the gain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
