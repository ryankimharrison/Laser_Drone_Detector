"""Does the laser wobble relative to the camera, or is the residual just noise?

    python tools/dot_residual.py                 # both runs
    python tools/dot_residual.py --plot <path>   # scatter if anything correlates

STAGED; belongs at tools/. Offline. Depends on dot_parallax.find_dot.

THE ARGUMENT
------------
The laser and the narrow camera are bolted to the same head, so the dot's
position in the camera frame is FIXED except for parallax, which depends only
on range. Anything left after removing a constant and a 1/R term is either
measurement noise or the two mounts moving relative to each other. If it tracks
commanded rate or pose pitch it is mechanical, and it puts a floor under the
aim objective that no control change can beat.

WHAT THE RESIDUAL ALSO CONTAINS, AND WHY THIS IS AN UPPER BOUND
----------------------------------------------------------------
The dot is measured where the beam lands ON THE DRONE, not on a plane. The
airframe is small, irregular and tilted, and the impact point slides across it
as the drone yaws and rolls -- a real change in the dot's image position with
nothing moving on the turret at all. Surface geometry, detection noise and
mount wobble are not separable here, so every number below is an UPPER BOUND on
wobble. Only a flat wall at fixed range isolates the mount, and that is a rig
measurement.

Correlation is reported as Spearman rho (rank), not Pearson: the covariates are
heavily skewed (rate is near zero most of the time) and a couple of large slews
would otherwise carry a Pearson r on their own.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dot_parallax as DP                                  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dot_cache")


def gather(run, use_cache=True):
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, "%s.json" % run)
    if use_cache and os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            return json.load(fh)
    got, path = DP.collect(run)
    recs = []
    prev_sum = None
    for g in got:
        if not g["dot"]:
            prev_sum = None
            continue
        r = g["row"]
        c = r.get("cmd") or {}
        ra, rb = c.get("rate_a"), c.get("rate_b")
        m = (r.get("interlock") or {}).get("margins") or {}
        pose = r.get("pose_deg") or [None, None]
        rate_sum = (abs(ra) + abs(rb)) if (ra is not None and rb is not None) else None
        reversal = None
        if rate_sum is not None and prev_sum is not None:
            reversal = 1.0 if (prev_sum[0] * ra < 0 or prev_sum[1] * rb < 0) else 0.0
        if ra is not None:
            prev_sum = (ra, rb)
        recs.append({
            "frame": g["frame"],
            "du": g["dot"][0] - g["goal"][0],
            "dv": g["dot"][1] - g["goal"][1],
            "area": g["dot"][2], "mode": g["dot"][5],
            "range_m": g["range_m"], "range_source": g["range_source"],
            "rate_sum": rate_sum, "rate_a": ra, "rate_b": rb,
            "reversal": reversal,
            "pitch": pose[0] if pose else None,
            "box_age_ms": m.get("box_age_ms"),
            "span": g["span"],
        })
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(recs, fh)
    return recs


def spearman(a, b):
    """Rank correlation, and the n it used. No scipy dependency."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None
             and not (isinstance(x, float) and math.isnan(x))]
    if len(pairs) < 30:
        return None, len(pairs)
    x = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)

    def rank(v):
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    if d <= 0:
        return None, len(pairs)
    return float((rx * ry).sum()) / d, len(pairs)


def report(run):
    recs = gather(run)
    if not recs:
        print("  (no dot rows)")
        return None
    du = [r["du"] for r in recs]
    dv = [r["dv"] for r in recs]
    print("\n================ %s ================" % run)
    print("  dot rows: %d   (coverage limited by LASER_PULSE_HZ 15 vs ~30 fps)" % len(recs))
    for name, d in (("du", du), ("dv", dv)):
        s = sorted(d)
        print("    %s: median %+6.2f  std %5.2f  p10 %+6.2f  p90 %+6.2f  IQR %5.2f"
              % (name, statistics.median(d), statistics.pstdev(d),
                 s[len(s) // 10], s[9 * len(s) // 10],
                 s[3 * len(s) // 4] - s[len(s) // 4]))
    # Residual about the median -- the constant is the calibration, not wobble.
    mu, mv = statistics.median(du), statistics.median(dv)
    res_u = [x - mu for x in du]
    res_v = [x - mv for x in dv]
    res_r = [math.hypot(a, b) for a, b in zip(res_u, res_v)]
    rs = sorted(res_r)
    print("    residual magnitude about the median: med %.2f  p90 %.2f px"
          % (rs[len(rs) // 2], rs[9 * len(rs) // 10]))

    covs = [("|rate_a|+|rate_b|", [r["rate_sum"] for r in recs]),
            ("reversal (0/1)", [r["reversal"] for r in recs]),
            ("pose pitch deg", [r["pitch"] for r in recs]),
            ("box_age_ms", [r["box_age_ms"] for r in recs]),
            ("range_m", [r["range_m"] for r in recs]),
            ("1/range", [1.0 / r["range_m"] if r["range_m"] else None for r in recs]),
            ("blob area px", [r["area"] for r in recs]),
            ("box span px", [r["span"] for r in recs])]
    print("\n  Spearman rho against the residual (|rho| > ~0.15 at these n is "
          "real but small; > 0.3 is worth acting on)")
    print("    %-20s %8s %8s %8s %7s" % ("covariate", "rho(du)", "rho(dv)", "rho(|r|)", "n"))
    flagged = []
    for name, v in covs:
        ru, n1 = spearman(v, res_u)
        rv, _ = spearman(v, res_v)
        rr, _ = spearman(v, res_r)
        def f(x):
            return "%+8.3f" % x if x is not None else "       -"
        print("    %-20s %s %s %s %7d" % (name, f(ru), f(rv), f(rr), n1))
        for tag, x in (("du", ru), ("dv", rv), ("|r|", rr)):
            if x is not None and abs(x) >= 0.30:
                flagged.append((name, tag, x))
    return {"recs": recs, "flagged": flagged, "res_u": res_u, "res_v": res_v}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args(argv)
    runs = a.run or ["run_2026-09-20_151635", "run_2026-09-20_145109"]
    results = {}
    for run in runs:
        results[run] = report(run)
    for run, res in results.items():
        if res and res["flagged"]:
            print("\n  %s CORRELATES: %s" % (run, ", ".join(
                "%s vs %s rho %+.2f" % (n, t, x) for n, t, x in res["flagged"])))
        elif res:
            print("\n  %s: nothing reaches |rho| 0.30 -- the residual is "
                  "consistent with measurement noise plus target geometry, not "
                  "with a mount that moves." % run)
    if a.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            run = runs[0]
            recs = results[run]["recs"]
            rv = results[run]["res_v"]
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
            for i, (lab, key) in enumerate((("|rate_a|+|rate_b| steps/s", "rate_sum"),
                                            ("pose pitch deg", "pitch"),
                                            ("range m", "range_m"))):
                x = [r[key] for r in recs]
                pts = [(a_, b_) for a_, b_ in zip(x, rv) if a_ is not None]
                ax[i].scatter([p[0] for p in pts], [p[1] for p in pts], s=3, alpha=0.25)
                ax[i].set_xlabel(lab)
                ax[i].set_ylabel("dv residual px")
                ax[i].axhline(0, color="k", lw=0.5)
                ax[i].set_title("%s" % lab)
            fig.suptitle("dot-minus-goal vertical residual, %s" % run)
            fig.tight_layout()
            fig.savefig(a.plot, dpi=110)
            print("\n  scatter -> %s" % a.plot)
        except ImportError:
            print("\n  (matplotlib not available; skipped the plot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
