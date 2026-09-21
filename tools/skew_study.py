"""Is the narrow<->wide frame skew constant, does it drift, is it load-driven?

    python tools/skew_study.py diag/flight/run_2026-09-20_151635 \
                               diag/flight/run_2026-09-20_145109

Offline, recorded runs only. No rig.

WHY
---
The fused overlay composites a narrow frame with a wide frame that were not
captured at the same instant, and nothing in the pipeline pairs them by time.
Measured on run_151635 the median gap is 28.4 ms, which at 30 deg/s of
platform motion pulls the two layers ~21 narrow px apart -- larger than the
depth-parallax term and comparable to the whole lens-distortion error that
the registration refit removes. The same gap is carried by the wide->narrow
handoff and the cascade, which is why this is worth characterising rather
than just noting.

A constant offset could be compensated. A drifting one could not. A
load-driven one is a scheduling problem with a different fix again. So:
constant, drifting, or load-driven?

TWO INDEPENDENT MEASURES, AND THEY ANSWER DIFFERENT QUESTIONS
--------------------------------------------------------------
  wide.age_ms   from control.jsonl. How old the wide DETECTION was when the
                control loop used it. This is the number the tracker and the
                cascade actually live with, and it is the authoritative one.

  paired skew   nearest recorded wide JPEG to each narrow frame, by capture
                time. This is what the DISPLAY composites, and it is the one
                the operator sees as a double image.

They are not the same thing and are not expected to agree: the recorder
drops wide frames under load (telemetry.wide_frames_dropped), so the paired
skew can overstate the pipeline's staleness. Reporting both, and their
disagreement, is the point -- a single number here would be quietly wrong for
one of the two audiences.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from turret_host import config                                  # noqa: E402


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _theil_sen(x: np.ndarray, y: np.ndarray, n: int = 4000) -> float:
    """Median pairwise slope. Robust to the outliers a p99 of 122 ms implies,
    where least squares would be led by them."""
    if len(x) < 3:
        return float("nan")
    rng = np.random.RandomState(0)
    i = rng.randint(0, len(x), n)
    j = rng.randint(0, len(x), n)
    ok = x[j] != x[i]
    if not ok.any():
        return float("nan")
    return float(np.median((y[j][ok] - y[i][ok]) / (x[j][ok] - x[i][ok])))


def load_control(run: str) -> List[dict]:
    out = []
    path = os.path.join(run, "control.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def wide_times(run: str) -> np.ndarray:
    ts = []
    for side in sorted(glob.glob(os.path.join(run, "wide", "*.txt"))):
        try:
            head = open(side, encoding="utf-8").readline()
            ts.append(float(head.split("t=")[1].split()[0]))
        except (OSError, IndexError, ValueError):
            pass
    return np.array(sorted(ts))


def study(run: str) -> Optional[dict]:
    rows = load_control(run)
    if not rows:
        print("  no control.jsonl")
        return None
    t0 = rows[0].get("frame_t") or 0.0
    dur = (rows[-1].get("frame_t") or t0) - t0

    # ---- authoritative: the age the loop actually saw -------------------
    age, infer_w, infer_n, rel = [], [], [], []
    for r in rows:
        w = r.get("wide") or {}
        a = w.get("age_ms")
        if a is None or r.get("frame_t") is None:
            continue
        age.append(float(a))
        infer_w.append(float(w.get("infer_ms") or np.nan))
        infer_n.append(float(r.get("infer_ms") or np.nan))
        rel.append(float(r["frame_t"]) - t0)
    age = np.array(age)
    infer_w = np.array(infer_w)
    infer_n = np.array(infer_n)
    rel = np.array(rel)

    # ---- what the display composites ------------------------------------
    wt = wide_times(run)
    paired, prel, signed = [], [], []
    for r in rows:
        ft = r.get("frame_t")
        if ft is None or not len(wt):
            continue
        k = int(np.searchsorted(wt, ft))
        best = None
        for c in (k - 1, k, k + 1):
            if 0 <= c < len(wt):
                d = wt[c] - ft
                if best is None or abs(d) < abs(best):
                    best = d
        paired.append(abs(best) * 1000.0)
        signed.append(best * 1000.0)
        prel.append(ft - t0)
    paired = np.array(paired)
    prel = np.array(prel)
    signed = np.array(signed)

    print("  duration %.0f s, %d control rows, %d recorded wide frames"
          % (dur, len(rows), len(wt)))
    if len(wt) > 1:
        # OVER THE WIDE RECORDING'S OWN SPAN, not the control log's. They are
        # not the same window -- on run_145109 the wide recorder started 148 s
        # before the first control row, and dividing by the control duration
        # reported 28 Hz for a camera actually delivering 9.8. The median
        # interval is quoted beside it because the delivery is bursty enough
        # (p10 32 ms, p90 240 ms on that run) that a mean rate flatters it.
        span = float(wt[-1] - wt[0])
        gaps = np.diff(wt) * 1000.0
        print("  wide delivery: %.1f Hz mean over its own %.0f s span, "
              "%.1f Hz by median interval" % ((len(wt) - 1) / span, span,
                                              1000.0 / np.median(gaps)))
        print("    interval ms: p10 %.0f  median %.0f  p90 %.0f   "
              "(configured throttle %.0f Hz = %.0f ms)"
              % (np.percentile(gaps, 10), np.median(gaps),
                 np.percentile(gaps, 90),
                 config.WIDE_SEARCH_FPS_TRACKING,
                 1000.0 / config.WIDE_SEARCH_FPS_TRACKING))
        if dur > 0:
            print("  narrow: %.1f Hz over %.0f s" % (len(rows) / dur, dur))

    print("\n  %-26s %7s %7s %7s %7s %7s"
          % ("", "median", "p75", "p90", "p99", "max"))
    for name, v in (("wide.age_ms (pipeline)", age),
                    ("paired |skew| (display)", paired)):
        if len(v):
            print("  %-26s %7.1f %7.1f %7.1f %7.1f %7.1f"
                  % (name, np.median(v), np.percentile(v, 75),
                     np.percentile(v, 90), np.percentile(v, 99), v.max()))

    # SIGNED, because it separates two different faults that the absolute
    # value merges: a CLOCK OFFSET between the cameras would show as a
    # non-zero mean and could be subtracted, whereas pure sampling jitter
    # averages to zero and cannot.
    if len(signed):
        print("\n  signed skew: mean %+.1f ms, median %+.1f -- %s"
              % (signed.mean(), np.median(signed),
                 "no constant offset between the cameras, so this is sampling, "
                 "not a clock error" if abs(signed.mean()) < 5.0
                 else "A CONSTANT OFFSET IS PRESENT and could be subtracted"))
        gaps = np.diff(wt) * 1000.0 if len(wt) > 1 else np.array([0.0])
        print("  |skew| predicted by the wide interval alone (T/4): %.1f ms; "
              "measured %.1f" % (np.median(gaps) / 4.0, np.median(paired)))

    # ---- constant, or drifting? -----------------------------------------
    print("\n  DRIFT -- median per fifth of the run (ms)")
    for name, v, x in (("wide.age_ms", age, rel),
                       ("paired |skew|", paired, prel)):
        if not len(v):
            continue
        edges = np.linspace(0, max(x.max(), 1e-9), 6)
        cells = []
        for i in range(5):
            m = (x >= edges[i]) & (x < edges[i + 1])
            cells.append(np.median(v[m]) if m.sum() > 20 else np.nan)
        slope = _theil_sen(x, v)
        spread = np.nanmax(cells) - np.nanmin(cells)
        print("    %-15s %s   slope %+.2f ms/s, fifths span %.1f ms"
              % (name, " ".join("%7.1f" % c for c in cells), slope, spread))

    # ---- load driven? ----------------------------------------------------
    print("\n  CORRELATION with inference cost")
    ok = np.isfinite(infer_w) & np.isfinite(age)
    print("    wide.age_ms vs wide infer_ms    r = %+.3f  (n=%d)"
          % (_pearson(infer_w[ok], age[ok]), int(ok.sum())))
    ok2 = np.isfinite(infer_n) & np.isfinite(age)
    print("    wide.age_ms vs narrow infer_ms  r = %+.3f  (n=%d)"
          % (_pearson(infer_n[ok2], age[ok2]), int(ok2.sum())))
    if len(paired) == len(age) and len(age):
        print("    paired skew vs wide.age_ms      r = %+.3f"
              % _pearson(age, paired))
    if ok.sum() > 100:
        # A correlation coefficient hides a threshold effect, so show the
        # conditional medians too.
        q = np.percentile(infer_w[ok], [25, 50, 75])
        print("    wide.age_ms by wide infer_ms quartile:")
        bounds = [-np.inf] + list(q) + [np.inf]
        for i in range(4):
            m = ok & (infer_w > bounds[i]) & (infer_w <= bounds[i + 1])
            if m.sum() > 20:
                print("      infer %6.1f-%-6.1f ms   age median %6.1f ms  (n=%d)"
                      % (max(bounds[i], 0), min(bounds[i + 1], 1e4),
                         np.median(age[m]), m.sum()))
    return {"age": age, "paired": paired, "infer_w": infer_w, "dur": dur,
            "n_wide": len(wt)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args(argv)
    got = {}
    for run in args.runs:
        run = run.rstrip("/\\")
        print("=" * 72)
        print(os.path.basename(run))
        print("=" * 72)
        if not os.path.isdir(run):
            print("  no such directory")
            continue
        g = study(run)
        if g:
            got[os.path.basename(run)] = g
        print()

    if len(got) > 1:
        print("=" * 72)
        print("ACROSS RUNS -- is the level itself a property of the run?")
        print("=" * 72)
        print("%-28s %10s %10s %10s" % ("run", "age med", "skew med", "wide Hz"))
        for k, g in got.items():
            print("%-28s %10.1f %10.1f %10.1f"
                  % (k, np.median(g["age"]) if len(g["age"]) else float("nan"),
                     np.median(g["paired"]) if len(g["paired"]) else float("nan"),
                     g["n_wide"] / g["dur"] if g["dur"] else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
