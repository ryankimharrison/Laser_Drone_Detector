"""THE metric: how far the beam lands from the drone's centre, by speed and range.

    python tools/centre_error.py                       # all known runs
    python tools/centre_error.py --run <dir> [--run <dir>]

STAGED; belongs at tools/. Offline, read-only.

WHY THIS IS NOT `cmd.error_px`
------------------------------
`error_px` is the distance from the goal pixel to the **predicted** target point
-- `est.u/v` carried forward by LATENCY_S, plus whatever `_aim_bias()` adds. It
is the right quantity for the control law and the wrong one for this question,
in two ways that both flatter the result:

  * it is measured against a PREDICTION, so a confident filter pointing at the
    wrong place scores well;
  * `_aim_bias()` deliberately pushes the aim point off centre by a fraction of
    the box height, so a perfectly-executed aim has a non-zero `error_px` by
    design (see the note in memory: 94% of one session's "lead error" was a
    deliberate 15% up-bias).

This measures what was asked for instead: the straight-line distance in the
narrow frame from where the BEAM LANDS (`cmd.goal_u`, `cmd.goal_v`) to the
CENTRE of the detector's box on that same frame. Both are measurements, neither
is a prediction, and no bias is subtracted.

Rows without a narrow box are excluded: with no detection there is no drone
centre to be near, and counting them would score the tracker's imagination.

SPEED AND RANGE
---------------
Image speed comes from consecutive raw box centres with `box_t` advancing, so a
re-stamped duplicate frame cannot contribute a false zero. Range is
`est.range_m`; rows where `range_source` is "assumed" are reported separately,
because that is a constant standing in for a measurement and binning by it
would invent a trend.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

FLIGHT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "diag", "flight")
DEFAULT_RUNS = ("run_2026-09-20_144709", "run_2026-09-20_145109",
                "run_2026-09-20_151635")

SPEED_BINS = [(0, 50), (50, 100), (100, 200), (200, 350), (350, 600),
              (600, float("inf"))]
RANGE_BINS = [(0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0),
              (4.0, float("inf"))]


def load(run):
    path = run if os.path.isdir(run) else os.path.join(FLIGHT, run)
    with open(os.path.join(path, "control.jsonl"), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh], path


def samples(rows, firing_only=False):
    """(centre_err_px, image_speed, range_m, range_source, row) per usable row."""
    prev_t = prev_c = None
    out = []
    for r in rows:
        e = r.get("est") or {}
        c = r.get("cmd") or {}
        box = e.get("box")
        if not box or c.get("goal_u") is None:
            prev_t = prev_c = None
            continue
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        span = min(box[2] - box[0], box[3] - box[1])
        bt = e.get("box_t")
        speed = None
        if prev_t is not None and bt is not None and bt != prev_t:
            dt = bt - prev_t
            if 0 < dt <= 0.25:
                speed = math.hypot(cx - prev_c[0], cy - prev_c[1]) / dt
        if bt is not None and bt != prev_t:
            prev_t, prev_c = bt, (cx, cy)
        if firing_only and r.get("laser") != "FIRING":
            continue
        err = math.hypot(c["goal_u"] - cx, c["goal_v"] - cy)
        # Scale-free companion: the same miss as a fraction of the airframe.
        # A px target is a MOVING PHYSICAL TARGET as range changes -- at half
        # the range the same physical miss is twice the pixels, and the drone
        # is twice as wide too. See the note in the range table.
        frac = (err / span) if span > 0 else None
        out.append((err, speed, e.get("range_m"), e.get("range_source"), r, frac))
    return out


def table(title, rows_of, key, bins, unit):
    print("\n  %s" % title)
    print("  %-14s %7s %9s %9s %9s %9s %9s" %
          (key, "n", "median", "p90", "<25px", "<60px", "err/span"))
    for lo, hi in bins:
        sel = [s for s in rows_of if s[1] is not None and lo <= s[1] < hi] \
            if key == "speed px/s" else \
            [s for s in rows_of if s[2] is not None and lo <= s[2] < hi]
        if len(sel) < 10:
            continue
        e = sorted(s[0] for s in sel)
        f = sorted(s[5] for s in sel if s[5] is not None)
        lab = "%g-%s" % (lo, "inf" if hi == float("inf") else "%g" % hi)
        print("  %-14s %7d %9.1f %9.1f %8.0f%% %8.0f%% %9s"
              % (lab, len(e), statistics.median(e), e[int(0.9 * (len(e) - 1))],
                 100.0 * sum(1 for x in e if x < 25) / len(e),
                 100.0 * sum(1 for x in e if x < 60) / len(e),
                 ("%.2f" % statistics.median(f)) if f else "-"))


def report(run):
    rows, path = load(run)
    allrows = samples(rows)
    fire = samples(rows, firing_only=True)
    print("\n================ %s ================" % os.path.basename(path))
    for label, s in (("ALL rows with a box", allrows), ("FIRING rows only", fire)):
        if not s:
            continue
        e = sorted(x[0] for x in s)
        print("\n  %s: n=%d   beam-to-centre median %.1f px   p90 %.1f   "
              "<25px %.0f%%   <60px %.0f%%"
              % (label, len(e), statistics.median(e), e[int(0.9 * (len(e) - 1))],
                 100.0 * sum(1 for x in e if x < 25) / len(e),
                 100.0 * sum(1 for x in e if x < 60) / len(e)))
        table("by image speed", s, "speed px/s", SPEED_BINS, "px/s")
        meas = [x for x in s if x[3] == "size"]
        if len(meas) >= 20:
            table("by range (measured only, range_source=size; n=%d of %d)"
                  % (len(meas), len(s)), meas, "range m", RANGE_BINS, "m")
        else:
            print("\n  by range: only %d of %d rows have a MEASURED range "
                  "(range_source=size); the rest are the flat ASSUMED_RANGE_M "
                  "constant, so a range curve would be an artefact."
                  % (len(meas), len(s)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    a = ap.parse_args(argv)
    for run in (a.run or list(DEFAULT_RUNS)):
        try:
            report(run)
        except FileNotFoundError:
            print("\n(%s: no control.jsonl)" % run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
