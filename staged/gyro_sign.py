"""Does the platform turn the way it was told? Commanded rate vs measured gyro.

    python tools/gyro_sign.py --run <dir> [--run <dir>] [--window A B]

STAGED; belongs at tools/. Offline, read-only.

WHY THIS IS A VALID IDENTIFICATION WHEN A JACOBIAN FIT IS NOT
--------------------------------------------------------------
Fitting the Jacobian from flight logs gives the wrong SIGN, because the loop
closes through the image: the command is a function of the error, so image
motion and command are correlated by the controller rather than by the plant
(see memory: closed-loop-jacobian-is-invalid).

The gyro is different. Nothing feeds the gyro back into the command -- the
control law never reads it. So `commanded motor rate -> measured body rate` is
the plant, measured open loop, and its SIGN is a fact about the wiring. That is
exactly the question here, and it is the only one of the two that these logs can
answer.

THE UNITS
---------
On the differential, with motor rates A and B in microsteps/s:
    yaw   deg/s = (A - B) / 2 * AXIS_STEP_DEG
    pitch deg/s = -(A + B) / 2 * AXIS_STEP_DEG
The gyro is a 3-vector in the payload IMU frame and WHICH COMPONENT IS WHICH IS
NOT ASSUMED. Each of the three is regressed against both commanded axes and the
pairing is read off the fits; a component that does not respond to either is
reported as such.

TIMING
------
The IMU runs at ~9.2 Hz against 30 Hz commands, so each gyro sample is paired
with the MEAN commanded rate over the interval that preceded it, with each
command expiring after VEL_WATCHDOG_MS -- a command the firmware has already
stopped acting on must not be counted as if it were still in force.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turret_host import config                            # noqa: E402

FLIGHT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "diag", "flight")
STEP = config.AXIS_STEP_DEG
WATCHDOG_S = config.VEL_WATCHDOG_MS / 1000.0
#: Below this commanded rate the gyro is reading noise, not response.
MIN_CMD_DPS = 2.0


def load(run):
    path = run if os.path.isdir(run) else os.path.join(FLIGHT, run)
    rows = [json.loads(l) for l in
            open(os.path.join(path, "control.jsonl"), encoding="utf-8")]
    imu = [json.loads(l) for l in
           open(os.path.join(path, "imu.jsonl"), encoding="utf-8")]
    return rows, imu, path


def commanded(rows):
    """[(t, yaw_dps, pitch_dps)] from the logged motor rates."""
    out = []
    for r in rows:
        c = r.get("cmd") or {}
        a, b = c.get("rate_a"), c.get("rate_b")
        if a is None or b is None:
            continue
        out.append((r["rel_t"],
                    (a - b) / 2.0 * STEP,
                    -(a + b) / 2.0 * STEP))
    return out


def mean_cmd(cmds, t0, t1):
    """Mean commanded (yaw, pitch) over [t0, t1], commands expiring at the
    watchdog. Returns None if nothing was in force for most of the interval."""
    if t1 <= t0:
        return None
    acc_y = acc_p = held = 0.0
    for i, (t, y, p) in enumerate(cmds):
        end = cmds[i + 1][0] if i + 1 < len(cmds) else t + WATCHDOG_S
        end = min(end, t + WATCHDOG_S)          # the firmware stops here
        lo, hi = max(t, t0), min(end, t1)
        if hi > lo:
            acc_y += y * (hi - lo)
            acc_p += p * (hi - lo)
            held += hi - lo
    if held < 0.5 * (t1 - t0):
        return None
    return acc_y / held, acc_p / held


def pair(rows, imu, lo=None, hi=None):
    cmds = commanded(rows)
    if not cmds:
        return []
    g = [x for x in imu if x.get("gyro_dps")]
    out = []
    for i in range(1, len(g)):
        t0, t1 = g[i - 1]["rel_t"], g[i]["rel_t"]
        if lo is not None and (t1 < lo or t0 > hi):
            continue
        if not 0.0 < t1 - t0 < 0.5:
            continue
        m = mean_cmd(cmds, t0, t1)
        if m is None:
            continue
        out.append((t1, m[0], m[1], list(g[i]["gyro_dps"])))
    return out


def fit(xs, ys):
    """slope, intercept, r2 for y = k x + c."""
    n = len(xs)
    if n < 8:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    c = my - k * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (k * x + c)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return k, c, r2


def report(run, lo=None, hi=None, label=""):
    rows, imu, path = load(run)
    P = pair(rows, imu, lo, hi)
    tag = "%s%s" % (os.path.basename(path),
                    "  window %.1f-%.1f s" % (lo, hi) if lo is not None else "")
    print("\n================ %s %s ================" % (tag, label))
    if not P:
        print("  no paired samples")
        return
    print("  paired gyro samples: %d" % len(P))
    for axis, idx in (("yaw", 1), ("pitch", 2)):
        moving = [p for p in P if abs(p[idx]) >= MIN_CMD_DPS]
        print("\n  commanded %s, |cmd| >= %.0f deg/s on %d samples"
              % (axis, MIN_CMD_DPS, len(moving)))
        if len(moving) < 8:
            print("    too few to fit")
            continue
        xs = [p[idx] for p in moving]
        print("    commanded range %+.1f .. %+.1f deg/s" % (min(xs), max(xs)))
        for comp in range(3):
            ys = [p[3][comp] for p in moving]
            f = fit(xs, ys)
            if f is None:
                continue
            k, c, r2 = f
            flag = ""
            if r2 >= 0.25:
                flag = "  <== responds, %s" % ("SAME sign" if k > 0 else "OPPOSITE sign")
            print("    gyro[%d] = %+.3f * cmd %+.2f   R2 %5.2f%s"
                  % (comp, k, c, r2, flag))


# ==========================================================================
#   DELIVERY vs RATE  -- where does the 0.63 collapse?
# ==========================================================================
#: Peak-motor-rate bins, steps/s.
RATE_BINS = [(0, 200), (200, 400), (400, 600), (600, 900), (900, 1300),
             (1300, 1800), (1800, 2500), (2500, 1e9)]

#: gyro component that carries each payload axis, established by the sign fits.
AXIS_GYRO = {"yaw": 2, "pitch": 1}


def paired_rich(rows, imu):
    """Like pair(), plus peak motor rate, command variation and pose pitch.

    `var` is how much the commanded axis rate MOVED inside the interval. A
    large value means the interval straddles a command change, so the platform
    spent part of it accelerating -- the VEL_ACCEL ramp -- and a low apparent
    delivery there is a measurement artefact, not a stall. That is the
    confound the brief asks about, so it is carried per sample.
    """
    cmds = []
    for r in rows:
        c = r.get("cmd") or {}
        a, b = c.get("rate_a"), c.get("rate_b")
        if a is None or b is None:
            continue
        p = r.get("pose_deg") or [None, None]
        cmds.append((r["rel_t"], a, b, p[0]))
    if not cmds:
        return []
    g = [x for x in imu if x.get("gyro_dps")]
    out = []
    for i in range(1, len(g)):
        t0, t1 = g[i - 1]["rel_t"], g[i]["rel_t"]
        if not 0.0 < t1 - t0 < 0.5:
            continue
        inside = [c for c in cmds if t0 <= c[0] <= t1]
        m = mean_cmd([(c[0], (c[1] - c[2]) / 2.0 * STEP,
                       -(c[1] + c[2]) / 2.0 * STEP) for c in cmds], t0, t1)
        if m is None:
            continue
        peak = max((max(abs(c[1]), abs(c[2])) for c in inside), default=None)
        if peak is None:
            continue
        yv = [(c[1] - c[2]) / 2.0 * STEP for c in inside]
        pv = [-(c[1] + c[2]) / 2.0 * STEP for c in inside]
        var = {"yaw": (max(yv) - min(yv)) if yv else 0.0,
               "pitch": (max(pv) - min(pv)) if pv else 0.0}
        tilt = g[i].get("tilt_deg") or [None, None]
        out.append({"t": t1, "yaw": m[0], "pitch": m[1],
                    "gyro": list(g[i]["gyro_dps"]), "peak": peak, "var": var,
                    "pose_pitch": inside[-1][3], "tilt_pitch": tilt[0]})
    return out


def delivery(run, steady_only=False):
    rows, imu, path = load(run)
    P = paired_rich(rows, imu)
    print("\n================ %s%s ================"
          % (os.path.basename(path), "  (steady commands only)" if steady_only else ""))
    if not P:
        print("  no paired samples")
        return
    for axis in ("yaw", "pitch"):
        gi = AXIS_GYRO[axis]
        print("\n  %s   (gyro[%d])" % (axis.upper(), gi))
        print("    %-14s %6s %9s %7s %9s %9s" %
              ("peak steps/s", "n", "delivery", "R2", "med|cmd|", "med ramp"))
        for lo, hi in RATE_BINS:
            sel = [p for p in P if lo <= p["peak"] < hi and abs(p[axis]) >= MIN_CMD_DPS]
            if steady_only:
                sel = [p for p in sel if p["var"][axis] <= 0.25 * max(abs(p[axis]), 1e-6)]
            if len(sel) < 8:
                continue
            xs = [p[axis] for p in sel]
            ys = [p["gyro"][gi] for p in sel]
            f = fit(xs, ys)
            if f is None:
                continue
            k, c, r2 = f
            lab = "%d-%s" % (lo, "inf" if hi > 1e8 else int(hi))
            note = "" if r2 >= 0.25 else "   (R2 low -- weak)"
            print("    %-14s %6d %9.2f %7.2f %9.1f %9.1f%s"
                  % (lab, len(sel), k, r2,
                     statistics.median([abs(x) for x in xs]),
                     statistics.median([p["var"][axis] for p in sel]), note))


def predictors(run):
    """Does ramp or gravity load explain the residual better than rate?"""
    rows, imu, path = load(run)
    P = [p for p in paired_rich(rows, imu)
         if abs(p["yaw"]) >= 5.0 or abs(p["pitch"]) >= 5.0]
    if len(P) < 40:
        return
    print("\n  --- what predicts a LOW delivery ratio? (%s, n=%d) ---"
          % (os.path.basename(path), len(P)))
    recs = []
    for p in P:
        for axis in ("yaw", "pitch"):
            if abs(p[axis]) < 5.0:
                continue
            ratio = p["gyro"][AXIS_GYRO[axis]] / p[axis]
            recs.append((ratio, p["peak"], p["var"][axis] / abs(p[axis]),
                         abs(p["tilt_pitch"]) if p["tilt_pitch"] is not None else None))
    def rho(idx, name):
        pts = [(r[idx], r[0]) for r in recs if r[idx] is not None]
        if len(pts) < 30:
            return
        xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
        def rank(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            rr = [0.0] * len(v)
            for j, i in enumerate(o):
                rr[i] = float(j)
            return rr
        rx, ry = rank(xs), rank(ys)
        mx = sum(rx) / len(rx); my = sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        print("    Spearman rho(delivery, %-22s) = %+.3f   n=%d"
              % (name, num / den if den else float("nan"), len(pts)))
    rho(1, "peak motor rate")
    rho(2, "command variation/|cmd|")
    rho(3, "|payload tilt| deg")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--window", nargs=2, type=float, default=None)
    a = ap.parse_args(argv)
    runs = a.run or ["run_2026-09-20_183021"]
    lo, hi = (a.window if a.window else (None, None))
    for r in runs:
        report(r, lo, hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
