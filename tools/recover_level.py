"""Bring the payload back to level from ANY attitude, without knowing the sign.

    python tools/recover_level.py            # do it
    python tools/recover_level.py --dry-run  # probe and report, no correction

WHY THIS EXISTS
---------------
Firmware `level` measures a sensitivity once, up front, and then drives
open-loop on it: `step = -t / k`. That is fine near yaw 0 and wrong elsewhere.
Which IMU axis a payload-pitch lands on depends on yaw, so k scales by roughly
cos(yaw) and INVERTS past 90 deg. Measured on this machine: k = +1.16 at yaw 0,
k = -0.976 at yaw ~180. A negative k makes the correction run away, and on
2026-09-18 it drove the payload into its own frame.

`level` now refuses a negative or too-small k rather than driving on it, which
is correct and leaves you stuck. This is the way out.

HOW IT DIFFERS
--------------
It never assumes a sign. It probes, measures what actually happened, and keeps
whichever direction REDUCED the tilt -- so it works at any yaw, with the axis
mapping unknown, and with no trust in the step counter (which measured 2.2x
wrong on 2026-09-19).

The error it minimises is the TOTAL angle between the measured gravity vector
and vertical, not a pitch component. A component can be small while the payload
is far from level, if the tilt has landed on the other axis.
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))
import send as sendmod                                    # noqa: E402

IMUF = re.compile(r"IMUF\s+\S+\s+\S+\s+\S+\s+([-+\d.]+)\s+([-+\d.]+)")

SAMPLES = 10
MAX_STEP_DEG = 10.0
MAX_ITERS = 14


def gravity_unit(pitch_deg, roll_deg):
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    return (-math.sin(p), math.sin(r) * math.cos(p), math.cos(r) * math.cos(p))


def tilt_from_level(pitch_deg, roll_deg):
    """Total angle from vertical, in degrees. Axis-agnostic."""
    g = gravity_unit(pitch_deg, roll_deg)
    return math.degrees(math.acos(max(-1.0, min(1.0, g[2]))))


def read(samples=SAMPLES):
    ps, rs = [], []
    for _ in range(samples):
        out = sendmod.send(["imu fast"], timeout=6, quiet=True)
        m = IMUF.search(out[0][1]) if out else None
        if m:
            ps.append(float(m.group(1)))
            rs.append(float(m.group(2)))
    if not ps:
        return None
    p, r = statistics.fmean(ps), statistics.fmean(rs)
    return p, r, tilt_from_level(p, r)


def move(dp):
    sendmod.send(["dmove %.3f 0" % dp], timeout=30, quiet=True)
    time.sleep(0.7)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1.0, help="degrees from level")
    ap.add_argument("--probe", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cur = read()
    if cur is None:
        print("no IMU reading -- is the 12 V supply on?")
        return 2
    print("start: pitch %+.2f  roll %+.2f  -> %.2f deg from level"
          % cur)
    if cur[2] <= args.tol:
        print("already within %.1f deg." % args.tol)
        return 0

    # Probe: which way reduces the total tilt? Measured, never assumed.
    print("\nprobing %+.1f deg to find the direction that helps..." % args.probe)
    before = cur
    move(args.probe)
    after = read()
    if after is None:
        print("lost the IMU mid-probe")
        return 2
    print("  %+.1f deg -> tilt %.2f -> %.2f (%+.2f)"
          % (args.probe, before[2], after[2], after[2] - before[2]))

    if after[2] < before[2]:
        sign = +1.0
    else:
        sign = -1.0
        print("  that made it worse; correcting the other way.")
    # Gain from what the probe actually achieved, floored so a near-zero
    # response cannot produce an enormous step.
    gain = abs(after[2] - before[2]) / abs(args.probe)
    if gain < 0.25:
        print("  response %.3f deg/deg is very small -- the payload may be "
              "against a stop. Continuing with small steps." % gain)
        gain = 0.25
    print("  response %.2f deg per deg commanded, direction %+d"
          % (gain, int(sign)))

    if args.dry_run:
        print("\n(dry run -- undoing the probe and stopping)")
        move(-args.probe)
        return 0

    # DAMPED, AND THE GAIN IS RE-MEASURED EVERY ITERATION.
    #
    # A gain measured once, at the start, is wrong later: total tilt is an
    # angle between vectors, so the response to a pitch command is not linear
    # in it. Measured at 66 deg and applied at 4 deg, the first version
    # commanded 10 deg for a 4 deg error and oscillated 4 -> 14 -> 4 -> 14
    # without ever converging.
    #
    # So: never command more than the error itself (the mechanism tracks at
    # ~1.1x, so a full-error step already slightly overshoots), damp it, and
    # update the gain from what the last move actually achieved.
    DAMP = 0.8
    cur = after
    for i in range(MAX_ITERS):
        if cur[2] <= args.tol:
            break
        step = sign * min(MAX_STEP_DEG, DAMP * cur[2] / max(0.6, gain))
        print("  iter %2d: %.2f deg from level -> commanding %+.2f (gain %.2f)"
              % (i + 1, cur[2], step, gain))
        move(step)
        nxt = read()
        if nxt is None:
            print("  lost the IMU")
            return 2
        achieved = abs(nxt[2] - cur[2])
        if achieved > 0.3 and abs(step) > 0.3:
            gain = 0.5 * gain + 0.5 * (achieved / abs(step))
        if nxt[2] > cur[2] + 0.5:
            sign = -sign
            print("     got worse (%.2f -> %.2f); reversing direction"
                  % (cur[2], nxt[2]))
        cur = nxt

    print("\nfinal: pitch %+.2f  roll %+.2f  -> %.2f deg from level" % cur)
    if cur[2] <= args.tol:
        print("RECOVERED. Firmware `level` can set the datum from here.")
        return 0
    print("did not converge. Check the mechanism by hand.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
