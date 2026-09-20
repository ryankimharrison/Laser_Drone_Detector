"""Does a position-mode move kill velocity-mode stepping on the same axis?

THE QUESTION
------------
Measured on the bench 2026-09-20: motor B (tilt) emitted nothing usable in
VELOCITY mode while its counter advanced, and worked normally in POSITION
mode. Motor A (pan) was fine. Every failure came after a position-mode move on
tilt; the one run that held for 88 s (run_2026-09-20_132413) had only pan
position moves before it.

THE HYPOTHESIS THIS TESTS
-------------------------
`coordinated_move()` has no velocity-mode guard -- `move()` and
`single_step()` both do -- and its `finally` clause calls
`_release_step_pin()` on BOTH axes, which hands the STEP pad back to SIO. If
the axis was still in velocity mode, `enter_vel_mode()` then returns early
(`if self.mode == "vel": return`) and NEVER re-claims the pad. The velocity
state machine keeps running, `vel_tick` keeps writing periods and keeps
integrating `_pos_micro`, and not one pulse reaches the driver -- until the
next reboot.

That predicts exactly what the bench saw: counter advances, shaft does not
move, per-axis, persistent.

WHAT THIS SCRIPT DOES
---------------------
Three legs, gravity read before and after each, comparing the gravity vector
rather than differencing pitch and roll (see `angle_between` -- the IMU turns
with the payload, so a given physical rotation lands in pitch and roll in a
ratio set by yaw, and subtracting the angles measures where yaw happened to
be):

  1. BASELINE   velocity-mode leg on the axis under test, straight after a
                reboot. Expect motion.
  2. DISTURB    one position-mode move on that same axis.
  3. RETEST     the identical velocity-mode leg again.

  Old firmware : leg 3 moves far less than leg 1, or not at all.
  Fixed        : leg 3 matches leg 1.

On firmware with the `pads` command (iteration 15+) it also prints who owns
each STEP pad before and after the disturbance, which settles the mechanism
outright: an axis whose mode is `vel` and whose pad reads SIO is the fault,
visible directly rather than inferred from a shaft that did not turn.

USAGE
-----
    .venv\\Scripts\\python.exe tools\\vel_pad_check.py --axis tilt
    .venv\\Scripts\\python.exe tools\\vel_pad_check.py --axis pan --rate 474

REBOOT THE BOARD FIRST. The whole point of leg 1 is that it runs on a pad
nothing has taken yet, so a stale `velmode` session invalidates the result.
Nothing here arms the laser and nothing here homes; the payload is left where
the legs put it, so start from somewhere with room to move.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import turret_host  # noqa: F401,E402  (MSMF env var, before cv2)

import argparse  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from turret_host import calibrate, config  # noqa: E402

_IMUF = re.compile(r"IMUF\s+\S+\s+\S+\s+\S+\s+([-+\d.]+)\s+([-+\d.]+)")

SETTLE_S = 0.40          # an 11 deg move rings; `level` waits the same
TILT_SAMPLES = 12        # single-sample noise is a MEASURED 0.76 deg
KEEPALIVE_S = 0.08       # well inside the 400 ms VEL_WATCHDOG_MS


def gravity_unit(pitch_deg: float, roll_deg: float):
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    return np.array([-math.sin(p),
                     math.sin(r) * math.cos(p),
                     math.cos(r) * math.cos(p)])


def angle_between(a, b) -> float:
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(d))


def read_tilt(link, samples: int = TILT_SAMPLES):
    """(pitch, roll) averaged, or None. None means NO MEASUREMENT.

    A standby-latched ADXL345 reads exactly -0.00/+0.00, which is a perfectly
    plausible answer and is how a dead sensor has passed as success before.
    """
    ps, rs = [], []
    for _ in range(samples):
        m = _IMUF.search(link.command("imu fast", timeout_s=2.0) or "")
        if m:
            ps.append(float(m.group(1)))
            rs.append(float(m.group(2)))
        time.sleep(0.01)
    if len(ps) < max(3, samples // 3):
        return None
    return float(np.median(ps)), float(np.median(rs))


def read_positions(link):
    ax = link.state()["axes"]
    return {k: int(ax[k]["position"]) for k in ("pan", "tilt")}


def show_pads(link, label):
    """Print the firmware's own pad-ownership view, if it has one."""
    reply = link.command("pads", timeout_s=2.0) or ""
    if "PADS" not in reply:
        print("  [%s] `pads` not supported on this firmware "
              "(iteration <= 14) -- inferring from motion only" % label)
        return None
    print("  [%s] pad ownership:" % label)
    for line in reply.splitlines():
        line = line.strip()
        if line.startswith(("pan", "tilt")):
            print("        " + line)
    return reply


def vel_leg(link, axis, rate, secs, label):
    """Hold a velocity-mode rate on ONE axis and measure what actually moved.

    The other axis is commanded zero, so a single motor drives pitch and yaw
    equally and any motion at all is unambiguous.
    """
    print("\n--- %s: vel leg, %s only, %+d steps/s for %.1f s ---"
          % (label, axis, rate, secs))
    before_t = read_tilt(link)
    before_p = read_positions(link)
    if before_t is None:
        raise SystemExit("IMU is not answering -- refusing to measure "
                         "motion against a sensor that is not there")

    ra, rb = (rate, 0) if axis == "pan" else (0, rate)
    t_end = time.time() + secs
    link.command("vel %.1f %.1f" % (ra, rb), timeout_s=2.0)
    while time.time() < t_end:
        time.sleep(KEEPALIVE_S)
        link.command("vel %.1f %.1f" % (ra, rb), timeout_s=2.0)
    link.command("vel 0 0", timeout_s=2.0)
    time.sleep(SETTLE_S)

    after_t = read_tilt(link)
    after_p = read_positions(link)
    if after_t is None:
        raise SystemExit("IMU stopped answering mid-leg")

    moved = angle_between(gravity_unit(*before_t), gravity_unit(*after_t))
    credited = {k: after_p[k] - before_p[k] for k in after_p}
    step_deg = config.AXIS_STEP_DEG
    expect = abs(credited[axis]) * step_deg / 2.0 * math.sqrt(2.0)

    print("  credited      pan %+6d  tilt %+6d microsteps"
          % (credited["pan"], credited["tilt"]))
    print("  tilt(grav)    %+7.2f,%+7.2f -> %+7.2f,%+7.2f"
          % (before_t[0], before_t[1], after_t[0], after_t[1]))
    print("  MOVED         %6.2f deg   (expected ~%.2f deg if fully delivered)"
          % (moved, expect))
    if expect > 0.5:
        print("  delivered     %5.1f%%" % (100.0 * moved / expect))
    return {"moved": moved, "expect": expect, "credited": credited}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", choices=("pan", "tilt"), default="tilt",
                    help="axis under test (default tilt, the one that failed)")
    ap.add_argument("--rate", type=float, default=474.0,
                    help="velocity-mode rate, steps/s (default 474, the rate "
                         "run_2026-09-20_132627 commanded)")
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--disturb-steps", type=int, default=0,
                    help="unused; the disturb move is now the exact return "
                         "of leg 1, so leg 3 starts from the same pose")
    ap.add_argument("--port", default=None)
    args = ap.parse_args()

    print(__doc__.split("USAGE")[0].rstrip())
    print("\n" + "=" * 70)
    print("axis=%s  rate=%+.0f steps/s  secs=%.1f  disturb=%d steps"
          % (args.axis, args.rate, args.secs, args.disturb_steps))
    print("REBOOT THE BOARD BEFORE THIS RUN, or leg 1 is not a baseline.")
    print("=" * 70)

    link = (calibrate.CalibrationLink(port=args.port) if args.port
            else calibrate.CalibrationLink()).open()
    try:
        link.command("enable both", timeout_s=5.0)
        show_pads(link, "after boot")

        one = vel_leg(link, args.axis, args.rate, args.secs, "LEG 1 BASELINE")
        show_pads(link, "after vel leg 1")

        # The disturbance does double duty: it is the position-mode move under
        # test AND it walks the payload back to where leg 1 started, so leg 3
        # runs from the same pose and cannot march the axis into its stop.
        back = -one["credited"][args.axis]
        print("\n--- LEG 2 DISTURB: position-mode move on %s, %+d steps ---"
              % (args.axis, back))
        refused = False
        try:
            link.command("move %s %d 800" % (args.axis, back), timeout_s=60.0)
        except RuntimeError as exc:
            # CalibrationLink.command RAISES on a rejected command, it does not
            # return the text -- so this has to be caught, not inspected.
            if "velocity mode" not in str(exc):
                raise
            refused = True
            # The firmware refusing is CORRECT (move() and, as of iteration 15,
            # coordinated_move() both guard against it). But it means the
            # disturbance never happened, so leg 3 would prove nothing. Leave
            # velocity mode properly and redo the move.
            print("  firmware REFUSED the move while in velocity mode "
                  "-- correct behaviour; leaving vel mode and retrying")
            # The refusal path calls safe_state(), which DE-ENERGISES both
            # axes. On a loaded tilt axis that drops holding torque, so put
            # the coils back before asking for motion.
            link.command("velmode off", timeout_s=5.0)
            link.command("enable both", timeout_s=5.0)
            link.command("move %s %d 800" % (args.axis, back), timeout_s=60.0)
        time.sleep(SETTLE_S)
        show_pads(link, "after the position move")

        three = vel_leg(link, args.axis, args.rate, args.secs,
                        "LEG 3 RETEST")
        show_pads(link, "after vel leg 3")

        print("\n" + "=" * 70)
        a, b = one["moved"], three["moved"]
        print("baseline moved %.2f deg, retest moved %.2f deg" % (a, b))
        if a < 1.0:
            print("VERDICT: INCONCLUSIVE -- the baseline leg barely moved "
                  "either, so there is nothing to lose. Check the axis is "
                  "enabled and free, and that the board was rebooted.")
        elif b < 0.25 * a:
            print("VERDICT: REPRODUCED. Velocity mode on %s is dead after a "
                  "position-mode move on the same axis (%.0f%% of baseline)."
                  % (args.axis, 100.0 * b / a))
            if not refused:
                print("         The move was NOT refused, so this board does "
                      "not have the coordinated_move/enter_vel_mode guard.")
        else:
            print("VERDICT: NOT reproduced -- retest delivered %.0f%% of "
                  "baseline." % (100.0 * b / a))
        print("=" * 70)
    finally:
        try:
            link.command("vel 0 0", timeout_s=2.0)
            link.command("velmode off", timeout_s=5.0)
        finally:
            link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
