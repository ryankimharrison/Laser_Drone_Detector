"""How much of a commanded pitch rate does the platform actually deliver,
and does the answer depend on which way gravity is pulling?

STAGED, NOT RUN. Belongs at tools/bench_pitch_delivery.py. Needs COM5 and an
operator in the room.

    python tools/bench_pitch_delivery.py                  # 0, -20, -40 deg
    python tools/bench_pitch_delivery.py --angles 0 -30   # pick your own
    python tools/bench_pitch_delivery.py --dry-run        # no port, no motion

THE QUESTION
------------
run_2026-09-20_142953 commanded -89.4 deg of pitch across 33 s of TRACK; the
gyro integrated -51.3 and gravity moved 60. So roughly 60% of the commanded
pitch happened, with both motors proven to turn the right way (per-motor gyro
regression on that run: A -0.28/+0.27, B -0.33/-0.27, matching the good static
run). That is an OPEN-LOOP number taken from a CLOSED-LOOP run, where the
control law was free to re-ask for whatever did not happen -- so it measures
the loop, not the mechanism.

This measures the mechanism: one fixed rate, one fixed duration, nothing
watching the result, at three payload pitches and in BOTH directions.

WHY BOTH DIRECTIONS IS THE WHOLE POINT
--------------------------------------
A pure SCALE error (AXIS_STEP_DEG wrong, DIFFERENTIAL_N wrong) is symmetric:
it under-delivers by the same fraction up and down, and it is invisible to an
out-and-back test -- see the note in scale-errors-are-invisible-here. Gravity
LOAD is not symmetric: descending, gravity helps and the payload can only
over-run; climbing, it opposes and the motor can stall or slip. So:

    down/commanded ~= up/commanded          -> scale error, not load
    down/commanded >> up/commanded          -> torque-limited climbing
    both ~1.0 here but 0.6 in flight        -> the loop, not the mechanism

Each leg is measured ONE WAY. An out-and-back cancels exactly the asymmetry
being looked for, and hides lash inside the reversal.

WHY THE ACCELEROMETER AND NOT A 500 Hz GYRO STREAM
--------------------------------------------------
There is no firmware command that streams gyro samples to the host. `imu
burst` measures how fast the board CAN read the part (and reports an implied
ODR); it does not emit the samples. `imu watch` cannot stream either -- it
exits on its own CRLF. Polling `imu fast` costs ~0.21 s per sample, so a 2 s
burst yields about ten points: enough for a coarse within-burst profile, not
enough to integrate.

That is fine, because integrating is not the best way to get this number
anyway. Delivered angle is the difference of two ABSOLUTE tilt readings, which
does not accumulate gyro bias over the burst, and the accelerometer's 0.76 deg
single-sample noise is beaten down by taking a median of TILT_SAMPLES at each
end. With the default 14.6 deg traverse that is a signal-to-noise of ~19:1 on
each leg.

If the within-burst profile is wanted properly -- to separate lash take-up at
the start from steady-state slip -- that needs a new firmware command to
stream the gyro. Say so and it can be written; it is not needed for the
delivered/commanded ratio this bench exists to produce.

BEFORE IT MEASURES ANYTHING IT CHECKS THE RIG
----------------------------------------------
Three known faults on this machine produce a confident wrong answer here, and
all three are cheap to exclude first:

  * tilt emits no steps in velocity mode after a position-mode tilt move --
    the counter advances and nothing moves (motor-b-dead-in-velocity-mode).
    A bench that hit this would report 0% delivery and call it hysteresis.
  * the STEP pad can be owned by SIO while the axis believes it is in vel
    mode, so it counts steps it never emits (step-pad-lost-to-sio). `pads`
    is the command that sees it.
  * a standby-latched ADXL345 reads exactly -0.00/+0.00, which is a plausible
    "level" (accel-zero-tilt-signature). Every tilt read here is rejected if
    it does not move at all across the run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turret_host import config                            # noqa: E402

_IMUF = re.compile(r"IMUF\s+\S+\s+\S+\s+\S+\s+([-+\d.]+)\s+([-+\d.]+)")

#: Pure pitch on a differential is BOTH motors at the same rate: the firmware's
#: own kinematics are pitch ~ (rate_a + rate_b), yaw ~ (rate_a - rate_b). A
#: single-motor `vel` would be half pitch and half yaw, which is not the axis
#: gravity loads. (The brief said "vel on tilt"; this is that question asked of
#: the payload axis rather than the motor.)
DEFAULT_RATE = 150.0            # steps/s per motor -> 7.31 deg/s of payload
DEFAULT_BURST_S = 2.0           # -> 14.6 deg of traverse
DEFAULT_ANGLES = (0.0, -20.0, -40.0)

#: Re-send the rate this often. The firmware watchdog is VEL_WATCHDOG_MS (400)
#: and zeroes both rates when it fires, so a bench that let it trip would
#: measure the watchdog.
VEL_HZ = 50.0

TILT_SAMPLES = 15               # median-of-15 at each end of every leg
SETTLE_S = 0.6                  # an abrupt stop rings; do not measure the ring

#: Refuse to start a leg that would end further than this from vertical.
#: app.ATTITUDE_MAX_DEG is 60 and PAYLOAD_LIMIT_DEG['pitch'] is +/-90.
MAX_TILT_DEG = 55.0

#: A leg whose measured traverse is below this is not a measurement -- it is
#: the dead-axis fault above wearing a number.
MIN_TRAVERSE_DEG = 2.0


def _gravity_unit(pitch_deg: float, roll_deg: float):
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    return (-math.sin(p), math.sin(r) * math.cos(p), math.cos(r) * math.cos(p))


def _angle_between(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))


def read_tilt(link, samples: int = TILT_SAMPLES):
    """(pitch, roll) medians, or None. None is NO READING, never 'level'."""
    ps, rs = [], []
    for _ in range(samples):
        m = _IMUF.search(link.try_probe("imu fast", timeout=0.5) or "")
        if m:
            ps.append(float(m.group(1)))
            rs.append(float(m.group(2)))
        time.sleep(0.01)
    if len(ps) < 3:
        return None
    return statistics.median(ps), statistics.median(rs)


# ---------------------------------------------------------------- preflight
def preflight(link, log) -> bool:
    """Exclude the three faults that would make this bench lie. See module doc."""
    ok = True

    st = link.state()
    axes = st.get("axes") or {}
    for name, ax in axes.items():
        if ax.get("fault") is True:
            log("FAULT on %s -- stop." % name)
            ok = False
        if ax.get("mode") == "vel":
            log("%s is already in velocity mode; sending `velmode off`." % name)
            link.command("velmode off", timeout=3.0)
    if any(on for n, on in (st.get("outputs") or {}).items()
           if "laser" in n.lower()):
        log("the LASER output is ON. Disarm before benching.")
        ok = False

    pads = link.command("pads", timeout=5.0)
    log("pads:\n" + pads.strip())
    if "MISMATCH" in pads:
        log("STEP pad MISMATCH -- an axis would count steps it never emits. "
            "Reboot the board (Ctrl-D at the REPL) before benching.")
        ok = False

    t = read_tilt(link, 5)
    if t is None:
        log("no IMU reading at all -- nothing here can be measured.")
        return False
    if t == (0.0, 0.0):
        log("tilt reads exactly -0.00/+0.00, which is the ADXL345 standby "
            "signature, not level. Power-cycle the sensor rail.")
        ok = False
    log("tilt at rest: pitch %+.2f roll %+.2f (%.2f deg from vertical)"
        % (t[0], t[1], _angle_between(_gravity_unit(*t), (0.0, 0.0, 1.0))))

    # Does each motor ACTUALLY move the payload in velocity mode? Short,
    # slow, one at a time -- this is the check that catches a tilt axis that
    # went dead after a position-mode move.
    for which, rates in (("A (pan)", (DEFAULT_RATE, 0.0)),
                         ("B (tilt)", (0.0, DEFAULT_RATE))):
        before = read_tilt(link, 7)
        _drive(link, rates[0], rates[1], 0.7)
        time.sleep(SETTLE_S)
        after = read_tilt(link, 7)
        if before is None or after is None:
            log("motor %s: no IMU reading around the check." % which)
            ok = False
            continue
        moved = _angle_between(_gravity_unit(*before), _gravity_unit(*after))
        log("motor %s single-axis check: payload moved %.2f deg" % (which, moved))
        if moved < 1.0:
            log("  motor %s EMITTED NOTHING (or the payload is jammed). This "
                "is the known dead-axis fault; reboot the board and retry."
                % which)
            ok = False
        # Put it back, so the next check starts where this one did.
        _drive(link, -rates[0], -rates[1], 0.7)
        time.sleep(SETTLE_S)
    return ok


# ------------------------------------------------------------------- driving
def _drive(link, rate_a: float, rate_b: float, seconds: float):
    """Hold a fixed pair of motor rates, petting the watchdog. Returns seconds
    actually held, measured, not assumed."""
    period = 1.0 / VEL_HZ
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        link.send_vel(rate_a, rate_b)
        time.sleep(min(period, deadline - now))
    held = time.perf_counter() - t0
    link.stop()
    return held


def _ramp_loss_deg(rate: float) -> float:
    """Degrees of payload pitch lost to the firmware's velocity ramp.

    VEL_ACCEL is 40000 steps/s^2, so reaching 150 steps/s takes 3.75 ms and
    costs half that interval's travel, twice (up and down). Tiny at these
    rates -- computed rather than ignored so the commanded figure is honest.
    """
    accel = 40000.0
    t_ramp = abs(rate) / accel
    return abs(rate) * t_ramp * config.AXIS_STEP_DEG      # 2 * (0.5*r*t) * deg


def run_leg(link, rate: float, burst_s: float, log):
    """One open-loop burst. Returns a dict, or None if it could not be read."""
    before = read_tilt(link)
    if before is None:
        return None
    g0 = _gravity_unit(*before)
    from_vert = _angle_between(g0, (0.0, 0.0, 1.0))

    profile = []
    period = 1.0 / VEL_HZ
    t0 = time.perf_counter()
    deadline = t0 + burst_s
    next_probe = t0
    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        link.send_vel(rate, rate)
        if now >= next_probe:
            # try_probe DROPS rather than delaying the vel stream, so this
            # cannot cause the watchdog trip it would otherwise measure.
            m = _IMUF.search(link.try_probe("imu fast") or "")
            if m:
                profile.append((now - t0, float(m.group(1)), float(m.group(2))))
            next_probe = now + 0.20
        time.sleep(min(period, max(0.0, deadline - time.perf_counter())))
    held = time.perf_counter() - t0
    link.stop()
    time.sleep(SETTLE_S)

    after = read_tilt(link)
    if after is None:
        return None
    g1 = _gravity_unit(*after)
    delivered = _angle_between(g0, g1)
    commanded = abs(rate) * config.AXIS_STEP_DEG * held - _ramp_loss_deg(rate)
    return {
        "rate_steps_s": rate,
        "held_s": held,
        "tilt_before": before,
        "tilt_after": after,
        "from_vertical_before": from_vert,
        "from_vertical_after": _angle_between(g1, (0.0, 0.0, 1.0)),
        "commanded_deg": commanded,
        "delivered_deg": delivered,
        "ratio": (delivered / commanded) if commanded else float("nan"),
        "profile": profile,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES),
                    help="payload pitches to test, degrees (default 0 -20 -40)")
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE,
                    help="steps/s per motor (both motors = pure pitch)")
    ap.add_argument("--burst", type=float, default=DEFAULT_BURST_S)
    ap.add_argument("--out", default=None, help="write the report as JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the numbers it depends on; open "
                         "no port and move nothing")
    a = ap.parse_args(argv)

    dps = a.rate * config.AXIS_STEP_DEG
    sweep = dps * a.burst
    print("bench: %.0f steps/s on BOTH motors = %.2f deg/s of payload pitch; "
          "%.1f s burst = %.1f deg per leg" % (a.rate, dps, a.burst, sweep))
    print("       AXIS_STEP_DEG %.5f  DIFFERENTIAL_N %.4f  watchdog %d ms"
          % (config.AXIS_STEP_DEG, config.DIFFERENTIAL_N,
             config.VEL_WATCHDOG_MS))
    for ang in a.angles:
        for d, s in (("down", -1.0), ("up", +1.0)):
            end = ang + s * sweep
            flag = "  REFUSED (past %.0f deg)" % MAX_TILT_DEG \
                if abs(end) > MAX_TILT_DEG else ""
            print("       %+6.1f deg %-4s -> %+6.1f%s" % (ang, d, end, flag))
    if a.dry_run:
        print("\n--dry-run: nothing opened, nothing moved.")
        return 0

    from turret_host.link import TurretLink
    link = TurretLink().start()
    log = lambda s: print(s, flush=True)                   # noqa: E731
    report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "rate_steps_s": a.rate, "burst_s": a.burst,
              "axis_step_deg": config.AXIS_STEP_DEG,
              "differential_n": config.DIFFERENTIAL_N, "legs": []}
    try:
        if not preflight(link, log):
            log("\nPREFLIGHT FAILED -- not benching. The numbers this would "
                "produce would be about the fault, not about the mechanism.")
            return 2
        for ang in a.angles:
            for name, sign in (("down", -1.0), ("up", +1.0)):
                if abs(ang + sign * sweep) > MAX_TILT_DEG:
                    log("\n%+.0f deg %s: SKIPPED, would end past %.0f deg "
                        "from vertical." % (ang, name, MAX_TILT_DEG))
                    continue
                log("\n=== payload pitch %+.0f deg, driving %s ===" % (ang, name))
                input("    place the payload at %+.0f deg and press Enter "
                      "(Ctrl-C to stop): " % ang)
                leg = run_leg(link, sign * a.rate, a.burst, log)
                if leg is None:
                    log("    no IMU reading -- leg discarded.")
                    continue
                leg["target_pitch_deg"] = ang
                leg["direction"] = name
                if leg["delivered_deg"] < MIN_TRAVERSE_DEG:
                    log("    delivered only %.2f deg: that is the dead-axis "
                        "signature, not hysteresis. Leg discarded."
                        % leg["delivered_deg"])
                    continue
                report["legs"].append(leg)
                log("    commanded %.2f deg, delivered %.2f deg  -> %.0f%%"
                    % (leg["commanded_deg"], leg["delivered_deg"],
                       100.0 * leg["ratio"]))
    except KeyboardInterrupt:
        log("\ninterrupted.")
    finally:
        try:
            link.stop()
            link.command("velmode off", timeout=3.0)
        except Exception:                                  # noqa: BLE001
            pass
        link.close()

    print("\n%-8s %-6s %10s %10s %8s" % ("pitch", "dir", "commanded",
                                         "delivered", "ratio"))
    for leg in report["legs"]:
        print("%+8.0f %-6s %10.2f %10.2f %7.0f%%"
              % (leg["target_pitch_deg"], leg["direction"],
                 leg["commanded_deg"], leg["delivered_deg"],
                 100.0 * leg["ratio"]))
    for ang in a.angles:
        legs = {l["direction"]: l for l in report["legs"]
                if l["target_pitch_deg"] == ang}
        if "down" in legs and "up" in legs:
            d, u = legs["down"]["ratio"], legs["up"]["ratio"]
            print("%+8.0f asymmetry: down %.0f%% vs up %.0f%% (%+.0f pts) -- %s"
                  % (ang, 100 * d, 100 * u, 100 * (d - u),
                     "gravity load" if abs(d - u) > 0.10 else
                     "symmetric, so this is SCALE, not load"))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print("\nwritten -> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
