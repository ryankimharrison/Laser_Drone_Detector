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
#: 2026-09-20: was 150 steps/s, which is 7.3 deg/s -- nowhere near the ~120
#: deg/s the tracking loop actually commands, and gravity load is a function of
#: the torque being asked for. Measured at an operating rate, with the burst
#: shortened to keep the traverse inside the same envelope.
DEFAULT_RATE = 1200.0           # steps/s per motor -> 58.5 deg/s of payload
DEFAULT_BURST_S = 0.4           # -> 23.4 deg of traverse
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


# ==========================================================================
#   SUSTAINED-RATE SHUTTLE SWEEP  --  rate-limited or heat-limited?
# ==========================================================================
#
# THE QUESTION. MAX_MOTOR_RATE is pinned at 1600 on one observation: in
# run_2026-09-20_184448 every 5 s window at or below 1607 steps/s delivered
# ~0.95 of the commanded angle, and the one window at 2468 delivered 0.02. One
# window is not a threshold, and it does not say WHY. Two mechanisms produce
# that shape and they have opposite fixes:
#
#   a RATE limit   -- the motor cannot follow the step train above some rate.
#                     Reproduces instantly, identically, every time, and does
#                     not care how long the axis has been running.
#   a THERMAL limit -- the driver is backing off its current, or the motor is
#                     losing torque as it heats. Comes on DURING a leg, gets
#                     worse the longer the axis runs, and recovers with rest.
#
# Separating them is what this sweep is for: delivery is measured at four
# rates, WITHIN each leg (first half vs second), and the whole set is repeated
# after REST_S. A rate limit repeats identically; a thermal one comes on
# sooner and reads worse on the second pass.
#
# WHY SHUTTLES AND NOT ONE LONG LEG. The axis has 180 deg of pitch travel and
# a matched pair at 2400 steps/s crosses all of it in 1.5 s, so a 20 s one-way
# leg is not slow -- it is impossible, and every leg would be a measurement of
# the end stop. Each leg therefore accelerates, plateaus, and reverses before
# the stop, inside SHUTTLE_ENVELOPE_DEG of the datum, and the sweep runs as
# many legs as it takes to accumulate PLATEAU_TARGET_S of plateau at each rate.
#
# WHY MATCHED MOTORS AND NOT ONE. A single motor on a differential is equal
# parts pitch and yaw, so the payload yaws -- and the IMU rides the payload, so
# its pitch axis stops being the mechanism's pitch axis the moment yaw moves.
# That is what voided nine of fifteen windows in run_2026-09-20_184448:
# per-second "delivery" read up to 410%, which is impossible, because the
# metric was mixing axes. A MATCHED pair is pure pitch by construction and
# holds yaw at the datum, which is what makes the reading mean anything. The
# single-motor checks stay where they belong, in preflight, where a dead axis
# has to announce itself before any number is taken.

#: Matched rate on BOTH motors, per leg.
STALL_RATES = (1200, 1600, 2000, 2400)

#: Plateau seconds to accumulate at each rate, per pass. Not a leg count: at
#: 2400 a leg holds ~0.9 s of plateau and at 1200 ~1.9 s, so a fixed count
#: would gather twice the evidence at the rate that needs it least.
PLATEAU_TARGET_S = 20.0

#: Pitch either side of the datum. PITCH_LIMIT_DEG is (-90, +90); 60 leaves
#: 30 deg of margin for the reversal to arrest in, which at 2400 steps/s and
#: VEL_ACCEL 40000 microsteps/s^2 takes about 3 deg.
SHUTTLE_ENVELOPE_DEG = 60.0

#: Rest between the two passes. A thermal limit recovers over this and comes
#: back SOONER on the second pass; a rate limit reproduces identically and does
#: not care that the axis was idle.
REST_S = 120.0

#: Gyro poll period inside a leg. `imu fast` costs ~0.21 s of wall time, but
#: try_probe DROPS rather than delaying when the vel stream needs the link, so
#: this cannot starve the watchdog it is measuring against.
STALL_POLL_S = 0.05

#: Blanking either side of a reversal. VEL_ACCEL is 40000 microsteps/s^2, so
#: the ramp to 2400 steps/s is 60 ms, and there is another ~2 deg of lash to
#: take up (wrist-hysteresis-under-gravity-load). Samples inside this window
#: are PLATEAU-EXCLUDED: they are real, but they are measuring the ramp.
RAMP_BLANK_S = 0.12

#: `imu fast` prints "IMUF gx gy gz pitch roll" -- the three gyro axes first.
#: Cheaper than `imu` and already in link.PROBE_COMMANDS, so it yields to the
#: vel stream instead of queueing in front of it.
_GYRO = re.compile(r"IMUF\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)")
_MAG = re.compile(r"MAG\s+-?\d+\s+-?\d+\s+-?\d+\s+[-+\d.]+\s+([-+\d.]+)")


def _yaw_now(link):
    """Payload yaw from the magnetometer, or None. Coarse (~1 deg), and the
    only absolute yaw this rig has; recorded per leg so a leg that drifted can
    be thrown out rather than silently averaged in."""
    m = _MAG.search(link.try_probe("imu mag", timeout=0.8) or "")
    return float(m.group(1)) if m else None


def shuttle_leg(link, rate: float, sign: float, start_deg: float, log):
    """ONE traverse: ramp, plateau, stop before the envelope. Matched motors.

    A leg is a single crossing, not a sustained back-and-forth, and consecutive
    legs alternate `sign`. That is what makes the set self-unwinding -- the
    payload ends each pair back where it started -- and it keeps the reversal
    OUTSIDE the measurement, where the VEL_ACCEL ramp and the ~2 deg of lash
    cannot be mistaken for a stall.

    Returns the samples, not a verdict: one leg is about a second, and the rate
    question is answered by pooling every leg at the same rate.

    SCORED ON |omega|. With yaw held at the datum by the matched pair, the
    payload's only rotation IS pitch, so the magnitude of the gyro vector is
    the pitch rate -- without this bench having to know which gyro axis the IMU
    calls pitch, and with no sign for the alternation to confuse. The per-axis
    medians are recorded alongside it, so the bench day settles that mapping as
    a by-product.
    """
    pitch_dps = abs(rate) * config.AXIS_STEP_DEG      # matched pair = pure pitch
    period = 1.0 / VEL_HZ
    plateau, ramp = [], []
    pos = float(start_deg)
    target = math.copysign(SHUTTLE_ENVELOPE_DEG, sign)
    # Open loop from the commanded rate. If the axis IS stalling the integral
    # over-estimates travel and ends the leg early -- erring toward the middle
    # of the envelope, which is the safe direction to be wrong in.
    t0 = time.perf_counter()
    last = t0
    deadline = t0 + SHUTTLE_LEG_MAX_S
    next_poll = t0
    r = sign * abs(rate)
    while True:
        now = time.perf_counter()
        if now >= deadline or (pos - target) * sign >= 0.0:
            break
        pos += sign * pitch_dps * (now - last)
        last = now
        link.send_vel(r, r)
        if now >= next_poll:
            m = _GYRO.search(link.try_probe("imu fast", timeout=0.5) or "")
            if m:
                g = [float(m.group(i)) for i in (1, 2, 3)]
                mag = math.sqrt(sum(x * x for x in g))
                row = (now - t0, mag, g)
                # The first RAMP_BLANK_S is the VEL_ACCEL ramp and the lash
                # take-up. Those samples are real, but they measure the ramp.
                (ramp if now - t0 < RAMP_BLANK_S else plateau).append(row)
            next_poll = now + STALL_POLL_S
        time.sleep(min(period, max(0.0, deadline - time.perf_counter())))
    link.stop()
    held = time.perf_counter() - t0
    return {"rate": rate, "commanded_dps": pitch_dps, "sign": sign,
            "start_deg": start_deg, "end_deg": pos, "held_s": held,
            "timed_out": held >= SHUTTLE_LEG_MAX_S,
            "plateau": plateau, "ramp": ramp}


#: Hard ceiling on one traverse, so a leg cannot run away if the open-loop
#: integral is wrong. The longest legitimate traverse is 120 deg at the slowest
#: rate, 2.1 s; 4 s leaves headroom without letting a runaway last long. A leg
#: that hits this is FLAGGED, because hitting it means the integral and the
#: mechanism disagree -- which is itself the stall being looked for.
SHUTTLE_LEG_MAX_S = 4.0


#: Walking the leftover residual back to the datum between rates. Half the
#: slowest rate under test, so a stall in the unwind can neither be mistaken
#: for nor hide a stall in the legs it sits between.
UNWIND_RATE = 600.0


def _unwind(link, residual_deg: float):
    """Return the payload to the datum. Matched pair, so this is pure pitch and
    leaves yaw where the leg left it."""
    if abs(residual_deg) <= 1.0:
        return
    back = -math.copysign(UNWIND_RATE, residual_deg)
    secs = abs(residual_deg) / (UNWIND_RATE * config.AXIS_STEP_DEG)
    _drive(link, back, back, secs)
    time.sleep(SETTLE_S)


def stall_sweep(link, log):
    """Both passes, with the rest between. Returns a list of per-rate dicts."""
    out = []
    for pass_no in (1, 2):
        if pass_no == 2:
            log("\nresting %.0f s before the repeat. A thermal limit recovers "
                "over this and reads WORSE on this pass; a rate limit "
                "reproduces identically." % REST_S)
            time.sleep(REST_S)
        for rate in STALL_RATES:
            log("\npass %d: matched %d steps/s (%.0f deg/s of pitch), "
                "traversing +/-%.0f deg until %.0f s of plateau"
                % (pass_no, rate, rate * config.AXIS_STEP_DEG,
                   SHUTTLE_ENVELOPE_DEG, PLATEAU_TARGET_S))
            pads = _pads_line(link)
            y0, t0 = _yaw_now(link), read_tilt(link, 7)
            got, legs, plateau, ramp, timeouts = 0.0, 0, [], [], 0
            pos, sign = 0.0, -1.0          # start downward; gravity assists
            while got < PLATEAU_TARGET_S:
                leg = shuttle_leg(link, float(rate), sign, pos, log)
                legs += 1
                timeouts += 1 if leg["timed_out"] else 0
                plateau.extend(leg["plateau"])
                ramp.extend(leg["ramp"])
                got += len(leg["plateau"]) * STALL_POLL_S
                pos = leg["end_deg"]
                sign = -sign               # alternate: the set unwinds itself
                time.sleep(SETTLE_S)
                if legs > 60:
                    log("   60 legs without reaching the target -- the IMU is "
                        "answering too rarely to finish this rate.")
                    break
            _unwind(link, pos)             # back to the datum for the next rate
            y1, t1 = _yaw_now(link), read_tilt(link, 7)
            r = _score(rate, plateau, ramp, legs, got)
            if r is None:
                log("   no plateau samples -- rate discarded.")
                continue
            r.update({"pass": pass_no, "pads": pads, "timeouts": timeouts,
                      "yaw_before": y0, "yaw_after": y1,
                      "tilt_before": t0, "tilt_after": t1})
            out.append(r)
            dy = (None if y0 is None or y1 is None else y1 - y0)
            log("   %d legs, %.0f s of plateau: commanded %.0f deg/s, "
                "measured %.0f -> delivery %.2f (1st half %.2f, 2nd %.2f)%s"
                % (legs, got, r["commanded_dps"], r["measured_dps"],
                   r["delivery"], r["first_half"], r["last_half"],
                   "" if dy is None else "; yaw moved %+.1f deg" % dy))
            if timeouts:
                log("   %d of %d legs hit the %.0f s ceiling: the commanded "
                    "traverse did not arrive, which is the stall itself."
                    % (timeouts, legs, SHUTTLE_LEG_MAX_S))
            if dy is not None and abs(dy) > 5.0:
                log("   YAW MOVED %+.1f deg on a matched pair, which should be "
                    "pure pitch. One motor is not keeping up, so this rate's "
                    "number is about that, not about a rate or a temperature."
                    % dy)
    return out


def _pads_line(link):
    """`pads` in one line, per leg: an axis that lost its STEP pad to SIO counts
    steps it never emits, and would read as a total stall at every rate."""
    try:
        return " ".join((link.command("pads", timeout=5.0) or "").split())
    except Exception:                                      # noqa: BLE001
        return ""


def _score(rate, plateau, ramp, legs, got_s):
    if not plateau:
        return None
    want = abs(rate) * config.AXIS_STEP_DEG
    mags = sorted(p[1] for p in plateau)
    med = mags[len(mags) // 2]
    half = len(plateau) // 2
    first = statistics.median([p[1] for p in plateau[:half]]) if half else float("nan")
    last = statistics.median([p[1] for p in plateau[half:]]) if half else float("nan")
    axes = [statistics.median([p[2][i] for p in plateau]) for i in range(3)]
    return {"rate": rate, "legs": legs, "plateau_s": got_s,
            "n_plateau": len(plateau), "n_ramp": len(ramp),
            "commanded_dps": want, "measured_dps": med,
            "delivery": med / want if want else float("nan"),
            "first_half": first / want if want else float("nan"),
            "last_half": last / want if want else float("nan"),
            "gyro_axis_medians": axes}


def report_stall(rows, log):
    if not rows:
        return
    log("")
    log("%-5s %6s %8s %9s %9s %9s %8s" %
        ("pass", "rate", "legs", "delivery", "1st half", "2nd half", "yaw"))
    for r in sorted(rows, key=lambda x: (x["rate"], x["pass"])):
        dy = (float("nan") if r["yaw_before"] is None or r["yaw_after"] is None
              else r["yaw_after"] - r["yaw_before"])
        log("%-5d %6d %8d %9.2f %9.2f %9.2f %8.1f"
            % (r["pass"], r["rate"], r["legs"], r["delivery"],
               r["first_half"], r["last_half"], dy))
    log("")
    log("READING IT:")
    log("  delivery falls with RATE and pass 2 matches pass 1  -> RATE limit;")
    log("    set MAX_MOTOR_RATE to the highest rate still at ~0.95.")
    log("  delivery falls WITHIN a leg (2nd half < 1st) and is")
    log("    WORSE on pass 2 at the same rate                  -> THERMAL;")
    log("    raising MAX_MOTOR_RATE would work cold and fail in a long track.")
    log("  delivery flat at ~0.95 at every rate                -> NEITHER, and")
    log("    MAX_MOTOR_RATE 1600 is leaving travel on the table. The 184448")
    log("    collapse would then be the firmware pitch clamp, as the trace says.")
    p1 = {r["rate"]: r for r in rows if r["pass"] == 1}
    p2 = {r["rate"]: r for r in rows if r["pass"] == 2}
    both = sorted(set(p1) & set(p2))
    if both:
        drop = [p1[k]["delivery"] - p2[k]["delivery"] for k in both]
        log("")
        log("  pass1 - pass2 by rate: %s"
            % ", ".join("%d: %+.2f" % (k, d) for k, d in zip(both, drop)))
        log("  (all near 0 -> repeatable, i.e. a rate limit. Systematically "
            "positive -> the axis is still warm, i.e. thermal.)")


def _run_stall(a) -> int:
    """The shuttle sweep, with the same preflight and cleanup as the angle legs.
    Preflight is not optional: a dead axis reads 0.00 delivery at every rate,
    which is indistinguishable from the stall this is looking for."""
    from turret_host.link import TurretLink
    link = TurretLink().start()
    log = lambda s: print(s, flush=True)                   # noqa: E731
    rows = []
    try:
        if not preflight(link, log):
            log("\nPREFLIGHT FAILED -- not benching. A dead axis reads 0.00 "
                "delivery at every rate, which is exactly what a stall reads "
                "like, and this sweep could not tell them apart.")
            return 2
        input("\n    level the payload by hand, clear +/-%.0f deg of arc, and "
              "press Enter (Ctrl-C to stop): " % SHUTTLE_ENVELOPE_DEG)
        rows = stall_sweep(link, log)
        report_stall(rows, log)
    except KeyboardInterrupt:
        log("\ninterrupted.")
        report_stall(rows, log)
    finally:
        try:
            link.stop()
            link.command("velmode off", timeout=3.0)
        except Exception:                                  # noqa: BLE001
            pass
        link.close()
    if a.out and rows:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "kind": "shuttle_stall_sweep",
                       "rates": list(STALL_RATES),
                       "plateau_target_s": PLATEAU_TARGET_S,
                       "envelope_deg": SHUTTLE_ENVELOPE_DEG,
                       "rest_s": REST_S,
                       "axis_step_deg": config.AXIS_STEP_DEG,
                       "rows": rows}, fh, indent=2)
        print("wrote %s" % a.out)
    return 0 if rows else 3


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
    ap.add_argument("--stall", action="store_true",
                    help="the shuttle sweep instead of the delivery legs: "
                         "matched motors at %s steps/s, %.0f s of plateau at "
                         "each, twice, with a %.0f s rest between, to separate "
                         "a rate limit from a thermal one"
                         % (list(STALL_RATES), PLATEAU_TARGET_S, REST_S))
    a = ap.parse_args(argv)

    dps = a.rate * config.AXIS_STEP_DEG
    sweep = dps * a.burst
    if a.stall:
        print("shuttle sweep: matched motors (pure pitch, yaw held at the "
              "datum), shuttling +/-%.0f deg." % SHUTTLE_ENVELOPE_DEG)
        print("       AXIS_STEP_DEG %.5f  watchdog %d ms  gyro poll %.2f s  "
              "ramp blanked %.0f ms"
              % (config.AXIS_STEP_DEG, config.VEL_WATCHDOG_MS, STALL_POLL_S,
                 1000 * RAMP_BLANK_S))
        print("       a one-way leg is impossible: %d steps/s crosses the "
              "whole %.0f deg of pitch travel in %.1f s."
              % (STALL_RATES[-1], config.PITCH_LIMIT_DEG[1] - config.PITCH_LIMIT_DEG[0],
                 (config.PITCH_LIMIT_DEG[1] - config.PITCH_LIMIT_DEG[0])
                 / (STALL_RATES[-1] * config.AXIS_STEP_DEG)))
        total = 0.0
        for rate in STALL_RATES:
            dps = rate * config.AXIS_STEP_DEG
            leg_s = 2 * SHUTTLE_ENVELOPE_DEG / dps
            per_leg = max(0.0, leg_s - RAMP_BLANK_S)
            legs = math.ceil(PLATEAU_TARGET_S / per_leg) if per_leg > 0 else 0
            total += legs * (leg_s + SETTLE_S)
            print("       %5d steps/s -> %5.1f deg/s of pitch, %4.1f s per "
                  "traverse (%4.1f s of plateau), ~%2d legs for %.0f s"
                  % (rate, dps, leg_s, per_leg, legs, PLATEAU_TARGET_S))
        print("       2 passes + %.0f s rest = about %.0f min."
              % (REST_S, (2 * total + REST_S) / 60.0))
        if a.dry_run:
            print("\n--dry-run: nothing opened, nothing moved.")
            return 0
        return _run_stall(a)
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
