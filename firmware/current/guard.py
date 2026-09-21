"""
IMU soft limits: stop the payload before it reaches its own frame.

WHY THIS EXISTS
---------------
The payload has been hitting the frame. Nothing on this machine could have
prevented that, because every limit the firmware had was quoted from the STEP
COUNTER, and the step counter is exactly what stops being true when something
collides: an A4988 raises no fault when a stepper skips against a hard stop,
the PIO goes on emitting pulses, and `position` goes on counting them. On
2026-09-18 the counter was 57 deg wrong while the payload sat against its own
yoke. A limit built on that number cannot see the one event it exists for.

Gravity can. The accelerometer is on the payload, it cannot be zeroed wrong,
and it does not drift. This module turns it into a travel limit.

WHAT EACH SENSOR IS ACTUALLY FOR
--------------------------------
A gyroscope measures angular RATE. It cannot tell you that you are vertical --
integrate it and you get an angle that walks away, which is the same class of
error as the step counter and no improvement on it. So:

  accel   WHERE WE ARE.  `tilt`, the angle between measured gravity and the
          payload's own up. Absolute, drift-free, and correct at every yaw.

  gyro    HOW FAST WE ARE GETTING THERE.  `tilt_rate`, the exact derivative of
          that same angle (see below). This is what makes the limit arrive in
          time: at 195 deg/s with VEL_ACCEL = 40000 the payload needs ~100 ms
          to stop, which is ~10 deg of travel, on top of ~6 deg of sampling
          latency. A limit that waits until `tilt` crosses the line is ~16 deg
          too late. One that brakes on `tilt + tilt_rate * T_STOP` is not.

  accel   THAT WE ALREADY HIT SOMETHING.  A payload at rest reads a constant
  spike   |a| at ANY attitude -- gravity's magnitude does not care which way up
          it is -- so a departure from the at-rest magnitude is pure
          acceleration. This is the only witness a collision has.

THE MATHS, AND WHY THERE IS NO SIGN TO GET WRONG
------------------------------------------------
Everything here is computed from the accel and gyro vectors alone. It uses no
yaw angle, no step counter, no mounting convention and no calibrated sign --
which matters, because this codebase has lost four homing runs and one payload
to exactly those. `level`'s docstring records the same lesson: a signed
sensitivity inverted past 90 deg of yaw and drove the payload into its frame.

  tilt = acos(g_hat . z_hat)

z_hat is the payload's own +Z, the direction `level` drives gravity onto. Yaw
rotates the IMU ABOUT z_hat, so this angle is invariant to yaw by construction
-- no de-rotation, nothing to calibrate. It is the same quantity `level`
already minimises (cli.settle_tilt), so the two agree by definition.

For the rate: gravity is fixed in the world, so in the body frame it moves at
g_hat_dot = -omega x g_hat. Therefore

  d/dt (g_hat . z_hat) = (-omega x g_hat) . z_hat = -omega . (g_hat x z_hat)

and with tilt = acos(c), d(tilt)/dt = -c_dot / sin(tilt), so

  tilt_rate = omega . n_hat,     n_hat = (g_hat x z_hat) / |g_hat x z_hat|

|g_hat x z_hat| is sin(tilt), so n_hat is defined for every tilt except zero
-- and at zero tilt we are as far from a stop as it is possible to be, so the
degenerate case is the harmless one. It is guarded anyway.

The one thing that IS learned rather than derived is which sign of motor push
increases the tilt. That is learned ONLINE, by watching the measured tilt_rate
against the commanded push, and it is re-learned continuously -- so a reversed
motor, a flipped INVERT_DIR or a re-cabled driver cannot leave it stale. See
PushSense.

WHAT IS CALIBRATED, AND HOW
---------------------------
Two numbers, one per side: the tilt at which the payload reaches its frame.
The operator drives the payload to just short of each stop and takes the
current value (`slimit set`). Nothing is assumed about where the stops are --
they are wherever the machine says they are.

The two sides are told apart by the sign of the step counter's pitch, which is
used ONLY as a label ("this is the up stop / this is the down stop"). It is
never used as a magnitude, so the 57-deg-wrong failure cannot propagate into
the limit itself; the worst a wrong counter sign can do is apply the other
side's tilt limit, and both are real limits.

WHERE IT RUNS
-------------
NOT in the velocity-loop ISR. An I2C transaction allocates and can raise, and
stepper.VelocityLoop._tick is an interrupt that must do neither. Instead this
samples in ordinary context -- on every `vel`/`pvel` command (~30 Hz while
tracking) and inside the coordinated_move guard callback (every 64 steps,
~16 ms) -- and publishes plain boolean attributes that the ISR reads. The ISR
does one comparison and zeroes a target rate. See VelocityLoop.apply_guard.
"""

import math
import time

import config


class GuardError(Exception):
    usage = True


# --------------------------------------------------------------------------
#   Tunables. Every one of these is a physical claim; the comment is the
#   argument for the number.
# --------------------------------------------------------------------------

#: Stopping time, seconds: how long the payload takes to come to rest once the
#: guard zeroes the target rate. Derived, not guessed -- VEL_ACCEL is the
#: velocity loop's deceleration in microsteps/s^2 and VEL_MAX_RATE... no: the
#: real ceiling is the platform's measured mechanical top of 195 deg/s, which
#: is 4000 microsteps/s at 1/16. 4000 / 40000 = 0.10 s.
#:
#: Recomputed from config rather than written as a literal so that raising
#: VEL_ACCEL (the comment in config.py invites exactly that) cannot silently
#: leave this stale.
STOP_S = 4000.0 / float(config.VEL_ACCEL)

#: Sampling latency, seconds. The guard only sees the world when something
#: calls poll(), and in velocity mode that is once per host `vel` command --
#: ~30 Hz measured, so ~33 ms between samples, plus the round trip. One full
#: sample period is the honest worst case for how stale `tilt` is when the
#: decision is made.
LATENCY_S = 0.040

#: Total lookahead. This is the number that decides how much room the guard
#: needs: at 195 deg/s it reserves 195 * 0.14 = 27 deg.  That sounds like a
#: lot and it is -- it is what stopping from full slew actually costs, and
#: pretending otherwise is how the payload reached the frame.
LOOKAHEAD_S = STOP_S + LATENCY_S

#: Extra fixed margin, degrees, on top of the predicted stopping distance.
#: Covers the accelerometer's own noise (0.76 deg/sample MEASURED on this
#: part) and the fact that the payload rings after a stop.
MARGIN_DEG = 3.0

#: How far back inside the limit the tilt must come before a latched trip
#: clears itself. Hysteresis, so a payload sitting exactly on the boundary
#: does not chatter between tripped and clear at the sample rate.
RELEASE_DEG = 6.0

#: A trip is not released until the payload has actually SLOWED to this.
#:
#: Without it the guard buzzes. A predictive trip fires early -- from full
#: slew it fires ~27 deg before the stop -- so by the time the payload has
#: halted it is well inside the limit and the position hysteresis alone says
#: "clear" on the very next sample. A host still commanding full rate then
#: gets released, re-accelerates, and re-trips, at the sample rate.
#:
#: Requiring near-rest turns that buzz into one clean stop per trip. It does
#: NOT stop a determined host from approaching the limit in a series of
#: shortening hops -- a gate cannot do rate limiting, and pretending
#: otherwise would be worse than saying so. What it does is make every hop a
#: deliberate, counted event (see guard_stops in STATE) instead of a
#: mechanical shudder nobody can diagnose.
RELEASE_RATE_DPS = 10.0

#: Absolute backstop, degrees of tilt, used when a side has NOT been
#: calibrated. The mechanism reaches its frame before the +/-90 the step
#: counter believes in; 75 is what `level` already refuses to pass
#: (LEVEL_MAX_TILT_DEG) and is the same claim about the same machine.
#:
#: An uncalibrated guard is therefore not an absent guard. It is a coarse one.
DEFAULT_MAX_TILT_DEG = 75.0

#: Hard ceiling on any calibrated limit. Capturing a stop while the payload is
#: already through the frame would otherwise store that as legal travel.
CEILING_TILT_DEG = 88.0

#: Departure from the at-rest |a|, in g, that counts as an impact WHILE
#: MOVING. Deliberately looser than cli.LEVEL_IMPACT_G (0.45), which is
#: checked after a 250 ms settle with the payload stopped. A slewing payload
#: is genuinely accelerating: the IMU is off the rotation axis, so it picks up
#: a lever-arm term on every ramp. Measured ramps on this rig reach ~0.5 g;
#: 1.2 g is clear of that and still far below a collision, which reverses the
#: payload in milliseconds.
IMPACT_G = 1.2

#: An impact is a TRANSIENT. One sample over threshold is noise or a ramp;
#: this many consecutive samples is a collision. At 30 Hz that is ~66 ms.
IMPACT_SAMPLES = 2

#: Below this tilt, n_hat is ill-conditioned (it is g_hat x z_hat normalised,
#: and its magnitude is sin(tilt)). Report the rate as zero rather than
#: amplify noise by 1/sin. At 2 deg from level nothing is near a stop.
RATE_MIN_TILT_DEG = 2.0

#: Learning a push sign needs both a clear command and a clear response.
#: Below either of these the sample says nothing and is discarded.
LEARN_MIN_PUSH = 200.0          # microsteps/s, summed over both motors
LEARN_MIN_RATE_DPS = 4.0        # deg/s of measured tilt rate

#: Where the calibration is persisted on the board's flash. Kept out of
#: config.py deliberately: config.py is source, edited by hand and deployed;
#: this is a measurement taken on the machine and it must survive a reboot
#: without anyone having to remember to save a file.
STORE_PATH = "/slimit.json"


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# --------------------------------------------------------------------------
#   Which way is "into the stop"?
# --------------------------------------------------------------------------
class PushSense:
    """
    Learns the sign relating commanded motor push to measured tilt rate.

    In the differential, pitch is the SUM of the two motor rates -- the
    velocity loop already calls that `push` -- so one signed number describes
    the whole pitch demand. What is NOT known a priori is whether a positive
    push increases or decreases the tilt, because that depends on which side
    of level the payload is on, on INVERT_DIR, on the wiring of both coil
    connectors, and on which way round the miter gears went in.

    NOTHING HERE IS DERIVED. It is measured, continuously, from data the guard
    already has: the commanded push and the gyro-derived tilt rate. That is
    deliberate. Every sign this project has derived from geometry has at some
    point been wrong -- motor A ran backwards in velocity mode for a whole
    session because a cached DIR bit disagreed with the pin, and the symptom
    was indistinguishable from a control-law bug. A sign that is re-measured
    every 33 ms cannot go stale like that.

    `confidence` is the count of consistent observations, capped. It is
    exposed so a caller can tell "I have never seen this move" from "I have
    seen it fifty times and it always agrees".
    """

    MAX_CONFIDENCE = 20

    def __init__(self):
        self.sign = 0           # +1: positive push increases tilt. 0: unknown.
        self.confidence = 0
        self.samples = 0
        self.flips = 0

    def observe(self, push, tilt_rate):
        """One (commanded push, measured tilt rate) pair."""
        if abs(push) < LEARN_MIN_PUSH or abs(tilt_rate) < LEARN_MIN_RATE_DPS:
            return
        self.samples += 1
        s = 1 if (push > 0) == (tilt_rate > 0) else -1
        if s == self.sign:
            if self.confidence < self.MAX_CONFIDENCE:
                self.confidence += 1
            return
        if self.sign == 0:
            self.sign = s
            self.confidence = 1
            return
        # Disagreement with an established sign. Decay rather than flip on one
        # sample: a single contrary observation during a direction reversal,
        # while the payload is still ringing, is not evidence that the
        # mechanism inverted.
        self.confidence -= 1
        if self.confidence <= 0:
            self.sign = s
            self.confidence = 1
            self.flips += 1

    def describe(self):
        if self.sign == 0:
            return "push->tilt sign UNKNOWN (no qualifying motion yet)"
        return ("push->tilt sign %+d (confidence %d/%d, %d samples, %d flips)"
                % (self.sign, self.confidence, self.MAX_CONFIDENCE,
                   self.samples, self.flips))


# --------------------------------------------------------------------------
#   The guard
# --------------------------------------------------------------------------
class CrashGuard:
    """
    Measured travel limits, from gravity.

    Owns no hardware of its own: it borrows the GY85 the console already has
    (one device, one bus, one owner) and the Platform, from which it reads
    nothing but the sign of the pitch counter.

    THE PUBLISHED STATE IS THREE BOOLEANS. `block_pos`, `block_neg` and
    `impact` are plain attributes, written here in ordinary context and read
    by the velocity-loop ISR. Nothing else crosses that boundary: no floats
    that could be half-written, no objects, no method calls into this class.
    """

    def __init__(self, imu, platform):
        self.imu = imu
        self.platform = platform

        #: Captured stop tilts, degrees, or None if that side is uncalibrated.
        #: "pos"/"neg" name the sign of the PITCH COUNTER at the stop, which
        #: is a label only -- see the module docstring.
        self.limit_pos = None
        self.limit_neg = None

        #: At-rest |a| in g, measured. This part reads 0.89-0.99 g stationary,
        #: so a threshold against a textbook 1.000 spends a third of its
        #: budget on scale error before anything moves.
        self.base_g = None

        self.enabled = True
        self.sense = PushSense()

        # --- published to the ISR -------------------------------------
        self.block_pos = False      # refuse push > 0
        self.block_neg = False      # refuse push < 0
        self.impact = False         # collision latched; refuse everything

        # --- last sample, for reporting and for STATE -----------------
        self.tilt = 0.0
        self.tilt_rate = 0.0
        self.predicted = 0.0
        self.a_dev = 0.0
        self.side = 0
        self.fresh_ms = 0
        self.samples = 0
        self.errors = 0
        self.last_error = ""
        self.trips = 0
        self.impacts = 0
        self._over = 0
        self._sampled_at = time.ticks_ms()

    # ------------------------------------------------------------------
    #   measurement
    # ------------------------------------------------------------------
    def measure(self):
        """
        One reading. Returns (tilt_deg, tilt_rate_dps, a_dev_g).

        Two I2C bursts, ~400 us at 400 kHz. Raises OSError if the bus is gone;
        callers decide what that means, because it means different things to a
        move (stop) and to a status print (say so).
        """
        ax, ay, az = self.imu.accel_g()
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if mag <= 0.0:
            # All-zero is the ADXL345 standby signature, and it reads as
            # PERFECTLY LEVEL -- the single most dangerous value this sensor
            # can return, because it is inside every tolerance. Never silently
            # turn it into a tilt.
            raise GuardError(
                "accelerometer read (0, 0, 0) -- the standby signature. A "
                "guard cannot run on a sensor that reports level at every "
                "attitude.")

        gx, gy, gz = ax / mag, ay / mag, az / mag
        tilt = math.degrees(math.acos(_clamp(gz, -1.0, 1.0)))

        # n_hat = (g_hat x z_hat) / |g_hat x z_hat|, with z_hat = (0, 0, 1):
        #   g_hat x z_hat = (gy*1 - gz*0, gz*0 - gx*1, 0) = (gy, -gx, 0)
        # whose magnitude is sqrt(gx^2 + gy^2) = sin(tilt).
        sin_t = math.sqrt(gx * gx + gy * gy)
        if tilt < RATE_MIN_TILT_DEG or sin_t <= 1e-6:
            rate = 0.0
        else:
            wx, wy, wz = self.imu.gyro_dps()
            rate = (wx * gy - wy * gx) / sin_t

        a_dev = abs(mag - self.base_g) if self.base_g else 0.0
        return (tilt, rate, a_dev)

    def measure_baseline(self, samples=16, wait_ms=5):
        """At-rest |a|, in g. THE PAYLOAD MUST BE STILL."""
        acc = 0.0
        n = 0
        for _ in range(samples):
            ax, ay, az = self.imu.accel_g()
            m = math.sqrt(ax * ax + ay * ay + az * az)
            if m > 0.0:
                acc += m
                n += 1
            time.sleep_ms(wait_ms)
        if n == 0:
            raise GuardError("every baseline sample read zero -- the "
                             "accelerometer is in standby")
        self.base_g = acc / n
        return self.base_g

    # ------------------------------------------------------------------
    #   the decision
    # ------------------------------------------------------------------
    def limit_for_side(self, side):
        """The tilt limit that applies on `side`, and where it came from."""
        cal = self.limit_pos if side >= 0 else self.limit_neg
        if cal is not None:
            return (cal, "calibrated")
        return (DEFAULT_MAX_TILT_DEG, "default")

    def poll(self, push=0.0):
        """
        Sample, decide, and publish. Returns True if motion is unobstructed.

        `push` is the pitch component of whatever is about to be commanded --
        the sum of the two motor rates, the same quantity the velocity loop
        clamps on. It is used for two things: to learn the push/tilt sign, and
        to decide which of block_pos/block_neg a trip should set. Pass 0 for a
        pure observation.

        Never raises. A guard whose failure mode is an exception in the middle
        of a move is not a guard -- an I2C hiccup would park the motors in an
        uncontrolled way, or worse, propagate out of the move loop and leave
        the STEP pads unclaimed. A read that fails leaves the previous
        decision standing and is counted.
        """
        if not self.enabled:
            self.block_pos = self.block_neg = self.impact = False
            return True

        try:
            tilt, rate, a_dev = self.measure()
        except Exception as exc:                            # noqa: BLE001
            self.errors += 1
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            # Hold the previous decision. Do NOT open the gate on a failed
            # read: if the guard was blocking a direction, a dead bus is not
            # evidence that it should stop.
            return not (self.block_pos or self.block_neg or self.impact)

        self.samples += 1
        self._sampled_at = time.ticks_ms()
        self.tilt = tilt
        self.tilt_rate = rate
        self.a_dev = a_dev
        self.sense.observe(push, rate)

        # --- impact -------------------------------------------------------
        # Latched, and NOT auto-cleared: a collision is not a position you can
        # retreat from by measurement. Someone has to look at the machine.
        if self.base_g and a_dev > IMPACT_G:
            self._over += 1
            if self._over >= IMPACT_SAMPLES and not self.impact:
                self.impact = True
                self.impacts += 1
        else:
            self._over = 0

        # --- which side are we on? ----------------------------------------
        # The step counter's pitch SIGN, used as a label only. If the counter
        # is unreadable, fall back to "whichever side has the tighter limit",
        # which is the conservative reading of an unknown.
        try:
            pitch_counter, _yaw = self.platform.position()
            side = 1 if pitch_counter >= 0.0 else -1
        except Exception:                                   # noqa: BLE001
            side = self._tightest_side()
        self.side = side

        limit, _src = self.limit_for_side(side)

        # --- predict ------------------------------------------------------
        # THE WHOLE POINT. `tilt` alone is a limit that arrives after the
        # crash. `tilt + rate * LOOKAHEAD_S` is where the payload will be once
        # it has actually stopped, and that is the number worth limiting.
        predicted = tilt + rate * LOOKAHEAD_S
        self.predicted = predicted

        trip = predicted > (limit - MARGIN_DEG)
        # Auto-clear on retreat, on TWO conditions that must both hold:
        #
        #   position  the MEASURED tilt (not the prediction -- once the rates
        #             are cut the prediction collapses onto the tilt, so
        #             releasing on it would release the instant we stopped)
        #             is back inside the limit by the hysteresis band, and
        #   rate      the payload has actually slowed. See RELEASE_RATE_DPS:
        #             without this the guard buzzes instead of stopping.
        #
        # The rate test is SIGNED, not abs(). `rate` is d(tilt)/dt and tilt
        # is distance from level, so rate > 0 is advancing toward the stop
        # and rate < 0 is retreating. abs() would refuse to clear during a
        # brisk retreat -- the exact motion the release exists to reward --
        # and left the latch set all the way back to level in testing.
        release = (tilt < (limit - MARGIN_DEG - RELEASE_DEG)
                   and rate < RELEASE_RATE_DPS)

        was = self.block_pos or self.block_neg
        if trip:
            # Which push made this happen? Prefer the learned sign; it is
            # measured and current. With no learned sign yet, block BOTH --
            # the conservative answer, and it self-corrects within one sample
            # of any real motion because that motion is what teaches the sign.
            s = self.sense.sign
            if s > 0:
                self.block_pos, self.block_neg = True, False
            elif s < 0:
                self.block_pos, self.block_neg = False, True
            else:
                self.block_pos = self.block_neg = True
            if not was:
                self.trips += 1
        elif release:
            self.block_pos = self.block_neg = False

        return not (self.block_pos or self.block_neg or self.impact)

    def _tightest_side(self):
        lp = self.limit_pos if self.limit_pos is not None else DEFAULT_MAX_TILT_DEG
        ln = self.limit_neg if self.limit_neg is not None else DEFAULT_MAX_TILT_DEG
        return 1 if lp <= ln else -1

    def blocks(self, push):
        """True if a push of this sign is currently refused."""
        if self.impact:
            return True
        if push > 0:
            return self.block_pos
        if push < 0:
            return self.block_neg
        return False

    def clear_impact(self):
        self.impact = False
        self._over = 0

    def age_ms(self):
        return time.ticks_diff(time.ticks_ms(), self._sampled_at)

    # ------------------------------------------------------------------
    #   calibration
    # ------------------------------------------------------------------
    def capture(self, side=None):
        """
        Take the CURRENT attitude as the travel limit for the side we are on.

        This is the whole calibration: drive the payload to just short of the
        frame, run it, and the number the machine reports is the limit. No
        geometry, no CAD, no assumption about where the stops ought to be.

        `side` may be forced to +1/-1 when the pitch counter is untrustworthy
        (right after a boot, before any home). Otherwise it is read from the
        counter's sign, which is all it is used for.

        Returns (side, tilt_deg).
        """
        tilt, _rate, _dev = self.measure()
        if tilt > CEILING_TILT_DEG:
            raise GuardError(
                "refusing to store %.2f deg as a travel limit: it is past the "
                "%.0f deg ceiling, which means the payload is already through "
                "its frame. Back it off and capture a pose it can hold."
                % (tilt, CEILING_TILT_DEG))
        if tilt < RATE_MIN_TILT_DEG:
            raise GuardError(
                "refusing to store %.2f deg as a travel limit: that is level. "
                "Drive the payload to the stop first -- capturing here would "
                "forbid all motion." % tilt)
        if side is None:
            pitch_counter, _yaw = self.platform.position()
            side = 1 if pitch_counter >= 0.0 else -1
        if side >= 0:
            self.limit_pos = tilt
        else:
            self.limit_neg = tilt
        return (side, tilt)

    def clear(self, side=None):
        if side is None:
            self.limit_pos = self.limit_neg = None
        elif side >= 0:
            self.limit_pos = None
        else:
            self.limit_neg = None

    # ------------------------------------------------------------------
    #   persistence
    # ------------------------------------------------------------------
    def save(self, path=STORE_PATH):
        """Write the calibration to flash.

        Persisted because it is a MEASUREMENT taken on the machine, not a
        setting: losing it on every reboot would mean the guard runs on
        DEFAULT_MAX_TILT_DEG for the first session after every power cycle,
        which is precisely the session where someone is most likely to be
        moving the thing by hand.
        """
        import json
        rec = {"limit_pos": self.limit_pos, "limit_neg": self.limit_neg,
               "base_g": self.base_g, "v": 1}
        with open(path, "w") as fh:
            json.dump(rec, fh)
        return rec

    def load(self, path=STORE_PATH):
        """Read the calibration back. Returns the record, or None if absent."""
        import json
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(rec, dict):
            return None
        for key, attr in (("limit_pos", "limit_pos"), ("limit_neg", "limit_neg")):
            v = rec.get(key)
            if v is None:
                setattr(self, attr, None)
            else:
                v = float(v)
                # A stored limit past the ceiling is a corrupt or hand-edited
                # file. Refuse it rather than run on it.
                if not (RATE_MIN_TILT_DEG <= v <= CEILING_TILT_DEG):
                    return None
                setattr(self, attr, v)
        bg = rec.get("base_g")
        if bg:
            bg = float(bg)
            if 0.5 <= bg <= 1.6:
                self.base_g = bg
        return rec

    # ------------------------------------------------------------------
    #   reporting
    # ------------------------------------------------------------------
    def state(self):
        """Machine-readable, for the STATE json."""
        return {
            "enabled": self.enabled,
            "tilt": round(self.tilt, 3),
            "rate": round(self.tilt_rate, 3),
            "predicted": round(self.predicted, 3),
            "limit_pos": self.limit_pos,
            "limit_neg": self.limit_neg,
            "side": self.side,
            "block_pos": self.block_pos,
            "block_neg": self.block_neg,
            "impact": self.impact,
            "a_dev": round(self.a_dev, 3),
            "base_g": round(self.base_g, 4) if self.base_g else None,
            "push_sign": self.sense.sign,
            "push_conf": self.sense.confidence,
            "age_ms": self.age_ms(),
            "samples": self.samples,
            "trips": self.trips,
            "impacts": self.impacts,
            "errors": self.errors,
        }

    def describe(self):
        out = ["soft travel limits (measured from gravity, not the counter)"]
        out.append("  status    %s" % ("ARMED" if self.enabled else "DISABLED"))
        for name, side in (("pos", 1), ("neg", -1)):
            lim, src = self.limit_for_side(side)
            cal = self.limit_pos if side > 0 else self.limit_neg
            out.append("  %-9s %6.2f deg  (%s)%s"
                       % ("limit " + name, lim, src,
                          "" if cal is not None
                          else "  <- run 'slimit set' at this stop"))
        out.append("  now       tilt %.2f deg, rate %+.2f deg/s, "
                   "predicted %.2f deg"
                   % (self.tilt, self.tilt_rate, self.predicted))
        lim, _src = self.limit_for_side(self.side)
        out.append("  room      %.2f deg to the %s limit (%.2f reserved for "
                   "stopping at this rate)"
                   % (lim - MARGIN_DEG - self.predicted,
                      "pos" if self.side >= 0 else "neg",
                      abs(self.tilt_rate) * LOOKAHEAD_S + MARGIN_DEG))
        out.append("  blocking  %s"
                   % ("IMPACT LATCHED -- 'slimit reset' to clear" if self.impact
                      else ("push>0" if self.block_pos else "")
                           + (" push<0" if self.block_neg else "")
                      or "nothing"))
        out.append("  %s" % self.sense.describe())
        out.append("  at-rest |a| %s"
                   % ("%.3f g" % self.base_g if self.base_g
                      else "NOT MEASURED -- impact detection is off"))
        out.append("  lookahead %.0f ms (%.0f stopping + %.0f latency), "
                   "margin %.1f deg"
                   % (LOOKAHEAD_S * 1000, STOP_S * 1000, LATENCY_S * 1000,
                      MARGIN_DEG))
        out.append("  samples   %d (%d errors%s)"
                   % (self.samples, self.errors,
                      ", last: " + self.last_error if self.last_error else ""))
        return "\n".join(out)
