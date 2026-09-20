"""
Differential wrist kinematics -- the layer between motors and payload axes.

The two stepper drivers are called "pan" and "tilt" everywhere else in this
suite. On this platform those names are wrong and actively misleading: the
motors drive the input bevels of a differential, so **neither one is a payload
axis**. Read them as MOTOR A (pan / A3) and MOTOR B (tilt / A2).

    both motors, same direction        -> carrier turns    -> PITCH
    both motors, opposite directions   -> output gear spins -> YAW
    one motor alone                    -> half of each, at once

    pitch = (thetaA + thetaB) / 2N        thetaA = N * (pitch + yaw)
    yaw   = (thetaA - thetaB) / 2N        thetaB = N * (pitch - yaw)

N is the belt reduction (60T miter pulley / 26T motor pulley = 2.3077), the
miters being 1:1. Everything here is linear, so the same transform maps
velocities as well as positions.

Pure Python, no dependencies -- runs on the Pico and on the host test stubs.
"""

import config

# --------------------------------------------------------------------------
# scalar helpers
# --------------------------------------------------------------------------


def motor_step_deg(microstep):
    """Degrees of motor shaft per microstep."""
    return 360.0 / (config.FULL_STEPS_PER_REV * microstep)


def axis_step_deg(microstep):
    """
    Payload degrees when BOTH motors take one step together.

    This is the platform's resolution on a cardinal axis -- pure pitch or
    pure yaw.
    """
    return motor_step_deg(microstep) / config.DIFFERENTIAL_N


def diagonal_step_deg(microstep):
    """
    Payload degrees in EACH axis when only ONE motor steps.

    Half the cardinal resolution, in both axes simultaneously. The platform
    can therefore position more finely than `axis_step_deg` suggests, just
    not along a single axis.
    """
    return axis_step_deg(microstep) / 2.0


def steps_per_axis_rev(microstep):
    """
    Motor steps for one full payload revolution on a cardinal axis.

    Deliberately returned as a float: with a 26T pulley this is 7384.6, not an
    integer, so a whole revolution does not land back on a step boundary.
    Track position in steps and convert to degrees on read -- never the other
    way round, or the error accumulates.
    """
    return 360.0 / axis_step_deg(microstep)


# --------------------------------------------------------------------------
# the transform
# --------------------------------------------------------------------------


def motor_to_payload(steps_a, steps_b, microstep):
    """Motor step counts -> (pitch_deg, yaw_deg)."""
    sd = motor_step_deg(microstep)
    n = config.DIFFERENTIAL_N
    tha = steps_a * sd * config.MOTOR_SIGN["pan"]
    thb = steps_b * sd * config.MOTOR_SIGN["tilt"]
    pitch = (tha + thb) / (2.0 * n) * config.PAYLOAD_SIGN["pitch"]
    yaw = (tha - thb) / (2.0 * n) * config.PAYLOAD_SIGN["yaw"]
    return pitch, yaw


def payload_to_motor(pitch_deg, yaw_deg, microstep):
    """
    (pitch_deg, yaw_deg) -> exact, FRACTIONAL motor step counts.

    Returned unrounded on purpose. Rounding belongs to the caller so that the
    remainder can be carried -- see Platform.steps_for. Rounding here would
    quietly lose up to half a step on every command, which accumulates over a
    tracking session.

    The sign constants are their own inverse (they are +/-1), so applying them
    in both directions round-trips exactly.
    """
    sd = motor_step_deg(microstep)
    n = config.DIFFERENTIAL_N
    p = pitch_deg * config.PAYLOAD_SIGN["pitch"]
    y = yaw_deg * config.PAYLOAD_SIGN["yaw"]
    tha = n * (p + y)
    thb = n * (p - y)
    return (tha / sd * config.MOTOR_SIGN["pan"],
            thb / sd * config.MOTOR_SIGN["tilt"])


def payload_rate_to_motor(pitch_dps, yaw_dps, microstep):
    """
    Payload deg/s -> motor steps/s, signed.

    Same linear map as the position transform; velocities transform
    identically.
    """
    return payload_to_motor(pitch_dps, yaw_dps, microstep)


def motor_rate_to_payload(sps_a, sps_b, microstep):
    """Motor steps/s -> payload (pitch_dps, yaw_dps)."""
    return motor_to_payload(sps_a, sps_b, microstep)


# --------------------------------------------------------------------------
# the platform
# --------------------------------------------------------------------------


class KinematicsError(Exception):
    usage = True


class Platform:
    """
    Both motors as one 2-DOF payload.

    Owns the fractional-step remainder so a long series of small moves does
    not drift, and enforces that the two motors share a microstep setting --
    they must, or the transform is wrong in a way nothing else would catch.
    """

    def __init__(self, axes):
        try:
            self.a = axes["pan"]      # MOTOR A, driver A3
            self.b = axes["tilt"]     # MOTOR B, driver A2
        except KeyError:
            raise KinematicsError("need both 'pan' and 'tilt' axes")
        self._res_a = 0.0
        self._res_b = 0.0
        self.home = dict(config.PAYLOAD_HOME_DEG)

    # ------------------------------------------------------------------
    #   travel limits
    # ------------------------------------------------------------------
    @staticmethod
    def limit_for(axis):
        """(min, max) for 'pitch'/'yaw', or None if the axis is continuous."""
        return config.PAYLOAD_LIMIT_DEG.get(axis)

    def clamp_target(self, pitch, yaw):
        """
        Clamp an absolute payload target to the travel limits.

        Returns (pitch, yaw, hit) where `hit` names the axes that were
        clamped. Clamping happens in PAYLOAD space, before the differential
        transform -- that is the only place it means anything. Clamping motor
        steps instead would be nonsense: one motor alone moves both axes, so a
        motor limit does not correspond to any payload limit.

        A continuous axis (limit None) is passed through untouched. Note that
        it is NOT wrapped here: the step counter is the position of record and
        wrapping it would lose track of how far the payload has actually
        turned, which matters for cable management even when the axis is
        mechanically free.
        """
        hit = []
        out = {}
        for name, want in (("pitch", pitch), ("yaw", yaw)):
            lim = self.limit_for(name)
            if lim is None:
                out[name] = want
                continue
            lo, hi = lim
            got = min(hi, max(lo, want))
            if abs(got - want) > 1e-9:
                hit.append(name)
            out[name] = got
        return out["pitch"], out["yaw"], hit

    def check_or_clamp(self, pitch, yaw):
        """Apply PAYLOAD_CLAMP policy to an absolute target."""
        p, y, hit = self.clamp_target(pitch, yaw)
        if hit and not config.PAYLOAD_CLAMP:
            names = ", ".join(hit)
            raise KinematicsError(
                "target out of travel on %s (pitch %.2f, yaw %.2f); "
                "limits pitch=%s yaw=%s"
                % (names, pitch, yaw,
                   self.limit_for("pitch"), self.limit_for("yaw")))
        return p, y, hit

    # ------------------------------------------------------------------
    #   home
    # ------------------------------------------------------------------
    def set_home(self, pitch=None, yaw=None):
        """
        Mark a home position. With no arguments, uses where the payload is now.

        Home is a datum you choose, not a mechanical reference -- there is no
        absolute encoder on this platform, so 'home' means "the pose I told it
        to remember", and it survives only until the board resets. Set it
        after levelling the payload by eye or with the IMU.
        """
        cp, cy = self.position()
        p = cp if pitch is None else float(pitch)
        y = cy if yaw is None else float(yaw)
        p, y, hit = self.check_or_clamp(p, y)
        self.home = {"pitch": p, "yaw": y}
        return dict(self.home), hit

    def go_home(self, **kw):
        """Move to the stored home position."""
        return self.move_to(self.home["pitch"], self.home["yaw"], **kw)

    # ------------------------------------------------------------------
    @property
    def microstep(self):
        if self.a.microstep != self.b.microstep:
            raise KinematicsError(
                "motors are on different microstep settings (%d and %d). "
                "The differential transform assumes they match -- run "
                "'ms both %d' first."
                % (self.a.microstep, self.b.microstep, self.a.microstep))
        return self.a.microstep

    def set_microstep(self, div):
        """Set both motors together, and reset the remainder."""
        self.a.set_microstep(div)
        self.b.set_microstep(div)
        self._res_a = 0.0
        self._res_b = 0.0

    def enable(self):
        self.a.enable()
        self.b.enable()

    def disable(self):
        self.a.disable()
        self.b.disable()

    @property
    def enabled(self):
        return self.a.enabled and self.b.enabled

    # ------------------------------------------------------------------
    def position(self):
        """
        Current (pitch_deg, yaw_deg).

        Uses `vel_position`, not `position`, and the difference is not
        cosmetic. In velocity mode the integer step counter is frozen -- the
        distance covered lives in the dead-reckoning accumulator until the
        mode is left -- so reading `position` reports a stationary payload
        while the motors are turning.

        That is fatal to a closed loop. A host servoing on this value sees an
        error that never shrinks no matter how far the turret travels, so it
        keeps commanding rate, and the turret runs until something stops it.
        It happened: the first motion looked correct, then it simply did not
        stop. `vel_position` equals `position` whenever the axis is idle, so
        this is always the right one to read.
        """
        return motor_to_payload(self.a.vel_position, self.b.vel_position,
                                self.microstep)

    def zero(self):
        self.a.zero()
        self.b.zero()
        self._res_a = 0.0
        self._res_b = 0.0

    def steps_for(self, dpitch, dyaw):
        """
        Motor steps for a relative payload move, carrying the remainder.

        The remainder matters: a 26T pulley gives 7384.6 steps per payload
        revolution, so almost no useful angle lands on an exact step. Dropping
        the fraction each time would bias every move in the same direction.
        """
        fa, fb = payload_to_motor(dpitch, dyaw, self.microstep)
        fa += self._res_a
        fb += self._res_b
        na = int(round(fa))
        nb = int(round(fb))
        self._res_a = fa - na
        self._res_b = fb - nb
        return na, nb

    # ------------------------------------------------------------------
    def move_by(self, dpitch, dyaw, rate=None, accel=None, guard=None,
                on_progress=None):
        """
        Relative payload move. Both motors run together and finish together.

        `rate` is the payload rate in deg/s on the dominant axis; None uses
        config.MAX_RATE translated through the transform.
        """
        import stepper

        # Clamp in payload space first. A move is only legal if the RESULTING
        # pose is inside the limits, so check the destination, not the delta.
        cp, cy = self.position()
        tp, ty, hit = self.check_or_clamp(cp + dpitch, cy + dyaw)
        dpitch, dyaw = tp - cp, ty - cy

        na, nb = self.steps_for(dpitch, dyaw)
        if na == 0 and nb == 0:
            return {"pitch": 0.0, "yaw": 0.0, "steps": (0, 0),
                    "aborted": False, "guarded": False, "elapsed_us": 0,
                    "clamped": hit}

        # `rate` is payload deg/s along the path. Convert to the dominant
        # motor's step rate: the move takes path_length / rate seconds, and
        # the busier motor has to fit its steps into that.
        sps = None
        if rate is not None and rate > 0:
            path = (dpitch * dpitch + dyaw * dyaw) ** 0.5
            if path > 0:
                seconds = path / float(rate)
                sps = max(abs(na), abs(nb)) / seconds
                if sps <= 0:
                    sps = None

        res = stepper.coordinated_move(self.a, self.b, na, nb, rate=sps,
                                       accel=accel, guard=guard,
                                       on_progress=on_progress)
        p, y = motor_to_payload(res["steps_a"], res["steps_b"], self.microstep)
        res["pitch"] = p
        res["yaw"] = y
        res["steps"] = (res["steps_a"], res["steps_b"])
        res["clamped"] = hit
        return res

    def move_to(self, pitch, yaw, **kw):
        """Absolute payload move, relative to wherever zero was set."""
        cp, cy = self.position()
        return self.move_by(pitch - cp, yaw - cy, **kw)

    # ------------------------------------------------------------------
    def describe(self):
        ms = self.microstep
        p, y = self.position()
        out = [
            "Differential wrist, 1/%d microstepping" % ms,
            "  belt        %dT motor -> %dT miter = %.4f:1"
            % (config.BELT_TEETH_MOTOR, config.BELT_TEETH_MITER,
               config.BELT_RATIO),
            "  miters      %.1f:1   ->  total N = %.4f"
            % (config.MITER_RATIO, config.DIFFERENTIAL_N),
            "",
            "  motor step          %.4f deg" % motor_step_deg(ms),
            "  axis-step  (both)   %.5f deg   <- pure pitch or pure yaw"
            % axis_step_deg(ms),
            "  diagonal   (one)    %.5f deg   <- in BOTH axes at once"
            % diagonal_step_deg(ms),
            "  steps / payload rev %.1f" % steps_per_axis_rev(ms),
            "",
            "  position    pitch %+8.3f deg   yaw %+8.3f deg" % (p, y),
            "  motors      A %+7d st        B %+7d st"
            % (self.a.position, self.b.position),
            "  remainder   A %+7.3f st        B %+7.3f st"
            % (self._res_a, self._res_b),
            "",
            "  signs       motor A %+d, motor B %+d | pitch %+d, yaw %+d"
            % (config.MOTOR_SIGN["pan"], config.MOTOR_SIGN["tilt"],
               config.PAYLOAD_SIGN["pitch"], config.PAYLOAD_SIGN["yaw"]),
        ]
        return "\n".join(out)


def beam_travel_mm(deg, range_m):
    """How far the beam moves on a target at `range_m` for `deg` of rotation."""
    import math
    return range_m * 1000.0 * math.tan(math.radians(deg))
