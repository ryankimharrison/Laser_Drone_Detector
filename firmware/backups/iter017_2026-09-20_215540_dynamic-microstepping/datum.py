"""
A HOME POSE THAT COMES BACK: home stored as gravity, not as a step count.

THE PROBLEM THIS SOLVES
-----------------------
Measured on this machine 2026-09-20:

    limits ->  pitch now -0.000,  home pitch -0.000
    imu    ->  tilt now -3.94 pitch, -1.97 roll   (4.40 deg from vertical)

The board says it is at its datum. Gravity says that pose is 4.40 deg off
vertical. Both are true: `dzero` wrote "this is zero" into a counter at
whatever pose the turret happened to be in, which is the `--skip-level` path.
That zero lives in RAM and dies at reset, so the next boot picks a different
arbitrary pose and calls it home. Pitch does not return. It never has.

Yaw already returns -- the motors' permanent-magnet field is fixed in the
base frame, so the magnetometer sweep plus a stored reference names a real
pose. Pitch had no equivalent, even though the reference it needs has been
bolted to the payload the whole time. Gravity is absolute, drift-free, and
cannot be zeroed wrong.

WHY `level` IS NOT THAT REFERENCE
---------------------------------
`level` drives to VERTICAL. Home is not vertical -- it is 4.40 deg off it. So
the one absolute reference on the machine was aimed at the wrong target, and
the gap was covered by levelling the rig by hand and passing --skip-level,
which throws the reference away entirely.

Worse, `level` cannot even reach its own target. Of that 4.40 deg, ~2 deg is
ROLL, and the mechanism has pitch and yaw only -- there is no roll axis. So
`level` minimises total tilt using pitch alone against a floor it cannot
remove, converges toward something unreachable, and settles 5-7 deg out.
That is the documented hunting failure, and it is why --skip-level became
standing practice.

Storing home as the measured gravity VECTOR fixes both at once. The stored
vector already contains the irreducible roll, so the target becomes
reachable and the loop converges normally instead of hunting.

THE ERROR METRIC, AND ITS ONE CONSTRAINT
----------------------------------------
    err = angle between (measured g_hat) and (stored home g_hat)

This is NOT yaw-invariant, and that is the one thing a caller has to respect.
Yaw rotates the IMU about the payload's own +Z, so gravity sweeps around in
the IMU frame as yaw turns: the same physical pitch reads a different vector
at a different yaw. Only the angle to +Z itself (guard.tilt) survives yaw,
and that one is unsigned and merges the roll in, so it cannot serve as a
setpoint.

So the yaw at capture is stored alongside the vector, and returning to the
datum must happen AT that yaw. Homing's sequence already ends at the yaw
datum, so this costs nothing -- but it does mean pitch cannot be homed first.
`check_yaw()` enforces it rather than trusting a caller to remember.

WHAT THIS DOES NOT DO
---------------------
It does not beat the lash. Wrist hysteresis measures ~2 deg direction-
dependent at 20-40 deg pitch (0.65 near level), so returning to a vector is
only as repeatable as the direction of approach. Homing's existing lash
preload -- overshoot, then come back the same way every time -- is what pins
that down, and it must stay.
"""

import math

#: Where the datum lives on the board's flash. Next to /slimit.json, and for
#: the same reason: it is a MEASUREMENT taken on the machine, not a setting.
#: A datum that has to be re-established by hand every power cycle is the
#: problem this file exists to remove.
STORE_PATH = "/datum.json"

#: How far the yaw counter may sit from the capture yaw before the vector
#: comparison stops meaning anything. At 4.4 deg of tilt, 2 deg of yaw error
#: moves the gravity vector by roughly 4.4*sin(2 deg) = 0.15 deg -- below the
#: accelerometer's own 0.76 deg/sample noise, so it cannot be the limiting
#: term. Wider than this and the "error" is mostly yaw, and the pitch loop
#: would chase it with the one axis that cannot fix it.
YAW_TOL_DEG = 2.0

#: Refuse to store a datum at an attitude the mechanism should not be holding.
#: Same claim as guard.CEILING_TILT_DEG, made about the same machine.
MAX_TILT_DEG = 88.0


def unit(v):
    """(x, y, z) scaled to length 1, or None if it has no length.

    None rather than a zero vector on purpose: an all-zero accelerometer read
    is the ADXL345 standby signature, and it is the single most dangerous
    value this sensor produces -- it turns into exactly (-0.00, +0.00) tilt,
    which is a perfectly plausible "we are level" and is inside every
    tolerance anything here would test against.
    """
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n <= 0.0:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def angle_between(a, b):
    """Angle between two vectors, in degrees. Order does not matter."""
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    d = ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]
    if d > 1.0:
        d = 1.0
    elif d < -1.0:
        d = -1.0
    return math.degrees(math.acos(d))


def tilt_from_vertical(g):
    """Angle between a gravity reading and the payload's own +Z, in degrees.

    The yaw-invariant one. Used for the SAFETY envelope, never as the
    setpoint -- see the module docstring.
    """
    return angle_between(g, (0.0, 0.0, 1.0))


class Datum:
    """The stored home attitude: a gravity direction plus the yaw it was at."""

    def __init__(self, vec=None, yaw=0.0, tilt=None):
        self.vec = vec              # unit gravity in the IMU frame, or None
        self.yaw = float(yaw)       # payload yaw counter at capture, degrees
        self.tilt = tilt            # its angle from vertical, for reporting

    # ------------------------------------------------------------------
    @property
    def is_set(self):
        return self.vec is not None

    def error_from(self, g):
        """Degrees between a gravity reading and this datum. None if unset."""
        if self.vec is None:
            return None
        return angle_between(g, self.vec)

    def check_yaw(self, yaw_now):
        """(ok, reason). Is the vector comparison meaningful at this yaw?

        Enforced rather than documented: the error metric is yaw-dependent,
        and a pitch loop chasing a yaw error would drive the one axis that
        cannot reduce it -- which is exactly the failure mode `level` already
        has with roll, arrived at from the other direction.
        """
        if self.vec is None:
            return (False, "no datum stored")
        dy = abs(float(yaw_now) - self.yaw)
        if dy > YAW_TOL_DEG:
            return (False,
                    "yaw is %+.2f but the datum was captured at %+.2f (%.2f "
                    "deg away, tolerance %.2f). The gravity vector rotates "
                    "with yaw, so the error here is mostly yaw and pitch "
                    "cannot fix it. Reach the home yaw first."
                    % (yaw_now, self.yaw, dy, YAW_TOL_DEG))
        return (True, "yaw %+.2f is within %.2f of the capture yaw %+.2f"
                      % (yaw_now, YAW_TOL_DEG, self.yaw))

    # ------------------------------------------------------------------
    def save(self, path=None):
        # path=None, resolved here rather than as a default argument: a
        # default binds STORE_PATH at DEFINITION time, so rebinding the
        # module constant (a test, or a board with a different filesystem
        # layout) would silently keep writing to the original path.
        import json
        path = path or STORE_PATH
        if self.vec is None:
            raise ValueError("nothing to save: no datum has been captured")
        rec = {"gx": self.vec[0], "gy": self.vec[1], "gz": self.vec[2],
               "yaw": self.yaw, "tilt": self.tilt, "v": 1}
        with open(path, "w") as fh:
            json.dump(rec, fh)
        return rec

    @classmethod
    def load(cls, path=None):
        """Read a stored datum, or None. Never raises on a bad file.

        A corrupt or hand-edited datum is refused rather than used: this
        value decides where the turret drives itself at every startup, so
        "unset, and say so" is the only safe reading of a file that does not
        parse.
        """
        import json
        path = path or STORE_PATH
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(rec, dict):
            return None
        try:
            v = unit((float(rec["gx"]), float(rec["gy"]), float(rec["gz"])))
            yaw = float(rec.get("yaw", 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        if v is None:
            return None
        t = tilt_from_vertical(v)
        if t is None or t > MAX_TILT_DEG:
            return None
        return cls(v, yaw, t)

    # ------------------------------------------------------------------
    def describe(self, g_now=None, yaw_now=None):
        out = ["home datum (stored as GRAVITY, so it survives a reset)"]
        if not self.is_set:
            out.append("  NOT SET -- pitch does not return to anything.")
            out.append("  Put the payload where home should be and run "
                       "'datum set'.")
            return "\n".join(out)
        out.append("  stored    g = (%+.4f, %+.4f, %+.4f) at yaw %+.3f"
                   % (self.vec[0], self.vec[1], self.vec[2], self.yaw))
        out.append("            %.2f deg from vertical  (this is where home "
                   "actually is, not where the counter says it is)"
                   % (self.tilt if self.tilt is not None else float("nan")))
        if g_now is not None:
            err = self.error_from(g_now)
            t = tilt_from_vertical(g_now)
            out.append("  now       %.2f deg from vertical, %.2f deg from the "
                       "datum" % (t if t is not None else float("nan"),
                                  err if err is not None else float("nan")))
        if yaw_now is not None:
            ok, why = self.check_yaw(yaw_now)
            out.append("  yaw       %s" % why)
            if not ok:
                out.append("            'datum go' is refused here.")
        return "\n".join(out)
