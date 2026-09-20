"""Automatic homing -- runs at every startup, before the tracker is allowed to move.

The machine must come up on a REPEATABLE datum, not "wherever it was left".
There is no absolute encoder on either axis, so the datum has to be rebuilt
from physics on every power cycle:

  pitch  gravity, via the firmware `level` command. Gravity does not move, so
         this datum is the same tomorrow as it is today.
  yaw    the motors' permanent-magnet field. The magnetometer sits inches from
         two steppers and cannot see the earth's field at all -- but that is
         the signal, not the problem. The motor field is FIXED IN THE BASE
         FRAME and rotates in the IMU frame as the payload yaws, which makes
         it the only absolute yaw reference the machine has.

Sequence (BUILD_SPEC "Auto-homing"):

  1. confirm the link and that the platform is idle       -- abort loudly
  2. `imu cal`      gyro zero-rate bias, payload must be still
  3. `level`        true-level PITCH datum from gravity
  4. yaw sweep      fit the motor-field trend, take the yaw datum from it
  5. lash preload   approach the datum from ONE direction, overshooting
                    BACKLASH_DEG and coming back
  6. `dzero` + `sethome`, report the datum

Everything here is blocking and slow on purpose -- tens of seconds. Every stage
reports through a progress callback so the GUI can show motion instead of
looking hung.

Run it against a real board with:   python -m turret_host.homing
"""
from __future__ import annotations

# ---- sys.path repair, and it MUST come before every other import ----------
# Running this file as a script (`python turret_host/homing.py`) puts the
# package DIRECTORY on sys.path[0] rather than the project root. That breaks
# two things, and the second one is vicious:
#   1. `turret_host.config` stops resolving, and
#   2. `turret_host/types.py` SHADOWS THE STDLIB `types` MODULE, so the next
#      stdlib import that reaches `from types import GenericAlias` -- which is
#      most of them, via enum -> re -> argparse -- dies with a circular-import
#      error naming our file. Nothing below this block would even get to run.
# Only `sys` and `os` are touched here; both are already loaded by interpreter
# startup, so neither can trip the shadow on the way in.
import os
import sys

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == _HERE:
        # REPLACE, do not append: leaving the package dir on the path leaves
        # the stdlib `types` shadowed.
        sys.path[0] = os.path.dirname(_HERE)
# --------------------------------------------------------------------------

import argparse
import inspect
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from turret_host import config


# ==========================================================================
#   Homing-only constants. Anything the rest of the stack shares lives in
#   config.py -- these are parameters of this routine and nothing else.
# ==========================================================================

# Sweep. +/-12 deg is inside the BUILD_SPEC's +/-10..15 window: wide enough
# that the coarse trend dominates sensor noise, narrow enough that the field's
# curvature (the motor near-field is anything but uniform) stays negligible.
SWEEP_HALF_DEG = 12.0
# 0.75 deg is ~4 samples per 3.12 deg ripple period. Nyquist alone would allow
# 1.5 deg; 4 samples/period is what makes the ripple PHASE estimable, and the
# phase is the lost-step detector.
SWEEP_STEP_DEG = 0.75
SWEEP_RATE_DPS = 8.0                  # slow enough that a step barely rings
SWEEP_SETTLE_S = 0.30                 # after the move, before the first sample

# The firmware configures the HMC5883L at 15 Hz with 8x averaging, so a new
# conversion lands every ~67 ms. Reading faster than that returns the SAME
# register contents, and averaging those would fake a noise reduction that is
# not there. 90 ms leaves margin.
MAG_SAMPLE_PERIOD_S = 0.09
MAG_SAMPLES_PER_STEP = 4

# HMC5883L at the firmware's default gain: output is +/-2048 counts, and the
# part reports -4096 as an explicit overflow sentinel. The measured field here
# is ~2000 counts -- about 2 % of headroom. Fitting clipped data produces a
# confident, WRONG datum, which is the worst failure available, so saturation
# is checked on every raw sample and is fatal to the yaw fit.
MAG_FULL_SCALE_LSB = 2048.0
MAG_OVERFLOW_SENTINEL = -4096
MAG_CLIP_LSB = 2040                   # any component at/over this is clipped
MAG_HEADROOM_WARN_FRAC = 0.95         # warn if |B| exceeds this much of FS

# Fit acceptance. The previously measured slope is about -4.96 LSB/deg on my.
EXPECTED_SLOPE_LSB_PER_DEG = -4.96
MIN_FIT_R2 = 0.90
MIN_ABS_SLOPE_LSB_PER_DEG = 2.0       # below this the datum is ill-conditioned
SLOPE_SANITY_FACTOR = 3.0             # report if slope is 3x off expectation
# The datum must sit inside the swept arc (plus a little). Extrapolating a
# local linear fit of a near-field magnet is meaningless.
MAX_EXTRAPOLATION_DEG = 3.0

# Rotor ripple. 7.2 deg of motor shaft is four full steps -- one magnetic
# period of a 200-step/rev rotor -- and the belt divides it by DIFFERENTIAL_N.
MOTOR_ROTOR_PERIOD_DEG = 7.2
RIPPLE_PERIOD_PREDICTED_DEG = MOTOR_ROTOR_PERIOD_DEG / config.DIFFERENTIAL_N
RIPPLE_SEARCH_DEG = (2.2, 4.6)        # brackets the 3.12 deg prediction
RIPPLE_SEARCH_N = 241
MIN_RIPPLE_AMP_LSB = 1.0              # below this the phase is noise

# Lash preload. PRELOAD_STEPS is 1.2x the measured lash; convert it to degrees
# once, here, so the "overshoot must exceed BACKLASH_DEG" rule is checkable.
PRELOAD_DEG = config.PRELOAD_STEPS * config.AXIS_STEP_DEG
PRELOAD_RATE_DPS = 6.0

# Lost-step check. One motor full step is 1.8 / N = 0.78 deg of payload, so a
# ripple-phase slip of half that is already past "measurement noise".
LOST_STEP_TOL_DEG = 0.5
LOST_STEP_POINTS = 9

# Keep clear of the loom limit when planning the sweep.
YAW_LIMIT_MARGIN_DEG = 5.0

# Post-level residual tilt we are willing to call "level".
PITCH_VERIFY_TOL_DEG = 0.5

# Timeouts, seconds. `level` iterates up to ten times and each iteration
# averages twelve accelerometer samples with the payload stopped, so it is
# genuinely slow; a short timeout here looks exactly like a dead board.
T_STATE = 3.0
T_MAG = 3.0
T_MOVE = 30.0
T_IMU_CAL = 30.0
T_LEVEL = 240.0
T_HOME = 10.0


class HomingError(RuntimeError):
    """Homing could not establish a datum. Never caught inside this module."""


class HomingAborted(HomingError):
    """The caller asked for the run to stop (shutdown). Also a failed datum.

    A subclass of HomingError so every existing handler still treats it as "no
    datum was established" -- which is the truth, and the safe reading.
    """


Progress = Callable[[str, float], None]


# ==========================================================================
#   Results
# ==========================================================================

@dataclass
class MagSample:
    """One `imu mag` reading: field AND the pose it was taken at.

    Both come from a single round trip on purpose -- sampling them with two
    commands lets the platform move in between and silently corrupts the fit.
    """
    mx: float
    my: float
    mz: float
    pitch_deg: float
    yaw_deg: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.mx ** 2 + self.my ** 2 + self.mz ** 2)

    @property
    def max_abs_component(self) -> float:
        return max(abs(self.mx), abs(self.my), abs(self.mz))


@dataclass
class YawFit:
    """Quality of the motor-field fit. Reported whether or not it passed."""
    slope_lsb_per_deg: float
    intercept_lsb: float
    r_squared: float
    rms_residual_lsb: float
    n_samples: int
    span_deg: float
    ripple_period_deg: float
    ripple_amp_lsb: float
    ripple_peak_yaw_deg: float          # yaw of a ripple peak, platform frame
    ripple_period_predicted_deg: float = RIPPLE_PERIOD_PREDICTED_DEG

    @property
    def ripple_period_error_frac(self) -> float:
        return (self.ripple_period_deg - self.ripple_period_predicted_deg) \
            / self.ripple_period_predicted_deg


@dataclass
class LostStepCheck:
    """Ripple phase vs commanded position -- a direct lost-step detector.

    The magnetometer caps at 75 Hz, so this is a datum and a health signal,
    never a servo input. It is also unambiguous only WITHIN one ripple period:
    a slip of exactly one rotor period is invisible to it.
    """
    ok: bool
    slip_deg: float
    ambiguity_deg: float
    amplitude_lsb: float
    tolerance_deg: float
    message: str


@dataclass
class HomingResult:
    """What homing established, and how much of it to believe."""
    ok: bool
    yaw_homed: bool                     # False => the yaw datum is arbitrary
    yaw_status: str                     # HOMED | UNHOMED (...) | REFERENCE SET
    datum_pitch_deg: float              # platform frame, after dzero
    datum_yaw_deg: float
    datum_yaw_counter_deg: float        # where the datum sat before dzero
    pitch_tilt_deg: float               # IMU tilt at the datum: the real check
    pitch_datum_ok: bool
    gyro_bias_dps: Tuple[float, float, float]
    mag_saturated: bool
    max_abs_component_lsb: float
    field_magnitude_lsb: float
    yaw_reference_lsb: Optional[float]  # the my value that DEFINES the datum
    yaw_fit: Optional[YawFit]
    elapsed_s: float
    messages: List[str] = field(default_factory=list)
    samples: List[MagSample] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["homing %s" % ("OK" if self.ok else "FAILED")]
        lines.append("  datum      pitch %+.3f  yaw %+.3f  (yaw %s)"
                     % (self.datum_pitch_deg, self.datum_yaw_deg,
                        self.yaw_status))
        lines.append("  pitch tilt %+.2f deg from gravity%s"
                     % (self.pitch_tilt_deg,
                        "" if self.pitch_datum_ok else "   *** OUT OF TOL ***"))
        lines.append("  gyro bias  %+.2f %+.2f %+.2f deg/s" % self.gyro_bias_dps)
        if self.yaw_fit is not None:
            f = self.yaw_fit
            lines.append("  yaw fit    %.3f LSB/deg  R2 %.4f  rms %.2f LSB  "
                         "n=%d over %.1f deg"
                         % (f.slope_lsb_per_deg, f.r_squared,
                            f.rms_residual_lsb, f.n_samples, f.span_deg))
            lines.append("  ripple     %.3f deg period vs %.3f predicted "
                         "(%+.1f%%)  amp %.2f LSB"
                         % (f.ripple_period_deg,
                            f.ripple_period_predicted_deg,
                            100.0 * f.ripple_period_error_frac,
                            f.ripple_amp_lsb))
        lines.append("  field      |B| %.0f LSB, peak component %.0f of %.0f"
                     % (self.field_magnitude_lsb, self.max_abs_component_lsb,
                        MAG_FULL_SCALE_LSB))
        lines.append("  took       %.1f s" % self.elapsed_s)
        for m in self.messages:
            lines.append("  . " + m)
        return "\n".join(lines)


# ==========================================================================
#   Reply parsing
# ==========================================================================

# The firmware turns any failure into a printed line rather than a silent
# no-op, so these prefixes are the only way to know a command was refused.
_ERROR_RE = re.compile(r"^\s*(error:|usage error:|unknown command)",
                       re.IGNORECASE | re.MULTILINE)
_NUM = r"([-+]?\d+(?:\.\d+)?)"
_MAG_RE = re.compile(r"MAG\s+([-+]?\d+)\s+([-+]?\d+)\s+([-+]?\d+)\s+"
                     + _NUM + r"\s+" + _NUM)
_STATE_RE = re.compile(r"STATE\s+(\{.*\})", re.DOTALL)
_NOWAT_RE = re.compile(r"now at pitch\s+" + _NUM + r"\s+yaw\s+" + _NUM)
_TILT_RE = re.compile(r"tilt now\s+" + _NUM + r"\s+pitch,\s*" + _NUM + r"\s+roll")
_BIAS_RE = re.compile(r"gyro bias.*?\(" + _NUM + r",\s*" + _NUM + r",\s*"
                      + _NUM + r"\s+deg/s\)", re.DOTALL)
_HOME_RE = re.compile(r"home set: pitch\s+" + _NUM + r"\s+yaw\s+" + _NUM)
_SENS_RE = re.compile(r"measured\s+" + _NUM + r"\s+deg of tilt per deg")


def parse_mag_line(reply: str) -> Tuple[float, float, float, float, float]:
    """`MAG <mx> <my> <mz> <pitch> <yaw>` -> five floats.

    Used only when the link does not supply its own parse_mag; the wire format
    should have exactly one owner.
    """
    m = _MAG_RE.search(reply)
    if not m:
        raise HomingError("no MAG line in the reply to 'imu mag':\n%s"
                          % reply.strip())
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)),
            float(m.group(4)), float(m.group(5)))


def _coerce_mag(obj) -> MagSample:
    """Accept whatever shape link.parse_mag returns, or say exactly why not."""
    if isinstance(obj, MagSample):
        return obj
    if isinstance(obj, dict):
        keys = ("mx", "my", "mz", "pitch", "yaw") if "pitch" in obj \
            else ("mx", "my", "mz", "pitch_deg", "yaw_deg")
        missing = [k for k in keys if k not in obj]
        if missing:
            raise HomingError("link.parse_mag returned a dict missing %s"
                              % missing)
        return MagSample(*(float(obj[k]) for k in keys))
    seq = tuple(obj)
    if len(seq) != 5:
        raise HomingError("link.parse_mag returned %d values, expected 5 "
                          "(mx my mz pitch yaw)" % len(seq))
    return MagSample(*(float(v) for v in seq))


# ==========================================================================
#   Fitting
# ==========================================================================

def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Least squares y = a*x + b. Returns (a, b, r2, rms_residual)."""
    if x.size < 3:
        raise HomingError("cannot fit a trend from %d samples" % x.size)
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # ss_tot == 0 means my never moved: a dead magnetometer, or an axis that
    # did not turn. Scoring that as a perfect fit would be exactly the silent
    # failure this routine exists to prevent, so it scores zero.
    r2 = 0.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    rms = math.sqrt(ss_res / x.size)
    return float(a), float(b), float(r2), rms


def _fit_ripple(x: np.ndarray, resid: np.ndarray,
                lo: float, hi: float, n: int) -> Tuple[float, float, float]:
    """Pull the rotor ripple out of the trend residual by swept least squares.

    A plain FFT would be the wrong tool: the samples are uniform in COMMANDED
    YAW rather than in time, and the record is only ~8 periods long, so the
    FFT's bin spacing near 3 deg is coarser than the number we are trying to
    report. Sweeping the period and solving a 3-column least squares at each
    candidate costs nothing at this size and resolves the period as finely as
    we ask.

    Returns (period_deg, amplitude_lsb, peak_x_deg).
    """
    best = None
    for period in np.linspace(lo, hi, n):
        w = 2.0 * math.pi / period
        basis = np.column_stack([np.cos(w * x), np.sin(w * x),
                                 np.ones_like(x)])
        coef, _res, _rank, _sv = np.linalg.lstsq(basis, resid, rcond=None)
        err = float(np.sum((resid - basis @ coef) ** 2))
        if best is None or err < best[0]:
            best = (err, float(period), coef)
    if best is None:
        raise HomingError("empty ripple period search range")
    _err, period, coef = best
    amp = float(math.hypot(coef[0], coef[1]))
    # A*cos(wx) + B*sin(wx) == R*cos(wx - phi) with phi = atan2(B, A); the peak
    # nearest x = 0 therefore sits at phi/w.
    phi = math.atan2(float(coef[1]), float(coef[0]))
    peak_x = phi / (2.0 * math.pi / period)
    return period, amp, peak_x


def _wrap(value: float, period: float) -> float:
    """Wrap into (-period/2, +period/2]."""
    half = 0.5 * period
    return (value + half) % period - half


# ==========================================================================
#   The sequence
# ==========================================================================

class Homing:
    """Blocking homing sequence. Construct it with an already-started link.

    Nothing here touches hardware at import time: the link is injected and
    every command goes out inside run().
    """

    def __init__(self, link, progress: Optional[Progress] = None, *,
                 yaw_reference_lsb: Optional[float] = None,
                 sweep_half_deg: float = SWEEP_HALF_DEG,
                 sweep_step_deg: float = SWEEP_STEP_DEG,
                 fast: bool = False,
                 skip_level: bool = False,
                 cancel: Optional[Callable[[], bool]] = None):
        self.link = link
        self._progress = progress
        # Optional predicate polled between commands -- see _cmd(). The caller
        # (app.shutdown()) needs a way to stop a 40 s sequence that is holding
        # the link's io lock, because link.close() runs on the Tk main thread
        # and queues behind it. Checked BETWEEN commands only: aborting one
        # mid-flight would leave the console half-read and the platform moving.
        self._cancel = cancel
        self.fast = bool(fast)
        # Take the CURRENT POSE as the pitch datum instead of driving to true
        # level. For a machine whose accelerometer is not measuring: `level`
        # refuses to set a datum from a sensor reading (0,0,0), which is
        # correct and which also stops the whole run.
        #
        # What this costs is real and is not hidden: the datum is arbitrary,
        # so it is NOT repeatable across boots, and PITCH_LIMIT_DEG is quoted
        # from true level -- with an arbitrary zero the soft stops sit
        # wherever the turret happened to be pointing, which is why
        # pitch_verified comes back False and stays False.
        #
        # What it does NOT cost is tracking. The control loop is closed on
        # PIXELS through the measured Jacobian and never consults the pitch
        # datum; only the travel-limit guard does.
        self.skip_level = bool(skip_level)
        # The my value that DEFINES the yaw datum. The motor field is fixed in
        # the base frame, so "my == this" names a real physical pose -- but
        # only once someone has measured it. Without it, the first run can
        # establish the reference and nothing more; see _solve_datum().
        self.yaw_reference_lsb = yaw_reference_lsb
        self.sweep_half_deg = float(sweep_half_deg)
        self.sweep_step_deg = float(sweep_step_deg)
        self.messages: List[str] = []
        self.samples: List[MagSample] = []
        self._fit: Optional[YawFit] = None
        self._datum_yaw_counter: Optional[float] = None

    # ---------------------------------------------------------------- link

    def _say(self, text: str, frac: float) -> None:
        """Progress the GUI shows AND a line kept in the result."""
        self.messages.append(text)
        self._tick(text, frac)

    def _tick(self, text: str, frac: float) -> None:
        """Progress only. Used for the per-sample sweep chatter, which has to
        reach the GUI (thirty seconds of silence reads as a hang) but must not
        bury the six lines that actually matter in HomingResult.messages."""
        if self._progress is not None:
            self._progress(text, max(0.0, min(1.0, frac)))

    def _cmd(self, line: str, timeout: float) -> str:
        if self._cancel is not None and self._cancel():
            # Raised, not returned: every caller treats a command that did not
            # happen as fatal, and unwinding is what releases the io lock.
            raise HomingAborted("homing aborted before %r: cancel requested"
                                % line)
        reply = self.link.command(line, timeout)
        if reply is None:
            raise HomingError("no reply to %r within %.1f s" % (line, timeout))
        if _ERROR_RE.search(reply):
            raise HomingError("board refused %r:\n%s" % (line, reply.strip()))
        return reply

    def _state(self) -> dict:
        reply = self._cmd("state", T_STATE)
        m = _STATE_RE.search(reply)
        if not m:
            raise HomingError("no STATE json in the reply to 'state':\n%s"
                              % reply.strip())
        return json.loads(m.group(1))

    def _position(self) -> Tuple[float, float]:
        p = self._state()["payload"]
        if "error" in p:
            raise HomingError("firmware cannot report payload position: %s"
                              % p["error"])
        return float(p["pitch"]), float(p["yaw"])

    def _dmove(self, dpitch: float, dyaw: float,
               rate: Optional[float] = None) -> Tuple[float, float]:
        cmd = "dmove %.4f %.4f" % (dpitch, dyaw)
        if rate is not None:
            cmd += " %.3f" % rate
        reply = self._cmd(cmd, T_MOVE)
        # A clamped or endstop-guarded move means the platform is NOT where the
        # caller thinks it is, and every angle computed afterwards would be
        # wrong. Fatal, not a warning.
        if "LIMITED on" in reply:
            raise HomingError("move %r hit a travel limit:\n%s"
                              % (cmd, reply.strip()))
        if "STOPPED EARLY" in reply:
            raise HomingError("move %r stopped on an endstop:\n%s"
                              % (cmd, reply.strip()))
        m = _NOWAT_RE.search(reply)
        if not m:
            raise HomingError("could not read the position back after %r:\n%s"
                              % (cmd, reply.strip()))
        return float(m.group(1)), float(m.group(2))

    def _read_mag(self) -> MagSample:
        parser = getattr(self.link, "parse_mag", None)
        # link.parse_mag may either do its own round trip or parse a reply we
        # already hold. Decide from the signature instead of guessing: getting
        # it wrong surfaces as a TypeError in the middle of a 30 s sweep.
        if parser is not None and not inspect.signature(parser).parameters:
            return _coerce_mag(parser())
        reply = self._cmd("imu mag", T_MAG)
        if parser is not None:
            return _coerce_mag(parser(reply))
        return _coerce_mag(parse_mag_line(reply))

    # -------------------------------------------------- 1. link and idle

    def _confirm_idle(self) -> None:
        self._say("checking link and platform state", 0.01)
        st = self._state()

        vel = st.get("velocity", {})
        if vel.get("running"):
            raise HomingError(
                "platform is in VELOCITY MODE -- homing needs an idle "
                "platform. In velocity mode the step counter is stale (the "
                "distance covered lives in the dead-reckoning accumulator), "
                "so every angle this routine reads back would be a lie. Stop "
                "tracking first.")
        if vel.get("tripped"):
            st = self._clear_vel_latch(st)
            vel = st.get("velocity", {})
            if vel.get("tripped"):
                raise HomingError(
                    "the velocity watchdog is TRIPPED and would not clear. "
                    "`velmode on` then `velmode off` is what resets it; both "
                    "were sent and `state` still reports it set, so the board "
                    "is not accepting commands the way this routine assumes.")

        for name, ax in st.get("axes", {}).items():
            if ax.get("mode") == "vel":
                raise HomingError("axis %s is still in velocity mode" % name)
            if ax.get("fault") is True:
                raise HomingError("axis %s reports a DRIVER FAULT" % name)
            if not ax.get("enabled"):
                # Not fatal: dmove auto-enables. Say so rather than hide it.
                self._say("note: axis %s was disabled; the first move will "
                          "enable it" % name, 0.02)

        # Homing swings the payload across ~25 deg of yaw. A live laser during
        # that sweep paints a 25 deg arc of the room.
        lit = [n for n, on in st.get("outputs", {}).items() if on]
        if any("laser" in n.lower() for n in lit):
            raise HomingError("the LASER output is ON. Homing sweeps the "
                              "payload across the room; disarm first.")

        p = st.get("payload", {})
        if "error" in p:
            raise HomingError("firmware kinematics are unhappy: %s" % p["error"])
        # Host and firmware must agree on the geometry, or every degree in this
        # file is scaled wrong -- and that error would look exactly like
        # backlash, which is the one thing we are trying to measure past.
        if abs(float(p["axis_step_deg"]) - config.AXIS_STEP_DEG) > 1e-4:
            raise HomingError(
                "firmware axis_step_deg %.5f != config.AXIS_STEP_DEG %.5f "
                "(the microstep setting disagrees)"
                % (p["axis_step_deg"], config.AXIS_STEP_DEG))
        if abs(float(p["N"]) - config.DIFFERENTIAL_N) > 1e-3:
            raise HomingError(
                "firmware DIFFERENTIAL_N %.4f != config.DIFFERENTIAL_N %.4f"
                % (p["N"], config.DIFFERENTIAL_N))

        self._say("link ok, platform idle at pitch %+.3f yaw %+.3f"
                  % (p["pitch"], p["yaw"]), 0.04)

    def _clear_vel_latch(self, st: dict) -> dict:
        """Reset the firmware watchdog latch, then re-read `state`.

        WHY THIS IS NOT A REAL ERROR ANY MORE. The latch is
        VelocityLoop._tripped in stepper.py. It is cleared by start() and by
        every accepted `vel`, and -- this is the bug -- NOT by stop(). So
        `velmode off` ends the velocity session and leaves the latch set for
        ever, and the NEXT run reads it here and refuses. A trip at the end of
        the previous run is now the ordinary case, not an anomaly: the host
        stops sending `vel` at shutdown and the 400 ms watchdog fires by
        design, every single time.

        The operator cleared this by hand three times on 2026-09-20 with
        exactly the sequence below, which is the whole argument for doing it
        here instead.

        WHAT IS NOT LOST BY CLEARING IT. The latch's stated meaning is "the
        motors were parked mid-command and the position is unknown". Homing is
        about to establish position from MEASUREMENT -- gravity for pitch, the
        magnetometer for yaw -- so a stale step counter is what it exists to
        replace. The cumulative `trips` counter is untouched and still reports
        the history; only the latch is reset. It is reported, not silenced.
        """
        trips = (st.get("velocity") or {}).get("trips")
        self._say("the velocity watchdog latch is set from the previous run "
                  "(%s trips on this board) -- clearing it" % trips, 0.015)
        try:
            # `velmode on` is the reset: VelocityLoop.start() sets
            # _tripped = False. `off` immediately after leaves the platform
            # idle, which is the state the rest of this routine needs.
            self._cmd("velmode on", 5.0)
            self._cmd("velmode off", 5.0)
        except Exception as exc:                           # noqa: BLE001
            self._say("could not clear the watchdog latch: %s" % exc, 0.015)
            return st
        return self._state()

    # ---------------------------------------------------------- 2. imu cal

    def _gyro_cal(self) -> Tuple[float, float, float]:
        self._say("calibrating gyro zero-rate -- PAYLOAD MUST BE STILL", 0.06)
        reply = self._cmd("imu cal", T_IMU_CAL)
        m = _BIAS_RE.search(reply)
        if not m:
            raise HomingError("'imu cal' did not report a bias:\n%s"
                              % reply.strip())
        bias = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        # Uncalibrated, X fabricates ~6 deg/s here, which integrates into an
        # angle that looks exactly like mechanical drift. Print what was
        # removed so that number is on the record.
        self._say("gyro bias %+.2f %+.2f %+.2f deg/s removed" % bias, 0.14)
        return bias

    # ------------------------------------------------------------ 3. level

    def _level(self) -> float:
        self._say("levelling off gravity (pitch datum) -- this is the slow one",
                  0.16)
        reply = self._cmd("level", T_LEVEL)
        if "datum set at TRUE LEVEL" not in reply:
            raise HomingError("'level' did not set a datum:\n%s" % reply.strip())
        sens = _SENS_RE.search(reply)
        if sens:
            # Free kinematics check: this should land near 1.0. It reads ~1.17
            # on a 2 deg probe here because the lash is taken up on the first
            # move and not on the second -- the same effect the yaw sweep
            # avoids by only ever moving one way.
            self._say("level measured %+.3f deg of tilt per deg commanded "
                      "(1.0 = kinematics agree)" % float(sens.group(1)), 0.30)
        m = re.search(r"settled at\s+" + _NUM + r"\s+deg", reply)
        if m is None:
            m = re.search(r"tilt now\s+" + _NUM + r"\s+deg", reply)
        if m is None:
            raise HomingError("'level' reported no final tilt:\n%s"
                              % reply.strip())
        tilt = float(m.group(1))
        self._say("pitch datum set at true level (%.2f deg residual)" % tilt,
                  0.38)
        return tilt

    # ------------------------------------------------ 4. yaw field sweep

    def _sample_point(self) -> Tuple[MagSample, bool]:
        """Average MAG_SAMPLES_PER_STEP readings taken at ONE fixed pose.

        Averaging is legitimate here only because the platform is stopped: the
        ripple is a function of position, not of time, so a still payload
        cannot smear it.
        """
        raws: List[MagSample] = []
        for i in range(MAG_SAMPLES_PER_STEP):
            if i:
                time.sleep(MAG_SAMPLE_PERIOD_S)
            raws.append(self._read_mag())
        saturated = any(
            r.max_abs_component >= MAG_CLIP_LSB
            or MAG_OVERFLOW_SENTINEL in (r.mx, r.my, r.mz)
            for r in raws)
        mean = MagSample(
            mx=float(np.mean([r.mx for r in raws])),
            my=float(np.mean([r.my for r in raws])),
            mz=float(np.mean([r.mz for r in raws])),
            # Pose comes from the reading, never from what we commanded: if the
            # firmware clamped a move, the commanded value is fiction.
            pitch_deg=raws[0].pitch_deg,
            yaw_deg=raws[0].yaw_deg,
        )
        return mean, saturated

    def _sweep(self) -> bool:
        """Sweep yaw one way across the arc, sampling the motor field.

        Returns True if any raw sample was clipped.
        """
        _pitch0, yaw0 = self._position()
        start = yaw0 - self.sweep_half_deg
        stop = yaw0 + self.sweep_half_deg
        lo, hi = config.YAW_LIMIT_DEG
        if (start - PRELOAD_DEG) < lo + YAW_LIMIT_MARGIN_DEG or \
                stop > hi - YAW_LIMIT_MARGIN_DEG:
            raise HomingError(
                "no room to sweep: yaw %+.2f needs %+.2f..%+.2f but travel is "
                "%+.2f..%+.2f. Unwind the loom and re-run."
                % (yaw0, start - PRELOAD_DEG, stop, lo, hi))

        # Approach the start of the sweep from BELOW, overshooting the lash.
        # Every sample in the fit is then taken with the backlash loaded the
        # same way, in the +yaw direction. Sweeping back and forth would fold
        # BACKLASH_DEG (0.57 deg -- ten times the step resolution) straight
        # into the slope and yield a confidently wrong datum.
        self._say("preloading lash, approaching the sweep start from -yaw",
                  0.40)
        self._dmove(0.0, (start - PRELOAD_DEG) - yaw0, PRELOAD_RATE_DPS)
        self._dmove(0.0, PRELOAD_DEG, PRELOAD_RATE_DPS)

        n_steps = int(round(2.0 * self.sweep_half_deg / self.sweep_step_deg)) + 1
        if n_steps < 8:
            raise HomingError("a %d-point sweep cannot separate the trend from "
                              "the ripple; widen --half or shrink --step"
                              % n_steps)
        self.samples = []
        saturated = False
        for i in range(n_steps):
            if i:
                self._dmove(0.0, self.sweep_step_deg, SWEEP_RATE_DPS)
            # Settle before sampling. Both the accelerometer and the
            # magnetometer ride the ringing payload, and a sample taken
            # mid-ring is not a measurement of anything.
            time.sleep(SWEEP_SETTLE_S)
            point, clipped = self._sample_point()
            saturated = saturated or clipped
            self.samples.append(point)
            self._tick("yaw sweep %2d/%d  yaw %+7.3f  my %+8.1f LSB"
                       % (i + 1, n_steps, point.yaw_deg, point.my),
                       0.40 + 0.48 * (i + 1) / n_steps)
        self._say("swept %d points over %.1f deg of yaw, one direction"
                  % (n_steps, 2.0 * self.sweep_half_deg), 0.88)
        return saturated

    # --------------------------------------------------- 4b. fit the sweep

    def _fit_sweep(self) -> YawFit:
        yaw = np.array([s.yaw_deg for s in self.samples], dtype=float)
        my = np.array([s.my for s in self.samples], dtype=float)
        slope, intercept, r2, rms = _linear_fit(yaw, my)
        resid = my - (slope * yaw + intercept)

        # The ripple is an INDEPENDENT measurement of where the mechanism
        # really is -- it comes from the rotor's own magnets, not from the step
        # counter -- which is what makes its phase a lost-step detector.
        period, amp, peak = _fit_ripple(yaw, resid, RIPPLE_SEARCH_DEG[0],
                                        RIPPLE_SEARCH_DEG[1], RIPPLE_SEARCH_N)
        fit = YawFit(
            slope_lsb_per_deg=slope,
            intercept_lsb=intercept,
            r_squared=r2,
            rms_residual_lsb=rms,
            n_samples=int(yaw.size),
            span_deg=float(yaw.max() - yaw.min()),
            ripple_period_deg=period,
            ripple_amp_lsb=amp,
            ripple_peak_yaw_deg=peak,
        )
        self._say("fit: %.3f LSB/deg (expected %.2f), R2 %.4f, rms %.2f LSB"
                  % (slope, EXPECTED_SLOPE_LSB_PER_DEG, r2, rms), 0.89)
        self._say("ripple: %.3f deg period vs %.3f predicted from the rotor "
                  "(%+.1f%%), amplitude %.2f LSB"
                  % (period, RIPPLE_PERIOD_PREDICTED_DEG,
                     100.0 * fit.ripple_period_error_frac, amp), 0.90)
        return fit

    def _solve_datum(self, fit: YawFit, saturated: bool,
                     yaw_now: float) -> Tuple[float, bool, str, Optional[float]]:
        """Turn the fit into a datum. Returns (yaw_datum, homed, status, ref).

        Every rejection path falls back to the CURRENT pose as yaw zero and
        says so in the status string. Nothing here ever returns homed=True on
        data it does not trust.
        """
        yaws = [s.yaw_deg for s in self.samples]
        lo, hi = min(yaws), max(yaws)

        if saturated:
            return (yaw_now, False, "UNHOMED (magnetometer saturated)", None)
        if fit.r_squared < MIN_FIT_R2:
            return (yaw_now, False,
                    "UNHOMED (fit R2 %.3f < %.2f)" % (fit.r_squared, MIN_FIT_R2),
                    None)
        if abs(fit.slope_lsb_per_deg) < MIN_ABS_SLOPE_LSB_PER_DEG:
            return (yaw_now, False,
                    "UNHOMED (trend only %.2f LSB/deg -- the datum is "
                    "ill-conditioned)" % fit.slope_lsb_per_deg, None)

        if self.yaw_reference_lsb is None:
            # First run on this machine: no stored field value names a pose, so
            # the middle of the swept arc becomes the datum by convention and
            # the fitted my there becomes the reference. That is an ARBITRARY
            # origin today -- honest status is UNHOMED -- but feed the
            # reference back in next boot and the same physical pose comes
            # back, which is the entire point of the exercise.
            center = 0.5 * (lo + hi)
            ref = fit.slope_lsb_per_deg * center + fit.intercept_lsb
            return (center, False,
                    "REFERENCE SET (my=%.1f LSB) -- yaw is arbitrary this run; "
                    "persist the reference to get a real datum" % ref, ref)

        datum = (self.yaw_reference_lsb - fit.intercept_lsb) \
            / fit.slope_lsb_per_deg
        if not (lo - MAX_EXTRAPOLATION_DEG <= datum <= hi + MAX_EXTRAPOLATION_DEG):
            # The motor near-field is not uniform, so this straight line only
            # means anything across the arc it was measured on. Extrapolating
            # it tens of degrees produces a number, not a datum.
            return (yaw_now, False,
                    "UNHOMED (datum %+.2f lies outside the swept arc "
                    "%+.2f..%+.2f -- sweep around it, or re-reference)"
                    % (datum, lo, hi), self.yaw_reference_lsb)
        return (datum, True, "HOMED", self.yaw_reference_lsb)

    # --------------------------------------------- 5/6. preload and datum

    def _approach_and_home(self,
                           yaw_datum: float) -> Tuple[float, float, float, bool]:
        """Approach the datum from one direction, verify level, set home."""
        pitch_now, yaw_now = self._position()
        # Pitch target is 0: `level` zeroed the counters at true level.
        pitch_datum = 0.0

        # Overshoot past the datum on BOTH axes, then come back the positive
        # way. PRELOAD_STEPS is 1.2x the measured lash by design -- check it,
        # because an overshoot smaller than the lash preloads nothing at all.
        if PRELOAD_DEG <= config.BACKLASH_DEG:
            raise HomingError(
                "PRELOAD_STEPS*AXIS_STEP_DEG = %.4f deg does not exceed "
                "BACKLASH_DEG = %.4f -- the preload would not take up the lash"
                % (PRELOAD_DEG, config.BACKLASH_DEG))

        self._say("lash preload: overshooting %.3f deg past the datum "
                  "(backlash is %.2f)" % (PRELOAD_DEG, config.BACKLASH_DEG),
                  0.92)
        self._dmove((pitch_datum - PRELOAD_DEG) - pitch_now,
                    (yaw_datum - PRELOAD_DEG) - yaw_now, PRELOAD_RATE_DPS)
        # The final move is +pitch/+yaw, the same direction the sweep ran, so
        # the teeth sit on the same flanks they sat on during the fit -- and,
        # more importantly, on the same flanks they will sit on next boot.
        pitch_now, yaw_now = self._dmove(PRELOAD_DEG, PRELOAD_DEG,
                                         PRELOAD_RATE_DPS)

        # The preload moved pitch off the level datum and back. The step
        # counter insists it returned; gravity is the only witness that it
        # actually did, so ask gravity.
        time.sleep(SWEEP_SETTLE_S)
        reply = self._cmd("imu", T_MAG)
        m = _TILT_RE.search(reply)
        if not m:
            raise HomingError("could not read the tilt back after the "
                              "preload:\n%s" % reply.strip())
        tilt = float(m.group(1))
        pitch_ok = abs(tilt) <= PITCH_VERIFY_TOL_DEG
        if self.skip_level:
            # There is no gravity datum to verify against, and the reading
            # that would "confirm" it is the one we do not trust: a standby
            # ADXL345 reads exactly -0.00, which sails through the tolerance
            # above and prints "pitch verified against gravity: +0.00 deg".
            # Saying nothing was verified is the only honest answer here.
            pitch_ok = False
            self._say("pitch NOT verified: levelling was skipped, so this "
                      "datum is the pose the turret was left in. Reported "
                      "tilt %+.2f deg is not evidence -- a standby "
                      "accelerometer reads exactly 0.00." % tilt, 0.95)
        elif not pitch_ok:
            self._say("WARNING: pitch reads %+.2f deg after the preload, past "
                      "the %.2f deg tolerance -- the pitch datum is suspect"
                      % (tilt, PITCH_VERIFY_TOL_DEG), 0.95)
        else:
            self._say("pitch verified against gravity: %+.2f deg" % tilt, 0.95)

        # dzero BEFORE sethome so the datum is literally (0, 0): the travel
        # limits in config.py are quoted from the datum, and a datum you cannot
        # name zero is not much of a datum.
        self._cmd("dzero", T_HOME)
        reply = self._cmd("sethome", T_HOME)
        m = _HOME_RE.search(reply)
        if not m:
            raise HomingError("'sethome' did not confirm a datum:\n%s"
                              % reply.strip())
        return float(m.group(1)), float(m.group(2)), tilt, pitch_ok

    # -------------------------------------------------------------- run

    def run(self) -> HomingResult:
        t0 = time.monotonic()
        self.messages = []

        self._confirm_idle()
        bias = self._gyro_cal()
        if self.skip_level:
            self._say("SKIPPING the gravity datum: taking the CURRENT POSE as "
                      "pitch zero. This datum is arbitrary and does not "
                      "survive a reboot, and the pitch soft stops are now "
                      "relative to it. Tracking is unaffected -- the control "
                      "loop closes on pixels, not on this.", 0.30)
        else:
            self._level()

        if self.fast:
            # FAST: pitch datum from gravity, yaw datum from the current pose.
            #
            # This skips the 33-point magnetometer sweep, which is ~20 s of the
            # ~40 s run. It costs NOTHING WE ARE USING: without a persisted
            # yaw_reference_lsb the sweep's own _solve_datum() falls back to
            # "current pose = yaw zero" anyway and reports the datum as
            # arbitrary. Doing that directly is the same answer, sooner.
            #
            # What it DOES cost: the ripple-phase lost-step check, and any hope
            # of absolute yaw. So it is wrong for a run that needs to come back
            # to a pose established on a previous boot -- use a full home and
            # --save-ref for that. Pitch is unaffected: gravity still sets it.
            if self.yaw_reference_lsb is not None:
                self._say("fast home: IGNORING the stored yaw reference. A "
                          "real yaw datum needs the sweep; run without --fast-home.",
                          0.90)
            _pitch_now, yaw_now = self._position()
            pitch_home, yaw_home, tilt, pitch_ok = self._approach_and_home(yaw_now)
            took = time.monotonic() - t0
            self._say("fast home complete in %.1f s: pitch from gravity, yaw "
                      "arbitrary (current pose)" % took, 1.0)
            return HomingResult(
                ok=True,
                yaw_homed=False,
                yaw_status="UNHOMED (fast home: magnetometer sweep skipped)",
                datum_pitch_deg=pitch_home,
                datum_yaw_deg=yaw_home,
                datum_yaw_counter_deg=yaw_now,
                pitch_tilt_deg=tilt,
                pitch_datum_ok=pitch_ok,
                gyro_bias_dps=bias,
                mag_saturated=False,
                # No sweep ran, so there are no field samples and no fit. These
                # are reported as zero/None rather than omitted, so a consumer
                # reading them gets "nothing was measured" instead of a stale
                # value from a previous run.
                max_abs_component_lsb=0.0,
                field_magnitude_lsb=0.0,
                yaw_reference_lsb=None,
                yaw_fit=None,
                elapsed_s=took,
                messages=list(self.messages),
                samples=[],
            )

        saturated = self._sweep()

        field_mag = float(np.mean([s.magnitude for s in self.samples]))
        peak_component = float(max(s.max_abs_component for s in self.samples))
        if saturated:
            # Loud, and fatal to the yaw datum -- but not to the whole run: the
            # pitch datum came from gravity and is untouched by this.
            self._say("MAGNETOMETER SATURATED: a component reached %.0f of "
                      "%.0f LSB full scale. Reduce the HMC5883L gain (CRB, "
                      "register 0x01) in the firmware and re-home. Refusing to "
                      "fit clipped data -- it would give a confident wrong "
                      "datum." % (peak_component, MAG_FULL_SCALE_LSB), 0.90)
        elif field_mag > MAG_HEADROOM_WARN_FRAC * MAG_FULL_SCALE_LSB:
            self._say("note: |B| = %.0f LSB is %.0f%% of the %.0f LSB full "
                      "scale -- nothing clipped this run, but a small change "
                      "in pose will"
                      % (field_mag, 100.0 * field_mag / MAG_FULL_SCALE_LSB,
                         MAG_FULL_SCALE_LSB), 0.90)

        fit = self._fit_sweep()
        self._fit = fit
        if abs(fit.slope_lsb_per_deg) > \
                SLOPE_SANITY_FACTOR * abs(EXPECTED_SLOPE_LSB_PER_DEG):
            self._say("note: the trend is %.2f LSB/deg against the %.2f "
                      "measured previously -- the IMU or a motor has moved "
                      "relative to the payload"
                      % (fit.slope_lsb_per_deg, EXPECTED_SLOPE_LSB_PER_DEG),
                      0.90)

        _pitch_now, yaw_now = self._position()
        yaw_datum, yaw_homed, status, ref = self._solve_datum(
            fit, saturated, yaw_now)
        if yaw_homed:
            self._say("yaw datum at %+.3f deg, from the motor field"
                      % yaw_datum, 0.91)
        else:
            self._say("YAW %s -- falling back to the current pose as yaw zero. "
                      "Absolute yaw is NOT trustworthy this run." % status,
                      0.91)

        pitch_home, yaw_home, tilt, pitch_ok = self._approach_and_home(yaw_datum)

        # Re-express the ripple peak in post-dzero coordinates so that
        # lost_step_check speaks the same frame the rest of the stack uses
        # from here on.
        self._datum_yaw_counter = yaw_datum
        fit.ripple_peak_yaw_deg = _wrap(fit.ripple_peak_yaw_deg - yaw_datum,
                                        fit.ripple_period_deg)

        result = HomingResult(
            ok=True,
            yaw_homed=yaw_homed,
            yaw_status=status,
            datum_pitch_deg=pitch_home,
            datum_yaw_deg=yaw_home,
            datum_yaw_counter_deg=yaw_datum,
            pitch_tilt_deg=tilt,
            pitch_datum_ok=pitch_ok,
            gyro_bias_dps=bias,
            mag_saturated=saturated,
            max_abs_component_lsb=peak_component,
            field_magnitude_lsb=field_mag,
            yaw_reference_lsb=ref,
            yaw_fit=fit,
            elapsed_s=time.monotonic() - t0,
            messages=list(self.messages),
            samples=list(self.samples),
        )
        self._say("homed: pitch %+.3f yaw %+.3f (%s) in %.1f s"
                  % (pitch_home, yaw_home, status, result.elapsed_s), 1.0)
        return result

    # ------------------------------------------------------ health check

    def lost_step_check(self, points: int = LOST_STEP_POINTS) -> LostStepCheck:
        """Compare the rotor ripple's phase against the commanded position.

        The ripple comes from the rotor's own magnets, so its phase says where
        the MECHANISM is. The step counter says where the firmware THINKS it
        is. A disagreement is a lost step, measured directly -- no camera, no
        target, no endstop.

        This MOVES the platform (one ripple period of yaw, ~3.1 deg, plus the
        lash preload) and takes a couple of seconds. It is a health check for
        an idle machine, not a servo input: the magnetometer caps at 75 Hz.
        """
        if self._fit is None:
            raise HomingError("no ripple model -- run() has not completed")
        fit = self._fit
        if fit.ripple_amp_lsb < MIN_RIPPLE_AMP_LSB:
            raise HomingError(
                "the ripple amplitude is only %.2f LSB; its phase is noise and "
                "cannot detect anything" % fit.ripple_amp_lsb)
        st = self._state()
        if st.get("velocity", {}).get("running"):
            raise HomingError("cannot run the lost-step check while the "
                              "platform is in velocity mode")

        _pitch0, yaw0 = self._position()
        period = fit.ripple_period_deg
        step = period / (points - 1)

        # Same approach direction as the sweep that built the model, for the
        # same reason: the phase has to be read with the lash on the same
        # flanks it was fitted on, or the answer is 0.57 deg of nothing.
        self._dmove(0.0, -(0.5 * period + PRELOAD_DEG), PRELOAD_RATE_DPS)
        self._dmove(0.0, PRELOAD_DEG, PRELOAD_RATE_DPS)

        yaws: List[float] = []
        mys: List[float] = []
        for i in range(points):
            if i:
                self._dmove(0.0, step, SWEEP_RATE_DPS)
            time.sleep(SWEEP_SETTLE_S)
            point, clipped = self._sample_point()
            if clipped:
                raise HomingError("the magnetometer clipped during the "
                                  "lost-step check; the phase would be wrong")
            yaws.append(point.yaw_deg)
            mys.append(point.my)

        x = np.array(yaws)
        y = np.array(mys)
        w = 2.0 * math.pi / period
        # Fit the sine AND a local line together: across one period the coarse
        # trend is a straight line, and removing it in a separate pass would
        # leak straight into the phase.
        basis = np.column_stack([np.cos(w * x), np.sin(w * x), x,
                                 np.ones_like(x)])
        coef, _r, _k, _s = np.linalg.lstsq(basis, y, rcond=None)
        amp = float(math.hypot(coef[0], coef[1]))
        peak_now = _wrap(math.atan2(float(coef[1]), float(coef[0])) / w, period)
        slip = _wrap(peak_now - fit.ripple_peak_yaw_deg, period)

        # Put the platform back where the caller left it, approached the same
        # way, so that having run the check does not itself change the lash
        # state the tracker is about to inherit.
        _p, y_end = self._position()
        self._dmove(0.0, (yaw0 - PRELOAD_DEG) - y_end, PRELOAD_RATE_DPS)
        self._dmove(0.0, PRELOAD_DEG, PRELOAD_RATE_DPS)

        ok = amp >= MIN_RIPPLE_AMP_LSB and abs(slip) <= LOST_STEP_TOL_DEG
        if amp < MIN_RIPPLE_AMP_LSB:
            msg = ("ripple amplitude collapsed to %.2f LSB -- no usable phase"
                   % amp)
        elif ok:
            msg = ("rotor phase agrees with the step counter to %+.3f deg "
                   "(tol %.2f)" % (slip, LOST_STEP_TOL_DEG))
        else:
            msg = ("LOST STEPS: the rotor phase is %+.3f deg from the "
                   "commanded position (tol %.2f; one motor full step is "
                   "%.2f deg of payload). Ambiguous modulo %.2f deg."
                   % (slip, LOST_STEP_TOL_DEG,
                      MOTOR_ROTOR_PERIOD_DEG / 4.0 / config.DIFFERENTIAL_N,
                      period))
        return LostStepCheck(ok=ok, slip_deg=slip, ambiguity_deg=period,
                             amplitude_lsb=amp,
                             tolerance_deg=LOST_STEP_TOL_DEG, message=msg)


def home(link, progress: Optional[Progress] = None,
         yaw_reference_lsb: Optional[float] = None) -> HomingResult:
    """One homing run against an already-started link. Convenience wrapper."""
    return Homing(link, progress, yaw_reference_lsb=yaw_reference_lsb).run()


# ==========================================================================
#   Manual run against a real board
# ==========================================================================

def _print_progress(text: str, frac: float) -> None:
    print("[%3d%%] %s" % (round(100 * frac), text))
    sys.stdout.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Home the turret against a real board.")
    ap.add_argument("--ref", type=Path, default=None,
                    help="JSON file holding the yaw reference my value. Read "
                         "if it exists; only written with --save-ref.")
    ap.add_argument("--save-ref", action="store_true",
                    help="write the reference this run established to --ref")
    ap.add_argument("--half", type=float, default=SWEEP_HALF_DEG,
                    help="sweep half-range, deg (default %.1f)" % SWEEP_HALF_DEG)
    ap.add_argument("--step", type=float, default=SWEEP_STEP_DEG,
                    help="sweep step, deg (default %.2f)" % SWEEP_STEP_DEG)
    ap.add_argument("--lost-step", action="store_true",
                    help="run the ripple-phase lost-step check afterwards")
    args = ap.parse_args(argv)

    reference = None
    if args.ref is not None and args.ref.exists():
        reference = float(json.loads(args.ref.read_text())["yaw_reference_lsb"])
        print("yaw reference my = %.2f LSB (from %s)" % (reference, args.ref))
    else:
        print("no yaw reference supplied -- this run can only ESTABLISH one")

    # Imported here, not at module scope: importing this module must not
    # require a board, and link.py opens the port when it is constructed.
    from turret_host.link import TurretLink

    link = TurretLink()
    starter = getattr(link, "start", None)
    if starter is not None:
        starter()
    try:
        seq = Homing(link, _print_progress, yaw_reference_lsb=reference,
                     sweep_half_deg=args.half, sweep_step_deg=args.step)
        result = seq.run()
        print("")
        print(result.summary())
        if args.lost_step:
            print("")
            print(seq.lost_step_check().message)
        if args.save_ref:
            if args.ref is None:
                raise SystemExit("--save-ref needs --ref PATH")
            if result.yaw_reference_lsb is None:
                raise SystemExit("this run produced no reference to save")
            args.ref.write_text(json.dumps(
                {"yaw_reference_lsb": result.yaw_reference_lsb,
                 "slope_lsb_per_deg": result.yaw_fit.slope_lsb_per_deg,
                 "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
            print("reference written to %s" % args.ref)
        return 0 if result.ok else 1
    finally:
        # close(), NOT stop(). On TurretLink those are different things:
        # stop() means "zero the rates and leave the servo path", which on an
        # idle board queues a `vel 0 0` that ENERGISES the coils and enters
        # velocity mode on the way out -- the opposite of what a homing run
        # wants, and it leaves the serial handle leaked besides.
        link.close()


if __name__ == "__main__":
    sys.exit(main())
