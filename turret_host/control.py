"""Image Jacobian, control law, travel limits and the laser interlock.

This module turns a filtered pixel estimate into signed MOTOR rates for the
firmware's `vel <A> <B>`, and decides -- separately and conservatively --
whether the laser may be on.

Three things here are load-bearing and look wrong until you know why:

1.  **Feedforward is the whole game.**  ``omega = J_inv @ (K*e + v_px)``.  With
    ~60 ms of latency the proportional term alone is stable only to about 2 Hz,
    which is nowhere near a hand.  The velocity term commands the turret at the
    target's *angular rate* continuously, so the P term only has to clean up
    the residual.  Deleting the second term "to simplify" reproduces a loop
    that visibly lags every hand movement.

2.  **An uncalibrated Jacobian refuses to command motion.**  ``J`` absorbs the
    belt ratio, the differential, the joint order, the 11 deg axis tilt, the
    focal length and every sign.  A guessed ``J`` with one column's sign wrong
    is a turret that accelerates away from the target.  ``config.J_PX_PER_STEP``
    is ``None`` until ``calibrate.py`` has run, and this module raises rather
    than inventing one.

3.  **The goal pixel is fitted, not derived from CAD.**  The laser sits 30 mm
    from the narrow camera, so where the beam lands in the image moves with
    range: ``g(R) = g_inf + c/R`` per axis.  The narrow camera is mounted
    rotated 90 deg, so that parallax term should come out on the goal ROW and
    not the column -- but this module does not hard-code that.  The fit decides
    it empirically, and the CAD prediction is used only as a sanity print.  The
    rotation sign is unverified in config.py, and a hard-coded axis would be a
    coin flip.

Nothing here touches hardware, so the module imports on a bare machine.
"""
from __future__ import annotations

# --- sys.path fix, and it MUST come before every other import --------------
# Running this file directly (`python turret_host/control.py`) puts
# turret_host/ itself at sys.path[0], where our types.py SHADOWS the stdlib
# `types` module -- and the interpreter then dies inside the very next
# `import json`, with a circular-import error that names our file and looks
# like our bug.  So: drop the script directory and put the project root on
# instead.  Done with string surgery rather than os.path because importing
# os/pathlib first is what triggers the collision.  turret_host has no
# __init__.py and resolves as a namespace package from the root.
import sys

_HERE = __file__.replace("\\", "/").rsplit("/", 1)[0]      # absolute since 3.9
_ROOT_STR = _HERE.rsplit("/", 1)[0]
for _p in list(sys.path):
    if _p and _p.replace("\\", "/").rstrip("/").lower() == _HERE.lower():
        sys.path.remove(_p)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

import json                                                      # noqa: E402
import math                                                      # noqa: E402
import time                                                      # noqa: E402
from datetime import datetime, timezone                          # noqa: E402
from pathlib import Path                                         # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple         # noqa: E402

import numpy as np                                               # noqa: E402

from turret_host import config                                   # noqa: E402
from turret_host.types import (                                  # noqa: E402
    ControlOutput,
    Detection,
    LaserState,
    TrackEstimate,
    TrackState,
)

# --------------------------------------------------------------------------
#   Module-local constants
#
#   These are NOT duplicates of config.py values -- they are policy internal to
#   the control law.  Anything the rest of the stack shares lives in config.py.
# --------------------------------------------------------------------------
# INTEGRATION: calibrate.py is the writer and it saves into turret_host/
# calibration/.  These are the reader's defaults, so they follow the writer --
# pointing them at the parent directory meant control.py looked where nothing
# is ever written and came up uncalibrated on a calibrated machine.
# calibrate.py's JSON is a deliberate superset: it writes `j_px_per_step`
# beside `J`, and `u_inf`/`c_u`/`v_inf`/`c_v` beside `g_inf`/`c`, so the two
# loaders read the same file unmodified.
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
JACOBIAN_PATH = CALIBRATION_DIR / "jacobian.json"
GOAL_MODEL_PATH = CALIBRATION_DIR / "goal_pixel.json"

# A `vel` older than this means the Pico is coasting on a stale rate.  The
# firmware watchdog is 400 ms; the interlock is deliberately 4x tighter,
# because "the beam is still moving on last frame's guess" is not a state in
# which we want to be emitting light.
VEL_FRESH_S = 0.100

# How long the control law keeps full authority on a target no detector has
# confirmed since, and how long it then takes to taper to zero. Taken from the
# coast constants config already defines for the TRACKER, because they describe
# the same physical judgement: how long a measurement stays evidence about now.
# Applied to the COMMAND as well as the estimate -- an estimate that decays in
# velocity still runs away in position, which is how a centred drone turned
# into a 200 deg/s slew in the opposite direction.
COAST_HOLD_S = config.COAST_HOLD_MS / 1000.0
COAST_DECAY_S = config.COAST_DECAY_MS / 1000.0

# The laser duty cap from config.LASER_MAX_ON_MS needs a matching cool-down,
# or the interlock chatters on/off at the cap instead of actually resting.
LASER_COOLDOWN_S = config.LASER_MAX_ON_MS / 1000.0

# Soft travel margin: start derating this far from a hard stop.  Larger than
# BACKLASH_DEG (0.57) so lash cannot carry us through the soft band in one
# reversal.
LIMIT_MARGIN_DEG = 3.0

# The tracker owns coast decay.  This only stops a late/stale estimate from
# being extrapolated on an absurd lever arm: at 1 m/s and 3 m a target moves
# ~470 px/s, so a 1 s extrapolation would aim half a room away.
MAX_EXTRAPOLATION_S = 0.25

# Geometric expectation for one motor microstep, used ONLY to sanity-check a
# loaded J: 0.04875 payload deg (belt ratio already folded in) x 1400 px/rad.
EXPECTED_PX_PER_STEP = config.NARROW_F_PX * math.radians(config.AXIS_STEP_DEG)

# A J whose two columns are nearly parallel means both motors push the target
# the same way in the image -- geometrically impossible on this head, so it
# means the calibration moved something other than what it thought.
MAX_J_CONDITION = 50.0

# Expected parallax coefficient from CAD, px*m: f_px * baseline_m.  Printed as
# a diagnostic against the fitted c; never used to override or replace it.
EXPECTED_PARALLAX_PX_M = config.NARROW_F_PX * abs(config.LASER_TO_NARROW_MM) / 1000.0

# Sampling pitch when measuring face distance along the beam path.  8 px of
# quantisation against a 120 px inhibit margin is noise.
_BEAM_SAMPLE_PX = 8.0

FRAME_CENTRE = ((config.NARROW_SIZE[0] - 1) / 2.0,
                (config.NARROW_SIZE[1] - 1) / 2.0)


class NotCalibratedError(RuntimeError):
    """Raised when the loop asks for motion the calibration cannot justify."""


_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    """Print once per process.  A per-frame warning is a warning nobody reads."""
    if key not in _warned:
        _warned.add(key)
        print(f"[control] {message}", file=sys.stderr)


# ==========================================================================
#   IMAGE JACOBIAN
# ==========================================================================
class Jacobian:
    """The 2x2 image Jacobian, px of target motion per MOTOR step.

        [du]   [ dU/dA  dU/dB ] [ a ]
        [dv] = [ dV/dA  dV/dB ] [ b ]

    Rows are narrow-frame image axes (u = column, v = row) in the UNROTATED
    capture buffer -- frames are never rotated for processing, the geometry
    carries the rotation.  Columns are motor A (firmware `pan`) and motor B
    (firmware `tilt`).  Because the head is a differential, neither column is
    expected to be axis-aligned in the image, and that is fine: the 2x2 absorbs
    it, which is why the loop tolerates J being 20% wrong.

    `is_calibrated` is False until a real measurement is loaded.  Every method
    that would produce motion raises NotCalibratedError while it is False.
    """

    __slots__ = ("_j", "_j_inv", "_calibrated", "source", "meta")

    def __init__(self, j=None, source: str = "uncalibrated",
                 meta: Optional[dict] = None):
        self.source = source
        self.meta: dict = dict(meta or {})
        if j is None:
            self._j = None
            self._j_inv = None
            self._calibrated = False
        else:
            self._j = self._validate(j)
            self._j_inv = self._invert(self._j)
            self._calibrated = True

    # -- construction ------------------------------------------------------
    @staticmethod
    def _validate(j) -> np.ndarray:
        arr = np.asarray(j, dtype=float)
        if arr.shape != (2, 2):
            raise ValueError(f"J must be 2x2, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"J contains non-finite entries: {arr.tolist()}")
        # Column norm = px moved per step by that motor alone.  Accept a decade
        # either side of the geometric prediction; outside that, the
        # calibration measured the wrong thing (wrong motor, wrong units,
        # target lost mid-move) and we want to hear about it now, not as a
        # runaway later.
        for col in (0, 1):
            norm = float(np.linalg.norm(arr[:, col]))
            if not (0.1 * EXPECTED_PX_PER_STEP <= norm <= 10.0 * EXPECTED_PX_PER_STEP):
                raise ValueError(
                    f"J column {col} moves {norm:.4f} px/step; geometry predicts "
                    f"~{EXPECTED_PX_PER_STEP:.3f} px/step. Refusing an implausible "
                    f"calibration -- re-run calibrate.py."
                )
        return arr

    @staticmethod
    def _invert(arr: np.ndarray) -> np.ndarray:
        cond = float(np.linalg.cond(arr))
        if not math.isfinite(cond) or cond > MAX_J_CONDITION:
            raise ValueError(
                f"J is near-singular (condition {cond:.1f} > {MAX_J_CONDITION}): "
                f"both motors move the target along nearly the same image "
                f"direction, which this head cannot do. {arr.tolist()}"
            )
        return np.linalg.inv(arr)

    @classmethod
    def from_config(cls) -> "Jacobian":
        """config.J_PX_PER_STEP, or an uncalibrated instance if it is None."""
        if config.J_PX_PER_STEP is None:
            return cls()
        return cls(config.J_PX_PER_STEP, source="config.J_PX_PER_STEP")

    @classmethod
    def load(cls, path: Path = JACOBIAN_PATH) -> "Jacobian":
        """Load a calibration JSON.  Missing file -> uncalibrated, not an error.

        A missing file is the normal state on a fresh machine.  A *corrupt* or
        implausible file is an error and propagates -- silently falling back to
        "uncalibrated" would hide a broken calibration run.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        if "j_px_per_step" not in blob:
            raise ValueError(f"{path} has no 'j_px_per_step' key: {sorted(blob)}")
        meta = {k: v for k, v in blob.items() if k != "j_px_per_step"}
        return cls(blob["j_px_per_step"], source=str(path), meta=meta)

    @classmethod
    def auto(cls, path: Path = JACOBIAN_PATH) -> "Jacobian":
        """config first (an explicit override), then the calibration file."""
        j = cls.from_config()
        if j.is_calibrated:
            return j
        return cls.load(path)

    def save(self, path: Path = JACOBIAN_PATH, **extra) -> Path:
        if not self._calibrated:
            raise NotCalibratedError("refusing to save an uncalibrated Jacobian")
        path = Path(path)
        blob = dict(self.meta)
        blob.update(extra)
        blob["j_px_per_step"] = self._j.tolist()
        blob["saved_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        blob["expected_px_per_step"] = EXPECTED_PX_PER_STEP
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True)
        return path

    # -- access ------------------------------------------------------------
    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def J(self) -> np.ndarray:
        self._require()
        return self._j.copy()

    @property
    def J_inv(self) -> np.ndarray:
        self._require()
        return self._j_inv.copy()

    @property
    def condition(self) -> float:
        self._require()
        return float(np.linalg.cond(self._j))

    def _require(self) -> None:
        if not self._calibrated:
            raise NotCalibratedError(
                "image Jacobian is not calibrated (config.J_PX_PER_STEP is None "
                "and no jacobian.json was loaded). Run calibrate.py. The loop "
                "will not command motion on a guessed J -- a guessed column sign "
                "is a turret that runs away from the target."
            )

    def motor_rates(self, pixel_rate) -> np.ndarray:
        """Desired px/s of target motion in the image -> motor steps/s."""
        self._require()
        return self._j_inv @ np.asarray(pixel_rate, dtype=float).reshape(2)

    def pixel_rates(self, motor_rate) -> np.ndarray:
        """Forward map: motor steps/s -> px/s.  Used by tests and the GUI."""
        self._require()
        return self._j @ np.asarray(motor_rate, dtype=float).reshape(2)

    def __repr__(self) -> str:
        if not self._calibrated:
            return "<Jacobian UNCALIBRATED>"
        return (f"<Jacobian {self._j.tolist()} px/step cond={self.condition:.2f} "
                f"src={self.source!r}>")


# ==========================================================================
#   GOAL PIXEL  --  g(R) = g_inf + c/R, per axis
# ==========================================================================
class GoalModel:
    """Where the beam lands in the narrow frame, as a function of range.

    The laser is 30 mm from the narrow camera, so the beam's image position is
    range dependent: a fixed bore-sight pixel is only right at one range.  The
    model is the textbook parallax form, fitted per axis:

        u_goal(R) = u_inf + c_u / R
        v_goal(R) = v_inf + c_v / R

    Both axes get a free c.  The narrow camera is mounted rotated 90 deg, so
    the expectation is |c_v| >> |c_u| -- the correction landing on the ROW.
    That expectation is NOT hard-coded: config flags the rotation sign as
    unverified, so the fit decides and `parallax_axis` merely reports what it
    decided.  Hard-coding the axis from CAD would silently aim 14 px off in the
    wrong direction if the mount is mirrored from the drawing.
    """

    __slots__ = ("u_inf", "c_u", "v_inf", "c_v", "_calibrated", "source", "meta")

    def __init__(self, u_inf=None, c_u=None, v_inf=None, c_v=None,
                 source: str = "uncalibrated", meta: Optional[dict] = None):
        self.meta: dict = dict(meta or {})
        self.source = source
        params = (u_inf, c_u, v_inf, c_v)
        if all(p is None for p in params):
            self.u_inf, self.c_u = FRAME_CENTRE[0], 0.0
            self.v_inf, self.c_v = FRAME_CENTRE[1], 0.0
            self._calibrated = False
            return
        if any(p is None for p in params):
            raise ValueError("GoalModel needs all four of u_inf, c_u, v_inf, c_v "
                             f"or none of them; got {params}")
        vals = [float(p) for p in params]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"GoalModel parameters must be finite: {vals}")
        self.u_inf, self.c_u, self.v_inf, self.c_v = vals
        self._calibrated = True

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: Path = GOAL_MODEL_PATH) -> "GoalModel":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        missing = [k for k in ("u_inf", "c_u", "v_inf", "c_v") if k not in blob]
        if missing:
            raise ValueError(f"{path} is missing goal-fit keys {missing}")
        meta = {k: v for k, v in blob.items()
                if k not in ("u_inf", "c_u", "v_inf", "c_v")}
        return cls(blob["u_inf"], blob["c_u"], blob["v_inf"], blob["c_v"],
                   source=str(path), meta=meta)

    def save(self, path: Path = GOAL_MODEL_PATH, **extra) -> Path:
        if not self._calibrated:
            raise NotCalibratedError("refusing to save an uncalibrated GoalModel")
        path = Path(path)
        blob = dict(self.meta)
        blob.update(extra)
        blob.update({"u_inf": self.u_inf, "c_u": self.c_u,
                     "v_inf": self.v_inf, "c_v": self.c_v,
                     "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "expected_parallax_px_m": EXPECTED_PARALLAX_PX_M,
                     "parallax_axis": self.parallax_axis})
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True)
        return path

    # -- use ---------------------------------------------------------------
    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def goal_pixel(self, range_m: Optional[float] = None) -> Tuple[float, float]:
        """Goal pixel at `range_m`.  Uncalibrated -> frame centre, and says so.

        Frame centre is a defensible fallback for *aiming the camera* -- the
        loop still converges, it just converges on the wrong pixel by the
        parallax offset (~14 px at 3 m, against a ~132 px target, so it still
        hits the body).  It is NOT a fallback that may fire the laser: the
        interlock is what keeps that honest, via MAX_ERROR_TO_FIRE_PX against
        an aim point the operator can see is uncalibrated in the GUI.
        """
        if range_m is None:
            range_m = config.ASSUMED_RANGE_M
        if not self._calibrated:
            _warn_once("goal-uncalibrated",
                       "goal pixel UNCALIBRATED -- using frame centre "
                       f"{FRAME_CENTRE}. Run calibrate.py goal fit.")
            return FRAME_CENTRE
        # Clamp range before dividing: a bad range estimate near zero turns the
        # 1/R term into a goal pixel in the next county.  RANGE_LIMITS_M is the
        # envelope the fit was taken over; extrapolating outside it is guessing.
        lo, hi = config.RANGE_LIMITS_M
        r = min(max(float(range_m), lo), hi)
        return (self.u_inf + self.c_u / r, self.v_inf + self.c_v / r)

    @property
    def source_text(self) -> str:
        """Human-readable provenance for the GUI banner."""
        if not self._calibrated:
            return "frame centre (GOAL UNCALIBRATED)"
        return f"g(R) fit [{self.parallax_axis}] from {self.source}"

    @property
    def parallax_axis(self) -> str:
        """Which axis the fitted parallax actually landed on.  Diagnostic only."""
        if not self._calibrated:
            return "unknown"
        if abs(self.c_v) >= abs(self.c_u):
            return "row (v)"
        return "column (u)"

    def cad_check(self) -> str:
        """Compare the fitted |c| with the CAD prediction.  Report, never correct.

        A fit that agrees with 42 px*m is a fit that measured parallax.  One
        that disagrees by 3x measured something else (drift, a mis-set range,
        a moving target) -- but the fit still wins, because CAD does not know
        which way the camera was actually screwed on.
        """
        if not self._calibrated:
            return "goal model uncalibrated; CAD predicts |c| = " \
                   f"{EXPECTED_PARALLAX_PX_M:.1f} px*m on one axis"
        mag = math.hypot(self.c_u, self.c_v)
        ratio = mag / EXPECTED_PARALLAX_PX_M if EXPECTED_PARALLAX_PX_M else float("inf")
        verdict = "consistent" if 0.5 <= ratio <= 2.0 else "DISAGREES with CAD"
        return (f"|c| = {mag:.1f} px*m on {self.parallax_axis} vs CAD "
                f"{EXPECTED_PARALLAX_PX_M:.1f} ({ratio:.2f}x) -- {verdict}")

    def __repr__(self) -> str:
        if not self._calibrated:
            return "<GoalModel UNCALIBRATED -> frame centre>"
        return (f"<GoalModel u={self.u_inf:.1f}{self.c_u:+.1f}/R "
                f"v={self.v_inf:.1f}{self.c_v:+.1f}/R>")


_DEFAULT_GOAL_MODEL: Optional[GoalModel] = None


def goal_pixel(range_m: Optional[float] = None,
               model: Optional[GoalModel] = None) -> Tuple[float, float]:
    """Module-level convenience: goal pixel at a range.

    Uses a lazily loaded default GoalModel so the GUI and calibrate.py can ask
    without threading an object through.  The control loop should hold its own
    model instance instead, so a recalibration takes effect without a restart.
    """
    global _DEFAULT_GOAL_MODEL
    if model is None:
        if _DEFAULT_GOAL_MODEL is None:
            _DEFAULT_GOAL_MODEL = GoalModel.load()
        model = _DEFAULT_GOAL_MODEL
    return model.goal_pixel(range_m)


# ==========================================================================
#   TRAVEL LIMITS
# ==========================================================================
class TravelLimits:
    """Soft-stop guard in payload angle space.

    The firmware owns the hard stops; this keeps the loop from driving into
    them and stalling mid-track.  It can only ever REMOVE commanded motion.

    Mapping motor rates to payload axis rates needs the differential mix, which
    is not in config.py.  types.ControlOutput names motor A `pan` and motor B
    `tilt`, but the head is a differential -- both motors move both axes -- so
    the conventional sum/difference form is used below (pitch is the SUM, yaw
    the DIFFERENCE; see DEFAULT_AXIS_MIX, and note that getting those two rows
    the wrong way round does NOT merely stall an axis -- it makes one soft stop
    unreachable and drives that axis into its mechanical stop).  If homing
    measures a different mix, call set_axis_mix().

    Pose comes from telemetry (LinkStatus.pitch_deg / .yaw_deg), which is NOT
    polled while servoing -- two requests contending for one serial link
    roughly doubles feedback latency and the watchdog parks the motors in the
    gap.  So the pose here is whatever homing and the idle link last reported,
    and when no pose has ever been reported the guard passes motion through and
    says `pose_known` is False.  The alternative -- refusing to move without a
    fresh pose -- would mean never moving at all, given we deliberately do not
    ask.
    """

    # rows: [pitch_deg/s, yaw_deg/s]; cols: [motor A, motor B], x AXIS_STEP_DEG
    #
    # PITCH IS THE SUM AND YAW IS THE DIFFERENCE. These two rows were
    # transposed, which made the pitch soft stop unreachable BY CONSTRUCTION:
    # real pitch landed in the dead-reckoned "yaw" slot and was derated against
    # the +-225 deg yaw limit that pitch can never reach, while the
    # dead-reckoned pitch sat pinned at 0.00 and the tilt axis drove into its
    # -90 deg mechanical stop at full rate with blocked = (). That destroys the
    # datum the whole homing sequence exists to establish.
    #
    # The firmware is the authority (firmware/current/kinematics.py:9-14,72-80):
    #     pitch = (thetaA + thetaB) / 2N,  yaw = (thetaA - thetaB) / 2N
    # with PAYLOAD_SIGN = {"pitch": -1, "yaw": +1} (firmware/current/config.py).
    # AXIS_STEP_DEG already carries the 1/N belt ratio, so in host units:
    DEFAULT_AXIS_MIX = np.array([[-0.5, -0.5],    # pitch = -(A + B) / 2
                                 [+0.5, -0.5]])   # yaw   =  (A - B) / 2

    def __init__(self,
                 pitch_limit: Tuple[float, float] = config.PITCH_LIMIT_DEG,
                 yaw_limit: Tuple[float, float] = config.YAW_LIMIT_DEG,
                 margin_deg: float = LIMIT_MARGIN_DEG):
        self.pitch_limit = tuple(float(x) for x in pitch_limit)
        self.yaw_limit = tuple(float(x) for x in yaw_limit)
        self.margin_deg = float(margin_deg)
        self._mix = self.DEFAULT_AXIS_MIX.copy()
        self._mix_inv = np.linalg.inv(self._mix)
        self.pitch_deg: Optional[float] = None
        self.yaw_deg: Optional[float] = None
        self.pose_t: float = 0.0
        self.blocked: Tuple[str, ...] = ()

    def set_axis_mix(self, mix) -> None:
        """Override the differential mix once homing has measured it."""
        arr = np.asarray(mix, dtype=float)
        if arr.shape != (2, 2):
            raise ValueError(f"axis mix must be 2x2, got {arr.shape}")
        self._mix = arr
        self._mix_inv = np.linalg.inv(arr)   # raises if the mix is degenerate

    def update_pose(self, pitch_deg: float, yaw_deg: float,
                    t: Optional[float] = None) -> None:
        self.pitch_deg = float(pitch_deg)
        self.yaw_deg = float(yaw_deg)
        self.pose_t = time.perf_counter() if t is None else float(t)

    @property
    def pose_known(self) -> bool:
        return self.pitch_deg is not None and self.yaw_deg is not None

    def axis_rates(self, omega) -> np.ndarray:
        """Motor steps/s -> [pitch deg/s, yaw deg/s]."""
        return config.AXIS_STEP_DEG * (self._mix @ np.asarray(omega, float).reshape(2))

    def motor_rates(self, axis_rates) -> np.ndarray:
        """[pitch deg/s, yaw deg/s] -> motor steps/s."""
        return self._mix_inv @ (np.asarray(axis_rates, float).reshape(2)
                                / config.AXIS_STEP_DEG)

    @staticmethod
    def _scale(pos: float, rate: float, limits: Tuple[float, float],
               margin: float) -> float:
        """Fraction of `rate` allowed at `pos`.  1.0 free, 0.0 hard against a stop.

        Only motion *towards* the near stop is derated; motion away is always
        allowed at full rate, so the guard can never trap the platform.
        """
        lo, hi = limits
        if rate > 0:
            headroom = hi - pos
        elif rate < 0:
            headroom = pos - lo
        else:
            return 1.0
        if headroom >= margin:
            return 1.0
        return max(0.0, headroom / margin)

    def apply(self, omega) -> np.ndarray:
        """Derate motor rates that would drive an axis into its stop."""
        omega = np.asarray(omega, dtype=float).reshape(2)
        self.blocked = ()
        if not self.pose_known:
            return omega
        pitch_rate, yaw_rate = self.axis_rates(omega)
        s_pitch = self._scale(self.pitch_deg, pitch_rate, self.pitch_limit,
                              self.margin_deg)
        s_yaw = self._scale(self.yaw_deg, yaw_rate, self.yaw_limit,
                            self.margin_deg)
        if s_pitch >= 1.0 and s_yaw >= 1.0:
            return omega
        blocked = []
        if s_pitch < 1.0:
            blocked.append(f"pitch@{self.pitch_deg:+.1f}deg x{s_pitch:.2f}")
        if s_yaw < 1.0:
            blocked.append(f"yaw@{self.yaw_deg:+.1f}deg x{s_yaw:.2f}")
        self.blocked = tuple(blocked)
        # ONE FACTOR ON BOTH AXES, NOT ONE EACH.
        #
        # Scaling pitch and yaw independently ROTATES the commanded velocity:
        # the turret then drives the beam somewhere the control law never
        # pointed. MEASURED on run_2026-09-19_132817: a median 71 deg of
        # rotation (p90 84.8, max 89.4) across 139 derated rows, and 51 rows
        # where s_pitch collapsed to 0 outright, leaving a pure-yaw command
        # (rate_a == -rate_b exactly). Those 51 are every such row in the
        # entire corpus -- one 8.9 s window.
        #
        # It is also guaranteed to rotate rather than merely scale on this
        # rig: yaw never comes within 124 deg of its own stop in any recorded
        # run, so s_yaw is 1.0 everywhere and s_pitch != s_yaw whenever the
        # derate fires at all.
        #
        # min() is the same direction-preserving principle this file already
        # applies to the rate clip and to the feedforward platform cap, and it
        # is strictly a REDUCTION: never larger than either factor, so it
        # cannot make the turret move faster or further than before.
        s = min(s_pitch, s_yaw)
        return self.motor_rates((pitch_rate * s, yaw_rate * s))


# ==========================================================================
#   FACE / BEAM GEOMETRY
# ==========================================================================
def _point_box_distance(px: float, py: float, box: Detection) -> float:
    """Distance from a point to an axis-aligned box.  0.0 inside.

    Box distance, not centre distance: a face 2 m from the camera fills a box
    200 px across, and centre distance would let the beam cross a cheek while
    reporting 150 px of clearance.
    """
    dx = max(box.x1 - px, 0.0, px - box.x2)
    dy = max(box.y1 - py, 0.0, py - box.y2)
    return math.hypot(dx, dy)


def _segment_box_distance(a: Sequence[float], b: Sequence[float],
                          box: Detection) -> float:
    """Min distance from segment a->b to a box, by sampling every ~8 px.

    Sampling rather than an exact segment/AABB solve: the inhibit margin is
    120 px, so 8 px of quantisation is irrelevant, and this stays obviously
    correct at 3 am.
    """
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(length / _BEAM_SAMPLE_PX) + 1)
    best = float("inf")
    for i in range(n):
        t = i / (n - 1)
        best = min(best, _point_box_distance(a[0] + (b[0] - a[0]) * t,
                                             a[1] + (b[1] - a[1]) * t, box))
        if best == 0.0:
            break
    return best


def nearest_face_px(faces: Optional[Sequence[Detection]],
                    aim: Sequence[float],
                    target: Optional[Sequence[float]] = None) -> float:
    """Closest face, in px, to the beam path in the narrow frame.

    The "beam path" in image space is the goal pixel -- where the beam lands --
    plus the segment from there to the predicted target, which is the ground
    the aim point sweeps while the error closes.  Faces must already be in
    NARROW frame pixels; wide-frame faces have to be mapped by the caller.

    inf when there are no faces, so a caller can print "clear".
    """
    if not faces:
        return float("inf")
    end = tuple(aim) if target is None else tuple(target)
    return min(_segment_box_distance(tuple(aim), end, f) for f in faces)


def _inset_box(box: Detection, frac: float) -> Tuple[float, float, float, float]:
    """Box shrunk toward its centre by `frac` of its own width/height per side.

    Fraction rather than pixels so it scales with range: the same airframe is
    ~198 px across at 2 m and ~79 px at 5 m, and a fixed inset would be a
    rounding error at one end and swallow the whole box at the other.

    Degenerate boxes collapse to their centre point, which no segment can be
    strictly inside -- a box with no area is not evidence of anything.
    """
    ix, iy = box.w * frac, box.h * frac
    x1, x2 = box.x1 + ix, box.x2 - ix
    y1, y2 = box.y1 + iy, box.y2 - iy
    if x1 > x2:
        x1 = x2 = box.cx
    if y1 > y2:
        y1 = y2 = box.cy
    return x1, y1, x2, y2


def beam_inside_box_px(box: Detection,
                       aim: Sequence[float],
                       target: Optional[Sequence[float]] = None,
                       frac: float = config.DRONE_CONTAINMENT_INSET_FRAC) -> float:
    """Signed clearance, in px, of the beam path INSIDE an inset box.

    Positive means the whole path is inside with that much room to the nearest
    inset edge; negative or zero means part of it is outside. This is the
    permission test, so it answers the opposite question to nearest_face_px():
    that one wants the beam far FROM a box, this one wants it well WITHIN one.

    An axis-aligned box is convex, so both endpoints being inside means the
    entire segment is -- no sampling needed, and the answer is exact rather
    than quantised the way _segment_box_distance is.

    -inf when there is no box at all, so "no detection" and "detection nowhere
    near the beam" read the same to the caller: not permitted.
    """
    if box is None:
        return float("-inf")
    x1, y1, x2, y2 = _inset_box(box, frac)
    end = tuple(aim) if target is None else tuple(target)
    worst = float("inf")
    for px, py in (tuple(aim), end):
        worst = min(worst, px - x1, x2 - px, py - y1, y2 - py)
    return worst


# ==========================================================================
#   LASER INTERLOCK
# ==========================================================================
class LaserInterlock:
    """Every condition, evaluated every frame, before anything else happens.

    FAIL-CLOSED: PERMISSION IS GRANTED, NOT ASSUMED
    -----------------------------------------------
    The beam is authorised by POSITIVE evidence that it is pointing at the
    drone, and vetoed on top of that by heads. It is never authorised by the
    absence of a veto.

    This is not a stylistic preference. "No face was detected" and "the face
    pass did not run, or ran blind" produce the identical value -- an empty
    list -- and this project has already shipped the blind case once: YuNet was
    fed a 90-degree-rotated frame and returned [] on every frame of a room with
    a person in it. A clearance-only interlock reads that as a clear room and
    fires. Under the rule below it reads as no `heads_valid` and holds the beam
    off, which is the same reading it gives a crashed detector, a dropped frame
    and a stale report.

    The permission is `drone_lock`: a detector box from THIS frame, no older
    than DRONE_BOX_MAX_AGE_S, with the whole beam path inside it. Every failure
    mode of the perception stack -- blind, crashed, starved, coasting on a
    filter prediction -- removes the box and therefore the beam. Note that the
    tracker deliberately holds TrackState.TRACK across brief detection misses
    (that is what the filter is for), so `track` alone is a statement about the
    FILTER, not about pixels. Only `drone_lock` is evidence about pixels.

    The laser may be on ONLY when all of these hold:

      armed        config.LASER_ENABLED / the GUI arm control
      track        TrackState.TRACK -- and COAST turns it off IMMEDIATELY, on
                   the frame the state changes, not after the coast decay: a
                   coasting estimate is a guess, and a guess must not emit
                   light
      goal_calib   the goal pixel is a MEASURED beam position, not the frame
                   centre fallback -- see below
      drone_lock   a fresh drone box CONTAINS the beam path -- the permission
      shape        that box is drone-shaped: w/h within FIRE_BOX_ASPECT_MIN..MAX.
                   run_145109 frame 006835 fired on the carrier's hand: the
                   drone was edge-on behind the fingers and YOLO boxed the
                   hand at conf 0.56, 202x91 px. Nothing else in the chain can
                   see that; containment is satisfied by any box
      heads_valid  the head pass reported recently enough to have a say
      face_clear   no head within FACE_INHIBIT_MARGIN_PX of the beam path
      settled      platform below SETTLED_RATE_DEG_S
      error        |e| < MAX_ERROR_TO_FIRE_PX
      vel_fresh    a `vel` went out in the last 100 ms
      duty         under LASER_MAX_ON_MS of continuous on-time

    CONSEQUENCE TO MEASURE ON THE LIVE DETECTOR:
    the tracker clears `_last_det` on every frame that fails to associate, so
    the beam goes off for that frame -- it does not ride the miss out. That is
    the correct default, and it is the same signal the GUI already draws (the
    box disappears exactly when permission does), but if the detector misses
    often enough the beam will strobe. DRONE_BOX_MAX_AGE_S cannot smooth it,
    because a missed frame produces no box to age rather than an old one. If
    that turns out to matter, the fix belongs in the tracker -- retain the last
    detection WITH its measurement time and let the age check here reject it --
    and never in loosening this condition.

    WHY `error` DOES NOT ALREADY DO THIS:
    MAX_ERROR_TO_FIRE_PX bounds the distance from the goal pixel to the
    PREDICTED target -- a number the filter computes from its own state. It is
    small and satisfied throughout a coast, because a coasting filter is
    confidently predicting a target that may no longer be there. Containment
    asks a different question: is there a drone-shaped thing in the pixels,
    now, where the beam is going. Both are needed; neither implies the other.

    WHY goal_calibrated IS AN INTERLOCK CONDITION AND NOT A WARNING:
    `face_clear` is measured by nearest_face_px() from `aim`, and `aim` is the
    goal pixel -- i.e. where we believe the BEAM lands. With no goal_pixel.json
    GoalModel.goal_pixel() returns the frame centre, which is where the CAMERA
    points, and the laser's boresight offset from it has never been measured.
    At NARROW_F_PX = 1400 one degree of offset is 24 px and three degrees is
    73 px, against FACE_INHIBIT_MARGIN_PX = 120 -- and the sign is unknown, so
    the error can eat the margin instead of adding to it. The interlock would
    report 130 px of clearance with the beam 30 px from a face.
    MAX_ERROR_TO_FIRE_PX does not bound this: it certifies the target is near
    the goal pixel, not that the goal pixel is near the beam. Frame centre
    stays a legitimate fallback for AIMING; it is never a fallback that may
    emit light.

    Every condition is evaluated -- no short-circuit -- so `failed` and
    `report()` can tell the GUI exactly which one is holding the laser off.
    A short-circuit here would make the demo say "unsettled" while a face sits
    in the frame, which is exactly the thing a spectator needs to see.
    """

    CONDITIONS = ("armed", "track", "goal_calibrated", "drone_lock", "shape",
                  "heads_valid", "face_clear", "settled", "error",
                  "vel_fresh", "duty")

    def __init__(self, armed: bool = config.LASER_ENABLED):
        self.armed = bool(armed)
        self.conditions: Dict[str, bool] = {k: False for k in self.CONDITIONS}
        self.margins: Dict[str, float] = {}
        self.failed: List[str] = list(self.CONDITIONS)
        self.state: LaserState = LaserState.DISARMED
        self.on: bool = False
        self._on_since: Optional[float] = None
        self._cooldown_until: float = 0.0
        self._box_source: str = "narrow"

    # -- arming ------------------------------------------------------------
    def arm(self) -> None:
        """Explicit and reversible, per the GUI contract."""
        self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self._on_since = None

    # -- the one entry point ----------------------------------------------
    def evaluate(self, *,
                 t_now: float,
                 track_state: TrackState,
                 error_px: float,
                 platform_rate_deg_s: float,
                 last_vel_sent_t: Optional[float],
                 faces: Optional[Sequence[Detection]] = None,
                 aim: Optional[Sequence[float]] = None,
                 target: Optional[Sequence[float]] = None,
                 goal_calibrated: bool = False,
                 drone_box: Optional[Detection] = None,
                 drone_box_t: Optional[float] = None,
                 drone_box_source: str = "narrow",
                 heads_t: Optional[float] = None) -> LaserState:
        """Evaluate all interlock conditions and return the resulting LaserState.

        `aim` is the goal pixel (where the beam lands) and `target` the
        predicted target pixel, both in the narrow frame; together they define
        the beam path that the containment and head checks run against.
        Passing faces without an aim point is a programming error, not a
        reason to fire.

        `drone_box` is the detector's box for THIS frame -- TrackEstimate.box,
        which the tracker clears to None on any frame that fails to associate
        -- and `drone_box_t` is when it was measured (TrackEstimate.box_t), not
        when the estimate was predicted to.  Together they are the permission.

        `heads_t` is when the head pass last reported, whatever it reported.
        An empty `faces` list means "clear" only if `heads_t` says a pass
        actually produced it.

        THE DEFAULTS ARE THE SAFETY PROPERTY.  `goal_calibrated` False,
        `drone_box` None, `heads_t` None: a caller that has not thought about
        where the beam goes, what it is pointing at, or whether anyone checked
        for heads does not get to fire.  Every call site that omits an argument
        fails safe rather than silently keeping older behaviour -- which is why
        these are keyword-only and why none of them defaults to a permissive
        value.
        """
        c = self.conditions
        m = self.margins

        c["armed"] = bool(self.armed)
        c["track"] = (track_state == TrackState.TRACK)
        # The face clearance below is measured FROM `aim`. If `aim` is the
        # frame-centre fallback, that clearance is not a beam clearance.
        c["goal_calibrated"] = bool(goal_calibrated)

        # -- PERMISSION: pixel evidence of a drone under the beam, right now --
        # A negative age means a clock mix-up between threads; treated as stale,
        # the same way vel_fresh does, because the failure is toward off.
        box_age = (float("inf") if drone_box_t is None
                   else t_now - float(drone_box_t))
        m["box_age_ms"] = box_age * 1000.0
        # TRACKING MAY USE EITHER CAMERA. FIRING MAY NOT.
        # A wide box mapped into the narrow frame is scaled up by ~2x, and its
        # edge error scales with it -- before any correction for the wide
        # camera's barrel distortion, which at 107.9 deg is real. Containment
        # is a statement about where the beam sits relative to the AIRFRAME
        # EDGE, so it is exactly the measurement that scaling degrades. Good
        # enough to slew the drone back into the narrow field; not good enough
        # to emit light against.
        narrow_sourced = (drone_box_source == "narrow")
        m["box_source"] = 1.0 if narrow_sourced else 0.0
        if (drone_box is None or aim is None or not narrow_sourced
                or not 0.0 <= box_age <= config.DRONE_BOX_MAX_AGE_S):
            inside_px = float("-inf")
        else:
            inside_px = beam_inside_box_px(drone_box, aim, target)
        m["inside_px"] = inside_px
        c["drone_lock"] = inside_px > 0.0
        self._box_source = drone_box_source

        # -- SHAPE: the box must look like a drone, not like the hand holding it
        # A hand seen side-on with the drone hidden behind it boxes wide and
        # small (2.22 aspect, smallest area of the run) while a quadcopter from
        # this camera sits between ~0.5 and ~1.5. Evaluated on the same box as
        # drone_lock, so a missing box fails here too rather than passing.
        if drone_box is None:
            aspect = float("nan")
        else:
            bw = float(drone_box.x2) - float(drone_box.x1)
            bh = float(drone_box.y2) - float(drone_box.y1)
            aspect = bw / bh if bh > 0.0 else float("inf")
        m["aspect"] = aspect
        c["shape"] = (config.FIRE_BOX_ASPECT_MIN <= aspect
                      <= config.FIRE_BOX_ASPECT_MAX)

        # -- VETO: heads, and only from a pass that actually reported ---------
        # This is the condition that separates "nobody is there" from "nobody
        # looked". Without it an empty `faces` list authorises the beam, and a
        # blind detector produces exactly that list -- which is not a
        # hypothetical here: a 90-degree-rotated frame made YuNet return [] on
        # every frame of an occupied room.
        head_age = (float("inf") if heads_t is None
                    else t_now - float(heads_t))
        m["heads_age_ms"] = head_age * 1000.0
        c["heads_valid"] = 0.0 <= head_age <= config.HEAD_REPORT_MAX_AGE_S

        if faces:
            if aim is None:
                raise ValueError("faces supplied without an aim point: the "
                                 "interlock cannot check a beam path it does "
                                 "not know")
            face_px = nearest_face_px(faces, aim, target)
        else:
            face_px = float("inf")
        m["face_px"] = face_px
        c["face_clear"] = face_px > config.FACE_INHIBIT_MARGIN_PX

        rate = abs(float(platform_rate_deg_s))
        m["rate_deg_s"] = rate
        c["settled"] = rate < config.SETTLED_RATE_DEG_S

        err = abs(float(error_px))
        m["error_px"] = err
        c["error"] = err < config.MAX_ERROR_TO_FIRE_PX

        if last_vel_sent_t is None:
            vel_age = float("inf")
        else:
            vel_age = t_now - float(last_vel_sent_t)
        m["vel_age_ms"] = vel_age * 1000.0
        # A negative age means a clock mix-up between threads; treat it as stale
        # rather than fresh.  Fail toward off.
        c["vel_fresh"] = 0.0 <= vel_age <= VEL_FRESH_S

        c["duty"] = t_now >= self._cooldown_until
        m["on_ms"] = 0.0 if self._on_since is None else (t_now - self._on_since) * 1000.0

        on = all(c[k] for k in self.CONDITIONS)

        # Duty accounting last: only a laser that WOULD be on accrues on-time.
        if on:
            if self._on_since is None:
                self._on_since = t_now
            elif (t_now - self._on_since) * 1000.0 > config.LASER_MAX_ON_MS:
                on = False
                c["duty"] = False
                self._on_since = None
                self._cooldown_until = t_now + LASER_COOLDOWN_S
        else:
            self._on_since = None

        self.failed = [k for k in self.CONDITIONS if not c[k]]
        self.on = on
        self.state = LaserState.FIRING if on else self._inhibit_state()
        return self.state

    def _inhibit_state(self) -> LaserState:
        """Pick the most informative failure for the banner.

        Face first, always: the inhibit banner is the demo moment and the one
        failure a spectator must see even when three others are also true.
        """
        c = self.conditions
        if not c["face_clear"]:
            return LaserState.INHIBITED_FACE
        if not c["heads_valid"]:
            # Reported as a FACE inhibit for the same reason as the case below:
            # what is unknown is whether a head is on the beam path, and unknown
            # is not clear. app.py's outer staleness guard returns the same
            # state, so the banner reads identically whichever layer catches it.
            return LaserState.INHIBITED_FACE
        if not c["goal_calibrated"]:
            # Reported as a FACE inhibit, and deliberately ahead of `armed`:
            # what is actually unknown is whether the beam path is clear of
            # faces, because the beam's position in the frame has never been
            # measured. Same reading as a stale face report -- unknown is not
            # clear. types.LaserState has no member for it; the reason string
            # carries the detail to the panel.
            return LaserState.INHIBITED_FACE
        if not c["armed"]:
            return LaserState.DISARMED
        if not c["drone_lock"] or not c["shape"]:
            # Nothing is wrong and nothing is unsafe -- there is simply no
            # confirmed drone under the beam, so there is no permission. This
            # is the resting state of a fail-closed interlock and it is the one
            # a spectator sees most of the time. A box that is not drone-shaped
            # is the same thing: no confirmed drone.
            return LaserState.INHIBITED_NO_LOCK
        if not c["settled"]:
            return LaserState.INHIBITED_UNSETTLED
        if not c["error"]:
            return LaserState.INHIBITED_ERROR
        # track / vel_fresh / duty have no dedicated enum member; the reason
        # string carries them to the GUI.
        return LaserState.OFF

    # -- reporting ---------------------------------------------------------
    @property
    def reason(self) -> str:
        if self.on:
            return "all interlock conditions met"
        if not self.failed:
            return "off"
        parts = []
        for k in self.failed:
            if k == "face_clear":
                parts.append(f"face {self.margins.get('face_px', 0):.0f}px "
                             f"(< {config.FACE_INHIBIT_MARGIN_PX})")
            elif k == "drone_lock":
                age = self.margins.get("box_age_ms", float("inf"))
                inside = self.margins.get("inside_px", float("-inf"))
                if getattr(self, "_box_source", "narrow") != "narrow":
                    parts.append("tracking on %s -- firing needs the narrow "
                                 "camera" % getattr(self, "_box_source", "?"))
                elif not math.isfinite(age):
                    parts.append("no drone box")
                elif age > config.DRONE_BOX_MAX_AGE_S * 1000.0 or age < 0.0:
                    parts.append(f"drone box {age:.0f}ms stale")
                else:
                    parts.append(f"beam {-inside:.0f}px outside the drone box")
            elif k == "shape":
                parts.append(f"box aspect {self.margins.get('aspect', float('nan')):.2f} "
                             f"not drone-shaped ({config.FIRE_BOX_ASPECT_MIN}.."
                             f"{config.FIRE_BOX_ASPECT_MAX})")
            elif k == "heads_valid":
                age = self.margins.get("heads_age_ms", float("inf"))
                parts.append("head pass never reported" if not math.isfinite(age)
                             else f"head pass {age:.0f}ms stale")
            elif k == "settled":
                parts.append(f"moving {self.margins.get('rate_deg_s', 0):.1f} deg/s "
                             f"(>= {config.SETTLED_RATE_DEG_S})")
            elif k == "error":
                parts.append(f"error {self.margins.get('error_px', 0):.1f}px "
                             f"(>= {config.MAX_ERROR_TO_FIRE_PX})")
            elif k == "vel_fresh":
                age = self.margins.get("vel_age_ms", float("inf"))
                parts.append("no vel sent" if not math.isfinite(age)
                             else f"vel {age:.0f}ms stale")
            elif k == "duty":
                parts.append(f"duty cap {config.LASER_MAX_ON_MS} ms")
            elif k == "track":
                parts.append("not TRACK")
            elif k == "armed":
                parts.append("disarmed")
            elif k == "goal_calibrated":
                parts.append("GOAL PIXEL UNCALIBRATED (face clearance would be "
                             "measured from the frame centre, not the beam) -- "
                             "run calibrate.py menu 2")
        return "; ".join(parts)

    def report(self) -> dict:
        """Full breakdown for the GUI.  Plain dict -- no new shared type."""
        return {"state": self.state, "on": self.on,
                "conditions": dict(self.conditions),
                "failed": list(self.failed),
                "margins": dict(self.margins),
                "reason": self.reason}


# ==========================================================================
#   CONTROLLER
# ==========================================================================
class Controller:
    """The control law: predict, error, feedforward, Jacobian, clip."""

    def __init__(self,
                 jacobian: Optional[Jacobian] = None,
                 goal_model: Optional[GoalModel] = None,
                 limits: Optional[TravelLimits] = None,
                 gain_k: float = config.CONTROL_GAIN_K,
                 latency_s: float = config.LATENCY_S):
        self.jacobian = jacobian if jacobian is not None else Jacobian.auto()
        self.goal_model = goal_model if goal_model is not None else GoalModel.load()
        self.limits = limits if limits is not None else TravelLimits()
        self.gain_k = float(gain_k)
        self.latency_s = float(latency_s)
        self.last_output: Optional[ControlOutput] = None
        # When a DETECTOR last actually saw the target. Not when the filter
        # last produced an estimate -- it produces one every frame regardless.
        # See the evidence decay in compute().
        self._last_evidence_t: Optional[float] = None
        #: Platform compensation state (config.PLATFORM_COMPENSATION).
        #: `_last_omega` is what the board was last asked for; `_omega_at_box`
        #: is what it was doing when the detection now in use arrived. Their
        #: DIFFERENCE is the part of the filter's velocity that belongs to the
        #: turret rather than the drone.
        self._last_omega = np.zeros(2)
        self._omega_at_box: Optional[np.ndarray] = None
        #: The feedforward input actually used, and the platform term removed
        #: from it. Logged every row -- without these the standing-lag claim
        #: cannot be checked against a recording.
        self._v_target_hat: tuple = (0.0, 0.0)
        self._ff_platform_px_s: float = 0.0
        #: Staleness responses, logged per row so their transitions are
        #: visible. p_scale gates the P term; authority scales the final rate.
        self._p_scale: float = 1.0
        self._authority: float = 1.0
        self._box_t_of_omega: Optional[float] = None
        self._platform_corr_px = 0.0

    @property
    def ready(self) -> bool:
        """False means the loop must not servo.  app.py gates on this and the
        GUI shows why, instead of the loop raising once per frame."""
        return self.jacobian.is_calibrated

    def require_ready(self) -> None:
        """Call at startup so an uncalibrated J fails loudly before the demo."""
        self.jacobian._require()

    # -- the control law ---------------------------------------------------
    def compute(self, est: TrackEstimate, t_now: float, *,
                range_m: Optional[float] = None,
                dot_pixel: Optional[Sequence[float]] = None,
                pose: Optional[Sequence[float]] = None) -> ControlOutput:
        """Filtered estimate -> signed motor rates for `vel <A> <B>`.

        Raises NotCalibratedError when J is uncalibrated.  It does not return
        zeros: zeros look like "on target" to every downstream consumer, and
        silently not moving is exactly the failure this project must not have.
        """
        if not self.jacobian.is_calibrated:
            self.jacobian._require()

        if pose is not None:
            self.limits.update_pose(pose[0], pose[1])

        # -- goal ----------------------------------------------------------
        if dot_pixel is not None:
            # Opportunistic: if the dot detector actually saw the beam this
            # frame, aim at the target using the real beam position and skip
            # the model entirely.  Never gated on -- it is a bonus, not a
            # requirement (config's own dot thresholds are estimates weakened
            # by the missing IR-cut filter).
            goal_u, goal_v = float(dot_pixel[0]), float(dot_pixel[1])
            dot_locked = True
        else:
            if range_m is None:
                range_m = est.range_m if est.range_m > 0 else config.ASSUMED_RANGE_M
            goal_u, goal_v = self.goal_model.goal_pixel(range_m)
            dot_locked = False

        # -- predict to the moment the beam will actually move --------------
        # est is already predicted to est.predicted_to_t; carry it the rest of
        # the way to t_now + LATENCY_S.  Commanding against where the target is
        # NOW is commanding against where it was 60 ms ago.
        dt = (t_now + self.latency_s) - est.predicted_to_t
        dt = min(dt, MAX_EXTRAPOLATION_S)
        u_hat = est.u + est.du * dt
        v_hat = est.v + est.dv * dt

        # -- error ----------------------------------------------------------
        e_u = u_hat - goal_u
        e_v = v_hat - goal_v

        # Deadband on the P term ONLY.  The feedforward keeps flowing inside
        # the band: the point of the band is to stop the P term dithering the
        # mechanism (0.57 deg of backlash means every reversal is 30 mm of beam
        # wander at 3 m), not to stop tracking a moving target.
        db = config.ERROR_DEADBAND_PX
        p_u = 0.0 if abs(e_u) < db else e_u
        p_v = 0.0 if abs(e_v) < db else e_v

        # -- how stale is the EVIDENCE, as opposed to the estimate? ----------
        # est is a number every frame whether or not anything was measured --
        # the filter predicts regardless. box_t is the last time a DETECTOR
        # actually saw the target, and it is the only honest input to "how
        # much should this frame be trusted".
        if est.box_t is not None:
            self._last_evidence_t = est.box_t
        age = (float("inf") if self._last_evidence_t is None
               else t_now - self._last_evidence_t)
        if age <= COAST_HOLD_S:
            authority = 1.0
        elif age >= COAST_HOLD_S + COAST_DECAY_S:
            authority = 0.0
        else:
            authority = 1.0 - (age - COAST_HOLD_S) / COAST_DECAY_S

        # -- control law ----------------------------------------------------
        # THE MINUS SIGN IS THE LOOP. Do not remove it.
        #
        # J as calibrate.py measures it (calibrate.py: point at a STATIC scene,
        # step motor A, record where the feature went) is
        #     J = d(image position of a world-fixed point) / d(motor step)
        # and the camera rides the payload, so a static feature moves OPPOSITE
        # to the aim direction -- that inversion is already inside the measured
        # numbers. With e = u_hat - goal:
        #     d(e)/dt = J @ omega + v_target
        # and we want d(e)/dt = -K*e, so
        #     omega = -J_inv @ (K*e + v_target)
        # Without the negation both terms are POSITIVE feedback: the P term
        # drives the target away from the goal and the feedforward doubles the
        # crossing rate instead of cancelling it, so the loop saturates at
        # MAX_MOTOR_RATE within a few frames and slews to a travel limit on the
        # first frame it acquires. One negation fixes both terms because their
        # relative sign is already right. It belongs HERE, not in the stored
        # matrix: calibrate_jacobian() re-derives J from measurement on every
        # run and would overwrite a hand-negated file.
        # THE P TERM NEEDS A MEASUREMENT; THE FEEDFORWARD DOES NOT.
        #
        # `p_scale` drops the proportional term once no detector has confirmed
        # the target recently. It is the term that runs away: the filter keeps
        # extrapolating position, so `e` grows every frame on no evidence and
        # K*e grows with it. Recorded: |e| went 200 -> 510 px with no
        # detection at all, and the command chased it to the rate limit.
        #
        # The feedforward is different in kind. est.du/dv is the target's last
        # MEASURED velocity, and continuing to match it is exactly what
        # config.COAST_HOLD_MS means by "hold last velocity" -- it is what
        # keeps the beam on a target that blinked out for three frames. It
        # decays with `authority` below rather than being dropped.
        #
        # So: coasting keeps matching the target's speed, and stops arguing
        # with a position nothing has seen.
        if config.P_SCALE_DECAY:
            # Same ramp `authority` uses, computed above: full while fresh,
            # then linear to zero over COAST_DECAY_S. The P term fades out
            # instead of being switched off mid-flight.
            p_scale = authority
        else:
            p_scale = 1.0 if age <= COAST_HOLD_S else 0.0
        self._p_scale = float(p_scale)
        self._authority = float(authority)

        # -- PLATFORM COMPENSATION (config.PLATFORM_COMPENSATION, default OFF)
        # The filter's velocity is what the camera SAW move, which is the drone
        # plus the turret. It embeds whatever slew was in effect while the last
        # detections arrived, and the lead prediction carries that forward as
        # though it were the drone's own motion.
        #
        # Only the CHANGE in commanded rate needs correcting: the part that was
        # steady is already in est.du/dv correctly. See the note in config.py.
        # Zero while the command holds, which is most of the time.
        self._platform_corr_px = 0.0
        if config.PLATFORM_COMPENSATION and self._omega_at_box is not None \
                and self.jacobian.is_calibrated:
            try:
                d_omega = self._last_omega - self._omega_at_box
                corr = self.jacobian.pixel_rates(d_omega) * self.latency_s
                mag = float(np.hypot(corr[0], corr[1]))
                if mag > config.PLATFORM_COMP_MAX_PX:
                    # A jump this large is a saturation or a state change, not
                    # a slew. Extrapolating it is how the runaway behaved.
                    corr = corr * (config.PLATFORM_COMP_MAX_PX / mag)
                    mag = config.PLATFORM_COMP_MAX_PX
                if mag >= config.PLATFORM_COMP_MIN_PX:
                    # The target's predicted position shifts WITH the platform,
                    # so the error the controller should act on shifts with it
                    # too. Sub-pixel corrections are Jacobian calibration noise.
                    p_u += float(corr[0])
                    p_v += float(corr[1])
                    self._platform_corr_px = mag
            except Exception:                            # noqa: BLE001
                # Never let a refinement break the loop it refines.
                self._platform_corr_px = 0.0

        # -- FEEDFORWARD ON THE TARGET'S INERTIAL VELOCITY -----------------
        # config.FEEDFORWARD_SUBTRACT_PLATFORM, default OFF. See config.py.
        #
        # The derivation above asks for v_target. est.du/dv is v_target PLUS
        # the platform's own contribution J @ omega, because the camera rides
        # the payload. Since the loop drives J @ omega toward -v_target, the
        # two cancel and the feedforward collapses to zero precisely when
        # tracking works -- leaving a standing error of v_target / K.
        #
        # omega_at_box is the rate that was in effect when the box was
        # measured, NOT the rate now: est.du describes motion that happened
        # then. It is already maintained for the position-compensation path.
        ff_u, ff_v = est.du, est.dv
        self._v_target_hat = (float(est.du), float(est.dv))
        self._ff_platform_px_s = 0.0
        if config.FEEDFORWARD_SUBTRACT_PLATFORM                 and self._omega_at_box is not None                 and self.jacobian.is_calibrated:
            try:
                plat = self.jacobian.pixel_rates(self._omega_at_box)
                mag = float(np.hypot(plat[0], plat[1]))
                cap = config.FEEDFORWARD_PLATFORM_MAX_PX_S
                if mag > cap:
                    # Direction-preserving, like every other saturation here:
                    # a per-axis clip would rotate the correction.
                    plat = plat * (cap / mag)
                    mag = cap
                ff_u = est.du - float(plat[0])
                ff_v = est.dv - float(plat[1])
                self._v_target_hat = (ff_u, ff_v)
                self._ff_platform_px_s = mag
            except Exception:                            # noqa: BLE001
                # Never let a refinement break the loop it refines.
                ff_u, ff_v = est.du, est.dv
                self._v_target_hat = (float(est.du), float(est.dv))
                self._ff_platform_px_s = 0.0

        # config.FEEDFORWARD_GAIN: 1.0 is the law above; 0.0 is pure P.
        _ffg = float(getattr(config, "FEEDFORWARD_GAIN", 1.0))
        ff_u, ff_v = ff_u * _ffg, ff_v * _ffg
        pixel_rate = np.array([self.gain_k * p_u * p_scale + ff_u,
                               self.gain_k * p_v * p_scale + ff_v])
        omega = -self.jacobian.motor_rates(pixel_rate)

        # -- EVIDENCE DECAY -------------------------------------------------
        # Nothing above knows whether a DETECTOR ever saw the target this
        # frame. `est` is always a number, because the filter predicts forward
        # whether or not anything was measured -- and both terms grow while it
        # does. The P term grows because the predicted position runs off
        # frame; the feedforward keeps commanding a velocity nothing has
        # confirmed since the last real box.
        #
        # MEASURED, run_2026-09-19_011729. It had the drone centred and was
        # converging -- error 111 -> 98 -> 68 -> 45 px, confidence 0.72. Then
        # one frame missed, and with NO detection at all for the next half
        # second:
        #
        #     t      box   |e| px   commanded yaw
        #     85.82   --     200      -139 deg/s
        #     85.94   --     279      -161
        #     86.09   --     377      -188
        #     86.16   --     423      -200.7   <- saturated
        #     86.30   --     510      -183
        #
        # The error GREW monotonically on no evidence, and the command chased
        # it to the rate limit, flinging the turret away from a drone it had
        # just centred. At 195 deg/s nothing is detectable, so it never came
        # back. That is the whole "it can't lock on" failure.
        #
        # config already names the intended behaviour -- COAST_HOLD_MS "hold
        # last velocity", COAST_DECAY_MS "then decay to zero" -- but those
        # shape the tracker's ESTIMATE, and an estimate that decays in
        # velocity still runs away in position. The decay has to be applied to
        # the COMMAND.
        #
        # Full authority while the evidence is fresh, so riding out a dropped
        # frame is unaffected; then a linear taper to zero. Commanding nothing
        # on no evidence is the honest action, and a stationary turret is one
        # that can still see.
        # The magnitude decay is APPLIED AFTER SATURATION, further down -- not
        # here. Scaling the raw demand does nothing when the demand is already
        # far past the rate limit: 0.6 x an enormous number is still enormous,
        # still clips to MAX_MOTOR_RATE, and the turret slews at full speed on
        # no evidence. Measured while building this: the decay applied to the
        # demand left the replayed runaway pinned at 311 deg/s throughout.

        # -- travel limits, then clip ---------------------------------------
        omega = self.limits.apply(omega)

        # DIRECTION-PRESERVING saturation, not componentwise. np.clip() clamps
        # each axis independently, so when one axis saturates and the other does
        # not, the ratio between them changes and the commanded image-space
        # motion ROTATES away from the direction the control law asked for --
        # the turret drives somewhere the law never pointed it. Observed as
        # `vel -4000.0 510.6` for a command whose true direction needed a much
        # larger second component.
        #
        # Scaling both axes by the worst-case factor keeps the direction exact
        # and only reduces the magnitude, which is what saturation should mean.
        # Low severity in practice -- the interlock's error < 25 px condition
        # keeps the beam off whenever the loop is saturated -- but a turret that
        # slews off-axis under saturation is worse than one that slews slowly
        # along the right line, and this costs nothing.
        limit = float(config.MAX_MOTOR_RATE)
        peak = float(np.max(np.abs(omega))) if omega.size else 0.0
        if peak > limit:
            clipped = omega * (limit / peak)
        else:
            clipped = omega
        saturated = bool(peak > limit) or bool(self.limits.blocked)

        # EVIDENCE DECAY, on the FINAL rate. See where `authority` is computed.
        # Here rather than upstream because this is the number the motors get:
        # scaling before the clip is scaling a demand that gets clipped back to
        # full rate anyway. Direction-preserving, like the saturation above --
        # it only ever reduces magnitude.
        if authority < 1.0:
            clipped = clipped * authority

        # Bookkeeping for the next pass's platform compensation. Recorded
        # AFTER the clip, because the clipped value is what the board is
        # actually asked to do and therefore what the platform actually does.
        if est.box_t is not None and est.box_t != self._box_t_of_omega:
            # A NEW detection: the rate in effect right now is the one its
            # motion was observed under.
            self._omega_at_box = self._last_omega.copy()
            self._box_t_of_omega = est.box_t
        self._last_omega = np.asarray(clipped, dtype=float).reshape(2).copy()

        ob = self._omega_at_box
        out = ControlOutput(rate_a=float(clipped[0]), rate_b=float(clipped[1]),
                            error_u=e_u, error_v=e_v,
                            goal_u=goal_u, goal_v=goal_v,
                            saturated=saturated, dot_locked=dot_locked,
                            v_target_u=self._v_target_hat[0],
                            v_target_v=self._v_target_hat[1],
                            ff_platform_px_s=self._ff_platform_px_s,
                            omega_at_box_a=(None if ob is None else float(ob[0])),
                            omega_at_box_b=(None if ob is None else float(ob[1])),
                            p_scale=self._p_scale, authority=self._authority)
        self.last_output = out
        return out

    # -- helpers the loop needs -------------------------------------------
    @staticmethod
    def zero_output(goal_u: float = FRAME_CENTRE[0],
                    goal_v: float = FRAME_CENTRE[1]) -> ControlOutput:
        """`vel 0 0` as a ControlOutput, for any transition out of TRACK."""
        return ControlOutput(rate_a=0.0, rate_b=0.0, error_u=0.0, error_v=0.0,
                             goal_u=goal_u, goal_v=goal_v)

    @staticmethod
    def platform_rate_deg_s(out: ControlOutput) -> float:
        """Commanded payload rate, deg/s, for the interlock's settled test.

        AXIS_STEP_DEG already has the belt ratio folded in, so a motor step is
        a payload step.  This is the COMMANDED rate; if the IMU rate is
        available, pass that to the interlock instead -- it also sees the
        ringing after a move stops, which this cannot.
        """
        return max(abs(out.rate_a), abs(out.rate_b)) * config.AXIS_STEP_DEG

    @staticmethod
    def error_magnitude(out: ControlOutput) -> float:
        return math.hypot(out.error_u, out.error_v)


# ==========================================================================
#   SELF-TEST -- control law and every interlock branch
# ==========================================================================
def _demo_jacobian() -> Jacobian:
    """A plausible differential J for the self-test.

    Both motors move both image axes -- that is what a differential does -- at
    roughly the geometric 1.19 px/step split across the two axes.  This is a
    TEST value and deliberately not written to config.py.
    """
    return Jacobian([[0.60, 0.59],
                     [0.58, -0.61]], source="self-test")


def _est(u, v, du=0.0, dv=0.0, t=0.0, state=TrackState.TRACK) -> TrackEstimate:
    return TrackEstimate(u=u, v=v, du=du, dv=dv, predicted_to_t=t,
                         state=state, q=config.Q_BASE, nis=1.0,
                         range_m=config.ASSUMED_RANGE_M)


def _face(cx, cy, w=160.0, h=200.0) -> Detection:
    return Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9, "face")


def _drone(cx, cy, w=132.0, h=132.0) -> Detection:
    """The 283 mm airframe as it appears at ASSUMED_RANGE_M -- ~132 px."""
    return Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9, "drone")


def _main() -> None:
    import tempfile

    line = "-" * 72
    print(line)
    print("1. UNCALIBRATED J REFUSES MOTION")
    print(line)
    bare = Controller(jacobian=Jacobian(), goal_model=GoalModel())
    print(f"   ready = {bare.ready}   {bare.jacobian!r}")
    try:
        bare.compute(_est(700.0, 400.0), t_now=0.0)
        raise AssertionError("uncalibrated J must not produce motion")
    except NotCalibratedError as exc:
        print(f"   refused: {str(exc).splitlines()[0]}")

    print()
    print(line)
    print("2. GOAL PIXEL  g(R) = g_inf + c/R")
    print(line)
    blank = GoalModel()
    print(f"   uncalibrated -> {blank.goal_pixel(3.0)}  [{blank.source_text}]")
    print(f"   {blank.cad_check()}")
    # Fit values standing in for calibrate.py's output.  The parallax lands on
    # the ROW here because of the 90 deg mount -- but note it is the numbers
    # that say so, not the code.
    fitted = GoalModel(u_inf=641.0, c_u=-1.2, v_inf=352.0, c_v=42.0,
                       source="self-test")
    for r in (2.0, 3.0, 5.0):
        gu, gv = fitted.goal_pixel(r)
        print(f"   R = {r:.1f} m -> goal ({gu:7.2f}, {gv:7.2f})")
    print(f"   parallax landed on: {fitted.parallax_axis}")
    print(f"   {fitted.cad_check()}")
    print(f"   range clamp: R=0.01 -> {fitted.goal_pixel(0.01)} "
          f"(clamped into {config.RANGE_LIMITS_M})")

    tmp = Path(tempfile.mkdtemp(prefix="turret_ctl_")) / "sub"
    jp = _demo_jacobian().save(tmp / "jacobian.json", note="self-test")
    gp = fitted.save(tmp / "goal_pixel.json", note="self-test")
    print(f"   JSON round-trip: {Jacobian.load(jp)!r}")
    print(f"                    {GoalModel.load(gp)!r}")

    print()
    print(line)
    print("3. CONTROL LAW")
    print(line)
    ctl = Controller(jacobian=_demo_jacobian(), goal_model=fitted)
    goal = fitted.goal_pixel(config.ASSUMED_RANGE_M)
    print(f"   K = {ctl.gain_k}  latency = {ctl.latency_s * 1000:.0f} ms  "
          f"goal = ({goal[0]:.1f}, {goal[1]:.1f})")

    on_goal = ctl.compute(_est(goal[0], goal[1], t=0.0), t_now=0.0)
    print(f"   a) on goal, still      -> vel {on_goal.rate_a:8.1f} "
          f"{on_goal.rate_b:8.1f}   |e| = {Controller.error_magnitude(on_goal):.2f} px")

    small = ctl.compute(_est(goal[0] + 0.5, goal[1] - 0.4, t=0.0), t_now=0.0)
    print(f"   b) 0.6 px off (deadband)-> vel {small.rate_a:8.1f} "
          f"{small.rate_b:8.1f}   P term suppressed, |e| = "
          f"{Controller.error_magnitude(small):.2f} px")

    off = ctl.compute(_est(goal[0] + 100.0, goal[1] + 60.0, t=0.0), t_now=0.0)
    print(f"   c) 100,60 px off, still -> vel {off.rate_a:8.1f} {off.rate_b:8.1f}"
          f"   (P only)")

    # 1 m/s across at 3 m with f = 1400 px is ~467 px/s.
    ff = ctl.compute(_est(goal[0], goal[1], du=467.0, dv=0.0, t=0.0), t_now=0.0)
    p_only = ctl.compute(_est(goal[0] + 467.0 * config.LATENCY_S, goal[1], t=0.0),
                         t_now=0.0)
    print(f"   d) on goal, 467 px/s    -> vel {ff.rate_a:8.1f} {ff.rate_b:8.1f}"
          f"   (feedforward; P term is zero)")
    print(f"      same lag, P term only-> vel {p_only.rate_a:8.1f} "
          f"{p_only.rate_b:8.1f}   -- {abs(ff.rate_a / max(p_only.rate_a, 1e-9)):.1f}x "
          f"smaller: this is why the velocity term exists")

    lat = ctl.compute(_est(goal[0], goal[1], du=467.0, dv=0.0, t=0.0), t_now=0.0)
    print(f"      predicted point at t+{ctl.latency_s * 1000:.0f} ms is "
          f"{lat.error_u:+.1f} px ahead of the goal")

    fast = ctl.compute(_est(goal[0] + 4000.0, goal[1] + 4000.0, du=9000.0,
                            dv=9000.0, t=0.0), t_now=0.0)
    print(f"   e) saturation           -> vel {fast.rate_a:8.1f} {fast.rate_b:8.1f}"
          f"   saturated = {fast.saturated} (clip +/-{config.MAX_MOTOR_RATE})")

    dot = ctl.compute(_est(goal[0], goal[1], t=0.0), t_now=0.0,
                      dot_pixel=(goal[0] + 20.0, goal[1] - 15.0))
    print(f"   f) dot-locked goal      -> goal ({dot.goal_u:.1f}, {dot.goal_v:.1f}) "
          f"dot_locked = {dot.dot_locked}")

    print()
    print(line)
    print("4. TRAVEL LIMITS")
    print(line)
    lim = ctl.limits
    print(f"   pitch {config.PITCH_LIMIT_DEG} yaw {config.YAW_LIMIT_DEG} "
          f"margin {lim.margin_deg} deg")
    print(f"   pose unknown -> guard passes motion, pose_known = {lim.pose_known}")
    lim.update_pose(pitch_deg=89.5, yaw_deg=0.0)
    into = ctl.compute(_est(goal[0] + 300.0, goal[1] + 300.0, t=0.0), t_now=0.0)
    print(f"   at pitch +89.5 deg      -> vel {into.rate_a:8.1f} {into.rate_b:8.1f}"
          f"   blocked = {lim.blocked}")
    lim.update_pose(pitch_deg=0.0, yaw_deg=0.0)
    free = ctl.compute(_est(goal[0] + 300.0, goal[1] + 300.0, t=0.0), t_now=0.0)
    print(f"   at pitch   0.0 deg      -> vel {free.rate_a:8.1f} {free.rate_b:8.1f}"
          f"   blocked = {lim.blocked}")

    print()
    print(line)
    print("5. LASER INTERLOCK -- every branch")
    print(line)
    t0 = 1000.0
    # goal_calibrated=True in the baseline so the scenarios below exercise the
    # condition each one is about. It is NOT the default -- see case q, and see
    # evaluate()'s docstring for why an omitted goal_calibrated fails safe.
    # goal_calibrated=True, a fresh drone box on the goal and a fresh head
    # report in the baseline so the scenarios below exercise the condition each
    # one is about. None of the three is a DEFAULT -- see cases q, s and v, and
    # see evaluate()'s docstring for why omitting any of them fails safe.
    ok = dict(t_now=t0, track_state=TrackState.TRACK, error_px=4.0,
              platform_rate_deg_s=1.0, last_vel_sent_t=t0 - 0.02,
              faces=[], aim=goal, target=goal, goal_calibrated=True,
              drone_box=_drone(goal[0], goal[1]), drone_box_t=t0 - 0.02,
              heads_t=t0 - 0.02)

    def fresh(t: float) -> dict:
        """Every time-relative field re-based to `t`.

        A scenario that moves the clock forward -- the duty cap does, by
        seconds -- would otherwise silently start testing box staleness
        instead of the thing it is about.
        """
        return {"t_now": t, "last_vel_sent_t": t - 0.02,
                "drone_box_t": t - 0.02, "heads_t": t - 0.02}

    def show(title: str, interlock: LaserInterlock, **over):
        kw = dict(ok)
        kw.update(over)
        state = interlock.evaluate(**kw)
        print(f"   {title:<26} -> {state.value:<18} "
              f"failed={interlock.failed or ['-']}")
        print(f"   {'':<26}    {interlock.reason}")
        return state

    disarmed = LaserInterlock(armed=config.LASER_ENABLED)
    print(f"   config.LASER_ENABLED = {config.LASER_ENABLED} (master arm)")
    show("a) disarmed", disarmed)

    il = LaserInterlock()
    il.arm()
    show("b) all conditions met", il)
    show("c) face on beam path", il, faces=[_face(goal[0] + 40, goal[1] + 20)])
    show("d) face far away", il, faces=[_face(50, 60)])
    # The 300 px sweep here is far wider than MAX_ERROR_TO_FIRE_PX allows in
    # practice; it exists to put the face at the far END of the path rather
    # than near the aim point. The box is widened to match so this case still
    # isolates face_clear -- a 132 px box cannot contain a 300 px sweep, and
    # without this it would trip drone_lock as well and stop testing what it
    # says it tests.
    show("e) face near target end", il,
         target=(goal[0] + 300, goal[1]), faces=[_face(goal[0] + 320, goal[1])],
         drone_box=_drone(goal[0] + 150.0, goal[1], w=500.0, h=500.0))
    show("f) unsettled platform", il, platform_rate_deg_s=12.0)
    show("g) error too large", il, error_px=config.MAX_ERROR_TO_FIRE_PX + 1.0)
    show("h) stale vel", il, last_vel_sent_t=t0 - 0.30)
    show("i) no vel ever sent", il, last_vel_sent_t=None)
    show("j) state ACQUIRE", il, track_state=TrackState.ACQUIRE)
    show("k) state COAST (off now)", il, track_state=TrackState.COAST)
    show("l) multiple failures", il, track_state=TrackState.COAST,
         platform_rate_deg_s=30.0, error_px=90.0,
         faces=[_face(goal[0], goal[1])])

    # Duty cap: hold every other condition true and let time run past
    # LASER_MAX_ON_MS, then confirm the cool-down keeps it off.
    duty = LaserInterlock(armed=True)
    t = t0
    duty.evaluate(**{**ok, **fresh(t)})
    t = t0 + config.LASER_MAX_ON_MS / 1000.0 + 0.1
    s = duty.evaluate(**{**ok, **fresh(t)})
    print(f"   {'m) duty cap exceeded':<26} -> {s.value:<18} "
          f"failed={duty.failed}")
    print(f"   {'':<26}    {duty.reason}")
    t += 0.5
    s = duty.evaluate(**{**ok, **fresh(t)})
    print(f"   {'n) still cooling down':<26} -> {s.value:<18} "
          f"failed={duty.failed}")
    t += LASER_COOLDOWN_S
    s = duty.evaluate(**{**ok, **fresh(t)})
    print(f"   {'o) cool-down elapsed':<26} -> {s.value:<18} "
          f"failed={duty.failed or ['-']}")

    # Disarming mid-fire must take effect on the very next evaluate().
    il.evaluate(**ok)
    il.disarm()
    s = il.evaluate(**ok)
    print(f"   {'p) disarmed mid-fire':<26} -> {s.value:<18} "
          f"failed={il.failed}")

    # The goal pixel is the point the face clearance is measured FROM. With no
    # goal_pixel.json it is the frame centre, which is where the camera points,
    # not where the beam goes -- so every other condition being true must NOT
    # produce FIRING. Two spellings, because the default is the one that
    # protects call sites written before this condition existed.
    il.arm()
    show("q) goal UNCALIBRATED", il, goal_calibrated=False)
    assert not il.on, "an uncalibrated goal pixel must not fire"
    bare = {k: v for k, v in ok.items() if k != "goal_calibrated"}
    s = il.evaluate(**bare)                  # the argument simply not passed
    print(f"   {'r) goal_calibrated omitted':<26} -> {s.value:<18} "
          f"failed={il.failed}")
    assert not il.on, "omitting goal_calibrated must not fire"
    assert il.evaluate(**ok) is LaserState.FIRING, \
        "the baseline scenario must still be able to fire"

    # -- FAIL-CLOSED: the beam is granted, never merely un-vetoed -------------
    # Every case below holds ALL the clearance conditions true -- no face on
    # the path, settled, small error, fresh vel, armed, calibrated -- and must
    # still refuse to fire, because nothing granted permission. A
    # clearance-only interlock fires on every one of them.
    print()
    print("   -- fail-closed: permission, not absence of veto --")

    show("s) no drone box", il, drone_box=None, drone_box_t=None)
    assert not il.on, "no detector box must not fire"
    assert il.state is LaserState.INHIBITED_NO_LOCK

    # The tracker holds TRACK across brief misses and keeps predicting, so
    # error stays small and `track` stays true through a coast. Only the
    # missing box distinguishes this from the baseline -- which is the entire
    # point of the condition.
    show("t) TRACK but box stale", il, drone_box_t=t0 - 0.5)
    assert not il.on, "a stale box is a statement about the past"

    show("u) beam outside the box", il,
         drone_box=_drone(goal[0] + 200.0, goal[1]))
    assert not il.on, "the beam must be inside the box, not merely near it"

    # Inside the raw box but inside the 15% inset margin: a box a few px loose
    # would put the beam past the airframe edge.
    show("v) beam on the box rim", il,
         drone_box=_drone(goal[0] + 60.0, goal[1]))
    assert not il.on, "the rim of the box is not a permitted aim point"

    # THE REGRESSION. YuNet fed a rotated frame returned [] on every frame of
    # an occupied room. `faces=[]` here is exactly that list; what separates it
    # from an empty room is whether a pass reported at all.
    show("w) head pass never ran", il, heads_t=None)
    assert not il.on, "an empty face list from a pass that never ran is not clear"
    assert il.state is LaserState.INHIBITED_FACE

    show("x) head pass stale", il, heads_t=t0 - 0.5)
    assert not il.on, "a stale head report is not clearance"

    # TRACK ON EITHER CAMERA, FIRE ONLY ON NARROW. A box mapped from the wide
    # camera is scaled ~2x, and its edge error with it -- which is precisely
    # the quantity containment measures. It may steer; it may not emit light.
    show("y) box came from wide", il, drone_box_source="wide->narrow")
    assert not il.on, "a mapped wide box must not grant permission to fire"
    assert il.state is LaserState.INHIBITED_NO_LOCK
    assert il.evaluate(**ok) is LaserState.FIRING, \
        "a narrow box must still fire after the wide case"

    assert il.evaluate(**ok) is LaserState.FIRING, \
        "the baseline must still fire after the fail-closed cases"

    print()
    print(line)
    print("6. REPORT (what the GUI reads)")
    print(line)
    il.arm()
    il.evaluate(**{**ok, "faces": [_face(goal[0] + 60, goal[1])],
                   "platform_rate_deg_s": 9.0})
    rep = il.report()
    for k in ("state", "on", "failed", "reason"):
        print(f"   {k:<12} {rep[k]}")
    print(f"   conditions   {rep['conditions']}")
    print(f"   margins      "
          f"{ {k: round(v, 2) for k, v in rep['margins'].items()} }")

    print()
    print("self-test complete -- no hardware touched")


if __name__ == "__main__":
    _main()
