"""Empirical calibrations: image Jacobian, goal pixel g(R), intrinsics, laser dot.

Everything in here is measured on the real machine and written to JSON under
`turret_host/calibration/`. Nothing here is derived from CAD or from a datasheet,
because every analytic path through this mechanism crosses at least one quantity
that is not trustworthy: the belt ratio, the differential joint order, the 11 deg
axis tilt, the narrow camera's 90 deg mount rotation (whose SIGN IS UNVERIFIED),
the M12 focal length (+/-10% part to part) and the direction of every motor. A
two-point measurement absorbs all of them at once and cannot get the sign wrong.

Design notes that look odd and are deliberate:

* Calibration is an OFFLINE, OPERATOR-PRESENT procedure with exclusive access to
  the hardware. It therefore does NOT import `cameras.py` or `link.py`: the
  tracking stack's capture threads and its fire-and-forget serial writer are
  built for a 30 Hz streaming loop, while this file needs blocking,
  request/response, one-move-at-a-time control. Two owners of one serial port is
  the exact failure the spec warns about. Every routine instead takes injected
  callables, so the GUI's calibration triggers can drive them with live stack
  objects when the stack is already up.
* Hardware is touched only inside `open()`/constructor bodies, never at import.

Protocols the routines expect (duck typed, no ABCs):
    grab()  -> np.ndarray     one FRESH narrow-camera BGR frame, blocking
    move(axis, steps) -> None one EXACT signed motor move, blocking until done;
                              axis is the firmware name "pan" (motor A) or
                              "tilt" (motor B). No backlash compensation --
                              the lash discipline belongs to the calibration.
    laser(on: bool) -> None   laser hard on/off
"""
from __future__ import annotations

# --------------------------------------------------------------------------
#   sys.path repair -- MUST run before any other import, and may use only
#   `os`/`sys`, which are both loaded before user code runs.
#
#   Running this file by path puts turret_host/ at sys.path[0], where types.py
#   SHADOWS THE STDLIB `types` MODULE. The next lazy stdlib import chain
#   (`threading -> functools -> from types import GenericAlias`, or
#   `re -> enum -> from types import MappingProxyType`) then picks up ours and
#   dies with a circular-import error that names this package and looks like
#   our bug. REPLACE the script directory with the project root: merely
#   inserting the root is not enough, because sys.path[0] still wins.
#
#   A no-op under `python -m turret_host.<mod>` and under a normal import.
# --------------------------------------------------------------------------
import os as _os
import sys as _sys

if __package__ in (None, ""):
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path[:] = [p for p in _sys.path
                    if _os.path.abspath(p or _os.getcwd()) != _here]
    _sys.path.insert(0, _os.path.dirname(_here))

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# The path repair that makes `python turret_host/calibrate.py` work is at the
# top of this file: it has to run before `import json`, not here.
from turret_host import config
from turret_host.types import Frame

# ==========================================================================
#   LOCAL TUNING -- not in config.py because nothing else needs them.
#   If any of these ever has to be shared, it moves to config.py.
# ==========================================================================
MOTOR_AXES = ("pan", "tilt")        # firmware names; column 0 = A = pan

JACOBIAN_STEPS = 200                # big enough for tens of px, small enough to stay in frame
TEMPLATE_HALF_PX = 24               # 49x49 patch: enough texture for TM_CCOEFF_NORMED
SEARCH_HALF_PX = 150                # search radius around the expected position
MIN_TEMPLATE_CORR = 0.55            # below this the feature is lost -- fail, do not guess
MIN_JACOBIAN_SHIFT_PX = 5.0         # a 200-step move that moves <5 px means something is wrong
MAX_JACOBIAN_SHIFT_PX = 400.0       # ...and >400 px means the feature is about to leave frame
MAX_JACOBIAN_COND = 30.0            # J_inv is what the loop uses; near-singular J is unusable
DEFAULT_SETTLE_S = 0.40             # mechanical ringing after a move, measured by eye

# Dot detection weights. DOT_CHROMA_TAU in config assumes an IR-cut sensor; the
# C270's filter is REMOVED, so near-IR passes every Bayer filter roughly equally
# and lit surfaces desaturate -- the true chroma of the dot can come in well
# under tau. The dot is, however, both the greenest AND the brightest thing in a
# 60 px window, so we add a local brightness-excess term and keep a chroma floor
# so that a plain white specular highlight still cannot pass on brightness alone.
DOT_BRIGHT_WEIGHT = 1.0
DOT_CHROMA_FLOOR_FRAC = 0.33        # chroma must reach 1/3 tau no matter how bright

# Goal-pixel fit. c is recovered from the DIFFERENCE of 1/R, so two nearby
# ranges make the fit explode. 2 m and 5 m give a spread of 0.30 1/m.
MIN_INV_RANGE_SPREAD = 0.08         # 1/m

GOOD_REPROJECTION_PX = 1.0          # above this, intrinsics are not good enough to trust
# Fisheye only wins if it wins clearly. On a lens the standard model can already
# describe, the two fits land within noise of each other, and the extra model is
# then pure liability: its distortion vector means something different and its
# undistort needs a different call, so a coin-flip between them would silently
# change what every consumer of intrinsics.json has to do.
FISHEYE_WIN_FRACTION = 0.95

CALIB_DIR = Path(__file__).resolve().parent / "calibration"


# ==========================================================================
#   STORAGE
# ==========================================================================
def _calib_path(name: str) -> Path:
    return CALIB_DIR / (name + ".json")


def save_calibration(name: str, data: dict) -> Path:
    """Write one calibration as JSON, atomically.

    Atomic because a half-written jacobian.json that still parses would be
    loaded silently at the next startup and send the turret off in some
    direction -- a truncated file must never look like a valid calibration.
    """
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["name"] = name
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = _calib_path(name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_calibration(name: str, required: bool = False) -> Optional[dict]:
    """Read one calibration. Missing is a legitimate state (config ships J=None).

    `required=True` raises instead -- use it wherever running uncalibrated would
    be worse than not running at all.
    """
    path = _calib_path(name)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                "no %s calibration at %s -- run `python -m turret_host.calibrate`"
                % (name, path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jacobian() -> Optional[np.ndarray]:
    """The 2x2 image Jacobian in px per motor step, or None if uncalibrated.

    control.py should prefer this over config.J_PX_PER_STEP, which is None until
    this file has run.
    """
    data = load_calibration("jacobian")
    if data is None:
        return None
    return np.asarray(data["J"], dtype=np.float64)


def load_goal_pixel() -> Optional[dict]:
    return load_calibration("goal_pixel")


def load_intrinsics(camera: str) -> Optional[dict]:
    return load_calibration("intrinsics_" + camera)


def goal_pixel_for_range(cal: dict, range_m: float) -> Tuple[float, float]:
    """g(R) = g_inf + c/R, per axis, in narrow-camera pixels.

    R is clamped to config.RANGE_LIMITS_M: outside the envelope the 1/R fit is
    an extrapolation, and a silly range estimate must not throw the aim point
    off the sensor.
    """
    lo, hi = config.RANGE_LIMITS_M
    r = min(max(float(range_m), lo), hi)
    g_inf = cal["g_inf"]
    c = cal["c"]
    return (g_inf[0] + c[0] / r, g_inf[1] + c[1] / r)


# ==========================================================================
#   LASER DOT  (opportunistic -- NOTHING in the tracking loop may gate on it)
# ==========================================================================
def _window_bounds(shape, around_xy, window_px) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    half = max(2, int(window_px) // 2)
    cx, cy = int(round(around_xy[0])), int(round(around_xy[1]))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half + 1)
    y1 = min(h, cy + half + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        raise ValueError("dot window %r falls outside the %dx%d frame"
                         % (around_xy, w, h))
    return x0, y0, x1, y1


def _blob_roundness(blob_u8: np.ndarray) -> float:
    """4*pi*A/P^2 with A as the pixel COUNT and P from the traced contour.

    On blobs this small (3-40 px) arcLength runs through pixel centres and so
    underestimates the perimeter, which pushes compact blobs above 1.0. That is
    fine: the test is here to throw out streaks and edges, and a 1x9 line still
    scores 0.44 while a 3x3 square scores 1.77.
    """
    cnts, _ = cv2.findContours(blob_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    contour = max(cnts, key=lambda c: len(c))
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0.0:
        return 0.0
    area = float(np.count_nonzero(blob_u8))
    return 4.0 * math.pi * area / (perimeter * perimeter)


def find_laser_dot(image: np.ndarray,
                   around_xy: Tuple[float, float],
                   window_px: int = config.DOT_WINDOW_PX,
                   tau: float = config.DOT_CHROMA_TAU,
                   area_px: Tuple[int, int] = config.DOT_AREA_PX,
                   roundness_min: float = config.DOT_ROUNDNESS_MIN,
                   ) -> Optional[Tuple[float, float]]:
    """Find the green dot near `around_xy`. Returns (u, v) in FULL-frame pixels.

    mask = (G - max(R, B)) > tau, as specified, PLUS a local brightness-excess
    term, because the C270 has no IR-cut filter: near-IR passes every Bayer
    filter roughly equally, the scene desaturates, and a pure chroma test tuned
    on an IR-cut sensor under-triggers. The dot is the greenest and the
    brightest thing in the window, so we use both and keep a chroma floor so a
    white highlight cannot pass on brightness alone.

    Returns None freely. Callers must treat that as "no information", never as
    an error -- the dot is invisible on dark cloth, in sunlight, and at range.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("find_laser_dot needs a BGR image")

    x0, y0, x1, y1 = _window_bounds(image.shape, around_xy, window_px)
    roi = image[y0:y1, x0:x1]

    bgr = roi.astype(np.int16)
    chroma = (bgr[:, :, 1] - np.maximum(bgr[:, :, 0], bgr[:, :, 2])).astype(np.float32)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Brightness EXCESS over the local background, not raw brightness: a bright
    # wall must not score, only a peak against its own surroundings.
    sigma = max(3.0, (x1 - x0) / 6.0)
    excess = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    score = chroma + DOT_BRIGHT_WEIGHT * excess
    mask = (score > float(tau)) & (chroma > float(tau) * DOT_CHROMA_FLOOR_FRAC)
    mask_u8 = (mask.astype(np.uint8)) * 255
    # Close: a dot bright enough to clip all three channels reads as WHITE at its
    # core, so chroma collapses exactly in the middle and the blob comes out as a
    # ring. Closing fills that hole instead of splitting one dot into arcs.
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    a_min, a_max = area_px
    yy, xx = np.indices(score.shape, dtype=np.float32)

    best = None
    best_weight = 0.0
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < a_min or area > a_max:
            continue
        blob = (labels == i).astype(np.uint8)
        if _blob_roundness(blob * 255) < roundness_min:
            continue
        w = np.where(blob > 0, np.maximum(score, 0.0), 0.0)
        total = float(w.sum())
        if total <= 0.0 or total <= best_weight:
            continue
        # Score-weighted centroid: sub-pixel, and biased towards the core of the
        # dot rather than the ragged edge of the threshold.
        best = (x0 + float((w * xx).sum()) / total,
                y0 + float((w * yy).sum()) / total)
        best_weight = total

    return best


def find_laser_dot_differential(on_image: np.ndarray,
                                off_image: np.ndarray,
                                around_xy: Tuple[float, float],
                                **kwargs) -> Optional[Tuple[float, float]]:
    """Same test, run on (laser on - laser off).

    The firmware's `laser pwm` prints the same hint: at LASER_PULSE_HZ = 15
    against 30 fps capture the dot is present on alternate frames, so
    differencing cancels the entire static scene -- including whatever near-IR
    is washing out the chroma -- and leaves the dot as almost the only signal.
    Used by the goal-pixel calibration, where the dot MUST be found.
    """
    diff = cv2.subtract(on_image, off_image)   # saturating: negatives clamp to 0
    return find_laser_dot(diff, around_xy, **kwargs)


# ==========================================================================
#   FEATURE MEASUREMENT for the Jacobian
# ==========================================================================
def _average_gray(grab: Callable[[], np.ndarray], n: int) -> np.ndarray:
    """Mean of n fresh frames as float32 gray.

    Averaging buys roughly sqrt(n) on sensor noise, and the Jacobian is measured
    once against a static scene, so there is no reason not to pay for it.
    """
    acc = None
    for _ in range(max(1, n)):
        img = grab()
        if img is None:
            raise RuntimeError("grab() returned None during calibration")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        acc = gray if acc is None else acc + gray
    return acc / float(max(1, n))


def _extract_template(gray: np.ndarray, p: Tuple[float, float], half: int) -> np.ndarray:
    h, w = gray.shape[:2]
    cx, cy = int(round(p[0])), int(round(p[1]))
    if cx - half < 0 or cy - half < 0 or cx + half + 1 > w or cy + half + 1 > h:
        raise ValueError("feature at %r is too close to the frame edge for a "
                         "%d px template" % (p, 2 * half + 1))
    return gray[cy - half:cy + half + 1, cx - half:cx + half + 1].copy()


def _subpixel_offset(resp: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """Parabola through the correlation peak and its two neighbours per axis."""
    def axis(a, b, c):        # a = left/up, b = peak, c = right/down
        denom = (a - 2.0 * b + c)
        if abs(denom) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))

    dx = dy = 0.0
    if 0 < x < resp.shape[1] - 1:
        dx = axis(resp[y, x - 1], resp[y, x], resp[y, x + 1])
    if 0 < y < resp.shape[0] - 1:
        dy = axis(resp[y - 1, x], resp[y, x], resp[y + 1, x])
    return dx, dy


def measure_feature(gray: np.ndarray,
                    template: np.ndarray,
                    expected_xy: Tuple[float, float],
                    search_half: int = SEARCH_HALF_PX,
                    min_corr: float = MIN_TEMPLATE_CORR) -> Tuple[float, float]:
    """Locate `template` near `expected_xy` to sub-pixel. Raises if it is lost.

    Raising is correct here: a Jacobian fitted to a mismatched feature is a
    plausible-looking 2x2 that sends the turret the wrong way, and the loop has
    no way to notice. Better to stop with the reason on the console.
    """
    th, tw = template.shape[:2]
    t_half_x, t_half_y = tw // 2, th // 2
    h, w = gray.shape[:2]

    sx0 = max(0, int(round(expected_xy[0])) - search_half - t_half_x)
    sy0 = max(0, int(round(expected_xy[1])) - search_half - t_half_y)
    sx1 = min(w, int(round(expected_xy[0])) + search_half + t_half_x + 1)
    sy1 = min(h, int(round(expected_xy[1])) + search_half + t_half_y + 1)
    search = gray[sy0:sy1, sx0:sx1]
    if search.shape[0] < th or search.shape[1] < tw:
        raise RuntimeError("search window collapsed at %r -- the feature has "
                           "left the frame" % (expected_xy,))

    resp = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(resp)
    if max_val < min_corr:
        raise RuntimeError("feature lost: best correlation %.2f < %.2f. Static "
                           "scene? Enough texture? Exposure hunting?"
                           % (max_val, min_corr))
    dx, dy = _subpixel_offset(resp, max_loc[0], max_loc[1])
    return (sx0 + max_loc[0] + dx + t_half_x,
            sy0 + max_loc[1] + dy + t_half_y)


def auto_pick_feature(gray: np.ndarray, margin: int) -> Tuple[float, float]:
    """Strongest corner in the central region, so the move cannot push it out."""
    h, w = gray.shape[:2]
    if margin * 2 + 20 > min(h, w):
        raise ValueError("frame too small for a %d px margin" % margin)
    mask = np.zeros((h, w), np.uint8)
    mask[margin:h - margin, margin:w - margin] = 255
    pts = cv2.goodFeaturesToTrack(gray.astype(np.uint8), maxCorners=1,
                                  qualityLevel=0.25, minDistance=40, mask=mask)
    if pts is None or len(pts) == 0:
        raise RuntimeError("no trackable feature in the central region -- point "
                           "the narrow camera at something with texture")
    return (float(pts[0, 0, 0]), float(pts[0, 0, 1]))


# ==========================================================================
#   1. IMAGE JACOBIAN
# ==========================================================================
def calibrate_jacobian(grab: Callable[[], np.ndarray],
                       move: Callable[[str, int], None],
                       p0: Optional[Tuple[float, float]] = None,
                       steps: int = JACOBIAN_STEPS,
                       repeats: int = 2,
                       settle_s: float = DEFAULT_SETTLE_S,
                       preload_steps: int = config.PRELOAD_STEPS,
                       avg_frames: int = 4,
                       min_shift_px: float = MIN_JACOBIAN_SHIFT_PX,
                       max_return_residual_px: float = 4.0,
                       max_cond: float = MAX_JACOBIAN_COND,
                       save: bool = True,
                       progress: Callable[[str], None] = print) -> dict:
    """Measure J, the 2x2 mapping motor steps -> narrow-camera pixels.

        J = [[ du/dA, du/dB ],
             [ dv/dA, dv/dB ]]     px per motor step

    Point the narrow camera at a STATIC, textured scene. We lock onto one
    feature, move motor A (firmware `pan`) +steps, measure, come back, then the
    same for motor B (`tilt`).

    200 steps is chosen so the shift is tens of pixels -- well above measurement
    noise -- while the feature stays in frame.

    BACKLASH: config.BACKLASH_DEG is 0.57 deg, about 12 microsteps against a
    200 step move -- 6%, not negligible. It is removed by APPROACH DISCIPLINE
    rather than by a correction term: every pose in this routine is reached
    moving in the + direction, so the flanks stay loaded on the same side
    throughout and the lash never enters the measured shift. A negative move of
    d steps is therefore done as -(d + preload) then +preload.

    The return leg is measured too. If the feature does not come back to where
    it started, steps were lost and the whole measurement is void -- we raise
    rather than bake the error into J.

    J absorbs the belt ratio, the differential, the joint order, the 11 deg axis
    tilt, the focal length, the 90 deg camera rotation and every sign at once,
    which is why the control loop tolerates it being 20% wrong.
    """
    if steps <= 0:
        raise ValueError("steps must be positive; the approach discipline "
                         "requires a known direction")

    def settle():
        time.sleep(settle_s)

    def goto_reference(axis: str):
        """Return to the current pose with the flanks loaded in +."""
        move(axis, -preload_steps)
        time.sleep(0.05)
        move(axis, +preload_steps)
        settle()

    progress("preloading both axes against the lash...")
    for axis in MOTOR_AXES:
        goto_reference(axis)

    ref_gray = _average_gray(grab, avg_frames)
    frame_h, frame_w = ref_gray.shape[:2]
    if p0 is None:
        p0 = auto_pick_feature(ref_gray, margin=TEMPLATE_HALF_PX + 60)
        progress("auto-picked feature at (%.1f, %.1f)" % p0)
    template = _extract_template(ref_gray, p0, TEMPLATE_HALF_PX)

    columns: Dict[str, np.ndarray] = {}
    scatter: Dict[str, float] = {}
    residuals: Dict[str, float] = {}

    for axis in MOTOR_AXES:
        per_repeat: List[np.ndarray] = []
        worst_residual = 0.0
        for k in range(max(1, repeats)):
            progress("%s: repeat %d/%d" % (axis, k + 1, max(1, repeats)))
            goto_reference(axis)
            base = np.array(measure_feature(_average_gray(grab, avg_frames),
                                            template, p0), dtype=np.float64)

            move(axis, +steps)
            settle()
            moved = np.array(measure_feature(_average_gray(grab, avg_frames),
                                             template, base), dtype=np.float64)

            # Come back the long way round so the final motion is still +.
            move(axis, -(steps + preload_steps))
            time.sleep(0.05)
            move(axis, +preload_steps)
            settle()
            back = np.array(measure_feature(_average_gray(grab, avg_frames),
                                            template, base), dtype=np.float64)

            residual = float(np.hypot(*(back - base)))
            worst_residual = max(worst_residual, residual)
            if residual > max_return_residual_px:
                raise RuntimeError(
                    "%s did not return to its start: %.1f px off after +%d/-%d "
                    "steps. Lost steps, a moving scene, or the preload is too "
                    "small for the lash. The measurement is void."
                    % (axis, residual, steps, steps))

            shift = moved - base
            magnitude = float(np.hypot(*shift))
            if magnitude < min_shift_px:
                raise RuntimeError(
                    "%s moved %d steps and the feature moved only %.1f px. "
                    "Motor not enabled, belt slipping, or the move never "
                    "reached the board." % (axis, steps, magnitude))
            if magnitude > MAX_JACOBIAN_SHIFT_PX:
                raise RuntimeError(
                    "%s moved %d steps and the feature moved %.0f px -- too "
                    "far to trust; reduce `steps`." % (axis, steps, magnitude))
            per_repeat.append(shift)
            progress("   shift %+.2f, %+.2f px   return residual %.2f px"
                     % (shift[0], shift[1], residual))

        stack = np.vstack(per_repeat)
        columns[axis] = stack.mean(axis=0)
        scatter[axis] = float(np.hypot(*stack.std(axis=0))) if len(stack) > 1 else 0.0
        residuals[axis] = worst_residual

    J = np.column_stack([columns["pan"] / float(steps),
                         columns["tilt"] / float(steps)])

    det = float(np.linalg.det(J))
    cond = float(np.linalg.cond(J))
    if not np.isfinite(cond) or cond > max_cond:
        raise RuntimeError(
            "J is near-singular (cond %.1f, det %.3e): the two motors move the "
            "feature along nearly the same pixel direction, so J_inv would "
            "amplify noise without bound. Check that both axes actually moved "
            "and that the camera is not looking straight down an axis."
            % (cond, det))

    J_inv = np.linalg.inv(J)
    # Angle between the two columns: 90 deg is ideal, small is the failure above.
    ca = columns["pan"] / (np.linalg.norm(columns["pan"]) + 1e-12)
    cb = columns["tilt"] / (np.linalg.norm(columns["tilt"]) + 1e-12)
    column_angle_deg = float(math.degrees(math.acos(float(np.clip(ca @ cb, -1, 1)))))

    result = {
        "J": J.tolist(),
        # Same matrix under the key control.py's Jacobian.load() looks for, so
        # this file can be handed straight to it. One measurement, two names --
        # never two matrices.
        "j_px_per_step": J.tolist(),
        "J_inv": J_inv.tolist(),
        "units": "pixels per motor microstep; columns = (motor A/pan, motor B/tilt)",
        "steps": int(steps),
        "repeats": int(max(1, repeats)),
        "preload_steps": int(preload_steps),
        "p0": [float(p0[0]), float(p0[1])],
        "shift_pan_px": columns["pan"].tolist(),
        "shift_tilt_px": columns["tilt"].tolist(),
        "repeat_scatter_px": scatter,
        "return_residual_px": residuals,
        "det": det,
        "cond": cond,
        "column_angle_deg": column_angle_deg,
        "image_size": [int(frame_w), int(frame_h)],
    }

    progress("J = [[%+.5f, %+.5f], [%+.5f, %+.5f]] px/step"
             % (J[0, 0], J[0, 1], J[1, 0], J[1, 1]))
    progress("columns %.1f deg apart, cond %.2f, scatter %.2f/%.2f px"
             % (column_angle_deg, cond, scatter["pan"], scatter["tilt"]))
    if save:
        progress("saved -> %s" % save_calibration("jacobian", result))
    return result


# ==========================================================================
#   2. GOAL PIXEL  g(R) = g_inf + c/R
# ==========================================================================
def measure_dot_at_range(grab: Callable[[], np.ndarray],
                         laser: Callable[[bool], None],
                         around_xy: Optional[Tuple[float, float]] = None,
                         samples: int = 15,
                         window_px: int = config.DOT_WINDOW_PX,
                         min_hits: int = 3,
                         progress: Callable[[str], None] = print) -> Tuple[float, float]:
    """Median laser-dot pixel over `samples` on/off pairs.

    Note the inversion of the usual rule: in the tracking loop the dot is
    OPPORTUNISTIC and nothing may gate on it, but this is a calibration whose
    entire purpose is to measure where the dot lands, so not finding it is a
    hard failure and we say so.
    """
    # One frame before anything is energised: proves the camera is delivering,
    # and gives the frame size for the default window centre.
    first = grab()
    if around_xy is None:
        around_xy = (first.shape[1] / 2.0, first.shape[0] / 2.0)

    hits: List[Tuple[float, float]] = []
    try:
        for _ in range(max(1, samples)):
            laser(True)
            time.sleep(0.06)        # let one whole frame be exposed with it on
            on_img = grab()
            laser(False)
            time.sleep(0.06)
            off_img = grab()
            p = find_laser_dot_differential(on_img, off_img, around_xy,
                                            window_px=window_px)
            if p is not None:
                hits.append(p)
    finally:
        # The laser goes off even if this raises. This is the one place in the
        # stack where an exception path must still touch hardware.
        laser(False)

    if len(hits) < min_hits:
        raise RuntimeError(
            "found the dot in only %d of %d frames near (%.0f, %.0f). Aim at a "
            "matte light surface, dim the room, and click closer to the dot."
            % (len(hits), samples, around_xy[0], around_xy[1]))

    arr = np.array(hits, dtype=np.float64)
    med = np.median(arr, axis=0)
    mad = float(np.median(np.hypot(arr[:, 0] - med[0], arr[:, 1] - med[1])))
    progress("   dot at (%.2f, %.2f) px from %d/%d frames, spread %.2f px"
             % (med[0], med[1], len(hits), samples, mad))
    return (float(med[0]), float(med[1]))


def fit_goal_pixel(observations: Sequence[Tuple[float, float, float]],
                   min_inv_range_spread: float = MIN_INV_RANGE_SPREAD) -> dict:
    """Least-squares fit of g(R) = g_inf + c/R per axis.

    observations: [(range_m, u_px, v_px), ...], at least two.

    1/R is the right basis because the parallax between the laser aperture and
    the narrow camera subtends f * baseline / R pixels, plus a fixed boresight
    offset that survives to infinity. But the CONSTANTS come from the fit, not
    from the CAD offsets -- fitting is what makes the narrow camera's 90 deg
    rotation and its unverified sign a non-issue: whatever they are, they are
    already inside g_inf and c.
    """
    obs = [(float(r), float(u), float(v)) for r, u, v in observations]
    if len(obs) < 2:
        raise ValueError("g(R) needs at least two measured ranges")

    R = np.array([o[0] for o in obs], dtype=np.float64)
    if np.any(R <= 0.0):
        raise ValueError("ranges must be positive metres")
    x = 1.0 / R
    spread = float(x.max() - x.min())
    if spread < min_inv_range_spread:
        raise ValueError(
            "ranges %s are too close together: spread in 1/R is %.3f 1/m, below "
            "%.3f. c is recovered from that difference, so a short baseline in "
            "1/R makes the fit meaningless. Use something like 2 m and 5 m."
            % (R.tolist(), spread, min_inv_range_spread))

    lo, hi = config.RANGE_LIMITS_M
    outside = [r for r in R if r < lo or r > hi]

    A = np.column_stack([np.ones_like(x), x])
    g_inf = []
    c = []
    resid = []
    for axis_idx in (1, 2):
        y = np.array([o[axis_idx] for o in obs], dtype=np.float64)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        g_inf.append(float(coef[0]))
        c.append(float(coef[1]))
        resid.append(float(np.sqrt(np.mean((A @ coef - y) ** 2))))

    # Sanity only, never a gate: the implied baseline should land near
    # LASER_TO_NARROW_MM, and with the camera rotated 90 deg most of it should
    # sit on the ROW (v). If it does not, the fit is telling you something true
    # about the machine that the CAD does not know.
    implied_mm = [1000.0 * c[0] / config.NARROW_F_PX,
                  1000.0 * c[1] / config.NARROW_F_PX]

    return {
        "model": "g(R) = g_inf + c/R, narrow-camera pixels, R in metres",
        "g_inf": g_inf,
        "c": c,
        # The same four numbers under the names control.py's GoalModel.load()
        # requires, so this file loads there unmodified.
        "u_inf": g_inf[0],
        "c_u": c[0],
        "v_inf": g_inf[1],
        "c_v": c[1],
        "observations": [list(o) for o in obs],
        "fit_residual_px": resid,
        "inv_range_spread": spread,
        "implied_baseline_mm": implied_mm,
        "cad_baseline_mm": config.LASER_TO_NARROW_MM,
        "ranges_outside_envelope_m": outside,
    }


def calibrate_goal_pixel(observations: Optional[Sequence[Tuple[float, float, float]]] = None,
                         grab: Optional[Callable[[], np.ndarray]] = None,
                         laser: Optional[Callable[[bool], None]] = None,
                         ranges_m: Sequence[float] = (2.0, 5.0),
                         around_xy: Optional[Tuple[float, float]] = None,
                         samples: int = 15,
                         confirm: Optional[Callable[[str], None]] = None,
                         save: bool = True,
                         progress: Callable[[str], None] = print) -> dict:
    """Fit g(R) from the laser dot at two (or more) MEASURED ranges.

    Either pass `observations` [(range_m, u, v), ...] already measured, or pass
    `grab`/`laser`/`confirm` and let this walk the operator through it.

    Do NOT replace this with an analytic parallax formula derived from the CAD
    offsets: the CAD does not know which way the narrow camera is rotated, and
    that sign is the difference between correcting the parallax and doubling it.
    """
    if observations is None:
        if grab is None or laser is None:
            raise ValueError("need grab() and laser() to measure g(R), or pass "
                             "observations directly")
        if confirm is None:
            raise ValueError("interactive g(R) turns the laser ON -- pass a "
                             "confirm() the operator has to answer")
        measured = []
        for r in ranges_m:
            confirm("Place a matte target at EXACTLY %.2f m from the laser "
                    "aperture, clear the room of eyes, then continue" % r)
            progress("measuring dot at %.2f m..." % r)
            u, v = measure_dot_at_range(grab, laser, around_xy=around_xy,
                                        samples=samples, progress=progress)
            measured.append((float(r), u, v))
            # Re-centre the search on the last hit: at the next range the dot
            # moves by the parallax, which is exactly what we are measuring, but
            # it moves by far less than the window from where it just was.
            around_xy = (u, v)
        observations = measured

    result = fit_goal_pixel(observations)
    progress("g_inf = (%.2f, %.2f) px   c = (%.1f, %.1f) px*m"
             % (result["g_inf"][0], result["g_inf"][1],
                result["c"][0], result["c"][1]))
    progress("implied baseline (%.1f, %.1f) mm vs %.1f mm in CAD -- FYI only"
             % (result["implied_baseline_mm"][0],
                result["implied_baseline_mm"][1],
                config.LASER_TO_NARROW_MM))
    for r in (config.RANGE_LIMITS_M[0], config.ASSUMED_RANGE_M, config.RANGE_LIMITS_M[1]):
        g = goal_pixel_for_range(result, r)
        progress("   g(%.1f m) = (%.1f, %.1f)" % (r, g[0], g[1]))
    if save:
        progress("saved -> %s" % save_calibration("goal_pixel", result))
    return result


# ==========================================================================
#   3. CAMERA INTRINSICS
# ==========================================================================
def _chessboard_corners(gray: np.ndarray, board: Tuple[int, int]):
    """Inner-corner detection. SB is used because the IR-cut-free sensor gives
    low-contrast, colour-cast images that the classic detector drops."""
    found, corners = cv2.findChessboardCornersSB(
        gray, board, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
    return bool(found), corners


def _object_grid(board: Tuple[int, int], square_mm: float) -> np.ndarray:
    objp = np.zeros((board[0] * board[1], 3), np.float64)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)
    return objp * float(square_mm)


def _mean_reprojection_error(obj_pts, img_pts, rvecs, tvecs, K, dist, fisheye: bool) -> float:
    total_sq = 0.0
    total_n = 0
    for i in range(len(obj_pts)):
        if fisheye:
            # (1, N, 3) layout, same as cv2.fisheye.calibrate demands below.
            proj, _ = cv2.fisheye.projectPoints(
                obj_pts[i].reshape(1, -1, 3), rvecs[i], tvecs[i], K, dist)
        else:
            proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        meas = img_pts[i].reshape(-1, 2)
        total_sq += float(np.sum((proj - meas) ** 2))
        total_n += len(meas)
    return math.sqrt(total_sq / max(1, total_n))


def calibrate_camera_intrinsics(images: Sequence,
                                camera: str = "narrow",
                                board: Tuple[int, int] = (9, 6),
                                square_mm: float = 25.0,
                                try_fisheye: Optional[bool] = None,
                                save: bool = True,
                                progress: Callable[[str], None] = print) -> dict:
    """Chessboard intrinsics. For the wide camera, fit BOTH models and keep the
    better one by reprojection error.

    `images` may be ndarrays or paths. `board` is INNER corners (cols, rows).

    The wide module is a 2.1 mm M12 at ~115 deg diagonal. The standard
    Brown-Conrady model runs out of expressiveness on a lens that wide, so we
    also fit cv2.fisheye and keep whichever reprojects tighter. Whichever wins,
    the winner is recorded by name -- an undistort that assumes the wrong model
    is worse than none.
    """
    if try_fisheye is None:
        try_fisheye = (camera == "wide")
    if len(images) < 6:
        raise ValueError("need at least 6 chessboard views (10-20 is better), "
                         "got %d" % len(images))

    objp = _object_grid(board, square_mm)
    obj_pts: List[np.ndarray] = []
    img_pts: List[np.ndarray] = []
    size = None
    for idx, item in enumerate(images):
        if isinstance(item, (str, Path)):
            img = cv2.imread(str(item), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError("cannot read chessboard image %s" % item)
        else:
            img = item
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif size != (gray.shape[1], gray.shape[0]):
            raise ValueError("view %d is %dx%d but the first was %dx%d -- one "
                             "calibration, one resolution"
                             % (idx, gray.shape[1], gray.shape[0], size[0], size[1]))
        found, corners = _chessboard_corners(gray, board)
        if not found:
            progress("view %d: board not found, skipped" % idx)
            continue
        obj_pts.append(objp.copy())
        img_pts.append(corners.reshape(-1, 1, 2).astype(np.float64))

    if len(obj_pts) < 6:
        raise RuntimeError("only %d usable views out of %d -- move the board "
                           "around more (corners, tilts, near and far)"
                           % (len(obj_pts), len(images)))
    progress("%d usable views at %dx%d" % (len(obj_pts), size[0], size[1]))

    # ---- pinhole -------------------------------------------------------
    rms_p, K_p, dist_p, rvecs_p, tvecs_p = cv2.calibrateCamera(
        [o.astype(np.float32) for o in obj_pts],
        [p.astype(np.float32) for p in img_pts],
        size, None, None)
    err_p = _mean_reprojection_error(obj_pts, img_pts, rvecs_p, tvecs_p,
                                    K_p, dist_p, fisheye=False)
    progress("pinhole:  rms %.3f px   mean reprojection %.3f px" % (rms_p, err_p))

    best = {
        "model": "pinhole",
        "K": np.asarray(K_p).tolist(),
        "dist": np.asarray(dist_p).reshape(-1).tolist(),
        "reprojection_px": err_p,
    }
    err_f = None

    # ---- fisheye -------------------------------------------------------
    if try_fisheye:
        K_f = np.zeros((3, 3))
        D_f = np.zeros((4, 1))
        # CALIB_CHECK_COND is deliberately NOT set: it aborts the entire fit on
        # one ill-conditioned view, which on a hand-held board is common and
        # tells us nothing. Fit, then judge by reprojection error.
        # OpenCV 5.0 moved these flags out of the cv2.fisheye namespace and onto
        # cv2 itself -- cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC no longer exists.
        flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        # The (1, N, k) layout is not optional: given the (N, 1, k) layout that
        # cv2.calibrateCamera is happy with, cv2.fisheye.calibrate dies inside
        # arithm_op with a size mismatch that names neither argument.
        rms_f, K_f, D_f, rvecs_f, tvecs_f = cv2.fisheye.calibrate(
            [o.reshape(1, -1, 3) for o in obj_pts],
            [p.reshape(1, -1, 2) for p in img_pts],
            size, K_f, D_f, None, None, flags, criteria)
        err_f = _mean_reprojection_error(obj_pts, img_pts, rvecs_f, tvecs_f,
                                         K_f, D_f, fisheye=True)
        progress("fisheye:  rms %.3f px   mean reprojection %.3f px" % (rms_f, err_f))
        if err_f < err_p * FISHEYE_WIN_FRACTION:
            best = {
                "model": "fisheye",
                "K": np.asarray(K_f).tolist(),
                "dist": np.asarray(D_f).reshape(-1).tolist(),
                "reprojection_px": err_f,
            }

    if best["reprojection_px"] > GOOD_REPROJECTION_PX:
        progress("WARNING: best model reprojects at %.2f px, above the %.1f px "
                 "target. More views, flatter board, better light."
                 % (best["reprojection_px"], GOOD_REPROJECTION_PX))

    fx, fy = best["K"][0][0], best["K"][1][1]
    reference = config.NARROW_F_PX if camera == "narrow" else config.WIDE_F_PX
    progress("f_px measured (%.1f, %.1f) vs %.1f in config -- config is an "
             "estimate; this measurement wins" % (fx, fy, reference))

    result = dict(best)
    result.update({
        "camera": camera,
        "image_size": [int(size[0]), int(size[1])],
        "board": [int(board[0]), int(board[1])],
        "square_mm": float(square_mm),
        "views_used": len(obj_pts),
        "reprojection_pinhole_px": err_p,
        "reprojection_fisheye_px": err_f,
        "config_f_px_reference": reference,
    })
    if save:
        progress("saved -> %s" % save_calibration("intrinsics_" + camera, result))
    return result


# ==========================================================================
#   HARDWARE ACCESS  (constructors/open() only -- never at import)
# ==========================================================================
class CalibrationCapture:
    """Minimal blocking capture for calibration only.

    Deliberately not cameras.py: that module runs grab threads and Slots for a
    30 Hz loop, and a calibration wants "give me one frame, now". The rules it
    does share with cameras.py are the ones that were paid for in debugging:
    CAP_MSMF (DSHOW silently delivers 10 fps), forced MJPEG (the YUYV modes are
    3-6x slower), grab-then-retrieve, and always release().
    """

    def __init__(self, index: int, size=config.NARROW_SIZE, name: str = "narrow"):
        self.index = int(index)
        self.size = size
        self.name = name
        self.cap = None            # nothing is opened until open()
        self._count = 0

    def open(self) -> "CalibrationCapture":
        cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not cap.isOpened():
            raise RuntimeError("camera index %d did not open on MSMF" % self.index)
        # MSMF REFUSES the FOURCC set and returns False. That is normal -- it
        # negotiates the compressed mode itself. Do not "fix" this with DSHOW.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap = cap
        for _ in range(5):          # let auto-exposure/auto-white-balance settle
            cap.grab()
        return self

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("CalibrationCapture.open() was never called")
        if not self.cap.grab():
            raise RuntimeError("grab() failed on camera %d" % self.index)
        ok, img = self.cap.retrieve()
        if not ok or img is None:
            raise RuntimeError("retrieve() failed on camera %d" % self.index)
        self._count += 1
        return img

    def read_frame(self) -> Frame:
        if self.cap is None:
            raise RuntimeError("CalibrationCapture.open() was never called")
        if not self.cap.grab():
            raise RuntimeError("grab() failed on camera %d" % self.index)
        t = time.perf_counter()     # taken at grab() return, as types.Frame documents
        ok, img = self.cap.retrieve()
        if not ok or img is None:
            raise RuntimeError("retrieve() failed on camera %d" % self.index)
        self._count += 1
        return Frame(image=img, t=t, index=self._count, camera=self.name)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()      # a leaked handle costs the NEXT run, not this one
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False


class CalibrationLink:
    """Blocking request/response console client for the Pico, for calibration.

    link.py owns the port during tracking and writes `vel` without waiting for
    anything. This does the opposite: one command, one complete reply, verified.
    Only one of the two may hold the port at a time -- two writers on one serial
    link is the contention failure the spec calls out.
    """

    def __init__(self, port: Optional[str] = None, default_rate: Optional[int] = 1200):
        self.port = port
        self.default_rate = default_rate
        self.ser = None            # nothing is opened until open()

    @staticmethod
    def find_port() -> str:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            if p.vid == config.PICO_VID and p.pid == config.PICO_PID:
                return p.device
        raise RuntimeError("no %04x:%04x board found -- resolve by VID/PID, "
                           "never by COM number" % (config.PICO_VID, config.PICO_PID))

    def open(self) -> "CalibrationLink":
        import serial
        if self.port is None:
            self.port = self.find_port()
        self.ser = serial.Serial(self.port, config.SERIAL_BAUD, timeout=0.3)
        time.sleep(0.3)
        self._drain(0.8)            # swallow the banner, not errors
        return self

    def _drain(self, quiet_s: float, timeout_s: float = 20.0) -> str:
        chunks = []
        last = time.monotonic()
        deadline = last + timeout_s
        while time.monotonic() < deadline:
            n = self.ser.in_waiting
            if n:
                chunks.append(self.ser.read(n))
                last = time.monotonic()
            elif chunks and (time.monotonic() - last) > quiet_s:
                break
            else:
                time.sleep(0.01)
        return b"".join(chunks).decode("utf-8", "replace")

    def command(self, text: str, quiet_s: float = 0.25, timeout_s: float = 30.0) -> str:
        """Send one line, read the whole reply, raise on anything the firmware
        calls an error. The firmware prints `error:`/`usage error:` and carries
        on; swallowing that here would turn a refused move into a silent one."""
        if self.ser is None:
            raise RuntimeError("CalibrationLink.open() was never called")
        self.ser.reset_input_buffer()
        self.ser.write((text + "\r\n").encode())
        self.ser.flush()
        reply = self._drain(quiet_s, timeout_s)
        if not reply.strip():
            raise TimeoutError("no reply to %r in %.0f s" % (text, timeout_s))
        low = reply.lower()
        for bad in ("error:", "unknown command", "refusing to move", "traceback"):
            if bad in low:
                raise RuntimeError("board rejected %r:\n%s" % (text, reply.strip()))
        return reply

    def move_motor(self, axis: str, steps: int, rate: Optional[int] = None) -> None:
        """One EXACT signed motor move, blocking. No lash compensation here --
        the calibration owns the approach direction."""
        if axis not in MOTOR_AXES:
            raise ValueError("axis must be 'pan' or 'tilt', got %r" % axis)
        steps = int(steps)
        if steps == 0:
            return
        rate = self.default_rate if rate is None else rate
        cmd = "move %s %d" % (axis, steps)
        if rate:
            cmd += " %d" % int(rate)
        reply = self.command(cmd, timeout_s=30.0)
        if "STOPPED EARLY" in reply:
            raise RuntimeError("endstop stopped the move short:\n%s" % reply.strip())
        # Verify the board actually moved what we asked. A short move silently
        # scales the Jacobian, and nothing downstream could ever detect it.
        moved = None
        for token in reply.replace("\n", " ").split():
            if token.startswith(("+", "-")) and token[1:].isdigit():
                moved = int(token)
                break
        if moved is None or moved != steps:
            raise RuntimeError("asked %s for %+d microsteps, board reported %s:\n%s"
                               % (axis, steps, moved, reply.strip()))

    def laser(self, on: bool, force: bool = False) -> None:
        """5 mW green. Only ever called from an operator-driven calibration."""
        self.command("laser on force" if (on and force) else ("laser on" if on else "laser off"))

    def state(self) -> dict:
        """The board's `state` JSON.

        The firmware prints it as `STATE {...}`, not as a bare object -- the
        prefix is what lets a reader pick the payload out of a stream that
        also carries the echoed command and the prompt. This used to require
        the line to START with "{" and so never matched, which turned every
        caller into "no JSON state line in:" followed by the JSON it was
        looking at. Take the first "{" on any line that has one.
        """
        reply = self.command("state")
        for line in reply.splitlines():
            i = line.find("{")
            if i >= 0:
                try:
                    return json.loads(line[i:].strip())
                except ValueError:
                    continue          # an echoed command can contain a brace
        raise RuntimeError("no JSON state line in:\n%s" % reply.strip())

    def stop(self) -> None:
        self.command("stop")

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False


# ==========================================================================
#   INTERACTIVE HELPERS  (used by __main__; need a window)
# ==========================================================================
def probe_camera_indices(max_index: int = 8) -> List[Tuple[int, int, int]]:
    """Open each index in turn on MSMF and report its resolution.

    SEQUENTIALLY, never concurrently: MSMF's source reader does not tolerate
    being raced and a concurrent second open never becomes ready.
    """
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            found.append((i, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
    return found


def pick_point(grab: Callable[[], np.ndarray],
               title: str = "click the feature - ENTER to accept, ESC to cancel"
               ) -> Optional[Tuple[float, float]]:
    """Live view; click a point, ENTER accepts. Returns full-resolution pixels."""
    picked: List[Tuple[float, float]] = []
    scale = [1.0]

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked.append((x / scale[0], y / scale[0]))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while True:
            img = grab()
            scale[0] = min(1.0, config.DISPLAY_SIZE[0] * 2.0 / img.shape[1])
            view = cv2.resize(img, None, fx=scale[0], fy=scale[0])
            if picked:
                p = picked[-1]
                cv2.drawMarker(view, (int(p[0] * scale[0]), int(p[1] * scale[0])),
                               (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and picked:
                return picked[-1]
            if key == 27:
                return None
    finally:
        cv2.destroyWindow(title)


def capture_chessboard_views(grab: Callable[[], np.ndarray],
                             board: Tuple[int, int],
                             n_views: int = 15) -> List[np.ndarray]:
    """SPACE captures a view (only if the board is detected), ESC finishes."""
    title = "chessboard - SPACE to capture, ESC when done"
    views: List[np.ndarray] = []
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        while len(views) < n_views:
            img = grab()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = _chessboard_corners(gray, board)
            scale = min(1.0, config.DISPLAY_SIZE[0] * 2.0 / img.shape[1])
            view = cv2.resize(img, None, fx=scale, fy=scale)
            if found:
                cv2.drawChessboardCorners(view, board, corners * scale, True)
            cv2.putText(view, "%d/%d  %s" % (len(views), n_views,
                                             "BOARD OK" if found else "no board"),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xFF
            if key == 32 and found:
                views.append(img.copy())
            elif key == 27:
                break
    finally:
        cv2.destroyWindow(title)
    return views


def _ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = "" if default is None else " [%s]" % default
    answer = input("%s%s: " % (prompt, suffix)).strip()
    return answer if answer else (default or "")


def _ask_float(prompt: str, default: float) -> float:
    return float(_ask(prompt, "%g" % default))


def _ask_int(prompt: str, default: int) -> int:
    return int(_ask(prompt, "%d" % default))


def _confirm(message: str) -> None:
    answer = input("%s [y/N]: " % message).strip().lower()
    if answer not in ("y", "yes"):
        raise KeyboardInterrupt("operator declined: %s" % message)


def _open_camera_interactive(name: str) -> CalibrationCapture:
    print("probing camera indices (sequentially -- MSMF cannot be raced)...")
    for idx, w, h in probe_camera_indices():
        print("   index %d: %dx%d" % (idx, w, h))
    index = _ask_int("%s camera index" % name, 0)
    size = config.NARROW_SIZE if name == "narrow" else config.WIDE_SIZE
    cam = CalibrationCapture(index, size=size, name=name).open()
    img = cam.read()
    print("opened index %d, delivering %dx%d" % (index, img.shape[1], img.shape[0]))
    return cam


# ==========================================================================
#   MENU
# ==========================================================================
def _menu_jacobian() -> None:
    # Nested finallys, not one: if the board fails to open, the camera must
    # still be released. A leaked camera handle is not this run's problem, it is
    # the next run's -- isOpened() returns True and every read() fails.
    cam = _open_camera_interactive("narrow")
    try:
        link = CalibrationLink().open()
        try:
            print("point the narrow camera at a STATIC, textured scene.")
            p0 = None
            if _ask("pick the feature by hand? (y/n)", "n").lower().startswith("y"):
                p0 = pick_point(cam.read)
                if p0 is None:
                    print("cancelled.")
                    return
            steps = _ask_int("steps per probe", JACOBIAN_STEPS)
            repeats = _ask_int("repeats per axis", 2)
            _confirm("BOTH MOTORS WILL MOVE +-%d steps. Mechanism clear?"
                     % (steps + config.PRELOAD_STEPS))
            calibrate_jacobian(cam.read, link.move_motor, p0=p0,
                               steps=steps, repeats=repeats)
        finally:
            link.close()
    finally:
        cam.close()


def _menu_goal_pixel() -> None:
    cam = _open_camera_interactive("narrow")
    try:
        link = CalibrationLink().open()
        try:
            r1 = _ask_float("near range, metres", 2.0)
            r2 = _ask_float("far range, metres", 5.0)
            print("LASER SAFETY: 5 mW green. Eyes out of the beam path, "
                  "target matte.")
            _confirm("Ready to fire the laser at a target?")
            print("click the dot in the live view so the search window lands on it.")
            link.laser(True)
            around = pick_point(cam.read, "click the LASER DOT - ENTER accepts")
            link.laser(False)
            if around is None:
                print("cancelled.")
                return
            calibrate_goal_pixel(grab=cam.read, laser=link.laser,
                                 ranges_m=(r1, r2), around_xy=around,
                                 confirm=_confirm)
        finally:
            # The laser goes off before the port is given up, and a failure to
            # turn it off is allowed to propagate: silence there is the one
            # outcome worse than a crash.
            link.laser(False)
            link.close()
    finally:
        cam.close()


def _menu_intrinsics(camera: str) -> None:
    cam = _open_camera_interactive(camera)
    try:
        cols = _ask_int("board inner corners across", 9)
        rows = _ask_int("board inner corners down", 6)
        square = _ask_float("square size, mm", 25.0)
        n = _ask_int("views to capture", 15)
        views = capture_chessboard_views(cam.read, (cols, rows), n)
        if len(views) < 6:
            print("only %d views -- not enough, aborting." % len(views))
            return
        calibrate_camera_intrinsics(views, camera=camera, board=(cols, rows),
                                    square_mm=square)
    finally:
        cam.close()


def _menu_dot_test() -> None:
    """Live find_laser_dot, so DOT_CHROMA_TAU can be retuned on real frames."""
    cam = _open_camera_interactive("narrow")
    title = "dot test - t/T lowers/raises tau, ESC to quit"
    tau = float(config.DOT_CHROMA_TAU)
    link = None
    try:
        link = CalibrationLink().open()
        _confirm("This turns the laser ON continuously. Ready?")
        link.laser(True)
        around = pick_point(cam.read, "click near the dot - ENTER accepts")
        if around is None:
            return
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        hits = 0
        total = 0
        while True:
            img = cam.read()
            p = find_laser_dot(img, around, tau=tau)
            total += 1
            hits += 1 if p is not None else 0
            scale = min(1.0, config.DISPLAY_SIZE[0] * 2.0 / img.shape[1])
            view = cv2.resize(img, None, fx=scale, fy=scale)
            half = config.DOT_WINDOW_PX // 2
            cv2.rectangle(view,
                          (int((around[0] - half) * scale), int((around[1] - half) * scale)),
                          (int((around[0] + half) * scale), int((around[1] + half) * scale)),
                          (255, 255, 0), 1)
            if p is not None:
                cv2.drawMarker(view, (int(p[0] * scale), int(p[1] * scale)),
                               (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(view, "tau %.0f   hit rate %.0f%%" % (tau, 100.0 * hits / total),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            if key == ord('t'):
                tau = max(5.0, tau - 5.0)
                hits = total = 0
            if key == ord('T'):
                tau += 5.0
                hits = total = 0
        print("final tau %.0f -- put it in config.DOT_CHROMA_TAU if it holds up" % tau)
    finally:
        cv2.destroyAllWindows()
        if link is not None:
            link.laser(False)
            link.close()
        cam.close()


def _menu_show() -> None:
    if not CALIB_DIR.exists():
        print("no calibrations yet (%s does not exist)" % CALIB_DIR)
        return
    files = sorted(CALIB_DIR.glob("*.json"))
    if not files:
        print("no calibrations in %s" % CALIB_DIR)
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        print("\n=== %s   (%s) ===" % (path.name, data.get("saved_at", "?")))
        print(json.dumps(data, indent=2, sort_keys=True))


def main() -> int:
    actions = {
        "1": ("image Jacobian J (moves both motors)", _menu_jacobian),
        "2": ("goal pixel g(R) = g_inf + c/R (fires the laser)", _menu_goal_pixel),
        "3": ("narrow camera intrinsics (chessboard)", lambda: _menu_intrinsics("narrow")),
        "4": ("wide camera intrinsics (chessboard + fisheye)", lambda: _menu_intrinsics("wide")),
        "5": ("laser dot detector live test / tau tuning", _menu_dot_test),
        "6": ("show saved calibrations", _menu_show),
    }
    while True:
        print("\n=== turret calibration ===")
        print("calibration directory: %s" % CALIB_DIR)
        for key in sorted(actions):
            print("  %s) %s" % (key, actions[key][0]))
        print("  q) quit")
        choice = input("choice: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return 0
        action = actions.get(choice)
        if action is None:
            print("no such option")
            continue
        try:
            action[1]()
        except KeyboardInterrupt as exc:
            # Only the operator aborting a menu item. Real faults propagate.
            print("\naborted: %s" % exc)


if __name__ == "__main__":
    sys.exit(main())
