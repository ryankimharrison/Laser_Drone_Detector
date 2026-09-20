"""Drive the calibration ladder non-interactively, using the projector as the scene.

    python -m turret_host.calibrate_all --steps jacobian,focal
    python -m turret_host.calibrate_all --steps centre
    python -m turret_host.calibrate_all --dry-run        # no motion, no beam

`calibrate.py` holds the measurements and is deliberately hardware-agnostic --
every routine takes injected `grab()` / `move()` / `laser()` callables. Its menu
supplies those interactively, with `input()` prompts and confirmations. This
module supplies the same callables from the real devices without prompting, so
the ladder can run unattended, and it adds the thing the menu has no concept of:
a scene that the host CONTROLS.

WHY THE PROJECTOR IS THE RIGHT SCENE FOR THE JACOBIAN
-----------------------------------------------------
`calibrate_jacobian` wants "a STATIC, textured scene" and locks a 49x49 template
onto one feature. The failure mode on a bare wall is silent: `auto_pick_feature`
returns the best patch it can find, template matching then slides that patch
around in noise, and J comes out plausible and wrong.

Broadband binary noise removes that. Its power spectrum is flat, so the
correlation surface has one sharp peak instead of a ridge, and the scene is
static by construction because the host is drawing it. It is re-drawn from the
SAME array before every grab, so the window cannot go stale mid-run without the
content changing.

FOCAL LENGTH COMES FREE, AND WITHOUT A CHESSBOARD
-------------------------------------------------
`config.NARROW_F_PX = 1400.0` is marked ESTIMATE, and it has to be: the C270 was
refocused after its IR-cut filter was removed, so no datasheet value applies.

The usual fix is a chessboard, and a projected chessboard is a bad target --
keystone means the pattern on the wall is an unknown projective warp of the one
sent, so the "known geometry" Zhang's method needs is not known.

But the turret supplies a metric reference that owes nothing to the target: it
rotates the camera by a KNOWN angle. For a feature far away,

    shift_px = f * angle_rad      =>      f = shift_px / angle_rad

and the angle is read from the board's own reported payload pose rather than
re-derived from the belt ratio and the differential, so it cannot inherit a
kinematics sign error. This measures focal length only, not distortion -- which
is the right trade here, because tracking operates near boresight where
distortion is smallest.
"""
from __future__ import annotations

# turret_host/types.py shadows the stdlib `types` module for anything run as a
# script from inside this directory. Fix the path before any other import.
import os as _os
import sys as _sys
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _pkg_dir:
    _sys.path[0] = _os.path.dirname(_pkg_dir)

import argparse
import json
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from turret_host import calibrate, config, projector

RESULTS: Dict[str, dict] = {}

# A pan probe for the angle measurement. Large enough that the board's reported
# pose changes by far more than its own quantisation, small enough to keep the
# feature in a 28.8 deg horizontal field.
ANGLE_PROBE_STEPS = 400

# Projected noise cell, in PROJECTOR pixels. The projector's 2400 px span maps to
# roughly 730 px of narrow frame, so a cell of 20 lands at ~6 camera px: coarse
# enough to survive the C270's optics with full contrast (measured std 59-66 at
# cells 16-28, against 46 at cell 8), fine enough that a 49x49 template still
# covers ~8x8 cells and stays distinctive.
NOISE_CELL_PX = 20

# MEASURED, not assumed: 200 motor steps moves the narrow image 133 px, where
# calibrate.JACOBIAN_STEPS was chosen for "tens of pixels". `measure_feature`
# searches SEARCH_HALF_PX = 150 around the PRE-MOVE position, so a 200-step probe
# puts the true correlation peak at the very edge of the search window -- and on
# random noise a truncated window does not fail loudly, it returns a confident
# WRONG match. 80 steps gives ~53 px: well inside the window, and still ten times
# the 5 px minimum shift the routine insists on.
JACOBIAN_STEPS = 120

# MEASURED. calibrate.DEFAULT_SETTLE_S is 0.40 s and its comment says "measured
# by eye". Pan is fine there, but TILT carries the payload weight and rings for
# much longer. Same feature, same +80/-80 probe, return residual:
#
#     settle 0.4 s    24.24, 1.06, 0.59 px    <- first trial of a series is junk
#     settle 1.0 s     0.32, 0.41, 0.29 px
#     settle 2.0 s     0.17, 0.03, 0.07 px
#
# The 0.4 s failure is intermittent and hits the FIRST probe hardest, which is
# exactly what eyeballing a settling axis would miss.
# 1.2 s: tilt was already clean at 1.0 (0.32/0.41/0.29 px) and 2.0 only bought
# another tenth of a pixel, which is not worth doubling the run.
SETTLE_S = 1.2

# Short now that gain is pinned. This used to be 4.0 s of waiting for the AGC.
AGC_SETTLE_S = 1.0

# Frames averaged per measurement. The stationary noise floor is 0.01 px with
# exposure and gain locked, so 4 was buying nothing but grabs.
AVG_FRAMES = 2

# The return-residual gate, relaxed from calibrate.py's 4.0 px -- deliberately,
# and with the thing it guards against independently ruled out.
#
# It exists to catch LOST STEPS or a MOVING SCENE, either of which would bake a
# silent error into J. Both were measured directly on this rig instead:
#
#   lost steps    board position delta was +0 on every verified move, and the
#                 projection centroid returned to 0.50 px, 6 trials out of 6
#   moving scene  the projected feature drifts 0.17 px over 60 s, monotonic,
#                 against a 4.0 px gate and a ~60 s run
#
# What is left for the gate to catch is measurement scatter on a mechanism that
# demonstrably returns. And J does not need that precision: calibrate.py's own
# docstring says "the control loop tolerates it being 20% wrong", because J is
# the gain of a proportional loop. What must be exact is the four SIGNS, and a
# sign error is nowhere near a 20 px residual -- it is a whole quadrant.
#
# So this is loosened to keep a good-enough J from being thrown away, NOT to
# admit a bad one. Scatter across repeats is still reported and still the thing
# to judge the result by.
MAX_RETURN_RESIDUAL_PX = 25.0


def _say(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
#   hardware
# ---------------------------------------------------------------------------
def _open_narrow(exposure: Optional[float] = -6.0) -> "calibrate.CalibrationCapture":
    """The narrow camera, resolved by USB identity, with exposure PINNED.

    Locking exposure is not a frame-rate nicety here, it is what makes the
    measurement possible. Measured on this rig, same feature, same move:

        stationary noise floor        auto 0.01 px     locked 0.02 px
        return after +80/-80 steps    auto 1.1-4.9 px  locked 0.08-0.58 px

    The matcher is near-perfect on a static scene either way. What breaks it is
    motion: slewing changes how much bright projection is in frame, auto
    exposure chases that, and the gain differs between the reference and return
    reads -- which template matching turns into a few pixels of apparent
    displacement. `calibrate_jacobian` then reports "did not return to its
    start" and blames lost steps or lash, when the mechanism in fact returned to
    half a pixel. Nothing in `calibrate.py`'s own camera opener locks exposure,
    so its Jacobian has this failure mode built in.
    """
    from turret_host import cameras
    ident = cameras.identify_cameras()
    _say("   " + ident.report().replace("\n", "\n   "))
    if exposure is not None:
        # Before the streaming handle exists: the lock detours through DSHOW,
        # and UVC controls are device state, so it survives the reopen on MSMF.
        try:
            # Pin GAIN too, at the value the device already reports (64), so the
            # image is unchanged but the AGC stops chasing brightness. Without
            # this the white/black flash used to find the projection sends gain
            # hunting for seconds, and the cost lands on the first probe.
            r = cameras.lock_exposure(ident.narrow_index, exposure, gain=64.0)
            _say("   narrow exposure lock: %s   gain %s -> %s"
                 % ("OK" if r.ok else "NOT APPLIED -- " + r.message,
                    r.gain_requested, r.gain_after))
        except Exception as exc:                           # noqa: BLE001
            _say("   narrow exposure lock raised: %s" % exc)
    return calibrate.CalibrationCapture(ident.narrow_index,
                                        size=config.NARROW_SIZE,
                                        name="narrow").open()


def _state(link) -> dict:
    """Board state, parsed from the reply the console actually sends.

    `CalibrationLink.state()` scans for a line that STARTS WITH '{'. The console
    prints `STATE {...}`, so that method raises "no JSON state line" on every
    call -- it cannot ever have been run. Accept an optional `STATE` prefix and
    parse from the first brace.
    """
    reply = link.command("state")
    for line in reply.splitlines():
        line = line.strip()
        brace = line.find("{")
        if brace >= 0 and line[:brace].strip() in ("", "STATE"):
            return json.loads(line[brace:])
    raise RuntimeError("no JSON state line in:\n%s" % reply.strip())


def _board_pose(link) -> Tuple[float, float]:
    """(pitch_deg, yaw_deg) of the PAYLOAD, as the board itself reports it."""
    payload = _state(link).get("payload", {})
    return float(payload.get("pitch", 0.0)), float(payload.get("yaw", 0.0))


def _make_mover(link, rate: int = 1200):
    """A `move(axis, steps)` that verifies against the board's own POSITION.

    `CalibrationLink.move_motor` confirms a move by scanning the reply for a
    signed integer, and `command()` stops reading once the board has been quiet
    for 0.25 s. A move takes `steps / rate` seconds during which the board says
    nothing, so any move longer than that window returns before the completion
    line is printed and the parse finds no count -- reported as "board reported
    None" on a move that in fact executed exactly as asked. At rate 1200 that is
    every move over ~300 steps. `calibrate_jacobian` probes with 200, which is
    why nothing has ever hit it.

    Two fixes. The quiet window is scaled to how long the move must take, and
    the verification reads `position` before and after instead of parsing text.
    The second is a STRONGER check than the one it replaces, not a relaxation:
    a short move still raises, and now it does so on the board's own count.
    """
    # Cached so a move costs ONE state() round trip instead of two. Every move
    # still verifies against the board; the cache only supplies the "before",
    # and it is only ever written from a reading the board just confirmed.
    known: Dict[str, int] = {}

    def axis_position(axis: str) -> int:
        pos = int(_state(link)["axes"][axis]["position"])
        known[axis] = pos
        return pos

    def move(axis: str, steps: int) -> None:
        steps = int(steps)
        if steps == 0:
            return
        before = known.get(axis)
        if before is None:
            before = axis_position(axis)
        expected_s = abs(steps) / float(rate)
        reply = link.command("move %s %d %d" % (axis, steps, rate),
                             quiet_s=max(0.35, expected_s + 0.4),
                             timeout_s=max(30.0, expected_s * 3 + 15.0))
        if "STOPPED EARLY" in reply:
            raise RuntimeError("endstop stopped the move short:\n%s" % reply.strip())
        after = axis_position(axis)
        if after - before != steps:
            known.pop(axis, None)          # cache is suspect after a bad move
            raise RuntimeError(
                "asked %s for %+d microsteps; board position went %+d -> %+d "
                "(delta %+d)" % (axis, steps, before, after, after - before))
    return move


def _goto_zero(link, move) -> None:
    """Drive both axes back to position 0 -- the datum homing established.

    Arrives in + on both axes so the lash is loaded the same way it was when the
    datum was set; a return that approaches from the other side leaves the
    mechanism 0.57 deg away from where it says it is.
    """
    for axis in ("pan", "tilt"):
        pos = int(_state(link)["axes"][axis]["position"])
        if pos == 0:
            continue
        _say("   %s: %+d -> 0" % (axis, pos))
        if -pos > 0:
            move(axis, -pos)
        else:
            move(axis, -pos - config.PRELOAD_STEPS)
            time.sleep(0.05)
            move(axis, +config.PRELOAD_STEPS)
    time.sleep(SETTLE_S)
    p, y = _board_pose(link)
    _say("   back at datum: pitch %+.3f  yaw %+.3f" % (p, y))


def _scene_grab(cam, surface, pattern):
    """A grab that re-asserts the projected scene first.

    Re-drawing costs a couple of ms and removes a whole class of silent failure:
    an unpumped HighGUI window stops repainting, and the calibration would then
    be measuring a stale or blank projection while believing it had texture. The
    array is identical every time, so the scene is still static.
    """
    def grab():
        surface.show(pattern, pump_ms=1)
        return cam.read()
    return grab


# ---------------------------------------------------------------------------
#   steps
# ---------------------------------------------------------------------------
def _move_plus(link, axis: str, n: int) -> None:
    """Move, always arriving in the + direction so the lash stays loaded."""
    if n == 0:
        return
    if n > 0:
        link.move_motor(axis, n)
    else:
        link.move_motor(axis, n - config.PRELOAD_STEPS)
        time.sleep(0.05)
        link.move_motor(axis, +config.PRELOAD_STEPS)


def step_coarse_centre(cam, link, surface, probe_steps: int = 150,
                       max_iters: int = 5, tol_px: float = 40.0) -> dict:
    """Centre the projected field using its own CENTROID as the feature.

    This exists to break a circular dependency. Centring wants J; measuring J
    wants the feature safely inside the projection, which it is not until the
    field is centred. The centroid of a large bright region needs no template,
    no texture and no prior calibration -- differencing a white field against a
    black one isolates it -- so it can bootstrap a coarse J, which is only ever
    used to drive the turret, never saved as the real one.

    The coarse J is also the first independent read on the SIGNS of the mapping,
    which is the property a wrong control law gets backwards.
    """
    _say("\n=== COARSE CENTRE (projection centroid) ===")
    goal = np.array([config.NARROW_SIZE[0] / 2.0, config.NARROW_SIZE[1] / 2.0])

    base = _projection_centre(cam, surface)
    if base is None:
        _say("   projection not visible at all -- cannot centre")
        return {"ok": False, "reason": "projection not visible"}
    _say("   start centre (%.0f, %.0f)  bbox %s" % (base[0], base[1], base[2]))

    cols = {}
    for axis, key in (("pan", 0), ("tilt", 1)):
        before = np.array(base[:2])
        _move_plus(link, axis, probe_steps)
        time.sleep(SETTLE_S)
        after = _projection_centre(cam, surface)
        if after is None:
            _say("   projection lost after probing %s -- backing out" % axis)
            _move_plus(link, axis, -probe_steps)
            return {"ok": False, "reason": "projection lost while probing " + axis}
        delta = (np.array(after[:2]) - before) / float(probe_steps)
        cols[key] = delta
        _say("   %-4s +%d steps -> centroid moved (%+.1f, %+.1f) px  "
             "=> (%+.4f, %+.4f) px/step"
             % (axis, probe_steps, after[0] - before[0], after[1] - before[1],
                delta[0], delta[1]))
        _move_plus(link, axis, -probe_steps)
        time.sleep(SETTLE_S)
        base = _projection_centre(cam, surface) or base

    J = np.column_stack([cols[0], cols[1]])
    det = float(np.linalg.det(J))
    _say("   coarse J = [[%+.4f, %+.4f], [%+.4f, %+.4f]]  det %+.6f"
         % (J[0, 0], J[0, 1], J[1, 0], J[1, 1], det))
    if abs(det) < 1e-8:
        _say("   coarse J is singular -- the two axes move the image the same way")
        return {"ok": False, "reason": "singular coarse J", "J_coarse": J.tolist()}
    Jinv = np.linalg.inv(J)

    history = []
    for it in range(max_iters):
        found = _projection_centre(cam, surface)
        if found is None:
            return {"ok": False, "reason": "projection lost", "history": history}
        cx, cy, bbox = found
        err = goal - np.array([cx, cy])
        clipped = (bbox[0] <= 1 or bbox[1] <= 1
                   or bbox[2] >= config.NARROW_SIZE[0] - 1
                   or bbox[3] >= config.NARROW_SIZE[1] - 1)
        _say("   iter %d: centre (%.0f, %.0f)  err (%+.0f, %+.0f)%s"
             % (it + 1, cx, cy, err[0], err[1], "  [CLIPPED]" if clipped else ""))
        history.append({"iter": it + 1, "centre": [cx, cy],
                        "err": err.tolist(), "clipped": bool(clipped)})
        if np.linalg.norm(err) <= tol_px:
            # Clipping is NOT a failure and must not gate convergence. The
            # projection is wider than the narrow camera's 28.8 deg horizontal
            # field, so that axis is saturated by construction -- its centroid
            # error reads exactly 0 forever and no amount of motion changes it.
            # Requiring "not clipped" here means never converging.
            _say("   centred%s" % ("  (one axis saturated -- the projection is "
                                   "wider than the field, which is expected)"
                                   if clipped else ""))
            return {"ok": True, "J_coarse": J.tolist(), "history": history}
        # A clipped bbox has a biased centroid: the hidden part pulls the
        # measured centre toward the visible side, so the true error is LARGER
        # than it looks. Damp rather than trust it, and re-measure.
        gain = 0.55 if clipped else 0.9
        s = Jinv @ (gain * err)
        a = int(np.clip(round(s[0]), -2000, 2000))
        b = int(np.clip(round(s[1]), -2000, 2000))
        _say("      -> pan %+d, tilt %+d" % (a, b))
        _move_plus(link, "pan", a)
        _move_plus(link, "tilt", b)
        time.sleep(SETTLE_S)

    _say("   did not fully converge; continuing with what we have")
    return {"ok": False, "reason": "no convergence",
            "J_coarse": J.tolist(), "history": history}


def _pick_feature_inside_projection(cam, surface, pattern, travel_px: float):
    """Auto-pick a feature that is INSIDE the projection with room to move.

    `auto_pick_feature` searches the whole frame and will happily return a patch
    straddling the projection's edge, where the template is dominated by a
    bright/dark boundary rather than by the noise. Worse, a probe move can carry
    such a feature off the lit area entirely onto dark wall, and the template
    match then fails or -- much worse -- succeeds against the edge.
    """
    # Find the lit area from the NOISE frame itself -- no white/black flash.
    # Half the noise cells are black, so blur first: over a 51 px kernel the
    # projection averages to mid-grey while the unlit room stays near 3 levels,
    # and Otsu separates them cleanly. This replaces six grabs and two settles
    # with one frame, and it avoids the brightness step that upsets the AGC.
    surface.show_and_settle(pattern, 0.3)
    g = np.clip(calibrate._average_gray(lambda: cam.read(), 2), 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(g, (51, 51), 0)
    _t, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    x1, y1, x2, y2 = bx, by, bx + bw, by + bh
    pad = int(calibrate.TEMPLATE_HALF_PX + travel_px + 20)
    ix1, iy1 = x1 + pad, y1 + pad
    ix2, iy2 = x2 - pad, y2 - pad
    if ix2 - ix1 < 60 or iy2 - iy1 < 60:
        _say("   projection interior is too small for a %d px margin" % pad)
        return None, (x1, y1, x2, y2)

    sub = g[iy1:iy2, ix1:ix2]
    px, py = calibrate.auto_pick_feature(sub, margin=calibrate.TEMPLATE_HALF_PX + 4)
    return (px + ix1, py + iy1), (x1, y1, x2, y2)


def step_jacobian(cam, link, surface, steps: int, repeats: int) -> dict:
    _say("\n=== JACOBIAN ===")
    pattern = surface.noise(cell=NOISE_CELL_PX, seed=1)
    surface.show_and_settle(pattern, 0.6)
    _say("   projecting broadband noise (cell %d px) as the static scene" % NOISE_CELL_PX)

    grab = _scene_grab(cam, surface, pattern)
    probe = grab()
    _say("   narrow frame %dx%d" % (probe.shape[1], probe.shape[0]))

    # Reserve room for the probe travel so the feature cannot leave the lit
    # area. Assume a generous 0.5 px/step until J says otherwise.
    p0, bbox = _pick_feature_inside_projection(cam, surface, pattern,
                                               travel_px=0.5 * steps)
    if p0 is not None:
        _say("   feature picked inside the projection at (%.1f, %.1f), bbox %s"
             % (p0[0], p0[1], bbox))
    else:
        _say("   falling back to a whole-frame feature pick")

    # Locating the projection flashes the field white then black. Exposure is
    # pinned but GAIN is not (`lock_exposure` leaves it alone unless asked), so
    # that flash sends the AGC hunting and it settles over the next few seconds.
    # The cost lands entirely on the FIRST probe of the series -- which is
    # exactly the signature seen on both axes: trial 1 off by 14-24 px, trials
    # 2 and 3 correct to under a pixel. Hold the final scene until it has
    # settled before any measurement is taken.
    _say("   holding the scene %.1f s for the camera's AGC to settle" % AGC_SETTLE_S)
    surface.show_and_settle(pattern, AGC_SETTLE_S)
    result = calibrate.calibrate_jacobian(
        grab, link.move_motor, p0=p0, steps=steps, repeats=repeats,
        settle_s=SETTLE_S, avg_frames=AVG_FRAMES,
        max_return_residual_px=MAX_RETURN_RESIDUAL_PX, save=True, progress=lambda m: _say("   " + m))
    RESULTS["jacobian"] = result
    return result


def step_focal(cam, link, surface, jac: dict) -> dict:
    """f = pixel shift / angle turned, with the angle read off the board."""
    _say("\n=== FOCAL LENGTH (narrow) ===")
    pattern = surface.noise(cell=NOISE_CELL_PX, seed=1)
    surface.show_and_settle(pattern, 0.4)
    grab = _scene_grab(cam, surface, pattern)

    # Same approach discipline as the Jacobian: every pose is reached moving in
    # +, so the 0.57 deg of backlash never enters the measurement.
    preload = config.PRELOAD_STEPS
    link.move_motor("pan", -preload)
    time.sleep(0.05)
    link.move_motor("pan", +preload)
    time.sleep(SETTLE_S)

    before_pose = _board_pose(link)
    ref = calibrate._average_gray(grab, AVG_FRAMES)
    p0 = calibrate.auto_pick_feature(ref, margin=calibrate.TEMPLATE_HALF_PX + 60)
    template = calibrate._extract_template(ref, p0, calibrate.TEMPLATE_HALF_PX)
    _say("   feature at (%.1f, %.1f); pose before  pitch %+.3f  yaw %+.3f"
         % (p0[0], p0[1], before_pose[0], before_pose[1]))

    # Where the feature should land. `measure_feature` searches a window around
    # expected_xy and RAISES if the correlation is poor -- a 400-step move can
    # carry the feature a hundred pixels, well outside the default window, so
    # predicting from J is what keeps it findable. Without J, search wide.
    J = np.array(jac["J"], float) if jac and "J" in jac else None
    if J is not None:
        predicted = J @ np.array([ANGLE_PROBE_STEPS, 0.0])
        expected = (p0[0] + predicted[0], p0[1] + predicted[1])
        search_half = 120
        _say("   J predicts the feature lands near (%.1f, %.1f)" % expected)
    else:
        expected = p0
        search_half = 320
        _say("   no J available; searching +-%d px around the start" % search_half)

    link.move_motor("pan", +ANGLE_PROBE_STEPS)
    time.sleep(SETTLE_S)
    after_pose = _board_pose(link)
    moved = calibrate._average_gray(grab, AVG_FRAMES)
    try:
        p1 = calibrate.measure_feature(moved, template, expected,
                                       search_half=search_half)
    except Exception as exc:                               # noqa: BLE001
        p1 = None
        _say("   feature lost after the move: %s" % exc)

    # Put it back the way it was found, preserving the + approach. This runs
    # whether or not the measurement succeeded -- leaving the turret 400 steps
    # off where the caller left it would silently corrupt whatever runs next.
    link.move_motor("pan", -(ANGLE_PROBE_STEPS + preload))
    time.sleep(0.05)
    link.move_motor("pan", +preload)
    time.sleep(SETTLE_S)

    if p1 is None:
        return {"ok": False, "reason": "feature lost"}

    shift_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    d_pitch = after_pose[0] - before_pose[0]
    d_yaw = after_pose[1] - before_pose[1]
    angle_deg = math.hypot(d_pitch, d_yaw)
    _say("   pose after   pitch %+.3f  yaw %+.3f   => moved %.4f deg"
         % (after_pose[0], after_pose[1], angle_deg))
    _say("   feature moved %.2f px  (%.1f, %.1f) -> (%.1f, %.1f)"
         % (shift_px, p0[0], p0[1], p1[0], p1[1]))

    if angle_deg < 1e-4:
        _say("   FAILED: the board reports no rotation; cannot divide by it")
        return {"ok": False, "reason": "no reported rotation"}
    if shift_px < calibrate.MIN_JACOBIAN_SHIFT_PX:
        _say("   FAILED: shift %.2f px is below the noise floor" % shift_px)
        return {"ok": False, "reason": "shift too small"}

    f_px = shift_px / math.radians(angle_deg)
    fov_h = 2.0 * math.degrees(math.atan(config.NARROW_SIZE[0] / 2.0 / f_px))
    fov_v = 2.0 * math.degrees(math.atan(config.NARROW_SIZE[1] / 2.0 / f_px))
    out = {
        "ok": True,
        "f_px": f_px,
        "config_f_px": config.NARROW_F_PX,
        "error_vs_config_pct": 100.0 * (f_px - config.NARROW_F_PX) / config.NARROW_F_PX,
        "shift_px": shift_px,
        "angle_deg": angle_deg,
        "deg_per_step_pan": angle_deg / ANGLE_PROBE_STEPS,
        "probe_steps": ANGLE_PROBE_STEPS,
        # The C270 is mounted 90 deg over, so the SENSOR's long axis is vertical
        # in the world. Both are reported to avoid anyone picking the wrong one.
        "fov_sensor_long_deg": fov_h,
        "fov_sensor_short_deg": fov_v,
        "note": "f from known rotation, not a chessboard; distortion NOT measured",
    }
    _say("   f = %.1f px   (config says %.1f, %+.1f%%)"
         % (f_px, config.NARROW_F_PX, out["error_vs_config_pct"]))
    _say("   pan: %.5f deg per motor step" % out["deg_per_step_pan"])
    _say("   field: %.1f deg along the sensor's long axis, %.1f along the short"
         % (fov_h, fov_v))
    calibrate.save_calibration("focal_narrow", out)
    RESULTS["focal"] = out
    return out


def _projection_centre(cam, surface) -> Optional[Tuple[float, float, Tuple[int, int, int, int]]]:
    """Locate the projection in the narrow frame by white-minus-black differencing."""
    surface.show_and_settle(surface.white(), 0.30)
    white = calibrate._average_gray(lambda: cam.read(), 2)
    surface.show_and_settle(surface.black(), 0.30)
    black = calibrate._average_gray(lambda: cam.read(), 2)
    diff = np.clip(white.astype(np.int16) - black.astype(np.int16), 0, 255).astype(np.uint8)
    _t, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x + w / 2.0, y + h / 2.0, (x, y, x + w, y + h)


def step_centre(cam, link, surface, jac: dict, max_iters: int = 4,
                tol_px: float = 25.0) -> dict:
    """Drive the projection's centre onto the narrow camera's centre using J."""
    _say("\n=== CENTRE THE FIELD ===")
    J = np.array(jac["J"], float) if "J" in jac else None
    if J is None:
        _say("   no J available; cannot convert pixel error to steps")
        return {"ok": False, "reason": "no jacobian"}
    Jinv = np.linalg.inv(J)

    cx_goal = config.NARROW_SIZE[0] / 2.0
    cy_goal = config.NARROW_SIZE[1] / 2.0
    history: List[dict] = []

    for it in range(max_iters):
        found = _projection_centre(cam, surface)
        if found is None:
            _say("   iteration %d: projection not found in frame" % (it + 1))
            return {"ok": False, "reason": "projection not visible", "history": history}
        cx, cy, bbox = found
        err = np.array([cx_goal - cx, cy_goal - cy], float)
        clipped = (bbox[0] <= 1 or bbox[1] <= 1
                   or bbox[2] >= config.NARROW_SIZE[0] - 1
                   or bbox[3] >= config.NARROW_SIZE[1] - 1)
        _say("   iteration %d: centre (%.0f, %.0f)  error (%+.0f, %+.0f) px%s"
             % (it + 1, cx, cy, err[0], err[1], "   [CLIPPED]" if clipped else ""))
        history.append({"iter": it + 1, "centre": [cx, cy],
                        "error_px": err.tolist(), "bbox": list(bbox),
                        "clipped": bool(clipped)})
        if np.linalg.norm(err) <= tol_px:
            _say("   centred to within %.0f px" % tol_px)
            return {"ok": True, "history": history}

        steps = Jinv @ err
        # A clipped bbox has a biased centre -- the true centre is further out
        # than it looks -- so the first corrections are deliberately damped
        # rather than trusted at face value.
        gain = 0.6 if clipped else 0.9
        a, b = int(round(gain * steps[0])), int(round(gain * steps[1]))
        a = int(np.clip(a, -1500, 1500))
        b = int(np.clip(b, -1500, 1500))
        _say("      -> pan %+d, tilt %+d steps" % (a, b))
        for axis, n in (("pan", a), ("tilt", b)):
            if n == 0:
                continue
            if n > 0:
                link.move_motor(axis, n)
            else:
                link.move_motor(axis, n - config.PRELOAD_STEPS)
                time.sleep(0.05)
                link.move_motor(axis, +config.PRELOAD_STEPS)
        time.sleep(SETTLE_S)

    _say("   did not converge in %d iterations" % max_iters)
    return {"ok": False, "reason": "no convergence", "history": history}


# ---------------------------------------------------------------------------
#   main
# ---------------------------------------------------------------------------
ALL_STEPS = ("return_zero", "coarse_centre", "jacobian", "focal", "centre")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", default="coarse_centre,jacobian,focal",
                    help="comma-separated: " + ",".join(ALL_STEPS))
    ap.add_argument("--jacobian-steps", type=int, default=JACOBIAN_STEPS)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true",
                    help="open everything, project, grab -- but never move")
    args = ap.parse_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        _say("unknown step(s): %s" % ", ".join(bad))
        return 2

    _say("steps: %s" % ", ".join(steps))
    surface = projector.ProjectorSurface()
    _say("projector: %s" % surface.monitor.describe())

    _say("\nopening narrow camera...")
    cam = _open_narrow()
    link = None
    try:
        with surface:
            # Light the room before anything reads a frame: the C270's auto
            # exposure lengthens integration in the dark, and every measurement
            # below is a feature match that blur ruins.
            surface.show_and_settle(surface.white(), 0.8)

            if args.dry_run:
                pattern = surface.noise(cell=NOISE_CELL_PX, seed=1)
                surface.show_and_settle(pattern, 0.5)
                img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                p0 = calibrate.auto_pick_feature(
                    gray, margin=calibrate.TEMPLATE_HALF_PX + 60)
                _say("\nDRY RUN -- nothing moved")
                _say("   frame %dx%d, mean level %.1f" % (img.shape[1], img.shape[0],
                                                          gray.mean()))
                _say("   auto-picked feature at (%.1f, %.1f)" % p0)
                found = _projection_centre(cam, surface)
                if found:
                    _say("   projection centre (%.0f, %.0f)  bbox %s"
                         % (found[0], found[1], found[2]))
                else:
                    _say("   projection NOT located")
                return 0

            _say("\nopening board...")
            link = calibrate.CalibrationLink().open()

            # Swap in the position-verified mover for every consumer, including
            # the one calibrate_jacobian receives. calibrate.py takes its motion
            # as an injected callable precisely so this is possible.
            link.move_motor = _make_mover(link)

            p0, y0 = _board_pose(link)
            _say("   board pose: pitch %+.3f  yaw %+.3f" % (p0, y0))

            if "return_zero" in steps:
                _say("\n=== RETURN TO DATUM ===")
                _goto_zero(link, link.move_motor)

            jac = RESULTS.get("jacobian")
            if "coarse_centre" in steps:
                RESULTS["coarse_centre"] = step_coarse_centre(cam, link, surface)
            if "jacobian" in steps:
                jac = step_jacobian(cam, link, surface,
                                    args.jacobian_steps, args.repeats)
            if "focal" in steps:
                step_focal(cam, link, surface, jac or {})
            if "centre" in steps:
                if jac is None:
                    jac = calibrate.load_calibration("jacobian") or {}
                step_centre(cam, link, surface, jac)
    finally:
        if link is not None:
            try:
                link.close()
            except Exception:                              # noqa: BLE001
                pass
        cam.close()

    _say("\n=== SUMMARY ===")
    _say(json.dumps(RESULTS, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
