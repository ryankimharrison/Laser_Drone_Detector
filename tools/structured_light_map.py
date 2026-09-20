"""Dense wide<->narrow correspondence from projected Gray codes.

    python tools/structured_light_map.py --self-test        # no hardware
    python tools/structured_light_map.py                    # capture + report
    python tools/structured_light_map.py --write            # ...and save the fit

TOUCHES NO SERIAL PORT AND COMMANDS NO MOTION. It opens the two cameras and
the projector, nothing else. `link`, `homing` and `control` are deliberately
not imported: this measurement needs the turret to be STATIONARY, not homed,
and the safest way to guarantee it does not move is to have no way to move it.

WHY THIS IS NOT "PROJECT A CALIBRATION TARGET"
----------------------------------------------
`turret_host/projector.py` argues against projecting a calibration target and
it is right -- for INTRINSICS. Keystone makes what lands on the wall an unknown
projective warp of what was sent, so the "known target geometry" Zhang's method
needs is not known.

That objection does not apply here, because the projected pattern's geometry is
never used. Gray codes are used only to ANSWER ONE QUESTION PER PIXEL: which
projector pixel is this camera pixel looking at? Whatever warped quadrilateral
lands on the wall, both cameras look at that same warped quadrilateral, and the
warp cancels in the correspondence -- exactly as it already cancels for the
ChArUco fit in `calibrate_wide_to_narrow.py`.

WHAT THIS BUYS OVER THE EXISTING FIT
------------------------------------
`calibration/wide_to_narrow.json` is a SIMILARITY transform -- scale, offset and
a fixed 90 deg rotation, four numbers -- fitted to 283 ChArUco corners at RMS
5.71 px. Its own note says the wide camera's barrel distortion is not modelled,
so that 5.71 is a pooled average hiding a small residual at the axis and a large
one at the rim.

This measures the map instead of modelling it:

    283 corners, 4 parameters      ->   ~10^5-10^6 correspondences, no model
    distortion unmodelled          ->   distortion included, it is in the data
    residual known only in pooled  ->   residual resolved against field radius
    RMS

The headline output is therefore not a better transform. It is the RESIDUAL
CURVE: how far the current linear model is wrong, as a function of distance
from the wide camera's axis. That answers whether a fancier model is needed at
all, with numbers instead of argument.

WHAT IT DOES NOT BUY
--------------------
The correspondence is measured THROUGH ONE PLANE -- the wall. It is exact for
objects on that plane and wrong elsewhere, by the disparity, which goes as
1/range. That is not a defect of the method, it is parallax: a 60 mm baseline
cannot be aligned by any fixed 2D map at every range.

So this gives a registered overlay at the calibration distance, and nothing
about depth. Repeat at two or three tape-measured wall distances and the map
becomes a function of 1/R, which is an empirical stereo calibration; that is a
separate run and a separate fit, not this script.

It also gives no metric intrinsics. Those still want a physical board that can
be tilted.

THE DECODE, AND WHY IT SURVIVES A LIT ROOM
------------------------------------------
Two choices do all the work:

  * GRAY CODE, not plain binary. Adjacent projector columns differ in exactly
    one bit, so a pixel that lands on a code boundary misreads by one column.
    Under plain binary the 511/512 boundary flips nine bits at once and the
    decode error is 512 columns, not one.

  * COMPLEMENTARY PATTERNS. Every bit is shown twice, normal and inverted, and
    the bit is read as `I_pattern > I_inverse`. The threshold is therefore
    LOCAL to the pixel, so it is immune to albedo variation, vignetting,
    projector falloff and ambient light -- none of which are equal across the
    frame, and a global threshold would have to assume they were. The gap
    |I_pattern - I_inverse| doubles as the per-pixel confidence, which is what
    the validity mask is built from.

Lights off helps contrast but is NOT required by the decode; the local
threshold is what makes it robust. Locked exposure IS required: an auto-exposure
loop hunting against the changing patterns suppresses exactly the signal being
measured.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# turret_host FIRST: its __init__ sets OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS,
# and that only has an effect before cv2 is imported.
from turret_host import config                            # noqa: E402
from turret_host.camera_fusion import WideToNarrow        # noqa: E402

import cv2                                                # noqa: E402
import numpy as np                                        # noqa: E402

OUT_PATH = _ROOT / "turret_host" / "calibration" / "wide_narrow_dense.npz"
FIT_PATH = _ROOT / "turret_host" / "calibration" / "wide_to_narrow.json"

# A bit is only believed when the normal and inverted frames differ by at least
# this many grey levels. Below it the pixel is either outside the projection,
# in shadow, or saturated -- all three decode to noise, and a wrong bit in a
# high-order plane moves the correspondence by hundreds of columns.
MIN_BIT_CONTRAST = 12.0

# Projection coverage below this in EITHER camera means the wall is not framed
# by both, and the overlap the whole measurement lives in does not exist.
MIN_COVERAGE_FRAC = 0.02

SETTLE_S = 0.35          # matches projector.SETTLE_S
FRESH_FRAMES = 3         # matches projector.FRESH_FRAMES
AVERAGE_FRAMES = 2       # per pattern, after the fresh ones, to beat sensor noise


# ==========================================================================
#   Gray code
# ==========================================================================
def n_bits(size: int) -> int:
    """Bits needed to index `size` columns."""
    return int(np.ceil(np.log2(max(2, size))))


def gray_of(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.uint32)
    return v ^ (v >> 1)


def gray_to_binary(gray: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of gray_of, vectorised. b_{n-1}=g_{n-1}; b_i = b_{i+1} ^ g_i."""
    g = gray.astype(np.uint32)
    out = np.zeros_like(g)
    prev = np.zeros_like(g)
    for i in range(bits - 1, -1, -1):
        bit = ((g >> i) & 1) ^ prev
        out |= bit << i
        prev = bit
    return out


def stripe_image(h: int, w: int, bit: int, axis: str, invert: bool) -> np.ndarray:
    """One Gray-code plane, full projector resolution, BGR uint8."""
    n = w if axis == "x" else h
    idx = np.arange(n, dtype=np.uint32)
    plane = ((gray_of(idx) >> bit) & 1).astype(np.uint8)
    if invert:
        plane = 1 - plane
    line = (plane * 255)
    if axis == "x":
        img = np.repeat(line[None, :], h, axis=0)
    else:
        img = np.repeat(line[:, None], w, axis=1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ==========================================================================
#   capture
# ==========================================================================
def _gray_u8(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _grab_fresh(thread, n_average: int = AVERAGE_FRAMES) -> Optional[np.ndarray]:
    """Mean of `n_average` frames that are all NEW since the display changed.

    The capture thread publishes into a newest-wins slot, so the frame sitting
    there when the pattern changed may have been exposed under the PREVIOUS
    pattern. Discard FRESH_FRAMES by frame index first, then average -- waiting
    on wall-clock alone cannot distinguish the two.
    """
    first, _ = thread.latest()
    last_index = first.index if first is not None else -1
    deadline = time.time() + SETTLE_S + 2.0
    seen = 0
    acc: List[np.ndarray] = []
    while time.time() < deadline and len(acc) < n_average:
        f, _ = thread.latest()
        if f is not None and f.index != last_index:
            last_index = f.index
            seen += 1
            if seen > FRESH_FRAMES:
                acc.append(_gray_u8(f.image).astype(np.float32))
        else:
            # waitKey, not sleep: this also services the HighGUI event queue, so
            # the projected pattern stays painted while we wait on the camera.
            cv2.waitKey(5)
    if not acc:
        return None
    return np.mean(acc, axis=0)


def capture_sequence(surface, threads: Dict[str, object],
                     progress=print) -> Dict[str, dict]:
    """Show every Gray plane and its inverse; return per-camera decoded maps.

    Returns {camera: {"proj_x", "proj_y", "valid", "white", "black"}} where
    proj_* are projector pixel indices per CAMERA pixel and `valid` is the mask
    of pixels whose every bit cleared MIN_BIT_CONTRAST.
    """
    ph, pw = surface.shape
    bx, by = n_bits(pw), n_bits(ph)
    total = 2 * (bx + by) + 2
    progress("projector %dx%d -> %d column bits, %d row bits, %d patterns"
             % (pw, ph, bx, by, total))

    names = list(threads)
    codes = {c: {"x": None, "y": None} for c in names}
    # Per-bit confidence is kept as a PLANE, not reduced to a running minimum,
    # because which bits are believable is not known until they have all been
    # seen. Stored as uint8: ~70 MB for both cameras and both axes at these
    # resolutions, which is cheaper than being unable to drop a bad bit.
    conf: Dict[str, Dict[str, List[np.ndarray]]] = {c: {"x": [], "y": []}
                                                    for c in names}
    refs: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in names}

    shown = 0

    def show(img):
        nonlocal shown
        surface.show_and_settle(img, SETTLE_S)
        shown += 1

    # White/black references first. They are not used by the decode -- the
    # complementary pairs handle thresholding -- but they give the coverage
    # check and a contrast figure for the report, which is how a run that
    # framed the wall badly gets caught before the fit rather than after.
    show(surface.white())
    for c in names:
        refs[c]["white"] = _grab_fresh(threads[c])
    show(surface.black())
    for c in names:
        refs[c]["black"] = _grab_fresh(threads[c])

    for axis, bits in (("x", bx), ("y", by)):
        acc = {c: None for c in names}
        for b in range(bits):
            show(stripe_image(ph, pw, b, axis, invert=False))
            pos = {c: _grab_fresh(threads[c]) for c in names}
            show(stripe_image(ph, pw, b, axis, invert=True))
            neg = {c: _grab_fresh(threads[c]) for c in names}
            progress("  %s bit %2d/%d   (%d/%d patterns)"
                     % (axis, b + 1, bits, shown, total))
            for c in names:
                if pos[c] is None or neg[c] is None:
                    raise RuntimeError(
                        "camera %r delivered no fresh frame for %s bit %d -- "
                        "it stalled or was taken by another process" % (c, axis, b))
                diff = pos[c] - neg[c]
                bit = (diff > 0).astype(np.uint32)
                if acc[c] is None:
                    acc[c] = np.zeros(bit.shape, np.uint32)
                acc[c] |= bit << b
                conf[c][axis].append(
                    np.clip(np.abs(diff), 0, 255).astype(np.uint8))
        for c in names:
            codes[c][axis] = gray_to_binary(acc[c], bits)

    return {c: {"word_x": codes[c]["x"], "word_y": codes[c]["y"],
                "conf_x": conf[c]["x"], "conf_y": conf[c]["y"],
                "white": refs[c]["white"], "black": refs[c]["black"]}
            for c in names}


def decode(raw: Dict[str, dict], proj_shape: Tuple[int, int],
           progress=print) -> Tuple[Dict[str, dict], int]:
    """Choose how many low bits to discard, then build the coarse maps.

    WHY BITS MUST BE DROPPED AT ALL. The finest Gray planes are stripes one
    projector pixel wide, and nothing in this optical chain can resolve them:
    Windows addresses the projector as 2400x1350 but the panel is 1920x1080, so
    every pattern is downscaled 1.25x before it leaves; the projector's own
    focus low-passes it further; and the wide camera sees the whole 2400-px
    projection across ~880 of its pixels, so ONE camera pixel spans nearly
    three projector pixels. Those bits do not decode to noise -- they decode to
    a coin flip, which is worse, because it looks like data.

    Dropping them is safe in a way it would NOT be under plain binary. Gray
    decoding runs MSB-first, b_i = b_{i+1} XOR g_i, so every high bit depends
    only on bits at or above it. Garbage in the low planes cannot corrupt the
    high ones -- it stays where it is and is shifted off.

    The number dropped is MEASURED per bit from its own contrast rather than
    assumed, and the same number is used for both cameras, because the join
    needs one shared projector grid.
    """
    ph, pw = proj_shape
    skip = 0
    for cam, d in raw.items():
        mask = projection_mask(d["white"], d["black"])["mask"] > 0
        if not np.any(mask):
            raise RuntimeError("camera %r saw no projection at all" % cam)
        for axis in ("x", "y"):
            meds = [float(np.median(p[mask])) for p in d["conf_" + axis]]
            progress("  %-7s %s per-bit median contrast: %s"
                     % (cam, axis, " ".join("%.0f" % m for m in meds)))
            failing = [b for b, m in enumerate(meds) if m < MIN_BIT_CONTRAST]
            if failing:
                skip = max(skip, max(failing) + 1)
    if skip:
        progress("  dropping the %d least significant bit(s): the projector "
                 "grid is quantised to %d px" % (skip, 1 << skip))

    out = {}
    for cam, d in raw.items():
        keep_x = d["conf_x"][skip:]
        keep_y = d["conf_y"][skip:]
        if not keep_x or not keep_y:
            raise RuntimeError(
                "every bit failed the contrast test for %r. The projection is "
                "too dim, out of focus, or not in frame." % cam)
        # The WEAKEST kept bit sets the pixel's confidence: one bad high-order
        # bit ruins the whole word, so the minimum is the honest summary.
        cx = np.minimum.reduce(keep_x).astype(np.float32)
        cy = np.minimum.reduce(keep_y).astype(np.float32)
        px = d["word_x"] >> skip
        py = d["word_y"] >> skip
        valid = ((cx >= MIN_BIT_CONTRAST) & (cy >= MIN_BIT_CONTRAST) &
                 (px < (pw >> skip)) & (py < (ph >> skip)))
        out[cam] = {"proj_x": px, "proj_y": py, "valid": valid,
                    "white": d["white"], "black": d["black"]}
    return out, skip


# ==========================================================================
#   aiming: get the projection framed by both cameras
# ==========================================================================
def projection_mask(white: np.ndarray, black: np.ndarray) -> dict:
    """Where the projection is in one camera's frame.

    Otsu rather than a fixed threshold, matching projector.run_check: the
    projector's brightness in frame depends on throw, room light and exposure,
    none of which are known here.
    """
    diff = np.clip(white - black, 0, 255).astype(np.uint8)
    _t, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    h, w = mask.shape[:2]
    out = {"coverage": float((mask > 0).mean()), "bbox": None,
           "centroid": None, "touches": [], "shape": (h, w), "mask": mask}
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return out
    big = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(big)
    out["bbox"] = (x, y, x + bw, y + bh)
    m = cv2.moments(big)
    if m["m00"] > 0:
        out["centroid"] = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    # A bbox against a frame edge means the projection is CLIPPED, so its
    # centroid is biased toward frame centre and the aim is not yet done.
    # Reported rather than inferred from coverage, because a projection that
    # fills the frame entirely also has high coverage and is equally clipped.
    for name, hit in (("left", x <= 1), ("right", x + bw >= w - 1),
                      ("top", y <= 1), ("bottom", y + bh >= h - 1)):
        if hit:
            out["touches"].append(name)
    return out


def _white_black(surface, threads) -> Dict[str, dict]:
    surface.show_and_settle(surface.white(), SETTLE_S)
    white = {c: _grab_fresh(t) for c, t in threads.items()}
    surface.show_and_settle(surface.black(), SETTLE_S)
    black = {c: _grab_fresh(t) for c, t in threads.items()}
    out = {}
    for c in threads:
        if white[c] is None or black[c] is None:
            raise RuntimeError("camera %r delivered no frame during aiming" % c)
        out[c] = projection_mask(white[c], black[c])
    return out


def _describe_aim(state: Dict[str, dict], progress=print) -> None:
    for cam, s in state.items():
        h, w = s["shape"]
        c = s["centroid"]
        progress("  %-7s coverage %5.1f%%  centroid %s  offset %s  clipped: %s"
                 % (cam, 100.0 * s["coverage"],
                    "(%7.1f,%7.1f)" % c if c else "      none      ",
                    "(%+7.1f,%+7.1f)" % (c[0] - w / 2.0, c[1] - h / 2.0) if c
                    else "     none     ",
                    ",".join(s["touches"]) if s["touches"] else "no"))


def aim(surface, threads, send_fn, max_iters: int = 4, probe_deg: float = 4.0,
        max_step_deg: float = 20.0, tol_px: float = 40.0,
        progress=print) -> Dict[str, dict]:
    """Slew until the projection is centred in the WIDE frame.

    NO SIGN IS ASSUMED ANYWHERE. Which way a `dmove` pushes the image depends on
    the axis mapping, the 90 deg narrow mount and the payload's yaw -- and this
    project has repeatedly been bitten by hand-derived signs. So both axes are
    PROBED: command a small move, measure how far the projection's centroid
    actually travelled, and keep the measured px-per-degree including its sign.

    The wide camera drives the loop because it sees more of the projection;
    the narrow camera is along for the ride and reported so its coverage can be
    watched. Centring in wide centres in narrow to within the boresight offset,
    which is far below the tolerance here.
    """
    state = _white_black(surface, threads)
    progress("aim: starting point")
    _describe_aim(state, progress)
    if state["wide"]["centroid"] is None:
        raise RuntimeError("the wide camera cannot see the projection at all")

    gain = {}
    for axis, (dp, dy) in (("x", (0.0, probe_deg)), ("y", (probe_deg, 0.0))):
        before = state["wide"]["centroid"]
        send_fn("dmove %.3f %.3f" % (dp, dy))
        time.sleep(0.8)
        after_state = _white_black(surface, threads)
        after = after_state["wide"]["centroid"]
        if after is None:
            send_fn("dmove %.3f %.3f" % (-dp, -dy))
            raise RuntimeError("lost the projection while probing %s" % axis)
        moved = (after[0] - before[0], after[1] - before[1])
        # The axis that MOVED is the one this command drives. Recording both
        # components keeps the cross-term visible instead of assuming the axes
        # are independent -- at non-zero yaw they are not.
        g = moved[0] / probe_deg if axis == "x" else moved[1] / probe_deg
        gain[axis] = g
        progress("  probe %s: %+.1f deg -> centroid moved (%+.1f, %+.1f) px"
                 "  -> %+.1f px/deg" % (axis, probe_deg, moved[0], moved[1], g))
        state = after_state
        if abs(g) < 1.0:
            raise RuntimeError(
                "axis %s moved the projection only %.2f px/deg. Either the "
                "motors are not driving or the measurement is noise; refusing "
                "to compute a correction from it." % (axis, g))

    # DAMPED, AND THE GAIN IS RE-MEASURED EVERY ITERATION -- the same lesson
    # recover_level.py already carries. A gain measured once is wrong later, and
    # here it is wrong from the start for a specific reason: the probe ran while
    # the projection was CLIPPED against the frame edge, and a clipped mask's
    # centroid travels less than the image does, because area entering at one
    # edge is invisible. That underestimates the gain, which overestimates every
    # correction. Measured on the first run: probe said 7.3 px/deg, the truth
    # once unclipped was 18. An overshoot factor above 2 diverges, and it did.
    DAMP = 0.7
    for i in range(max_iters):
        c = state["wide"]["centroid"]
        h, w = state["wide"]["shape"]
        ex, ey = c[0] - w / 2.0, c[1] - h / 2.0
        progress("  iter %d: offset (%+.1f, %+.1f) px  gain (%+.1f, %+.1f) px/deg"
                 % (i + 1, ex, ey, gain["x"], gain["y"]))
        if abs(ex) <= tol_px and abs(ey) <= tol_px and not state["wide"]["touches"]:
            progress("  centred.")
            break
        d_yaw = float(np.clip(DAMP * -ex / gain["x"], -max_step_deg, max_step_deg))
        d_pitch = float(np.clip(DAMP * -ey / gain["y"], -max_step_deg, max_step_deg))
        progress("          commanding dmove %+.2f pitch, %+.2f yaw" % (d_pitch, d_yaw))
        send_fn("dmove %.3f %.3f" % (d_pitch, d_yaw))
        time.sleep(0.8)
        nxt = _white_black(surface, threads)
        _describe_aim(nxt, progress)
        if nxt["wide"]["centroid"] is None:
            raise RuntimeError("lost the projection while correcting")
        # Update each gain from what the move actually achieved, but only when
        # the command was big enough for the result to be signal rather than
        # measurement noise.
        nc = nxt["wide"]["centroid"]
        for axis, cmd, moved in (("x", d_yaw, nc[0] - c[0]),
                                 ("y", d_pitch, nc[1] - c[1])):
            if abs(cmd) > 1.0 and abs(moved) > 5.0:
                gain[axis] = 0.5 * gain[axis] + 0.5 * (moved / cmd)
        state = nxt
    return state


# ==========================================================================
#   the laser dot, in BOTH frames at once
# ==========================================================================
def green_excess(img: np.ndarray) -> np.ndarray:
    """G - max(R, B). Lifted from measure_latency._green_excess32, unchanged.

    SIGNED, and that is the point: the beam can only push a pixel greener, so
    anything turning REDDER is not evidence of it. An absolute difference would
    score a warm reflection as a dot.
    """
    bgr = img.astype(np.int16)
    return (bgr[:, :, 1] -
            np.maximum(bgr[:, :, 0], bgr[:, :, 2])).astype(np.float32)


def find_dot(ref: np.ndarray, lit: np.ndarray) -> dict:
    """Locate the beam by the change it caused, in chroma. -> peak, centroid.

    Blurred before the peak because DOT_AREA_PX says a real dot spans 3-40 px:
    it survives a 3x3 blur and a hot single pixel from sensor noise does not.
    The centroid is taken over the pixels above half the peak, which is
    sub-pixel and far steadier than the argmax.
    """
    d = cv2.GaussianBlur(cv2.subtract(green_excess(lit), green_excess(ref)),
                         (3, 3), 0)
    peak = float(d.max())
    floor = float(np.median(d))
    noise = float(np.std(d))
    mask = d >= max(0.5 * peak, floor + 6.0 * noise)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"peak": peak, "noise": noise, "centroid": None, "n_px": 0}
    wts = d[ys, xs].astype(np.float64)
    return {"peak": peak, "noise": noise, "n_px": int(len(xs)),
            "snr": (peak - floor) / max(noise, 1e-6),
            "centroid": (float((xs * wts).sum() / wts.sum()),
                         float((ys * wts).sum() / wts.sum()))}


def measure_laser(surface, threads, link, n_average: int = 4,
                  progress=print) -> Dict[str, dict]:
    """Beam off -> reference, beam on -> lit, in BOTH cameras simultaneously.

    NO MOTION. The laser and both cameras are bolted to one payload, so the
    beam's position in camera coordinates does not depend on where the turret
    points -- only on range, through parallax. So the dot can simply be pulsed
    where the platform already looks, and read off. Centring it on a target
    would measure the same quantity with a servo loop in the way.

    The projector is held BLACK throughout. Chroma alone would tolerate a lit
    wall (that is the whole argument in measure_latency's --wall mode), but a
    projected image can contain real green, and there is no reason to make the
    detector fight one.
    """
    surface.show_and_settle(surface.black(), 0.5)
    link.laser(False)
    time.sleep(0.4)
    ref = {c: _grab_fresh_colour(t, n_average) for c, t in threads.items()}

    progress("  beam ON")
    link.laser(True)
    time.sleep(0.5)
    lit = {c: _grab_fresh_colour(t, n_average) for c, t in threads.items()}
    link.laser(False)
    progress("  beam OFF")

    out = {}
    for c in threads:
        if ref[c] is None or lit[c] is None:
            raise RuntimeError("camera %r delivered no frame for the dot" % c)
        out[c] = find_dot(ref[c], lit[c])
    return out


def _grab_fresh_colour(thread, n_average: int):
    """As _grab_fresh but keeps colour -- the chroma test needs all 3 channels."""
    first, _ = thread.latest()
    last_index = first.index if first is not None else -1
    deadline = time.time() + SETTLE_S + 2.0
    seen = 0
    acc: List[np.ndarray] = []
    while time.time() < deadline and len(acc) < n_average:
        f, _ = thread.latest()
        if f is not None and f.index != last_index:
            last_index = f.index
            seen += 1
            if seen > FRESH_FRAMES:
                acc.append(f.image.astype(np.float32))
        else:
            cv2.waitKey(5)
    if not acc:
        return None
    return np.mean(acc, axis=0)


# ==========================================================================
#   join the two cameras through the projector
# ==========================================================================
def join(decoded: Dict[str, dict], proj_shape: Tuple[int, int],
         cam_a: str = "wide", cam_b: str = "narrow"):
    """-> (pts_a, pts_b) sub-pixel camera coordinates, one row per shared
    projector pixel.

    The projector pixel is the JOIN KEY. Each camera independently answers
    "which projector pixel am I looking at"; inverting both and intersecting on
    that key gives correspondences without ever matching image features across
    a 40 deg and a 108 deg lens, which is the step that would otherwise be hard.

    Several camera pixels usually see one projector pixel, so the inverse is
    the CENTROID of those pixels. That is an average over a handful of samples,
    so the recovered camera coordinate is sub-pixel -- better than the integer
    grid either image is sampled on.
    """
    ph, pw = proj_shape
    n = ph * pw
    centroids = {}
    for cam in (cam_a, cam_b):
        d = decoded[cam]
        v = d["valid"]
        ys, xs = np.nonzero(v)
        key = d["proj_y"][v].astype(np.int64) * pw + d["proj_x"][v].astype(np.int64)
        count = np.bincount(key, minlength=n)
        sum_x = np.bincount(key, weights=xs.astype(np.float64), minlength=n)
        sum_y = np.bincount(key, weights=ys.astype(np.float64), minlength=n)
        centroids[cam] = (count, sum_x, sum_y)

    ca, sxa, sya = centroids[cam_a]
    cb, sxb, syb = centroids[cam_b]
    both = (ca > 0) & (cb > 0)
    if not np.any(both):
        raise RuntimeError(
            "no projector pixel was decoded by BOTH cameras. The two fields do "
            "not overlap on the projection -- check that the wall is framed by "
            "each camera, and that the turret has not moved between them.")
    pts_a = np.stack([sxa[both] / ca[both], sya[both] / ca[both]], axis=1)
    pts_b = np.stack([sxb[both] / cb[both], syb[both] / cb[both]], axis=1)
    return pts_a, pts_b


# ==========================================================================
#   report
# ==========================================================================
def _residuals(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - truth, axis=1)


def radius_report(wide_pts: np.ndarray, resid: np.ndarray,
                  n_bins: int = 6) -> List[tuple]:
    """Residual against distance from the WIDE camera's axis.

    This is the measurement the whole run exists for. Barrel distortion is zero
    on-axis and grows with field radius, so a model that ignores it produces a
    residual that is small in the middle and large at the rim. A single pooled
    RMS cannot show that and this can.
    """
    cx, cy = config.WIDE_SIZE[0] / 2.0, config.WIDE_SIZE[1] / 2.0
    r = np.hypot(wide_pts[:, 0] - cx, wide_pts[:, 1] - cy)
    r_max = float(r.max()) if r.size else 1.0
    edges = np.linspace(0.0, r_max, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1] if i < n_bins - 1 else r <= edges[i + 1])
        if not np.any(m):
            continue
        rr = resid[m]
        rows.append((edges[i], edges[i + 1], int(m.sum()),
                     float(np.sqrt(np.mean(rr ** 2))),
                     float(np.percentile(rr, 95)), float(rr.max())))
    return rows


def fit_models(wide_pts: np.ndarray, narrow_pts: np.ndarray,
               sample: int = 60000, seed: int = 0) -> Dict[str, dict]:
    """Fit the candidate models and score each on the SAME points.

    Subsampled for the fit because a million-row design matrix buys nothing a
    sixty-thousand-row one does not; every model is scored on all points.
    """
    rng = np.random.default_rng(seed)
    n = len(wide_pts)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    w_s, n_s = wide_pts[idx], narrow_pts[idx]
    out: Dict[str, dict] = {}

    # 1. The CURRENT stored similarity, as-is. Not refitted -- this is the
    #    number that says how wrong today's overlay actually is.
    stored = WideToNarrow.load(str(FIT_PATH))
    pred = np.array([stored.point(float(x), float(y)) for x, y in wide_pts])
    out["stored similarity (wide_to_narrow.json)"] = {
        "resid": _residuals(pred, narrow_pts), "params": 4, "model": stored}

    # 2. Similarity refitted on these correspondences, same 4-parameter form.
    pairs = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
             for a, b in zip(w_s, n_s)]
    refit = WideToNarrow.measure(pairs)
    pred = np.array([refit.point(float(x), float(y)) for x, y in wide_pts])
    out["similarity refitted"] = {"resid": _residuals(pred, narrow_pts),
                                  "params": 4, "model": refit}

    # 3. Affine: adds shear and independent axis scales.
    # Plain least squares, NOT cv2.estimateAffine2D: that defaults to a robust
    # estimator which minimises the MEDIAN residual, so its RMS can come out
    # WORSE than the 4-parameter similarity it strictly contains -- which it
    # did on the first run here (4.28 vs 3.95) and which is a scoring artefact,
    # not a property of the model. Every model in this table must be fitted
    # under the same loss as the table reports.
    design = np.hstack([w_s, np.ones((len(w_s), 1))])
    coef, *_ = np.linalg.lstsq(design, n_s, rcond=None)
    pred = np.hstack([wide_pts, np.ones((len(wide_pts), 1))]) @ coef
    out["affine"] = {"resid": _residuals(pred, narrow_pts), "params": 6,
                     "model": coef}

    # 4. Homography: the correct model for two views of ONE PLANE through
    #    distortion-free lenses. What it cannot absorb is lens distortion --
    #    so the residual left here is a direct read on how non-rectilinear
    #    the wide lens is.
    H, _ = cv2.findHomography(w_s.astype(np.float64), n_s.astype(np.float64),
                              method=0)
    if H is not None:
        ones = np.ones((len(wide_pts), 1))
        proj = np.hstack([wide_pts, ones]) @ H.T
        pred = proj[:, :2] / proj[:, 2:3]
        out["homography"] = {"resid": _residuals(pred, narrow_pts),
                             "params": 8, "model": H}
    return out


def print_report(wide_pts, narrow_pts, models, progress=print) -> None:
    progress("")
    progress("=" * 72)
    progress("CORRESPONDENCES: %d projector pixels decoded by both cameras" % len(wide_pts))
    progress("=" * 72)
    progress("")
    progress("%-42s %6s %8s %8s %8s" % ("model", "params", "RMS px", "p95 px", "max px"))
    progress("-" * 76)
    for name, m in models.items():
        r = m["resid"]
        progress("%-42s %6d %8.2f %8.2f %8.2f"
                 % (name, m["params"], float(np.sqrt(np.mean(r ** 2))),
                    float(np.percentile(r, 95)), float(r.max())))

    progress("")
    progress("RESIDUAL vs FIELD RADIUS -- stored similarity (today's overlay)")
    progress("%12s %10s %9s %9s %9s" % ("radius px", "points", "RMS", "p95", "max"))
    progress("-" * 54)
    for lo, hi, n, rms, p95, mx in radius_report(
            wide_pts, models["stored similarity (wide_to_narrow.json)"]["resid"]):
        progress("%5.0f-%5.0f %10d %9.2f %9.2f %9.2f" % (lo, hi, n, rms, p95, mx))

    if "homography" in models:
        progress("")
        progress("RESIDUAL vs FIELD RADIUS -- homography (distortion is what is left)")
        progress("%12s %10s %9s %9s %9s" % ("radius px", "points", "RMS", "p95", "max"))
        progress("-" * 54)
        for lo, hi, n, rms, p95, mx in radius_report(
                wide_pts, models["homography"]["resid"]):
            progress("%5.0f-%5.0f %10d %9.2f %9.2f %9.2f" % (lo, hi, n, rms, p95, mx))


# ==========================================================================
#   self-test
# ==========================================================================
def self_test() -> int:
    print("SELF-TEST: no hardware, no projector, no cameras\n")
    bad = 0

    # 1. Gray round-trip over every column of a real projector width.
    for w in (1920, 1280, 1024, 800):
        b = n_bits(w)
        idx = np.arange(w, dtype=np.uint32)
        back = gray_to_binary(gray_of(idx), b)
        ok = bool(np.array_equal(back, idx))
        bad += 0 if ok else 1
        print("  gray round-trip w=%4d bits=%2d  [%s]" % (w, b, "OK" if ok else "FAIL"))

    # 2. Adjacent columns must differ in exactly ONE bit -- the property the
    #    whole choice of Gray over binary rests on.
    g = gray_of(np.arange(1920, dtype=np.uint32))
    popcount = np.array([bin(int(a) ^ int(b)).count("1") for a, b in zip(g[:-1], g[1:])])
    ok = bool(np.all(popcount == 1))
    bad += 0 if ok else 1
    print("  adjacent columns differ in 1 bit: max=%d  [%s]"
          % (popcount.max(), "OK" if ok else "FAIL"))

    # 3. Stripe images must actually carry the bit they claim.
    img = stripe_image(64, 1920, 3, "x", invert=False)
    plane = (img[0, :, 0] > 127).astype(np.uint32)
    ok = bool(np.array_equal(plane, (gray_of(np.arange(1920, dtype=np.uint32)) >> 3) & 1))
    bad += 0 if ok else 1
    print("  stripe_image bit 3 matches gray plane  [%s]" % ("OK" if ok else "FAIL"))

    inv = stripe_image(64, 1920, 3, "x", invert=True)
    ok = bool(np.array_equal((inv[0, :, 0] > 127).astype(np.uint32), 1 - plane))
    bad += 0 if ok else 1
    print("  inverted plane is the complement      [%s]" % ("OK" if ok else "FAIL"))

    # 4. End-to-end join against a KNOWN transform. Synthesise what each camera
    #    would decode if it viewed the projector through a known map, then check
    #    the join recovers correspondences consistent with it.
    ph, pw = 270, 480                      # small projector, keeps the test quick
    truth = WideToNarrow(scale=1.4137, offset_x=0.32, offset_y=-12.55,
                         rotate_cw_deg=90, provenance="synthetic truth")
    wide_h, wide_w = 200, 360
    decoded = {}
    yy, xx = np.mgrid[0:wide_h, 0:wide_w]
    # wide camera sees the projector as a simple scaled window
    px = np.clip((xx * (pw / wide_w)).astype(np.uint32), 0, pw - 1)
    py = np.clip((yy * (ph / wide_h)).astype(np.uint32), 0, ph - 1)
    decoded["wide"] = {"proj_x": px, "proj_y": py,
                       "valid": np.ones((wide_h, wide_w), bool)}
    # narrow camera: the same projector pixels, but its own image coordinates
    # are the truth transform of the wide ones
    nx, ny = [], []
    for x, y in zip(xx.ravel(), yy.ravel()):
        a, b = truth.point(float(x), float(y))
        nx.append(a)
        ny.append(b)
    nx = np.array(nx).reshape(wide_h, wide_w)
    ny = np.array(ny).reshape(wide_h, wide_w)
    keep = (nx >= 0) & (nx < 4000) & (ny >= 0) & (ny < 4000)
    # build the narrow decode on its own integer grid by rounding
    n_h, n_w = int(np.ceil(ny[keep].max())) + 1, int(np.ceil(nx[keep].max())) + 1
    n_px = np.zeros((n_h, n_w), np.uint32)
    n_py = np.zeros((n_h, n_w), np.uint32)
    n_valid = np.zeros((n_h, n_w), bool)
    ix = np.clip(np.round(nx[keep]).astype(int), 0, n_w - 1)
    iy = np.clip(np.round(ny[keep]).astype(int), 0, n_h - 1)
    n_px[iy, ix] = px[keep]
    n_py[iy, ix] = py[keep]
    n_valid[iy, ix] = True
    decoded["narrow"] = {"proj_x": n_px, "proj_y": n_py, "valid": n_valid}

    w_pts, n_pts = join(decoded, (ph, pw))
    pred = np.array([truth.point(float(a), float(b)) for a, b in w_pts])
    err = _residuals(pred, n_pts)
    ok = float(np.sqrt(np.mean(err ** 2))) < 2.0
    bad += 0 if ok else 1
    print("  join recovers the known transform: %d pts, RMS %.2f px  [%s]"
          % (len(w_pts), float(np.sqrt(np.mean(err ** 2))), "OK" if ok else "FAIL"))

    print("\n%s" % ("SELF-TEST PASSED" if bad == 0 else "SELF-TEST FAILED (%d)" % bad))
    return 1 if bad else 0


# ==========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="verify the codec and the join with no hardware")
    p.add_argument("--exposure", type=float, default=-6.0,
                   help="locked exposure for BOTH cameras (default -6, what "
                        "app.NARROW_EXPOSURE uses in flight)")
    p.add_argument("--narrow-gain", type=float, default=64.0)
    p.add_argument("--laser", action="store_true",
                   help="pulse the beam and locate the dot in BOTH cameras, "
                        "then check the dense map against it. Fires the laser "
                        "at the wall the platform already points at; commands "
                        "no motion.")
    p.add_argument("--aim", action="store_true",
                   help="slew until the projection is centred in the wide "
                        "frame, THEN capture. Moves the platform.")
    p.add_argument("--aim-only", action="store_true",
                   help="aim and stop, without the Gray-code capture")
    p.add_argument("--wall-in", type=float, default=None,
                   help="distance from the camera plate to the wall, INCHES. "
                        "Recorded with the map and used to predict the "
                        "parallax, which is the error bound when the map is "
                        "used at any other range.")
    p.add_argument("--write", action="store_true",
                   help="overwrite wide_to_narrow.json with the refitted "
                        "similarity. OFF by default: the dense map and the "
                        "residual report are the deliverable, and replacing a "
                        "measured file should be a decision, not a side effect.")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    from turret_host import cameras, projector

    # -- projector -------------------------------------------------------
    mons = projector.enumerate_monitors()
    print("monitors: %s" % mons)
    proj_mon = projector.pick_projector(mons)
    if proj_mon is None:
        print("\nNo second display. ProjectorSurface would SILENTLY fall back to\n"
              "the primary panel and report PRIMARY in describe(); the cameras\n"
              "would then see a wall with nothing on it and every bit would\n"
              "decode as noise. Extend (not mirror) the projector and re-run.")
        return 2
    print("projector: %s" % (proj_mon,))

    # -- cameras ---------------------------------------------------------
    ident = cameras.identify_cameras()
    print(ident.report())
    for name, idx, gain in (("narrow", ident.narrow_index, args.narrow_gain),
                            ("wide", ident.wide_index, None)):
        res = cameras.lock_exposure(idx, args.exposure, gain=gain)
        print("  %-7s %s" % (name, res.message))
        if not res.ok:
            print("\nREFUSING TO RUN. Auto-exposure will hunt against the\n"
                  "patterns and suppress the very contrast the decode reads.")
            return 3

    threads = {}
    surface = None
    try:
        threads["narrow"] = cameras.narrow_thread(ident.narrow_index).start()
        # Full 1920x1080, never --wide-fast: wide_to_narrow.json's convention
        # is WIDE_SIZE, and the fast mode is a CENTRE CROP, so a map measured
        # in it would be offset by (320, 180) wide px against every consumer.
        threads["wide"] = cameras.wide_thread(ident.wide_index, fast=False).start()
        for n, t in threads.items():
            print("  %-7s %.1f fps at startup" % (n, t.startup_fps))

        surface = projector.ProjectorSurface(proj_mon).open()
        if surface.mirrored:
            print("\nProjectorSurface reports MIRRORED -- it fell back to the "
                  "primary display. Refusing.")
            return 2

        if args.laser:
            from turret_host import calibrate as calmod
            link = calmod.CalibrationLink()
            link.open()
            try:
                print("\nLASER: 5 mW 532 nm, pulsed at the wall the platform "
                      "already points at. No motion.")
                dots = measure_laser(surface, threads, link)
            finally:
                try:
                    link.laser(False)
                finally:
                    link.close()

            for cam, d in dots.items():
                print("  %-7s peak %+7.1f  noise %5.2f  snr %6.1f  %4d px  "
                      "centroid %s"
                      % (cam, d["peak"], d["noise"], d.get("snr", 0.0),
                         d["n_px"],
                         "(%7.2f, %7.2f)" % d["centroid"] if d["centroid"]
                         else "NOT FOUND"))

            if dots["wide"]["centroid"] and dots["narrow"]["centroid"]:
                wx, wy = dots["wide"]["centroid"]
                nx, ny = dots["narrow"]["centroid"]
                print("\nINDEPENDENT CHECK OF THE MAP")
                print("  The dot is ONE physical point seen by both cameras, and")
                print("  it is NOT part of the data any transform below was fitted")
                print("  to. Mapping the wide dot should land on the narrow dot.")
                stored = WideToNarrow.load(str(FIT_PATH))
                px, py = stored.point(wx, wy)
                print("    stored similarity -> (%7.2f, %7.2f)  err %6.2f px"
                      % (px, py, float(np.hypot(px - nx, py - ny))))
                try:
                    d = np.load(args.out, allow_pickle=True)
                    W = d["wide_pts"].astype(np.float64)
                    N = d["narrow_pts"].astype(np.float64)
                    i = np.random.default_rng(0).choice(len(W),
                                                        min(60000, len(W)),
                                                        replace=False)
                    refit = WideToNarrow.measure(
                        [((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                         for a, b in zip(W[i], N[i])])
                    rx, ry = refit.point(wx, wy)
                    print("    refitted similarity -> (%7.2f, %7.2f)  err %6.2f px"
                          % (rx, ry, float(np.hypot(rx - nx, ry - ny))))
                    H, _ = cv2.findHomography(W[i], N[i], method=0)
                    q = np.array([wx, wy, 1.0]) @ H.T
                    hx, hy = q[0] / q[2], q[1] / q[2]
                    print("    homography         -> (%7.2f, %7.2f)  err %6.2f px"
                          % (hx, hy, float(np.hypot(hx - nx, hy - ny))))
                except (OSError, KeyError, ValueError) as exc:
                    print("    (no dense map to compare against: %s)" % exc)

                print("\n  goal_pixel.json says the beam sits at (635.08, 327.77)")
                print("  in the narrow frame; measured here (%7.2f, %7.2f), "
                      "delta %.2f px." % (nx, ny,
                                          float(np.hypot(nx - 635.08, ny - 327.77))))
            return 0

        if args.aim or args.aim_only:
            sys.path.insert(0, str(_ROOT / "tools"))
            import send as sendmod

            def send_fn(cmd: str) -> None:
                sendmod.send([cmd], timeout=30, quiet=True)

            print("\nAIMING -- the platform will move.")
            aim(surface, threads, send_fn)
            if args.aim_only:
                return 0

        print("\nDO NOT MOVE THE TURRET AND STAY OUT OF FRAME UNTIL THIS FINISHES.")
        if args.wall_in:
            r_m = args.wall_in * 0.0254
            par_n = config.NARROW_F_PX * (config.STEREO_BASELINE_MM / 1000.0) / r_m
            par_w = config.WIDE_F_PX * (config.STEREO_BASELINE_MM / 1000.0) / r_m
            print("wall at %.1f in = %.3f m. Predicted disparity between the two "
                  "views at this range: %.1f narrow px / %.1f wide px."
                  % (args.wall_in, r_m, par_n, par_w))
            print("That is the scale of the error if this map is used at a "
                  "DIFFERENT range -- it falls as 1/R.")
        t0 = time.time()
        raw = capture_sequence(surface, threads)
        print("capture took %.1f s" % (time.time() - t0))
        decoded, skip = decode(raw, surface.shape)
        coarse = (surface.shape[0] >> skip, surface.shape[1] >> skip)

        # Coverage check BEFORE the fit: a run where one camera barely saw the
        # projection produces a confident-looking fit over a tiny patch, which
        # is the failure mode that is hardest to notice afterwards.
        for cam, d in decoded.items():
            frac = float(d["valid"].mean())
            swing = float(np.mean(d["white"] - d["black"]))
            print("  %-7s decoded %.1f%% of its pixels, white-black swing %.1f levels"
                  % (cam, 100.0 * frac, swing))
            if frac < MIN_COVERAGE_FRAC:
                print("\n%s decoded almost nothing. The projection is not in its "
                      "field, or the room is too bright for the swing to clear "
                      "%.0f levels." % (cam, MIN_BIT_CONTRAST))
                return 4

        wide_pts, narrow_pts = join(decoded, coarse)
        models = fit_models(wide_pts, narrow_pts)
        print_report(wide_pts, narrow_pts, models)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, wide_pts=wide_pts.astype(np.float32),
            narrow_pts=narrow_pts.astype(np.float32),
            projector_shape=np.array(surface.shape),
            wide_size=np.array(config.WIDE_SIZE),
            narrow_size=np.array(config.NARROW_SIZE),
            wall_m=np.array(args.wall_in * 0.0254 if args.wall_in else -1.0),
            note=np.array(
                "Dense wide<->narrow correspondence through the projected wall "
                "plane. Valid ON THAT PLANE ONLY: off it the error is the "
                "disparity, which goes as 1/range. narrow_pts are STORED "
                "(unrotated) narrow pixels, matching wide_to_narrow.json."))
        print("\nsaved -> %s  (%d correspondences)" % (out, len(wide_pts)))

        if args.write:
            refit = models["similarity refitted"]["model"]
            refit.save(str(FIT_PATH),
                       saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            print("wrote -> %s" % FIT_PATH)
        else:
            print("wide_to_narrow.json NOT modified (pass --write to replace it)")
        return 0
    finally:
        if surface is not None:
            surface.close()
        for t in threads.values():
            t.stop()


if __name__ == "__main__":
    raise SystemExit(main())
