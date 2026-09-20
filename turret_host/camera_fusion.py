"""Map wide-camera detections into the narrow frame, so BOTH cameras can track.

    from turret_host.camera_fusion import WideToNarrow, FusedDetection
    w2n = WideToNarrow.load()          # measured if available, derived if not
    narrow_box = w2n.box(wide_box)

WHY THIS EXISTS
---------------
The control law consumes narrow pixels and nothing else, because the Jacobian
is measured in narrow pixels (`jacobian.json` records image_size 1280x720) and
the goal pixel lives in the narrow frame. So a wide detection currently cannot
command motion at all -- `app.py` routes it to `wide_det_slot`, commented
"display only", and the tracker never sees it.

That is the wrong trade for this machine. The narrow camera's horizontal field
is 28.8 degrees against the wide camera's 107.9 -- nearly four times narrower.
A drone moving fast leaves the narrow frame routinely, and at the exact moment
it does, the only camera that can still see it is wired to a display panel.

Mapping wide pixels into the narrow frame lets the same tracker and the same
Jacobian consume both. Wide keeps the target while it is outside the narrow
field; narrow takes over for the precision work.

TRACK ON EITHER. FIRE ONLY ON NARROW.
-------------------------------------
A mapped wide box is a much coarser statement than a narrow one: one wide pixel
covers about two narrow pixels at the axis and more toward the edge, before
any distortion is accounted for. That precision is fine for "slew until the
drone is back in the narrow field" and NOT fine for "the beam path is inside
the airframe".

So `FusedDetection.source` is carried all the way through. The interlock's
`drone_lock` must require source == "narrow": tracking may be driven by either
camera, firing may not. That is not conservatism for its own sake -- a wide
box scaled up by 2x has its edge error scaled up by 2x too, and the beam is
positioned against that edge.

WHAT THE DEFAULT MAPPING ASSUMES, AND WHY IT IS ONLY A START
-------------------------------------------------------------
With no measured fit, the transform is derived from focal lengths alone:
both cameras are rigidly co-mounted and approximately boresighted, so a ray at
angle theta from the axis lands at f*tan(theta) in each. The scale is therefore
NARROW_F_PX / WIDE_F_PX = 2.0, with the narrow frame's 90-degree mount applied
on top.

Three things that assumption ignores, in order of how much they hurt:

  * DISTORTION. At 107.9 degrees the wide camera has real barrel distortion,
    so a single linear scale is only valid near the axis. This is why the
    intrinsics calibration matters more for the wide camera than the narrow.
  * BORESIGHT OFFSET. The two optical axes are not identical; the residual
    shows up as a constant pixel offset, which `measure()` fits.
  * PARALLAX. The cameras are separated, so the mapping is range-dependent.
    Small at 2-5 m for a ~60 mm baseline, but not zero.

`measure()` fits scale and offset from correspondences and is strictly better
than the derived default. `load()` prefers a stored fit and falls back to the
derivation, reporting which one it used, because a mapping of unknown
provenance is exactly the kind of thing that gets trusted by mistake.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _pkg_dir:
    _sys.path[0] = _os.path.dirname(_pkg_dir)

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from turret_host import config
from turret_host.types import Detection

FIT_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "calibration", "wide_to_narrow.json")


@dataclass
class FusedDetection:
    """A detection plus WHICH camera produced it. The source is not decoration.

    `drone_lock` in the interlock must refuse anything that is not "narrow";
    the field exists so that decision can be made at all. A bare Detection
    reaching the control law carries no way to tell a 1-pixel narrow edge from
    a 2-pixel-equivalent mapped wide one.
    """
    box: Detection
    source: str                      # "narrow" | "wide->narrow"
    frame_t: float
    mapped: bool = False


@dataclass
class WideToNarrow:
    """Similarity transform, wide pixels -> narrow (stored, unrotated) pixels."""
    scale: float
    offset_x: float
    offset_y: float
    rotate_cw_deg: int               # narrow mount; 90 on this machine
    provenance: str                  # "measured" | "derived from focal lengths"
    rms_px: Optional[float] = None
    n_points: int = 0
    # 0 = parallax correction DISABLED. +1/-1 once the direction is measured.
    # Off by default on purpose: the magnitude is computable from CAD, the sign
    # is not, and applying the wrong one is worse than applying none.
    parallax_sign: int = 0

    # -- construction ------------------------------------------------------
    @staticmethod
    def derived() -> "WideToNarrow":
        """From focal lengths alone. A starting point, not a calibration."""
        scale = config.NARROW_F_PX / config.WIDE_F_PX
        return WideToNarrow(
            scale=scale, offset_x=0.0, offset_y=0.0,
            rotate_cw_deg=config.NARROW_ROTATION_DEG
            if config.NARROW_ROTATE_CLOCKWISE else -config.NARROW_ROTATION_DEG,
            provenance="derived from focal lengths (NARROW_F_PX/WIDE_F_PX="
                       "%.2f); boresight offset and distortion NOT accounted"
                       % scale)

    @staticmethod
    def load(path: str = FIT_PATH) -> "WideToNarrow":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            return WideToNarrow(
                scale=float(blob["scale"]),
                offset_x=float(blob["offset_x"]),
                offset_y=float(blob["offset_y"]),
                rotate_cw_deg=int(blob["rotate_cw_deg"]),
                provenance="measured %s" % blob.get("saved_at", "?"),
                rms_px=blob.get("rms_px"),
                n_points=int(blob.get("n_points", 0)))
        except (OSError, ValueError, KeyError, TypeError):
            return WideToNarrow.derived()

    def save(self, path: str = FIT_PATH, saved_at: str = "") -> None:
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"scale": self.scale, "offset_x": self.offset_x,
                       "offset_y": self.offset_y,
                       "rotate_cw_deg": self.rotate_cw_deg,
                       "rms_px": self.rms_px, "n_points": self.n_points,
                       "saved_at": saved_at,
                       "note": "wide pixels -> narrow STORED (unrotated) "
                               "pixels. Linear, so valid near the axis; the "
                               "wide camera's barrel distortion is not "
                               "modelled here."}, fh, indent=2)
        _os.replace(tmp, path)

    # -- the mapping -------------------------------------------------------
    def parallax_px(self, range_m: float) -> float:
        """Separation between the two views of one object, in narrow pixels.

        CAD puts the cameras 60 mm apart (wide x=-30, narrow x=+30, both on
        y=0), so an object at range R appears displaced between them by
        baseline/R radians, which is f_narrow * baseline / R pixels. At the
        assumed 3 m that is 1400 * 0.060 / 3 = 28 px -- small against a 132 px
        target, large against the 25 px MAX_ERROR_TO_FIRE_PX.

        Returned as a MAGNITUDE with no sign applied, and `parallax_sign`
        defaults to 0 so the correction is OFF until someone measures which way
        it goes. config.py is explicit about why: "DO NOT hand-derive the
        parallax sign ... The sign ambiguity only ever bites someone who
        hard-codes the correction from the CAD numbers." A wrong sign does not
        halve the error, it doubles it.
        """
        r = max(1e-3, float(range_m))
        return config.NARROW_F_PX * (config.STEREO_BASELINE_MM / 1000.0) / r

    def point(self, x: float, y: float,
              range_m: Optional[float] = None) -> Tuple[float, float]:
        """One wide pixel -> narrow stored pixel.

        INCOMPATIBLE WITH `app.py --wide-fast`, AND SILENTLY SO. The wide frame
        centre below is config.WIDE_SIZE/2 = (960, 540), but --wide-fast opens
        the camera at WIDE_FAST_SIZE (1280x720), which cameras.wide_thread
        documents as a CENTRE CROP rather than a downscale. A cropped frame's
        centre is (640, 360), so every mapped point is displaced by (320, 180)
        wide px -- about 450 narrow px after the scale, against a GATE_PX of 60.
        The fit in wide_to_narrow.json was captured at full WIDE_SIZE, so it
        cannot be reinterpreted either; the correct fix is to carry the capture
        size on the transform and use the ACTUAL frame centre here.

        This matters because PLAN_NEXT_STAGE.md 5.4 proposes forcing 720p60 for
        the sample-rate win, which would walk straight into it.
        """
        wcx, wcy = config.WIDE_SIZE[0] / 2.0, config.WIDE_SIZE[1] / 2.0
        ncx, ncy = config.NARROW_SIZE[0] / 2.0, config.NARROW_SIZE[1] / 2.0
        dx, dy = (x - wcx) * self.scale, (y - wcy) * self.scale
        # The narrow sensor is mounted 90 deg over, so a displacement that is
        # horizontal in the world is VERTICAL in the stored narrow frame. The
        # rotation belongs here, in the mapping, and nowhere else -- labels,
        # the Jacobian and the goal pixel are all in stored coordinates.
        th = math.radians(self.rotate_cw_deg)
        rx = dx * math.cos(th) + dy * math.sin(th)
        ry = -dx * math.sin(th) + dy * math.cos(th)
        # Parallax lands on the narrow sensor's VERTICAL axis, not its
        # horizontal one: the baseline is horizontal in the PAYLOAD frame and
        # the C270 is mounted 90 deg over, so in stored coordinates that is the
        # ROW. config.py makes the same point about the laser offset.
        par = 0.0
        if self.parallax_sign and range_m is not None:
            par = self.parallax_sign * self.parallax_px(range_m)
        return ncx + rx + self.offset_x, ncy + ry + self.offset_y + par

    def box(self, d: Detection, range_m: Optional[float] = None) -> Detection:
        """A wide detection -> the same object in narrow stored coordinates.

        Corners are mapped and then re-min/maxed rather than mapped as a
        (centre, w, h) triple: the 90-degree rotation swaps which corner is
        which, and carrying width and height through unchanged would produce a
        box that is transposed relative to the object it describes.
        """
        xs, ys = [], []
        for px, py in ((d.x1, d.y1), (d.x2, d.y1), (d.x2, d.y2), (d.x1, d.y2)):
            mx, my = self.point(px, py, range_m)
            xs.append(mx)
            ys.append(my)
        return Detection(min(xs), min(ys), max(xs), max(ys), d.conf, d.label,
                         source="wide->narrow")

    def in_narrow_view(self, d: Detection, margin_px: float = 0.0) -> bool:
        """Would this mapped box land inside the narrow frame at all?"""
        w, h = config.NARROW_SIZE
        return not (d.x2 < -margin_px or d.x1 > w + margin_px or
                    d.y2 < -margin_px or d.y1 > h + margin_px)

    # -- fitting -----------------------------------------------------------
    @staticmethod
    def measure(pairs: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
                rotate_cw_deg: Optional[int] = None) -> "WideToNarrow":
        """Least-squares scale and offset from (wide_px, narrow_px) pairs.

        The rotation is NOT fitted. It is a property of how the camera is
        bolted on, known exactly to be 90 degrees, and fitting a known
        quantity from noisy correspondences can only make it worse -- it would
        absorb error that belongs to the offset.
        """
        if len(pairs) < 2:
            raise ValueError("need at least 2 correspondences, got %d" % len(pairs))
        rot = (rotate_cw_deg if rotate_cw_deg is not None
               else (config.NARROW_ROTATION_DEG
                     if config.NARROW_ROTATE_CLOCKWISE
                     else -config.NARROW_ROTATION_DEG))
        base = WideToNarrow(1.0, 0.0, 0.0, rot, "fitting")

        # SOLVE SCALE AND OFFSET JOINTLY, about their own centroids.
        #
        # Solving scale first with the offset assumed zero and then taking the
        # offset as the mean residual is only correct when the points happen to
        # be centred on the axis. With real correspondences -- wherever the
        # target happened to be -- it is not, and the offset leaks into the
        # scale: measured on a synthetic transform it recovered 1.8305 for a
        # true 1.8300 and left 0.38 px of RMS that should have been zero.
        # Centring both sets first makes the two parameters separable for real.
        ncx, ncy = config.NARROW_SIZE[0] / 2.0, config.NARROW_SIZE[1] / 2.0
        rotated: List[Tuple[float, float]] = []
        targets: List[Tuple[float, float]] = []
        for (wx, wy), (nx, ny) in pairs:
            ux, uy = base.point(wx, wy)          # scale 1, no offset
            rotated.append((ux - ncx, uy - ncy))
            targets.append((nx - ncx, ny - ncy))

        mux = sum(p[0] for p in rotated) / len(rotated)
        muy = sum(p[1] for p in rotated) / len(rotated)
        mtx = sum(p[0] for p in targets) / len(targets)
        mty = sum(p[1] for p in targets) / len(targets)

        num = den = 0.0
        for (ux, uy), (tx, ty) in zip(rotated, targets):
            cux, cuy = ux - mux, uy - muy
            num += cux * (tx - mtx) + cuy * (ty - mty)
            den += cux * cux + cuy * cuy
        if den <= 0:
            raise ValueError("degenerate correspondences: every point is at "
                             "the same place, so no scale is observable")
        scale = num / den
        ox, oy = mtx - scale * mux, mty - scale * muy

        fit = WideToNarrow(scale, ox, oy, rot, "measured", n_points=len(pairs))
        err = 0.0
        for (wx, wy), (nx, ny) in pairs:
            mx, my = fit.point(wx, wy)
            err += (mx - nx) ** 2 + (my - ny) ** 2
        fit.rms_px = math.sqrt(err / len(pairs))
        return fit

    def describe(self) -> str:
        s = ("wide->narrow: scale %.3f, offset (%+.1f, %+.1f) px, rot %+d deg\n"
             "  %s" % (self.scale, self.offset_x, self.offset_y,
                       self.rotate_cw_deg, self.provenance))
        if self.rms_px is not None:
            s += "\n  fit RMS %.2f px over %d points" % (self.rms_px, self.n_points)
        return s


def fuse(narrow: Sequence[Detection], wide: Sequence[Detection],
         w2n: WideToNarrow, frame_t: float,
         prefer_narrow: bool = True) -> Optional[FusedDetection]:
    """Pick the detection the tracker should consume this frame.

    Narrow wins whenever it has one, at any confidence: it is the camera the
    Jacobian, the goal pixel and the beam all live in, and a mapped wide box
    is a coarser statement about the same object. Wide is the fallback that
    keeps the target alive while it is outside the narrow field -- which is the
    whole point, because that is when the narrow camera has nothing to say.

    Returns None when neither camera sees anything, which the tracker already
    handles as a missed frame.
    """
    if prefer_narrow and narrow:
        best = max(narrow, key=lambda d: d.conf)
        return FusedDetection(best, "narrow", frame_t, mapped=False)
    if wide:
        best = max(wide, key=lambda d: d.conf)
        return FusedDetection(w2n.box(best), "wide->narrow", frame_t, mapped=True)
    if narrow:
        best = max(narrow, key=lambda d: d.conf)
        return FusedDetection(best, "narrow", frame_t, mapped=False)
    return None
