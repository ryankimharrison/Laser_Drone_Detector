"""Wide <-> narrow image registration, distortion and range aware.

    from turret_host.registration import WideNarrowRegistration
    reg = WideNarrowRegistration.load()          # None if never fitted
    mapx, mapy, inside = reg.warp_maps(display_size, wide_frame.shape, range_m)
    narrow_on_display = cv2.remap(narrow, mapx, mapy, cv2.INTER_LINEAR)

WHAT THIS IS FOR
----------------
The fused pane draws both cameras in one frame so a physical point lands in
one place. Everything below exists because three different things stop that
from happening, and they fail in three different-looking ways:

  1. THE WIDE LENS IS NOT A PINHOLE. A homography is the exact model for two
     views of one plane through rectilinear lenses; the wide module is a
     2.1 mm M12 at ~108 degrees. Fitting a homography to it leaves a purely
     RADIAL residual -- flat-ish in the middle, growing hard at the edges --
     which reads on screen as an overlay that is sharp at centre and splits
     into a double image toward the top and bottom. Measured against the
     shipped homography: 3 narrow px at centre, 13 at mid-field, 67-104 at
     the corners of the narrow field. `undistort()` removes it, and after it
     the residual is FLAT across radius (0.41/0.40/0.42/0.43 px median by
     ring), which is the real evidence the model is right -- not the rms.

  2. THE CAMERAS ARE 60 mm APART. The registration was measured on one wall
     at 2.083 m and a homography is exact for one plane only. Off it the two
     views separate along the epipolar direction by
     `parallax_scale_px_m * (1/R - 1/range_m)` -- 0 at 2.08 m, -12 narrow px
     at 3 m, -22 at 5 m, -38 at infinity. That reads as a uniform sideways
     shift that CHANGES WITH TARGET DISTANCE, quite unlike (1). Correcting it
     needs a range, which the tracker already has.

     THE SIGN IS NOT IN THE CALIBRATION FILE AND MUST NOT BE GUESSED. Both
     Gray-code captures are at the same range, so the sign is not observable
     from them; `parallax_sign` therefore defaults to 0, meaning the
     correction is OFF. config.py is emphatic about this and it has already
     cost this project one bug: "A wrong sign does not halve the error, it
     doubles it." See MEASURING THE SIGN below -- it takes about a minute.

  3. THE TWO FRAMES ARE FROM DIFFERENT MOMENTS. Not this module's problem,
     but worth knowing while reading an overlay: nothing here can fix a
     composite of two different instants, and while the turret slews that
     term is larger than both of the above. The renderer times it instead.

MEASURING THE SIGN
------------------
Put something at a range CLEARLY different from 2.08 m -- 4-5 m is ideal,
the correction is 22 px there and 0 at the calibration plane where you can
learn nothing. Turn the fused pane on and cycle the sign. One of -1/+1 closes
the double image and the other visibly doubles it; that is the whole test.
Write the winner to `config.FUSED_PARALLAX_SIGN`. Do NOT derive it from the
CAD numbers instead: the narrow camera is mounted 90 degrees over, so the
chain from "wide is at x=-30" to "which way the narrow row moves" has two
sign flips in it and being right about both on paper is not the same as
having checked.

THE FRAME SIZE TRAP -- READ BEFORE CHANGING `warp_maps`
--------------------------------------------------------
`--wide-fast` opens the wide camera at 1280x720, which cameras.wide_thread
documents as a CENTER CROP of the 1920x1080 readout, not a downscale. Any
code that converts wide pixels using `config.WIDE_SIZE` instead of the size
of the frame IN ITS HAND is then wrong by (320, 180) wide px -- about 450
narrow px, which is not a subtle misregistration, it is a completely
different part of the room. camera_fusion.point() documents this same trap
and still has it.

So nothing in this module reads `config.WIDE_SIZE` to scale a frame. Every
entry point takes the ACTUAL frame size and routes it through
`frame_offset()`, which returns the mapping to capture coordinates and
refuses -- returns None -- for any size it cannot account for. A caller that
ignores a None gets no overlay, which is the correct outcome: an overlay that
looks registered and is off by a third of a field is worse than no overlay.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from turret_host import config

CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "calibration", "wide_narrow_registration.json")

# How far the range may drift before the warp maps are rebuilt, metres. The
# maps are a few ms to build and the parallax term moves by well under a
# pixel over this at any range in RANGE_LIMITS_M, so rebuilding more often
# than this buys nothing at all.
_RANGE_QUANTUM_M = 0.05


@dataclass(frozen=True)
class FrameMapping:
    """How a delivered wide frame relates to the one the fit was measured on.

    `offset` is added to a frame pixel to get a capture-coordinate pixel.
    `assumed` is True when that offset comes from the documented centre-crop
    behaviour rather than from the frame matching the capture size -- the
    caller is expected to SAY SO on screen, because the crop being exactly
    centred has never been verified on this rig.
    """
    offset: Tuple[float, float]
    assumed: bool
    note: str


@dataclass(frozen=True)
class WideNarrowRegistration:
    h: np.ndarray                       # undistorted wide -> narrow
    k1: float
    k2: float
    cx: float
    cy: float
    f_norm_px: float
    wide_capture_size: Tuple[int, int]
    narrow_size: Tuple[int, int]
    range_m: float
    parallax_scale_px_m: float
    epipolar_narrow: Tuple[float, float]
    parallax_sign: int
    rms_px: float
    n_points: int
    verified_radius_px: float
    narrow_footprint_radius_px: float
    source: str

    # -- construction ----------------------------------------------------
    @staticmethod
    def load(path: str = CAL_PATH,
             parallax_sign: Optional[int] = None
             ) -> Optional["WideNarrowRegistration"]:
        """Read the fitted registration, or None if it has never been fitted.

        None is a normal outcome, not an error: a machine that has not run
        tools/fit_wide_narrow_registration.py still has to bring the panel up.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                b = json.load(fh)
            d = b["wide_distortion"]
            sign = (parallax_sign if parallax_sign is not None
                    else int(getattr(config, "FUSED_PARALLAX_SIGN",
                                     b.get("parallax_sign", 0))))
            epi = b["epipolar_narrow"]
            return WideNarrowRegistration(
                h=np.array(b["H_undistorted_wide_to_narrow"], float),
                k1=float(d["k1"]), k2=float(d["k2"]),
                cx=float(d["cx"]), cy=float(d["cy"]),
                f_norm_px=float(d["f_norm_px"]),
                wide_capture_size=tuple(int(v) for v in b["wide_capture_size"]),
                narrow_size=tuple(int(v) for v in b["narrow_size"]),
                range_m=float(b["range_m"]),
                parallax_scale_px_m=float(b["parallax_scale_px_m"]),
                epipolar_narrow=(float(epi[0]), float(epi[1])),
                parallax_sign=max(-1, min(1, sign)),
                rms_px=float(b.get("rms_px", 0.0)),
                n_points=int(b.get("n_points", 0)),
                verified_radius_px=float(b.get("verified_radius_px", 0.0)),
                narrow_footprint_radius_px=float(
                    b.get("narrow_footprint_radius_px", 0.0)),
                source=os.path.basename(path),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def with_sign(self, sign: int) -> "WideNarrowRegistration":
        """A copy with a different parallax sign, for cycling it live."""
        return WideNarrowRegistration(**{**self.__dict__,
                                         "parallax_sign": max(-1, min(1, int(sign)))})

    # -- the frame size trap ---------------------------------------------
    def frame_offset(self, frame_size: Tuple[int, int]) -> Optional[FrameMapping]:
        """Map a DELIVERED wide frame onto the capture coordinates.

        Returns None when the frame cannot be accounted for, and a caller that
        gets None must not draw an overlay. See THE FRAME SIZE TRAP above.
        """
        fw, fh = int(frame_size[0]), int(frame_size[1])
        cw, ch = self.wide_capture_size
        if (fw, fh) == (cw, ch):
            return FrameMapping((0.0, 0.0), False, "")

        fast = tuple(int(v) for v in config.WIDE_FAST_SIZE)
        if (fw, fh) == fast and (cw, ch) == tuple(int(v) for v in config.WIDE_SIZE):
            # The ONLY other size wide_thread can deliver, and it documents it
            # as a centre crop (~108 -> ~85 deg horizontal), not a downscale.
            # A downscale would need a SCALE here and a crop needs an OFFSET;
            # applying the wrong one of those is the 450-narrow-px error.
            return FrameMapping(((cw - fw) / 2.0, (ch - fh) / 2.0), True,
                                "--wide-fast centre crop assumed (unverified)")
        return None

    # -- distortion ------------------------------------------------------
    def undistort(self, pts: np.ndarray) -> np.ndarray:
        """Distorted wide pixels (capture coordinates) -> rectilinear."""
        p = np.asarray(pts, float).reshape(-1, 2)
        x = (p - (self.cx, self.cy)) / self.f_norm_px
        r2 = (x * x).sum(1)
        s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return x * s[:, None] * self.f_norm_px + (self.cx, self.cy)

    def distort(self, pts: np.ndarray, iters: int = 12) -> np.ndarray:
        """The inverse of `undistort`: rectilinear wide pixels -> distorted.

        NEWTON ON THE RADIUS, not the obvious fixed point. Undistortion is
        radial, so inverting it is a scalar root find along each ray:

            f(r) = r + k1*r^3 + k2*r^5 - r_undistorted = 0

        The tempting `r <- r_u / s(r)` iteration is a contraction only while
        the distortion is mild, and this lens's is not -- at the wide frame's
        corners k1*r^2 reaches 0.59 and that loop leaves 184 px of error.
        Newton converges there in a handful of steps instead, because f is
        strictly increasing (f' = 1 + 3*k1*r^2 + 5*k2*r^4 > 0 for these
        positive coefficients) so the root is unique and approached
        monotonically from r_u.

        This runs on the handful of overlay points per frame that go the
        narrow->display way (a beam marker, a box, the field outline), never
        on a whole image; whole images go through `warp_maps`, which only
        needs the forward direction.
        """
        p = np.asarray(pts, float).reshape(-1, 2)
        d = (p - (self.cx, self.cy)) / self.f_norm_px
        ru = np.linalg.norm(d, axis=1)
        r = ru.copy()
        for _ in range(iters):
            r2 = r * r
            f = r * (1.0 + self.k1 * r2 + self.k2 * r2 * r2) - ru
            fp = 1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r2 * r2
            r = r - f / np.where(np.abs(fp) < 1e-12, 1e-12, fp)
        # Scale along the ray. At the exact centre ru is 0 and the direction
        # is undefined, so leave that point where it is.
        scale = np.where(ru > 1e-12, r / np.where(ru > 1e-12, ru, 1.0), 1.0)
        return d * scale[:, None] * self.f_norm_px + (self.cx, self.cy)

    # -- the range term --------------------------------------------------
    def parallax_shift_px(self, range_m: Optional[float]) -> float:
        """Signed narrow-pixel shift to add on top of the plane homography.

        Zero whenever the sign has not been measured, which is the default --
        so this is a no-op until somebody does the one-minute test in the
        module docstring, and it is a no-op in a way that leaves the overlay
        exactly as good as it was rather than randomly worse.
        """
        if not self.parallax_sign or range_m is None:
            return 0.0
        r = float(range_m)
        lo, hi = config.RANGE_LIMITS_M
        r = max(lo, min(hi, r))
        return (self.parallax_sign * self.parallax_scale_px_m
                * (1.0 / r - 1.0 / self.range_m))

    # -- point mapping ---------------------------------------------------
    def wide_to_narrow(self, pts: np.ndarray, frame_size: Tuple[int, int],
                       range_m: Optional[float] = None) -> Optional[np.ndarray]:
        """Wide pixels IN THE DELIVERED FRAME -> narrow pixels."""
        fm = self.frame_offset(frame_size)
        if fm is None:
            return None
        p = np.asarray(pts, float).reshape(-1, 2) + fm.offset
        q = self.undistort(p)
        q = np.hstack([q, np.ones((len(q), 1))]) @ self.h.T
        out = q[:, :2] / q[:, 2:3]
        shift = self.parallax_shift_px(range_m)
        if shift:
            out = out + np.array(self.epipolar_narrow) * shift
        return out

    def narrow_to_wide(self, pts: np.ndarray, frame_size: Tuple[int, int],
                       range_m: Optional[float] = None) -> Optional[np.ndarray]:
        """Narrow pixels -> wide pixels IN THE DELIVERED FRAME."""
        fm = self.frame_offset(frame_size)
        if fm is None:
            return None
        p = np.asarray(pts, float).reshape(-1, 2)
        shift = self.parallax_shift_px(range_m)
        if shift:
            p = p - np.array(self.epipolar_narrow) * shift
        q = np.hstack([p, np.ones((len(p), 1))]) @ np.linalg.inv(self.h).T
        q = q[:, :2] / q[:, 2:3]
        return self.distort(q) - fm.offset

    def narrow_outline(self, frame_size: Tuple[int, int],
                       range_m: Optional[float] = None,
                       per_edge: int = 24) -> Optional[np.ndarray]:
        """The narrow field's border, in delivered-wide pixels.

        Sampled along each edge rather than taken as four corners: once the
        lens distortion is in the model this boundary is CURVED, and drawing
        it as a quadrilateral would hide the very bow the model exists to
        correct.
        """
        nw, nh = self.narrow_size
        t = np.linspace(0.0, 1.0, per_edge, endpoint=False)
        edges = [
            np.c_[t * (nw - 1), np.zeros_like(t)],
            np.c_[np.full_like(t, nw - 1), t * (nh - 1)],
            np.c_[(1 - t) * (nw - 1), np.full_like(t, nh - 1)],
            np.c_[np.zeros_like(t), (1 - t) * (nh - 1)],
        ]
        return self.narrow_to_wide(np.vstack(edges), frame_size, range_m)

    # -- the warp --------------------------------------------------------
    def warp_maps(self, display_size: Tuple[int, int],
                  frame_size: Tuple[int, int],
                  range_m: Optional[float] = None
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Remap tables that paint the narrow frame into the wide display.

        Returns (map_x, map_y, inside) for cv2.remap, all display-sized.
        `inside` is the boolean coverage mask -- computed from the maps
        themselves rather than by warping a second all-white image, which is
        both exact and one fewer full-frame resample per frame.

        A remap rather than warpPerspective because the undistortion is not
        projective, so no 3x3 can express this chain. That is not a cost: the
        tables only change when the display size, the frame size or the range
        changes, and `Warper` below caches them.
        """
        fm = self.frame_offset(frame_size)
        if fm is None:
            return None
        dw, dh = int(display_size[0]), int(display_size[1])
        fw, fh = float(frame_size[0]), float(frame_size[1])

        # Display is the delivered frame scaled to the pane -- so the scale
        # comes from the FRAME, never from config.WIDE_SIZE. See the trap.
        u = (np.arange(dw, dtype=np.float64) + 0.5) * (fw / dw) - 0.5
        v = (np.arange(dh, dtype=np.float64) + 0.5) * (fh / dh) - 0.5
        uu, vv = np.meshgrid(u, v)
        pts = np.stack([uu.ravel(), vv.ravel()], axis=1) + fm.offset

        q = self.undistort(pts)
        q = np.hstack([q, np.ones((len(q), 1))]) @ self.h.T
        w = q[:, 2:3]
        # Points behind the narrow camera project through a sign flip and
        # would otherwise alias onto valid-looking pixels. Push them out of
        # range instead of letting them paint.
        bad = (np.abs(w[:, 0]) < 1e-9) | (w[:, 0] < 0)
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        nrw = q[:, :2] / w

        shift = self.parallax_shift_px(range_m)
        if shift:
            nrw = nrw + np.array(self.epipolar_narrow) * shift

        nw, nh = self.narrow_size
        mx = nrw[:, 0].reshape(dh, dw)
        my = nrw[:, 1].reshape(dh, dw)
        inside = ((mx >= 0) & (mx <= nw - 1) & (my >= 0) & (my <= nh - 1)
                  & ~bad.reshape(dh, dw))
        return (mx.astype(np.float32), my.astype(np.float32), inside)

    # -- reporting -------------------------------------------------------
    def describe(self) -> str:
        sign = {0: "OFF (sign never measured)", 1: "+1", -1: "-1"}[self.parallax_sign]
        return ("wide->narrow: undistort(k1=%+.4f k2=%+.4f c=%.0f,%.0f) + "
                "homography\n  rms %.2f narrow px on the plane at %.2f m, "
                "%d points\n  verified to wide radius %.0f px; the narrow "
                "field reaches %.0f\n  parallax %s, scale %.1f px*m"
                % (self.k1, self.k2, self.cx, self.cy, self.rms_px,
                   self.range_m, self.n_points, self.verified_radius_px,
                   self.narrow_footprint_radius_px, sign,
                   self.parallax_scale_px_m))

    def extrapolating(self) -> bool:
        """Does the overlay reach past what the held-out test supports?"""
        return self.narrow_footprint_radius_px > self.verified_radius_px + 1.0


class Warper:
    """Caches `warp_maps` so the render loop rebuilds them only when needed."""

    def __init__(self, reg: WideNarrowRegistration) -> None:
        self.reg = reg
        self._key: Optional[tuple] = None
        self._maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def maps(self, display_size, frame_size, range_m):
        rq = (None if range_m is None
              else round(float(range_m) / _RANGE_QUANTUM_M))
        # The sign is in the key because cycling it live must rebuild.
        key = (tuple(display_size), tuple(frame_size), rq,
               self.reg.parallax_sign)
        if key != self._key:
            self._maps = self.reg.warp_maps(display_size, frame_size, range_m)
            self._key = key
        return self._maps

    def invalidate(self) -> None:
        self._key = None


# ==========================================================================
#   SELF TEST -- geometry only, no camera, no board
# ==========================================================================
def _main() -> int:
    reg = WideNarrowRegistration.load()
    if reg is None:
        print("no %s -- run tools/fit_wide_narrow_registration.py --write"
              % os.path.basename(CAL_PATH))
        return 1
    print(reg.describe())

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print("  %-46s %s%s" % (name, "ok" if cond else "FAIL",
                                ("  " + detail) if detail else ""))

    print("\ndistort/undistort round trip (the overlay direction):")
    cw, ch = reg.wide_capture_size
    g = np.array([[x, y] for x in np.linspace(0, cw - 1, 9)
                  for y in np.linspace(0, ch - 1, 9)], float)
    back = reg.distort(reg.undistort(g))
    err = np.linalg.norm(back - g, axis=1).max()
    check("round trip over the whole wide frame", err < 0.01,
          "max %.2e px" % err)

    print("\nwide <-> narrow round trip:")
    full = (cw, ch)
    n = reg.wide_to_narrow(g, full)
    w2 = reg.narrow_to_wide(n, full)
    err = np.linalg.norm(w2 - g, axis=1).max()
    check("wide -> narrow -> wide", err < 0.01, "max %.2e px" % err)

    print("\nTHE FRAME SIZE TRAP:")
    fm = reg.frame_offset(full)
    check("capture size maps with no offset",
          fm is not None and fm.offset == (0.0, 0.0) and not fm.assumed)
    fast = tuple(int(v) for v in config.WIDE_FAST_SIZE)
    fm = reg.frame_offset(fast)
    want = ((cw - fast[0]) / 2.0, (ch - fast[1]) / 2.0)
    check("--wide-fast maps as a centre crop",
          fm is not None and fm.offset == want and fm.assumed,
          "offset %s" % (fm.offset if fm else None,))
    # The whole point: the fast frame's CENTRE is the capture frame's centre,
    # so it must land on the same narrow pixel, not 450 px away.
    c_full = reg.wide_to_narrow([[cw / 2.0, ch / 2.0]], full)[0]
    c_fast = reg.wide_to_narrow([[fast[0] / 2.0, fast[1] / 2.0]], fast)[0]
    check("both frames' centres hit one narrow pixel",
          float(np.linalg.norm(c_full - c_fast)) < 0.01,
          "%.3f px apart" % float(np.linalg.norm(c_full - c_fast)))
    # And the failure it replaces, for scale.
    naive = reg.wide_to_narrow([[fast[0] / 2.0, fast[1] / 2.0]], full)[0]
    print("      (ignoring the crop would have put it %.0f narrow px away)"
          % float(np.linalg.norm(c_full - naive)))
    for bogus in ((640, 480), (1920, 1200), (800, 600)):
        check("refuses an unaccountable %dx%d frame" % bogus,
              reg.frame_offset(bogus) is None)

    print("\nwarp maps:")
    maps = reg.warp_maps((640, 360), full, None)
    check("built at 640x360", maps is not None)
    mx, my, inside = maps
    check("coverage is a sane fraction of the pane",
          0.10 < inside.mean() < 0.60, "%.1f%% covered" % (100 * inside.mean()))
    check("refuses to build for an unaccountable frame",
          reg.warp_maps((640, 360), (640, 480), None) is None)

    print("\nparallax term:")
    check("off by default", reg.parallax_shift_px(5.0) == 0.0
          if reg.parallax_sign == 0 else True)
    signed = reg.with_sign(1)
    check("zero at the calibration plane",
          abs(signed.parallax_shift_px(reg.range_m)) < 1e-9)
    for r_m in (1.5, 3.0, 5.0):
        print("      R=%.1f m -> %+6.1f narrow px"
              % (r_m, signed.parallax_shift_px(r_m)))
    check("clamped outside RANGE_LIMITS_M",
          signed.parallax_shift_px(100.0)
          == signed.parallax_shift_px(config.RANGE_LIMITS_M[1]))
    check("the two signs are opposite",
          abs(signed.parallax_shift_px(5.0)
              + reg.with_sign(-1).parallax_shift_px(5.0)) < 1e-9)

    print("\nepipolar direction:")
    ex, ey = reg.epipolar_narrow
    # CAD says the baseline is horizontal in the payload frame and the C270 is
    # mounted 90 deg over, so the shift must land on the narrow ROW. That is
    # asserted in config.py and camera_fusion.py from the mechanics; this
    # comes from the fitted homography instead, so agreement is a real check.
    check("lands on the narrow row axis, as config.py says", abs(ex) < 0.10,
          "(%+.4f, %+.4f)" % (ex, ey))

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOMETHING FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
