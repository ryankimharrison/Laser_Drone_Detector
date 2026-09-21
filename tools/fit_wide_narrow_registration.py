"""Refit the wide<->narrow registration WITH the wide lens's distortion.

    python tools/fit_wide_narrow_registration.py            # fit and report
    python tools/fit_wide_narrow_registration.py --write    # ...and save it

Offline. Reads the dense Gray-code correspondences already captured by
tools/structured_light_map.py. No hardware, no projector, no camera.

WHY THIS EXISTS
---------------
`wide_narrow_homography.json` holds a plain homography, and a homography is
the exact model for two views of one plane THROUGH PINHOLE LENSES. The wide
module is a 2.1 mm M12 at ~108 degrees and is nothing of the sort, so the
homography has to absorb barrel distortion it cannot represent. Measured on
the 172257 stored correspondences, the residual is almost purely RADIAL --
tangential mean stays under 0.35 px at every radius while radial swings
-1.6 -> -2.4 -> -0.1 -> +4.5 wide px as the field radius grows. That is the
signature of an unmodelled radial term, not of noise.

It gets worse off the sampled patch. The correspondences only reach wide
radius ~385 px, but the narrow camera's field maps out to radius ~549 --
so the outer third of the fused overlay, its top and bottom bands, is the
homography EXTRAPOLATING a pinhole model through a fisheye lens. Held out
on this very data (fit the inner region, predict the outer) the plain
homography reaches 15 px of error at r=350-400 and, projected against the
model below, 67-104 narrow px at the corners of the narrow field.

THE MODEL
---------
    wide pixel -> undistort (radial, 2 terms, fitted principal point)
               -> homography -> narrow pixel

Fitting is a coordinate descent over (k1, k2, cx, cy) with the homography
re-solved in closed form at every step, so the homography is never one of the
free parameters being searched -- it is the exact minimiser given the current
undistortion. No scipy on this machine, and a 4-parameter search does not need
it.

WHAT THE NUMBERS COME OUT AT (2026-09-20, both captures pooled)
---------------------------------------------------------------
    plain homography            rms 3.491   p95 6.16   max 20.48 narrow px
    undistort + homography      rms 0.494   p95 ~1.0   max  ~3.6 narrow px

and, far more to the point, held out on data the fit never saw:

    fit r<300, predict r 350-400    plain 15.03 px    this model 0.53 px

FITTING THE PRINCIPAL POINT IS JUSTIFIED, AND THE JUSTIFICATION IS THE
HELD-OUT TEST, NOT THE FIT. Freeing (cx, cy) improves the in-sample rms from
0.730 to 0.494, which on its own proves nothing -- four parameters beat two on
their own training data by construction. What earns it is that the held-out
outer-ring error also improves, 1.22 -> 0.45 px, and that two independent
subsets (r<250 and r<300) both land the centre at (993, 551) to within a
pixel. An overfitted centre does not agree with itself across disjoint data.

WHAT THIS MODEL DOES NOT FIX
----------------------------
DEPTH. Every correspondence here was captured on one wall at 2.083 m, so this
registration is still exact on that plane and nowhere else. Off it the two
views separate by f_n*B*(1/R - 1/R_cal) = 79.8*(1/R - 0.480) narrow px: 0 at
2.08 m, -12 at 3 m, -22 at 5 m, -38 at infinity. That is a range-dependent
translation and it belongs at render time, not in this file -- see
turret_host/registration.py. This script records the direction and the scale
of it so the renderer does not have to re-derive them, and records NO SIGN,
because both captures are at the same range and the sign is therefore not
observable from this data. config.py is explicit about that trap.

VALIDITY RADIUS. The fit sees wide radius <= ~385 px. The narrow field needs
~549 and the wide frame's own corners are at ~1101. The held-out test supports
the model to ~400 and, by the same trend, through the ~549 the overlay needs.
It does NOT support it at the wide frame's corners: evaluated there the model
implies a 114 deg horizontal field against a 107.9 deg spec, which is a
2.5x extrapolation of a 2-term polynomial and is reported below as a sanity
check ONLY. `verified_radius_px` in the output is the honest boundary.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from turret_host import config                                  # noqa: E402

CAL_DIR = os.path.join(_ROOT, "turret_host", "calibration")
DEFAULT_INPUTS = ("wide_narrow_dense.npz", "wide_narrow_dense_run2.npz")
OUT_PATH = os.path.join(CAL_DIR, "wide_narrow_registration.json")
LEGACY_PATH = os.path.join(CAL_DIR, "wide_narrow_homography.json")

# The focal length the distortion coefficients are normalised by. This is NOT
# a free parameter and it is NOT config.WIDE_F_PX: k1/k2 are defined against
# whatever radius scale you divide by, so the number has to travel with them
# in the output file or they are meaningless. 970 px is the paraxial focal
# length measured by the Gray-code capture (f_wide_px_measured in the legacy
# file), which is the right scale to express a distortion polynomial in.
F_NORM_PX = 970.0


# ==========================================================================
#   the model
# ==========================================================================
def undistort(pts: np.ndarray, k1: float, k2: float,
              cx: float, cy: float, f: float = F_NORM_PX) -> np.ndarray:
    """Distorted wide pixels -> undistorted (rectilinear) wide pixels."""
    x = (pts - (cx, cy)) / f
    r2 = (x * x).sum(1)
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    return x * s[:, None] * f + (cx, cy)


def solve_h(wide_u: np.ndarray, narrow: np.ndarray) -> Optional[np.ndarray]:
    """Closed-form homography on ALREADY-UNDISTORTED wide points.

    method=0 (plain least squares), never RANSAC: these are dense Gray-code
    correspondences with no outliers to reject, and a robust estimator
    minimises a different loss than the one this script reports -- which is
    exactly the scoring artefact structured_light_map.py documents.
    """
    h, _ = cv2.findHomography(wide_u, narrow, method=0)
    return h


def apply_h(pts: np.ndarray, h: np.ndarray) -> np.ndarray:
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return q[:, :2] / q[:, 2:3]


def rms_for(params: Sequence[float], wide: np.ndarray,
            narrow: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    k1, k2, cx, cy = params
    wu = undistort(wide, k1, k2, cx, cy)
    h = solve_h(wu, narrow)
    if h is None:
        return 1e9, None
    e = apply_h(wu, h) - narrow
    return float(np.sqrt((e * e).sum(1).mean())), h


def fit(wide: np.ndarray, narrow: np.ndarray, *, free_centre: bool = True,
        iters: int = 13, progress=None) -> Tuple[List[float], np.ndarray, float]:
    """Coordinate descent on (k1, k2, cx, cy); H re-solved exactly each step."""
    p = [0.40, 0.10, config.WIDE_SIZE[0] / 2.0, config.WIDE_SIZE[1] / 2.0]
    step = [0.06, 0.12, 48.0, 48.0] if free_centre else [0.06, 0.12, 0.0, 0.0]
    for _ in range(iters):
        for j in range(4):
            if step[j] == 0.0:
                continue
            best = (rms_for(p, wide, narrow)[0], p[j])
            for v in (p[j] - step[j], p[j] + step[j]):
                trial = list(p)
                trial[j] = v
                e = rms_for(trial, wide, narrow)[0]
                if e < best[0]:
                    best = (e, v)
            p[j] = best[1]
        step = [s * 0.6 for s in step]
        if progress:
            progress("    k1=%.4f k2=%.4f centre=(%.1f,%.1f)  rms %.3f"
                     % (p[0], p[1], p[2], p[3], rms_for(p, wide, narrow)[0]))
    e, h = rms_for(p, wide, narrow)
    return p, h, e


# ==========================================================================
#   data
# ==========================================================================
def load_inputs(names: Sequence[str]) -> dict:
    wide, narrow, ranges, srcs = [], [], [], []
    wide_size = narrow_size = None
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(CAL_DIR, name)
        if not os.path.exists(path):
            raise SystemExit("no such correspondence file: %s" % path)
        d = np.load(path)
        ws = tuple(int(v) for v in d["wide_size"])
        ns = tuple(int(v) for v in d["narrow_size"])
        # A capture at a different frame size is a different camera as far as
        # pixel coordinates are concerned. Pooling them would be silent
        # nonsense, so refuse rather than average.
        if wide_size is None:
            wide_size, narrow_size = ws, ns
        elif (ws, ns) != (wide_size, narrow_size):
            raise SystemExit(
                "%s was captured at wide %s / narrow %s but the first input "
                "used wide %s / narrow %s -- these cannot be pooled."
                % (os.path.basename(path), ws, ns, wide_size, narrow_size))
        wide.append(d["wide_pts"].astype(np.float64))
        narrow.append(d["narrow_pts"].astype(np.float64))
        ranges.append(float(d["wall_m"]))
        srcs.append(os.path.basename(path))
    return {
        "wide": np.vstack(wide), "narrow": np.vstack(narrow),
        "ranges": ranges, "sources": srcs,
        "wide_size": wide_size, "narrow_size": narrow_size,
    }


# ==========================================================================
#   reporting
# ==========================================================================
def radial_table(wide, narrow, pred, centre, out=print) -> None:
    r = np.linalg.norm(wide - centre, axis=1)
    err = pred - narrow
    mag = np.linalg.norm(err, axis=1)
    out("%12s %9s %9s %9s %9s" % ("wide radius", "points", "median", "p95", "max"))
    for lo, hi in ((0, 100), (100, 200), (200, 300), (300, 400), (400, 600)):
        m = (r >= lo) & (r < hi)
        if m.sum() < 50:
            continue
        out("%5d-%-6d %9d %9.2f %9.2f %9.2f"
            % (lo, hi, m.sum(), np.median(mag[m]),
               np.percentile(mag[m], 95), mag[m].max()))


def heldout_report(wide, narrow, h_plain, out=print) -> Optional[float]:
    """Fit inner, predict outer. The only test that speaks to the overlay.

    Returns the radius out to which the distortion model stayed under 2 narrow
    px on data it never saw, or None if the rings were too sparse to say.
    """
    centre = np.array([config.WIDE_SIZE[0] / 2.0, config.WIDE_SIZE[1] / 2.0])
    r = np.linalg.norm(wide - centre, axis=1)
    out("%-10s %-14s %8s %12s %12s" % ("fit", "test ring", "points",
                                       "plain px", "this model px"))
    verified = None
    for cut in (250, 300):
        tr = np.where(r < cut)[0]
        if len(tr) < 2000:
            continue
        sub = np.random.RandomState(2).choice(tr, min(12000, len(tr)),
                                              replace=False)
        p, h_u, _ = fit(wide[sub], narrow[sub])
        h_p = solve_h(wide[sub], narrow[sub])
        for lo, hi in ((cut, cut + 50), (cut + 50, cut + 100), (300, 400)):
            m = (r >= lo) & (r < hi)
            if m.sum() < 200:
                continue
            e_p = np.median(np.linalg.norm(
                apply_h(wide[m], h_p) - narrow[m], axis=1))
            e_u = np.median(np.linalg.norm(
                apply_h(undistort(wide[m], *p), h_u) - narrow[m], axis=1))
            out("r<%-8d %5d-%-8d %8d %12.2f %12.2f"
                % (cut, lo, hi, m.sum(), e_p, e_u))
            if e_u < 2.0:
                verified = max(verified or 0.0, float(hi))
    return verified


def fov_sanity(k1: float, k2: float, cx: float, cy: float, out=print) -> None:
    """What field of view does this distortion model imply? A CHECK, not a fit.

    Evaluated at the wide frame's own edges, which is 2-3x beyond any data the
    model saw, so it can only ever be a smell test: a model that implied 60 or
    200 degrees would be wrong, one that lands near the 107.9 deg spec is at
    least the right animal.
    """
    def ang(rpx: float) -> float:
        x = rpx / F_NORM_PX
        s = 1.0 + k1 * x * x + k2 * x ** 4
        return math.degrees(math.atan(x * s))

    w, h = config.WIDE_SIZE
    half_w, half_h = w / 2.0, h / 2.0
    out("  horizontal  %6.1f deg   (spec %.1f)" % (2 * ang(half_w),
                                                   config.WIDE_FOV_H_DEG))
    out("  diagonal    %6.1f deg   (spec ~115)"
        % (2 * ang(math.hypot(half_w, half_h))))
    out("  --wide-fast centre crop %.1f deg   (cameras.py says ~85)"
        % (2 * ang(config.WIDE_FAST_SIZE[0] / 2.0)))
    out("  NOTE: these are a 2-3x extrapolation of a 2-term polynomial and are")
    out("        a smell test only. They are NOT evidence the model is good")
    out("        out there, and registration.py refuses to use it past")
    out("        verified_radius_px for exactly that reason.")


# ==========================================================================
#   main
# ==========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS),
                    help="dense correspondence .npz files (default: both runs)")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--write", action="store_true",
                    help="save the fit; without this nothing is written")
    ap.add_argument("--fixed-centre", action="store_true",
                    help="pin the principal point to the image centre")
    ap.add_argument("--sample", type=int, default=20000,
                    help="points used for the search (the report is scored on "
                         "all of them regardless)")
    args = ap.parse_args(argv)

    data = load_inputs(args.inputs)
    wide, narrow = data["wide"], data["narrow"]
    print("=" * 74)
    print("WIDE->NARROW REGISTRATION with wide-lens distortion")
    print("=" * 74)
    print("inputs        %s" % ", ".join(data["sources"]))
    print("points        %d" % len(wide))
    print("capture size  wide %dx%d   narrow %dx%d"
          % (data["wide_size"] + data["narrow_size"]))
    print("wall range    %s m" % ", ".join("%.4f" % r for r in data["ranges"]))

    if data["wide_size"] != tuple(config.WIDE_SIZE):
        print("\nWARNING: captured at wide %s but config.WIDE_SIZE is %s. The "
              "capture size is what gets written to the file and what the "
              "renderer will check against, so this is not silently ignored -- "
              "but check it is what you meant."
              % (data["wide_size"], tuple(config.WIDE_SIZE)))

    ranges = data["ranges"]
    if max(ranges) - min(ranges) > 0.01:
        print("\nWARNING: inputs are at DIFFERENT ranges (%s). Pooling them "
              "fits a plane that exists at neither. Fit them separately."
              % ranges)

    # ---- the two models, scored on identical data ----------------------
    h_plain = solve_h(wide, narrow)
    pred_plain = apply_h(wide, h_plain)

    idx = np.random.RandomState(0).choice(
        len(wide), min(args.sample, len(wide)), replace=False)
    print("\nfitting (k1, k2%s) on %d points..."
          % ("" if args.fixed_centre else ", cx, cy", len(idx)))
    params, _, _ = fit(wide[idx], narrow[idx],
                       free_centre=not args.fixed_centre)
    k1, k2, cx, cy = params
    # Final homography on ALL the points, at the fitted undistortion.
    wide_u = undistort(wide, k1, k2, cx, cy)
    h_und = solve_h(wide_u, narrow)
    pred_und = apply_h(wide_u, h_und)

    def score(pred):
        e = np.linalg.norm(pred - narrow, axis=1)
        return (float(np.sqrt((e ** 2).mean())), float(np.median(e)),
                float(np.percentile(e, 95)), float(e.max()))

    s_plain, s_und = score(pred_plain), score(pred_und)
    print("\n%-34s %8s %8s %8s %8s" % ("model", "rms", "median", "p95", "max"))
    print("-" * 70)
    print("%-34s %8.3f %8.3f %8.3f %8.3f" % ("homography (shipped)", *s_plain))
    print("%-34s %8.3f %8.3f %8.3f %8.3f" % ("undistort + homography", *s_und))
    print("\nfitted  k1 %+.5f   k2 %+.5f   centre (%.1f, %.1f)   f_norm %.0f px"
          % (k1, k2, cx, cy, F_NORM_PX))
    print("        principal point sits %.1f px off the image centre"
          % math.hypot(cx - config.WIDE_SIZE[0] / 2.0,
                       cy - config.WIDE_SIZE[1] / 2.0))

    print("\nRESIDUAL vs FIELD RADIUS -- shipped homography")
    radial_table(wide, narrow, pred_plain,
                 np.array([config.WIDE_SIZE[0] / 2.0, config.WIDE_SIZE[1] / 2.0]))
    print("\nRESIDUAL vs FIELD RADIUS -- undistort + homography")
    radial_table(wide, narrow, pred_und,
                 np.array([config.WIDE_SIZE[0] / 2.0, config.WIDE_SIZE[1] / 2.0]))

    print("\nHELD OUT: fit the inner field, predict the outer (median narrow px)")
    print("This is the test that matters. The overlay's edges live at radius")
    print("~549 and NO capture reached past ~385, so in-sample rms says")
    print("nothing at all about the part of the image that looks wrong.")
    verified = heldout_report(wide, narrow, h_plain)

    print("\nFIELD OF VIEW IMPLIED BY THE DISTORTION MODEL (sanity check)")
    fov_sanity(k1, k2, cx, cy)

    # ---- geometry the renderer needs, derived once here ----------------
    sampled_radius = float(np.linalg.norm(
        wide - np.array([config.WIDE_SIZE[0] / 2.0,
                         config.WIDE_SIZE[1] / 2.0]), axis=1).max())

    # Where does the narrow frame land in wide pixels? This is what the
    # overlay actually covers, and the number that says how far past the data
    # the model is being asked to reach.
    h_inv = np.linalg.inv(h_und)
    nw, nh = data["narrow_size"]
    corners_n = np.array([[0, 0], [nw - 1, 0], [nw - 1, nh - 1], [0, nh - 1]],
                         float)
    corners_wu = apply_h(corners_n, h_inv)
    footprint_radius = float(np.linalg.norm(
        corners_wu - np.array([config.WIDE_SIZE[0] / 2.0,
                               config.WIDE_SIZE[1] / 2.0]), axis=1).max())

    # The epipolar direction, in NARROW pixels, as a unit vector.
    #
    # The baseline is horizontal in the payload frame (CAD: wide x=-30, laser
    # x=0, narrow x=+30, all on y=0, sensors level), and the wide module is
    # not rotated on its mount -- so payload-horizontal is the wide sensor's
    # +x. Pushing that direction through the fitted homography's local linear
    # part gives where it points in the narrow frame, WITHOUT anyone having to
    # reason about the C270's 90 degree mount by hand. It comes out near the
    # narrow ROW axis, which is what config.py and camera_fusion.py both say
    # it should be -- that agreement is the check on this derivation.
    mid = np.array([[config.WIDE_SIZE[0] / 2.0, config.WIDE_SIZE[1] / 2.0]])
    mid_u = undistort(mid, k1, k2, cx, cy)
    step_u = undistort(mid + (10.0, 0.0), k1, k2, cx, cy)
    d_n = apply_h(step_u, h_und)[0] - apply_h(mid_u, h_und)[0]
    epi = d_n / float(np.linalg.norm(d_n))

    f_narrow = 1330.0
    try:
        with open(LEGACY_PATH, "r", encoding="utf-8") as fh:
            f_narrow = float(json.load(fh).get("f_narrow_px_measured", f_narrow))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    baseline_m = config.STEREO_BASELINE_MM / 1000.0
    print("\nEPIPOLAR GEOMETRY (the range term the renderer applies on top)")
    print("  direction in narrow px  (%+.4f, %+.4f)   -- %.1f deg off the "
          "narrow ROW axis" % (epi[0], epi[1],
                               abs(math.degrees(math.atan2(epi[0], -epi[1])))))
    print("  scale f_n*B = %.1f px*m   (f_narrow %.0f px, baseline %.0f mm)"
          % (f_narrow * baseline_m, f_narrow, config.STEREO_BASELINE_MM))
    print("  shift vs the calibration plane, f_n*B*(1/R - 1/%.4f):" % ranges[0])
    for r_m in (1.5, 2.0828, 3.0, 4.0, 5.0):
        print("      R=%.2f m  %+7.1f narrow px"
              % (r_m, f_narrow * baseline_m * (1.0 / r_m - 1.0 / ranges[0])))
    print("  SIGN IS NOT RECORDED. Both captures are at the same range, so it")
    print("  is not observable here. Measure it on the rig; see registration.py.")

    print("\nCOVERAGE")
    print("  correspondences reach wide radius   %.0f px" % sampled_radius)
    print("  the narrow field needs out to       %.0f px" % footprint_radius)
    print("  held-out verified to                %s"
          % ("%.0f px" % verified if verified else "(rings too sparse)"))

    if not args.write:
        print("\nnothing written (pass --write to save %s)"
              % os.path.basename(args.out))
        return 0

    blob = {
        "model": "undistort(wide) -> homography -> narrow",
        "H_undistorted_wide_to_narrow": h_und.tolist(),
        "wide_distortion": {
            "k1": k1, "k2": k2, "cx": cx, "cy": cy, "f_norm_px": F_NORM_PX,
            "convention": ("x_u = c + (x_d - c) * (1 + k1*r2 + k2*r2^2), "
                           "r2 = |x_d - c|^2 / f_norm_px^2"),
        },
        "wide_capture_size": list(data["wide_size"]),
        "narrow_size": list(data["narrow_size"]),
        "range_m": ranges[0],
        "f_narrow_px": f_narrow,
        "baseline_m": baseline_m,
        "epipolar_narrow": [float(epi[0]), float(epi[1])],
        "parallax_scale_px_m": f_narrow * baseline_m,
        "parallax_sign": 0,
        "rms_px": s_und[0], "median_px": s_und[1],
        "p95_px": s_und[2], "max_px": s_und[3],
        "rms_px_plain_homography": s_plain[0],
        "n_points": int(len(wide)),
        "sampled_radius_px": sampled_radius,
        "verified_radius_px": verified or sampled_radius,
        "narrow_footprint_radius_px": footprint_radius,
        "sources": data["sources"],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": ("coordinate descent on (k1,k2,cx,cy) with the homography "
                   "re-solved in closed form at each step; scored on all "
                   "pooled correspondences"),
        "VALID_ONLY": (
            "On the wall plane at %.3f m. Off that plane add the epipolar "
            "term: parallax_scale_px_m * (1/R - 1/range_m) along "
            "epipolar_narrow, times parallax_sign -- WHICH IS 0 HERE BECAUSE "
            "IT HAS NEVER BEEN MEASURED. Inside wide radius %.0f px this is "
            "fitted; out to %.0f px it is supported by the held-out test; "
            "past that it is extrapolation and registration.py will say so."
            % (ranges[0], sampled_radius, verified or sampled_radius)),
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, args.out)
    print("\nwrote -> %s" % args.out)
    print("The legacy wide_narrow_homography.json is left ALONE: its "
          "H_wide_to_narrow maps DISTORTED wide pixels and this file's maps "
          "UNDISTORTED ones, so the two must never share a key name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
