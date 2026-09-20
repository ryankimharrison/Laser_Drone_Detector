"""Fit the wide->narrow mapping from a projected ChArUco board.

    python tools/calibrate_wide_to_narrow.py --self-test
    python tools/calibrate_wide_to_narrow.py --distance-in 96
    python tools/calibrate_wide_to_narrow.py --distance-in 96 --board-width-in 42

WHY A PROJECTED BOARD IS LEGITIMATE HERE
----------------------------------------
turret_host/projector.py argues against projecting a calibration target, and
it is right -- for INTRINSICS. Keystone makes what lands on the wall an unknown
projective warp of what was sent, so the "known target geometry" Zhang's method
needs is not known; and a wall-welded pattern cannot be tilted to condition a
focal-length fit.

NEITHER OBJECTION APPLIES TO AN INTER-CAMERA FIT. This measures a mapping
between two views of the SAME physical thing. Whatever warped quadrilateral
lands on the wall, both cameras look at that same warped quadrilateral, and the
warp cancels in the correspondence. The board's true geometry is never used --
only that a corner with a given ID is the same physical point in both images.

ChArUco rather than a chessboard because its corners carry IDs. Matching two
views of a plain chessboard means solving the correspondence yourself, and the
narrow camera's ~29-60 deg field sees a different SUBSET of the board than the
wide camera's 108 deg -- so the two views do not even contain the same corners.
IDs make a partial overlap a non-issue.

WHAT IT DOES NOT FIX
--------------------
The wide lens is not rectilinear (config: "At 115 deg diagonal this lens is NOT
rectilinear -- expect to need cv2.fisheye"). A similarity transform cannot model
barrel distortion, so the residual will grow off-axis and the reported RMS is
the honest bound on that. This replaces a mapping that was never measured at
all; it does not make the mapping exact.

--board-width-in is optional and separate: with the distance to the wall it
gives a real focal length per camera by similar triangles, which is the one
number `config.NARROW_F_PX = 1400` is currently ESTIMATING. Measure the
projected board's white border-to-border width on the wall with a tape.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# turret_host FIRST: its __init__ sets the MSMF env var, and that only works
# before cv2 is imported.
from turret_host import config                            # noqa: E402
from turret_host.camera_fusion import WideToNarrow        # noqa: E402

import cv2                                                # noqa: E402
import numpy as np                                        # noqa: E402

SQUARES_X, SQUARES_Y = 9, 6
DICT = cv2.aruco.DICT_5X5_100


def build_board():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    # Square/marker lengths are in arbitrary units: nothing here uses metric
    # board geometry, only corner IDs and their pixel positions.
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), 1.0, 0.75, d)
    return d, board


def board_image(board, w: int, h: int, margin_frac: float = 0.06) -> np.ndarray:
    m = int(min(w, h) * margin_frac)
    img = board.generateImage((max(16, w - 2 * m), max(16, h - 2 * m)),
                              marginSize=0)
    out = np.zeros((h, w), np.uint8)
    out[:] = 255
    ih, iw = img.shape[:2]
    y0, x0 = (h - ih) // 2, (w - iw) // 2
    out[y0:y0 + ih, x0:x0 + iw] = img
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def detect(img, board, dictionary):
    """-> {corner_id: (x, y)} in RAW image pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    det = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _m, _mi = det.detectBoard(gray)
    out = {}
    if ch_ids is None or ch_corners is None:
        return out
    for cid, pt in zip(ch_ids.flatten(), ch_corners.reshape(-1, 2)):
        out[int(cid)] = (float(pt[0]), float(pt[1]))
    return out


# ==========================================================================
#   self-test: does the fit recover a known transform?
# ==========================================================================
def self_test() -> int:
    print("SELF-TEST: fit a KNOWN wide->narrow transform from synthetic pairs\n")
    truth = WideToNarrow(scale=1.58, offset_x=-23.0, offset_y=+11.0,
                         rotate_cw_deg=90, provenance="synthetic truth")
    rng = np.random.default_rng(5)
    bad = 0
    for noise_px in (0.0, 0.5, 2.0):
        pairs = []
        for _ in range(40):
            wx = float(rng.uniform(200, config.WIDE_SIZE[0] - 200))
            wy = float(rng.uniform(150, config.WIDE_SIZE[1] - 150))
            nx, ny = truth.point(wx, wy)
            nx += float(rng.normal(0, noise_px))
            ny += float(rng.normal(0, noise_px))
            pairs.append(((wx, wy), (nx, ny)))
        fit = WideToNarrow.measure(pairs)
        ds = abs(fit.scale - truth.scale)
        dx = abs(fit.offset_x - truth.offset_x)
        dy = abs(fit.offset_y - truth.offset_y)
        ok = ds < 0.02 and dx < 3.0 + noise_px and dy < 3.0 + noise_px
        bad += 0 if ok else 1
        print("  noise %.1f px -> scale %.4f (truth %.4f, d=%.4f)  "
              "offset (%+.1f, %+.1f) (truth %+.1f, %+.1f)  rms %.2f  [%s]"
              % (noise_px, fit.scale, truth.scale, ds,
                 fit.offset_x, fit.offset_y, truth.offset_x, truth.offset_y,
                 fit.rms_px, "OK" if ok else "FAIL"))

    # A fit from points clustered in one corner must not be trusted -- report
    # that it is ill-conditioned rather than quietly returning a number.
    pairs = []
    for _ in range(20):
        wx = float(rng.uniform(100, 260))
        wy = float(rng.uniform(100, 220))
        nx, ny = truth.point(wx, wy)
        pairs.append(((wx, wy), (nx + rng.normal(0, 1.0),
                                 ny + rng.normal(0, 1.0))))
    clustered = WideToNarrow.measure(pairs)
    print("\n  clustered points -> scale %.4f (truth %.4f): %s"
          % (clustered.scale, truth.scale,
             "spread matters -- this is why coverage is reported below"))
    print("\n%s" % ("SELF-TEST PASSED" if bad == 0 else "SELF-TEST FAILED"))
    return 1 if bad else 0


# ==========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="verify the fit against a synthetic transform, no hardware")
    p.add_argument("--distance-in", type=float, default=None,
                   help="inches from the CAMERA FACE to the projected wall")
    p.add_argument("--board-width-in", type=float, default=None,
                   help="measured width of the projected board on the wall, "
                        "inches. With --distance-in this gives a real focal "
                        "length per camera.")
    p.add_argument("--shots", type=int, default=5,
                   help="frames captured per pose")
    p.add_argument("--sweep", action="store_true",
                   help="POOL CORRESPONDENCES OVER SEVERAL POSES. Both cameras "
                        "are bolted to one payload, so slewing the turret "
                        "moves the board across BOTH sensors -- which is the "
                        "only way to separate scale from offset. A single pose "
                        "covers ~13%% of the wide frame and gives a local fit "
                        "of a lens that is not rectilinear.")
    p.add_argument("--sweep-pitch", type=float, default=5.0,
                   help="degrees of pitch either side of centre")
    p.add_argument("--sweep-yaw", type=float, default=7.0,
                   help="degrees of yaw either side of centre")
    p.add_argument("--min-pairs", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="fit and report, but do not write the json")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    from turret_host import cameras, projector

    mons = projector.enumerate_monitors()
    proj = projector.pick_projector(mons)
    print("monitors: %s" % mons)
    if proj is None:
        print("\nNo second display found. Connect the projector, extend (not "
              "mirror) the desktop, and re-run.\n"
              "Mirrored would put the board on the laptop panel, and the "
              "cameras would see a wall with nothing on it.")
        return 2
    print("projector: %s" % (proj,))

    dictionary, board = build_board()

    print("\nopening cameras...")
    ident = cameras.identify_cameras()
    for name, idx, expo, gain in (("narrow", ident.narrow_index, -6, 64),
                                  ("wide", ident.wide_index, -6, None)):
        cameras.lock_exposure(idx, expo, gain=gain)
    narrow = cameras.narrow_thread(ident.narrow_index).start()
    wide = cameras.wide_thread(ident.wide_index).start()
    print("  narrow %.1f fps, wide %.1f fps"
          % (narrow.startup_fps, wide.startup_fps))

    # Poses to visit, as (dpitch, dyaw) RELATIVE to where it starts. The
    # centre pose is first and last so any drift shows up as a disagreement
    # between two fits of the same place.
    dp, dy = args.sweep_pitch, args.sweep_yaw
    poses = [(0.0, 0.0)]
    if args.sweep:
        poses = [(0.0, 0.0), (0.0, -dy), (0.0, +dy), (-dp, 0.0), (+dp, 0.0),
                 (-dp, -dy), (-dp, +dy), (+dp, -dy), (+dp, +dy), (0.0, 0.0)]

    mover = None
    read_tilt = None
    moves: List[Tuple[float, float, float]] = []   # (cmd_pitch, meas_dpitch, roll)
    if args.sweep:
        sys.path.insert(0, str(_ROOT / "tools"))
        import re as _re
        import send as sendmod                             # noqa: E402

        _IMUF = _re.compile(r"IMUF\s+\S+\s+\S+\s+\S+\s+([-+\d.]+)\s+([-+\d.]+)")

        def read_tilt():                                   # noqa: F811
            """(pitch, roll) from gravity, or None. `imu fast` is the cheap
            form -- the verbose one runs a 16-sample liveness check."""
            try:
                out = sendmod.send(["imu fast"], timeout=6, quiet=True)
            except Exception:                              # noqa: BLE001
                return None
            m = _IMUF.search(out[0][1]) if out else None
            return (float(m.group(1)), float(m.group(2))) if m else None

        def mover(dpitch, dyaw):                           # noqa: F811
            """Move, and CHECK IT AGAINST GRAVITY.

            The step counter cannot tell you the payload moved -- it counts
            steps it emitted, which is exactly what stays right while a motor
            skips against a stop. The accelerometer is the only witness that
            the mechanism did what it was told.

            PITCH ONLY. Gravity is unchanged by rotation about the vertical
            axis, so yaw is unobservable here and is not checked. That is a
            property of gravity, not a gap in the test.
            """
            before = read_tilt()
            sendmod.send(["dmove %.3f %.3f" % (dpitch, dyaw)],
                         timeout=25, quiet=True)
            time.sleep(0.5)                 # stop ringing before measuring
            after = read_tilt()
            if before is None or after is None:
                print("      (no IMU reading -- move unverified)")
                return
            meas = after[0] - before[0]
            moves.append((dpitch, meas, after[1]))
            if abs(dpitch) < 0.05:
                print("      gravity: pitch %+.2f -> %+.2f (yaw-only move)"
                      % (before[0], after[0]))
                return
            ratio = meas / dpitch
            flag = "" if 0.75 <= ratio <= 1.25 else "   <-- OFF"
            print("      gravity: commanded %+.1f, measured %+.2f deg "
                  "(ratio %.2f)%s" % (dpitch, meas, ratio, flag))

    pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    n_seen = w_seen = 0
    at = (0.0, 0.0)
    try:
        with projector.ProjectorSurface(proj) as surface:
            img = board_image(board, surface.monitor.width, surface.monitor.height)
            surface.show(img)
            time.sleep(1.2)                 # let exposure settle on the board

            for pi, pose in enumerate(poses):
                if mover is not None:
                    step = (pose[0] - at[0], pose[1] - at[1])
                    if abs(step[0]) > 1e-6 or abs(step[1]) > 1e-6:
                        print("  pose %d/%d: dmove %+.1f pitch %+.1f yaw"
                              % (pi + 1, len(poses), step[0], step[1]))
                        mover(step[0], step[1])
                        at = pose
                        time.sleep(0.6)     # let the platform stop ringing
                for shot in range(args.shots):
                    time.sleep(0.25)
                    nf, _ = narrow.slot.get()
                    wf, _ = wide.slot.get()
                    if nf is None or wf is None:
                        continue
                    nd = detect(nf.image, board, dictionary)
                    wd = detect(wf.image, board, dictionary)
                    n_seen = max(n_seen, len(nd))
                    w_seen = max(w_seen, len(wd))
                    common = sorted(set(nd) & set(wd))
                    print("    shot %d: narrow %2d, wide %2d, common %2d"
                          % (shot + 1, len(nd), len(wd), len(common)))
                    for cid in common:
                        pairs.append((wd[cid], nd[cid]))
    finally:
        # Put it back where it started, whatever happened above.
        if mover is not None and (abs(at[0]) > 1e-6 or abs(at[1]) > 1e-6):
            print("  returning to the starting pose")
            try:
                mover(-at[0], -at[1])
            except Exception as exc:                       # noqa: BLE001
                print("  WARNING: could not return to start: %s" % exc)
        narrow.stop()
        wide.stop()

    if moves:
        pitched = [m for m in moves if abs(m[0]) >= 0.05]
        print("\n" + "=" * 68)
        print("MOTION CHECKED AGAINST GRAVITY  (pitch only -- gravity cannot "
              "see yaw)")
        print("=" * 68)
        if pitched:
            ratios = [m[1] / m[0] for m in pitched]
            resid = [m[1] - m[0] for m in pitched]
            print("  %d pitch moves" % len(pitched))
            print("  commanded vs measured ratio: mean %.3f  min %.3f  max %.3f"
                  % (float(np.mean(ratios)), min(ratios), max(ratios)))
            print("  residual deg:                mean %+.2f  worst %+.2f"
                  % (float(np.mean(resid)),
                     max(resid, key=abs)))
            # JUDGE THE SCATTER, NOT THE MEAN.
            #
            # A mean ratio of 1.0 says nothing on its own: lash gives back on
            # the next move in the same direction exactly what it swallowed on
            # a reversal, so a set of moves that individually range 0.50..1.41
            # averages to 0.97 and reads as perfect. The mean is the one
            # statistic blind to the failure this test exists to find.
            mr = float(np.mean(ratios))
            spread = max(ratios) - min(ratios)
            worst = max(resid, key=abs)
            if spread > 0.4:
                print("  -> INCONSISTENT. Mean ratio %.2f looks fine, but "
                      "individual moves range %.2f..%.2f and the worst move "
                      "missed by %+.2f deg." % (mr, min(ratios), max(ratios),
                                                worst))
                print("     That pattern is BACKLASH or LOST STEPS, not a gain "
                      "error: a reversal swallows travel and the next move in "
                      "the same direction hands it back, so the mean cancels "
                      "out while no single move is right.")
                print("     config.BACKLASH_DEG = %.2f; a worst residual of "
                      "%+.2f deg is %.1fx that."
                      % (config.BACKLASH_DEG, worst,
                         abs(worst) / max(1e-6, config.BACKLASH_DEG)))
            elif mr < 0.75:
                print("  -> the payload moved LESS than commanded: lost steps, "
                      "belt slip, or a stop.")
            elif mr > 1.25:
                print("  -> the payload moved MORE than commanded: check "
                      "DIFFERENTIAL_N and the microstep setting.")
            else:
                print("  -> consistent AND correctly scaled. The step counter "
                      "agreeing with gravity is the only way to know a motor "
                      "did not skip.")
        rolls = [m[2] for m in moves]
        print("  roll over the sweep: %.2f .. %.2f deg (should barely move -- "
              "these are pitch and yaw commands)" % (min(rolls), max(rolls)))

    print("\ncorrespondences: %d (narrow saw up to %d corners, wide %d)"
          % (len(pairs), n_seen, w_seen))
    if len(pairs) < args.min_pairs:
        print("\nNot enough. Things that cause this:")
        print("  * the narrow camera's field is small -- aim the turret at the")
        print("    board's CENTRE so its view lands inside the board")
        print("  * board out of focus on the wall, or the room is too bright")
        print("  * projector mirrored rather than extended")
        return 3

    fit = WideToNarrow.measure(pairs)
    derived = WideToNarrow.derived()

    wx = [p[0][0] for p in pairs]
    wy = [p[0][1] for p in pairs]
    cover = ((max(wx) - min(wx)) / config.WIDE_SIZE[0],
             (max(wy) - min(wy)) / config.WIDE_SIZE[1])

    print("\n" + "=" * 68)
    print("FIT")
    print("=" * 68)
    print("  scale      %.4f      (derived guess was %.4f, %+.1f%%)"
          % (fit.scale, derived.scale,
             100.0 * (fit.scale - derived.scale) / derived.scale))
    print("  offset     (%+.1f, %+.1f) px   (derived assumed 0, 0)"
          % (fit.offset_x, fit.offset_y))
    print("  rotate_cw  %+d deg (not fitted -- it is how the camera is bolted on)"
          % fit.rotate_cw_deg)
    print("  rms        %.2f px over %d points" % (fit.rms_px, fit.n_points))
    print("  coverage   %.0f%% x %.0f%% of the wide frame" % (100 * cover[0],
                                                              100 * cover[1]))
    if min(cover) < 0.25:
        print("  WARNING: points span a small part of the frame, so scale and")
        print("           offset are poorly separated. Move the board or the")
        print("           turret and re-run for a fit you can trust.")
    if fit.rms_px > 8.0:
        print("  WARNING: rms is high. Expected off-axis -- the wide lens is")
        print("           not rectilinear and this transform cannot model it.")

    off = math.hypot(fit.offset_x, fit.offset_y)
    print("\n  what this changes: the derived mapping was off by %.0f px of"
          % off)
    print("  boresight alone, against config.GATE_PX = %.0f." % config.GATE_PX)

    if args.distance_in and args.board_width_in:
        d_m = args.distance_in * 0.0254
        wreal = args.board_width_in * 0.0254
        span_w = max(wx) - min(wx)
        nx = [p[1][0] for p in pairs]
        ny = [p[1][1] for p in pairs]
        span_n = max(max(nx) - min(nx), max(ny) - min(ny))
        # Corners span only part of the board; scale the measured physical
        # width by the same fraction the detected corners cover.
        frac = span_w / config.WIDE_SIZE[0]
        print("\n  focal length, from %0.1f in to the wall and a %0.1f in board:"
              % (args.distance_in, args.board_width_in))
        print("    NOTE: this uses the DETECTED corner span, so it assumes the")
        print("    board fills the frame fraction shown above. Treat as a")
        print("    sanity check on NARROW_F_PX, not a replacement for a")
        print("    chessboard intrinsics run.")
        if frac > 0.05:
            f_wide = span_w * d_m / (wreal * frac)
            f_narrow = span_n * d_m / (wreal * frac)
            print("    wide   f_px ~ %.0f   (config.WIDE_F_PX  = %.0f)"
                  % (f_wide, config.WIDE_F_PX))
            print("    narrow f_px ~ %.0f   (config.NARROW_F_PX = %.0f, ESTIMATE)"
                  % (f_narrow, config.NARROW_F_PX))

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    fit.save(saved_at=stamp)
    print("\nwritten to turret_host/calibration/wide_to_narrow.json")
    print("WideToNarrow.load() will now report 'measured %s'." % stamp)
    if args.distance_in:
        print("Fit taken at %.1f in (%.2f m). The mapping is range-dependent "
              "through parallax; parallax_sign stays 0 until someone measures "
              "its direction." % (args.distance_in, args.distance_in * 0.0254))
    return 0


if __name__ == "__main__":
    sys.exit(main())
