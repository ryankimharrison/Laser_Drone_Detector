"""Per-motor lash probe: is the tilt-axis return residual lash, or something else?

WHY THIS EXISTS
---------------
`calibration/jacobian.json` (2026-09-18 19:27, steps=120, repeats=2,
preload=14) reports

    return_residual_px   pan  4.44   tilt 19.27
    repeat_scatter_px    pan  1.43   tilt  6.86

19.27 px against `config.MAX_ERROR_TO_FIRE_PX = 25` is most of the interlock's
budget on one axis. `calibrate_jacobian` stores only the MAGNITUDE of the worst
residual over the repeats, which cannot distinguish:

    (a) lash larger than PRELOAD_STEPS on the B path   -- static, repeatable,
        along motor B's own pixel direction, and removable by a bigger preload
    (b) the mechanism not having settled yet           -- decays with time
    (c) slip / lost steps                              -- accumulates run over run
    (d) the scene or the camera moving                 -- not along either column

This probe separates them. It measures the residual as a VECTOR, decomposes it
into motor-A/motor-B microsteps through J_inv, sweeps the preload, watches the
residual decay after the move stops, and repeats the cycle to look for drift.

THE MODEL IT TESTS
------------------
With a deadband of L microsteps and the calibration's approach discipline
(every pose reached moving in +, a -d move done as -(d+p) then +p):

    residual = max(0, L - p)  microsteps, along that motor's column of J

So residual falls linearly with p and reaches zero at p = L. p = 0 measures L
directly. If the residual does NOT fall with p, it is not lash.

Firmware note: there is no backlash compensation anywhere in `firmware/current`
(grep: no hits in stepper.py, config.py, cli.py). All lash handling is host-side
approach discipline, so this probe sees the whole of it.

SAFETY
------
This script never touches the laser. It does not import or call anything that
does. Every probe is commanded net-zero, and the board's own position counter is
verified on every move by `calibrate_all._make_mover`; the start position is
asserted at the end.

USAGE
-----
    python -m tools.lash_probe                    # full sweep, both axes
    python -m tools.lash_probe --axes tilt        # one axis
    python -m tools.lash_probe --quick            # fewer points
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turret_host import calibrate, calibrate_all, config, projector  # noqa: E402

OUT_DIR = ROOT / "diag"

# Always re-seat from this far out, so every trial starts with the flanks
# loaded in + regardless of how the previous trial left them. Must exceed any
# plausible lash; 120 microsteps is 5.85 deg of payload, ten times the
# BACKLASH_DEG figure it is meant to swamp.
SEAT_STEPS = 120

# Seconds after the final move at which the feature is sampled. `back` is taken
# at the entry nearest calibrate_all.SETTLE_S so it reproduces what the real
# calibration measures; the rest show whether the residual is decaying.
DECAY_SCHEDULE_S = (0.30, 1.20, 2.50, 5.00, 9.00)


def _say(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
#   measurement
# ---------------------------------------------------------------------------
def _make_measurer(grab, template, p0, avg_frames: int):
    """A measure() that returns the feature position, searching around p0.

    Searching around the ORIGINAL p0 rather than the last reading keeps every
    sample in one coordinate frame and stops a bad match from dragging the
    search window along with it. Travel here is under 90 px against
    SEARCH_HALF_PX = 150, so the peak is always inside the window.
    """
    def measure() -> np.ndarray:
        gray = calibrate._average_gray(grab, avg_frames)
        return np.array(calibrate.measure_feature(gray, template, p0),
                        dtype=np.float64)
    return measure


def _static_noise_floor(measure, seconds: float, period: float = 1.0) -> dict:
    """Sample a motionless scene, so every later number has a floor to beat.

    This is the control for (d): if the scene or the camera is drifting, it
    shows up here, in pixels, before any motor has moved.
    """
    t0 = time.monotonic()
    samples: List[Tuple[float, np.ndarray]] = []
    while time.monotonic() - t0 < seconds:
        samples.append((time.monotonic() - t0, measure()))
        time.sleep(period)
    pts = np.vstack([p for _, p in samples])
    drift = pts[-1] - pts[0]
    return {
        "n": len(samples),
        "seconds": round(time.monotonic() - t0, 2),
        "std_px": [float(pts[:, 0].std()), float(pts[:, 1].std())],
        "peak_to_peak_px": [float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))],
        "drift_px": [float(drift[0]), float(drift[1])],
        "drift_magnitude_px": float(np.hypot(*drift)),
        "samples": [[round(t, 2), float(p[0]), float(p[1])] for t, p in samples],
    }


# ---------------------------------------------------------------------------
#   the probe
# ---------------------------------------------------------------------------
def _seat(move, axis: str, settle_s: float) -> None:
    """Leave the axis with its flanks loaded in +, from a known long way out."""
    move(axis, -SEAT_STEPS)
    time.sleep(0.05)
    move(axis, +SEAT_STEPS)
    time.sleep(settle_s)


def probe_once(axis: str, steps: int, preload: int, move, measure,
               settle_s: float, decay: bool = False) -> dict:
    """One out-and-back at a given preload. Returns residual as a VECTOR.

    Identical in structure to `calibrate.calibrate_jacobian`'s inner loop, so
    the numbers are comparable to the ones in jacobian.json -- except that the
    preload is a parameter and the residual keeps its direction.
    """
    _seat(move, axis, settle_s)
    base = measure()

    move(axis, +steps)
    time.sleep(settle_s)
    moved = measure()

    # Come back the long way round so the final motion is still +.
    move(axis, -(steps + preload))
    if preload:
        time.sleep(0.05)
        move(axis, +preload)
    t_done = time.monotonic()

    series: List[Tuple[float, np.ndarray]] = []
    schedule = DECAY_SCHEDULE_S if decay else (settle_s,)
    for target in schedule:
        remaining = target - (time.monotonic() - t_done)
        if remaining > 0:
            time.sleep(remaining)
        series.append((time.monotonic() - t_done, measure()))

    # `back` is the sample taken closest to the real calibration's settle time.
    back = min(series, key=lambda s: abs(s[0] - settle_s))[1]

    shift = moved - base
    residual = back - base
    return {
        "axis": axis,
        "steps": steps,
        "preload": preload,
        "base_px": base.tolist(),
        "moved_px": moved.tolist(),
        "back_px": back.tolist(),
        "shift_px": shift.tolist(),
        "shift_magnitude_px": float(np.hypot(*shift)),
        "residual_px": residual.tolist(),
        "residual_magnitude_px": float(np.hypot(*residual)),
        "decay_series": [[round(t, 2), float(p[0]) - float(base[0]),
                          float(p[1]) - float(base[1])] for t, p in series],
    }


# ---------------------------------------------------------------------------
#   interpretation
# ---------------------------------------------------------------------------
def decompose(residual_px, J_inv: Optional[np.ndarray]) -> Optional[dict]:
    """Residual expressed in the motor steps that would have produced it.

    A tilt probe whose residual is lash in the B path must land almost entirely
    in `motor_b_steps`. A large `motor_a_steps` on a tilt probe means the
    residual did not come from the axis that moved -- scene, camera, or a
    coupling the model does not have.
    """
    if J_inv is None:
        return None
    d = J_inv @ np.asarray(residual_px, dtype=np.float64)
    total = abs(d[0]) + abs(d[1])
    return {
        "motor_a_steps": float(d[0]),
        "motor_b_steps": float(d[1]),
        "off_axis_fraction": float(abs(d[0]) / total) if total > 1e-9 else 0.0,
    }


def _column_projection(residual_px, column: np.ndarray) -> dict:
    """Residual resolved along and across the axis's own pixel direction."""
    c = np.asarray(column, dtype=np.float64)
    n = float(np.linalg.norm(c))
    if n < 1e-12:
        return {"along_px": 0.0, "across_px": 0.0, "along_steps": 0.0}
    unit = c / n
    r = np.asarray(residual_px, dtype=np.float64)
    along = float(r @ unit)
    across = float(r[0] * unit[1] - r[1] * unit[0])
    return {"along_px": along, "across_px": across, "along_steps": along / n}


# ---------------------------------------------------------------------------
#   main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--axes", default="pan,tilt")
    ap.add_argument("--steps", type=int, default=calibrate_all.JACOBIAN_STEPS)
    ap.add_argument("--preloads", default="0,14,28,56",
                    help="preload sweep, microsteps")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--settle", type=float, default=calibrate_all.SETTLE_S)
    ap.add_argument("--rate", type=int, default=1200)
    ap.add_argument("--noise-floor-s", type=float, default=20.0)
    ap.add_argument("--quick", action="store_true",
                    help="preloads 0,14,56 and 2 repeats")
    args = ap.parse_args(argv)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    preloads = [int(p) for p in args.preloads.split(",") if p.strip()]
    repeats = args.repeats
    if args.quick:
        preloads = [0, 14, 56]
        repeats = 2

    for a in axes:
        if a not in calibrate.MOTOR_AXES:
            _say("unknown axis %r; expected one of %s" % (a, calibrate.MOTOR_AXES))
            return 2

    prior = calibrate.load_calibration("jacobian")
    J = np.asarray(prior["J"], dtype=np.float64) if prior else None
    J_inv = np.linalg.inv(J) if J is not None else None
    columns = {"pan": J[:, 0], "tilt": J[:, 1]} if J is not None else {}
    if prior:
        _say("prior J from %s: columns |A| %.4f  |B| %.4f px/step"
             % (prior.get("saved_at"), np.linalg.norm(J[:, 0]),
                np.linalg.norm(J[:, 1])))
        _say("  its residuals: pan %.2f  tilt %.2f px"
             % (prior["return_residual_px"]["pan"],
                prior["return_residual_px"]["tilt"]))

    _say("\nlash probe: axes %s  steps %d  preloads %s  repeats %d  settle %.2f s"
         % (",".join(axes), args.steps, preloads, repeats, args.settle))
    _say("PRELOAD_STEPS in config is %d; BACKLASH_DEG %.3f = %.1f microsteps"
         % (config.PRELOAD_STEPS, config.BACKLASH_DEG,
            config.BACKLASH_DEG / config.AXIS_STEP_DEG))

    results: Dict[str, object] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": args.steps, "preloads": preloads, "repeats": repeats,
        "settle_s": args.settle, "rate": args.rate,
        "seat_steps": SEAT_STEPS,
        "config": {"PRELOAD_STEPS": config.PRELOAD_STEPS,
                   "BACKLASH_DEG": config.BACKLASH_DEG,
                   "AXIS_STEP_DEG": config.AXIS_STEP_DEG},
        "prior_jacobian": prior,
        "trials": [],
    }

    surface = projector.ProjectorSurface()
    _say("projector: %s" % surface.monitor.describe())
    _say("\nopening narrow camera...")
    cam = calibrate_all._open_narrow()
    link = None
    try:
        with surface:
            surface.show_and_settle(surface.white(), 0.8)
            pattern = surface.noise(cell=calibrate_all.NOISE_CELL_PX, seed=1)
            surface.show_and_settle(pattern, 0.6)
            grab = calibrate_all._scene_grab(cam, surface, pattern)

            p0, bbox = calibrate_all._pick_feature_inside_projection(
                cam, surface, pattern, travel_px=0.7 * args.steps)
            if p0 is None:
                _say("no feature inside the projection; falling back to frame")
                ref = calibrate._average_gray(grab, calibrate_all.AVG_FRAMES)
                p0 = calibrate.auto_pick_feature(
                    ref, margin=calibrate.TEMPLATE_HALF_PX + 60)
            _say("feature at (%.1f, %.1f)  projection bbox %s" % (p0[0], p0[1], bbox))

            _say("holding %.1f s for the AGC" % calibrate_all.AGC_SETTLE_S)
            surface.show_and_settle(pattern, calibrate_all.AGC_SETTLE_S)

            ref = calibrate._average_gray(grab, calibrate_all.AVG_FRAMES)
            template = calibrate._extract_template(ref, p0, calibrate.TEMPLATE_HALF_PX)
            measure = _make_measurer(grab, template, p0, calibrate_all.AVG_FRAMES)

            _say("\n=== STATIC NOISE FLOOR (%.0f s, nothing moves) ===" % args.noise_floor_s)
            floor = _static_noise_floor(measure, args.noise_floor_s)
            results["noise_floor"] = floor
            _say("   std %.3f, %.3f px   drift %.2f px over %.0f s"
                 % (floor["std_px"][0], floor["std_px"][1],
                    floor["drift_magnitude_px"], floor["seconds"]))

            _say("\nopening board...")
            link = calibrate.CalibrationLink().open()
            move = calibrate_all._make_mover(link, rate=args.rate)
            start_pos = {ax: int(calibrate_all._state(link)["axes"][ax]["position"])
                         for ax in calibrate.MOTOR_AXES}
            pitch, yaw = calibrate_all._board_pose(link)
            _say("   board pose: pitch %+.3f  yaw %+.3f   positions %s"
                 % (pitch, yaw, start_pos))
            results["start_position"] = start_pos
            results["start_pose"] = [pitch, yaw]

            for axis in axes:
                _say("\n=== %s (MOTOR %s) ===" % (axis.upper(),
                                                  "A" if axis == "pan" else "B"))
                for preload in preloads:
                    for k in range(repeats):
                        # Only the first repeat of each preload pays for the
                        # long decay series; the rest are for scatter.
                        t = probe_once(axis, args.steps, preload, move, measure,
                                       args.settle, decay=(k == 0))
                        t["repeat"] = k
                        t["decompose"] = decompose(t["residual_px"], J_inv)
                        if axis in columns:
                            t["projection"] = _column_projection(
                                t["residual_px"], columns[axis])
                        results["trials"].append(t)

                        msg = ("   preload %3d  rep %d   shift %6.2f px   "
                               "residual %6.2f px" %
                               (preload, k, t["shift_magnitude_px"],
                                t["residual_magnitude_px"]))
                        if t.get("projection"):
                            msg += "  (%+.1f steps along axis)" % \
                                   t["projection"]["along_steps"]
                        if t.get("decompose"):
                            msg += "  [A %+.1f  B %+.1f]" % \
                                   (t["decompose"]["motor_a_steps"],
                                    t["decompose"]["motor_b_steps"])
                        _say(msg)
                        if k == 0:
                            _say("      decay " + "  ".join(
                                "%.1fs:%.2f" % (s[0], math.hypot(s[1], s[2]))
                                for s in t["decay_series"]))

            end_pos = {ax: int(calibrate_all._state(link)["axes"][ax]["position"])
                       for ax in calibrate.MOTOR_AXES}
            results["end_position"] = end_pos
            if end_pos != start_pos:
                _say("\nWARNING: board position moved %s -> %s" % (start_pos, end_pos))
            else:
                _say("\nboard back at its starting counts %s" % end_pos)
    finally:
        if link is not None:
            try:
                link.close()
            except Exception:                                  # noqa: BLE001
                pass
        cam.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ("lash_probe_%s.json" % time.strftime("%Y-%m-%d_%H%M%S"))
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _say("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
