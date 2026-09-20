"""Measure per-motor lash with the payload accelerometer. No camera, no projector.

WHY
---
`calibration/jacobian.json` reports a return residual of 19.27 px on MOTOR B
against 4.44 px on MOTOR A. Under the calibration's approach discipline that
residual is `max(0, L - PRELOAD_STEPS)` microsteps on that motor's own path, so
it implies L_B = 54 and L_A = 21 microsteps -- against the 11.7 microsteps that
`config.BACKLASH_DEG = 0.57` asserts. Those cannot all be true: a both-motor
reversal with L_A = 21 and L_B = 54 would have read 1.83 deg of payload
hysteresis, not 0.57.

The camera route to settling this needs the projector (a static textured scene)
and sole ownership of the narrow camera. Gravity needs neither. The ADXL345
sees the payload directly, so this measures the mechanism instead of a template
match on a projected pattern -- an independent instrument, not a repeat.

METHOD
------
Per motor, two measurements, both self-calibrating:

  1. SENSITIVITY.  Seated in +, move +PROBE (no lash enters a same-direction
     move) and divide the change in the measured (pitch, roll) vector by PROBE.
     That gives k, degrees of accelerometer reading per microstep of THIS motor,
     including whatever mix of payload pitch and yaw this pose happens to show.
     Measuring it removes the need to know that mix.

  2. HYSTERESIS.  Return to one nominal commanded position twice -- once
     arriving in +, once arriving in - -- and difference the two readings.
     Projected onto k, that difference IS the deadband, in microsteps.

Gravity cannot see yaw, and one motor alone moves pitch and yaw together, so
only part of the motion is visible. That is exactly why the sensitivity is
measured rather than derived: the projection cancels in `h / k`.

DEAD-SENSOR GUARD
-----------------
An ADXL345 in standby returns zeros, `tilt_deg()` turns that into -0.00/+0.00,
and every tolerance check downstream reads it as success. Zero variance across
samples is the test, so this refuses to run if the readings do not jitter.

SAFETY
------
Never touches the laser. Every probe is commanded net-zero and the board's step
counters are asserted back at their starting values. Total excursion is
SEAT + PROBE microsteps from the start pose, ~8.8 deg of payload.

USAGE
-----
    python -m tools.lash_imu                 # both motors
    python -m tools.lash_imu --axes tilt     # motor B only
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turret_host import calibrate, calibrate_all, config  # noqa: E402

OUT_DIR = ROOT / "diag"

SEAT_STEPS = 120        # far enough out to seat past any plausible deadband
PROBE_STEPS = 240       # 5.85 deg per payload axis: swamps the IMU noise floor
SETTLE_S = 1.5          # after a move, before sampling

# "gyro  +0.00 +0.00 +0.00 deg/s   tilt  +19.22  -19.85"
_TILT_RE = re.compile(r"tilt\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)")
_GYRO_RE = re.compile(r"gyro\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)")


def _say(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
#   sampling
# ---------------------------------------------------------------------------
def read_tilt(link, seconds: float) -> dict:
    """Stream `imu watch` for `seconds` and return the mean (pitch, roll).

    `imu watch` emits a line every 150 ms until a key arrives, which is the
    only fast read on this firmware -- the plain `imu` command reprints the
    whole describe() block per call and cannot support averaging.
    """
    ser = link.ser
    ser.reset_input_buffer()
    ser.write(b"imu watch\r\n")
    ser.flush()

    deadline = time.monotonic() + seconds
    buf = b""
    tilts: List[Tuple[float, float]] = []
    gyros: List[Tuple[float, float, float]] = []
    while time.monotonic() < deadline:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", "replace")
                mt = _TILT_RE.search(text)
                if mt:
                    tilts.append((float(mt.group(1)), float(mt.group(2))))
                mg = _GYRO_RE.search(text)
                if mg:
                    gyros.append(tuple(float(g) for g in mg.groups()))
        else:
            time.sleep(0.01)

    ser.write(b"\r\n")            # any key stops the watch loop
    ser.flush()
    time.sleep(0.35)
    ser.reset_input_buffer()

    if len(tilts) < 4:
        raise RuntimeError("only %d tilt samples in %.1f s -- is `imu watch` "
                           "streaming?" % (len(tilts), seconds))
    arr = np.array(tilts, dtype=np.float64)
    g = np.array(gyros, dtype=np.float64) if gyros else np.zeros((1, 3))
    return {
        "n": len(tilts),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "sem": arr.std(axis=0) / max(1.0, np.sqrt(len(tilts))),
        "distinct": int(len({tuple(t) for t in tilts})),
        "gyro_abs_max": float(np.abs(g).max()),
    }


def assert_live(sample: dict) -> None:
    """Refuse to measure through a standby ADXL345.

    A part in standby ACKs its address, accepts every register write and
    returns zeros forever. The reading it produces is indistinguishable from
    'perfectly level and perfectly still', so no tolerance check downstream can
    ever catch it. Variance is the only test that can.
    """
    if sample["distinct"] < max(3, sample["n"] // 4):
        raise RuntimeError(
            "accelerometer looks DEAD: only %d distinct values in %d samples "
            "(std %.3f, %.3f). A standby ADXL345 returns zeros and every "
            "tolerance check reads that as success. Check `imu` reports "
            "DEVID 0xE5 and LIVE before trusting anything here."
            % (sample["distinct"], sample["n"], sample["std"][0], sample["std"][1]))


# ---------------------------------------------------------------------------
#   the probe
# ---------------------------------------------------------------------------
def _seat(move, axis: str, direction: int, seat_steps: int) -> None:
    """Return to the current commanded position arriving in `direction`."""
    move(axis, -direction * seat_steps)
    time.sleep(0.05)
    move(axis, +direction * seat_steps)
    time.sleep(SETTLE_S)


def measure_axis(axis: str, link, move, sample_s: float, repeats: int,
                 seat_steps: int, probe_steps: int) -> dict:
    out: Dict[str, object] = {"axis": axis,
                              "motor": "A" if axis == "pan" else "B"}

    # ---- 1. sensitivity, measured with the flanks already loaded in + ----
    _seat(move, axis, +1, seat_steps)
    t0 = read_tilt(link, sample_s)
    assert_live(t0)
    move(axis, +probe_steps)
    time.sleep(SETTLE_S)
    t1 = read_tilt(link, sample_s)

    k = (t1["mean"] - t0["mean"]) / float(probe_steps)
    k_norm = float(np.linalg.norm(k))
    out["sensitivity_deg_per_step"] = k.tolist()
    out["sensitivity_magnitude"] = k_norm
    out["tilt_before"] = t0["mean"].tolist()
    out["tilt_after"] = t1["mean"].tolist()
    out["sample_std"] = [t0["std"].tolist(), t1["std"].tolist()]

    geometric = config.AXIS_STEP_DEG / 2.0
    _say("   sensitivity %s deg/step  |k| %.5f   (one motor alone moves each "
         "payload axis %.5f deg/step)" % (np.round(k, 5), k_norm, geometric))
    _say("   visible fraction of the geometric motion: %.2f"
         % (k_norm / (geometric * np.sqrt(2.0))))

    if k_norm < geometric * 0.15:
        raise RuntimeError(
            "%s moved %d microsteps and the accelerometer barely changed "
            "(|k| %.5f deg/step). Either the motor is not turning, or this "
            "pose hides the motion from gravity -- lash cannot be measured "
            "here." % (axis, probe_steps, k_norm))

    # back to the start, arriving in + so the next stage begins seated
    move(axis, -(probe_steps + seat_steps))
    time.sleep(0.05)
    move(axis, +seat_steps)
    time.sleep(SETTLE_S)

    # ---- 2. hysteresis at one nominal commanded position ----
    k_hat = k / k_norm
    trials = []
    for r in range(repeats):
        _seat(move, axis, +1, seat_steps)
        plus = read_tilt(link, sample_s)
        _seat(move, axis, -1, seat_steps)
        minus = read_tilt(link, sample_s)

        h = plus["mean"] - minus["mean"]
        along = float(h @ k_hat)
        across = float(h[0] * k_hat[1] - h[1] * k_hat[0])
        lash_steps = abs(along) / k_norm
        sem = float(np.linalg.norm(plus["sem"]) + np.linalg.norm(minus["sem"]))
        trials.append({
            "repeat": r,
            "tilt_from_plus": plus["mean"].tolist(),
            "tilt_from_minus": minus["mean"].tolist(),
            "hysteresis_deg": h.tolist(),
            "along_k_deg": along,
            "across_k_deg": across,
            "lash_microsteps": lash_steps,
            "lash_payload_deg": lash_steps * config.AXIS_STEP_DEG,
            "uncertainty_microsteps": sem / k_norm,
        })
        _say("   rep %d: hysteresis %s deg -> %.1f microsteps "
             "(+-%.1f)  = %.2f deg payload-equivalent   [across %.3f deg]"
             % (r, np.round(h, 3), lash_steps, sem / k_norm,
                lash_steps * config.AXIS_STEP_DEG, across))

    vals = np.array([t["lash_microsteps"] for t in trials])
    out["trials"] = trials
    out["lash_microsteps_mean"] = float(vals.mean())
    out["lash_microsteps_std"] = float(vals.std())
    out["lash_payload_deg_mean"] = float(vals.mean() * config.AXIS_STEP_DEG)
    return out


# ---------------------------------------------------------------------------
#   main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--axes", default="pan,tilt")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sample-s", type=float, default=6.0)
    ap.add_argument("--seat", type=int, default=SEAT_STEPS)
    ap.add_argument("--probe", type=int, default=PROBE_STEPS)
    ap.add_argument("--rate", type=int, default=1200)
    args = ap.parse_args(argv)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for a in axes:
        if a not in calibrate.MOTOR_AXES:
            _say("unknown axis %r" % a)
            return 2

    _say("IMU lash probe -- no camera, no projector, laser never touched")
    _say("config says BACKLASH_DEG %.3f = %.1f microsteps per motor path; "
         "PRELOAD_STEPS %d"
         % (config.BACKLASH_DEG, config.BACKLASH_DEG / config.AXIS_STEP_DEG,
            config.PRELOAD_STEPS))
    _say("excursion from the start pose: %d microsteps = %.1f deg per payload axis"
         % (args.seat + args.probe,
            (args.seat + args.probe) * config.AXIS_STEP_DEG / 2.0))

    results: Dict[str, object] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "accelerometer hysteresis, sensitivity measured in-run",
        "seat_steps": args.seat, "probe_steps": args.probe,
        "repeats": args.repeats, "sample_s": args.sample_s,
        "config": {"BACKLASH_DEG": config.BACKLASH_DEG,
                   "PRELOAD_STEPS": config.PRELOAD_STEPS,
                   "AXIS_STEP_DEG": config.AXIS_STEP_DEG},
        "axes": {},
    }

    link = calibrate.CalibrationLink().open()
    try:
        move = calibrate_all._make_mover(link, rate=args.rate)
        start = {ax: int(calibrate_all._state(link)["axes"][ax]["position"])
                 for ax in calibrate.MOTOR_AXES}
        results["start_position"] = start
        _say("start positions %s" % start)

        for axis in axes:
            _say("\n=== %s (MOTOR %s) ===" % (axis.upper(),
                                              "A" if axis == "pan" else "B"))
            results["axes"][axis] = measure_axis(
                axis, link, move, args.sample_s, args.repeats,
                args.seat, args.probe)

        end = {ax: int(calibrate_all._state(link)["axes"][ax]["position"])
               for ax in calibrate.MOTOR_AXES}
        results["end_position"] = end
        if end != start:
            _say("\nWARNING: board position moved %s -> %s" % (start, end))
        else:
            _say("\nboard back at its starting counts %s" % end)
    finally:
        try:
            link.close()
        except Exception:                                      # noqa: BLE001
            pass

    _say("\n=== RESULT ===")
    for axis, r in results["axes"].items():
        _say("%-5s (MOTOR %s)  lash %5.1f +- %.1f microsteps = %.2f deg "
             "payload-equivalent"
             % (axis, r["motor"], r["lash_microsteps_mean"],
                r["lash_microsteps_std"], r["lash_payload_deg_mean"]))
    _say("config BACKLASH_DEG %.3f deg = %.1f microsteps"
         % (config.BACKLASH_DEG, config.BACKLASH_DEG / config.AXIS_STEP_DEG))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ("lash_imu_%s.json" % time.strftime("%Y-%m-%d_%H%M%S"))
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _say("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
