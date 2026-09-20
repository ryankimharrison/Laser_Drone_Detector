"""Did the motors actually go where they were told? Rewind and look.

    python tools/step_check.py --capture           # before the run
    ... fly the session ...
    python tools/step_check.py --verify            # after: rewind and measure

THE IDEA, AND WHY IT IS THE RIGHT ONE
--------------------------------------
Everything downstream of the Jacobian assumes a commanded step becomes a
degree of payload. Nothing in the system can see the one failure that breaks
that assumption: an A4988 raises no fault when a stepper skips, and the step
counter keeps counting, so a lost step is permanent, silent, and cumulative.

So: note where the board says it is, run the session, then command EXACTLY the
negative of the microsteps it emitted. If nothing slipped, the payload comes
back to the pose it started in. Whatever it is off by IS the accumulated loss.

Two readouts, because they fail differently:

  CAMERA   phase-correlate the before and after frames. The camera rides the
           payload, so if the pose returns the image is identical and any
           residual shift is the error -- in BOTH axes, at ~0.1 px, which at
           24 px/deg is ~0.004 deg. Needs a static, textured scene.

  GRAVITY  the angle between the before and after gravity vectors. Coarser
           (~0.22 deg after averaging) and BLIND TO YAW, because at level,
           yawing does not move gravity at all. But it is absolute: it is the
           only one of the two that notices if the whole rig was bumped, which
           the camera would happily report as a step loss.

Disagreement between them is informative, not a problem. Camera says moved,
gravity says level -> a yaw-axis loss, or the scene changed. Both say moved ->
a pitch-axis loss. Gravity says moved, camera says still -> somebody moved the
tripod.

WHY THE BOARD'S COUNTERS AND NOT THE HOST'S
--------------------------------------------
`app._integrate_pose` is MEASURED 2.2x wrong (71.45 deg actual vs 32.21
believed): it only runs on frames that issue a command, while the Pico holds
the last rate between them, and it applies a new rate over a dt that already
elapsed. The board's `state` reports the microsteps it actually emitted per
motor. That is the number dead reckoning is built on, so that is the number to
test.

REVERSAL IS PER MOTOR, NOT PER PAYLOAD AXIS
--------------------------------------------
On a differential, pitch is the SUM of the two motor positions and yaw is
their difference, so `dmove` cannot undo an arbitrary pair exactly. Each motor
is walked back by its own delta with `move <axis> <-steps>`.

BACKLASH IS NOT STEP LOSS. The measured 0.57 deg of lash means the last
direction of travel matters: arriving at the datum from the other side leaves
a real offset that no step was lost to. `--verify` therefore approaches from
the same direction it left, with an overshoot-and-return, unless --no-lash.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import turret_host  # noqa: F401,E402  (MSMF env var, before cv2)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from turret_host import calibrate, cameras, config  # noqa: E402

STATE_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "diag", "step_check.json")
REF_FRAME = STATE_PATH.replace(".json", "_ref.png")

_IMUF = re.compile(r"IMUF\s+\S+\s+\S+\s+\S+\s+([-+\d.]+)\s+([-+\d.]+)")

#: Extra microsteps to overshoot by before coming back, so the datum is
#: approached from the same side it was left from. Backlash here is a MEASURED
#: 0.57 deg; at AXIS_STEP_DEG this is comfortably more than that.
LASH_STEPS = 40

#: Settle before believing the accelerometer. An 11 deg move rings, and a
#: sample taken during the ring measures the ring. `level` uses the same.
SETTLE_S = 0.40

#: Averaged accelerometer samples per reading. Single-sample noise is a
#: MEASURED 0.76 deg; 12 brings the mean to ~0.22 deg.
TILT_SAMPLES = 12


def gravity_unit(pitch_deg: float, roll_deg: float):
    """Unit gravity in the IMU frame. Same construction as app._gravity_unit.

    Compared as a VECTOR, never as two angles: the IMU turns with the payload,
    so a given physical tilt lands in pitch and roll in a ratio set by yaw.
    Differencing the angles makes the answer depend on where yaw happened to
    be, which is exactly the mistake that made `level` drive the payload into
    its own frame.
    """
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    return np.array([-math.sin(p),
                     math.sin(r) * math.cos(p),
                     math.cos(r) * math.cos(p)])


def angle_between(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb),
                                                -1.0, 1.0))))


def read_tilt(link, samples: int = TILT_SAMPLES):
    """(pitch, roll) averaged. Returns None if the IMU is not answering.

    None means NO MEASUREMENT, never "level". A standby-latched ADXL345 reads
    exactly -0.00/+0.00, which is a perfectly plausible answer and is how a
    dead sensor previously passed as success.
    """
    ps, rs = [], []
    for _ in range(samples):
        m = _IMUF.search(link.command("imu fast", timeout_s=2.0) or "")
        if m:
            ps.append(float(m.group(1)))
            rs.append(float(m.group(2)))
        time.sleep(0.01)
    if len(ps) < max(3, samples // 3):
        return None
    return float(np.median(ps)), float(np.median(rs))


def read_positions(link):
    st = link.state()
    ax = st["axes"]
    return {k: int(ax[k]["position"]) for k in ("pan", "tilt")}


def grab_frame(index):
    cam = calibrate.CalibrationCapture(index, size=config.NARROW_SIZE,
                                       name="narrow").open()
    try:
        for _ in range(12):
            cam.read()                      # let exposure settle
        return cv2.cvtColor(cam.read(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    finally:
        cam.close()


def image_shift(ref: np.ndarray, cur: np.ndarray):
    """(dx, dy, response) between two greyscale frames.

    Hanning-windowed: phaseCorrelate assumes periodicity, and without the
    window the frame edges act as a huge step discontinuity that dominates the
    correlation. `response` is the peak confidence -- a blank wall gives a low
    value and the shift from it means nothing, so it is reported alongside.
    """
    if ref.shape != cur.shape:
        return None
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref, cur, win)
    return float(dx), float(dy), float(resp)


def _open(port):
    return (calibrate.CalibrationLink(port=port) if port
            else calibrate.CalibrationLink()).open()


def capture(port, use_camera: bool) -> int:
    link = _open(port)
    try:
        pos = read_positions(link)
        tilt = read_tilt(link)
    finally:
        link.close()

    rec = {"positions": pos, "tilt": tilt,
           "axis_step_deg": config.AXIS_STEP_DEG,
           "wall": time.strftime("%Y-%m-%d %H:%M:%S")}
    if use_camera:
        ident = cameras.identify_cameras()
        cameras.lock_exposure(ident.narrow_index, exposure=-6)
        ref = grab_frame(ident.narrow_index)
        cv2.imwrite(REF_FRAME, ref.astype(np.uint8))
        rec["ref_frame"] = REF_FRAME
        rec["narrow_index"] = ident.narrow_index

    _os.makedirs(_os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as fh:
        json.dump(rec, fh, indent=2)

    print("datum captured")
    print("  pan %+d   tilt %+d   microsteps" % (pos["pan"], pos["tilt"]))
    print("  tilt %s" % ("pitch %+.2f roll %+.2f" % tilt if tilt
                         else "IMU DID NOT ANSWER -- gravity check unavailable"))
    print("  frame %s" % (REF_FRAME if use_camera else "(skipped)"))
    print("\nrun the session, then: python tools/step_check.py --verify")
    return 0


def verify(port, use_camera: bool, lash: bool, rate: int) -> int:
    if not _os.path.exists(STATE_PATH):
        print("no datum. Run --capture BEFORE the session.")
        return 1
    ref_rec = json.load(open(STATE_PATH))
    p0 = ref_rec["positions"]

    link = _open(port)
    try:
        p1 = read_positions(link)
        delta = {k: p1[k] - p0[k] for k in p0}
        print("net microsteps emitted this session:")
        for k in ("pan", "tilt"):
            print("  %-5s %+7d  (%.2f deg of motor)"
                  % (k, delta[k], delta[k] * config.AXIS_STEP_DEG))
        if all(v == 0 for v in delta.values()):
            print("\nnothing moved; nothing to check.")
            return 0

        print("\nrewinding...")
        for k in ("pan", "tilt"):
            if delta[k] == 0:
                continue
            back = -delta[k]
            if lash:
                # Approach the datum from the same side it was left from, so
                # the 0.57 deg of measured backlash is taken up in the same
                # direction as the outbound travel and does not read as loss.
                over = LASH_STEPS if back > 0 else -LASH_STEPS
                link.command("move %s %d %d" % (k, back + over, rate),
                             timeout_s=60.0)
                time.sleep(0.15)
                link.command("move %s %d %d" % (k, -over, rate), timeout_s=60.0)
            else:
                link.command("move %s %d %d" % (k, back, rate), timeout_s=60.0)

        time.sleep(SETTLE_S)
        p2 = read_positions(link)
        tilt2 = read_tilt(link)
    finally:
        link.close()

    resid = {k: p2[k] - p0[k] for k in p0}
    print("\ncounter residual (should be 0 -- this only checks the ARITHMETIC,")
    print("not the mechanism):  pan %+d  tilt %+d" % (resid["pan"], resid["tilt"]))

    print("\n%s" % ("=" * 64))
    print("DID THE PAYLOAD COME BACK?")
    verdict = []

    t0 = ref_rec.get("tilt")
    if t0 and tilt2:
        g = angle_between(gravity_unit(*t0), gravity_unit(*tilt2))
        steps = g / config.AXIS_STEP_DEG
        print("  GRAVITY  %.2f deg from the datum  (~%.0f microsteps)" % (g, steps))
        print("           resolution ~0.22 deg; BLIND TO YAW")
        verdict.append(("gravity", g, 0.30))
    else:
        print("  GRAVITY  unavailable (no IMU reading at one end)")

    if use_camera and ref_rec.get("ref_frame") \
            and _os.path.exists(ref_rec["ref_frame"]):
        ident_idx = ref_rec.get("narrow_index")
        cameras.lock_exposure(ident_idx, exposure=-6)
        ref = cv2.imread(ref_rec["ref_frame"], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        cur = grab_frame(ident_idx)
        sh = image_shift(ref, cur)
        if sh is None:
            print("  CAMERA   frame size changed; cannot compare")
        else:
            dx, dy, resp = sh
            px = math.hypot(dx, dy)
            deg = px / (config.NARROW_F_PX * math.pi / 180.0)
            print("  CAMERA   %.1f px  = %.3f deg  (~%.0f microsteps)  [dx %+.1f dy %+.1f]"
                  % (px, deg, deg / config.AXIS_STEP_DEG, dx, dy))
            print("           correlation peak %.3f %s" % (
                resp, "" if resp > 0.05
                else "-- TOO LOW, the scene is blank or changed; ignore this"))
            if resp > 0.05:
                verdict.append(("camera", deg, 0.10))

    print("\n%s" % ("-" * 64))
    if not verdict:
        print("NO USABLE READOUT. Both sensors declined to answer -- this is")
        print("not a pass. Check the IMU, or point the camera at texture.")
        return 1
    worst = max(verdict, key=lambda v: v[1] / v[2])
    if all(v < tol for _n, v, tol in verdict):
        print("PASS -- the payload returned. No net step loss above the noise")
        print("floor of the readouts above.")
    else:
        print("STEP LOSS: %s reads %.2f deg off (tolerance %.2f)."
              % (worst[0], worst[1], worst[2]))
        print("That is ~%.0f microsteps the board believes it emitted and the"
              % (worst[1] / config.AXIS_STEP_DEG))
        print("payload did not make. Dead reckoning is wrong by that much, and")
        print("it accumulates -- it does not wash out.")
    if len(verdict) == 2:
        g = dict((n, v) for n, v, _t in verdict)
        if abs(g["camera"] - g["gravity"]) > 0.5:
            print("\nNOTE: the two disagree (camera %.2f, gravity %.2f)."
                  % (g["camera"], g["gravity"]))
            print("Camera-only -> a YAW-axis loss, which gravity cannot see, or")
            print("the scene moved. Gravity-only -> the whole rig was bumped.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--port", default=None)
    ap.add_argument("--no-camera", action="store_true",
                    help="gravity only -- leaves the camera free for another "
                         "session, at the cost of the yaw axis")
    ap.add_argument("--no-lash", action="store_true",
                    help="rewind directly instead of overshoot-and-return; "
                         "the 0.57 deg of backlash will then read as loss")
    ap.add_argument("--rate", type=int, default=800, help="microsteps/s")
    a = ap.parse_args(argv)

    use_cam = not a.no_camera
    if a.capture:
        return capture(a.port, use_cam)
    return verify(a.port, use_cam, not a.no_lash, a.rate)


if __name__ == "__main__":
    raise SystemExit(main())
