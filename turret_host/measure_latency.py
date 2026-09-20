"""Measure LATENCY_S: flash the laser, see which frame the dot lands in.

    python -m turret_host.measure_latency               # 16 flashes
    python -m turret_host.measure_latency --trials 30

WHY THE LASER AND NOT THE MOTOR
-------------------------------
A motor probe was written first and thrown away, for two reasons.

Backlash here is a MEASURED 0.57 deg. Command a motor and the first steps take
up the lash while the payload -- and therefore the image -- does not move at
all. That dead time is mechanical slop, but a command-to-image-motion test
counts it as latency and inflates the answer by however loose the gearing
happens to be that day.

And the only non-blocking writer is `link.TurretLink`, which owns the port
during tracking; `CalibrationLink.move_motor()` waits for the whole reply
before returning, so it cannot be timed at all.

The laser has nothing mechanical in its path. MOSFET, diode, photons, sensor.
It is a genuine step change, so the onset is sharp rather than a ramp whose
start depends on the detection threshold.

AND NOT THE PROJECTOR
---------------------
A projector carries 30-60 ms of its own input lag -- the same order as the
quantity being measured, and not separable from it without a second
instrument. It was never the light source here. It was used only for what it
is genuinely good for: a BLACK field, which kills the room's reflected light
and is what makes a 5 mW dot stand out. `projector.py` argues the same point
for the goal-pixel fit.

`--wall` drops it entirely: aim at a plain wall, room light on. The black
field was buying CONTRAST, so `--wall` buys that back a different way, by
changing what counts as signal. Instead of "this pixel got brighter" it asks
"this pixel got GREENER" -- `G - max(R, B)`, the same chroma term
`calibrate.find_laser_dot` uses. A grey or white wall has G ~= R ~= B no
matter how the room is lit, so it sits near zero; a 532 nm dot is the only
thing in the scene with a large positive value. Brightness cannot separate
those two. Colour can, and it does not care that the lights are on.

The exposure lock does the rest: at -6 the wall is dim and the dot still
clips. Drop `--exposure` further if the wall is sunlit.

WHAT THIS DOES AND DOES NOT INCLUDE
-----------------------------------
INCLUDED: host serial write, firmware parse, MOSFET switch, diode turn-on,
camera exposure and readout, USB transport, MJPEG decode, host timestamp.
That is the sensing-and-command latency the control loop cannot avoid.

NOT INCLUDED: inference (app.py reports ~15 ms) and the motor's mechanical
response. Add the inference figure before setting config.LATENCY_S. The
mechanical term is left unmeasured on purpose -- see above.

WHAT "ONSET" MEANS
------------------
Never the first nonzero reading -- sensor noise, rolling shutter and JPEG
artifacts all move that. The noise floor is MEASURED first from frames with
the beam off, and onset is the first frame exceeding `floor + k*sigma`. The
threshold is reported with the answer, because a latency quoted without the
detection threshold that produced it cannot be checked by anyone.

THE BEAM IS ON FOR AT MOST ONE FRAME INTERVAL PER TRIAL, with the platform
stationary, pointed wherever it already points. `laser(False)` runs in a
`finally`. Look where it is aimed before starting.
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
import time
from contextlib import ExitStack
from typing import List, Optional

# BEFORE cv2, not after. `turret_host/__init__.py` sets
# OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0, which OpenCV reads once at import
# and which is worth 60 s on the narrow camera's open. Running this as
# `python -m turret_host.measure_latency` imports the package first anyway, but
# running the FILE does not, and the only symptom is a minute of apparent hang.
import turret_host  # noqa: F401  (import for its side effect, see above)

import cv2
import numpy as np

from turret_host import calibrate, cameras, config, projector

FLOOR_FRAMES = 40          # frames used to characterise the beam-off noise
SIGMA_K = 6.0              # onset threshold = floor_mean + SIGMA_K * floor_std
TIMEOUT_S = 1.0
# Held-beam peak the aim check demands before spending any trials. Deliberately
# low: this is "can the camera see the dot AT ALL", not the onset test. The
# onset test uses a floor measured on the day, and must stay the stricter of
# the two -- this one only exists to separate a pointing problem from a timing
# one, and a high bar here would reject marginal aims that still measure fine.
AIM_MIN_SIGNAL = 12.0


def _grey32(img: np.ndarray) -> np.ndarray:
    """Brightness. Correct against a BLACK field, where the dot is the light."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _green_excess32(img: np.ndarray) -> np.ndarray:
    """G - max(R, B): how GREEN a pixel is, regardless of how bright it is.

    The signal for `--wall`. A lit wall is near-neutral, so this sits near
    zero and STAYS there through anything that changes the light without
    changing its colour -- a cloud, someone walking past, the camera's own
    gain drifting. All of those move brightness, and none of them move a 5 mW
    green dot's 532 nm signature. That is the whole reason the projector's
    black field can be dropped.

    Signed, not absolute: something redder than the wall gives a negative
    value, and must never look like a dot.
    """
    bgr = img.astype(np.int16)
    return (bgr[:, :, 1]
            - np.maximum(bgr[:, :, 0], bgr[:, :, 2])).astype(np.float32)


def _dot_map(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Per-pixel change against the beam-off reference, in the prepped signal.

    SIGNED, not `absdiff`. The beam coming on can only push a pixel one way --
    brighter in `_grey32`, greener in `_green_excess32` -- so a pixel that
    moved the other way is not evidence of the beam, and an absolute
    difference would score it as though it were. That matters most in the
    chroma mode this was written for, where anything turning REDDER produces a
    large negative number: a hand, a warm reflection, a red LED on the bench.
    Taking the absolute value would let all three fire the trigger.

    Blurred before the max: DOT_AREA_PX says the dot covers 3-40 px, so a real
    dot survives a 3x3 blur while a hot single pixel from sensor noise does
    not. Taking a raw max would track the noisiest pixel in the frame.
    """
    return cv2.GaussianBlur(cv2.subtract(cur, ref), (3, 3), 0)


def _dot_signal(ref: np.ndarray, cur: np.ndarray) -> float:
    return float(_dot_map(ref, cur).max())


def aim_check(cam, link, prep) -> dict:
    """Hold the beam on and confirm the dot is in the narrow camera's field.

    The narrow camera sees 28.8 deg. A turret pointed a little off the wall,
    or at a patch outside that cone, produces a run of silent misses and a
    "NO FLASH DETECTED" at the end -- which looks like a broken instrument
    rather than a pointing problem. One steady-state look costs a second and
    tells the two apart, and its peak value also says whether the margin over
    the noise floor is comfortable or marginal before 16 trials are spent.

    Steady state, so this is NOT a latency sample: the beam is held on, well
    past any onset.
    """
    link.laser(False)
    time.sleep(0.25)
    for _ in range(5):
        cam.read()
    ref = prep(cam.read())

    link.laser(True)
    try:
        time.sleep(0.5)
        for _ in range(3):
            cam.read()                          # drop frames straddling the edge
        cur = prep(cam.read())
    finally:
        link.laser(False)

    d = _dot_map(ref, cur)
    _, peak, _, loc = cv2.minMaxLoc(d)
    return {"peak": float(peak), "xy": (int(loc[0]), int(loc[1])),
            "shape": (int(d.shape[1]), int(d.shape[0]))}


def measure_laser(cam, link, surface, trials: int, prep=_grey32) -> dict:
    """Command `laser on`, find the first frame containing the dot."""
    if surface is not None:
        surface.show_and_settle(surface.black(), 0.8)
    for _ in range(20):
        cam.read()                              # warm-up; first frames are black

    # -- 1. noise floor with the beam OFF ------------------------------------
    ref = prep(cam.read())
    floor: List[float] = []
    for _ in range(FLOOR_FRAMES):
        floor.append(_dot_signal(ref, prep(cam.read())))
    f_mean, f_std = float(np.mean(floor)), float(np.std(floor))
    threshold = f_mean + SIGMA_K * max(f_std, 1e-3)

    # -- 2. flash, and time the first frame that breaks the threshold --------
    # THE TIMESTAMP GOES AT THE SERIAL WRITE, NOT AROUND link.laser().
    # CalibrationLink.laser() calls command(), which writes and then drains the
    # reply for quiet_s = 0.25 s before returning. Timing around that would
    # measure the drain timeout: 250 ms would elapse before the first frame was
    # even requested, the dot would already be present in it, and the result
    # would look like a suspiciously small number rather than an obvious error.
    ser = link.ser
    lat: List[float] = []
    misses = 0
    try:
        for _ in range(trials):
            link.laser(False)
            time.sleep(0.15)
            ref = prep(cam.read())              # fresh reference each trial:
            cam.read()                          # exposure drifts over a run

            ser.reset_input_buffer()
            t_cmd = time.perf_counter()
            ser.write(b"laser on\r\n")
            ser.flush()

            hit: Optional[float] = None
            while time.perf_counter() - t_cmd < TIMEOUT_S:
                cur = prep(cam.read())
                t_frame = time.perf_counter()
                if _dot_signal(ref, cur) > threshold:
                    hit = t_frame - t_cmd
                    break
            # Raw write again: the beam goes off now, not after a drain.
            ser.write(b"laser off\r\n")
            ser.flush()
            # Only now let the link resynchronise with the replies to both.
            link._drain(0.15, 3.0)
            if hit is None:
                misses += 1
            else:
                lat.append(hit)
            time.sleep(0.1)
    finally:
        link.laser(False)

    return {"axis": "laser flash", "floor_px": f_mean, "floor_std": f_std,
            "threshold_px": threshold, "samples": lat, "misses": misses,
            "signal": getattr(prep, "__name__", "?"),
            "surface": "projector black field" if surface is not None else "wall"}


def _report(r: dict) -> None:
    s = r["samples"]
    print("\n%s" % ("-" * 66))
    print("%s   (%s, signal %s)" % (r["axis"], r["surface"], r["signal"]))
    print("  beam-off frame noise  %.2f (sd %.2f)  -> onset above %.2f"
          % (r["floor_px"], r["floor_std"], r["threshold_px"]))
    if not s:
        print("  NO FLASH DETECTED in any trial, though the aim check passed.")
        print("  That combination points at the DETECTION, not the pointing:")
        print("  the dot is visible held on but not within one frame of the")
        print("  command. Raise --trials, or lower --exposure so the dot")
        print("  clears the floor by more.")
        return
    a = np.array(s)
    print("  n=%d  misses=%d" % (len(s), r["misses"]))
    print("  median %.1f ms   mean %.1f ms   sd %.1f ms   min %.1f   max %.1f"
          % (np.median(a) * 1e3, a.mean() * 1e3, a.std() * 1e3,
             a.min() * 1e3, a.max() * 1e3))
    # The camera frame interval quantises this: an onset can only be seen on a
    # frame boundary, so the true value is up to one interval EARLIER than
    # measured. Say so rather than quoting a precision that is not there.
    print("  NOTE: quantised by the 33 ms frame interval -- the true value lies")
    print("        roughly between %.1f and %.1f ms."
          % (max(0.0, np.median(a) * 1e3 - 33.3), np.median(a) * 1e3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--port", default=None)
    ap.add_argument("--wall", action="store_true",
                    help="no projector: aim at a plain wall and detect the dot "
                         "by its GREEN excess instead of raw brightness")
    ap.add_argument("--exposure", type=float, default=-6.0,
                    help="narrow camera exposure lock (default -6; lower is "
                         "darker, which widens the dot's margin over the wall)")
    a = ap.parse_args(argv)

    ident = cameras.identify_cameras()
    print(ident.report())
    cameras.lock_exposure(ident.narrow_index, exposure=a.exposure)

    cam = calibrate.CalibrationCapture(ident.narrow_index,
                                       size=config.NARROW_SIZE,
                                       name="narrow").open()
    link = calibrate.CalibrationLink(port=a.port).open() if a.port \
        else calibrate.CalibrationLink().open()
    surface = None if a.wall else projector.ProjectorSurface()
    prep = _green_excess32 if a.wall else _grey32
    results = []
    print("\n" + "!" * 66)
    print("THE BEAM WILL FLASH %d TIMES, plus one steady second for the aim"
          % a.trials)
    print("check. The platform does not move, so it stays pointed wherever it")
    print("already points. LOOK AT WHERE THAT IS, and keep eyes out of it.")
    print("Each flash lasts one frame interval (~33 ms).")
    print("!" * 66)
    try:
        with ExitStack() as stack:
            if surface is not None:
                stack.enter_context(surface)

            aim = aim_check(cam, link, prep)
            print("\naim check: peak %+.1f at (%d, %d) in %dx%d"
                  % (aim["peak"], aim["xy"][0], aim["xy"][1],
                     aim["shape"][0], aim["shape"][1]))
            if aim["peak"] < AIM_MIN_SIGNAL:
                print("  THE DOT IS NOT VISIBLE (needs > %.0f). Nothing below"
                      % AIM_MIN_SIGNAL)
                print("  would have been measured, so stopping here instead of")
                print("  spending %d trials to say the same thing:" % a.trials)
                print("   - is the beam lighting at all? `laser on` at the")
                print("     firmware prompt, and look at the wall.")
                print("   - is the dot inside the narrow camera's 28.8 deg")
                print("     field? It is the NARROW one that has to see it.")
                if a.wall:
                    print("   - is the surface neutral? Green paint reads as")
                    print("     dot everywhere and leaves no contrast.")
                    print("   - try a lower --exposure, or dim the room.")
                return 1
            results.append(measure_laser(cam, link, surface, a.trials, prep))
    finally:
        # stop() before close(): the platform is stationary throughout, but a
        # link that is closed without a stop leaves the board holding whatever
        # it last had, and that is not a state to leave a turret in.
        link.laser(False)
        link.stop()
        link.close()
        cam.close()

    for r in results:
        _report(r)

    samples = [float(np.median(r["samples"])) for r in results if r["samples"]]
    print("\n%s" % ("=" * 66))
    print("config.LATENCY_S is currently %.3f s, marked ESTIMATE"
          % config.LATENCY_S)
    if not samples:
        print("NOTHING MEASURED -- see the note above.")
        return 1
    base = samples[0]
    print("  laser flash (command -> photons seen)   %.3f s" % base)
    infer = 0.015
    print("\nLATENCY_S wants the whole path the control law waits on, so ADD")
    print("the inference the loop pays first (app.py reports ~%.0f ms):" % (infer * 1e3))
    print("    LATENCY_S ~= %.3f s" % (base + infer))
    print("Usable P-loop bandwidth ~ 1/(8L) = %.1f Hz" % (1.0 / (8 * (base + infer))))
    if base + infer > config.LATENCY_S * 1.25:
        print("\nTHIS IS %.0f%% HIGHER than the configured estimate. The tracker"
              % (100.0 * (base + infer) / config.LATENCY_S - 100.0))
        print("leads the aim point by LATENCY_S, so an underestimate means it")
        print("aims BEHIND a moving target -- worst exactly when the drone is")
        print("fastest, which is the case the deliverable is about.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
