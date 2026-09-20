"""The projector as a CONTROLLABLE VISUAL FIELD, and the check that it works.

    python -m turret_host.projector --check      # do the cameras see it?
    python -m turret_host.projector --hold       # just put a pattern up
    python -m turret_host.projector --displays   # list monitors and exit

WHY THIS IS NOT "PROJECT A CHESSBOARD"
--------------------------------------
Projecting a calibration TARGET is a weak idea and this module does not do it.
Two reasons, both fatal for intrinsics: keystone means the pattern that lands on
the wall is an unknown projective warp of the one you sent, so the "known target
geometry" that Zhang's method depends on is not known; and the pattern is welded
to the wall plane, so you cannot present it at the varied orientations that
condition a focal-length estimate.

What the projector IS good for is the thing a printed board can never do -- it
changes on command, in closed loop with what the cameras measure:

  * GUARANTEED TEXTURE. `cv2.phaseCorrelate` needs a textured scene to find the
    pixel shift for the image Jacobian. A blank wall has none. Broadband noise
    is the ideal input for phase correlation -- flat power spectrum, one sharp
    correlation peak, no periodic structure to alias against.
  * A TARGET THAT MOVES ON COMMAND. The control law's sign has never been
    checked against a moving mechanism, and a wrong sign is runaway. A projected
    blob that steps to a known place turns that into a scripted test with no
    human holding anything and no beam.
  * AN INSTANT, TIMESTAMPED LIGHT CHANGE. End-to-end latency L sets the control
    loop's gain ceiling (~1/(8L)) and `config.LATENCY_S` is an estimate. The
    projector is a perfect substitute for the LED that measurement wanted,
    because the host knows exactly when it commanded the change.
  * A BLACK FIELD. The laser-dot calibration differences consecutive frames to
    find the dot. Killing the room's reflected light raises that contrast.

HOW THE CHECK PROVES DETECTION
------------------------------
Two independent tests, because either one alone can be fooled.

TEMPORAL -- display a PSEUDO-RANDOM binary sequence and correlate each camera's
mean brightness against it. Random rather than alternating on purpose: a regular
flash can correlate with mains flicker or with auto-exposure hunting, and a
random one cannot. Exposure is locked first for the same reason -- an AE loop
fighting the flashes suppresses exactly the signal being measured.

SPATIAL -- difference a white field against a black one. The static scene
cancels and the projected region is what is left, which gives the bounding box
of the projection in camera pixels and the contrast in grey levels. This is the
same trick the goal-pixel calibration uses on the laser dot.

Correlation alone is not enough: a camera pointed at a wall the projector merely
spills light onto tracks the sequence perfectly. The contrast swing is what says
the projector is actually IN FRAME rather than lighting the room.
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
import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from turret_host import config

ROOT = _os.path.dirname(_pkg_dir)
OUT_DIR = _os.path.join(ROOT, "turret_host", "calibration", "projector")

# Each displayed state is held this long before a frame is trusted. It has to
# cover the projector's own response, the camera's integration time and the USB
# pipeline. Frames are additionally required to be NEW since the change, so this
# is a floor and not the whole guarantee.
SETTLE_S = 0.35
FRESH_FRAMES = 3                 # discard this many after a change, then sample

# A fixed pseudo-random sequence -- fixed so a rerun is comparable, random so it
# cannot be matched by mains flicker or an auto-exposure loop. 0 = black field.
FLASH_SEQUENCE = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]

# Verdict thresholds. Correlation says "it tracks what I displayed"; swing says
# "and it is in frame, not just lighting the room".
MIN_CORRELATION = 0.80
MIN_SWING_LEVELS = 12.0          # mean grey-level difference, white vs black


# ---------------------------------------------------------------------------
#   displays
# ---------------------------------------------------------------------------
@dataclass
class Monitor:
    left: int
    top: int
    width: int
    height: int

    @property
    def is_primary(self) -> bool:
        return self.left == 0 and self.top == 0

    def describe(self) -> str:
        return "%dx%d at (%d,%d)%s" % (self.width, self.height, self.left,
                                       self.top, "  PRIMARY" if self.is_primary else "")


def enumerate_monitors() -> List[Monitor]:
    """Every monitor rectangle on the virtual desktop, primary first."""
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    found: List[Monitor] = []

    proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                               ctypes.POINTER(wintypes.RECT), ctypes.c_double)

    def _cb(_h, _hdc, lprc, _data):
        r = lprc.contents
        found.append(Monitor(r.left, r.top, r.right - r.left, r.bottom - r.top))
        return 1

    user32.EnumDisplayMonitors(0, 0, proto(_cb), 0)
    found.sort(key=lambda m: (not m.is_primary, m.left))
    return found


def pick_projector(monitors: List[Monitor]) -> Optional[Monitor]:
    """The projector is the non-primary monitor, if the desktop is EXTENDED.

    In DUPLICATE mode Windows reports a single rectangle, because a mirrored set
    is one desktop. That is not a failure -- the projector shows whatever the
    primary shows -- so the caller falls back to the primary and says so.
    """
    for m in monitors:
        if not m.is_primary:
            return m
    return None


# ---------------------------------------------------------------------------
#   the display surface
# ---------------------------------------------------------------------------
class ProjectorSurface:
    """A fullscreen window on the projector. Everything drawn is BGR uint8."""

    WINDOW = "projector"

    def __init__(self, monitor: Optional[Monitor] = None, mirrored: bool = False):
        mons = enumerate_monitors()
        self.mirrored = mirrored or (pick_projector(mons) is None)
        self.monitor = monitor or pick_projector(mons) or mons[0]
        self._open = False

    def open(self) -> "ProjectorSurface":
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        # Move BEFORE going fullscreen: fullscreen applies to whichever monitor
        # currently holds the window, so the move is what selects the display.
        cv2.moveWindow(self.WINDOW, self.monitor.left, self.monitor.top)
        cv2.resizeWindow(self.WINDOW, self.monitor.width, self.monitor.height)
        cv2.setWindowProperty(self.WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        self._open = True
        self.show(self.black())
        return self

    def close(self) -> None:
        if self._open:
            try:
                cv2.destroyWindow(self.WINDOW)
                cv2.waitKey(1)
            except cv2.error:
                pass
            self._open = False

    def __enter__(self) -> "ProjectorSurface":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def shape(self) -> Tuple[int, int]:
        return self.monitor.height, self.monitor.width

    # -- content ----------------------------------------------------------
    def black(self) -> np.ndarray:
        return np.zeros((self.monitor.height, self.monitor.width, 3), np.uint8)

    def white(self, level: int = 255) -> np.ndarray:
        return np.full((self.monitor.height, self.monitor.width, 3), level, np.uint8)

    def noise(self, cell: int = 8, seed: int = 0) -> np.ndarray:
        """Broadband binary noise -- the ideal input for phase correlation.

        `cell` sets the smallest feature. Too fine and the camera's optics and
        the projector's own focus low-pass it into flat grey; 8 px on a 2400-wide
        panel survives both comfortably.
        """
        h, w = self.monitor.height, self.monitor.width
        rng = np.random.default_rng(seed)
        small = rng.integers(0, 2, size=(h // cell + 1, w // cell + 1),
                             dtype=np.uint8) * 255
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    def blob(self, cx_frac: float, cy_frac: float, radius_frac: float = 0.06,
             level: int = 255) -> np.ndarray:
        """A filled disc at a fractional position -- the moving target."""
        img = self.black()
        h, w = img.shape[:2]
        cv2.circle(img, (int(cx_frac * w), int(cy_frac * h)),
                   max(2, int(radius_frac * min(h, w))), (level, level, level), -1)
        return img

    def identify(self) -> np.ndarray:
        """A human-readable pattern: corner markers, grid, and a label."""
        img = self.black()
        h, w = img.shape[:2]
        step = max(40, w // 24)
        for x in range(0, w, step):
            cv2.line(img, (x, 0), (x, h), (40, 40, 40), 1)
        for y in range(0, h, step):
            cv2.line(img, (0, y), (w, y), (40, 40, 40), 1)
        m = int(0.06 * min(h, w))
        for (px, py) in ((m, m), (w - m, m), (m, h - m), (w - m, h - m)):
            cv2.circle(img, (px, py), m // 2, (255, 255, 255), -1)
        cv2.circle(img, (w // 2, h // 2), m // 3, (0, 255, 255), -1)
        cv2.putText(img, "TURRET PROJECTOR", (w // 2 - 320, h // 2 - m),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        cv2.putText(img, "%dx%d" % (w, h), (w // 2 - 110, h // 2 + 2 * m),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        return img

    def show(self, img: np.ndarray, pump_ms: int = 1) -> None:
        cv2.imshow(self.WINDOW, img)
        cv2.waitKey(pump_ms)         # the pump is what actually paints it

    def show_and_settle(self, img: np.ndarray, seconds: float) -> None:
        """Show a frame and KEEP PUMPING while it settles.

        `imshow` alone does not paint. A bare `time.sleep()` after it leaves the
        window unrendered and the event queue unserviced, so the projector can
        still be showing the previous frame -- which reads downstream as a
        camera that does not respond to the display, not as a display bug.
        """
        end = time.time() + seconds
        cv2.imshow(self.WINDOW, img)
        while time.time() < end:
            cv2.imshow(self.WINDOW, img)
            cv2.waitKey(10)

    def hold(self, img: np.ndarray, seconds: float) -> None:
        """Keep a frame up and keep the window pumping, so it cannot go stale."""
        end = time.time() + seconds
        while time.time() < end:
            cv2.imshow(self.WINDOW, img)
            if (cv2.waitKey(15) & 0xFF) == 27:
                break


# ---------------------------------------------------------------------------
#   the check
# ---------------------------------------------------------------------------
@dataclass
class CameraVerdict:
    name: str
    samples: int = 0
    correlation: float = 0.0
    swing: float = 0.0
    mean_white: float = 0.0
    mean_black: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    coverage: float = 0.0
    error: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def sees_projector(self) -> bool:
        return (not self.error
                and self.correlation >= MIN_CORRELATION
                and self.swing >= MIN_SWING_LEVELS)

    def line(self) -> str:
        if self.error:
            return "  %-7s FAILED  %s" % (self.name, self.error)
        verdict = "SEES IT" if self.sees_projector else "NO"
        why = ""
        if not self.sees_projector:
            if self.correlation < MIN_CORRELATION:
                why = "  (brightness does not track the sequence)"
            elif self.swing < MIN_SWING_LEVELS:
                why = "  (tracks it, but only %.1f levels -- spill light, not in frame)" % self.swing
        return ("  %-7s %-8s r=%+.3f  swing=%.1f  (white %.1f / black %.1f)  "
                "coverage=%.1f%%%s"
                % (self.name, verdict, self.correlation, self.swing,
                   self.mean_white, self.mean_black, 100.0 * self.coverage, why))


def _grab_settled(thread, hold_s: float = SETTLE_S):
    """A frame that is definitely NEW since the display changed.

    Waiting on wall-clock alone is not enough: the capture thread publishes into
    a newest-wins slot, so the frame sitting there at the moment of the change
    may have been exposed before it. Require FRESH_FRAMES new indices, then take
    what is current.
    """
    first, _seq = thread.latest()
    last_index = first.index if first is not None else -1
    deadline = time.time() + hold_s + 1.5
    seen = 0
    frame = None
    while time.time() < deadline and seen < FRESH_FRAMES:
        f, _s = thread.latest()
        if f is not None and f.index != last_index:
            last_index = f.index
            seen += 1
            frame = f
        else:
            # waitKey, not sleep: this also services the HighGUI event queue, so
            # the projected frame stays painted while we wait on the camera.
            cv2.waitKey(5)
    return frame


def _mean_level(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(g.mean())


def run_check(surface: ProjectorSurface, threads: Dict[str, object],
              save: bool = True) -> Dict[str, CameraVerdict]:
    verdicts = {n: CameraVerdict(n) for n in threads}

    # -- temporal: does brightness track a random sequence? ---------------
    levels: Dict[str, List[float]] = {n: [] for n in threads}
    shown: List[int] = []
    for state in FLASH_SEQUENCE:
        surface.show_and_settle(surface.white() if state else surface.black(),
                                SETTLE_S)
        ok = True
        pending = {}
        for name, th in threads.items():
            f = _grab_settled(th)
            if f is None:
                verdicts[name].error = "no frames delivered"
                ok = False
            else:
                pending[name] = _mean_level(f.image)
        if ok:
            shown.append(state)
            for name, v in pending.items():
                levels[name].append(v)

    for name, v in verdicts.items():
        vals = levels[name]
        v.samples = len(vals)
        if v.error:
            continue
        if len(vals) < 6 or len(set(shown)) < 2:
            v.error = "not enough samples (%d)" % len(vals)
            continue
        if float(np.std(vals)) < 1e-6:
            v.correlation = 0.0
            v.notes.append("brightness perfectly flat -- exposure may be pinned or "
                           "the projector is not in this camera's view")
        else:
            v.correlation = float(np.corrcoef(np.array(shown, float),
                                              np.array(vals, float))[0, 1])
        w = [x for x, s in zip(vals, shown) if s]
        b = [x for x, s in zip(vals, shown) if not s]
        v.mean_white, v.mean_black = float(np.mean(w)), float(np.mean(b))
        v.swing = v.mean_white - v.mean_black

    # -- spatial: where in frame is it? ------------------------------------
    surface.show_and_settle(surface.white(), SETTLE_S)
    white = {n: _grab_settled(t) for n, t in threads.items()}
    surface.show_and_settle(surface.black(), SETTLE_S)
    black = {n: _grab_settled(t) for n, t in threads.items()}

    if save:
        _os.makedirs(OUT_DIR, exist_ok=True)

    for name, v in verdicts.items():
        fw, fb = white.get(name), black.get(name)
        if fw is None or fb is None:
            continue
        gw = cv2.cvtColor(fw.image, cv2.COLOR_BGR2GRAY).astype(np.int16)
        gb = cv2.cvtColor(fb.image, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = np.clip(gw - gb, 0, 255).astype(np.uint8)
        # Otsu rather than a fixed threshold: the projector's brightness in frame
        # depends on throw, room light and exposure, none of which are known here.
        _t, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        v.coverage = float((mask > 0).mean())
        cnts, _h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w_, h_ = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            v.bbox = (x, y, x + w_, y + h_)

        if save:
            vis = fw.image.copy()
            if v.bbox:
                cv2.rectangle(vis, v.bbox[:2], v.bbox[2:], (0, 255, 0), 3)
            # The C270 is mounted 90 deg over; rotate the DIAGNOSTIC only, so the
            # saved image is human-readable. Nothing measured above is rotated.
            if name == "narrow":
                vis = cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE
                                 if config.NARROW_ROTATE_CLOCKWISE
                                 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                diff = cv2.rotate(diff, cv2.ROTATE_90_CLOCKWISE
                                  if config.NARROW_ROTATE_CLOCKWISE
                                  else cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(_os.path.join(OUT_DIR, "%s_white.jpg" % name), vis)
            cv2.imwrite(_os.path.join(OUT_DIR, "%s_diff.jpg" % name), diff)

    return verdicts


# ---------------------------------------------------------------------------
#   hardware bring-up
# ---------------------------------------------------------------------------
def _start_cameras(exposure: Optional[float],
                   min_fps: Optional[float] = None) -> Tuple[Dict[str, object], List[str]]:
    """Open both cameras.

    `min_fps` relaxes `CameraThread`'s startup floor. The floor exists to stop a
    30 Hz TRACKING loop running on a camera that cannot feed it; this check is a
    stepped measurement that holds each state for a third of a second, so a
    camera delivering 22 fps is perfectly adequate here and refusing it costs a
    result. The floor is not relaxed anywhere near the tracking path.
    """
    from turret_host import cameras
    notes: List[str] = []
    ident = cameras.identify_cameras()
    notes.append(ident.report())

    threads: Dict[str, object] = {}
    for name, index, factory in (("narrow", ident.narrow_index, cameras.narrow_thread),
                                 ("wide", ident.wide_index, cameras.wide_thread)):
        if exposure is not None:
            try:
                r = cameras.lock_exposure(index, exposure)
                notes.append("%s exposure lock: %s" %
                             (name, "OK" if r.ok else "NOT APPLIED -- " + r.message))
            except Exception as exc:                      # noqa: BLE001
                notes.append("%s exposure lock raised: %s" % (name, exc))
        try:
            th = factory(index)
            if min_fps is not None:
                th.min_fps = min_fps
            threads[name] = th.start()
            notes.append("%s started at %.1f fps" % (name, th.startup_fps))
        except Exception as exc:                          # noqa: BLE001
            notes.append("%s camera FAILED to start: %s" % (name, exc))
    return threads, notes


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="prove the cameras see the projector (default)")
    ap.add_argument("--hold", action="store_true",
                    help="display the identify pattern until Esc")
    ap.add_argument("--displays", action="store_true", help="list monitors and exit")
    ap.add_argument("--seconds", type=float, default=60.0, help="--hold duration")
    ap.add_argument("--exposure", type=float, default=-6.0,
                    help="locked exposure, log2 seconds (-6 = 1/64 s). "
                         "A fixed value matters more than which value: an "
                         "auto-exposure loop suppresses the flashing being measured.")
    ap.add_argument("--no-exposure-lock", action="store_true")
    ap.add_argument("--min-fps", type=float, default=15.0,
                    help="startup fps floor for this check only (default 15). "
                         "The tracking stack's %.0f fps floor is not touched."
                         % config.MIN_ACCEPTABLE_FPS)
    args = ap.parse_args(argv)

    mons = enumerate_monitors()
    proj = pick_projector(mons)
    print("displays:")
    for i, m in enumerate(mons):
        tag = "  <-- projector" if (proj and m is proj) else ""
        print("   %d: %s%s" % (i, m.describe(), tag))
    if proj is None:
        print("\nNo secondary display. Either the desktop is DUPLICATED (in which")
        print("case the primary IS the projector and this still works), or the")
        print("projector is not attached.")
    if args.displays:
        return 0

    surface = ProjectorSurface()
    if args.hold:
        with surface:
            print("\nholding the identify pattern on %s -- Esc in the window to stop"
                  % surface.monitor.describe())
            surface.hold(surface.identify(), args.seconds)
        return 0

    exposure = None if args.no_exposure_lock else args.exposure
    threads: Dict[str, object] = {}
    try:
        with surface:
            print("\nprojecting on %s" % surface.monitor.describe())
            if surface.mirrored:
                print("   (duplicate mode: this also covers the laptop screen)")

            # Light the room BEFORE opening the cameras. The C270's startup fps
            # probe is what decides whether it is allowed to run, and its auto
            # exposure lengthens integration in a dim room until it falls under
            # the floor -- so the projector is the light source that lets its own
            # test run. This is the closed loop the projector is for.
            surface.show_and_settle(surface.white(), 1.0)

            print("\nstarting cameras (projector lighting the room)...")
            threads, notes = _start_cameras(exposure, args.min_fps)
            for n in notes:
                print("   " + n.replace("\n", "\n   "))
            if not threads:
                print("\nno cameras started -- nothing to check")
                return 1

            print("\nflashing a %d-state pseudo-random sequence..." % len(FLASH_SEQUENCE))
            verdicts = run_check(surface, threads)
    finally:
        for th in threads.values():
            try:
                th.stop()
            except Exception:                             # noqa: BLE001
                pass

    print("\nRESULT")
    for v in verdicts.values():
        print(v.line())
        if v.bbox:
            print("           projection bbox in frame: %s" % (v.bbox,))
        for n in v.notes:
            print("           note: %s" % n)
    print("\nthresholds: correlation >= %.2f AND swing >= %.0f grey levels"
          % (MIN_CORRELATION, MIN_SWING_LEVELS))
    print("diagnostics -> %s" % OUT_DIR)

    ok = [v for v in verdicts.values() if v.sees_projector]
    return 0 if len(ok) == len(verdicts) and verdicts else 1


if __name__ == "__main__":
    raise SystemExit(main())
