"""app.py -- thread wiring, lifecycle and shutdown for the turret host.

    capture-narrow --> infer-narrow --> tracker --> control --> link --> Pico
          |                                 |          |
          +--> faces-narrow ----------------+----------+
    capture-wide   --> infer-wide (throttled, display + situational awareness)
          |                                 |          |
          +---------------------------------+----------+--> SystemStatus --> GUI

Every hand-off is a `types.Slot`. A queue would deliver stale frames that the
filter treats as current; a slot delivers the newest thing there is and lets a
slow consumer skip.

Three things in here are load-bearing and look odd:

1.  **Homing runs automatically at startup and tracking is refused until it
    finishes.** The machine must come up on a repeatable datum, not "wherever
    it was left". START is disabled until `_ready` is set.

2.  **The narrow face pass has its own thread, and STALE FACE DATA INHIBITS
    THE LASER.** YuNet is CPU-only in OpenCV 5.0 and costs ~40 ms on a
    1280x720 frame -- longer than the frame period -- so it cannot live in the
    inference thread without halving the tracking rate. And because it can
    fall behind, "no face reported recently" is treated as *unknown*, not as
    *clear*: unknown does not get to emit light.

3.  **Only NARROW faces gate the laser.** control.nearest_face_px measures
    against the beam path in narrow-frame pixels and there is no wide->narrow
    mapping anywhere in this stack (it needs both intrinsics and the baseline,
    which are uncalibrated). Wide faces are drawn for the operator and gate
    nothing. This is stated at startup in the log so nobody assumes otherwise.

Run:
    python -m turret_host.app                 real hardware
    python -m turret_host.app --no-hardware   synthetic source, simulated link
    python -m turret_host.app --no-gui        headless, Ctrl-C to stop
"""
from __future__ import annotations

# --------------------------------------------------------------------------
#   sys.path repair -- MUST run before any other import, and may use only
#   `os`/`sys`, which are both loaded before user code runs.
#
#   Running this file by path puts turret_host/ at sys.path[0], where types.py
#   SHADOWS THE STDLIB `types` MODULE. The next lazy stdlib import chain
#   (`threading -> functools -> from types import GenericAlias`) then picks up
#   ours and dies with a circular-import error that names this package and
#   looks like our bug. REPLACE the script directory with the project root:
#   merely inserting the root is not enough, because sys.path[0] still wins.
# --------------------------------------------------------------------------
import os as _os
import sys as _sys

if __package__ in (None, ""):
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path[:] = [p for p in _sys.path
                    if _os.path.abspath(p or _os.getcwd()) != _here]
    _sys.path.insert(0, _os.path.dirname(_here))

import argparse                                                   # noqa: E402
import collections                                                # noqa: E402
import gc                                                         # noqa: E402
import math                                                       # noqa: E402
import signal                                                     # noqa: E402
import sys                                                        # noqa: E402
import threading                                                  # noqa: E402
import time                                                       # noqa: E402
import traceback                                                  # noqa: E402
from dataclasses import dataclass                                 # noqa: E402
from dataclasses import replace as _dc_replace                    # noqa: E402
from pathlib import Path                                          # noqa: E402
from typing import List, Optional                                 # noqa: E402

import numpy as np                                                # noqa: E402

from turret_host import config                                    # noqa: E402
import re                                                         # noqa: E402

from turret_host.camera_fusion import WideToNarrow                # noqa: E402
# MODULE LEVEL, deliberately. step_integrity was reachable only through the
# local `from turret_host import cameras, step_integrity` inside _start_cameras,
# so the two call sites in OTHER methods -- the homing datum hook and the
# end-of-run verify -- raised NameError and were swallowed by their own
# try/except, printing "datum failed (name 'step_integrity' is not defined)"
# and silently writing no step_check.json at all. It is safe here: unlike
# `cameras` it pulls only stdlib and turret_host.config, no cv2, no torch.
from turret_host import step_integrity                             # noqa: E402
from turret_host.flowvel import FlowVelocity                       # noqa: E402

#: `imu fast` -> "IMUF gx gy gz pitch roll". Groups 4 and 5 are the angles.
_IMUF_RE = re.compile(r"IMUF\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+"
                      r"([-+\d.]+)\s+([-+\d.]+)")
from turret_host.types import (                                   # noqa: E402
    ControlOutput,
    Detection,
    DetectionResult,
    Frame,
    LaserState,
    AttitudeStatus,
    LinkStatus,
    ReckonedPose,
    Slot,
    SystemStatus,
    TrackEstimate,
    TrackState,
)

# --------------------------------------------------------------------------
#   App-level policy. Not duplicates of config.py -- these are wiring
#   decisions that belong to the orchestrator and nothing else needs them.
# --------------------------------------------------------------------------

# A face report older than this is UNKNOWN, and unknown inhibits. YuNet at
# 1280x720 measures ~40 ms on this machine, so a healthy pass lands every
# ~45 ms; 200 ms means several consecutive misses before the beam drops, and
# still a fifth of a second of ignorance at most.
FACE_MAX_AGE_S = 0.200

# A wide frame older than this is not evidence about where the drone is NOW.
# The wide detector is throttled to WIDE_SEARCH_FPS_TRACKING (10 Hz) while
# tracking, so its newest frame can legitimately be ~100 ms old; 150 ms allows
# for that plus one dropped frame and no more. Past it, the fallback is
# refused and the tracker sees a miss -- which is the honest answer.
WIDE_FALLBACK_MAX_AGE_S = 0.150

# How long the control loop waits for a new track estimate before it treats the
# pipeline as STALLED and drops the beam. It is the est_slot wait timeout, and
# it is deliberately the same order as the interlock's own VEL_FRESH_S (100 ms):
# a gap this long would already have failed vel_fresh INSIDE evaluate(), if
# evaluate() were still being called -- which, with no new estimate, it is not.
# One frame period is 33 ms, so this is three missed frames.
_EST_STALL_S = 0.250   # 2026-09-20: was 0.100, which assumed 30 Hz frames. The narrow camera delivers ~15 DISTINCT fps (the rest were re-stamped duplicates that fed this watchdog until DEDUP_FRAMES dropped them); at 0.100 it tripped ~1/s and zeroed the rates. Firmware vel watchdog is 400 ms.

# How long shutdown() waits for a running platform task (homing, Jacobian
# calibration) to finish before giving up on it. homing.py's own sequence is
# ~40 s; this is that plus margin for a single blocking command in flight, and
# it is a bound on how long the window takes to close.
_PLATFORM_JOIN_S = 6.0
#: How long shutdown will wait for the closing re-home. A full home measures
#: ~40 s (35.9 s on the first real run), and `level` alone can iterate for a
#: while on a rig that is settling, so this is roughly 2x the expected worst
#: case. Finite because a window-close must not be holdable hostage by a rig
#: that has stopped answering.
_CLOSING_HOME_BUDGET_S = 90.0

# Exposure setpoint, DirectShow log2-seconds. config.py has no exposure
# constant and cameras.lock_exposure() deliberately leaves it to the caller:
# 2^-6 = 15.6 ms, which fits inside a 33 ms frame with margin and was the
# value verified on this hardware (29.8 fps after the lock).
NARROW_EXPOSURE = -6
NARROW_GAIN = 64
WIDE_EXPOSURE = -6


# ==========================================================================
#   ATTITUDE ENVELOPE  --  measured, not dead-reckoned
# ==========================================================================
# The travel-limit guard in control.TravelLimits runs on `self._pose`, which
# is INTEGRATED FROM COMMANDS. Measured on run_2026-09-19_002851, over 61 s of
# tracking:
#
#     pitch by gravity        -0.25 -> +71.20 deg   (change +71.45, max +78.33)
#     pitch dead-reckoned      0.00 -> +32.21 deg   (change +32.21)
#
# The mechanism moved 2.2x what the host believed, so the guard was reasoning
# about a pose 39 deg from reality against a LIMIT_MARGIN_DEG of 3.0. It let
# the turret run to 78 deg -- near the +/-90 stop -- while thinking it had
# room. Two reasons, both structural: _integrate_pose only runs on frames that
# COMMAND (202 of 621 here), while the Pico holds the last rate until
# superseded or the 400 ms watchdog; and it applies the newly-computed rate
# over a dt that has already elapsed.
#
# So: a second guard, on MEASURED attitude. Gravity cannot drift and cannot be
# zeroed wrong.
#
# Deliberately the TOTAL angle from the datum attitude rather than a pitch
# component. Which IMU axis a payload-pitch lands on depends on yaw -- that is
# what made `level` measure a negative sensitivity and drive the payload into
# its own frame -- so any per-axis reading needs a yaw the host does not
# reliably have. The angle between the current gravity vector and the one
# recorded at the datum needs no axis assignment at all.
#
# 60 deg against a 90 deg mechanical stop. The sampler runs at ~10 Hz, so a
# reading can be 100 ms old, and full-rate payload pitch is 195 deg/s = 19.5
# deg in that time. 30 deg of margin covers it with room to spare.
ATTITUDE_MAX_DEG = 85.0   # 2026-09-20 operator: 60 -> 85 (mechanical stop is +/-90 from level; the skip-level datum can be several deg off)
# 2026-09-20 operator: on an envelope trip, do not freeze -- REVERSE the last
# ATTITUDE_REVERSE_S of commanded motion (direction-preserving, at least
# ATTITUDE_REVERSE_MIN_RATE on the faster motor, at most 600 steps/s) until gravity
# says the payload is ATTITUDE_REVERSE_HYST_DEG back inside, then resume tracking.
ATTITUDE_REVERSE_S = 1.0
ATTITUDE_REVERSE_MIN_RATE = 600.0   # 2026-09-20 run_203933: reverse at 200..600 recovered only 8 deg from 86; a 1600-rate excursion needs a 1600-rate reverse
ATTITUDE_REVERSE_HYST_DEG = 10.0    # reverse starts at 75 (limit 85 stays, Ryan's number); 5 deg was not enough margin at 48 deg/s

#: A measured attitude older than this is not evidence about now, and the
#: guard falls back to the dead-reckoned one. Two sample periods at 10 Hz.
ATTITUDE_MAX_AGE_S = 0.250

#: How often to ask the board for its attitude while servoing. Every sample
#: costs a bounded attempt at the io lock and is DROPPED rather than delaying
#: a vel -- see link.try_probe.
ATTITUDE_SAMPLE_HZ = 10.0

#: Consecutive board-level refusals before the sampler backs off. `imu fast`
#: raises OSError on the board when the GY-85 is off the bus, and the firmware
#: answers that by disabling every axis -- so a monitor that keeps asking
#: disables the machine continuously. Small, because there is no value in
#: asking a second time once the answer is an error.
ATTITUDE_FAIL_LIMIT = 3

#: How long to wait before trying again once suspended. Long enough that a
#: missing sensor costs nothing, short enough that reseating a connector is
#: noticed without a restart.
ATTITUDE_BACKOFF_S = 10.0

#: Datum tilt off the yaw axis above which the pose overlay says so. Set just
#: under the 4.4 deg the bench actually measured on 2026-09-18, so the
#: condition is reported rather than silently degrading the display.
POSE_BASE_TILT_WARN_DEG = 3.0


def _now() -> float:
    return time.perf_counter()


# ==========================================================================
#   Hand-off payloads
#
#   Local to this module: they are wiring, not contract. Anything another
#   module needs to understand is already a types.py dataclass.
# ==========================================================================
# Seconds of live frames the ARM-time orientation check watches. 6 s at 30 fps
# is ~180 frames against a `need` of 8 detections, so a face that is actually
# visible clears it with wide margin while a blind detector cannot.
FACE_VERIFY_SECONDS = 6.0


@dataclass
class DetectedFrame:
    """One narrow frame and what the target detector found in it."""
    frame: Frame
    result: DetectionResult


@dataclass
class TrackedFrame:
    """...and what the filter made of it."""
    frame: Frame
    result: DetectionResult
    estimate: TrackEstimate


@dataclass
class FaceReport:
    """The narrow face pass. `t` is the CAPTURE time, not the finish time --
    staleness has to be measured against the world, not against our own
    scheduling."""
    t: float
    faces: List[Detection]
    infer_ms: float


# ==========================================================================
#   Synthetic hardware for --no-hardware
# ==========================================================================
class SyntheticSource(threading.Thread):
    """A camera that is not there: frames, timing and ground truth.

    Exists so the whole thread graph, the filter, the interlock and the GUI
    can be exercised on a machine with no camera and no board. It publishes
    real types.Frame objects into a real Slot at the real frame period, so
    what it tests is the actual wiring rather than a mock of it.
    """

    def __init__(self, name: str, size, fps: int, slot: Slot, seed: int = 0):
        super().__init__(name="synthetic-%s" % name, daemon=True)
        self.camera_name = name
        self.size = tuple(size)
        self.requested_fps = fps
        self.slot = slot
        self.startup_fps = float(fps)
        self.fps = 0.0
        self.frames = 0
        self.failed_grabs = 0
        self.duplicates_dropped = 0  # synthetic frames are always newly generated
        self.max_gap_ms = 0.0
        self.error: Optional[str] = None

        self._stop_evt = threading.Event()
        self._rng = np.random.default_rng(seed)
        self._truth_lock = threading.Lock()
        self._truth = {}
        self._t0 = _now()

        w, h = self.size
        # One background, rendered once. Allocating a fresh 1920x1080 noise
        # field per frame would dominate the run and measure nothing useful.
        bg = self._rng.integers(28, 46, size=(h, w, 1), dtype=np.uint8)
        self._bg = np.repeat(bg, 3, axis=2)
        for gx in range(0, w, 160):
            self._bg[:, gx:gx + 2] = 70
        for gy in range(0, h, 160):
            self._bg[gy:gy + 2, :] = 70

    # -- the moving world ------------------------------------------------
    def _truth_boxes(self, t: float):
        """Target and face, in this camera's pixels.

        The target jinks (that is what the adaptive Q exists for) and the face
        sweeps across the frame every 11 s so the inhibit path actually fires
        during a dry run instead of being theory.
        """
        w, h = self.size
        cx = w * (0.5 + 0.26 * math.sin(t * 0.9) + 0.06 * math.sin(t * 5.3))
        cy = h * (0.5 + 0.17 * math.sin(t * 1.4 + 1.0) + 0.04 * math.sin(t * 6.7))
        bw = w * 0.10
        bh = bw * 0.55
        target = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)

        phase = (t % 11.0) / 11.0
        fx = w * (-0.15 + 1.3 * phase)
        fy = h * 0.52
        fw, fh = w * 0.09, w * 0.11
        face = (fx - fw / 2, fy - fh / 2, fx + fw / 2, fy + fh / 2)
        return target, face

    def truth_for(self, frame: Frame):
        with self._truth_lock:
            return self._truth.get(frame.index)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "SyntheticSource":
        super().start()
        return self

    def run(self) -> None:
        period = 1.0 / float(self.requested_fps)
        next_t = _now()
        times = []
        while not self._stop_evt.is_set():
            now = _now()
            if now < next_t:
                self._stop_evt.wait(next_t - now)
                continue
            next_t += period
            if next_t < now - period:
                next_t = now + period           # we fell behind; do not spiral

            t = _now()
            elapsed = t - self._t0
            target, face = self._truth_boxes(elapsed)
            img = self._bg.copy()
            self._fill(img, target, (210, 215, 205))
            self._fill(img, face, (150, 130, 120))

            self.frames += 1
            with self._truth_lock:
                self._truth[self.frames] = (target, face)
                if len(self._truth) > 240:
                    for k in sorted(self._truth)[:120]:
                        del self._truth[k]

            self.slot.put(Frame(image=img, t=t, index=self.frames,
                                camera=self.camera_name))
            times.append(t)
            if len(times) > 31:
                times.pop(0)
            if len(times) > 1 and times[-1] > times[0]:
                self.fps = (len(times) - 1) / (times[-1] - times[0])

    @staticmethod
    def _fill(img, box, colour) -> None:
        h, w = img.shape[:2]
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
        if x2 > x1 and y2 > y1:
            img[y1:y2, x1:x2] = colour

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)
        self.fps = 0.0


class SyntheticDetector:
    """Ground truth plus noise. Same surface as detector.Detector."""

    def __init__(self, source: SyntheticSource, jitter_px: float = 2.0,
                 miss_rate: float = 0.03, seed: int = 1):
        self.source = source
        self.jitter_px = float(jitter_px)
        self.miss_rate = float(miss_rate)
        self._rng = np.random.default_rng(seed)
        self.last_infer_ms = 0.0
        self.missing_classes: List[str] = []
        self.class_ids: List[int] = [0]
        self.weights_path = "<synthetic>"

    def start(self):
        return self

    def close(self) -> None:
        pass

    def _jitter(self, box):
        j = self._rng.normal(0.0, self.jitter_px, 4)
        return tuple(float(b + k) for b, k in zip(box, j))

    def detect(self, frame: Frame) -> DetectionResult:
        t0 = _now()
        truth = self.source.truth_for(frame)
        targets: List[Detection] = []
        if truth is not None and self._rng.random() >= self.miss_rate:
            x1, y1, x2, y2 = self._jitter(truth[0])
            targets.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2,
                                     conf=float(0.55 + 0.4 * self._rng.random()),
                                     label="drone"))
        # Spend something, so the loop timing is not a fantasy.
        time.sleep(0.004)
        self.last_infer_ms = (_now() - t0) * 1000.0
        return DetectionResult(frame_t=frame.t, frame_index=frame.index,
                               camera=frame.camera, targets=targets, faces=[],
                               infer_ms=self.last_infer_ms)


class SyntheticFaceDetector:
    """Same surface as detector.FaceDetector: detect(frame) -> [Detection]."""

    def __init__(self, source: SyntheticSource, jitter_px: float = 2.0, seed: int = 2):
        self.source = source
        self.jitter_px = float(jitter_px)
        self._rng = np.random.default_rng(seed)
        self.last_infer_ms = 0.0
        self.model_path = "<synthetic>"

    def start(self):
        return self

    def close(self) -> None:
        pass

    def detect(self, frame: Frame) -> List[Detection]:
        t0 = _now()
        truth = self.source.truth_for(frame)
        out: List[Detection] = []
        if truth is not None:
            j = self._rng.normal(0.0, self.jitter_px, 4)
            x1, y1, x2, y2 = (float(b + k) for b, k in zip(truth[1], j))
            out.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=0.93, label="face"))
        # YuNet costs ~40 ms on a 1280x720 frame. Simulating that cost is the
        # point: it is why the face pass has a thread at all.
        time.sleep(0.010)
        self.last_infer_ms = (_now() - t0) * 1000.0
        return out

    def detect_into(self, result: DetectionResult, frame: Frame) -> DetectionResult:
        result.faces = self.detect(frame)
        result.infer_ms += self.last_infer_ms
        return result


class SimulatedLink:
    """A board that is not there. Same surface as link.TurretLink.

    It accepts rates, integrates them into a pose, and refuses nothing -- the
    point is to exercise the control loop's call pattern and the shutdown
    path, not to model the firmware.
    """

    def __init__(self):
        self.vel_slot = Slot()
        self.status_slot = Slot()
        self._lock = threading.Lock()
        self._sent = 0
        self._last_vel_ok_t: Optional[float] = None
        self._servo = False
        self._laser = False
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self._closed = False
        # Mirrors TurretLink's safety epoch so the dry run exercises the real
        # call pattern, including the epoch drop.
        self._safety_epoch = 0
        # Mirrors TurretLink's hard beam latch and abandoned-"on" flag. These
        # are part of the link contract the control loop depends on, so the
        # simulated link has to implement them or --no-hardware stops
        # exercising the real code path -- and an AttributeError the moment the
        # interlock reaches FIRING is a poor way to find that out.
        self._beam_latched = False
        self._laser_on_failed = False

    def start(self):
        return self

    def close(self):
        self.set_laser(False)
        self.send_vel(0.0, 0.0)
        self._closed = True
        with self._lock:
            self._servo = False

    def submit(self, out: ControlOutput, epoch=None):
        return self.send_vel(out.rate_a, out.rate_b, epoch=epoch)

    def send_vel(self, rate_a, rate_b, epoch=None):
        ra, rb = float(rate_a), float(rate_b)
        if not (math.isfinite(ra) and math.isfinite(rb)):
            raise ValueError("simulated link got a non-finite rate: %r %r" % (ra, rb))
        with self._lock:
            if epoch is not None and epoch != self._safety_epoch:
                return False
            self._sent += 1
            self._last_vel_ok_t = _now()
            self._servo = True
        return True

    def stop(self):
        self.safety_veto()
        self.set_laser(False)
        self.send_vel(0.0, 0.0)
        with self._lock:
            self._servo = False

    @property
    def safety_epoch(self):
        with self._lock:
            return self._safety_epoch

    def safety_veto(self, latch=False):
        with self._lock:
            self._safety_epoch += 1
            self._laser = False
            if latch:
                self._beam_latched = True

    @property
    def beam_latched(self):
        with self._lock:
            return self._beam_latched

    @property
    def laser_on_failed(self):
        with self._lock:
            return self._laser_on_failed

    def clear_beam_latch(self):
        with self._lock:
            was = self._beam_latched
            self._beam_latched = False
            self._laser_on_failed = False
        return was

    @property
    def laser_path_ok(self):
        return not self._closed

    def set_laser(self, on, epoch=None):
        with self._lock:
            if on and self._beam_latched:
                return False
            if on and epoch is not None and epoch != self._safety_epoch:
                return False
            if on and self._closed:
                return False
            self._laser = bool(on)
        return True

    @property
    def laser_on(self):
        with self._lock:
            return self._laser

    @property
    def connected(self):
        return not self._closed

    @property
    def servoing(self):
        with self._lock:
            return self._servo

    @property
    def port(self):
        return "SIMULATED"

    def ms_since_vel(self):
        with self._lock:
            t = self._last_vel_ok_t
        return float("inf") if t is None else (_now() - t) * 1000.0

    def vel_overdue(self):
        return self.ms_since_vel() > config.VEL_COMMAND_PERIOD_MAX_MS

    def status(self) -> LinkStatus:
        with self._lock:
            return LinkStatus(connected=not self._closed, port="SIMULATED",
                              rtt_ms=3.0, sent=self._sent, errors=0,
                              last_error="", pitch_deg=self.pitch_deg,
                              yaw_deg=self.yaw_deg)

    def command(self, cmd, timeout=None, settle=None):
        raise RuntimeError("simulated link has no console: %r" % cmd)


# ==========================================================================
#   The application
# ==========================================================================
class TurretApp:
    """Owns every thread and every handle, and gives all of them back."""

    def __init__(self, args):
        from turret_host.microstepping import configure
        configure(getattr(args, "microstep", None) or 16,
                  dynamic=getattr(args, "dynamic_microstep", False))
        self.args = args
        self.sim = bool(args.no_hardware)
        # Flight recorder, or None. Set up in startup() once the link exists;
        # the control loop checks for None on every pass.
        self.recorder = None
        # Kept separately from `recorder`, because the step check runs AFTER
        # recorder.close() has written session.json and set recorder to None,
        # and its report still has to land in the same directory as the
        # control/IMU streams it has to be read alongside.
        self._run_dir = None
        # Was the datum actually at IMU zero? None until homing has run.
        self._datum_level_ok = None
        self._datum_pitch_deg = None

        # -- hand-offs ---------------------------------------------------
        self.narrow_slot = Slot()
        #: Last flow velocity and which tier produced it, handed from the
        #: detect thread to the control thread. Plain assignment of a tuple is
        #: atomic enough here: the control thread wants the most recent value
        #: and a one-frame-old one is not a fault.
        self._ff_velocity = (0.0, 0.0)
        self._ff_source = "none"
        self._ff_points = 0
        self._ff_ms = 0.0
        self.wide_slot = Slot()
        self.det_slot = Slot()          # DetectedFrame, narrow
        # Wide detections now reach the TRACKER as a fallback, not just the
        # display -- see _tracker_loop. They still never gate the beam.
        self.wide_det_slot = Slot()     # DetectedFrame, wide (display + fallback)
        self._w2n = WideToNarrow.load()
        #: Target image velocity for the FEEDFORWARD only, from optical flow on
        #: the box patch. Lives on the detect thread, which is the only place
        #: the raw narrow frame and the box exist together. See flowvel.py.
        self.flowvel = FlowVelocity()

        # Measured attitude, from gravity. (t, pitch_deg, roll_deg).
        self.attitude_slot = Slot()
        # Unit gravity vector at the datum, captured when homing completes.
        # None until then -- the guard is inert without a reference.
        self._tilt_datum = None
        # Pose overlay: (t, pitch) of the last smoothed measurement, the
        # one-shot sign-check latch, and the base tilt measured at the datum.
        self._pose_filt = (0.0, None)
        self._pose_scale: list = []
        self._pose_scale_warned = False
        self._base_tilt_deg = None
        self._attitude_tripped = False
        self._cmd_hist: collections.deque = collections.deque(maxlen=120)   # (t, rate_a, rate_b) sent
        self._reverse_rates = (0.0, 0.0)
        self.face_slot = Slot()         # FaceReport, narrow (SAFETY)
        self.est_slot = Slot()          # TrackedFrame
        self.status_slot = Slot()       # SystemStatus -> GUI

        # -- components, all constructed empty; nothing opens yet --------
        self.link = None
        self.narrow_cam = None
        self.wide_cam = None
        self.detector = None
        self.wide_detector = None
        self.face_detector = None
        self.wide_face_detector = None
        self.tracker = None
        self.controller = None
        self.interlock = None
        self.gui = None

        self.narrow_index: Optional[int] = None
        self.wide_index: Optional[int] = None

        # -- lifecycle flags ---------------------------------------------
        self._stop = threading.Event()      # process is going down
        self._tracking = threading.Event()  # operator pressed START
        self._ready = threading.Event()     # homing finished; START is allowed
        self._platform_busy = threading.Event()   # homing/calibration owns the motors
        self._threads: List[threading.Thread] = []
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        # Latched by the control loop so a dead camera or a dead port is
        # reported (and disarmed) once, not 10 times a second.
        self._faulted = False
        # Has the face-interlock orientation self-check passed in THIS process?
        # False until proven. on_arm() refuses while it is False, so the beam
        # cannot be permitted by a build whose rotation constant has never been
        # shown to produce a detection. See _check_face_orientation().
        self._face_orientation_ok = False
        self._face_orientation_detail = ""
        # A platform task (homing / Jacobian calibration) runs on its own
        # thread and issues BLOCKING console commands -- `level` is a 240 s
        # timeout. shutdown() has to join it before it touches the link, or
        # link.close() queues behind it on the io lock, on the Tk main thread.
        self._platform_thread: Optional[threading.Thread] = None
        self._platform_cancel = threading.Event()

        # -- telemetry ----------------------------------------------------
        self._loop_hz = 0.0
        self._loop_times: List[float] = []
        self._status_message = "starting"
        self._last_laser = LaserState.DISARMED
        self._pose = [0.0, 0.0]
        self._pose_valid = False
        #: Latched so the rejection is logged once, not every frame.
        self._pose_distrusted = False
        #: COARSE acquisition phase. _coarse_t0 is when the current approach
        #: began; _coarse_gave_up_at is when the last one timed out.
        self._coarse_t0 = None
        self._coarse_gave_up_at = None
        self._pose_t = 0.0
        self._gc_frozen = False

    # ------------------------------------------------------------------
    #   logging -- one place, so headless and GUI runs say the same thing
    # ------------------------------------------------------------------
    def log(self, text: str, level: str = "info") -> None:
        print("[%s] %s" % (level.upper()[:4], text))
        sys.stdout.flush()
        if self.gui is not None:
            self.gui.log(text, level)

    def progress(self, text: str, fraction: Optional[float] = None,
                 done: bool = False, failed: bool = False,
                 title: str = "HOMING") -> None:
        if self.gui is not None:
            self.gui.set_homing_progress(text, fraction, done=done,
                                         failed=failed, title=title)
        else:
            pct = "" if fraction is None else "[%3d%%] " % round(100 * fraction)
            print("%s%s: %s" % (pct, title, text))
            sys.stdout.flush()

    # ==================================================================
    #   STARTUP  -- runs on a worker thread so the GUI paints progress
    # ==================================================================
    def startup(self) -> None:
        """Bring the whole machine up, in the only order that works.

        Order matters and is not arbitrary:
          link  -> we must know the board is there before moving anything
          cameras -> SEQUENTIALLY, and the exposure lock happens before the
                     streaming handle opens (MSMF ignores exposure writes)
          models -> the 1-2 s shape-compile stall must not land mid-track
          gc     -> only once everything long-lived is allocated
          homing -> tens of seconds of motion, and nothing may track before it
        """
        try:
            self._start_link()
            if not self.sim and (getattr(self.args, "microstep", None) is not None
                                 or config.DYNAMIC_MICROSTEPPING):
                from turret_host.microstepping import apply_to_board
                profile = apply_to_board(self.link, config.MICROSTEP_DIVISOR,
                                         dynamic=config.DYNAMIC_MICROSTEPPING)
                self.log("microstep profile: %s; narrow angular cap %.2f deg/s" %
                         (profile, config.MAX_MOTOR_RATE * config.AXIS_STEP_DEG))
            self._start_telemetry()
            # HOMING FIRST, BEFORE THE CAMERAS AND MODELS.
            #
            # It used to run last, so the panel could show live frames while
            # the platform moved. That trade is wrong when homing is the step
            # that fails: four runs tonight paid ~20 s for cameras and CUDA
            # before `level` refused, and every one of those seconds was spent
            # on a pipeline that was then thrown away. Homing needs only the
            # link, so it can fail in 3 s instead of 25.
            #
            # Nothing can track before homing completes regardless -- the
            # travel limits, the pitch datum and every angle the loop reasons
            # about are quoted from the datum -- so nothing is lost by moving
            # the cameras after it.
            self._run_homing(first=True)
            self._capture_tilt_datum()
            # STEP-LOSS DATUM. Taken here because homing has just put the
            # payload at a pose defined by MEASUREMENT -- pitch from gravity,
            # yaw from the magnetometer reference -- rather than at whatever
            # the step counter happened to believe. Shutdown rewinds to this
            # and reports what did not come back; see step_integrity.py.
            try:
                self._step_datum = step_integrity.capture(
                    self.link, level_ok=self._datum_level_ok)
                if self._step_datum is None:
                    self.log("step check: no datum (board did not report "
                             "position); the end-of-run check will be skipped",
                             "warn")
            except Exception as e:                       # noqa: BLE001
                self._step_datum = None
                self.log("step check: datum failed (%s)" % e, "warn")
            # Homing and datum capture are finished and no setup command owns
            # the serial request/reply stream. Only now may the recorder issue
            # opportunistic `imu fast` probes. Starting it before homing can
            # cross the replies: homing's `state` receives an `IMUF ...` line
            # and aborts even though both commands individually succeeded.
            if self.recorder is not None and not self.sim:
                self.recorder.enable_imu(self.link)
            self._start_cameras()
            self._load_models()
            self._freeze_gc()
            self._build_loop()
            # Only now. Every worker below dereferences a detector, the
            # tracker or the controller, and startup runs on its own thread so
            # the GUI can paint -- starting them any earlier is a race that
            # shows up as an AttributeError on None a few milliseconds in.
            # They run DURING homing so the panel shows live frames while the
            # platform moves; _tracking gates the part that commands motion.
            self.start_threads()
        except BaseException as exc:
            self._status_message = "STARTUP FAILED: %s" % exc
            self.log("startup failed: %s" % exc, "error")
            traceback.print_exc()
            self.progress("startup failed: %s" % exc, 1.0, done=True, failed=True)
            # Do not leave the operator looking at a panel that merely refuses
            # to start. Bring the process down the same way every other exit
            # path does.
            self.shutdown()
            if self.gui is not None:
                self.gui.request_close()
            return

        # CLOSE THE PROGRESS STREAM. The GUI derives `_busy` from it, and
        # `_busy` is what disables START -- correctly, because a platform task
        # owns the motors while it runs.
        #
        # This became necessary when homing moved to the FRONT of startup. It
        # used to be last, so its own done=True was the final report and the
        # panel went idle behind it. Now the camera and model steps report
        # after homing finishes, the last of them says "active", and START
        # stayed greyed out with the machine sitting at READY.
        self.progress("ready", 1.0, done=True)

        self._ready.set()
        self._status_message = "homed -- ready. START to track."
        self.log("READY. Homing complete; tracking is now permitted.", "good")

    # -- link -----------------------------------------------------------
    def _start_link(self) -> None:
        if self.sim:
            self.link = SimulatedLink().start()
            self.log("SIMULATED link (--no-hardware): no board is being commanded.", "warn")
            return
        from turret_host.link import TurretLink
        self.progress("opening the serial link", 0.0)
        self.link = TurretLink(port=self.args.port,
                               on_message=lambda m: self.log("link: %s" % m))
        self.link.start()
        self.log("link up on %s" % self.link.port, "good")

    # -- telemetry ------------------------------------------------------
    def _start_telemetry(self) -> None:
        """Open the flight recorder if --record was given.

        Files and image-writer threads start right after the link so no tracked
        frame is missed. Board IMU probing is deliberately deferred until
        homing and datum capture finish; they share one serial request/reply
        stream and must not race a background probe. Once tracking can start,
        sampling uses link.try_probe so a busy vel stream drops telemetry
        instead of being delayed.
        """
        if not getattr(self.args, "record", None):
            return
        from turret_host.telemetry import FlightRecorder
        out = Path(self.args.record)
        if out.is_dir() or not out.suffix:
            out = out / time.strftime("run_%Y-%m-%d_%H%M%S")
        self.recorder = FlightRecorder(
            out, link=None,
            imu_hz=float(self.args.record_imu_hz),
            on_message=lambda m: self.log(m, "warn"),
        ).start()
        self._run_dir = out
        self.log("FLIGHT RECORDER -> %s (IMU at %.0f Hz, dropped rather than "
                 "delaying a vel)" % (out, self.args.record_imu_hz), "good")

    # -- cameras --------------------------------------------------------
    def _start_cameras(self) -> None:
        if self.sim:
            self.narrow_cam = SyntheticSource("narrow", config.NARROW_SIZE,
                                              config.NARROW_FPS, self.narrow_slot,
                                              seed=1).start()
            self.wide_cam = SyntheticSource("wide", config.WIDE_SIZE,
                                            config.WIDE_FPS, self.wide_slot,
                                            seed=2).start()
            self.log("SYNTHETIC cameras: %dx%d and %dx%d at %d fps."
                     % (config.NARROW_SIZE + config.WIDE_SIZE + (config.NARROW_FPS,)),
                     "warn")
            return

        from turret_host import cameras, step_integrity
        self.progress("identifying cameras", 0.05)
        ident = cameras.identify_cameras()
        self.narrow_index, self.wide_index = ident.narrow_index, ident.wide_index
        for line in ident.report().splitlines():
            self.log(line)

        # Exposure BEFORE the streaming handle exists. MSMF silently ignores
        # exposure writes (set() returns True and nothing changes), so the
        # lock goes through a short DSHOW open; doing it after the capture
        # thread owns the device would just fail quietly.
        self.progress("locking exposure", 0.10)
        for name, index, exposure, gain in (
                ("narrow", self.narrow_index, NARROW_EXPOSURE, NARROW_GAIN),
                ("wide", self.wide_index, WIDE_EXPOSURE, None)):
            res = cameras.lock_exposure(index, exposure, gain=gain)
            level = "good" if res.ok else "warn"
            # Not fatal: the exposure lock is an fps optimisation, not an
            # interlock, and the fps floor below is the real gate. But it is
            # reported either way -- a silent failure here shows up later as
            # "the tracker is bad in a dim room".
            self.log("%s exposure lock: %s" % (name, res.message or
                                               ("ok" if res.ok else "did not take")),
                     level)

        # SEQUENTIALLY. MSMF's source reader does not tolerate being raced:
        # a concurrent second open never becomes ready, and reports a lying
        # ERROR_DEVICE_NOT_CONNECTED.
        self.progress("opening the narrow camera", 0.14)
        self.narrow_cam = cameras.narrow_thread(self.narrow_index, self.narrow_slot).start()
        self.log("narrow camera index %d: %.1f fps measured at startup"
                 % (self.narrow_index, self.narrow_cam.startup_fps), "good")

        self.progress("opening the wide camera", 0.18)
        self.wide_cam = cameras.wide_thread(self.wide_index, self.wide_slot,
                                            fast=self.args.wide_fast).start()
        self.log("wide camera index %d: %.1f fps measured at startup"
                 % (self.wide_index, self.wide_cam.startup_fps), "good")
        # Hand the camera threads to the recorder so session.json can report
        # how many byte-identical re-deliveries each one discarded. Set here
        # rather than at construction because the recorder starts BEFORE the
        # cameras do.
        if self.recorder is not None:
            self.recorder.cameras = {"narrow": self.narrow_cam,
                                     "wide": self.wide_cam}

    # -- models ---------------------------------------------------------
    def _load_models(self) -> None:
        if self.sim and not self.args.real_detect:
            self.detector = SyntheticDetector(self.narrow_cam, seed=11).start()
            self.wide_detector = SyntheticDetector(self.wide_cam, seed=12).start()
            self.face_detector = SyntheticFaceDetector(self.narrow_cam, seed=13).start()
            self.wide_face_detector = SyntheticFaceDetector(self.wide_cam, seed=14).start()
            self.log("SYNTHETIC detectors: ground truth plus noise, no model loaded.", "warn")
            return

        from turret_host.detector import Detector, FaceDetector
        device = "cpu" if self.args.cpu else "cuda:0"
        half = config.YOLO_HALF and not self.args.cpu

        self.progress("loading YOLO weights and building the CUDA context", 0.22)
        self.detector = Detector(device=device, half=half,
                                 warmup_sizes=(config.NARROW_SIZE,)).start()
        self.log("target detector: %s on %s, classes %s"
                 % (self.detector.weights_path, device,
                    [self.detector.model_names.get(i) for i in self.detector.class_ids]),
                 "good")
        if self.detector.missing_classes:
            self.log("weights have no class(es) %s -- detecting the rest"
                     % self.detector.missing_classes, "warn")

        # A second model instance for the wide camera. One ultralytics model
        # driven from two threads is not a documented-safe thing to do, and
        # the failure would be an intermittent wrong-frame detection, which is
        # exactly the kind of bug nobody finds during a demo.
        self.progress("loading the wide-camera detector", 0.28)
        self.wide_detector = Detector(device=device, half=half,
                                      warmup_sizes=(config.WIDE_SIZE,)).start()

        self.progress("loading YuNet (the face interlock)", 0.32)
        # NARROW -- the only face detector that gates the beam. Rotated to
        # upright (the C270 is mounted 90 deg over and YuNet only detects
        # upright faces) and run at NATIVE resolution.
        #
        # max_side is passed explicitly, not left to the default, because this
        # is the beam-gating detector and the value is a safety parameter: a
        # 640 downscale here was measured to halve the range at which a face is
        # seen at all (2.80 m -> 1.40 m), which is a bystander behind the drone
        # holder going undetected. See the table in detector.py.
        self.face_detector = FaceDetector(max_side=None).start()
        # WIDE -- draws an overlay and gates nothing. The wide camera is NOT
        # mounted rotated, so rotating its frames would be wrong; and its faces
        # are small (~22 px at 5 m), so it keeps native resolution, which is
        # what detector.py's cost note argues for.
        self.wide_face_detector = FaceDetector(rotation_deg=0,
                                               max_side=None).start()
        self.log("face model: %s (CPU; narrow pass rotated %d deg CW to upright "
                 "and run at max_side=%s -- YuNet is an UPRIGHT detector and "
                 "the C270 is mounted rotated)"
                 % (self.face_detector.model_path,
                    self.face_detector.rotation_deg,
                    self.face_detector.max_side), "good")
        self.log("SAFETY SCOPE: only NARROW faces gate the laser. Wide faces "
                 "are drawn for the operator -- there is no wide->narrow "
                 "mapping in this stack to put them on the beam path.", "warn")

        # The pipeline check runs at startup and is FATAL: a face detector that
        # cannot find a face in a frame we constructed for it is not going to
        # find one in the room, and every other safety property here is
        # downstream of it.
        ok, detail = self.face_detector.pipeline_self_check()
        if not ok:
            raise RuntimeError(
                "FACE INTERLOCK PIPELINE SELF-CHECK FAILED: %s -- refusing to "
                "start. This is the detector that keeps light off a face."
                % detail)
        self.log("face interlock pipeline self-check: %s" % detail, "good")
        self.log("ROTATION SIGN IS STILL UNPROVEN. The check above is "
                 "self-consistent by construction and cannot validate the "
                 "sense of config.NARROW_ROTATION_DEG. ARM is refused until "
                 "verify_face_interlock() has seen a REAL face through the "
                 "narrow camera -- see STATUS.md, pre-arm checklist item 1.",
                 "warn")

    # -- runtime tuning -------------------------------------------------
    def _freeze_gc(self) -> None:
        """Only after every long-lived allocation exists.

        A 35 ms gc.collect() pause was measured here. At 30 fps that is a
        dropped frame and a velocity discontinuity the filter has to absorb --
        it looks exactly like a tracking fault. gc.freeze() moves everything
        allocated so far (torch, ultralytics, cv2, Tk) into the permanent
        generation so it is never rescanned; gc.disable() stops the cyclic
        collector entirely.

        The cost: reference CYCLES now leak. Refcounting still frees
        everything acyclic, which is every frame buffer and every dataclass on
        the hot path. Run for hours and watch RSS; `--keep-gc` turns this off
        wholesale if a leak ever shows up.
        """
        if self.args.keep_gc:
            self.log("gc left enabled (--keep-gc): expect occasional ~35 ms pauses.", "warn")
            return
        gc.collect()
        gc.freeze()
        gc.disable()
        self._gc_frozen = True
        # 5 ms (the default) means a thread holding the GIL can sit on it for
        # 5 ms while the control thread has a frame to service. 1 ms is a
        # closer fit to a 33 ms budget split across five threads.
        sys.setswitchinterval(0.001)
        self.log("gc frozen and disabled; switch interval 1 ms "
                 "(%d objects made permanent)" % gc.get_freeze_count(), "good")

    # -- loop objects ---------------------------------------------------
    def _build_loop(self) -> None:
        from turret_host import control
        from turret_host.tracker import PixelTracker

        self.tracker = PixelTracker()
        self.interlock = control.LaserInterlock(armed=False)

        jac = control.Jacobian.auto()
        if not jac.is_calibrated and self.sim:
            # A dry run with no Jacobian cannot exercise the control law at
            # all, and the control law is most of what there is to exercise.
            # Clearly labelled so nothing mistakes it for a calibration.
            jac = control.Jacobian(self._simulated_jacobian(),
                                   source="SIMULATED (--no-hardware)")
        self.controller = control.Controller(jacobian=jac)

        if self.controller.ready:
            self.log("Jacobian: %r" % self.controller.jacobian, "good")
        else:
            # Explicit failure at startup over silent degradation at runtime:
            # the loop will track and display, and will refuse to servo.
            self.log("NO JACOBIAN CALIBRATION (looked in %s). Tracking and the "
                     "display will run; the turret will NOT be commanded. Run "
                     "`python -m turret_host.calibrate`."
                     % control.JACOBIAN_PATH, "error")
        self.log("goal pixel: %s" % self.controller.goal_model.source_text,
                 "good" if self.controller.goal_model.is_calibrated else "warn")

    @staticmethod
    def _simulated_jacobian() -> np.ndarray:
        """A plausible differential J for the dry run: both motors move both
        image axes, at roughly the geometric px/step."""
        s = config.NARROW_F_PX * math.radians(config.AXIS_STEP_DEG)
        return np.array([[+0.72 * s, +0.70 * s],
                         [-0.69 * s, +0.71 * s]])

    # -- homing ---------------------------------------------------------
    def _run_homing(self, first: bool = False) -> None:
        """Automatic at startup, and re-runnable from the GUI.

        Tracking is refused until this has completed once: the travel limits,
        the pitch datum and every angle the loop reasons about are quoted from
        the datum, and "wherever it was left" is not a datum.
        """
        self._platform_busy.set()
        # Register whichever thread is running this as THE platform task, so
        # shutdown() can join it before it touches the link. Done here rather
        # than only in on_home() because the automatic startup run happens on
        # the startup thread, which on_home() never sees.
        self._platform_thread = threading.current_thread()
        try:
            if self.sim:
                self.progress("SIMULATED homing -- no motion commanded", 0.0)
                for i, step in enumerate(("gyro bias", "level (pitch datum)",
                                          "yaw sweep", "lash preload", "sethome")):
                    if self._stop.is_set():
                        return
                    self.progress("simulated: %s" % step, (i + 1) / 5.0)
                    time.sleep(0.35)
                self._pose = [0.0, 0.0]
                self._pose_valid = True
                self._pose_t = _now()
                self.progress("SIMULATED homing complete (nothing moved)", 1.0, done=True)
                return

            from turret_host.homing import Homing, PITCH_VERIFY_TOL_DEG

            if getattr(self.args, "fast_home", False):
                self.log("--fast-home is no longer supported and is being "
                         "IGNORED: it skipped the magnetometer sweep, which "
                         "is the only absolute yaw reference on this machine. "
                         "Running the full home.", "warn")

            self.progress("homing: this takes tens of seconds and moves the "
                          "platform", 0.0)
            seq = Homing(self.link, self.gui.progress_callback("HOMING")
                         if self.gui is not None
                         else (lambda text, frac=None: self.progress(text, frac)),
                         # None means "load the stored reference from disk",
                         # which homing.py now does by default. That file has
                         # existed since 2026-09-19 and was never read,
                         # because the only way in was typing the float on the
                         # command line -- so every run swept the motor field
                         # for 20 s and threw the datum away.
                         yaw_reference_lsb=self.args.yaw_reference,
                         # fast= is gone. The magnetometer sweep is the only
                         # absolute yaw reference this machine has and it runs
                         # every time now; homing.py refuses fast outright.
                         # A stale --fast-home is reported and ignored rather
                         # than allowed to fail the whole startup.
                         skip_level=getattr(self.args, 'skip_level', False),
                         # Polled between commands so shutdown() can stop a
                         # 40 s sequence that is holding the link's io lock.
                         cancel=self._platform_cancel.is_set)
            try:
                result = seq.run()
            except Exception as exc:
                # Caught to colour the progress bar red and name the failure,
                # then re-raised unchanged. homing.py raises HomingError;
                # link.py raises LinkError on a refused command and means the
                # same thing to us. Neither is swallowed.
                self.progress("homing FAILED: %s" % exc, 1.0, done=True, failed=True)
                raise

            for line in result.summary().splitlines():
                self.log(line, "good" if result.ok else "error")
            if not result.ok:
                raise RuntimeError("homing did not complete: %s"
                                   % "; ".join(result.messages[-2:]))
            if not result.yaw_homed:
                self.log("yaw is UNHOMED this run (%s). Pitch is on gravity and "
                         "repeatable; yaw is arbitrary until a reference is "
                         "saved -- see homing.py --save-ref." % result.yaw_status,
                         "warn")

            # THE DATUM MUST ACTUALLY BE AT IMU ZERO, not merely recorded.
            #
            # homing computes `pitch_datum_ok` by reading gravity back after
            # the lash preload, and until now NOTHING consumed it -- it was
            # printed inside result.summary() and dropped. That made the
            # end-of-run step check meaningless in the one case it exists for:
            # if `level` settles 3 deg off, the step datum is captured at 3 deg,
            # the rewind returns to 3 deg, and the check reports "the payload
            # returned" while dead reckoning is 3 deg wrong from the first
            # frame. A reference that is not verified is not a reference.
            #
            # AND the post-sethome re-measurement now gates it too. The
            # preload check above runs BEFORE dzero/sethome, so it answers
            # "did the preload come back", not "is the datum we wrote level".
            # Both have to hold.
            self._datum_level_ok = bool(result.pitch_datum_ok
                                        and result.datum_verified)
            self._datum_pitch_deg = float(result.pitch_tilt_deg)
            if not result.datum_verified:
                self.log("datum NOT verified after sethome: %.2f deg from "
                         "vertical. See the homing log above for whether that "
                         "is pitch (re-homing may fix it) or roll (`level` "
                         "cannot correct roll -- level the rig by hand)."
                         % result.datum_residual_deg, "error")
            if result.ref_source == "none":
                self.log("yaw came up ARBITRARY this run: no stored reference. "
                         "One has just been saved, so the NEXT home will be "
                         "absolute.", "warn")
            else:
                self.log("yaw reference %s -- the magnetometer sweep produced "
                         "an absolute datum." % result.ref_source, "good")
            if result.soft_limit_margin_deg:
                self.log("measured soft-limit room from this datum: %s"
                         % ", ".join("%s %.1f deg" % (k, v) for k, v in
                                     sorted(result.soft_limit_margin_deg.items())),
                         "good")
            if not self._datum_level_ok:
                self.log("DATUM NOT AT IMU ZERO: gravity reads %+.2f deg after "
                         "homing (tolerance %.2f). Dead reckoning, the travel "
                         "limits and the end-of-run step check are all quoted "
                         "from this pose, so every one of them now carries that "
                         "offset. Re-run homing on a settled rig before "
                         "trusting a step-loss result."
                         % (result.pitch_tilt_deg, PITCH_VERIFY_TOL_DEG),
                         "error")
            else:
                self.log("datum verified at IMU zero: gravity reads %+.2f deg "
                         "(tolerance %.2f)"
                         % (result.pitch_tilt_deg, PITCH_VERIFY_TOL_DEG),
                         "good")

            # The datum IS (0, 0): homing runs `dzero` before `sethome`, and
            # config's travel limits are quoted from it. This is the only pose
            # fix the control loop ever gets -- see _integrate_pose().
            self._pose = [0.0, 0.0]
            self._pose_valid = True
            self._pose_t = _now()
            self.progress("homed: pitch %+.3f yaw %+.3f (%s)"
                          % (result.datum_pitch_deg, result.datum_yaw_deg,
                             result.yaw_status), 1.0, done=True)

            if self.args.lost_step_check:
                check = seq.lost_step_check()
                if self.gui is not None:
                    self.gui.set_health(check.ok, check.message)
                self.log(check.message, "good" if check.ok else "error")
        finally:
            self._platform_busy.clear()
            if self._platform_thread is threading.current_thread():
                self._platform_thread = None

    # ==================================================================
    #   THREADS
    # ==================================================================
    def start_threads(self) -> None:
        """Called once, from startup(), after every component exists."""
        if self._threads:
            return
        for target, name in ((self._infer_narrow_loop, "infer-narrow"),
                             (self._infer_wide_loop, "infer-wide"),
                             (self._faces_loop, "faces-narrow"),
                             (self._tracker_loop, "tracker"),
                             (self._control_loop, "control"),
                             (self._attitude_loop, "attitude"),
                             (self._display_loop, "display")):
            th = threading.Thread(target=self._guard(target), name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def _guard(self, fn):
        """A thread that dies must take the beam with it.

        No try/except anywhere else in this file swallows anything; this one
        exists because a silently dead inference thread would leave the
        control loop servoing on a frozen estimate, which is the worst
        possible failure of this machine.
        """
        def wrapped():
            try:
                fn()
            except BaseException as exc:
                cause = "%s thread died: %s" % (
                    threading.current_thread().name, exc)
                self.log(cause, "error")
                traceback.print_exc()
                self._tracking.clear()
                # DISARM, like every other interlock failure. A dead worker is
                # not a transient: it will not fix itself, and leaving the
                # interlock armed means the beam is one re-acquire away from
                # being permitted again by a pipeline that has lost a stage.
                if self.interlock is not None:
                    self.interlock.disarm()
                self._safe_state("a worker thread died")
                # LAST, so it is the message that survives. _safe_state() and
                # the control loop's _leave_servo() both write _status_message,
                # so the real cause used to be overwritten twice and the panel
                # ended up reading "stopped: tracking stopped" for a dead
                # detector thread. The beam was off either way, but the
                # operator could not see why -- same class as a frozen banner.
                self._status_message = cause
        return wrapped

    # -- narrow target detection ---------------------------------------
    def _infer_narrow_loop(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            frame, seq = self.narrow_slot.wait(0.1)
            if seq == last_seq or frame is None:
                continue
            last_seq = seq
            result = self.detector.detect(frame)
            self.det_slot.put(DetectedFrame(frame=frame, result=result))

    # -- wide target + face detection (display / situational awareness) --
    def _infer_wide_loop(self) -> None:
        last_seq = 0
        next_t = 0.0
        #: Wide FACE pass runs every config.WIDE_FACE_EVERY_N iterations; the
        #: rest reuse the last result. See that constant for the measurement.
        face_tick = 0
        last_wide_faces: list = []
        while not self._stop.is_set():
            frame, seq = self.wide_slot.wait(0.1)
            if seq == last_seq or frame is None:
                continue
            now = _now()
            if now < next_t:
                continue                    # throttled, not skipped in error
            last_seq = seq
            # Throttled hard while tracking: the wide camera is a search and
            # awareness device, and every millisecond it spends on the GPU is
            # a millisecond the narrow loop is not getting.
            hz = (config.WIDE_SEARCH_FPS_TRACKING
                  if self.tracker.state is TrackState.TRACK
                  else config.WIDE_SEARCH_FPS_SEARCHING)
            next_t = now + 1.0 / max(1.0, float(hz))

            result = self.wide_detector.detect(frame)

            # PUBLISH THE DRONE BOXES IMMEDIATELY.
            #
            # This used to run the wide FACE pass first and publish once, at
            # the end. Measured on this machine, per wide frame:
            #
            #     wide YOLO  (drone, GPU)            17.8 ms
            #     wide YuNet (faces, CPU)            99.9 ms
            #     total before the put()            117.7 ms
            #
            # The face pass was 85% of the latency on the one path gated by
            # freshness -- _tracker_loop refuses the wide fallback when the
            # detection is older than WIDE_FALLBACK_MAX_AGE_S (150 ms). Worse,
            # 117.7 ms exceeds the 33 ms throttle period, so the loop could
            # never reach WIDE_SEARCH_FPS_SEARCHING and free-ran at ~8.5 Hz;
            # by the time the control loop read it the age compounded to a
            # MEASURED 275 ms mean, over the limit on 447 of 447 frames. The
            # fallback never fired once, all night.
            #
            # And it bought nothing: wide faces are drawn for the operator and
            # gate NOTHING -- only narrow faces gate the beam, because there is
            # no wide->narrow mapping on the interlock path. A cosmetic pass
            # was silently disabling the search-and-reacquire path.
            self.wide_det_slot.put(DetectedFrame(frame=frame, result=result))

            # KEEP THE WIDE PIXELS. Before this the recorder saved no wide
            # imagery at all, so a wide-sourced box could never be checked by
            # eye -- and a 228 px box at conf 0.74 in native wide pixels is
            # either the drone at 0.9 m or the operator's torso. Only the frame
            # can say. Placed after the FIRST put so publishing the boxes is
            # never delayed by a picture; offer_wide_frame drops rather than
            # blocks, so the wide loop cannot stall on disk.
            #
            # The wide->narrow mapping is done HERE, not in the recorder: this
            # is where _w2n lives, and the sidecar should carry the SAME mapped
            # box the fallback would have used, at the same assumed range.
            if self.recorder is not None and result is not None:
                try:
                    trip = []
                    for d in result.targets:
                        m = self._w2n.box(d, config.ASSUMED_RANGE_M)
                        trip.append((d.conf, (d.x1, d.y1, d.x2, d.y2),
                                     (m.x1, m.y1, m.x2, m.y2)))
                    self.recorder.offer_wide_frame(
                        frame.image, trip, frame.index, frame.t)
                except Exception:                          # noqa: BLE001
                    pass       # a picture is never worth the detector thread

            # Display-only face pass, AFTER, into a SEPARATE result object.
            # detect_into() mutates what it is given, and the object above has
            # already been handed to another thread -- mutating it now would be
            # a race on a live reader. targets is shared read-only.
            shown = DetectionResult(frame_t=result.frame_t,
                                    frame_index=result.frame_index,
                                    camera=result.camera,
                                    targets=result.targets,
                                    infer_ms=result.infer_ms)
            # THE WIDE FACE PASS, THROTTLED SEPARATELY FROM THE DRONE PASS.
            # ~100 ms of CPU against the drone pass's 21.7 ms of GPU, and it
            # was making the wide loop slower than the 150 ms freshness cap the
            # fallback is gated on. Display-only: nothing on the interlock path
            # reads a wide face (see config.WIDE_FACE_EVERY_N).
            #
            # On a skipped iteration the PREVIOUS face list is reused rather
            # than cleared. Chosen over an empty list because clearing makes
            # the operator's wide overlay strobe at 1-in-N, and a face box up
            # to N-1 frames stale still reads correctly as "a person is over
            # there", while an empty list reads as "nobody is there" -- the
            # more misleading of the two to show a human.
            n_every = max(1, int(getattr(config, "WIDE_FACE_EVERY_N", 1)))
            if face_tick % n_every == 0:
                self.wide_face_detector.detect_into(shown, frame)
                last_wide_faces = list(shown.faces)
            else:
                shown.faces = list(last_wide_faces)
            face_tick += 1
            # BOTH puts happen on EVERY iteration regardless, with identical
            # targets: the tracker's wide fallback and the display must both
            # see every drone box, whatever the face pass did.
            self.wide_det_slot.put(DetectedFrame(frame=frame, result=shown))

    # -- coarse acquisition ---------------------------------------------
    def _coarse_targets(self, mapped, now):
        """Wide boxes to steer on during COARSE, or [] meaning "hold".

        The coarse phase puts the target inside the narrow camera's field and
        then GETS OUT OF THE WAY. It is bounded two ways, and the bounds are
        the whole point of it:

          ARRIVAL -- once the mapped box lands inside the narrow frame the
            approach has done its job. It stops driving and waits for the
            narrow detector. Continuing to steer on a box narrow can already
            see means steering on the WRONG one of two measurements.

          GIVE-UP -- if narrow has not confirmed within COARSE_MAX_S the
            approach stops, logs once, and refuses to restart for
            COARSE_COOLDOWN_S. The 1065 px runaway of 2026-09-20 was exactly
            this phase without a clock: a box mapped ~400 px outside the frame,
            driven at until the measured-attitude envelope stopped the payload
            at 68 deg pointing at the floor.
        """
        if not config.CASCADE_COARSE_ENABLED:
            return mapped                      # legacy substitution behaviour

        if self._coarse_gave_up_at is not None:
            if now - self._coarse_gave_up_at < config.COARSE_COOLDOWN_S:
                return []                      # cooling off: do not steer
            self._coarse_gave_up_at = None

        if self._coarse_t0 is None:
            self._coarse_t0 = now

        # ARRIVED? in_narrow_view's margin EXPANDS the acceptance region, so a
        # NEGATIVE margin requires the box to be that far INSIDE the edge
        # before the approach is called done.
        if any(self._w2n.in_narrow_view(m, float(config.COARSE_ARRIVE_MARGIN_PX))
               for m in mapped):
            return []                          # in frame: hold for narrow

        if now - self._coarse_t0 >= config.COARSE_MAX_S:
            self._coarse_gave_up_at = now
            self._coarse_t0 = None
            self.log("COARSE APPROACH GAVE UP: steered on the wide camera for "
                     "%.1f s and the narrow detector never confirmed. Holding "
                     "%.1f s rather than slewing further -- an unbounded "
                     "approach is what drove the payload into the attitude "
                     "envelope on 2026-09-20."
                     % (config.COARSE_MAX_S, config.COARSE_COOLDOWN_S), "warn")
            return []

        return mapped

    # -- measured attitude ---------------------------------------------
    @staticmethod
    def _gravity_unit(pitch_deg: float, roll_deg: float):
        """Unit gravity direction in the IMU frame, from (pitch, roll).

        The inverse of imu.tilt_deg(): that takes the accelerometer vector to
        two angles, this takes them back to a direction. Going through the
        vector rather than comparing angles is what makes the guard
        axis-agnostic -- see ATTITUDE_MAX_DEG on why that matters here.
        """
        p = math.radians(pitch_deg)
        r = math.radians(roll_deg)
        return (-math.sin(p),
                math.sin(r) * math.cos(p),
                math.cos(r) * math.cos(p))

    @staticmethod
    def _angle_between(a, b) -> float:
        dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        na = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        nb = math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))

    def tilt_from_datum(self, now: float):
        """(degrees, age_s) from the datum attitude, or None if no fresh read.

        None means "no measurement", NEVER "level" -- the caller falls back to
        the dead-reckoned guard rather than assuming anything.
        """
        if self._tilt_datum is None:
            return None
        item, _seq = self.attitude_slot.get()
        if item is None:
            return None
        t, pitch, roll = item[0], item[1], item[2]
        age = now - t
        if not 0.0 <= age <= ATTITUDE_MAX_AGE_S:
            return None
        return (self._angle_between(self._gravity_unit(pitch, roll),
                                    self._tilt_datum), age)

    def tilt_from_vertical(self, now: float):
        """(degrees from TRUE VERTICAL, age_s), or None if no fresh read.

        2026-09-20: the envelope is checked against this, not the datum. The
        mechanical stop is +/-90 from LEVEL; a skip-level datum that is 8 deg
        off would let a datum-relative 85 reach 93 on one side. Still gated on
        `_tilt_datum` so the guard stays inert if the IMU never answered.
        """
        if self._tilt_datum is None:
            return None
        item, _seq = self.attitude_slot.get()
        if item is None:
            return None
        t, pitch, roll = item[0], item[1], item[2]
        age = now - t
        if not 0.0 <= age <= ATTITUDE_MAX_AGE_S:
            return None
        return (self._angle_between(self._gravity_unit(pitch, roll),
                                    (0.0, 0.0, 1.0)), age)

    # -- the pose overlay's solid model --------------------------------
    def _measured_pitch(self, ghost_pitch_deg: float, now: float):
        """`AttitudeStatus` for the pose overlay: where the payload really is.

        **Magnitude is measured, sign is not.** The magnitude is the angle
        between the gravity vector now and the gravity vector at the datum --
        a pure angle between two measured vectors, assuming nothing about how
        the GY-85 is clocked on its plate. The sign comes from the ghost.

        Taking the sign from the ghost is a real limitation and it is stated
        in the HUD. It costs one blind spot: a payload sitting at the exact
        mirror of its commanded pitch reads as agreement. Nothing else does --
        backlash, lost steps, slip and a hand on the head all change the
        magnitude, and the magnitude is what the display is watching. The
        alternative needs the sensor's clocking about the yaw axis, which has
        never been measured; guessing it would put a fifth unverified sign
        constant into a project that has shipped four. What CAN be checked
        independently is the magnitude -- see `_check_pose_scale`.

        **There is no complementary filter here, and §3 of the brief asked for
        one.** The reason is the same: a filter needs the gyro's pitch axis in
        the payload frame, the ITG3205 and the ADXL345 are separate parts with
        independent axis conventions on the GY-85, and that mapping has never
        been checked. A wrong gyro sign does not look like noise -- it
        fabricates smooth, confident motion in the wrong direction, which is
        strictly worse than the wobble it removes. So the gyro is used only
        for its MAGNITUDE, which no mounting convention can corrupt, to mark
        the accelerometer untrustworthy while the head is moving. To do better,
        measure the mapping: command a slow pure-pitch move and regress the
        three gyro axes against d(tilt)/dt.
        """
        item, _seq = self.attitude_slot.get()
        if item is None or self._tilt_datum is None:
            return AttitudeStatus(ok=False, note="no attitude datum")
        t, p_imu, r_imu = item[0], item[1], item[2]
        gyro = item[3] if len(item) > 3 else (0.0, 0.0, 0.0)
        age = now - t
        if not 0.0 <= age <= ATTITUDE_MAX_AGE_S:
            return AttitudeStatus(ok=False, age_s=max(0.0, age), roll_deg=r_imu,
                                  note="attitude stale")

        mag = self._angle_between(self._gravity_unit(p_imu, r_imu), self._tilt_datum)
        rate = math.sqrt(sum(float(g) ** 2 for g in gyro))
        moving = rate > config.POSE_STILL_RATE_DPS
        prev_t, prev = self._pose_filt

        # THIS IS CALLED FASTER THAN THE SAMPLES ARRIVE. `_publish_status` runs
        # at the control loop's ~30 Hz and `attitude_slot` refills at
        # ATTITUDE_SAMPLE_HZ = 10, so the same reading is normally seen three
        # times. Re-filtering it would drag the state towards the raw value
        # three times per sample (and re-running the scale check would count it
        # three times), so a repeat is returned unchanged.
        if prev is not None and t == prev_t:
            return AttitudeStatus(ok=True, pitch_deg=prev, roll_deg=r_imu,
                                  age_s=age, moving=moving, rate_dps=rate,
                                  scale=self.pose_scale())

        pitch = (-1.0 if ghost_pitch_deg < 0.0 else 1.0) * mag
        # Exponential smoothing at the interval we actually got, not the one we
        # asked for: try_probe DROPS a sample rather than delay a motor
        # command, so the spacing is irregular by design.
        if prev is not None and 0.0 < (t - prev_t) < 1.0:
            a = math.exp(-(t - prev_t) / max(1e-3, config.POSE_SMOOTH_TAU_S))
            pitch = a * prev + (1.0 - a) * pitch
        self._pose_filt = (t, pitch)

        # Once per SAMPLE, and only while the accelerometer means anything.
        if not moving:
            self._check_pose_scale(pitch, ghost_pitch_deg)
        return AttitudeStatus(ok=True, pitch_deg=pitch, roll_deg=r_imu, age_s=age,
                              moving=moving, rate_dps=rate,
                              scale=self.pose_scale())

    def _check_pose_scale(self, measured: float, ghost: float) -> None:
        """Track measured tilt per commanded degree, and say when it is off.

        THE SIGN CANNOT BE CHECKED HERE and an earlier draft of this pretended
        it could: `_measured_pitch` takes the sign from the ghost, so a
        measured-vs-commanded sign comparison compares the ghost with itself
        and can never fail. What IS independent is the MAGNITUDE, so that is
        what this watches.

        The ratio is the number AGENT_HANDOFF.md 472 records as 1.072 and
        calls a confirmation of the kinematics. It is not one. Three
        measurements of the same quantity exist -- 1.171 (PLAN.md, 2.0 deg
        probe), 1.072 (`level`), 0.972 (homing, a larger probe) -- and a true
        scale error is a constant multiplier that cannot produce a 20 % spread.
        The trend is monotonic in probe size, which is the signature of an
        ADDITIVE offset, and 0.65-2.3 deg of measured backlash against a 2 deg
        probe is exactly that size.

        So the ratio is worth measuring properly and has never been measured
        properly. This does it continuously, from angles far enough out that
        lash is a small fraction of the probe -- which is the one-way, large-
        angle measurement that is not blind to scale. `AXIS_STEP_DEG` is
        derived from `DIFFERENTIAL_N` rather than measured, so if N is wrong
        every angle in the system is wrong together and no internal check can
        see it. This one is external: it comes off gravity.
        """
        lim = config.POSE_SCALE_MIN_DEG
        if abs(ghost) < lim or abs(measured) < lim:
            return
        self._pose_scale.append(abs(measured) / abs(ghost))
        if len(self._pose_scale) > config.POSE_SCALE_WINDOW:
            self._pose_scale.pop(0)
        if len(self._pose_scale) < config.POSE_SCALE_WINDOW or self._pose_scale_warned:
            return
        k = sorted(self._pose_scale)[len(self._pose_scale) // 2]
        if abs(k - 1.0) > config.POSE_SCALE_TOL:
            self._pose_scale_warned = True
            self.log("POSE OVERLAY: the payload is measuring %.3f deg of real "
                     "tilt per commanded degree, over %d samples past %.0f deg. "
                     "That is a SCALE error, not backlash -- lash is additive "
                     "and shrinks as a fraction of a larger angle. DIFFERENTIAL_N "
                     "(%.4f) is the only unmeasured constant it could come from, "
                     "and AXIS_STEP_DEG is derived from it, so every logged angle "
                     "is wrong by the same factor."
                     % (k, len(self._pose_scale), lim, config.DIFFERENTIAL_N), "warn")

    def pose_scale(self):
        """Median measured-per-commanded tilt ratio, or None if not enough yet."""
        if len(self._pose_scale) < 5:
            return None
        return sorted(self._pose_scale)[len(self._pose_scale) // 2]

    def _capture_tilt_datum(self) -> None:
        """Record the attitude homing just established as the reference."""
        if self.sim or self.link is None:
            return
        # ONE sample sets the reference for the whole run, so a sample from an
        # ADXL345 still in standby (reads exactly -0.00/+0.00; run_144709's
        # first five samples) would arm the guard against a "level" the
        # payload was not at, silently. Retry past the standby signature and
        # refuse rather than accept it.
        pitch = roll = None
        for _attempt in range(5):
            reply = self.link.try_probe("imu fast", timeout=0.4, lock_wait=0.5)
            m = _IMUF_RE.search(reply or "")
            if not m:
                continue
            p_, r_ = float(m.group(4)), float(m.group(5))
            if p_ == 0.0 and r_ == 0.0:
                self.log("attitude datum sample read exactly 0.00/0.00 (the "
                         "accelerometer standby signature); retrying.", "warn")
                time.sleep(0.25)
                continue
            pitch, roll = p_, r_
            break
        if pitch is None:
            self.log("attitude guard INERT: could not read a datum attitude. "
                     "The travel limits fall back to the dead-reckoned pose, "
                     "which measured 2.2x wrong on 2026-09-19.", "warn")
            return
        self._tilt_datum = self._gravity_unit(pitch, roll)
        self.log("attitude guard armed at the datum (pitch %+.2f, roll %+.2f); "
                 "motion refused past %.0f deg from here, measured by gravity."
                 % (pitch, roll, ATTITUDE_MAX_DEG), "good")

        # How far the datum sits from the payload's own yaw axis -- which the
        # CAD puts parallel to the GY-85's board normal (sensor +Z) to within
        # 0.3 deg, so this angle is the TURRET's tilt on its bench, not a
        # sensor artifact.
        #
        # It matters to the pose overlay specifically. Yawing the head sweeps
        # gravity around a cone of this half-angle, so the overlay's measured
        # pitch -- an angle from the datum vector -- picks up a yaw-dependent
        # error. That error is SECOND ORDER, not first: worst case it is
        # beta^2 / (2 * pitch), in degrees, which for beta = 4.4 is 0.97 deg at
        # 10 deg of payload pitch, 0.29 at 30 and 0.17 at 45. So it is worst
        # near level, where the backlash it might be confused with is also
        # smallest (0.65 deg), and it shrinks as the payload pitches -- the
        # opposite dependence to a scale error, which is how the two are told
        # apart. Levelling the base removes it; nothing in software can.
        self._base_tilt_deg = self._angle_between(self._tilt_datum, (0.0, 0.0, 1.0))
        if self._base_tilt_deg > POSE_BASE_TILT_WARN_DEG:
            self.log("pose overlay: the datum is %.1f deg off the yaw axis, so "
                     "the base is not level. The measured pitch picks up up to "
                     "%.2f deg of yaw-dependent error at 10 deg of payload pitch "
                     "(less as it pitches further). Comparable to the backlash "
                     "near level; level the base to read small gaps there."
                     % (self._base_tilt_deg, self._base_tilt_deg ** 2 / 20.0),
                     "warn")

    def _attitude_loop(self) -> None:
        """Sample measured attitude while servoing, yielding to the vel stream.

        try_probe drops the sample rather than delaying a motor command, so
        this cannot cause the watchdog trips it exists to help prevent.

        AND IT BACKS OFF WHEN THE SENSOR IS GONE. `imu fast` raises OSError on
        the BOARD when the GY-85 is off the bus, and the firmware's error path
        answers that by disabling every axis. Polling at 10 Hz therefore
        disabled the axes ten times a second and spilled error text into the
        next reader's buffer -- which killed the `state` query in startup and
        took the whole run down with it. A monitor must not be able to break
        the machine it is monitoring when its sensor is absent.
        """
        period = 1.0 / ATTITUDE_SAMPLE_HZ
        fails = 0
        complained = False
        while not self._stop.is_set():
            if self.link is not None and not self.sim:
                try:
                    reply = self.link.try_probe("imu fast")
                except Exception:                          # noqa: BLE001
                    reply = None
                m = _IMUF_RE.search(reply or "")
                if m:
                    if complained:
                        self.log("attitude sampling resumed.", "good")
                        complained = False
                    fails = 0
                    # Gyro appended as a fourth element. The two existing
                    # consumers (tilt_from_datum, tilt_from_vertical) slice the
                    # first three, so this cannot reach them. It is here for
                    # the pose overlay's `moving` flag, which needs only |w|.
                    self.attitude_slot.put((_now(), float(m.group(4)),
                                            float(m.group(5)),
                                            (float(m.group(1)), float(m.group(2)),
                                             float(m.group(3)))))
                elif reply and "error" in reply.lower():
                    # The BOARD refused, which means the sensor is gone.
                    # Anything other than backing off makes it worse.
                    fails += 1
                else:
                    # No reply at all is ordinary lock contention, not a fault.
                    pass
            if fails >= ATTITUDE_FAIL_LIMIT:
                if not complained:
                    self.log("attitude guard SUSPENDED: the board refuses "
                             "`imu fast` (%d times). Backing off to %.0f s -- "
                             "polling a missing IMU disables the axes on every "
                             "call. The travel limits fall back to the "
                             "dead-reckoned pose."
                             % (fails, ATTITUDE_BACKOFF_S), "warn")
                    complained = True
                self._stop.wait(ATTITUDE_BACKOFF_S)
            else:
                self._stop.wait(period)

    # -- narrow face detection: THE INTERLOCK --------------------------
    def _faces_loop(self) -> None:
        """Its own thread because YuNet is CPU-only and costs ~40 ms.

        Polls rather than waits: the narrow slot has two consumers and
        Slot.wait() clears a shared event, so a waiter can lose a wakeup to
        the other one. A 4 ms poll is cheaper than that class of bug.
        """
        last_index = -1
        while not self._stop.is_set():
            frame, seq = self.narrow_slot.get()
            if frame is None or frame.index == last_index:
                self._stop.wait(0.004)
                continue
            last_index = frame.index
            faces = self.face_detector.detect(frame)
            infer_ms = self.face_detector.last_infer_ms
            self.face_slot.put(FaceReport(t=frame.t, faces=faces, infer_ms=infer_ms))

    # -- filter ---------------------------------------------------------
    def _tracker_loop(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            item, seq = self.det_slot.wait(0.1)
            if seq == last_seq or item is None:
                continue
            last_seq = seq
            # Wall time for the coarse-phase timers in _coarse_targets(). It
            # was never defined in this loop: the first wide-only detection on
            # hardware (run_2026-09-20_141701) raised NameError and killed the
            # tracker thread. The unit tests called _coarse_targets directly.
            now = _now()
            # WIDE FALLBACK. The narrow camera's horizontal field is 28.8 deg
            # against the wide camera's 107.9, so a fast target leaves the
            # narrow frame routinely -- and at exactly that moment the only
            # camera still holding it used to be wired to a display panel.
            #
            # Narrow always wins when it has anything at all. The wide box is
            # mapped into narrow coordinates so the EXISTING narrow Jacobian
            # applies unchanged: both cameras are bolted to one payload, so a
            # rotation moves both by the same angle and the mapping is a pure
            # scale-and-rotate. No second Jacobian is needed.
            #
            # The mapped box is tagged source="wide->narrow" and the interlock
            # refuses to fire on it. That is not caution: CAD puts the cameras
            # 60 mm apart, which is 42 px of parallax at 2 m against a 25 px
            # firing tolerance. It can steer; it cannot aim.
            targets = item.result.targets
            # See config.WIDE_FALLBACK_AFTER_MISSES: a live track coasts through
            # a short narrow dropout instead of taking a wide box.
            _need = int(getattr(config, "WIDE_FALLBACK_AFTER_MISSES", 0) or 0)
            # 2026-09-20 run_183021: this gate BLOCKED THE COARSE PHASE. With the
            # tracker in ACQUIRE on a wide box, wide was allowed again only
            # after WIDE_FALLBACK_AFTER_MISSES (3) misses, which is exactly
            # ACQUIRE_MISSES (3), so every approach was one wide box, three
            # misses, SEARCH, repeat: 62 three-frame stints, a command on 59%
            # of rows, and the head never closed a 33 deg gap. The gate is for a
            # NARROW-sourced track coasting through a dropout; while a coarse
            # approach is live (_coarse_t0 set) the wide box IS the track.
            _wide_ok = (self.tracker.state is TrackState.SEARCH
                        or self._coarse_t0 is not None
                        or getattr(self.tracker, "_misses", 0) >= _need)

            if targets:
                # FINE PHASE. Narrow sees it, so the wide box is DROPPED
                # ENTIRELY -- not blended, not weighted. Its ~0.5 deg of
                # parallax/range error is 4% of the coarse budget but HALF the
                # firing budget, so it must not touch this loop. Clearing the
                # timers here is what makes the give-up per-approach rather
                # than per-run.
                self._coarse_t0 = None
                self._coarse_gave_up_at = None
            elif self._tracking.is_set() and _wide_ok:
                wide_item, _ = self.wide_det_slot.get()
                if wide_item is not None and wide_item.result.targets:
                    age = abs(item.result.frame_t - wide_item.result.frame_t)
                    if age <= WIDE_FALLBACK_MAX_AGE_S:
                        mapped = [self._w2n.box(d, config.ASSUMED_RANGE_M)
                                  for d in wide_item.result.targets]
                        targets = self._coarse_targets(mapped, now)
            item.result.targets = targets
            if self._tracking.is_set():
                # The estimate AT the frame time. The control law does not
                # want this one -- it calls estimate_for_control() itself,
                # which leads by LATENCY_S from the moment the command
                # actually goes out.
                est = self.tracker.update(item.result.targets,
                                          item.result.frame_t)
                # FLOW GOES HERE, NOT IN THE CONTROL THREAD.
                #
                # This is the only point that has the RAW narrow frame and the
                # box for the same instant. `item.frame.image` is the
                # unannotated capture -- the display overlay is drawn later,
                # onto a copy, and tracking the overlay would feed the loop its
                # own output. Measured cost 2.78 ms median (cropped LK), inside
                # the 33 ms frame period alongside the 10.7 ms detector.
                # A flow failure is a lost feedforward sample, never a lost
                # detect thread: with FEEDFORWARD_GAIN at 0 it is inert, and
                # even at gain 1 "pure P this frame" is the designed fallback.
                try:
                    ffv = self.flowvel.update(
                        item.frame.image if item.frame is not None else None,
                        (est.box.x1, est.box.y1, est.box.x2, est.box.y2)
                        if est.box is not None else None,
                        item.result.frame_t)
                    self._ff_velocity = ffv
                    self._ff_source = self.flowvel.source
                    self._ff_points = self.flowvel.points
                    self._ff_ms = self.flowvel.ms
                except Exception as exc:                      # noqa: BLE001
                    if not getattr(self, "_flow_complained", False):
                        self._flow_complained = True
                        self.log(f"flow velocity failed ({exc!r}); feedforward "
                                 "source is 'none' until it recovers", "warn")
                    self._ff_velocity = (0.0, 0.0)
                    self._ff_source = "none"
                    self._ff_points = 0
            else:
                # Detections are NOT fed to the filter before the operator
                # presses START. Otherwise the banner reads TRACK during the
                # startup homing run -- a panel claiming to be tracking while
                # the platform is homing itself is the kind of thing that gets
                # believed. Frames and boxes still flow to the display.
                if self.tracker.state is not TrackState.SEARCH:
                    self.tracker.reset()
                est = self.tracker.predict_to(item.result.frame_t)
                # Not tracking: drop the flow history too, so a velocity
                # measured before START cannot survive into the first
                # commanded frame.
                self.flowvel.reset()
                self._ff_velocity = (0.0, 0.0)
                self._ff_source = "none"
                self._ff_points = 0
            self.est_slot.put(TrackedFrame(frame=item.frame, result=item.result,
                                           estimate=est))

    # -- control --------------------------------------------------------
    def _control_loop(self) -> None:
        """One vel per detection, sent right after the detection lands."""
        last_seq = 0
        was_tracking = False
        last_t = None
        while not self._stop.is_set():
            item, seq = self.est_slot.wait(_EST_STALL_S)

            # ---- SAFETY EPOCH FIRST, BEFORE ANY INPUT IS READ ---------------
            # This MUST be the first thing the pass reads, ahead of the
            # tracking gate and ahead of estimate_for_control().
            #
            # It used to be read after both. estimate_for_control() takes the
            # tracker lock, so it is a preemption point: an E-STOP landing
            # between the tracking gate and the epoch read gave this frame the
            # NEW epoch, and send_vel()/set_laser() then could not tell it from
            # a legitimate frame. Measured at 60/60 trials leaving the board
            # with the beam ON, and 7/60 holding a full tracking rate.
            #
            # Read here, the invariant is restored and is now the simple one:
            # any safety action that happens after this line bumps the epoch,
            # so every command this frame issues is rejected as stale. There is
            # no window left, because there is nothing before it to race with.
            epoch = self.link.safety_epoch

            # ---- HEALTH FIRST, ON EVERY PASS --------------------------------
            # Including the passes where NO new estimate arrived. This used to
            # be inside the fresh-estimate path, which meant a dead camera or a
            # dead serial writer left the beam latched on having evaluated the
            # interlock zero times -- with the panel still reading TRACK
            # FIRING. A component that has died is an interlock FAILURE, and it
            # will not fix itself, so it also disarms.
            fault = self._hardware_fault()
            if fault is not None:
                if not self._faulted:
                    self._faulted = True
                    self._fail_interlock(fault, disarm=True)
                was_tracking = False
                last_seq = seq
                continue
            self._faulted = False

            if seq == last_seq or item is None:
                # NO NEW ESTIMATE IS NOT "THE PATH IS CLEAR". It is exactly the
                # same class of fact as "the face report is stale", which
                # _evaluate_laser already refuses to fire on: nothing has told
                # us what the beam is pointing at since the last frame, and the
                # interlock's own vel_fresh and duty-cap checks live INSIDE
                # evaluate(), which is not being called. So the beam goes off
                # and the servo path is left.
                if was_tracking and self._tracking.is_set():
                    self._fail_interlock(
                        "frame pipeline stalled: no new track estimate for "
                        ">%.0f ms" % (_EST_STALL_S * 1000.0), disarm=False)
                    was_tracking = False
                elif was_tracking:
                    self._leave_servo("tracking stopped")
                    was_tracking = False
                else:
                    # Belt and braces: no path out of this loop may leave the
                    # beam on without having evaluated the interlock.
                    self.link.set_laser(False)
                continue
            last_seq = seq

            now = _now()
            if last_t is not None:
                self._note_loop(now - last_t)
            last_t = now

            tracking = self._tracking.is_set() and not self._platform_busy.is_set()
            if not tracking:
                if was_tracking:
                    self._leave_servo("tracking stopped")
                    was_tracking = False
                self._publish_status(item, None, LaserState.DISARMED, None)
                continue
            was_tracking = True

            # ONE read of the tracker, not two. TrackEstimate.state is taken
            # from inside PixelTracker's lock, so it describes the same instant
            # as the numbers beside it. Reading `self.tracker.state` separately
            # let the tracker thread step TRACK->COAST between the two lines,
            # and the interlock was then told track_state=TRACK for an estimate
            # that is a coast extrapolation -- one frame of beam past the "off
            # immediately on COAST" rule.
            est = self.tracker.estimate_for_control(now)
            state = est.state

            # `epoch` was read at the top of this pass, before the tracking gate
            # and before the line above -- see the comment there. It is handed
            # back to the link with the rate and the laser request, so an
            # E-STOP or disarm that lands while this frame is computing wins
            # instead of being overwritten by it.

            out: Optional[ControlOutput] = None
            # COAST still commands. BUILD_SPEC says "vel 0 0 on any transition
            # out of TRACK"; taken literally that makes the tracker's coast
            # ramp dead code. The reading here -- the same one control.py
            # took -- is that the LASER goes off immediately on COAST (it
            # does, and it is the interlock that enforces it), while the
            # mechanism keeps the last velocity for COAST_HOLD_MS and decays
            # over COAST_DECAY_MS, which is what keeps the beam on a target
            # that blinked out for three frames. SEARCH, below, is the real
            # transition out, and that does send vel 0 0.
            # ---- MEASURED ATTITUDE ENVELOPE --------------------------------
            # Purely additive: it can only REFUSE motion the dead-reckoned
            # guard would have allowed, so a missing sample leaves behaviour
            # exactly as it was. `None` is "no measurement", not "level".
            att = self.tilt_from_vertical(now)
            _limit = (ATTITUDE_MAX_DEG - ATTITUDE_REVERSE_HYST_DEG
                      if self._attitude_tripped else ATTITUDE_MAX_DEG)
            if att is not None and att[0] > _limit:
                if not self._attitude_tripped:
                    self._attitude_tripped = True
                    # REVERSE what was commanded over the last
                    # ATTITUDE_REVERSE_S: that is the motion that took the
                    # payload out, whichever axis it was on.
                    da = db = 0.0; t_prev = None
                    for t_c, ra, rb in reversed(self._cmd_hist):
                        if now - t_c > ATTITUDE_REVERSE_S:
                            break
                        dt_c = (t_prev - t_c) if t_prev is not None else 0.033
                        da += ra * dt_c; db += rb * dt_c; t_prev = t_c
                    ba, bb = -da / ATTITUDE_REVERSE_S, -db / ATTITUDE_REVERSE_S
                    peak = max(abs(ba), abs(bb))
                    if peak < 1.0 and self._cmd_hist:
                        ba, bb = -self._cmd_hist[-1][1], -self._cmd_hist[-1][2]
                        peak = max(abs(ba), abs(bb))
                    if peak > 0.0:
                        f = min(1500.0 / peak, max(1.0, ATTITUDE_REVERSE_MIN_RATE / peak))
                        ba, bb = ba * f, bb * f
                    self._reverse_rates = (ba, bb)
                    self.log("ATTITUDE REVERSE: undoing the last %.1f s of "
                             "commanded motion at (%.0f, %.0f) steps/s until "
                             "gravity reads under %.0f deg"
                             % (ATTITUDE_REVERSE_S, ba, bb,
                                ATTITUDE_MAX_DEG - ATTITUDE_REVERSE_HYST_DEG), "warn")
                    self._fail_interlock(
                        "ATTITUDE ENVELOPE: gravity says the payload is %.1f "
                        "deg from VERTICAL (limit %.0f, sample %.0f ms old). "
                        "The dead-reckoned pose says %.1f -- believe gravity. "
                        "Motion stopped before the mechanical stop."
                        % (att[0], ATTITUDE_MAX_DEG, att[1] * 1000.0,
                           abs(self._pose[0]) if self._pose_valid else float("nan")),
                        disarm=False)
                # Drive the reverse; the interlock keeps the beam off while
                # tripped, and tracking resumes once inside by the hysteresis.
                _rv = ControlOutput(rate_a=self._reverse_rates[0],
                                    rate_b=self._reverse_rates[1],
                                    error_u=0.0, error_v=0.0,
                                    goal_u=0.0, goal_v=0.0)
                if self.link.submit(_rv, epoch=epoch) is not False:
                    self._integrate_pose(_rv, now)
                was_tracking = False
                continue
            self._attitude_tripped = False

            # ---- IS THE DEAD-RECKONED POSE STILL EVIDENCE? -----------------
            # The envelope check above already says "believe gravity" when the
            # two disagree enough to stop motion. The SAME disagreement, below
            # that threshold, is still enough to make the dead-reckoned pose
            # useless as an input to the travel-limit derate -- and the derate
            # fires on pose, not on gravity. Left unchecked it freezes an axis
            # on a pose the payload never reached.
            #
            # Gravity's total angle is blind to yaw, so it bounds |true pitch|
            # from BELOW. A dead-reckoned pitch far above it cannot be real.
            # Passing pose=None makes TravelLimits return the command
            # unchanged (`if not self.pose_known: return omega`), so the
            # measured ATTITUDE_MAX_DEG guard above becomes the only limit --
            # which is the one that was right all along.
            pose_trusted = self._pose_valid
            if self._pose_valid and att is not None:
                if abs(self._pose[0]) > att[0] + config.POSE_DR_DISAGREE_MAX_DEG:
                    pose_trusted = False
                    if not self._pose_distrusted:
                        self._pose_distrusted = True
                        self.log(
                            "DEAD-RECKONED POSE REJECTED: it claims pitch "
                            "%.1f deg while gravity measures only %.1f deg of "
                            "total tilt from the datum (tolerance %.0f). The "
                            "travel-limit derate will be skipped until they "
                            "agree -- it would otherwise brake an axis on a "
                            "pose the payload never reached. The measured "
                            "attitude envelope still applies."
                            % (self._pose[0], att[0],
                               config.POSE_DR_DISAGREE_MAX_DEG), "warn")
                elif self._pose_distrusted:
                    self._pose_distrusted = False
                    self.log("dead-reckoned pose agrees with gravity again "
                             "(pitch %.1f vs %.1f deg measured)"
                             % (self._pose[0], att[0]), "good")

            if state in (TrackState.TRACK, TrackState.ACQUIRE, TrackState.COAST):
                if self.controller.ready:
                    out = self.controller.compute(
                        est, now,
                        pose=self._pose if pose_trusted else None,
                        # The feedforward's velocity, from optical flow on the
                        # box patch rather than from differencing box centres.
                        # `est` still supplies the aim point and everything
                        # else; this replaces the feedforward term alone.
                        ff_velocity=self._ff_velocity,
                        ff_source=self._ff_source,
                        ff_points=self._ff_points)
                    # WIDE-SOURCED SLEW CAP -- see config.WIDE_MAX_MOTOR_RATE.
                    # The handoff box is stale and coarse; approach at a rate
                    # the narrow camera can still detect through, so it can
                    # take over. Direction-preserving, like the main clip.
                    #
                    # KEYED ON THE PHASE, NOT ONLY THE BOX SOURCE. Testing
                    # `est.box.source` alone left a hole: on the ACQUIRE rows
                    # immediately after a wide box is CONSUMED, est.box is None
                    # (nothing associated that frame) and the cap silently
                    # stopped applying -- while the turret was still slewing on
                    # a coarse-derived estimate. MEASURED on the first hardware
                    # engagement, run_2026-09-20_142026: five such rows
                    # commanded 1358, 1358, 1182, 1182, 1165 steps/s against a
                    # 600 cap. `_coarse_t0 is not None` means the coarse phase
                    # is live, which is the condition the cap is actually FOR.
                    #
                    # Read across threads (_tracker_loop writes it, this loop
                    # reads it). That is safe for a cap: a stale read can only
                    # make it apply when it need not, never the reverse, and it
                    # is cleared the instant narrow acquires.
                    cap = float(getattr(config, "WIDE_MAX_MOTOR_RATE", 0) or 0)
                    _coarse_live = self._coarse_t0 is not None
                    _wide_box = (est.box is not None
                                 and getattr(est.box, "source", "narrow") != "narrow")
                    if cap > 0 and (_coarse_live or _wide_box):
                        peak = max(abs(out.rate_a), abs(out.rate_b))
                        if peak > cap:
                            f = cap / peak
                            out = _dc_replace(out, rate_a=out.rate_a * f,
                                              rate_b=out.rate_b * f,
                                              saturated=True)
                    # Send every frame, right after the detection lands: the
                    # Pico holds the rate between frames and that is what
                    # keeps the beam moving smoothly at 30 fps.
                    self._cmd_hist.append((now, float(out.rate_a), float(out.rate_b)))
                    if self.link.submit(out, epoch=epoch) is False:
                        # A safety action landed mid-frame; the rate was
                        # dropped on purpose. Do not integrate a pose from a
                        # command the board never got, and do not let this
                        # frame's laser decision stand either.
                        self.link.set_laser(False)
                        self._publish_status(item, None, LaserState.OFF, est)
                        was_tracking = False
                        continue
                    self._integrate_pose(out, now)
            if state is TrackState.SEARCH and self.link.servoing:
                # Nothing to chase. `vel 0 0` on any transition out of TRACK.
                self._leave_servo("SEARCH")

            laser = self._evaluate_laser(state, out, now, epoch, est)
            self._publish_status(item, out, laser, est)

            # Flight recorder. Last in the pass so it observes the decision
            # rather than sitting in front of it, and it can never raise into
            # this thread -- log_control() swallows and disables itself.
            if self.recorder is not None:
                # KEEP THE PIXELS FOR ANYTHING THE TRACKER LOCKED ONTO.
                # The numbers say a box was at (1002, 165), conf 0.48, 272 px
                # across; only the image says whether that was the drone.
                # EVERY FRAME, NOT ONLY THE ONES THAT LOCKED.
                #
                # This used to be gated on `est.box is not None`, so the
                # recorder kept exactly the frames where the tracker had
                # ALREADY SUCCEEDED and threw away every frame where it lost
                # the target -- the precise subset any failure analysis needs.
                # run_140019 has 43 JPEGs against 215 control rows for that
                # reason. A miss is evidence; it is often the only evidence.
                if item.frame is not None:
                    b = est.box if est is not None else None
                    # THE BOX IS OLDER THAN THE IMAGE IT IS DRAWN ON.
                    # `item.frame` is this pass's frame; `est.box` is the last
                    # ASSOCIATED detection, from est.box_t, typically 32-54 ms
                    # earlier. During a fast move that is over 100 px, so the
                    # rectangle sits visibly BEHIND the drone and the picture
                    # looks like a tracking failure that the numbers say is not
                    # there (filter velocity tracks truth at 0.92-0.95, and the
                    # real staleness cost is p90 17 px).
                    #
                    # Two fixes, both here: the age goes in the caption so no
                    # one reads the rectangle as current, and the AIM POINT --
                    # which is current, and is what the turret is actually
                    # driving at -- is passed through to be drawn as a cross.
                    # A frame that cannot be read correctly is worse than no
                    # frame, because it gets believed.
                    if b is not None:
                        box_age_ms = (0.0 if est.box_t is None
                                      else (item.frame.t - est.box_t) * 1000.0)
                        self.recorder.offer_frame(
                            item.frame.image, (b.x1, b.y1, b.x2, b.y2),
                            "%06d t=%.2f %s conf=%.2f %.0fpx BOXAGE=%.0fms"
                            % (item.frame.index, now - self.recorder.t0,
                               getattr(state, "value", state), b.conf,
                               max(b.w, b.h), box_age_ms),
                            aim=(est.u, est.v))
                    else:
                        # NO BOX. Raw only -- there is nothing to annotate, and
                        # the label still carries the index and the state so the
                        # frame joins to control.jsonl. Written with box=None,
                        # which also means no sidecar: an unlabelled frame is a
                        # candidate negative, never a claim about position.
                        self.recorder.offer_frame(
                            item.frame.image, None,
                            "%06d t=%.2f %s NOBOX"
                            % (item.frame.index, now - self.recorder.t0,
                               getattr(state, "value", state)),
                            aim=(None if est is None else (est.u, est.v)))
                wide_rec = None
                wide_item, _wseq = self.wide_det_slot.get()
                if wide_item is not None and wide_item.result is not None:
                    wide_rec = {
                        "n": len(wide_item.result.targets),
                        # GPU cost of the wide detector, now that it runs at
                        # WIDE_SEARCH_FPS_TRACKING = 20 instead of 10.
                        "infer_ms": getattr(wide_item.result, "infer_ms", None),
                        "age_ms": (now - wide_item.result.frame_t) * 1000.0,
                        "conf": max((d.conf for d in wide_item.result.targets),
                                    default=None),
                    }
                self.recorder.log_control(
                    wide=wide_rec, tracker=self.tracker,
                    t=now, state=state, est=est, out=out, laser=laser,
                    interlock=self.interlock, link=self.link,
                    pose=self._pose if self._pose_valid else None,
                    frame_t=item.frame.t if item and item.frame else None,
                    frame_index=item.frame.index if item and item.frame else None,
                    detections=item.result.targets if item and item.result else None,
                    det_frame_t=(item.result.frame_t
                                 if item and item.result else None),
                    # NARROW detector cost, beside wide.infer_ms. Both sides
                    # of any GPU contention have to be visible or a regression
                    # in one is indistinguishable from a throttle in the other.
                    infer_ms=(getattr(item.result, "infer_ms", None)
                              if item and item.result else None),
                    dup_dropped=(self.narrow_cam.duplicates_dropped
                                 if self.narrow_cam is not None else None),
                )

    def _evaluate_laser(self, state: TrackState, out: Optional[ControlOutput],
                        now: float, epoch: Optional[int] = None,
                        est: Optional[TrackEstimate] = None) -> LaserState:
        """Every condition, every frame, before anything else happens.

        `est` carries the frame's drone box, which is the interlock's PERMISSION
        to fire. It defaults to None so an omitted argument grants nothing --
        see LaserInterlock.evaluate on why every one of these defaults is the
        safety property rather than a convenience.
        """
        report, _ = self.face_slot.get()
        face_age = float("inf") if report is None else (now - report.t)

        if report is None or face_age > FACE_MAX_AGE_S:
            # UNKNOWN IS NOT CLEAR. The face pass has not reported recently
            # enough for its answer to describe the room the beam is in, so
            # the beam does not get to be on. Reported as a face inhibit
            # because that is what it is: we cannot say the path is clear.
            self.link.set_laser(False)
            self._status_message = (
                "face detector stale (%s) -- laser inhibited"
                % ("never reported" if report is None else "%.0f ms" % (face_age * 1000)))
            return LaserState.INHIBITED_FACE

        if out is None:
            self.link.set_laser(False)
            self.interlock.evaluate(
                t_now=now, track_state=state, error_px=float("inf"),
                platform_rate_deg_s=float("inf"), last_vel_sent_t=None,
                goal_calibrated=self.controller.goal_model.is_calibrated)
            self._status_message = self.interlock.reason
            return self.interlock.state

        aim = (out.goal_u, out.goal_v)
        target = (out.goal_u + out.error_u, out.goal_v + out.error_v)
        last_vel_t = None
        ms = self.link.ms_since_vel()
        if math.isfinite(ms):
            last_vel_t = now - ms / 1000.0

        laser = self.interlock.evaluate(
            t_now=now,
            track_state=state,
            error_px=self.controller.error_magnitude(out),
            # Commanded rate, not measured: the IMU rate is unavailable while
            # servoing (polling `state` is the contention trap), and the
            # commanded rate is the honest bound on what the platform is doing.
            platform_rate_deg_s=self.controller.platform_rate_deg_s(out),
            last_vel_sent_t=last_vel_t,
            faces=report.faces,
            aim=aim,
            target=target,
            # `aim` above is out.goal_u/goal_v, which is FRAME CENTRE when no
            # goal_pixel.json exists. Frame centre is a defensible fallback for
            # aiming the camera; it is not a fallback that may emit light,
            # because nearest_face_px measures face-to-BEAM clearance from it.
            # At NARROW_F_PX = 1400 one degree of unmeasured laser boresight
            # offset is 24 px against a 120 px margin, of unknown sign, so it
            # can eat the margin rather than add to it.
            goal_calibrated=self.controller.goal_model.is_calibrated,
            # THE PERMISSION. `est.box` is raw detector output and the tracker
            # clears it to None on any frame that fails to associate, so a
            # coasting filter -- which still reports TRACK, and still produces a
            # small error, because it is confidently predicting -- arrives here
            # with no box and gets no beam.
            drone_box=None if est is None else est.box,
            drone_box_t=None if est is None else est.box_t,
            # Travels with the box, so it cannot be read against a different
            # frame's detection. A wide-derived box steers but never fires.
            drone_box_source=("narrow" if est is None or est.box is None
                              else est.box.source),
            # Not report.faces being empty: the TIME the pass reported. An empty
            # list from a blind detector is indistinguishable from an empty room
            # without it. The guard at the top of this method catches the same
            # staleness one layer out; this is the one that binds.
            heads_t=report.t,
        )
        if not self.link.set_laser(laser is LaserState.FIRING, epoch=epoch):
            # The link refused the request -- no live writer, a safety action
            # overtook this frame, the beam latch is set, or the board has
            # refused `laser on` and the writer abandoned it. Either way the
            # beam is not under this frame's control, so the frame does not get
            # to claim FIRING.
            if laser is LaserState.FIRING:
                laser = LaserState.OFF
                if self.link.beam_latched:
                    why = ("a safety action is holding the beam off -- press "
                           "ARM to clear it")
                elif self.link.laser_on_failed:
                    why = ("the board REFUSED `laser on` and it was abandoned; "
                           "check LASER_ENABLED in the FIRMWARE config")
                else:
                    why = self.link.status().last_error
                self._status_message = ("laser request not accepted by the "
                                        "link: %s" % why)
                return laser
        # Cross-check the COMMANDED state against what the board was actually
        # told. Nothing used to read link.laser_on at all, so a beam the board
        # never lit was still reported as FIRING on the panel -- a false
        # readout on the one indicator this machine's safety case rests on.
        # Fail-safe in the beam direction, but the operator was being lied to.
        if laser is LaserState.FIRING and self.link.laser_on_failed:
            self._status_message = ("interlock permits the beam but the board "
                                    "REFUSED `laser on` -- the beam is NOT lit")
            return LaserState.OFF
        self._status_message = self.interlock.reason
        return laser

    def _integrate_pose(self, out: ControlOutput, now: float) -> None:
        """Dead-reckon the payload pose for the travel-limit guard.

        We deliberately never poll `state` while servoing, so this is the only
        pose available between homing and the next idle moment. It drifts --
        that is what the 3 deg soft-stop margin is for -- and it is reset from
        the datum on every homing run.
        """
        if not self._pose_valid:
            return
        dt = now - self._pose_t
        self._pose_t = now
        if dt <= 0.0 or dt > 0.5:
            return                          # a stall; integrating it is fiction
        pitch_rate, yaw_rate = self.controller.limits.axis_rates(
            (out.rate_a, out.rate_b))
        self._pose[0] += pitch_rate * dt
        self._pose[1] += yaw_rate * dt

    def _leave_servo(self, why: str) -> None:
        """vel 0 0, beam off. Called on every transition out of tracking."""
        self.link.set_laser(False)
        self.link.stop()
        self._status_message = "stopped: %s" % why

    # -- liveness -------------------------------------------------------
    def _hardware_fault(self) -> Optional[str]:
        """The one-line reason the beam may not be lit, or None if all is well.

        cameras.py sets CameraThread.error and RETURNS when the handle dies
        (30 consecutive failed grabs -- isOpened() still True, every grab
        fails, the documented C270 failure). Nothing read that field, so the
        trigger for the latched-beam failure was completely silent. Same for
        the serial writer: it exits on a lost port and nothing noticed, so
        every later set_laser(False) had nothing to carry it.
        """
        for cam, label in ((self.narrow_cam, "narrow"), (self.wide_cam, "wide")):
            if cam is None:
                continue
            if cam.error:
                return "%s camera: %s" % (label, cam.error)
            if not cam.is_alive():
                # Only the narrow camera gates the beam, but a wide camera that
                # has died is still a component that has died.
                return ("%s camera thread has exited without an error message"
                        % label)
        if self.link is not None:
            if not self.link.connected:
                return "serial link lost (%s)" % self.link.status().last_error
            if not getattr(self.link, "laser_path_ok", True):
                return ("the serial writer thread is gone: a `laser off` "
                        "cannot reach the board (%s)"
                        % self.link.status().last_error)
        return None

    def _fail_interlock(self, why: str, disarm: bool) -> None:
        """Treat a failure of the machinery itself as an interlock failure.

        Beam off, servo left, and the operator told -- in that order. `disarm`
        for faults that will not fix themselves (dead camera, dead port): a
        re-arm should be a deliberate act once the hardware is back.
        """
        self._status_message = "INTERLOCK FAILURE: %s" % why
        if self.link is not None:
            # safety_veto() bumps the link's epoch as well as dropping the
            # beam, so any frame still in flight loses its rate AND its laser
            # decision.
            self.link.safety_veto()
            self.link.set_laser(False)
            self.link.stop()
        if disarm and self.interlock is not None:
            self.interlock.disarm()
        self._publish_fault_status(why)
        self.log("INTERLOCK FAILURE: %s -- beam off, rates zeroed%s."
                 % (why, ", DISARMED" if disarm else ""), "error")

    def _publish_fault_status(self, why: str) -> None:
        """A status the GUI can believe when there is no frame to build one on.

        The whole hazard here is a panel that keeps reading TRACK FIRING at a
        live-looking fps while the interlock is not running at all, so this
        publishes the real (zero) fps and an explicit OFF.
        """
        status = SystemStatus(
            track=self.tracker.state if self.tracker is not None else TrackState.SEARCH,
            laser=LaserState.OFF,
            narrow_fps=self.narrow_cam.fps if self.narrow_cam else 0.0,
            wide_fps=self.wide_cam.fps if self.wide_cam else 0.0,
            loop_hz=0.0,
            infer_ms=0.0,
            q_level=self.tracker.q_level if self.tracker is not None else 0.0,
            error_px=0.0,
            range_m=0.0,
            face_count=0,
            link=self.link.status() if self.link is not None else LinkStatus(),
            attitude=self._measured_pitch(
                self.link.status().pitch_deg if self.link is not None else 0.0,
                _now()),
            message="INTERLOCK FAILURE: %s" % why,
        )
        self.status_slot.put(status)
        self._last_laser = LaserState.OFF
        self._loop_hz = 0.0

    def _note_loop(self, dt: float) -> None:
        if dt <= 0:
            return
        self._loop_times.append(dt)
        if len(self._loop_times) > 30:
            self._loop_times.pop(0)
        self._loop_hz = len(self._loop_times) / sum(self._loop_times)

    # -- status ---------------------------------------------------------
    def _publish_status(self, item: TrackedFrame, out: Optional[ControlOutput],
                        laser: LaserState, est: Optional[TrackEstimate]) -> None:
        report, _ = self.face_slot.get()
        link_status = self.link.status()
        if (not self.link.servoing and link_status.connected
                and (link_status.pitch_deg or link_status.yaw_deg)):
            # A blocking command reported a real pose while we were idle;
            # prefer it to the dead-reckoned one.
            self._pose = [link_status.pitch_deg, link_status.yaw_deg]
            self._pose_valid = True
            self._pose_t = _now()

        infer_ms = item.result.infer_ms
        if report is not None:
            infer_ms += report.infer_ms

        # The pose overlay. The ghost is the DEAD-RECKONED pose, never
        # `link_status.pitch_deg`: `command()` refuses to run while servoing,
        # so that one is frozen for the whole of a track and the overlay would
        # have drawn a stationary ghost against a moving solid -- which is the
        # picture of a seized mechanism, invented out of a link that is working
        # exactly as designed. See ReckonedPose.
        _t = _now()
        ghost = ReckonedPose(pitch_deg=self._pose[0], yaw_deg=self._pose[1],
                             valid=self._pose_valid,
                             age_s=max(0.0, _t - self._pose_t) if self._pose_t else 0.0)
        # The solid model's sign comes from the ghost, so both must be the same
        # pose -- pass the one the GUI will actually draw.
        attitude = self._measured_pitch(ghost.pitch_deg if ghost.valid
                                        else link_status.pitch_deg, _t)

        status = SystemStatus(
            track=self.tracker.state,
            laser=laser,
            narrow_fps=self.narrow_cam.fps if self.narrow_cam else 0.0,
            wide_fps=self.wide_cam.fps if self.wide_cam else 0.0,
            loop_hz=self._loop_hz,
            infer_ms=infer_ms,
            q_level=self.tracker.q_level,
            error_px=self.controller.error_magnitude(out) if out is not None else 0.0,
            range_m=item.estimate.range_m,
            face_count=len(report.faces) if report is not None else 0,
            link=link_status,
            attitude=attitude,
            pose=ghost,
            message=self._status_message,
        )
        self.status_slot.put(status)
        self._last_laser = laser

    # -- display --------------------------------------------------------
    def _display_loop(self) -> None:
        """Feeds the GUI at DISPLAY_FPS, decoupled from the control loop.

        Deliberately separate: the GUI must never be able to slow the control
        thread down, and the control thread must never be able to stall the
        panel. Both read the newest of everything and neither waits.
        """
        if self.gui is None:
            return
        period = 1.0 / float(config.DISPLAY_FPS)
        last_wide = 0
        while not self._stop.is_set():
            self._stop.wait(period)
            tracked, _ = self.est_slot.get()
            wide_item, wide_seq = self.wide_det_slot.get()

            kwargs = {}
            if tracked is not None:
                kwargs.update(narrow=tracked.frame,
                              narrow_det=self._narrow_overlay(tracked),
                              estimate=tracked.estimate,
                              control=self.controller.last_output)
            if wide_item is not None and wide_seq != last_wide:
                last_wide = wide_seq
                kwargs.update(wide=wide_item.frame, wide_det=wide_item.result)
            if kwargs:
                self.gui.set_frames(**kwargs)

    def _narrow_overlay(self, tracked: TrackedFrame) -> DetectionResult:
        """The narrow DetectionResult the GUI draws, with the faces the
        interlock actually used folded in.

        A fresh object every time: DetectionResult from the inference thread
        is shared with the tracker, and writing the face list into it would be
        a cross-thread mutation of something another thread is reading.
        """
        report, _ = self.face_slot.get()
        return DetectionResult(
            frame_t=tracked.result.frame_t,
            frame_index=tracked.result.frame_index,
            camera=tracked.result.camera,
            targets=tracked.result.targets,
            faces=list(report.faces) if report is not None else [],
            infer_ms=tracked.result.infer_ms,
        )

    # ==================================================================
    #   OPERATOR ACTIONS  (called on the Tk main thread)
    # ==================================================================
    def on_start(self) -> None:
        if not self._ready.is_set():
            # Belt and braces: the GUI already refuses START while a platform
            # task is active (startup homing reports itself as one), so this
            # only catches a startup that failed outright.
            self.log("START refused: homing has not completed. The datum is "
                     "not established, so nothing may be commanded.", "error")
            self._tracking.clear()
            return
        if not self.controller.ready:
            self.log("START: tracking and display only -- no Jacobian, so the "
                     "turret will not be commanded.", "warn")
        self.tracker.reset()
        self._tracking.set()
        self._status_message = "tracking"

    def verify_face_interlock(self, seconds: float = 6.0,
                              need: int = 8) -> bool:
        """PRE-ARM CHECK 1. Prove the face interlock sees a REAL face.

        Stand in front of the narrow camera and run this. It watches live
        frames and, for each one, scores the face detector at the CONFIGURED
        rotation and at the opposite sense. To pass, the configured rotation
        must reach FACE_CONF on at least `need` frames AND must beat the
        opposite sense on the clear majority of them.

        This is the only check in the stack that can settle the sign of
        config.NARROW_ROTATION_DEG, because it is the only one whose input
        comes from the real camera on the real mount instead of from a buffer
        this program built using the constant it is trying to verify. The
        startup pipeline check passes for all four rotations -- verified -- so
        it cannot stand in for this.

        Until this passes, on_arm() refuses and the beam is never permitted.
        """
        if self.face_detector is None:
            self._face_orientation_detail = "no face detector"
            return False
        self.log("FACE INTERLOCK VERIFICATION: put a real face in front of the "
                 "narrow camera for %.0f s. Configured rotation is %d deg."
                 % (seconds, self.face_detector.rotation_deg), "warn")

        # SAVE WHAT IT SCORED, OR THE VERDICT CANNOT BE AUDITED.
        #
        # run_2026-09-20_183021 refused ARM at 0/38 and the frames were gone:
        # the flight recorder runs from launch (imu.jsonl covers the whole
        # session) but the control loop publishes nothing until START, so the
        # 6 s before it leaves no images at all. "0/38" then cannot be told
        # apart from "nobody stood in front of the camera", which is exactly
        # the question that mattered -- and it is a SAFETY gate.
        #
        # Cheap on purpose: 38 JPEGs over 6 s, written on the thread that is
        # already blocking for those 6 s. Nothing here may raise: this runs
        # inside on_arm(), and a failed write must never be the reason the beam
        # is refused or permitted.
        # Imported here, not in the loop: `verdict` below needs it even when
        # the loop never ran (frames = 0), which is precisely the case worth
        # recording.
        import json as _json
        shot_dir = None
        try:
            base = (Path(self._run_dir) if self._run_dir is not None
                    else Path("diag") / "face_checks")
            shot_dir = base / ("face_check_%s" % time.strftime("%H%M%S"))
            shot_dir.mkdir(parents=True, exist_ok=True)
        except Exception:                                  # noqa: BLE001
            shot_dir = None
        scored = []

        deadline = _now() + float(seconds)
        hits = wins = losses = frames = 0
        best_here = best_there = 0.0
        last_seq = None
        while _now() < deadline:
            frame, seq = self.narrow_slot.get()
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            frames += 1
            here, there = self.face_detector.orientation_evidence(frame.image)
            best_here = max(best_here, here)
            best_there = max(best_there, there)
            if here >= config.FACE_CONF:
                hits += 1
            if here >= config.FACE_CONF or there >= config.FACE_CONF:
                if here > there:
                    wins += 1
                elif there > here:
                    losses += 1
            if shot_dir is not None:
                try:
                    import cv2
                    name = "%03d_here%.3f_there%.3f.jpg" % (frames, here, there)
                    cv2.imwrite(str(shot_dir / name), frame.image)
                    scored.append({"n": frames, "frame_index": frame.index,
                                   "here": float(here), "there": float(there),
                                   "file": name})
                except Exception:                          # noqa: BLE001
                    shot_dir = None                        # stop trying, keep checking

        ok = hits >= int(need) and wins > losses
        self._face_orientation_detail = (
            "%d/%d frames detected at the configured %d deg (best conf %.3f); "
            "opposite sense best %.3f, won %d frames vs %d"
            % (hits, frames, self.face_detector.rotation_deg, best_here,
               best_there, losses, wins))
        self._face_orientation_ok = ok
        if shot_dir is not None:
            try:
                with open(shot_dir / "verdict.json", "w", encoding="utf-8") as fh:
                    _json.dump({"passed": bool(ok), "hits": hits, "frames": frames,
                               "need": int(need), "seconds": float(seconds),
                               "rotation_deg": self.face_detector.rotation_deg,
                               "face_conf": config.FACE_CONF,
                               "best_here": float(best_here),
                               "best_there": float(best_there),
                               "wins": wins, "losses": losses,
                               "detail": self._face_orientation_detail,
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
                               "scored": scored}, fh, indent=2)
                self.log("face check frames saved -> %s" % shot_dir, "info")
            except Exception:                              # noqa: BLE001
                pass
        if ok:
            self.log("FACE INTERLOCK VERIFIED: %s" % self._face_orientation_detail,
                     "good")
        else:
            self.log("FACE INTERLOCK NOT VERIFIED: %s. %s"
                     % (self._face_orientation_detail,
                        "The opposite rotation sense detected more faces -- "
                        "config.NARROW_ROTATE_CLOCKWISE is probably backwards."
                        if losses > wins else
                        "No face was reliably detected. Do not arm."), "error")
        return ok

    def on_stop(self) -> None:
        self._tracking.clear()
        self.interlock.disarm()
        self._safe_state("operator STOP")

    def on_arm(self) -> None:
        if getattr(self.args, "tracking_only", False):
            self.log("ARM refused: --tracking-only comparison run", "warn")
            return
        # The orientation self-check is the one thing that proves the face
        # interlock can SEE a face through this build's rotation constant. A
        # wrong NARROW_ROTATION_DEG sign makes YuNet return [] for every frame,
        # which control.py cannot distinguish from an empty room -- the beam
        # would be permitted against a face it is structurally blind to. No
        # proof, no arm.
        # The gate applies where light can actually be emitted. A simulated
        # link commands no board and no laser diode exists to protect anyone
        # from, and the rotation constant it would be checking describes a
        # physical camera mount that is not in the loop -- so requiring a live
        # face there would only stop --no-hardware from exercising the FIRING
        # path, which is how this gate would quietly become untested.
        if self.sim:
            self.log("SIMULATED LINK: face-interlock orientation gate skipped. "
                     "Nothing can emit light. This proves NOTHING about the "
                     "real camera mount.", "warn")
        elif not self._face_orientation_ok:
            # Run the check HERE rather than only refusing. Nothing in the GUI
            # could invoke verify_face_interlock() -- the operator hooks are
            # arm/disarm/estop/home/calibrate/start/stop -- so the gate was
            # unreachable from a running app and ARM could never be satisfied.
            #
            # This does NOT weaken it: the check still has to PASS, on live
            # frames, against a real face, or ARM still returns below. It only
            # makes it runnable at the one moment the operator is guaranteed to
            # be standing in front of the narrow camera: when they press ARM.
            self.log("ARM: running the face-interlock orientation check -- "
                     "STAND IN FRONT OF THE NARROW CAMERA for %.0f s."
                     % FACE_VERIFY_SECONDS, "warn")
            try:
                self.verify_face_interlock(seconds=FACE_VERIFY_SECONDS)
            except Exception as exc:                       # noqa: BLE001
                self.log("ARM REFUSED: the orientation check raised: %s" % exc,
                         "error")
                return
        if not (self.sim or self._face_orientation_ok):
            self.log("ARM REFUSED: the face-interlock orientation check did not "
                     "pass (%s). The beam is not permitted until the face "
                     "detector has been proven to see a REAL face through "
                     "config.NARROW_ROTATION_DEG."
                     % (self._face_orientation_detail or "not run"), "error")
            return
        self.interlock.arm()
        # Release the hard beam latch a prior safety action may have set. This
        # is the ONLY caller of clear_beam_latch(): re-lighting after an E-STOP
        # is a human decision, made here, deliberately.
        if self.link is not None and self.link.clear_beam_latch():
            self.log("beam latch from the last safety action cleared by ARM.",
                     "warn")
        self.log("laser ARMED -- the interlock still decides every frame.", "warn")

    def on_disarm(self) -> None:
        self.interlock.disarm()
        # safety_veto(), not just set_laser(False): the control thread samples
        # interlock.armed inside evaluate() and calls set_laser() a few hundred
        # microseconds later, so a disarm landing in that window used to be
        # followed by one frame of beam. Bumping the epoch makes the link
        # refuse that frame's "on"; latch=True makes the refusal hold no matter
        # what that frame read or when.
        self.link.safety_veto(latch=True)
        self.link.set_laser(False)
        self.log("laser DISARMED", "good")

    def on_estop(self) -> None:
        self._tracking.clear()
        self.interlock.disarm()
        self._safe_state("E-STOP")
        self.log("E-STOP: beam off, rates zeroed, disarmed.", "error")

    def on_clear_estop(self) -> None:
        self.log("E-STOP cleared. Still disarmed; still stopped.", "warn")

    def on_home(self) -> None:
        if self._tracking.is_set():
            self.log("homing refused while tracking. STOP first.", "warn")
            return
        self._start_platform_thread(
            "homing", self._guard(lambda: self._run_homing()))

    def _start_platform_thread(self, name: str, target) -> None:
        """Start a platform task and REMEMBER it.

        Anything that issues blocking console commands has to be joinable by
        shutdown(): `level` alone holds the link's io lock for up to 240 s, and
        link.close() runs on the Tk main thread. An unrecorded thread meant the
        window never closed.
        """
        prev = self._platform_thread
        if prev is not None and prev.is_alive():
            self.log("a platform task is already running; ignoring the "
                     "request.", "warn")
            return
        self._platform_cancel.clear()
        th = threading.Thread(target=target, name=name, daemon=True)
        self._platform_thread = th
        th.start()

    def on_calibrate(self, name: str) -> None:
        if self._tracking.is_set():
            self.log("calibration refused while tracking. STOP first.", "warn")
            return
        if name == "jacobian":
            self._start_platform_thread("calibrate-jacobian",
                                        self._guard(self._calibrate_jacobian))
        elif name == "goal_pixel":
            # Needs an operator with a tape measure placing a target at two
            # known ranges and confirming each. That is a console dialogue,
            # not a button, so it stays where it was built.
            self.log("goal-pixel calibration needs measured ranges from an "
                     "operator: run `python -m turret_host.calibrate` (menu 2) "
                     "with this app stopped -- two owners of one serial port "
                     "is the contention failure the spec calls out.", "warn")
        else:
            self.log("unknown calibration %r" % name, "error")

    def _calibrate_jacobian(self) -> None:
        """Drive calibrate.py's routine with this app's camera and link.

        calibrate.py deliberately does not import cameras.py or link.py -- it
        takes injected callables precisely so the running stack can drive it
        without a second owner of the port.
        """
        if self.sim:
            self.log("Jacobian calibration needs real hardware.", "error")
            return
        from turret_host import calibrate
        import cv2

        self._platform_busy.set()
        self._platform_thread = threading.current_thread()
        progress = (self.gui.progress_callback("CALIBRATE J") if self.gui is not None
                    else (lambda text, frac=None: self.progress(text, frac, title="CALIBRATE J")))
        try:
            self.link.stop()                # the servo path must be idle first

            def grab():
                frame, _ = self.narrow_slot.get()
                if frame is None:
                    raise RuntimeError("no narrow frame to calibrate against")
                return cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

            def move(axis: str, steps: int) -> None:
                reply = self.link.command("move %s %d" % (axis, int(steps)))
                if "STOPPED EARLY" in reply:
                    raise RuntimeError("endstop stopped the move short: %s" % reply.strip())

            # save=True (the default) writes turret_host/calibration/jacobian.json,
            # which is exactly where control.Jacobian.load() now looks.
            calibrate.calibrate_jacobian(
                grab, move, progress=progress,
                steps=int(calibrate.JACOBIAN_STEPS * config.MICROSTEP_DIVISOR / 16),
                preload_steps=int(config.PRELOAD_STEPS))

            from turret_host import control
            self.controller.jacobian = control.Jacobian.load()
            self.log("Jacobian recalibrated: %r" % self.controller.jacobian, "good")
        finally:
            self._platform_busy.clear()
            if self._platform_thread is threading.current_thread():
                self._platform_thread = None
            if self.gui is not None:
                self.gui.set_homing_progress("calibration finished", 1.0, done=True,
                                             title="CALIBRATE J")

    # ==================================================================
    #   SHUTDOWN  -- every exit path lands here
    # ==================================================================
    def _safe_state(self, why: str) -> None:
        """Beam off, rates zero. Ordered: light before motion, always.

        Runs on the Tk MAIN THREAD (E-STOP, STOP, window close) while the
        control thread is mid-frame. safety_veto() first: it bumps the link's
        epoch, so the in-flight frame's `vel` is DROPPED by the writer instead
        of overwriting these zeros in the newest-wins slot, and its FIRING
        decision is refused instead of arriving one frame after the disarm.
        """
        if self.link is None:
            return
        # latch=True: this is a safety action, not a routine transition. The
        # beam stays refused at the link until the operator presses ARM, which
        # is the only thing that calls clear_beam_latch(). Without the latch,
        # a control-loop frame that had already read a consistent epoch could
        # re-light the beam in the window after an E-STOP -- measured at 60/60.
        self.link.safety_veto(latch=True)
        self.link.set_laser(False)
        self.link.stop()
        self._status_message = "safe state: %s" % why

    def _closing_home(self) -> None:
        """Re-home pitch and yaw on the way out. Never raises.

        THE CANCEL FLAG HAS TO BE CLEARED FIRST. shutdown() sets
        _platform_cancel early, to stop a 40 s homing run that is holding the
        link's io lock -- and homing._cmd() checks that predicate before every
        single command, so leaving it set would make this abort on its first
        line while looking like a board fault. Clearing it is safe here: the
        task it was aimed at has already been joined above.

        The cancel it installs instead is a DEADLINE plus a second close
        request, so a window-close cannot be held hostage by a rig that is
        not answering. A homing run is ~40 s; the budget is generous but
        finite, and expiring it is reported rather than swallowed.
        """
        if self.sim or self.link is None:
            return
        if not getattr(self.args, "home_on_close", True):
            self.log("closing home skipped (--no-home-on-close)", "warn")
            return
        # Nothing to return to if we never established a datum this session:
        # a home on the way out would be the FIRST home, on an unattended rig,
        # with the operator already walking away. Refuse rather than start
        # 40 s of unwatched motion.
        if not self._pose_valid:
            self.log("closing home skipped: this session never homed, so "
                     "there is no datum to return to.", "warn")
            return

        deadline = _now() + _CLOSING_HOME_BUDGET_S
        self._platform_cancel.clear()

        def _expired() -> bool:
            return _now() > deadline

        try:
            from turret_host.homing import Homing

            self.log("re-homing pitch and yaw before close (up to %.0f s)"
                     % _CLOSING_HOME_BUDGET_S, "warn")
            seq = Homing(self.link,
                         lambda text, frac=None: self.log("  home: " + text),
                         yaw_reference_lsb=self.args.yaw_reference,
                         skip_level=getattr(self.args, "skip_level", False),
                         cancel=_expired)
            result = seq.run()
            self.log("closed on a verified datum: pitch %+.3f yaw %+.3f (%s), "
                     "%.2f deg from vertical"
                     % (result.datum_pitch_deg, result.datum_yaw_deg,
                        result.yaw_status, result.datum_residual_deg),
                     "good" if result.datum_verified else "warn")
        except BaseException as exc:                       # noqa: BLE001
            # BaseException on purpose: a KeyboardInterrupt here must not skip
            # the port close and camera release below.
            self.log("closing home did not finish (%s). The turret is being "
                     "left where it is -- check it is clear of its frame "
                     "before the next run." % exc, "error")
        finally:
            self._platform_cancel.set()

    def shutdown(self) -> None:
        """Idempotent, and callable from any thread and any exception path."""
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

        self.log("shutting down", "warn")
        self._tracking.clear()
        if self.interlock is not None:
            self.interlock.disarm()


        # Close the recorder FIRST: its IMU thread contends for the same io
        # lock that every step below needs, and a half-written JSONL is worth
        # less than the two seconds this costs.
        if self.recorder is not None:
            try:
                self.recorder.close()
                self.log("flight recorder: %d control rows, %d IMU samples, "
                         "%d locked frames saved (%d dropped), %d probes "
                         "dropped to protect the vel stream -> %s"
                         % (self.recorder.control_rows, self.recorder.imu_rows,
                            self.recorder.frames_saved,
                            self.recorder.frames_dropped,
                            self.recorder.probe_dropped_count, self.recorder.dir),
                         "good")
            except Exception as exc:                       # noqa: BLE001
                self.log("flight recorder close failed: %s" % exc, "warn")
            self.recorder = None

        # 0. A platform task owns the serial port while it runs, and its
        #    commands BLOCK for up to 240 s (`level`) holding link._io_lock.
        #    Everything below touches the link, and shutdown() is called on the
        #    Tk MAIN THREAD from the window-close handler, so this has to be
        #    resolved first or the window simply never closes. link.close() is
        #    bounded independently (_CLOSE_IO_TIMEOUT_S) as the backstop for a
        #    task that ignores the cancel.
        self._platform_cancel.set()
        th = self._platform_thread
        # `is not current_thread()`: a startup failure calls shutdown() from
        # the very thread that was running the automatic homing run, and
        # join()ing the calling thread raises RuntimeError.
        if (th is not None and th is not threading.current_thread()
                and th.is_alive()):
            self.log("waiting up to %.0f s for platform task %r to finish "
                     "(it owns the serial port)"
                     % (_PLATFORM_JOIN_S, th.name), "warn")
            th.join(timeout=_PLATFORM_JOIN_S)
            if th.is_alive():
                self.log("platform task %r did not finish; the motors may "
                         "still be moving and the port park will be skipped."
                         % th.name, "error")
        self._platform_thread = None

        # 1. Beam off and rates zero, while the link still works.
        if self.link is not None:
            try:
                self._safe_state("shutdown")
            except BaseException as exc:
                # Recorded, never swallowed -- and we keep going, because the
                # cameras and the port still have to be released.
                self.log("could not park the motors: %s" % exc, "error")

        # 2. Stop the threads that feed the loop.
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2.0)
            if th.is_alive():
                self.log("thread %s did not join in 2 s" % th.name, "warn")
        self._threads = []

        # 2b. THE END-OF-RUN STEP CHECK. Placement is the whole trick:
        #     it must come AFTER the loop threads are joined and after
        #     _safe_state() has left velocity mode, because link.state() is
        #     SETUP ONLY behind _require_idle and `move` would fight the
        #     velocity loop for the same motors -- and BEFORE step 4 closes
        #     the port, because it needs both to read and to command.
        #
        #     Wrapped completely, and it must stay that way: a shutdown path
        #     that can raise is one that leaves motors enabled, and no
        #     diagnostic is worth that.
        try:
            if getattr(self, "_step_datum", None) is not None and self.link:
                rep = step_integrity.verify(self.link, self._step_datum)
                lines = step_integrity.describe(rep)
                for line in lines:
                    self.log(line,
                             "error" if (rep or {}).get("verdict") == "STEPS LOST"
                             else "info")
                # Onto disk, in the run directory, so the verdict can be read
                # alongside control.jsonl and imu.jsonl during after-test
                # analysis instead of living only in a log panel.
                path = step_integrity.write_report(rep, lines, self._run_dir)
                if path:
                    self.log("step check written -> %s" % path, "good")
        except Exception as exc:                          # noqa: BLE001
            self.log("step check failed (%s) -- continuing shutdown" % exc,
                     "warn")

        # 2c. REHOME BEFORE CLOSING.
        #
        #     Placed after the step check (which is a diagnostic ABOUT the run
        #     that just ended, and a rehome would destroy the datum it
        #     compares against) and before the port closes.
        #
        #     Why home on the way out at all: the machine is left on a
        #     repeatable pose rather than wherever tracking abandoned it, so
        #     the next session starts from a known attitude instead of an
        #     arbitrary one, and -- the part that matters for the payload --
        #     it is left LEVEL and inside its measured soft limits rather than
        #     parked against a stop with the coils released.
        #
        #     Wrapped completely, and it must stay that way. Shutdown is the
        #     one path that cannot be allowed to raise: it still has to close
        #     the port and release the cameras below.
        self._closing_home()

        # 3. Release the cameras. A leaked handle costs the NEXT run:
        #    isOpened() returns True and every read() fails.
        for cam in (self.narrow_cam, self.wide_cam):
            if cam is not None:
                cam.stop()
        self.narrow_cam = self.wide_cam = None

        # 4. Close the port. link.close() turns the laser off, ramps the
        #    rates down and leaves velocity mode before releasing the handle.
        if self.link is not None:
            self.link.close()
            self.link = None

        # 5. Models last: they hold GPU memory, nothing else waits on them.
        for det in (self.detector, self.wide_detector,
                    self.face_detector, self.wide_face_detector):
            if det is not None:
                det.close()
        self.detector = self.wide_detector = None
        self.face_detector = self.wide_face_detector = None

        if self._gc_frozen:
            gc.enable()
            gc.unfreeze()
        self.log("shutdown complete", "good")


# ==========================================================================
#   ENTRY POINT
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m turret_host.app",
        description="Run the turret tracking stack.")
    p.add_argument("--no-hardware", action="store_true",
                   help="synthetic cameras and a simulated link: exercises "
                        "every thread, the filter, the interlock and the GUI "
                        "on a machine with nothing attached")
    p.add_argument("--no-gui", action="store_true",
                   help="headless; Ctrl-C to stop")
    p.add_argument("--real-detect", action="store_true",
                   help="with --no-hardware, run the real YOLO/YuNet models "
                        "against the synthetic frames")
    p.add_argument("--cpu", action="store_true",
                   help="run the target detector on the CPU (explicitly slow; "
                        "it will not make the frame deadline)")
    p.add_argument("--wide-fast", action="store_true",
                   help="wide camera at 1280x720@60 -- a CENTER CROP, not a "
                        "downscale: the horizontal field drops ~108 to ~85 deg")
    p.add_argument("--port", default=None,
                   help="force a serial port instead of resolving by VID/PID")
    micro = p.add_mutually_exclusive_group()
    micro.add_argument("--microstep", type=int, choices=(8, 16), default=16,
                   help="fixed microstepping (default: 16); preserves angular rates and acceleration; requires microprofile firmware")
    micro.add_argument("--dynamic-microstep", action="store_true",
                   help="automatic 1/16 to 1/8, twice the narrow angular speed ceiling; unchanged angular acceleration; requires updated firmware")
    p.add_argument("--tracking-only", action="store_true",
                   help="disable ARM for this run (motor tracking remains available)")
    p.add_argument("--yaw-reference", type=float, default=None,
                   help="override the `my` LSB value that defines the yaw "
                        "datum. Normally unnecessary: homing loads the stored "
                        "reference from turret_host/calibration/ by itself, "
                        "and saves one on the first run that establishes it.")
    p.add_argument("--fast-home", action="store_true",
                   help="DEPRECATED and ignored. It skipped the magnetometer "
                        "yaw sweep, which is the only absolute yaw reference "
                        "this machine has; the sweep now runs every time.")
    p.add_argument("--no-home-on-close", dest="home_on_close",
                   action="store_false", default=True,
                   help="do not re-home pitch and yaw during shutdown. By "
                        "default the turret is left on a verified level datum "
                        "inside its soft limits rather than wherever tracking "
                        "abandoned it.")
    p.add_argument("--lost-step-check", action="store_true",
                   help="run the ripple-phase lost-step check after homing")
    p.add_argument("--run-seconds", type=float, default=None,
                   help="stop and shut down cleanly after N seconds, so the "
                        "exit path can actually be tested")
    p.add_argument("--skip-level", action="store_true",
                   help="take the CURRENT POSE as the pitch datum instead of "
                        "driving to true level. For a machine whose "
                        "accelerometer is not measuring -- `level` refuses to "
                        "set a datum from a sensor reading (0,0,0), which "
                        "stops the whole run. The datum is then arbitrary and "
                        "does not survive a reboot, and the pitch soft stops "
                        "are relative to it. Tracking is unaffected: the "
                        "control loop closes on pixels, not on this datum.")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="flight recorder: write control.jsonl (estimate, pixel "
                        "error, COMMANDED MOTOR RATES, track and laser state) "
                        "and imu.jsonl (gyro and gravity tilt sampled while "
                        "servoing) on one perf_counter clock, for cross-"
                        "referencing against the camera frames")
    p.add_argument("--record-imu-hz", type=float, default=10.0,
                   help="IMU sample rate for --record. Samples are DROPPED "
                        "rather than delaying a vel, so raising this costs "
                        "accuracy of the IMU stream, never watchdog margin.")
    p.add_argument("--keep-gc", action="store_true",
                   help="do not freeze/disable the cyclic collector "
                        "(costs an occasional ~35 ms pause)")
    return p


class _Tee:
    """Mirror a stream to a file, line-buffered, without swallowing anything.

    Deliberately not `logging`: everything here already prints, and the point is
    to keep a verbatim transcript of a run -- including the OpenCV and torch
    chatter that goes straight to the C-level stream and never touches a Python
    logger.
    """

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
            self._fh.flush()          # a crashed run must still leave a log
        except (OSError, ValueError):
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except (OSError, ValueError):
            pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def _open_run_log(log_dir: str):
    """One timestamped file per run, so nothing overwrites anything.

    Every run having its own file is the whole point: a single fixed path gets
    truncated by the next launch, and two overlapping launches interleave into
    a file that is not even valid text. Both happened.
    """
    import datetime as _dt
    # `_os`, not `os`: this module imports os AS _os at the top (the sys.path
    # repair needs it before anything else loads). A bare `os` here is a
    # NameError that only fires when --log-dir is passed -- i.e. the first time
    # anyone tries to capture a log of a run that misbehaved.
    _os.makedirs(log_dir, exist_ok=True)
    path = _os.path.join(log_dir, "app_%s.log"
                         % _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    fh = open(path, "w", encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print("[log] this run -> %s" % path)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "log_dir", None):
        _open_run_log(args.log_dir)
    app = TurretApp(args)

    # Ctrl-C and SIGTERM land in the same shutdown as everything else.
    def _signal(signum, _frame):
        app.log("signal %s" % signum, "warn")
        app._stop.set()
        if app.gui is not None:
            app.gui.request_close()
    signal.signal(signal.SIGINT, _signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal)

    try:
        if args.no_gui:
            return _run_headless(app)
        return _run_gui(app)
    finally:
        app.shutdown()


def _run_gui(app: TurretApp) -> int:
    from turret_host.gui import TurretGUI

    gui = TurretGUI(
        status_slot=app.status_slot,
        on_start=app.on_start,
        on_stop=app.on_stop,
        on_arm=app.on_arm,
        on_disarm=app.on_disarm,
        on_estop=app.on_estop,
        on_clear_estop=app.on_clear_estop,
        on_home=app.on_home,
        on_calibrate=app.on_calibrate,
        on_shutdown=app.shutdown,
    )
    app.gui = gui
    gui.log("app: startup running -- homing must finish before START is "
            "permitted.", "warn")

    # Startup on a worker thread so the panel paints the progress of a
    # 40-second homing run instead of looking hung.
    threading.Thread(target=app.startup, name="startup", daemon=True).start()

    if app.args.run_seconds is not None:
        # Smoke-test hook: close the window from a worker thread through the
        # same path the window manager uses, so the shutdown being exercised
        # is the real one.
        gui.root.after(int(app.args.run_seconds * 1000), gui.request_close)

    # The Tk loop owns the main thread and does not return until the window
    # is gone. gui._on_close() calls on_shutdown (= app.shutdown) itself; the
    # finally in main() makes the other exit paths land there too.
    gui.run()
    return 0


def _run_headless(app: TurretApp) -> int:
    app.startup()
    if not app._ready.is_set():
        return 1
    app.log("headless: tracking started. Ctrl-C to stop. The laser stays "
            "DISARMED -- there is no arm control without the GUI.", "warn")
    app.on_start()
    deadline = (None if app.args.run_seconds is None
                else _now() + float(app.args.run_seconds))
    try:
        while not app._stop.is_set():
            if deadline is not None and _now() >= deadline:
                print("\n--run-seconds elapsed; shutting down.")
                break
            time.sleep(0.5)
            status, seq = app.status_slot.get()
            if status is not None:
                print("%-8s %-18s  n %4.1f/w %4.1f fps  loop %5.1f Hz  "
                      "infer %5.1f ms  err %6.1f px  faces %d"
                      % (status.track.value, status.laser.value,
                         status.narrow_fps, status.wide_fps, status.loop_hz,
                         status.infer_ms, status.error_px, status.face_count))
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nCtrl-C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
