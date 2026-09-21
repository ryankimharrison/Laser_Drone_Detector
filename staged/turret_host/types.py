"""Shared types and the slot primitive. Every module trades in these.

This is the contract between capture, detection, filtering, control, link and
GUI. Do not invent parallel structures in those modules -- extend this one.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import threading

import numpy as np


class TrackState(Enum):
    SEARCH = "SEARCH"
    ACQUIRE = "ACQUIRE"
    TRACK = "TRACK"
    COAST = "COAST"


class LaserState(Enum):
    OFF = "OFF"
    INHIBITED_FACE = "INHIBITED - face"
    INHIBITED_ERROR = "INHIBITED - error"
    INHIBITED_UNSETTLED = "INHIBITED - unsettled"
    # No positive permission to fire: no fresh drone box under the beam, or the
    # head pass has not reported recently enough to veto. Distinct from
    # INHIBITED_FACE, which means a head WAS seen and it was too close. This one
    # means we do not know what the beam is pointing at.
    INHIBITED_NO_LOCK = "INHIBITED - no lock"
    DISARMED = "DISARMED"
    FIRING = "FIRING"


@dataclass
class Frame:
    """One captured frame. `t` is taken at grab() return -- the best available."""
    image: np.ndarray
    t: float
    index: int
    camera: str                      # "narrow" | "wide"


@dataclass
class Detection:
    """A bounding box in the frame that produced it. xyxy, pixels."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    label: str
    # WHICH CAMERA THIS CAME FROM, and it travels with the box on purpose.
    # A wide detection mapped into narrow coordinates carries up to 42 px of
    # parallax error at 2 m -- larger than MAX_ERROR_TO_FIRE_PX -- so it may
    # steer the turret but must never grant permission to fire. Carried on the
    # Detection rather than in a shared variable because the tracker and the
    # control loop are different threads: a flag set beside the data can be
    # read against a box it does not describe.
    source: str = "narrow"

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class DetectionResult:
    """Everything one inference pass produced for one frame."""
    frame_t: float
    frame_index: int
    camera: str
    targets: list = field(default_factory=list)   # list[Detection]
    faces: list = field(default_factory=list)     # list[Detection]
    infer_ms: float = 0.0


@dataclass
class TrackEstimate:
    """Filter output, already predicted forward to `predicted_to_t`."""
    u: float                         # predicted pixel position
    v: float
    du: float                        # predicted pixel velocity, px/s
    dv: float
    predicted_to_t: float
    state: TrackState
    q: float                         # current adaptive process noise
    nis: float                       # last normalised innovation squared
    occluded: bool = False
    box: Optional[Detection] = None
    # When `box` was MEASURED, which is not `predicted_to_t`: u/v are led forward
    # by LATENCY_S while the box is raw detector output from the last frame that
    # actually associated. The interlock needs the measurement time to decide
    # whether the box is still evidence about now. None whenever box is None.
    box_t: Optional[float] = None
    range_m: float = 0.0
    range_source: str = "assumed"    # "assumed" | "size" | "stereo"


@dataclass
class ControlOutput:
    """What the control law decided, and why."""
    rate_a: float                    # motor A (firmware `pan`) steps/s
    rate_b: float                    # motor B (firmware `tilt`) steps/s
    error_u: float                   # pixel error against the goal
    error_v: float
    goal_u: float
    goal_v: float
    saturated: bool = False
    dot_locked: bool = False         # goal came from a detected laser dot
    #: The feedforward input the law actually used, px/s. Equals (est.du,
    #: est.dv) unless config.FEEDFORWARD_SUBTRACT_PLATFORM removed the
    #: platform's own contribution. Logged so the standing-lag claim can be
    #: checked against a recording rather than argued from theory.
    v_target_u: float = 0.0
    v_target_v: float = 0.0
    #: Magnitude of the platform term removed from the feedforward, px/s.
    #: Zero when the flag is off.
    ff_platform_px_s: float = 0.0
    #: Which tier supplied the feedforward velocity this frame:
    #: "flow"  optical flow answered, all three guards passed
    #: "held"  flow was silent; the last value, ramped linearly toward zero
    #: "none"  no trustworthy velocity -- pure P this frame
    #: "est"   no flow source wired up; the old est.du/dv path
    #: Logged so an AAR can tell a lag caused by pure-P frames from one caused
    #: by the control law. See flowvel.py.
    #: The feedforward velocity actually applied, px/s, AFTER the platform
    #: term -- i.e. the target's inertial image velocity. Logged beside
    #: v_target_u/v so an AAR can see both the raw flow and what the law used.
    ff_inertial_u: float = 0.0
    ff_inertial_v: float = 0.0
    ff_source: str = "est"
    #: Surviving LK points behind a "flow" row; 0 on every other tier.
    ff_points: int = 0
    #: The commanded rate in effect when the box was measured, steps/s.
    #: NOT the achieved rate -- VEL_ACCEL ramps at 40000 steps/s^2, so a
    #: command needs 100 ms to be realised. Logged so that gap is measurable.
    omega_at_box_a: Optional[float] = None
    omega_at_box_b: Optional[float] = None
    #: Scale applied to the PROPORTIONAL term (evidence staleness gate), and
    #: the separate `authority` scale applied to the FINAL rate. Logged so a
    #: recording shows where each transitioned.
    p_scale: float = 1.0
    authority: float = 1.0


@dataclass
class LinkStatus:
    connected: bool = False
    port: str = ""
    rtt_ms: float = 0.0
    sent: int = 0
    errors: int = 0
    last_error: str = ""
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class ReckonedPose:
    """Where the controller BELIEVES the payload is: counted steps, integrated.

    **Not `LinkStatus.pitch_deg`.** `link.command()` refuses outright to run
    while servoing -- a second request on the port contends with the vel
    stream, which is the bug `link.py` exists to make impossible -- and
    `_note_pose` only runs inside `command()`. So the pose the board last
    reported is FROZEN for the whole of a track, which is exactly the period
    anything watching the mechanism cares about. `TurretApp._integrate_pose`
    keeps this one live by integrating the commanded rates instead.

    It drifts, by construction; that is what the soft-stop margin is for. A
    drifting number is still the right one to show as the controller's belief,
    because drift IS part of what the controller believes.
    """
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    valid: bool = False
    age_s: float = 0.0               # since the last integration step


@dataclass
class AttitudeStatus:
    """What gravity says about where the payload actually is.

    The only INDEPENDENT witness on this rig. `LinkStatus.pitch_deg` is the
    controller's belief -- it counts steps and trusts them -- so if the
    mechanism slips, stalls or is nudged, that number stays confident and
    wrong. This one comes off the payload's own accelerometer and does not.

    `ok` False means "no measurement", never "level": the reader must fall
    back to showing the dead-reckoned pose alone rather than holding the last
    measured one, which would look exactly like a seized mechanism.

    Only PITCH is measured. An accelerometer sees gravity, and yaw rotates the
    payload about the gravity vector itself at pitch zero, so yaw is
    unobservable there and only weakly observable (as sin(pitch)) anywhere
    this turret works. See `app.TurretApp._measured_pitch`.
    """
    ok: bool = False
    pitch_deg: float = 0.0           # signed, payload frame, relative to the datum
    roll_deg: float = 0.0            # mounting tilt or flex: no joint moves this
    age_s: float = 0.0
    #: The accelerometer is only valid at rest -- any real acceleration adds to
    #: gravity and reads as tilt. True while the gyro says otherwise.
    moving: bool = False
    rate_dps: float = 0.0            # |gyro|, the evidence behind `moving`
    #: Median measured tilt per commanded degree, once enough large-angle
    #: samples exist. 1.0 means the kinematics are right. This is the number
    #: AGENT_HANDOFF.md 472 reports as 1.072 from a probe too small to measure
    #: it -- see app.TurretApp._check_pose_scale. None until it is known.
    scale: Optional[float] = None
    note: str = ""


@dataclass
class SystemStatus:
    """One snapshot for the GUI. Published by the control thread, read-only."""
    track: TrackState = TrackState.SEARCH
    laser: LaserState = LaserState.DISARMED
    narrow_fps: float = 0.0
    wide_fps: float = 0.0
    loop_hz: float = 0.0
    infer_ms: float = 0.0
    q_level: float = 0.0             # 0..1, for the jink bar
    error_px: float = 0.0
    range_m: float = 0.0
    face_count: int = 0
    link: LinkStatus = field(default_factory=LinkStatus)
    attitude: "AttitudeStatus" = field(default_factory=lambda: AttitudeStatus())
    #: The dead-reckoned pose. Read this, not `link.pitch_deg`, for anything
    #: that must stay live while the turret is servoing -- see ReckonedPose.
    pose: "ReckonedPose" = field(default_factory=lambda: ReckonedPose())
    message: str = ""


class Slot:
    """Newest-wins hand-off: a lock and one variable.

    A queue would deliver stale frames that the filter treats as current, so
    every producer/consumer boundary in this stack uses a slot instead. The
    producer overwrites unconditionally; the consumer takes whatever is newest
    and may see the same item twice (check the timestamp) or miss one entirely.
    """

    __slots__ = ("_lock", "_item", "_seq", "_event")

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self._seq = 0
        self._event = threading.Event()

    def put(self, item) -> None:
        with self._lock:
            self._item = item
            self._seq += 1
        self._event.set()

    def get(self) -> Tuple[object, int]:
        """Return (item, seq) without blocking. seq == 0 means nothing yet."""
        with self._lock:
            return self._item, self._seq

    def wait(self, timeout: float = 0.1) -> Tuple[object, int]:
        """Block until something has been put, then return the newest."""
        self._event.wait(timeout)
        self._event.clear()
        return self.get()
