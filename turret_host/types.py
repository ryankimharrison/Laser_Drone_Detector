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
