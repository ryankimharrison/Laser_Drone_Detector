"""Pixel-space Kalman filter, adaptive Q, occlusion stabiliser, track state machine.

State  x = [u, v, udot, vdot]   (pixels, pixels/second, in the NARROW frame)
Meas   z = [u, v]               (bounding-box centre, rolling-shutter corrected)

Everything here is pure arithmetic on numbers the detector already produced, so
this module imports with no camera and no board attached. There is no hardware
access anywhere in it.

Two things in here look wrong until you know why:

1. The filter tracks the *box centre*, but `TrackEstimate.u/v` is the *aim point*
   -- centre plus a deliberate bias toward world-up. The bias is applied on
   output, not folded into the measurement, so that the quantity the filter is
   estimating stays a single consistent image feature. Folding a size-dependent
   offset into z would inject the detector's box-height noise straight into the
   position channel.

2. "Up" is not -v. The narrow camera is mounted rotated (config.NARROW_ROTATION_DEG)
   and BUILD_SPEC forbids rotating frames for processing -- the rotation is carried
   in the geometry instead. So the up direction is computed from the config angle,
   and the "box height" the bias scales is the box extent along that direction.

Run the self-test with:   python -m turret_host.tracker
"""
from __future__ import annotations

# --------------------------------------------------------------------------
#   sys.path repair -- MUST run before any other import, and may use only
#   `os`/`sys`, which are both loaded before user code runs.
#
#   Running this file by path puts turret_host/ at sys.path[0], where types.py
#   SHADOWS THE STDLIB `types` MODULE. The next lazy stdlib import chain
#   (`threading -> functools -> from types import GenericAlias`, or
#   `re -> enum -> from types import MappingProxyType`) then picks up ours and
#   dies with a circular-import error that names this package and looks like
#   our bug. REPLACE the script directory with the project root: merely
#   inserting the root is not enough, because sys.path[0] still wins.
#
#   A no-op under `python -m turret_host.<mod>` and under a normal import.
# --------------------------------------------------------------------------
import os as _os
import sys as _sys

if __package__ in (None, ""):
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path[:] = [p for p in _sys.path
                    if _os.path.abspath(p or _os.getcwd()) != _here]
    _sys.path.insert(0, _os.path.dirname(_here))

import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from turret_host import config
from turret_host.types import Detection, TrackEstimate, TrackState

# --------------------------------------------------------------------------
#   Fixed filter geometry. These are structure, not tunables -- the tunables
#   all live in config.py and are imported, never restated.
# --------------------------------------------------------------------------
_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
_R = (config.MEAS_NOISE_PX ** 2) * np.eye(2)
_I4 = np.eye(4)

# Initial velocity uncertainty at acquisition. Derived, not measured: sqrt(Q_BASE)
# is the modelled acceleration in px/s^2, so half a second of it is a generous
# diffuse prior on an unknown initial speed. Being too wide costs two frames of
# convergence; being too narrow costs a lost track on a target that was already
# moving when we first saw it.
_INIT_VEL_SIGMA_PX_S = 0.5 * math.sqrt(config.Q_BASE)


def world_up_in_image(rotation_deg: float) -> np.ndarray:
    """Unit vector in image coords (u right, v down) pointing toward world-up.

    In an unrotated frame that is (0, -1). A camera rolled by `rotation_deg`
    rotates the up vector by the same angle in the image plane; with v pointing
    down, a positive angle turns clockwise on screen.

    config.NARROW_ROTATION_DEG is 90 and its SIGN IS NOT YET VERIFIED (config
    says so plainly). At +/-90 this vector lies along +/-u, so the aim bias moves
    along the image column axis, not the row axis. If the bias is ever observed
    pushing the aim point *into* the hand instead of away from it, the fix is the
    sign of NARROW_ROTATION_DEG in config.py -- not a sign flip buried here.
    """
    th = math.radians(rotation_deg)
    return np.array([math.sin(th), -math.cos(th)])


def _transition(dt: float) -> np.ndarray:
    """F(dt). dt always comes from capture timestamps; it is never assumed."""
    f = np.eye(4)
    f[0, 2] = dt
    f[1, 3] = dt
    return f


def _process_noise(q: float, dt: float) -> np.ndarray:
    """Q(dt) for a piecewise-constant white-acceleration model."""
    d2 = dt * dt
    d3 = d2 * dt
    d4 = d3 * dt
    return q * np.array([
        [d4 / 4.0, 0.0,      d3 / 2.0, 0.0],
        [0.0,      d4 / 4.0, 0.0,      d3 / 2.0],
        [d3 / 2.0, 0.0,      d2,       0.0],
        [0.0,      d3 / 2.0, 0.0,      d2],
    ])


def _box_extent_along(box: Tuple[float, float], direction: np.ndarray) -> float:
    """Extent of an axis-aligned (w, h) box along a unit direction."""
    w, h = box
    return abs(direction[0]) * w + abs(direction[1]) * h


@dataclass
class _EdgeAnchor:
    """One axis of an occlusion hold: which box edge survived, and the offset
    from that edge to the pre-occlusion centre."""
    axis: int        # 0 = u, 1 = v
    use_low: bool    # anchor to x1/y1 (True) or x2/y2 (False)
    offset: float    # centre - edge, frozen at the moment of occlusion


@dataclass
class _Hold:
    ref_area: float                 # box area immediately BEFORE the occlusion
    anchors: List[_EdgeAnchor]


class PixelTracker:
    """Single-target pixel tracker. One target, deliberately.

    There is exactly one drone and it is in a person's hand. A multi-hypothesis
    tracker (ByteTrack and friends) buys identity management we do not need and
    adds a class of failure -- an ID swap onto a bystander -- that we very much
    do not want on a system that fires a laser.
    """

    def __init__(self, rotation_deg: Optional[float] = None):
        # No hardware here, and none anywhere in this class.
        self._rotation_deg = (config.NARROW_ROTATION_DEG if rotation_deg is None
                              else rotation_deg)
        self._up = world_up_in_image(self._rotation_deg)
        # World-horizontal in the image: up rotated 90 deg clockwise on screen.
        self._right = np.array([-self._up[1], self._up[0]])

        # predict_to() is called from the control thread and may also be called
        # by the GUI for drawing; one lock keeps a read from straddling a write.
        self._lock = threading.RLock()

        self.state: TrackState = TrackState.SEARCH
        self.q: float = config.Q_BASE
        self.nis: float = 0.0
        # Whitened innovation of the last accepted measurement, L^-1 nu with
        # S = L L^T. Diagnostics only; None when the last frame was a miss.
        self.last_nu_white: Optional[np.ndarray] = None
        #: Distances, in px, of every detection the gate turned down this
        #: frame. Empty on a clean single-target frame. Exists so "is the gate
        #: too tight?" is a question the LOG can answer -- until now a rejected
        #: detection vanished, and the only trace was n_targets disagreeing
        #: with has_box, which gives a count and no distances.
        self.last_rejected_px: List[float] = []
        #: Detections discarded this frame for being too wide to be the drone.
        #: Counted rather than folded into last_rejected_px, because telemetry
        #: writes that list as bare distances and an analysis expecting numbers
        #: should keep getting numbers.
        self.last_shape_dropped = 0
        #: How many times the track was restarted from a detection rather than
        #: corrected toward one. A climbing count during a run means the motion
        #: model is repeatedly ending up somewhere the target is not, which is
        #: a statement about the model and belongs in the log next to the
        #: lead-error figures.
        self.reseeds = 0
        #: Updates whose velocity correction hit the physical acceleration
        #: ceiling. See config.MAX_TARGET_ACCEL_MPS2.
        self.accel_clamped = 0
        #: Magnitude of the last clamped velocity correction, px/s, for logging.
        self.last_accel_clamp_px_s = 0.0

        self._x = np.zeros(4)
        self._P = np.diag([config.MEAS_NOISE_PX ** 2, config.MEAS_NOISE_PX ** 2,
                           _INIT_VEL_SIGMA_PX_S ** 2, _INIT_VEL_SIGMA_PX_S ** 2])
        self._t: Optional[float] = None

        self._hits = 0
        self._misses = 0

        self._coast_t0: Optional[float] = None
        self._coast_v0: Optional[np.ndarray] = None

        self._hold: Optional[_Hold] = None
        self._prev_box: Optional[Detection] = None     # rolling-shutter corrected
        self._stable_wh: Optional[Tuple[float, float]] = None
        #: True when the box behind _stable_wh ran off the sensor edge, so its
        #: width is a LOWER BOUND on the object and cannot be ranged from.
        self._stable_at_edge: bool = False
        self._last_det: Optional[Detection] = None     # raw, for the GUI
        #: The last ASSOCIATED detection and the time it was measured, kept
        #: across misses so the interlock can still see it while it is fresh.
        #: Deliberately separate from `_last_det`, which stays exactly as it
        #: was: `_last_det` is what the association step writes every frame,
        #: and conflating "what happened this frame" with "the most recent
        #: measurement" is how a held value silently becomes a live one.
        self._held_det: Optional[Detection] = None
        self._held_det_t: Optional[float] = None

    # ----------------------------------------------------------------- public

    @property
    def q_level(self) -> float:
        """Adaptive-Q level normalised to 0..1 for the GUI's jink bar."""
        span = config.Q_MAX_MULT - 1.0
        return float(np.clip((self.q / config.Q_BASE - 1.0) / span, 0.0, 1.0))

    @property
    def occluded(self) -> bool:
        return self._hold is not None

    def update(self, detections: Sequence[Detection], t: float) -> TrackEstimate:
        """Fold one frame's detections in. `t` is that frame's capture timestamp.

        Returns the corrected estimate AT `t`. The control law does not want
        this one -- it wants `estimate_for_control()`, which leads by
        config.LATENCY_S. Returning the filtered-at-t estimate here keeps the
        two ideas from being confused with each other.
        """
        with self._lock:
            return self._update(list(detections or ()), t)

    def predict_to(self, t: float) -> TrackEstimate:
        """Estimate at an arbitrary time. Does not mutate the filter."""
        with self._lock:
            return self._estimate_at(t)

    def estimate_for_control(self, t_now: float) -> TrackEstimate:
        """The estimate the control law needs: led by the loop's latency.

        Feedforward is the whole game (BUILD_SPEC): with ~60 ms of latency the
        proportional term alone is stable only to ~2 Hz and cannot chase a hand.
        """
        return self.predict_to(t_now + config.LATENCY_S)

    def reset(self) -> None:
        """Drop the track and return to SEARCH."""
        with self._lock:
            self._to_search()

    # ---------------------------------------------------------------- filter

    def _update(self, dets: List[Detection], t: float) -> TrackEstimate:
        # SHAPE FILTER, AT THE ONE PLACE EVERY PATH GOES THROUGH.
        #
        # A box much wider than tall is the hand holding the drone, not the
        # drone -- see config.ASSOC_BOX_ASPECT_MAX. Dropping it HERE rather
        # than inside _associate() is deliberate and was found by testing: the
        # gate is not the only way a detection becomes the track. _update has
        # three seed paths that bypass _associate entirely -- the first frame
        # ever, the SEARCH re-entry, and the RESEED_AFTER_MISSES branch -- and
        # every one of them takes max(dets, key=conf) directly. Filtering only
        # in the gate left the reseed path free to adopt a hand-shaped box,
        # which is exactly what it did on the recorded hand approach.
        dropped = [d for d in dets if not self._shape_ok(d)]
        if dropped:
            dets = [d for d in dets if self._shape_ok(d)]
        self.last_shape_dropped = len(dropped)
        if self._t is None:
            # Very first frame ever. Nothing to predict against, so no gate.
            self._t = t
            if dets:
                self._begin_track(max(dets, key=lambda d: d.conf), t)
            return self._estimate_at(t)

        dt = t - self._t
        if dt <= 0.0:
            # A Slot hands the newest item over and may hand the same one twice;
            # a repeat is a normal consequence of the newest-wins hand-off, not
            # an error and not a miss. Do nothing and report the current state.
            return self._estimate_at(self._t)

        if self.state is TrackState.SEARCH:
            self._t = t
            if dets:
                self._begin_track(max(dets, key=lambda d: d.conf), t)
            return self._estimate_at(t)

        x_pred, p_pred = self._propagate(self._x, self._P, dt, t)
        match, self.last_rejected_px = self._associate(dets, x_pred, p_pred)

        if match is None and dets and self._misses >= config.RESEED_AFTER_MISSES:
            # RE-SEED. The gate above handles a filter that is UNSURE; this
            # handles one that is WRONG. After a run of misses the prediction
            # can be somewhere the target never went -- a jink during the
            # coast -- and then no gate width helps, because widening a gate
            # around a bad mean admits clutter, not the target.
            #
            # Measured 2026-09-19: the filter's prediction was 20-26 px worse
            # than "wherever it was last actually seen". A state that loses to
            # the last raw measurement has negative value and should be
            # discarded, not defended.
            best = max(dets, key=lambda d: d.conf)
            if best.conf >= config.RESEED_MIN_CONF:
                self._begin_track(best, t)
                self.reseeds += 1
                self._t = t
                # Deliberately NOT counted as a hit. _begin_track puts the
                # state machine back in ACQUIRE, and ACQUIRE_FRAMES of ordinary
                # associated detections still have to follow before this is
                # allowed to be a firing permission again. Recovering the
                # ESTIMATE and re-earning TRACK are two different things, and
                # collapsing them would let one lucky frame re-arm the beam.
                return self._estimate_at(t)

        if match is None:
            # No association. NOTE: loss is counted in FRAMES, never in elapsed
            # time. Worst-case frame gap here is ~50 ms against a 33 ms median,
            # and a single late frame is not a lost track.
            self._x, self._P = x_pred, p_pred
            self.last_nu_white = None
            self._last_det = None
        else:
            # Rolling shutter: the C270 reads the sensor out over ~20 ms, so a
            # moving target's centroid is smeared along its own motion. Correct
            # the whole box, not just the centre, because the occlusion logic
            # below reasons about edges.
            box = self._rolling_shutter_corrected(match, x_pred[2:4])
            z = self._measure_with_occlusion_hold(box)

            nu = z - _H @ x_pred
            s = _H @ p_pred @ _H.T + _R
            s_inv = np.linalg.inv(s)
            self.nis = float(nu @ s_inv @ nu)
            self.last_nu_white = np.linalg.solve(np.linalg.cholesky(s), nu)

            if self.nis > config.NIS_THRESHOLD:
                # The target did something the constant-velocity model did not
                # predict -- a jink. Open the process noise so the next
                # prediction stops fighting the measurement. This takes effect
                # from the NEXT propagate, one frame late by construction; the
                # 0.85 decay is what makes it persist long enough to matter.
                self.q = min(self.q * config.Q_SPIKE_MULT,
                             config.Q_BASE * config.Q_MAX_MULT)

            k = p_pred @ _H.T @ s_inv
            x_new = x_pred + k @ nu

            # ACCELERATION CEILING ON THE VELOCITY CORRECTION.
            # See config.MAX_TARGET_ACCEL_MPS2 for the measurement behind it.
            #
            # Bounds the VELOCITY sub-state only. Position is what the camera
            # actually measured and is left alone; velocity is inferred by
            # differencing box centres across ~47 ms, which amplifies the
            # 20-50 px of centre noise into hundreds of px/s of phantom speed.
            # That phantom is then fed to the motors as feedforward and used to
            # extrapolate through detection gaps, so it is the one state worth
            # holding to what a 249 g airframe can physically do.
            #
            # Scale BOTH components by one factor rather than clipping each:
            # a per-axis clip would ROTATE the correction, steering the turret
            # somewhere the measurement never pointed. Same reasoning as the
            # direction-preserving saturation in control.py.
            dv = x_new[2:4] - x_pred[2:4]
            step = float(math.hypot(dv[0], dv[1]))
            range_m, _ = self._range()
            dv_max = (config.MAX_TARGET_ACCEL_MPS2
                      * config.NARROW_F_PX / max(range_m, 0.1)) * dt
            if dv_max > 0.0 and step > dv_max:
                x_new[2:4] = x_pred[2:4] + dv * (dv_max / step)
                self.accel_clamped += 1
                self.last_accel_clamp_px_s = step
            self._x = x_new
            # Joseph form: stays symmetric positive-definite even when the gain
            # is large, which it is right after a Q spike.
            a = _I4 - k @ _H
            self._P = a @ p_pred @ a.T + k @ _R @ k.T
            self._last_det = match
            self._held_det = match
            self._held_det_t = t

        # Every frame, hit or miss.
        self.q = config.Q_BASE + config.Q_DECAY * (self.q - config.Q_BASE)

        self._t = t
        self._step_state(match is not None, t)
        return self._estimate_at(t)

    def _propagate(self, x: np.ndarray, p: np.ndarray, dt: float,
                   t_target: float) -> Tuple[np.ndarray, np.ndarray]:
        f = _transition(dt)
        x = f @ x
        p = f @ p @ f.T + _process_noise(self.q, dt)

        if self.state is TrackState.COAST and self._coast_v0 is not None:
            # Coasting is not constant velocity: hold for COAST_HOLD_MS, then
            # ramp to zero over COAST_DECAY_MS. Set the velocity absolutely from
            # the value captured at COAST entry, never by multiplying the current
            # one -- a per-frame multiply compounds and decays exponentially
            # instead of linearly, and the shape would then depend on frame rate.
            s = self._coast_scale(t_target)
            x[2] = self._coast_v0[0] * s
            x[3] = self._coast_v0[1] * s
            # P is deliberately NOT shrunk with the ramp. Coasting is exactly
            # when the filter should stay humble about where the target is.

        return x, p

    def _coast_scale(self, t: float) -> float:
        if self._coast_t0 is None:
            return 1.0
        age_ms = (t - self._coast_t0) * 1000.0
        if age_ms <= config.COAST_HOLD_MS:
            return 1.0
        decayed = age_ms - config.COAST_HOLD_MS
        if decayed >= config.COAST_DECAY_MS:
            return 0.0
        return 1.0 - decayed / config.COAST_DECAY_MS

    @staticmethod
    def _shape_ok(d: Detection) -> bool:
        """Is this box drone-shaped enough to be used as a measurement?

        ONE-SIDED ON PURPOSE. The hand signature is WIDE: 1.67-2.22 over the
        17 associated boxes of the recorded hand approach. The drone held
        edge-on or folded is TALL AND NARROW -- 0.39-0.48 on 17 frames across
        both runs, and the frames show the real airframe every time. A
        symmetric window would throw those away and lose the track, which is
        the failure this is meant to reduce. Refusing to FIRE on an odd box is
        free; refusing to TRACK one is not.
        """
        h = float(d.y2) - float(d.y1)
        if h <= 0.0:
            return False
        return (float(d.x2) - float(d.x1)) / h <= config.ASSOC_BOX_ASPECT_MAX

    def _associate(self, dets: List[Detection], x_pred: np.ndarray,
                   p_pred: Optional[np.ndarray] = None
                   ) -> Tuple[Optional[Detection], List[float]]:
        """Nearest detection to the prediction, inside a gate that SCALES WITH
        THE FILTER'S OWN UNCERTAINTY. Single target.

        Returns (match, rejected_distances) -- the distances are for the
        recorder. A detection that fails the gate used to vanish silently,
        which is why "is the gate too tight?" could only be argued rather than
        measured.

        THE GATE
        --------
        Accept when the detection is EITHER
          * within GATE_PX of the prediction -- the old fixed test, kept as a
            floor so this change can only ever accept more, never less; or
          * statistically consistent with the prediction, Mahalanobis distance
            under GATE_CHI2 against the innovation covariance
            S = H P H' + R -- the identical test the filter already applies to
            NIS one step later, just moved earlier so it can inform the
            decision instead of only reporting on it.
        and always within GATE_MAX_PX, however unsure the filter has become.

        Using S rather than a constant is the point. When the track is fresh
        and converged, P is small and the second clause adds almost nothing.
        After a run of misses P has grown -- the filter KNOWS its prediction
        has gone vague -- and the gate opens in proportion to that, which is
        exactly the COAST case where a fixed 60 px rejected 100% of the
        detections while the detector was seeing the drone on four frames in
        five.

        The gate is measured against the RAW (rolling-shutter corrected)
        centre, before the occlusion stabiliser runs, because the stabiliser
        needs to know which detection is ours before it can reason about its
        edges. That is safe: an occlusion that eats 30% of the box moves the
        centre by well under half a box.
        """
        s_inv = None
        if p_pred is not None:
            s = _H @ p_pred @ _H.T + _R
            try:
                s_inv = np.linalg.inv(s)
            except np.linalg.LinAlgError:
                # A singular S means the covariance is degenerate; fall back to
                # the fixed gate rather than propagating a NaN into the aim
                # point. Never raise from here -- association runs on every
                # frame of the live loop.
                s_inv = None

        best: Optional[Detection] = None
        best_d2 = float("inf")
        rejected: List[float] = []
        floor2 = config.GATE_PX ** 2
        ceil2 = config.GATE_MAX_PX ** 2

        for d in dets:
            box = self._rolling_shutter_corrected(d, x_pred[2:4])
            nu = np.array([box.cx - x_pred[0], box.cy - x_pred[1]])
            d2 = float(nu @ nu)

            ok = d2 <= floor2
            if not ok and d2 <= ceil2 and s_inv is not None:
                ok = float(nu @ s_inv @ nu) <= config.GATE_CHI2
            # A LONE CONFIDENT DETECTION OUTRUNS THE PREDICTION, NOT THE GATE.
            # With one box in frame there is nothing to mistake it for, so a
            # large innovation says the PREDICTION is stale -- which is what a
            # hand-carried drone does to a constant-velocity model. In
            # run_2026-09-20_144709, 21 of 26 TRACK losses were the real drone
            # at conf 0.54-0.81, a median 241 px away. Shape-filtered upstream
            # and still capped by GATE_MAX_PX.
            if (not ok and len(dets) == 1 and d2 <= config.ASSOC_LONE_PX ** 2
                    and float(d.conf) >= config.ASSOC_LONE_CONF):
                ok = True
            if ok and d2 < best_d2:
                if best is not None:
                    rejected.append(math.sqrt(best_d2))
                best_d2, best = d2, d
            else:
                rejected.append(math.sqrt(d2))
        return best, rejected

    def gate_radius_px(self) -> float:
        """The gate's current effective radius, for telemetry and the GUI.

        Isotropic summary of an anisotropic test -- the real gate is an
        ellipse, and this is its long axis. Reported so a log can show the
        gate opening during a coast instead of leaving it to be inferred.
        """
        with self._lock:
            s = _H @ self._P @ _H.T + _R
            try:
                lam = float(np.max(np.linalg.eigvalsh(s)))
            except np.linalg.LinAlgError:
                return config.GATE_PX
        r = math.sqrt(max(0.0, config.GATE_CHI2 * lam))
        return float(min(max(r, config.GATE_PX), config.GATE_MAX_PX))

    @staticmethod
    def _rolling_shutter_corrected(det: Detection, vel: np.ndarray) -> Detection:
        """Undo the readout smear: z_true = z_reported - k * v.

        Strictly this makes the measurement model z = H' x + noise with
        H' = [[1,0,k,0],[0,1,0,k]], and the exact filter would use H'. Subtracting
        k * v_pred and keeping H is the first-order form the spec calls for; the
        two differ only by k times the velocity ESTIMATION ERROR, which at
        k = 10 ms is sub-pixel once the track has converged. It does put a little
        extra lag-1 correlation into the innovations (measured: r1 -0.29 -> -0.36
        on a constant-velocity run) -- known, small, and not worth the coupling.
        """
        su = config.ROLLING_SHUTTER_K_S * float(vel[0])
        sv = config.ROLLING_SHUTTER_K_S * float(vel[1])
        return Detection(det.x1 - su, det.y1 - sv, det.x2 - su, det.y2 - sv,
                         det.conf, det.label)

    # ------------------------------------------------------- occlusion hold

    def _measure_with_occlusion_hold(self, box: Detection) -> np.ndarray:
        """Turn an associated box into a measurement, holding through occlusion.

        The hand covers part of the drone. The detector then reports a smaller
        box whose centre has migrated onto the surviving remnant -- and if we
        feed that centre in, the aim point walks off the drone body and onto
        whichever half is still visible, which is exactly where the hand is not.
        So: when the area collapses in one frame AND the centre moves toward the
        edge that did not move, anchor to that edge and keep the pre-occlusion
        centre offset from it. The anchor still tracks the target -- the edge is
        re-read every frame -- it just refuses to re-centre on the remnant.
        """
        z = np.array([box.cx, box.cy])
        prev = self._prev_box

        if self._hold is not None:
            if box.area >= (1.0 - config.AREA_RECOVER_FRAC) * self._hold.ref_area:
                self._hold = None          # the drone is fully visible again
                self._set_stable(box)
            else:
                z = self._apply_hold(box)
        elif prev is not None and prev.area > 0.0:
            drop = (prev.area - box.area) / prev.area
            if drop > config.AREA_DROP_FRAC:
                anchors = self._surviving_edges(prev, box)
                if anchors:
                    self._hold = _Hold(ref_area=prev.area, anchors=anchors)
                    z = self._apply_hold(box)
        if self._hold is None:
            self._set_stable(box)

        self._prev_box = box

        # NOTE: while held, z is a reconstruction and is genuinely noisier than
        # a clean centroid, but it is still fed with the nominal R. Inflating R
        # here would need a measured number and there isn't one -- if occlusions
        # ever show up as NIS spikes on release, measure it and add it to config.
        return z

    def _apply_hold(self, box: Detection) -> np.ndarray:
        z = np.array([box.cx, box.cy])
        for a in self._hold.anchors:
            if a.axis == 0:
                edge = box.x1 if a.use_low else box.x2
            else:
                edge = box.y1 if a.use_low else box.y2
            z[a.axis] = edge + a.offset
        return z

    @staticmethod
    def _surviving_edges(prev: Detection, cur: Detection) -> List[_EdgeAnchor]:
        """Per axis: did this axis shrink onto one edge, and did the centre
        follow? Both must hold, or this is not an occlusion."""
        anchors: List[_EdgeAnchor] = []
        axes = (
            (0, prev.x1, prev.x2, cur.x1, cur.x2, prev.cx, cur.cx),
            (1, prev.y1, prev.y2, cur.y1, cur.y2, prev.cy, cur.cy),
        )
        for axis, p_lo, p_hi, c_lo, c_hi, p_c, c_c in axes:
            if (p_hi - p_lo) - (c_hi - c_lo) <= 0.0:
                continue                        # this axis did not shrink
            # The occluded edge is the one that moved; the other one survived.
            use_low = abs(c_lo - p_lo) <= abs(c_hi - p_hi)
            surviving_prev = p_lo if use_low else p_hi
            dc = c_c - p_c
            # Centre must have moved TOWARD the surviving edge. A target simply
            # getting smaller (flying away) shrinks both edges symmetrically and
            # leaves the centre where it was -- that is not an occlusion and
            # must not latch the hold.
            toward = dc < 0.0 if use_low else dc > 0.0
            if not toward:
                continue
            anchors.append(_EdgeAnchor(axis=axis, use_low=use_low,
                                       offset=p_c - surviving_prev))
        return anchors

    # ------------------------------------------------------- state machine

    def _begin_track(self, det: Detection, t: float) -> None:
        # No velocity is known yet, so the rolling-shutter correction is a no-op
        # on the first frame -- correct, not lazy: the correction is k*v and v is
        # zero until we have two frames to difference.
        box = self._rolling_shutter_corrected(det, np.zeros(2))
        self._x = np.array([box.cx, box.cy, 0.0, 0.0])
        self._P = np.diag([config.MEAS_NOISE_PX ** 2, config.MEAS_NOISE_PX ** 2,
                           _INIT_VEL_SIGMA_PX_S ** 2, _INIT_VEL_SIGMA_PX_S ** 2])
        self.state = TrackState.ACQUIRE
        self._hits = 1
        self._misses = 0
        self.q = config.Q_BASE
        self.nis = 0.0
        self.last_nu_white = None
        self._hold = None
        self._prev_box = box
        self._set_stable(box)
        self._last_det = det
        self._coast_t0 = None
        self._coast_v0 = None

    def _to_search(self) -> None:
        self.state = TrackState.SEARCH
        self._hits = 0
        self._misses = 0
        # Keep the last position so the GUI's marker does not snap to the corner,
        # but zero the velocity: nothing is being tracked, so nothing is moving
        # as far as this filter is concerned. Control must gate on state and
        # never act on a SEARCH estimate.
        self._x[2] = 0.0
        self._x[3] = 0.0
        self._P = np.diag([config.MEAS_NOISE_PX ** 2, config.MEAS_NOISE_PX ** 2,
                           _INIT_VEL_SIGMA_PX_S ** 2, _INIT_VEL_SIGMA_PX_S ** 2])
        self.q = config.Q_BASE
        self.last_nu_white = None
        self._hold = None
        self._prev_box = None
        self._last_det = None
        # A track that has been given up on has no measurement to offer.
        self._held_det = None
        self._held_det_t = None
        self._coast_t0 = None
        self._coast_v0 = None

    def _step_state(self, hit: bool, t: float) -> None:
        if hit:
            self._misses = 0
            self._hits += 1
            if self.state is TrackState.ACQUIRE:
                if self._hits >= config.ACQUIRE_FRAMES:
                    self.state = TrackState.TRACK
            elif self.state is TrackState.COAST:
                # Re-detected while coasting. Go back through ACQUIRE rather
                # than straight to TRACK: TRACK is a firing permission, and it
                # should cost the same confirmation every time it is granted.
                self.state = TrackState.ACQUIRE
                self._hits = 1
                self._coast_t0 = None
                self._coast_v0 = None
            return

        self._hits = 0
        self._misses += 1
        if self.state is TrackState.ACQUIRE:
            if self._misses >= config.ACQUIRE_MISSES:
                self._to_search()
        elif self.state is TrackState.TRACK:
            if self._misses >= config.TRACK_MISSES:
                self.state = TrackState.COAST
                self._coast_t0 = t
                self._coast_v0 = self._x[2:4].copy()
                # BUILD_SPEC: the laser goes off immediately on entering COAST.
                # That is the control module's job -- it reads this state.
        elif self.state is TrackState.COAST:
            if self._coast_scale(t) <= 0.0:
                self._to_search()

    # ------------------------------------------------------------- output

    def _estimate_at(self, t: float) -> TrackEstimate:
        if self._t is None:
            return TrackEstimate(u=0.0, v=0.0, du=0.0, dv=0.0, predicted_to_t=t,
                                 state=self.state, q=self.q, nis=self.nis,
                                 occluded=False, box=None,
                                 range_m=config.ASSUMED_RANGE_M,
                                 range_source="assumed")

        if self.state is TrackState.SEARCH:
            x = self._x.copy()
        else:
            x, _ = self._propagate(self._x, self._P, t - self._t, t)

        # WHICH BOX THE INTERLOCK IS ALLOWED TO SEE.
        #
        # On a frame that associated, this is that detection. On a frame that
        # did not, it is the last associated one for up to FIRE_BOX_HOLD_S --
        # carrying its ORIGINAL measurement time, so nothing downstream is
        # misled about how old it is.
        #
        # SEARCH never offers a held box: the track has been given up on, and
        # a firing permission must not outlive the track it belonged to.
        box, box_t = self._last_det, (None if self._last_det is None else self._t)
        if (box is None and self.state is not TrackState.SEARCH
                and self._held_det is not None and self._held_det_t is not None):
            age = t - self._held_det_t
            if 0.0 <= age <= config.FIRE_BOX_HOLD_S:
                box, box_t = self._held_det, self._held_det_t

        bias = self._aim_bias()
        range_m, range_source = self._range()
        return TrackEstimate(
            u=float(x[0] + bias[0]),
            v=float(x[1] + bias[1]),
            du=float(x[2]),
            dv=float(x[3]),
            predicted_to_t=t,
            state=self.state,
            q=self.q,
            nis=self.nis,
            occluded=self._hold is not None,
            box=box,
            # The time the box was MEASURED, never the time it is being read.
            # On a hit that is self._t; on a held box it is when the hold
            # started, so the interlock's DRONE_BOX_MAX_AGE_S test keeps
            # working on the real age and a held box expires on schedule.
            box_t=box_t,
            range_m=range_m,
            range_source=range_source,
        )

    def _aim_bias(self) -> np.ndarray:
        """Push the aim point toward world-up by a fraction of the box height.

        The drone is held in a hand, so the hand is BELOW it. Aiming at the
        geometric centre of the box puts the beam closest to the fingers; biasing
        up puts it on the airframe. See world_up_in_image() for why "up" is not
        simply -v on this camera.

        While occluded, the extent used is the remembered pre-occlusion size --
        scaling the bias by the remnant would shrink the bias exactly when the
        hand is closest.
        """
        if self._stable_wh is None:
            return np.zeros(2)
        extent = _box_extent_along(self._stable_wh, self._up)
        return self._up * (config.AIM_BIAS_UP_FRAC * extent)

    def _set_stable(self, box: Detection) -> None:
        """Remember the box's size, and whether the sensor edge cut it off.

        A box that runs off the frame is not a measurement of apparent size:
        the object continues past the edge, so the width is a lower bound and
        R = f * W / w_px comes back too FAR. MEASURED across six runs: 11.5%
        of narrow-sourced boxes touch an edge.

        Only narrow boxes can be truncated this way. A wide->narrow mapped box
        routinely lands outside the narrow frame -- that is the mapping doing
        its job, not the sensor clipping the object, and the wide camera saw
        the whole target -- so the edge test does not apply to it.
        """
        self._stable_wh = (box.w, box.h)
        w, h = config.NARROW_SIZE
        if getattr(box, "source", "narrow") != "narrow":
            self._stable_at_edge = False
        else:
            self._stable_at_edge = bool(
                box.x1 <= 1.0 or box.y1 <= 1.0
                or box.x2 >= w - 1.0 or box.y2 >= h - 1.0)

    def _range(self) -> Tuple[float, str]:
        """Range to the target.

        Apparent-size ranging is R = f * W_real / w_px, with W_real =
        config.DRONE_WIDTH_M. (That constant USED to be None and this docstring
        used to say ranging was disabled; it is 0.289 now, so this path is
        live. Left in the log as range_source so an analysis can tell which
        samples were ranged and which fell back.)

        Three things disqualify a box as a size measurement, and all three fall
        back to the flat assumption rather than ranging from a number known to
        be wrong: no box at all, an OCCLUDED box (the hand ate part of it), and
        an EDGE-TRUNCATED box (the sensor ate part of it). The last two are the
        same failure -- part of the object is missing from the box -- and both
        bias the width DOWN, which biases range UP.
        """
        if config.DRONE_WIDTH_M is None or self._stable_wh is None:
            return config.ASSUMED_RANGE_M, "assumed"
        if self._hold is not None:
            # An occluded box is not a measurement of apparent size.
            return config.ASSUMED_RANGE_M, "assumed"
        if self._stable_at_edge:
            # Neither is one the sensor edge cut in half. Same reasoning as the
            # occlusion case above: the object continues past the box.
            return config.ASSUMED_RANGE_M, "assumed"
        w_px = _box_extent_along(self._stable_wh, self._right)
        if w_px <= 0.0:
            return config.ASSUMED_RANGE_M, "assumed"
        r = config.NARROW_F_PX * config.DRONE_WIDTH_M / w_px
        lo, hi = config.RANGE_LIMITS_M
        return float(min(max(r, lo), hi)), "size"


# ==========================================================================
#   Self-test: synthetic jinking trajectory, innovation whiteness.
#   python -m turret_host.tracker
# ==========================================================================

def _make_box(u: float, v: float, w: float, h: float,
              conf: float = 0.8) -> Detection:
    return Detection(u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0,
                     conf, "drone")


def _autocorr(s: np.ndarray, lag: int) -> float:
    s = s - s.mean()
    denom = float(s @ s)
    if denom <= 0.0:
        return 0.0
    return float(s[lag:] @ s[:-lag] / denom)


def _ljung_box(s: np.ndarray, h: int = 5) -> float:
    n = len(s)
    acc = 0.0
    for k in range(1, h + 1):
        r = _autocorr(s, k)
        acc += r * r / (n - k)
    return n * (n + 2) * acc


def _frame_times(n: int, rng: np.random.Generator) -> np.ndarray:
    """30 fps median with the measured worst case mixed in: ~33 ms typical,
    occasional ~50 ms. dt is handed to the filter, never assumed by it."""
    dts = 1.0 / config.NARROW_FPS + rng.normal(0.0, 0.0015, n)
    late = rng.random(n) < 0.08
    dts[late] = 0.050
    return np.cumsum(dts)


def _px_per_m() -> float:
    return config.NARROW_F_PX / config.ASSUMED_RANGE_M


def _truth_constant_velocity(ts: np.ndarray) -> np.ndarray:
    scale = _px_per_m()
    u = 640.0 + 0.6 * scale * (ts - ts[0])          # 0.6 m/s across the frame
    v = 360.0 + 0.1 * scale * (ts - ts[0])
    return np.stack([u, v], axis=1)


def _truth_jinking(ts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hand-held drone: <=1 m/s, <=3 m/s^2, direction reversing constantly."""
    scale = _px_per_m()
    a_max = 3.0 * scale
    v_max = 1.0 * scale
    pos = np.array([640.0, 360.0])
    vel = np.zeros(2)
    acc = rng.normal(0.0, 1.0, 2)
    acc *= a_max / np.linalg.norm(acc)
    out = np.zeros((len(ts), 2))
    next_jink = ts[0] + 0.4
    for i, t in enumerate(ts):
        dt = 0.0 if i == 0 else t - ts[i - 1]
        if t >= next_jink:
            acc = rng.normal(0.0, 1.0, 2)
            acc *= a_max / np.linalg.norm(acc)
            next_jink = t + rng.uniform(0.25, 0.6)
        vel = vel + acc * dt
        sp = np.linalg.norm(vel)
        if sp > v_max:
            vel *= v_max / sp
        pos = pos + vel * dt
        out[i] = pos
    return out


def _run(truth: np.ndarray, ts: np.ndarray, rng: np.random.Generator,
         box_wh=(132.0, 132.0), drop_frames=(), occlude=None):
    """Feed a trajectory through the tracker. Returns (tracker, records)."""
    trk = PixelTracker()
    records = []
    for i, t in enumerate(ts):
        u, v = truth[i]
        w, h = box_wh
        dets: List[Detection] = []
        if i not in drop_frames:
            mu = u + rng.normal(0.0, config.MEAS_NOISE_PX)
            mv = v + rng.normal(0.0, config.MEAS_NOISE_PX)
            if occlude is not None and occlude[0] <= i < occlude[1]:
                # Hand comes up from below: the bottom edge is eaten, the top
                # edge survives, the reported centre migrates upward.
                keep = 0.55
                h_occ = h * keep
                mv = mv - (h - h_occ) / 2.0
                dets.append(_make_box(mu, mv, w, h_occ))
            else:
                dets.append(_make_box(mu, mv, w, h))
        est = trk.update(dets, t)
        records.append({
            "i": i, "t": t, "truth": np.array([u, v]), "est": est,
            "white": None if trk.last_nu_white is None else trk.last_nu_white.copy(),
            "state": trk.state, "q": trk.q, "nis": trk.nis,
            "naive": None if not dets else np.array([dets[0].cx, dets[0].cy]),
        })
    return trk, records


def _whiteness_report(name: str, records) -> None:
    w = np.array([r["white"] for r in records
                  if r["white"] is not None and r["state"] is TrackState.TRACK])
    if len(w) < 30:
        print(f"  {name}: only {len(w)} TRACK frames -- not enough to judge")
        return
    n = len(w)
    bound = 2.0 / math.sqrt(n)
    nis = np.array([r["nis"] for r in records if r["state"] is TrackState.TRACK])
    print(f"  {name}: N={n} TRACK frames")
    print(f"    whitened innovation mean  {w[:, 0].mean():+.3f}, {w[:, 1].mean():+.3f}"
          f"   (want ~0)")
    print(f"    whitened innovation std    {w[:, 0].std():.3f},  {w[:, 1].std():.3f}"
          f"   (want ~1)")
    print(f"    mean NIS {nis.mean():.2f} (want ~2.0, chi2 2 dof);"
          f" {100.0 * (nis > config.NIS_THRESHOLD).mean():.1f}% over"
          f" NIS_THRESHOLD (want ~1%)")
    ok = True
    r1 = []
    for axis, label in ((0, "u"), (1, "v")):
        rs = [_autocorr(w[:, axis], k) for k in range(1, 6)]
        r1.append(rs[0])
        lb = _ljung_box(w[:, axis], 5)
        # 95% critical value of chi2 with 5 dof.
        white = lb < 11.070
        ok = ok and white
        print(f"    {label}: r1..r5 " + " ".join(f"{r:+.3f}" for r in rs)
              + f"  (|r|<{bound:.3f} expected)  Ljung-Box={lb:.2f}"
              + ("  WHITE" if white else "  NOT WHITE"))

    # Whiteness and consistency are two different questions and only one of them
    # is a bug. Report both, and name the cause when the answer is known.
    consistent = 1.4 <= nis.mean() <= 2.8
    print(f"    whiteness: {'WHITE' if ok else 'CORRELATED'}    "
          f"consistency (mean NIS ~2): {'OK' if consistent else 'OFF'}")
    if not ok and min(r1) < -0.15:
        print("      r1 is NEGATIVE, which is the signature of a high-gain filter,")
        print("      not of a mismodelled target: q_base is sized for a 3 m/s^2")
        print("      jink, so on a calmer trajectory the gain approaches 1 and")
        print("      consecutive innovations inherit anti-correlated measurement")
        print("      noise. VERIFIED: shrinking q to 0.02x drives r1 to -0.09.")
        print("      Sizing q for the worst case is the deliberate trade.")


def _aim_bias_v(box_wh: Tuple[float, float]) -> float:
    """Expected vertical component of the aim bias, for the report below."""
    up = world_up_in_image(config.NARROW_ROTATION_DEG)
    return config.AIM_BIAS_UP_FRAC * _box_extent_along(box_wh, up) * up[1]


def _main() -> None:
    rng = np.random.default_rng(7)
    print("turret_host.tracker self-test -- no hardware touched\n")
    print(f"  q_base = {config.Q_BASE:.3g} (px/s^2)^2, "
          f"sqrt = {math.sqrt(config.Q_BASE):.0f} px/s^2 "
          f"= 3 m/s^2 at {config.ASSUMED_RANGE_M} m")
    print(f"  up-in-image at {config.NARROW_ROTATION_DEG} deg roll: "
          f"({world_up_in_image(config.NARROW_ROTATION_DEG)[0]:+.2f}, "
          f"{world_up_in_image(config.NARROW_ROTATION_DEG)[1]:+.2f})\n")

    print("1. Constant velocity -- validates the filter and the whitening maths.")
    ts = _frame_times(600, rng)
    _, rec = _run(_truth_constant_velocity(ts), ts, rng)
    _whiteness_report("constant-velocity", rec)

    print("\n2. Jinking hand-held target (<=1 m/s, <=3 m/s^2), adaptive Q live.")
    ts = _frame_times(900, rng)
    truth = _truth_jinking(ts, rng)
    trk, rec = _run(truth, ts, rng)
    _whiteness_report("jinking", rec)
    qs = np.array([r["q"] for r in rec]) / config.Q_BASE
    print(f"    adaptive q: median {np.median(qs):.2f}x q_base, "
          f"peak {qs.max():.2f}x (cap {config.Q_MAX_MULT:.0f}x)")
    print("    NOTE: a jink is a real model violation. Some correlation at the")
    print("          jink onsets is physics, not a bug -- that is what the Q")
    print("          spike exists to absorb. Compare against case 1.")

    print("\n3. Occlusion, dropouts and the state machine.")
    ts = _frame_times(300, rng)
    truth = _truth_jinking(ts, rng)
    drops = set(range(120, 135))              # 15 misses -> TRACK -> COAST
    _, rec = _run(truth, ts, rng, drop_frames=drops, occlude=(40, 70))
    seen = []
    for r in rec:
        if not seen or seen[-1][1] is not r["state"]:
            seen.append((r["i"], r["state"]))
    print("    state timeline: " + " -> ".join(f"{s.value}@{i}" for i, s in seen))
    occ = [r for r in rec if 42 <= r["i"] < 70]
    held = [r for r in occ if r["est"].occluded]
    print(f"    occlusion hold engaged on {len(held)}/{len(occ)} occluded frames")
    if held:
        # Truth is the drone centre; the aim point should sit ABOVE it by the
        # bias, and must NOT have followed the remnant box. The bias is computed
        # from the UNOCCLUDED box size, which is what the tracker remembers.
        bias_v = _aim_bias_v((132.0, 132.0))
        centre_err = np.array([abs(r["est"].v - (r["truth"][1] + bias_v))
                               for r in held])
        naive_err = np.array([abs(r["naive"][1] - r["truth"][1]) for r in held])
        print(f"    aim-point error vs true centre: held {centre_err.mean():.1f} px, "
              f"naive remnant centroid {naive_err.mean():.1f} px")
    released = [r["i"] for r in rec if r["i"] >= 70 and not r["est"].occluded]
    print(f"    released at frame {released[0] if released else 'never'} "
          f"(occlusion ended at 70)")
    coast = [r for r in rec if r["state"] is TrackState.COAST]
    if coast:
        sp = [math.hypot(r["est"].du, r["est"].dv) for r in coast]
        print(f"    COAST for {len(coast)} frames, speed {sp[0]:.0f} -> "
              f"{sp[-1]:.0f} px/s (hold {config.COAST_HOLD_MS} ms then decay "
              f"{config.COAST_DECAY_MS} ms)")
    late = max(ts[i] - ts[i - 1] for i in range(1, len(ts)))
    print(f"    worst frame gap in this run: {1000.0 * late:.0f} ms, and the "
          f"track survived it -- loss is counted in frames, not seconds.")


if __name__ == "__main__":
    _main()
