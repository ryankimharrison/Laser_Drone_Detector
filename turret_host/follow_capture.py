"""Let the turret follow the drone while both cameras capture training frames.

    python -m turret_host.follow_capture                 # lazy follow, capture both
    python -m turret_host.follow_capture --dry-run       # detect and log, NEVER move
    python -m turret_host.follow_capture --shot handheld_kitchen

WHY THE TRACKER IS DELIBERATELY BAD AT ITS JOB
----------------------------------------------
The obvious build centres the drone in frame. Do that and every frame you
capture has the drone in the middle, and a detector trained on those frames
gets worse at finding it anywhere else -- which is exactly the case the control
loop depends on, because off-centre is WHEN IT NEEDS TO SLEW. Closed-loop
capture would quietly destroy the positional variety the current static-camera
dataset has, while looking like an improvement.

So this follows LAZILY, with hysteresis: it does nothing until the drone drifts
past `--outer` of the way to the frame edge, then slews only until it is back
inside `--inner`, then stops again. Between corrections the drone wanders across
the frame on its own. You want a sloppy tracker here, not a good one.

What this DOES buy, and the static rig cannot:
  * the narrow camera stays on target. Its horizontal field is 28.8 deg, and
    two whole tracks in the first session turned out to contain no drone at all
    because the shot was framed on the wide view.
  * backgrounds sweep as the platform moves, so one walk round a room yields
    what a dozen static setups would.
  * MOTION BLUR FROM A MOVING PLATFORM, which the live system will have on
    every frame and the current dataset has on none.

THE MODEL POINTS. IT DOES NOT LABEL.
------------------------------------
Boxes from the tracking model are never written as labels. If the detector
locks onto a ceiling fan -- it did, on 52 frames, before the fan was named as a
distractor -- the turret follows the fan and you capture a sequence perfectly
centred on it. Label that with the model's own box and you have trained it to
be more confident about its own mistake, and the error compounds every round.
Labels come from `autolabel` (a different model) and then human review, same as
every other frame in this dataset.

THE LASER, AND WHAT IT IS AND IS NOT PROTECTED BY
-------------------------------------------------
Off unless `--laser` is passed. When passed, the beam goes through the real
`LaserInterlock`, not a bare on/off: it lights only when a fresh drone box
contains the beam path AND no face is within FACE_INHIBIT_MARGIN_PX of it AND
the platform is settled AND a `vel` went out recently. Any one failing, or any
detector going quiet, turns it off -- that is the fail-closed rule, and it is
the same code `app.py` uses.

On top of that, `--duty` (default 0.05) caps the long-run on-time with a leaky
bucket. At 5% the average power is 0.25 mW rather than 5 mW. That is a real
reduction in dose. It does NOT improve pointing, and pointing is the failure
mode, so do not read it as making a mispointed beam safe.

WHAT IS STILL NOT COVERED, stated plainly because the banner cannot:
  * HEADS FROM BEHIND. The veto is YuNet, which is a FACE detector. Someone
    turned away is invisible to it, and an operator holding a drone is turned
    away a large fraction of a session. PLAN.md 3 specs the fix (union with
    YOLO11n-pose and COCO person, inhibit on the head region); it is not built.
  * IR LEAKAGE. 1064/808 nm from the module is unmeasured. Invisible light
    triggers no blink reflex, and head detection does nothing about it.
  * This interlock has never run against hardware. It passes its self-tests.

Wear eye protection, keep it off faces, and treat the first run as a test of
the interlock rather than a demo of the turret.
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
import datetime
import json
import math
import queue
import threading
import time
from dataclasses import asdict
from typing import Dict, Optional

import cv2
import numpy as np

from turret_host import cameras, config
from turret_host.capture_dataset import (
    ClipRecord, DATASET_DIR, JPEG_QUALITY, MAX_GAP_S, MIN_GAP_S,
    _differs_enough, _thumb,
)
from turret_host.control import Controller, LaserInterlock
from turret_host.detector import Detector, FaceDetector
from turret_host.link import TurretLink
from turret_host.tracker import PixelTracker
from turret_host.types import LaserState, TrackState


class FrameWriter(threading.Thread):
    """JPEG encoding and disk I/O off the control path.

    `cv2.imwrite` of a 1280x720 JPEG is single-digit milliseconds, and this
    loop also runs detection, tracking, the control law, the face pass and the
    display. Doing the write inline puts disk latency directly into the
    interval between `vel` commands, which is the one number the control law's
    stability depends on -- and a stalled loop is also a stalled interlock.

    This is the same reason app.py separates infer / faces / tracker / control
    / display into their own threads. This module does not go that far, but
    the blocking I/O at least belongs somewhere else.
    """

    def __init__(self):
        super().__init__(name="frame-writer", daemon=True)
        self._q: "queue.Queue" = queue.Queue(maxsize=256)
        # NOT `self._stop`: threading.Thread already has a private _stop()
        # method, and shadowing it with an Event breaks join() with
        # "'Event' object is not callable" only at shutdown -- after the
        # session, when the frames are already captured and the manifest is
        # about to be written.
        self._done = threading.Event()
        self.dropped = 0

    def put(self, path: str, image) -> bool:
        try:
            self._q.put_nowait((path, image))
            return True
        except queue.Full:
            # Never block the caller. A dropped frame is a lost training
            # sample; a stalled control loop is a moving turret nobody is
            # steering. Counted so the count can be reported rather than
            # discovered later as a hole in the data.
            self.dropped += 1
            return False

    def run(self) -> None:
        while not (self._done.is_set() and self._q.empty()):
            try:
                path, image = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            except Exception:
                self.dropped += 1

    def close(self) -> None:
        """Drain before returning: the clip manifest claims these frames exist."""
        self._done.set()
        self.join(timeout=20.0)


class HardExampleMiner:
    """Keep the frames the detector MISSED while the drone was demonstrably there.

    A frame the model already gets right teaches it almost nothing. A frame
    where it fails, with the drone plainly in view, sits exactly on the wrong
    side of its decision boundary -- that is the frame worth training on, and
    this session produces them for free because the operator can see the box
    blink out while the drone is still in shot.

    WHERE THE LABEL COMES FROM, WHICH IS THE WHOLE ARGUMENT
    ------------------------------------------------------
    NOT from the model. The model's answer on these frames is "no drone", which
    is the error being corrected; writing that down would train it to be more
    confident about the miss. Instead a run of misses is only kept when it is
    BRACKETED by confident hits, and the box is interpolated between those two
    observations. The label is evidence from frames either side, never a
    prediction about the frame itself. That is what separates this from
    self-training on your own output, which degrades a model every round.

    A run is discarded, not interpolated, when:
      * it is longer than `max_gap` frames -- over a long gap the drone may have
        left, turned, or been occluded, and a straight line between the ends is
        an invention rather than an inference;
      * either bracketing box is missing (the run hit the end of the clip);
      * the bracketing boxes disagree in area by more than `max_area_ratio`,
        which means something discontinuous happened between them and linear
        interpolation has no basis.
    """

    def __init__(self, max_gap_s: float = 0.40, max_area_ratio: float = 2.5,
                 min_conf: float = 0.45):
        # BOUNDED IN TIME, NOT IN LOOP ITERATIONS. This loop is not locked to
        # the camera: `latest()` returns the newest frame, so a fast loop sees
        # one frame several times and a slow loop never sees some at all.
        # Counting iterations would make the interpolation window depend on
        # how busy the machine was -- and the window is the entire basis for
        # claiming a straight line between the brackets is honest.
        self.max_gap_s = float(max_gap_s)
        self.max_area_ratio = float(max_area_ratio)
        self.min_conf = float(min_conf)
        self._last_hit: Optional[tuple] = None      # (t, box)
        self._pending: list = []                    # (t, image) since that hit
        self._seen_index: Optional[int] = None      # dedupe: same Frame twice
        self.mined: list = []
        self.runs_kept = 0
        self.runs_dropped = 0

    @staticmethod
    def _lerp(a, b, f: float):
        return tuple(a[i] + (b[i] - a[i]) * f for i in range(4))

    def note(self, frame, box, conf: float) -> bool:
        """One camera frame. `box` is the detection, or None if it missed.

        Returns False if this Frame was already seen -- the caller is handing
        us whatever `latest()` holds, which repeats when the loop outruns the
        camera. Counting a repeat as another missed frame would manufacture
        gaps that never happened and interpolate across them.
        """
        if frame.index == self._seen_index:
            return False
        self._seen_index = frame.index
        t = frame.t

        if box is None:
            if self._last_hit is not None:
                self._pending.append((t, frame.image))
                if t - self._last_hit[0] > self.max_gap_s:
                    # Too long to interpolate honestly. Drop the whole run and
                    # wait for a fresh bracket rather than keeping a prefix.
                    self.runs_dropped += 1
                    self._pending.clear()
                    self._last_hit = None
            return True

        if conf < self.min_conf:
            return True                 # a weak hit is not a trustworthy bracket

        prev = self._last_hit
        xyxy = (box.x1, box.y1, box.x2, box.y2)
        if prev is not None and self._pending:
            span = t - prev[0]
            pa = max(1.0, (prev[1][2] - prev[1][0]) * (prev[1][3] - prev[1][1]))
            na = max(1.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            ratio = max(pa, na) / min(pa, na)
            if span <= self.max_gap_s and ratio <= self.max_area_ratio:
                # Interpolate on TIMESTAMP, not on position in the list. Frames
                # do not arrive evenly spaced, and spacing them evenly would
                # put the box where the drone was not.
                for ft, img in self._pending:
                    f = (ft - prev[0]) / span if span > 0 else 0.5
                    self.mined.append((img, self._lerp(prev[1], xyxy, f)))
                self.runs_kept += 1
            else:
                self.runs_dropped += 1
        self._pending.clear()
        self._last_hit = (t, xyxy)
        return True

    def drain(self):
        out, self.mined = self.mined, []
        return out


class DutyLimiter:
    """Leaky bucket capping long-run beam on-time to `fraction`.

    Fills at 1 s/s while the beam is on and drains at `fraction` s/s always, so
    the sustained duty cannot exceed `fraction` however the interlock behaves
    above it. Capacity sets how long a single continuous burst may be before
    the bucket is full -- 0.4 s here, so the beam is visibly a blink rather
    than a steady dot, which is also the honest way to show a 5% duty.

    This is a DOSE limiter, not a pointing check. It reduces average power; it
    does nothing whatever about where the beam is going.
    """

    def __init__(self, fraction: float, capacity_s: float = 0.4):
        self.fraction = max(0.0, min(1.0, float(fraction)))
        self.capacity = max(0.05, float(capacity_s))
        self._level = 0.0
        self._last: Optional[float] = None

    def allow(self, t_now: float, want_on: bool) -> bool:
        dt = 0.0 if self._last is None else max(0.0, t_now - self._last)
        self._last = t_now
        self._level = max(0.0, self._level - dt * self.fraction)
        if not want_on:
            return False
        if self._level + dt > self.capacity:
            return False
        self._level += dt
        return True

    @property
    def load(self) -> float:
        return self._level / self.capacity


def _to_display(x: float, y: float, stored_h: int, scale: float):
    """Stored-frame pixel -> rotated, resized preview pixel.

    ROTATE_90_CLOCKWISE maps stored (x, y) to (h-1-y, x). Everything the
    control law produces is in STORED coordinates -- that is the convention the
    whole stack uses -- so the rotation belongs here, at the point of drawing,
    and nowhere else.
    """
    if config.NARROW_ROTATE_CLOCKWISE:
        rx, ry = stored_h - 1 - y, x
    else:
        rx, ry = y, stored_h - 1 - x
    return int(rx * scale), int(ry * scale)


def _draw_aim(panel, goal, out, box, stored_shape, scale, goal_calibrated):
    """Crosshair where the BEAM would land, plus the box and the error vector.

    This is the substitute for switching the laser on to see where the turret
    is pointing, and it is strictly more informative: it shows the aim point on
    frames where the beam would be inhibited, needs no surface to land on, and
    does not care how far away the wall is.

    It shows where the software BELIEVES the beam goes. When the goal pixel is
    uncalibrated that belief is just the frame centre, so it is drawn hollow
    and labelled -- a confident crosshair over an unmeasured aim point is
    exactly the kind of thing that gets trusted by mistake.
    """
    h, w = stored_shape
    if box is not None:
        a = _to_display(box.x1, box.y1, h, scale)
        b = _to_display(box.x2, box.y2, h, scale)
        cv2.rectangle(panel, (min(a[0], b[0]), min(a[1], b[1])),
                      (max(a[0], b[0]), max(a[1], b[1])), (0, 220, 60), 2)
    # The crosshair does NOT depend on there being a track. Where the beam
    # points is a property of the mechanism, and "nothing is detected" is
    # exactly when you want to see where it is aimed.
    if goal is None:
        return panel
    gx, gy = _to_display(goal[0], goal[1], h, scale)
    if out is not None:
        tx, ty = _to_display(out.goal_u + out.error_u,
                             out.goal_v + out.error_v, h, scale)
        cv2.arrowedLine(panel, (gx, gy), (tx, ty), (0, 200, 255), 2, tipLength=0.25)
    colour = (60, 80, 255) if not goal_calibrated else (80, 220, 255)
    cv2.line(panel, (gx - 14, gy), (gx + 14, gy), colour, 2)
    cv2.line(panel, (gx, gy - 14), (gx, gy + 14), colour, 2)
    cv2.circle(panel, (gx, gy), 9, colour, 1 if not goal_calibrated else 2)
    if not goal_calibrated:
        cv2.putText(panel, "goal UNCALIBRATED", (max(4, gx - 70), gy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)
    return panel


def _banner(narrow, wide, state, err, outer, slewing, kept, laser_text, msg,
            goal=None, out=None, box=None, goal_calibrated=False):
    panels = []
    if narrow is not None:
        stored_shape = narrow.shape[:2]
        img = cv2.rotate(narrow, cv2.ROTATE_90_CLOCKWISE
                         if config.NARROW_ROTATE_CLOCKWISE
                         else cv2.ROTATE_90_COUNTERCLOCKWISE)
        panel = cv2.resize(img, (360, 640))
        _draw_aim(panel, goal, out, box, stored_shape, 360.0 / img.shape[1],
                  goal_calibrated)
        panels.append(panel)
    if wide is not None:
        pad = np.zeros((640, 640, 3), np.uint8)
        pad[140:500, :, :] = cv2.resize(wide, (640, 360))
        panels.append(pad)
    if not panels:
        return None
    canvas = np.hstack(panels) if len(panels) > 1 else panels[0]

    bar = np.zeros((116, canvas.shape[1], 3), np.uint8)
    colour = (60, 200, 255) if slewing else (160, 160, 160)
    cv2.putText(bar, "%s   err %5.0f px / outer %.0f   %s" %
                (state, err, outer, "SLEWING" if slewing else "drifting"),
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2)
    cv2.putText(bar, "kept %d frames" % kept, (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    # The beam state is on screen at all times, in red when it is not provably
    # off. There is no path here that lights it, but "I thought it was off" is
    # not a thing anyone should have to rely on.
    cv2.putText(bar, laser_text, (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (80, 220, 80) if laser_text.endswith("OFF") else (60, 60, 255), 2)
    if msg:
        cv2.putText(bar, msg[:70], (330, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 120), 1)
    return np.vstack([canvas, bar])


def run(args) -> int:
    ident = cameras.identify_cameras()
    print(ident.report())

    ctl = Controller()
    if not ctl.ready:
        print("\nJacobian is NOT calibrated -- the loop cannot servo.")
        print("Run: python -m turret_host.calibrate_all --steps jacobian")
        return 1
    print("Jacobian calibrated. goal pixel calibrated = %s"
          % ctl.goal_model.is_calibrated)

    cameras.lock_exposure(ident.narrow_index, exposure=-6)
    narrow = cameras.narrow_thread(ident.narrow_index).start()
    wide = cameras.wide_thread(ident.wide_index).start()

    # MATCH THE TRAINING SIZE. config.YOLO_IMGSZ is 1280 (chosen for the COCO
    # model on a native narrow frame), but drone_y11n was fine-tuned at 640.
    # Infer at 1280 and the target subtends about twice the pixels the model
    # ever saw, which costs recall silently -- it does not error, it just finds
    # fewer drones. Override here rather than in config, because the right size
    # is a property of the WEIGHTS, not of the machine.
    det = Detector(weights=args.weights, imgsz=args.imgsz, conf=args.conf).start()
    print("detector: %s  imgsz %d  conf %.2f  classes %s"
          % (det.weights_path, det.imgsz, det.conf, det.class_names))
    if det.missing_classes:
        print("  (not in this model, ignored: %s)" % ", ".join(det.missing_classes))
    tracker = PixelTracker()

    faces = None
    interlock = None
    duty = DutyLimiter(args.duty)
    if args.laser:
        if args.dry_run:
            print("--laser and --dry-run together: dry run wins, beam stays off")
            args.laser = False
        else:
            # The face pass is a HARD requirement for arming, not an optional
            # extra: without it `heads_t` is never fresh and the interlock
            # refuses every frame anyway. Start it before the beam is armed so
            # a failure here is a startup error, not a silent fail-open.
            faces = FaceDetector().start()
            interlock = LaserInterlock(armed=True)
            print("\n" + "!" * 68)
            print("LASER ARMED. Interlock: drone containment + face veto + duty.")
            print("Face veto is YuNet -- FACE detection. Someone turned AWAY is")
            print("invisible to it. Duty capped at %.0f%% (avg ~%.2f mW of 5 mW)."
                  % (args.duty * 100, 5.0 * args.duty))
            print("Eye protection on. Q to stop.")
            print("!" * 68 + "\n")

    link = None
    if not args.dry_run:
        link = TurretLink(port=args.port)
        link.start()
        # Assert the beam off before anything else moves. The firmware flag is
        # still True from iter004, so the board WOULD accept `laser on`; this
        # is the host saying it never will.
        link.set_laser(False)
        print("link up on %s -- beam asserted OFF" % link.port)
    else:
        print("DRY RUN: detector and control law only, motors never commanded")

    session = args.session or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    session_dir = _os.path.join(DATASET_DIR, session)
    clips_dir = _os.path.join(session_dir, "clips")
    n = 0
    while _os.path.isdir(_os.path.join(clips_dir, "%s_%02d" % (args.shot, n))):
        n += 1
    clip_id = "%s_%02d" % (args.shot, n)
    clip_dir = _os.path.join(clips_dir, clip_id)
    _os.makedirs(clip_dir, exist_ok=True)
    clip = ClipRecord(clip_id, args.shot, True,
                      {"capture": "follow", "motion": "platform"},
                      started_utc=datetime.datetime.now(datetime.timezone.utc)
                      .isoformat(timespec="seconds"))

    # Deadband in pixels, from the SHORT side of the narrow frame -- that is the
    # axis with the least angular room, so it is the one that loses the target.
    half = min(config.NARROW_SIZE) / 2.0
    outer, inner = args.outer * half, args.inner * half
    print("\nlazy follow: slew above %.0f px, stop below %.0f px" % (outer, inner))
    print("Q or ESC to finish.\n")

    slewing, kept, msg = False, 0, ""
    out = None
    est = None
    laser_now = "beam off"
    miner = HardExampleMiner(max_gap_s=args.max_gap_s)
    writer = FrameWriter()
    writer.start()
    hard_labels: Dict[str, list] = {}
    last_thumb: Dict[str, Optional[np.ndarray]] = {}
    last_kept_t: Dict[str, float] = {}
    started = time.perf_counter()
    err = 0.0
    state = TrackState.SEARCH

    try:
        while True:
            fn, _ = narrow.latest()
            fw, _ = wide.latest()
            now = time.perf_counter()

            if fn is not None:
                result = det.detect(fn)
                # Mine BEFORE the tracker folds it in: the miner wants the raw
                # per-frame detector verdict, and the tracker deliberately
                # smooths exactly the dropouts being collected.
                best = max(result.targets, key=lambda d: d.conf, default=None)
                miner.note(fn, best, best.conf if best else 0.0)
                tracker.update(result.targets, fn.t)
                est = tracker.estimate_for_control(now)
                state = est.state

                if state in (TrackState.TRACK, TrackState.ACQUIRE):
                    out = ctl.compute(est, now)
                    err = ctl.error_magnitude(out)
                    # Hysteresis: start on `outer`, stop on `inner`. Without the
                    # two thresholds this chatters on and off around a single
                    # boundary and the platform buzzes instead of drifting.
                    if slewing and err < inner:
                        slewing = False
                        if link:
                            link.stop()
                    elif not slewing and err > outer:
                        slewing = True
                    if slewing and link:
                        link.submit(out)
                else:
                    err = float("inf") if state is TrackState.SEARCH else err
                    out = None
                    if slewing:
                        slewing = False
                        if link:
                            link.stop()

            # -- beam ------------------------------------------------------
            # Every frame, after the control decision and before anything else.
            # `out is None` covers SEARCH/COAST and any frame the control law
            # did not produce: no aim point means no beam, which is the same
            # answer the interlock gives for a missing box or a stale face pass.
            if interlock is not None and link is not None:
                face_list, heads_t = [], None
                if fn is not None and faces is not None:
                    face_list = faces.detect(fn)
                    heads_t = fn.t
                if out is None:
                    laser_state = interlock.evaluate(
                        t_now=now, track_state=state, error_px=float("inf"),
                        platform_rate_deg_s=float("inf"), last_vel_sent_t=None,
                        goal_calibrated=ctl.goal_model.is_calibrated)
                else:
                    ms = link.ms_since_vel()
                    laser_state = interlock.evaluate(
                        t_now=now, track_state=state, error_px=err,
                        platform_rate_deg_s=ctl.platform_rate_deg_s(out),
                        last_vel_sent_t=(now - ms / 1000.0
                                         if math.isfinite(ms) else None),
                        faces=face_list,
                        aim=(out.goal_u, out.goal_v),
                        target=(out.goal_u + out.error_u,
                                out.goal_v + out.error_v),
                        goal_calibrated=ctl.goal_model.is_calibrated,
                        drone_box=est.box if est else None,
                        drone_box_t=est.box_t if est else None,
                        heads_t=heads_t)
                # The duty limiter can only ever REMOVE permission. It is
                # applied after the interlock, never instead of it.
                beam_on = duty.allow(now, laser_state is LaserState.FIRING)
                link.set_laser(beam_on)
                msg = interlock.reason
                laser_now = ("BEAM ON" if beam_on else
                             "beam off - %s" % laser_state.value)

            # Sample frames exactly as capture_dataset does, so this session is
            # interchangeable with a manual one downstream.
            for name, fr in (("narrow", fn), ("wide", fw)):
                if fr is None:
                    continue
                gap = now - last_kept_t.get(name, 0.0)
                if gap < MIN_GAP_S:
                    continue
                th = _thumb(fr.image)
                if gap < MAX_GAP_S and not _differs_enough(th, last_thumb.get(name)):
                    continue
                i = clip.frames.get(name, 0)
                writer.put(_os.path.join(clip_dir, "%s_%04d.jpg" % (name, i)),
                           fr.image)
                clip.frames[name] = i + 1
                last_thumb[name], last_kept_t[name] = th, now

            # Mined misses bypass the difference sampler entirely. They have to:
            # the sampler keeps ~4 frames a second and the dropouts happen at
            # 30, so the frames worth having are exactly the ones it discards.
            for img, xyxy in miner.drain():
                i = clip.frames.get("narrow", 0)
                stem = "narrow_%04d" % i
                writer.put(_os.path.join(clip_dir, stem + ".jpg"), img)
                h, w = img.shape[:2]
                x1, y1, x2, y2 = xyxy
                hard_labels[stem] = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                     (x2 - x1) / w, (y2 - y1) / h]
                clip.frames["narrow"] = i + 1
            kept = sum(clip.frames.values())

            if interlock is None:
                laser_text = ("LASER OFF (dry run, no link)" if link is None
                              else "LASER OFF (not armed)")
            else:
                laser_text = "%s   duty %3.0f%%" % (laser_now, duty.load * 100)
            canvas = _banner(fn.image if fn else None, fw.image if fw else None,
                             state.value, err if math.isfinite(err) else 0.0,
                             outer, slewing, kept, laser_text, msg,
                             goal=ctl.goal_model.goal_pixel(config.ASSUMED_RANGE_M),
                             out=out if state in (TrackState.TRACK,
                                                  TrackState.ACQUIRE) else None,
                             box=est.box if est else None,
                             goal_calibrated=ctl.goal_model.is_calibrated)
            if canvas is not None:
                cv2.imshow("follow capture", canvas)
            if (cv2.waitKey(5) & 0xFF) in (ord('q'), 27):
                break
    finally:
        # Drain the writer BEFORE the manifest is written: clip.json states a
        # frame count, and a count that outruns the files on disk is a dataset
        # with holes in it that nothing downstream will notice.
        writer.close()
        clip.seconds = round(time.perf_counter() - started, 1)
        json.dump(asdict(clip), open(_os.path.join(clip_dir, "clip.json"), "w"),
                  indent=2)
        # Sidecar, not a .txt beside the jpg: `--split` truncates every label
        # file when it rebuilds the tree, so a label written there would be
        # destroyed by the very next pipeline step. autolabel reads this and
        # prefers it, because an interpolation between two confident
        # observations beats anything a detector can say about a frame it
        # already failed on.
        if hard_labels:
            json.dump({"source": "follow_capture.HardExampleMiner",
                       "note": "boxes INTERPOLATED between bracketing hits; "
                               "never predicted on the frame itself",
                       "max_gap_s": args.max_gap_s,
                       "frames": hard_labels},
                      open(_os.path.join(clip_dir, "hard_examples.json"), "w"),
                      indent=2)
        if link:
            link.set_laser(False)
            link.stop()
            link.close()
        if faces is not None:
            faces.close()
        det.close()
        narrow.close()
        wide.close()
        cv2.destroyAllWindows()

    print("\n%s: %s in %.1f s" % (clip_id, dict(clip.frames), clip.seconds))
    print("hard examples: %d frames from %d bracketed miss-runs (%d runs dropped "
          "as too long/inconsistent)" % (len(hard_labels), miner.runs_kept,
                                         miner.runs_dropped))
    print("Labels do NOT come from the tracking model. Next:")
    print("  python -m turret_host.capture_dataset --split")
    print("  python -m turret_host.autolabel")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shot", default="follow", help="shot key for the clip")
    ap.add_argument("--session", default=None)
    ap.add_argument("--weights", default="drone_y11n.pt")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size. Default matches how drone_y11n was "
                         "trained (640), NOT config.YOLO_IMGSZ (1280)")
    ap.add_argument("--conf", type=float, default=config.YOLO_CONF)
    ap.add_argument("--port", default=None)
    ap.add_argument("--laser", action="store_true",
                    help="ARM THE BEAM, behind the full interlock and --duty")
    ap.add_argument("--duty", type=float, default=0.05,
                    help="max long-run beam duty cycle (0.05 = 5%%)")
    ap.add_argument("--max-gap-s", type=float, default=0.40,
                    help="longest dropout, IN SECONDS, still interpolated into "
                         "hard examples. Beyond this the run is DROPPED: a "
                         "straight line over a long gap is invention. In "
                         "seconds rather than frames because the loop is not "
                         "locked to the camera")
    ap.add_argument("--outer", type=float, default=0.35,
                    help="slew once the error exceeds this fraction of the "
                         "half-frame. Bigger = lazier = more positional variety")
    ap.add_argument("--inner", type=float, default=0.12,
                    help="stop slewing below this fraction. Hysteresis.")
    ap.add_argument("--dry-run", action="store_true",
                    help="detector and control law only; motors never commanded")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
