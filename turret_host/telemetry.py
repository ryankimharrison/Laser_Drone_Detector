"""Flight recorder: what the loop saw, what it commanded, what the IMU felt.

The question this exists to answer is "the turret sees the drone and slews
AWAY from it". That cannot be diagnosed from frames alone, because the frames
do not say what the control law asked for, and it cannot be diagnosed from
commands alone, because the commands do not say whether the platform obeyed.
Three streams, one clock:

  control.jsonl   one record per control-loop pass: the estimate, the pixel
                  error, the goal, the MOTOR RATES COMMANDED, track state,
                  laser state and every interlock condition
  imu.jsonl       gyro rate and gravity tilt sampled from the board WHILE it
                  servos, via link.try_probe(), which drops samples rather
                  than delaying a vel
  frames/         periodic JPEGs with the detection and goal drawn

Everything is stamped with time.perf_counter(), which is also what
cameras.CameraThread puts in Frame.t, so a control record joins to the frame
it was computed from by `frame_t` and to the nearest IMU sample by time.

The recorder must never be able to break the loop it observes. Every logging
call is wrapped: a telemetry failure disables the recorder and is reported
once, rather than raising inside the control thread.
"""
from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from pathlib import Path
from typing import Optional

from turret_host import config

_RATE_RE = re.compile(r"rate now\s+([-+\d.]+),\s*([-+\d.]+),\s*([-+\d.]+)")
_TILT_RE = re.compile(r"tilt now\s+([-+\d.]+) pitch,\s*([-+\d.]+) roll")
# `imu fast` -> "IMUF gx gy gz pitch roll [ticks_ms]". One line, two I2C burst
# reads. The trailing tick arrived in firmware iteration 13; this pattern is
# unanchored and takes the first five fields, so it reads both old and new
# boards. See clocksync.py for what the tick is for.
_IMUF_RE = re.compile(r"IMUF\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+"
                      r"([-+\d.]+)\s+([-+\d.]+)")
#: The same line's board tick, when present. Separate pattern so a pre-13
#: board simply yields None instead of failing the whole parse.
_IMUF_TICK_RE = re.compile(r"IMUF\s+(?:[-+\d.]+\s+){5}(\d+)")


def _jsonable(x):
    """Config values are scalars and tuples -- but not all of them, and a
    Path or an Enum sneaking in must not take the whole session file with it.
    """
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return repr(x)


def _f(x):
    """JSON-safe float: inf/nan are not valid JSON and break every reader."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class FlightRecorder:
    def __init__(self, out_dir, link=None, imu_hz: float = 10.0,
                 on_message=None, save_frames: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.link = link
        self.imu_period = 1.0 / imu_hz if imu_hz > 0 else None
        self._on_message = on_message

        self.t0 = time.perf_counter()
        self.wall0 = time.time()
        self._control = (self.dir / "control.jsonl").open("w", encoding="utf-8")
        self._imu = (self.dir / "imu.jsonl").open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.failed: Optional[str] = None
        self.control_rows = 0
        self.imu_rows = 0

        # -- locked-frame capture ------------------------------------------
        # control.jsonl records that the tracker locked a box at (1002, 165),
        # conf 0.48, 272 px across. It cannot say WHAT that was, and "what did
        # it lock onto" is the question that matters when the turret chases
        # something that is not the drone. So: keep the pixels.
        #
        # On a WORKER THREAD behind a bounded queue. A JPEG write is 5-15 ms
        # and the control loop has a 33 ms budget it shares with inference --
        # writing inline would make the recorder cause the stalls it exists to
        # measure. The queue drops rather than blocks, for the same reason.
        self.save_frames = bool(save_frames)
        self.frames_dir = self.dir / "frames"
        #: Unannotated frames + YOLO sidecar labels. This is the
        #: directory the annotation loop and any retraining read;
        #: `frames/` is the drawn-on copy, for eyeballing only.
        self.raw_dir = self.dir / "raw"
        #: WIDE imagery. The recorder saved none until 2026-09-20, so no wide
        #: box could ever be adjudicated by eye -- a 228 px box at conf 0.74 in
        #: native wide pixels is a drone at 0.9 m or it is the operator's
        #: torso, and only the frame can say which. Half resolution and clean
        #: (no overlays): this exists to be LOOKED AT and to be annotated.
        self.wide_dir = self.dir / "wide"
        self.frames_saved = 0
        self.frames_dropped = 0
        #: Frames whose box could not be expressed as a label for
        #: THIS image -- almost always a wide-sourced box mapped
        #: outside the narrow frame. Counted, not silent.
        self.labels_skipped = 0
        self.wide_frames_saved = 0
        self.wide_frames_dropped = 0
        #: Camera threads, for the duplicate counters in session.json. Set by
        #: the app once the threads exist; empty is fine.
        self.cameras: dict = {}
        # 32, not 8. Every frame is offered now (not just the ones that
        # locked), so a short queue drops exactly during the fast motion that
        # makes the writer fall behind -- again losing the interesting subset.
        # 32 x ~2.7 MB is ~86 MB of headroom; frames_dropped still counts what
        # does not fit, so a shortfall stays visible rather than silent.
        self._fq: "queue.Queue" = queue.Queue(maxsize=32)
        #: Wide imagery rides its OWN queue and worker. Sharing the narrow
        #: queue would let wide writes evict narrow frames, and the narrow
        #: frames are what the interlock and the aim point are judged on.
        self._wq: "queue.Queue" = queue.Queue(maxsize=16)
        self._writer: Optional[threading.Thread] = None
        self._wide_writer: Optional[threading.Thread] = None
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.wide_dir.mkdir(parents=True, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "FlightRecorder":
        if self.link is not None and self.imu_period:
            self._thread = threading.Thread(target=self._imu_loop,
                                            name="telemetry-imu", daemon=True)
            self._thread.start()
        if self.save_frames:
            self._writer = threading.Thread(target=self._frame_writer,
                                            name="telemetry-frames", daemon=True)
            self._writer.start()
            self._wide_writer = threading.Thread(
                target=self._wide_frame_writer,
                name="telemetry-wide", daemon=True)
            self._wide_writer.start()
        return self

    # -- locked frames -----------------------------------------------------
    def offer_frame(self, image, box, label: str, aim=None) -> None:
        """Queue a frame for annotated capture. NEVER blocks the caller.

        `box` is the last ASSOCIATED detection and is OLDER than `image` --
        see the caller. `aim` is the aim point the turret is driving at right
        now, drawn as a cross so the two can be told apart by eye.
        """
        if not self.save_frames or self.failed:
            return
        try:
            self._fq.put_nowait((image, box, label, aim))
        except queue.Full:
            # Dropping a picture is free; stalling the control loop is not.
            self.frames_dropped += 1

    def offer_wide_frame(self, image, dets, index: int, frame_t: float) -> None:
        """Queue a WIDE frame and its detections. NEVER blocks the caller.

        `dets` is a sequence of (conf, native_box, mapped_box) triples, both
        boxes as (x1, y1, x2, y2) -- native in WIDE pixels, mapped already in
        NARROW coordinates. The mapping is done by the caller so this module
        stays free of fusion geometry.
        """
        if not self.save_frames or self.failed:
            return
        try:
            self._wq.put_nowait((image, list(dets or ()), int(index),
                                 float(frame_t)))
        except queue.Full:
            # Dropping wide imagery is free; stalling the wide detector is not
            # -- it feeds the fallback the tracker uses when narrow goes blind.
            self.wide_frames_dropped += 1

    def _wide_frame_writer(self) -> None:
        import cv2
        while not self._stop.is_set() or not self._wq.empty():
            try:
                image, dets, index, frame_t = self._wq.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                stem = "%06d" % index
                h, w = image.shape[:2]
                # HALF RESOLUTION, CLEAN. Half because wide is 1920x1080 and
                # this runs at up to 20 Hz alongside everything else; clean
                # because a burned-in box teaches an annotator the model's own
                # answer, which is the one thing their verdict must not know.
                small = cv2.resize(image, (w // 2, h // 2),
                                   interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(self.wide_dir / (stem + ".jpg")), small,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                # Sidecar. Line 1 carries the wide frame's own capture time so
                # this joins to control.jsonl by TIME, not by a frame index
                # that counts a different camera's frames. Boxes are in FULL
                # resolution wide pixels, not the halved image -- the jpeg is a
                # convenience, the numbers are the record.
                lines = ["# t=%.6f n=%d native=%dx%d" % (frame_t, len(dets), w, h)]
                for conf, nb, mb in dets:
                    lines.append(
                        "%.4f  %.1f %.1f %.1f %.1f  %.1f %.1f %.1f %.1f"
                        % (conf, nb[0], nb[1], nb[2], nb[3],
                           mb[0], mb[1], mb[2], mb[3]))
                (self.wide_dir / (stem + ".txt")).write_text(
                    chr(10).join(lines) + chr(10), encoding="utf-8")
                self.wide_frames_saved += 1
            except Exception:                              # noqa: BLE001
                self.wide_frames_dropped += 1

    def _frame_writer(self) -> None:
        import cv2
        while not self._stop.is_set() or not self._fq.empty():
            try:
                image, box, label, aim = self._fq.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                stem = label.split()[0]
                # RAW FIRST, AND IT IS THE ONE THAT MATTERS.
                #
                # This used to write only the annotated copy, which made every
                # frame the recorder captured useless for the two things frames
                # are actually for:
                #   * TRAINING -- the green box and the label text are burned
                #     into the pixels, so a model fine-tuned on them learns to
                #     find green rectangles.
                #   * BLIND ANNOTATION -- the label carries the confidence, and
                #     an annotator who can read "conf=0.37" anchors to it. The
                #     human verdict is the only information entering the
                #     retraining loop that the model does not already hold;
                #     showing it the model's own answer destroys exactly that.
                cv2.imwrite(str(self.raw_dir / (stem + ".jpg")), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                # The box as data rather than as pixels, in YOLO's own
                # normalised cx cy w h, so a confirmed frame drops straight
                # into dataset/yolo without a conversion step.
                if box:
                    h, w = image.shape[:2]
                    # CLIP TO THE IMAGE, AND REFUSE WHAT DOES NOT BELONG TO IT.
                    #
                    # A WIDE-sourced box has been pushed through
                    # wide_to_narrow into NARROW pixel coordinates, and the
                    # wide camera sees 108 deg against the narrow camera's
                    # 28.8 -- so most of the wide field maps OUTSIDE the narrow
                    # frame, legitimately. Writing that as a label against the
                    # narrow image annotates a region of a picture that does
                    # not contain the object. Measured on the first run with
                    # sidecars: 28% of labels had out-of-range values, one with
                    # cx = -0.046.
                    #
                    # A bad label is worse than a missing one -- a missing
                    # label costs one training example, a wrong one actively
                    # teaches the model that empty wall is a drone.
                    x1 = max(0.0, min(float(box[0]), float(box[2])))
                    y1 = max(0.0, min(float(box[1]), float(box[3])))
                    x2 = min(float(w), max(float(box[0]), float(box[2])))
                    y2 = min(float(h), max(float(box[1]), float(box[3])))
                    cx, cy = 0.5 * (x1 + x2) / w, 0.5 * (y1 + y2) / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    # Require real overlap, not a sliver clipped off an edge:
                    # a 2-px strip of a drone is not a training example of one.
                    if bw > 0.02 and bh > 0.02 and 0.0 <= cx <= 1.0 \
                            and 0.0 <= cy <= 1.0:
                        (self.raw_dir / (stem + ".txt")).write_text(
                            "0 %.6f %.6f %.6f %.6f\n" % (cx, cy, bw, bh))
                    else:
                        # The JPEG is still kept. An unlabelled frame is a
                        # candidate negative and a human can still judge it;
                        # it just carries no claim about where anything is.
                        self.labels_skipped += 1

                if not box:
                    # NO BOX: the raw frame is the whole point and there is
                    # nothing to draw on it. Skipping the annotated copy keeps
                    # the added I/O of recording every frame to ONE jpeg per
                    # frame rather than two.
                    self.frames_saved += 1
                    continue

                img = image.copy()
                if box:
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[2]), int(box[3]))
                    cv2.rectangle(img, p1, p2, (60, 220, 60), 2)
                if aim is not None:
                    # The CURRENT aim point, as a cross. The rectangle is a
                    # stale detection; this is what the turret is driving at.
                    ax, ay = int(aim[0]), int(aim[1])
                    cv2.drawMarker(img, (ax, ay), (40, 120, 255),
                                   cv2.MARKER_CROSS, 26, 2)
                cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.imwrite(str(self.frames_dir / (stem + ".jpg")),
                            img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                self.frames_saved += 1
            except Exception:                              # noqa: BLE001
                self.frames_dropped += 1

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._writer is not None:
            self._writer.join(timeout=5.0)   # let queued pictures land
        if getattr(self, "_wide_writer", None) is not None:
            self._wide_writer.join(timeout=5.0)
        with self._lock:
            for fh in (self._control, self._imu):
                try:
                    fh.close()
                except Exception:                          # noqa: BLE001
                    pass
        self._write_session()

    # -- the IMU stream ----------------------------------------------------
    def _imu_loop(self) -> None:
        """Sample the board while it servos, yielding to the vel stream."""
        while not self._stop.is_set():
            t = time.perf_counter()
            reply = None
            try:
                # `imu fast`, NOT `imu`. The verbose form runs describe(),
                # which proves liveness over 16 samples (~80-160 ms) and can
                # RECONFIGURE the accelerometer as a side effect. Both are
                # wrong here: at 10 Hz it would hold the io lock most of the
                # time and starve the vel stream, and a sampler that silently
                # revives the part would hide the fault it exists to catch.
                reply = self.link.try_probe("imu fast")
            except Exception as exc:                       # noqa: BLE001
                self._fail("imu probe: %s" % exc)
                return
            if reply:
                rec = {"t": t, "rel_t": t - self.t0}
                m = _IMUF_RE.search(reply)
                if m:
                    g = [float(x) for x in m.groups()]
                    rec["gyro_dps"], rec["tilt_deg"] = g[:3], g[3:]
                else:
                    # Fall back to the verbose form's wording, so an older
                    # firmware without `imu fast` still records something.
                    m = _RATE_RE.search(reply)
                    if m:
                        rec["gyro_dps"] = [float(x) for x in m.groups()]
                    m = _TILT_RE.search(reply)
                    if m:
                        rec["tilt_deg"] = [float(x) for x in m.groups()]
                # THE BOARD'S OWN CLOCK, alongside the host's.
                # Without it every IMU row is timed only by when the HOST got
                # round to reading it, which folds the serial round trip and
                # the io lock into what is meant to be a sensor timestamp.
                #
                # It costs NOTHING to record: `imu fast` already answers
                # "IMUF gx gy gz pitch roll ms" and that trailing ms IS the
                # board's ticks_ms. _IMUF_RE captures only the first five
                # fields, so the tick was being parsed off and thrown away on
                # every sample. No extra probe, no extra round trip, and in
                # particular no extra contention on the io lock the comment
                # above warns about -- it is the same reply, read properly.
                if config.IMU_LOG_BOARD_TICKS:
                    mt = _IMUF_TICK_RE.search(reply)
                    if mt:
                        rec["board_ticks_ms"] = int(mt.group(1))
                        # The host time this reply was READ. Pairing the two
                        # over a run is what clocksync.py fits offset and skew
                        # from -- it reached 0.29 ms residual offline and has
                        # never been wired into a live recording until now.
                        rec["tick_host_t"] = t
                ok, dropped = self.link.probe_stats
                rec["probe_ok"], rec["probe_dropped"] = ok, dropped
                self._write(self._imu, rec)
                self.imu_rows += 1
            self._stop.wait(self.imu_period)

    # -- the control stream ------------------------------------------------
    def log_control(self, *, t, state, est, out, laser, interlock, link,
                    pose=None, frame_t=None, frame_index=None,
                    detections=None, wide=None, tracker=None,
                    det_frame_t=None, dup_dropped=None,
                    infer_ms=None) -> None:
        """One control-loop pass. Called from the control thread; must be cheap."""
        if self.failed:
            return
        try:
            rec = {
                "t": t, "rel_t": t - self.t0,
                "frame_t": _f(frame_t), "frame_index": frame_index,
                # THE DETECTION'S OWN FRAME TIME, as its own field.
                # `est.box_t` is stamped from the tracker's self._t, which
                # equals the frame time only because _update() happens to be
                # called with it -- a coincidence of wiring, not a guarantee,
                # and it conflates "when the picture was taken" with "when the
                # box associated". Offline replay needs the former explicitly.
                "det_frame_t": _f(det_frame_t),
                # NARROW detector inference cost, ms. Pairs with wide.infer_ms.
                "infer_ms": _f(infer_ms),
                # Frames the capture thread discarded as byte-identical
                # re-deliveries, cumulative. See config.DEDUP_FRAMES.
                "dup_dropped": dup_dropped,
                "track": getattr(state, "value", str(state)),
                "laser": getattr(laser, "value", str(laser)),
            }
            if est is not None:
                rec["est"] = {
                    "u": _f(est.u), "v": _f(est.v),
                    "du": _f(est.du), "dv": _f(est.dv),
                    "q": _f(est.q), "nis": _f(est.nis),
                    "occluded": bool(est.occluded),
                    "range_m": _f(est.range_m),
                    # WHICH ESTIMATOR PRODUCED range_m: "assumed" (the flat
                    # ASSUMED_RANGE_M fallback), "size" (apparent-size ranging
                    # off the box width), or "stereo". Never logged until now,
                    # and it matters more than it looks: DRONE_WIDTH_M was None
                    # for every run recorded before 2026-09-20, so range_m is
                    # EXACTLY 3.0 on all 6621 historical rows and ranging never
                    # once fired. It is live now, so new runs will vary where
                    # every baseline is constant -- and range feeds the
                    # acceleration clamp (a_px = a_max * f / range_m) and the
                    # goal pixel g(R). Without this field those two populations
                    # cannot be told apart after the fact.
                    "range_source": getattr(est, "range_source", None),
                    "has_box": est.box is not None,
                    "box_t": _f(est.box_t),
                    "box_age_ms": (None if est.box_t is None
                                   else _f((t - est.box_t) * 1000.0)),
                }
                if est.box is not None:
                    b = est.box
                    rec["est"]["box"] = [_f(b.x1), _f(b.y1), _f(b.x2), _f(b.y2)]
                    rec["est"]["conf"] = _f(b.conf)
            # WHAT ASSOCIATION THREW AWAY, and how wide the gate was when it
            # did. Until now a rejected detection left no trace at all: the
            # only evidence was n_targets disagreeing with has_box, which
            # gives a count and no distances -- so "is the gate too tight?"
            # could be argued but never measured. These three fields are what
            # turn that into a number.
            if tracker is not None:
                rej = getattr(tracker, "last_rejected_px", None)
                rec["assoc"] = {
                    "rejected_px": ([_f(x) for x in rej[:8]] if rej else []),
                    "n_rejected": (len(rej) if rej else 0),
                    "gate_px": _f(tracker.gate_radius_px()),
                    "reseeds": getattr(tracker, "reseeds", 0),
                    # Velocity corrections bounded by the physical
                    # acceleration ceiling. See config.MAX_TARGET_ACCEL_MPS2.
                    "accel_clamped": getattr(tracker, "accel_clamped", 0),
                    "last_clamp_px_s": _f(getattr(
                        tracker, "last_accel_clamp_px_s", 0.0)),
                }
            # THE COMMAND. This is the half of the picture frames cannot show.
            if out is not None:
                rec["cmd"] = {
                    "rate_a": _f(out.rate_a), "rate_b": _f(out.rate_b),
                    "error_u": _f(out.error_u), "error_v": _f(out.error_v),
                    "error_px": _f(math.hypot(out.error_u, out.error_v)),
                    "goal_u": _f(out.goal_u), "goal_v": _f(out.goal_v),
                    "saturated": bool(out.saturated),
                    "dot_locked": bool(out.dot_locked),
                    # The feedforward input the law actually used. Compare
                    # against est.du/dv: when FEEDFORWARD_SUBTRACT_PLATFORM is
                    # off they are identical, and the gap when it is on IS the
                    # platform's contribution to the measured image velocity.
                    "v_target_u": _f(getattr(out, "v_target_u", None)),
                    "v_target_v": _f(getattr(out, "v_target_v", None)),
                    "ff_platform_px_s": _f(getattr(out, "ff_platform_px_s", None)),
                    # COMMANDED rate at box time, not achieved -- see types.py.
                    "omega_at_box": [_f(getattr(out, "omega_at_box_a", None)),
                                     _f(getattr(out, "omega_at_box_b", None))],
                    "p_scale": _f(getattr(out, "p_scale", None)),
                    "authority": _f(getattr(out, "authority", None)),
                }
            else:
                rec["cmd"] = None
            if interlock is not None:
                rec["interlock"] = {
                    "failed": list(interlock.failed),
                    "margins": {k: _f(v) for k, v in interlock.margins.items()},
                }
            if link is not None:
                rec["link"] = {
                    "ms_since_vel": _f(link.ms_since_vel()),
                    "servoing": bool(link.servoing),
                    "sent": getattr(link.status(), "sent", None),
                    "errors": getattr(link.status(), "errors", None),
                }
            if pose is not None:
                rec["pose_deg"] = [_f(pose[0]), _f(pose[1])]
            if detections is not None:
                rec["n_targets"] = len(detections)
                # EVERY detection the frame produced, not only the one that
                # associated. Without this an offline replay cannot re-run the
                # filter against the true measurement stream -- it only ever
                # sees the boxes the current association already chose, so any
                # alternative gate or filter is unfalsifiable. `source`
                # distinguishes a narrow box from a wide->narrow mapped one,
                # which matters because a mapped box legitimately lands
                # outside the narrow frame.
                rec["detections"] = [
                    {"box": [_f(d.x1), _f(d.y1), _f(d.x2), _f(d.y2)],
                     "conf": _f(d.conf),
                     "source": getattr(d, "source", "narrow")}
                    for d in detections[:12]
                ]
            # The WIDE detector, separately. `n_targets` above is taken after
            # app._tracker_loop has already folded the wide fallback in, so a
            # zero there cannot distinguish "the wide camera saw nothing" from
            # "it saw something and the mapping put it outside GATE_PX". This
            # is the number that tells them apart.
            if wide is not None:
                rec["wide"] = wide
            self._write(self._control, rec)
            self.control_rows += 1
        except Exception as exc:                           # noqa: BLE001
            self._fail("log_control: %s" % exc)

    @property
    def probe_dropped_count(self) -> int:
        """IMU samples skipped because the writer held the port. Not an error."""
        return self.link.probe_stats[1] if self.link is not None else 0

    # -- plumbing ----------------------------------------------------------
    def _write(self, fh, rec) -> None:
        with self._lock:
            fh.write(json.dumps(rec) + "\n")

    def _fail(self, why: str) -> None:
        """Disable the recorder. It may never take the loop down with it."""
        if self.failed:
            return
        self.failed = why
        if self._on_message:
            self._on_message("telemetry DISABLED: %s" % why)

    #: Constants a later analysis cannot reinterpret the log without. LATENCY_S
    #: is the one that bites: it decides what time each logged prediction was
    #: FOR, so scoring an old run against a newer value silently measures the
    #: wrong thing. It changed from an 0.060 estimate to a measured 0.066 on
    #: 2026-09-19, and none of the runs before that recorded which they used.
    _CONFIG_SNAPSHOT = (
        "LATENCY_S", "CONTROL_GAIN_K", "GATE_PX", "GATE_MAX_PX", "GATE_CHI2",
        "RESEED_AFTER_MISSES", "RESEED_MIN_CONF", "MAX_ERROR_TO_FIRE_PX",
        "COAST_HOLD_MS", "COAST_DECAY_MS", "ACQUIRE_FRAMES", "ACQUIRE_MISSES",
        "TRACK_MISSES", "YOLO_WEIGHTS", "YOLO_IMGSZ", "YOLO_CONF",
        "MEAS_NOISE_PX", "Q_BASE", "NARROW_F_PX", "MAX_MOTOR_RATE",
    )

    def _dup_counts(self) -> dict:
        """Per-camera duplicate-frame drops. Never raises."""
        out = {}
        for name, cam in (self.cameras or {}).items():
            try:
                out[name] = {
                    "dropped": int(getattr(cam, "duplicates_dropped", 0)),
                    "published": int(getattr(cam, "frames", 0)),
                }
            except Exception:                              # noqa: BLE001
                continue
        return out

    def _write_session(self) -> None:
        probe = self.link.probe_stats if self.link is not None else (0, 0)
        (self.dir / "session.json").write_text(json.dumps({
            "clock_anchor": {"perf_counter": self.t0, "wall": self.wall0,
                             "note": "wall = wall0 + (t - t0)"},
            # EVERY public constant, not a hand-picked list. The curated
            # tuple below is kept only as a documented floor -- the failure it
            # could not prevent was AIM_BIAS, which was not in it, so
            # analyze_run.py had to ASSUME 0.15 for older runs and drew a wrong
            # conclusion from that assumption. A snapshot that depends on
            # someone remembering to extend it will be wrong again.
            "config": {k: _jsonable(getattr(config, k, None))
                       for k in dir(config)
                       if k.isupper() and not k.startswith("_")},
            "config_required": sorted(self._CONFIG_SNAPSHOT),
            "control_rows": self.control_rows,
            "imu_rows": self.imu_rows,
            "frames_saved": self.frames_saved,
            "frames_dropped": self.frames_dropped,
            "labels_skipped": self.labels_skipped,
            "wide_frames_saved": self.wide_frames_saved,
            "wide_frames_dropped": self.wide_frames_dropped,
            # Byte-identical re-deliveries the capture threads discarded.
            # Before de-duplication this ran at ~35-43%; a run that still
            # reports a high rate here means the fix is not holding.
            "duplicates_dropped": dict(self._dup_counts()),
            "imu_probe_delivered": probe[0],
            "imu_probe_dropped": probe[1],
            "failed": self.failed,
        }, indent=2), encoding="utf-8")
