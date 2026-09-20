"""Capture a training set for the drone detector.

    python -m turret_host.capture_dataset            # run a capture session
    python -m turret_host.capture_dataset --split    # build a YOLO train/val split
    python -m turret_host.capture_dataset --stats    # summarise what exists

WHY THIS EXISTS RATHER THAN "just record a video and pull frames out"
---------------------------------------------------------------------
Three things go wrong when you do that, and all three are silent:

1. **Domain mismatch.** Record H.264 to disk and you train on H.264 artifacts
   while inference sees MJPEG decoded by Media Foundation. This tool captures
   through `cameras.CameraThread` -- the exact path `app.py` uses -- so the
   training frames and the live frames come off the same pipeline.

2. **Near-duplicate frames.** At 30 fps consecutive frames teach nothing. Worse,
   if near-duplicates land on both sides of a train/val split, validation
   measures memorisation: the score looks excellent and the model fails live.
   This tool samples on frame DIFFERENCE (so it keeps more during motion, fewer
   when the scene is static), and `--split` splits **by clip, never by frame**.

3. **Missing variation.** 300 near-identical frames are worth less than 80
   varied ones. The shot list below is the variation matrix from
   DETECTION_BRIEF.md 5, turned into prompts so nobody has to remember it.

NEGATIVES ARE NOT OPTIONAL. Frames of people holding nothing, and holding a
phone or a bottle, are what stop the model learning "person therefore drone" --
which fails in a room full of spectators, i.e. exactly the demo.
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
import glob as _glob
import json
import random
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import cv2
import numpy as np

from turret_host import cameras, config

ROOT = _os.path.dirname(_pkg_dir)
DATASET_DIR = _os.path.join(ROOT, "dataset")

# --------------------------------------------------------------------------
#   SAMPLING
# --------------------------------------------------------------------------
# A frame is kept only when it differs enough from the last KEPT frame. This is
# better than a fixed interval: it naturally keeps more frames while the drone
# is moving (where the model needs blur and pose variety) and fewer while
# someone stands still holding it.
DIFF_THRESHOLD = 6.0          # mean abs difference, 0-255, on a 64x64 grey thumb
MIN_GAP_S = 0.25              # never keep faster than this, however fast things move
MAX_GAP_S = 3.0               # ...but keep at least this often, so static poses appear
JPEG_QUALITY = 95


# --------------------------------------------------------------------------
#   THE SHOT LIST  (DETECTION_BRIEF.md 5)
# --------------------------------------------------------------------------
@dataclass
class Shot:
    key: str
    prompt: str
    has_drone: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    target_frames: int = 25


def negatives_only_shot_list(total_frames: int = 500) -> List[Shot]:
    """Just the NO-DRONE shots, sized to `total_frames` between them.

    WHY A NEGATIVES-ONLY SESSION IS WORTH ITS OWN RUN
    -------------------------------------------------
    In every positive frame captured so far the drone is IN A HAND. So across
    the training set "drone" and "a person gripping an object" are almost
    perfectly correlated, and nothing forces the model to learn which of the
    two carries the signal. It demonstrably learned the wrong one: measured
    2026-09-19, `drone_y11n_v2` put conf 0.74 on a hooded shoulder while the
    real drone sat visible and unboxed in the same frame, and scored real
    drone locks at 0.37-0.47 in that same run. The false positive OUTSCORED
    the true one, which is why raising the confidence threshold makes it
    worse rather than better.

    These shots break the correlation from one side: a hand holding a phone
    or a bottle, and a person holding nothing at all, both labelled empty.

    They do NOT break it from the other side -- an unheld drone is still
    missing from the set entirely, and the deliverable is a FLYING drone with
    no hand anywhere near it. That gap needs its own positive shots.

    Frame budget is split evenly. The difference-based sampler is left exactly
    as it is: it keeps more frames while things move and fewer while they do
    not, which is the right behaviour here too.
    """
    negs = [s for s in build_shot_list() if not s.has_drone]
    per = max(1, int(round(total_frames / float(len(negs)))))
    for s in negs:
        s.target_frames = per
    return negs


def build_shot_list() -> List[Shot]:
    """The variation matrix, as prompts.

    Ordered so the most valuable shots come first: if the session gets cut
    short, what got captured is still the part that mattered. Occlusion leads
    because the demo literally contains a hand wrapped around the airframe, and
    it is the case every public drone dataset lacks.
    """
    shots: List[Shot] = []

    # -- occlusion: the hardest case, and the one the demo actually contains --
    for grip, where in (("under", "from underneath, fingers under the body"),
                        ("over", "over the top, palm covering the body"),
                        ("arm", "by one arm, hand covering a motor"),
                        ("two-hand", "in both hands, most of the airframe hidden")):
        shots.append(Shot(
            "occl_%s" % grip,
            "Hold the drone %s. At ~3 m. Turn it slowly while you hold it." % where,
            tags={"distance_m": "3", "occlusion": grip, "motion": "slow"}))

    # -- distance sweep: 132 px at 3 m, ~80 px at 5 m --
    for d in ("2", "3", "4", "5"):
        shots.append(Shot(
            "dist_%sm" % d,
            "Stand at about %s m. Normal grip. Move the drone around the frame." % d,
            tags={"distance_m": d, "occlusion": "normal", "motion": "slow"}))

    # -- orientation: nose-on is the smallest silhouette --
    for name, how in (("nose_on", "pointing straight at the camera (smallest silhouette)"),
                      ("side", "side on"),
                      ("top", "tilted so you see the top plate"),
                      ("tumbling", "rotating through all three axes, slowly")):
        shots.append(Shot(
            "orient_%s" % name,
            "At ~3 m, hold the drone %s." % how,
            tags={"distance_m": "3", "orientation": name, "motion": "slow"}))

    # -- background --
    shots.append(Shot("bg_cluttered", "Stand so the background is the CLUTTERED part of the room.",
                      tags={"background": "cluttered", "distance_m": "3"}))
    shots.append(Shot("bg_plain", "Stand against a PLAIN wall.",
                      tags={"background": "plain", "distance_m": "3"}))
    shots.append(Shot("bg_people", "Have someone walk around BEHIND you while you hold the drone.",
                      tags={"background": "people", "distance_m": "3"}))

    # -- motion: the live system sees motion blur; the training set must too --
    shots.append(Shot("motion_fast", "Move the drone QUICKLY -- jink it side to side, like evading.",
                      tags={"motion": "fast", "distance_m": "3"}, target_frames=35))
    shots.append(Shot("motion_walk", "Walk toward and away from the camera holding the drone.",
                      tags={"motion": "walking"}))

    # -- lighting: whatever range the demo room will actually have --
    shots.append(Shot("light_bright", "Room lights FULLY ON. Normal grip at ~3 m.",
                      tags={"lighting": "bright", "distance_m": "3"}))
    shots.append(Shot("light_dim", "Lights DIMMED or partly off. Same pose.",
                      tags={"lighting": "dim", "distance_m": "3"}))

    # -- NEGATIVES: without these the model learns "person => drone" ----------
    shots.append(Shot("neg_empty", "NO DRONE. Just stand there, empty hands. Move around.",
                      has_drone=False, tags={"negative": "empty_hands"}, target_frames=30))
    shots.append(Shot("neg_phone", "NO DRONE. Hold a PHONE, the way you would hold the drone.",
                      has_drone=False, tags={"negative": "phone"}))
    shots.append(Shot("neg_bottle", "NO DRONE. Hold a BOTTLE or a mug.",
                      has_drone=False, tags={"negative": "bottle"}))
    shots.append(Shot("neg_room", "NO DRONE, NO PERSON. Pan the empty room, including the clutter.",
                      has_drone=False, tags={"negative": "empty_room"}))

    return shots


# --------------------------------------------------------------------------
#   CLIP RECORDING
# --------------------------------------------------------------------------
def _write_manifest(session_dir: str, session: str, records: List["ClipRecord"],
                    wide_fast: bool) -> dict:
    """Write the session manifest.

    Called after EVERY clip, not just at the end. A session is a person and a
    drone in a room for 40 minutes; losing it to a closed window or a crash
    because the index only existed in memory would be unforgivable, and the
    clips themselves are already safely on disk by then.
    """
    manifest = {
        "session": session,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "capture_path": "turret_host.cameras.CameraThread (MSMF + MJPEG, same as app.py)",
        "narrow_size": list(config.NARROW_SIZE),
        "wide_size": list(config.WIDE_FAST_SIZE if wide_fast else config.WIDE_SIZE),
        "sampling": {"diff_threshold": DIFF_THRESHOLD,
                     "min_gap_s": MIN_GAP_S, "max_gap_s": MAX_GAP_S},
        "clips": [asdict(r) for r in records],
    }
    tmp = _os.path.join(session_dir, "manifest.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    _os.replace(tmp, _os.path.join(session_dir, "manifest.json"))
    return manifest


@dataclass
class ClipRecord:
    clip_id: str
    shot_key: str
    has_drone: bool
    tags: Dict[str, str]
    frames: Dict[str, int] = field(default_factory=dict)   # camera -> count
    started_utc: str = ""
    seconds: float = 0.0


def _thumb(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16)


def _differs_enough(thumb: np.ndarray, previous: Optional[np.ndarray]) -> bool:
    if previous is None:
        return True
    return float(np.abs(thumb - previous).mean()) >= DIFF_THRESHOLD


def _beep(times: int = 1, freq: int = 880, ms: int = 120) -> None:
    """Non-blocking beep. Hands-free capture needs an AUDIBLE cue.

    The operator is 2-5 m away holding a drone in both hands, which is exactly
    the posture in which nobody can read a countdown on a laptop screen. Run it
    on a daemon thread: winsound.Beep blocks for its full duration, and 120 ms
    inside the capture loop is four dropped frames at 30 fps.
    """
    def run():
        try:
            import winsound
            for i in range(times):
                winsound.Beep(freq, ms)
                if i + 1 < times:
                    time.sleep(0.06)
        except Exception:
            pass                      # no audio device, or not Windows -- visual only
    threading.Thread(target=run, daemon=True).start()


class AutoPilot:
    """Walk the shot list on a timer, so the operator never touches a key.

    This drives the SAME key handling the manual path uses rather than
    duplicating clip creation and saving -- it returns a synthetic keycode and
    the main loop cannot tell the difference. Anything that is true of a manual
    session (debounce, manifest writes, frame-difference sampling) stays true
    here for free, and there is no second code path to keep in step.

    Phases: ready -> rec -> advance -> ready. A shot also ends early once it has
    hit `target_frames`, because past that point the extra frames are
    near-duplicates of ones already kept and cost review time for nothing.
    """

    def __init__(self, ready_s: float = 6.0, shot_s: float = 20.0):
        self.ready_s, self.shot_s = ready_s, shot_s
        self.phase, self.t0 = "ready", time.perf_counter()

    def tick(self, kept: int, target: int, last_shot: bool, cameras_live: int = 1):
        """(synthetic key, big on-screen caption).

        `kept` is the sum over ALL cameras (`sum(clip.frames.values())`) while
        `target_frames` is per camera, so the early exit has to scale by how
        many are streaming. Comparing them directly stops a two-camera shot at
        roughly half the frames it was asked for -- and silently, because the
        clip still looks like it completed.
        """
        el = time.perf_counter() - self.t0
        if self.phase == "ready":
            left = self.ready_s - el
            if left <= 0:
                self.phase, self.t0 = "rec", time.perf_counter()
                _beep(1, 988, 150)
                return ord(' '), ""
            return 255, "GET READY  %d" % int(left + 1)
        if self.phase == "rec":
            left = self.shot_s - el
            if left <= 0 or kept >= target * max(1, cameras_live):
                self.phase, self.t0 = "advance", time.perf_counter()
                _beep(2, 660, 110)
                return ord(' '), ""
            return 255, "RECORDING  %d" % int(left + 1)
        # advance: one tick after the stop, so the save has completed
        self.phase, self.t0 = "ready", time.perf_counter()
        if last_shot:
            _beep(3, 523, 180)
            return ord('q'), ""
        return ord('n'), ""


def _big_caption(canvas, text: str, recording: bool):
    """Centred caption sized to be read from across the room, not at the desk."""
    if not text or canvas is None:
        return canvas
    scale, thick = 2.4, 5
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(0, (canvas.shape[1] - tw) // 2)
    y = canvas.shape[0] // 2
    colour = (60, 220, 60) if recording else (60, 200, 255)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 6)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)
    return canvas


def _display(frame_narrow, frame_wide, shot, index, total, recording, kept, message):
    """Operator view. Narrow is rotated FOR DISPLAY ONLY (config.NARROW_ROTATION_DEG)."""
    panels = []
    if frame_narrow is not None:
        img = frame_narrow
        if config.NARROW_ROTATE_CLOCKWISE:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        panels.append(cv2.resize(img, (360, 640)))
    if frame_wide is not None:
        w = cv2.resize(frame_wide, (640, 360))
        pad = np.zeros((640, 640, 3), np.uint8)
        pad[140:500, :, :] = w
        panels.append(pad)
    if not panels:
        return None
    canvas = np.hstack(panels) if len(panels) > 1 else panels[0]

    banner = np.zeros((130, canvas.shape[1], 3), np.uint8)
    colour = (60, 220, 60) if recording else (200, 200, 200)
    cv2.putText(banner, "[%d/%d] %s" % (index + 1, total, shot.key), (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    # Wrap the prompt; it is the whole point of the panel.
    words, line, y = shot.prompt.split(), "", 56
    for word in words:
        if len(line) + len(word) > 64:
            cv2.putText(banner, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
            y += 22
            line = ""
        line += word + " "
    cv2.putText(banner, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

    # Show progress toward the target, per camera, so the operator knows when to
    # stop. Without this the only cue is a raw count with nothing to compare it to.
    target = shot.target_frames
    done = kept >= target
    state = "%s  %d / %d per camera%s" % (
        "RECORDING" if recording else "paused", kept, target,
        "   ENOUGH - press SPACE, then N" if done else "")
    colour = (60, 220, 60) if done else ((60, 200, 255) if recording else (160, 160, 160))
    cv2.putText(banner, state, (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    # Progress bar under the text, so it reads at a glance from 3 m away.
    bar_w = int(min(1.0, kept / float(target)) * (canvas.shape[1] - 24))
    cv2.rectangle(banner, (12, 126), (12 + bar_w, 129), colour, -1)
    if not shot.has_drone:
        cv2.putText(banner, "NEGATIVE - no drone in frame", (canvas.shape[1] - 330, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 160, 255), 2)
    if message:
        cv2.putText(banner, message, (canvas.shape[1] - 330, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    return np.vstack([banner, canvas])


def print_shot_list() -> None:
    """The whole plan, in the terminal, before anything opens.

    The prompts live on the video window's banner, which is the right place
    while shooting -- but you cannot plan a session from a banner you have not
    reached yet. Print it all up front so the run can be read, or printed out,
    in advance.
    """
    shots = build_shot_list()
    per_cam = sum(s.target_frames for s in shots)
    print("\n%d shots.  Target ~%d frames PER CAMERA (~%d across both)."
          % (len(shots), per_cam, per_cam * 2))
    print("Roughly 20-30 s of recording per shot WHILE MOVING THE DRONE.\n")
    print("  %-16s %-5s %s" % ("shot", "frames", "prompt"))
    print("  " + "-" * 94)
    for s in shots:
        mark = " " if s.has_drone else "*"
        prompt = s.prompt if len(s.prompt) <= 66 else s.prompt[:63] + "..."
        print("  %-16s %-5d %s%s" % (s.key, s.target_frames, mark, prompt))
    print("\n  * = NEGATIVE shot, no drone in frame. These are not optional:")
    print("    without them the model learns 'person => drone' and fails in a")
    print("    room full of spectators, which is exactly the demo.\n")


def run_session(session_name: Optional[str] = None,
                cameras_wanted: str = "both",
                wide_fast: bool = False,
                auto: Optional["AutoPilot"] = None,
                shots: Optional[List[Shot]] = None) -> int:
    ident = cameras.identify_cameras()
    print(ident.report())

    use_narrow = cameras_wanted in ("both", "narrow")
    use_wide = cameras_wanted in ("both", "wide")

    # Lock exposure BEFORE the MSMF stream exists. MSMF accepts exposure writes
    # and silently ignores them; the DSHOW detour is the only thing that works.
    if use_narrow:
        res = cameras.lock_exposure(ident.narrow_index, exposure=-6)
        print("narrow exposure lock: %s  (took=%s persisted=%s)  %s"
              % ("OK" if res.ok else "NOT LOCKED", res.took, res.persisted, res.message))
        if not res.ok:
            print("  Exposure is still automatic. Frame timing will vary and the")
            print("  green dot will bloom. Capture is still usable; tracking is worse.")

    threads = []
    narrow = wide = None
    # Sequential opens only: MSMF's source reader cannot be raced.
    if use_narrow:
        narrow = cameras.narrow_thread(ident.narrow_index).start()
        threads.append(narrow)
    if use_wide:
        wide = cameras.wide_thread(ident.wide_index, fast=wide_fast).start()
        threads.append(wide)

    session = session_name or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    session_dir = _os.path.join(DATASET_DIR, session)
    clips_dir = _os.path.join(session_dir, "clips")
    _os.makedirs(clips_dir, exist_ok=True)

    shots = shots or build_shot_list()
    records: List[ClipRecord] = []
    index, recording, kept = 0, False, 0
    clip: Optional[ClipRecord] = None
    clip_dir = ""
    last_thumb: Dict[str, Optional[np.ndarray]] = {}
    last_kept_t: Dict[str, float] = {}
    message = ""
    started = 0.0
    last_toggle = 0.0

    print_shot_list()
    if auto is None:
        print("KEYS:  SPACE start/stop   N next shot   B back   R redo this shot   Q finish")
        print("The prompt for each shot is on the video window banner.\n")
    else:
        print("AUTO: %.0fs to get into position, then %.0fs recording, then the next shot."
              % (auto.ready_s, auto.shot_s))
        print("      One beep = recording. Two = stop. Three = session over.")
        print("      No keys needed. Q still quits early.\n")
    shown = -1
    caption = ""

    try:
        while True:
            fn = narrow.latest()[0] if narrow else None
            fw = wide.latest()[0] if wide else None
            shot = shots[index]
            if index != shown:
                # Echo to the terminal too, so the session leaves a readable log
                # of what was asked for, not just what was saved.
                print("\n[%d/%d] %-16s target %d/camera%s\n      %s"
                      % (index + 1, len(shots), shot.key, shot.target_frames,
                         "" if shot.has_drone else "   *** NEGATIVE - no drone ***",
                         shot.prompt))
                shown = index

            if recording and clip is not None:
                now = time.perf_counter()
                for name, frame in (("narrow", fn), ("wide", fw)):
                    if frame is None:
                        continue
                    gap = now - last_kept_t.get(name, 0.0)
                    if gap < MIN_GAP_S:
                        continue
                    th = _thumb(frame.image)
                    if gap < MAX_GAP_S and not _differs_enough(th, last_thumb.get(name)):
                        continue
                    n = clip.frames.get(name, 0)
                    path = _os.path.join(clip_dir, "%s_%04d.jpg" % (name, n))
                    cv2.imwrite(path, frame.image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    clip.frames[name] = n + 1
                    last_thumb[name] = th
                    last_kept_t[name] = now
                kept = sum(clip.frames.values())

            canvas = _display(fn.image if fn else None, fw.image if fw else None,
                              shot, index, len(shots), recording, kept, message)
            if canvas is not None:
                _big_caption(canvas, caption, recording)
                cv2.imshow("turret capture", canvas)
            key = cv2.waitKey(15) & 0xFF

            # The auto-pilot only speaks when the operator has not. A real key
            # always wins, so Q works at any moment and the manual keys stay
            # usable to override a shot mid-session.
            if auto is not None and key == 255:
                key, caption = auto.tick(kept, shot.target_frames,
                                         last_shot=(index == len(shots) - 1),
                                         cameras_live=len(threads))

            # Debounce the toggle. Holding SPACE (or an OS key-repeat) otherwise
            # fires start/stop dozens of times a second: a real session produced
            # 333 clips of ~2 frames each on one shot, and every clip-start
            # FORCES a keep (the difference history is empty), so they were all
            # near-duplicates. 0.4 s is far longer than any repeat interval and
            # far shorter than a deliberate press.
            if key == ord(' ') and time.perf_counter() - last_toggle < 0.4:
                key = 255

            if key == ord(' '):
                last_toggle = time.perf_counter()
                if not recording:
                    # Number from what is ON DISK, not just this run's records.
                    # Resuming into an existing session with --session would
                    # otherwise restart at _00 and overwrite earlier clips.
                    n = 0
                    while _os.path.isdir(_os.path.join(clips_dir, "%s_%02d" % (shot.key, n))):
                        n += 1
                    clip_id = "%s_%02d" % (shot.key, n)
                    clip_dir = _os.path.join(clips_dir, clip_id)
                    _os.makedirs(clip_dir, exist_ok=True)
                    clip = ClipRecord(clip_id, shot.key, shot.has_drone, dict(shot.tags),
                                      started_utc=datetime.datetime.now(datetime.timezone.utc)
                                      .isoformat(timespec="seconds"))
                    last_thumb, last_kept_t, kept, started = {}, {}, 0, time.perf_counter()
                    recording, message = True, ""
                else:
                    recording = False
                    clip.seconds = round(time.perf_counter() - started, 1)
                    json.dump(asdict(clip), open(_os.path.join(clip_dir, "clip.json"), "w"), indent=2)
                    records.append(clip)
                    message = "saved %d frames" % kept
                    print("  %-16s %3d frames in %4.1f s" % (clip.clip_id, kept, clip.seconds))
                    _write_manifest(session_dir, session, records, wide_fast)
                    clip = None
            elif key in (ord('n'), 13) and not recording:
                index = min(index + 1, len(shots) - 1)
                kept, message = 0, ""
            elif key == ord('b') and not recording:
                index = max(index - 1, 0)
                kept, message = 0, ""
            elif key == ord('r') and not recording and records and records[-1].shot_key == shot.key:
                gone = records.pop()
                shutil.rmtree(_os.path.join(clips_dir, gone.clip_id), ignore_errors=True)
                message = "deleted %s" % gone.clip_id
                kept = 0
            elif key == ord('q') and not recording:
                break
    finally:
        # Always release. A leaked handle costs the NEXT run: isOpened() returns
        # True and every read() fails.
        for t in threads:
            t.stop()
        cv2.destroyAllWindows()

    _report(_write_manifest(session_dir, session, records, wide_fast))
    return 0


def _report(manifest: dict) -> None:
    clips = manifest["clips"]
    total = sum(sum(c["frames"].values()) for c in clips)
    pos = sum(sum(c["frames"].values()) for c in clips if c["has_drone"])
    neg = total - pos
    print("\nsession %s" % manifest["session"])
    print("  clips   %d" % len(clips))
    print("  frames  %d   (%d with drone, %d negatives)" % (total, pos, neg))
    if total and neg / total < 0.15:
        print("  WARNING: under 15% negatives. Without them the model learns")
        print("           'person => drone' and fails in a room of spectators.")
    covered = {k for c in clips for k in c["tags"]}
    for axis in ("occlusion", "distance_m", "orientation", "background", "lighting", "motion"):
        if axis not in covered:
            print("  WARNING: nothing captured varying '%s'" % axis)


# --------------------------------------------------------------------------
#   SPLIT  --  by clip, never by frame
# --------------------------------------------------------------------------
def _temporal_val_frames(src_dir: str, val_fraction: float) -> set:
    """The chronologically LAST `val_fraction` of each camera's frames.

    Used when there is only one clip per shot, where a clip-wise split cannot
    produce a validation set without deleting a whole condition from training.

    This is weaker than a clip-wise split and the score it produces is
    OPTIMISTIC: frames from one continuous recording share a background, a
    person, a grip and a lighting setup, so the model has seen very similar
    frames. It is still far better than a random frame split, because frames
    seconds apart are different poses while adjacent frames are the same pose.

    The honest fix is a second short capture session on another day -- that is a
    real held-out set. This exists so you are not blocked before then.
    """
    by_cam: Dict[str, List[str]] = {}
    for name in sorted(_os.listdir(src_dir)):
        if name.endswith(".jpg"):
            by_cam.setdefault(name.split("_")[0], []).append(name)
    out = set()
    for names in by_cam.values():
        cut = int(len(names) * (1.0 - val_fraction))
        out.update(names[cut:])
    return out


def build_split(session: Optional[str] = None, val_fraction: float = 0.25,
                seed: int = 0, val_mode: str = "auto", force: bool = False) -> int:
    """Emit a YOLO-format directory tree, splitting BY CLIP.

    Splitting by frame is the classic silent mistake: near-duplicate frames land
    on both sides, validation measures memorisation, the score looks excellent
    and the model fails live. Every frame of a clip goes to exactly one side.

    DESTRUCTIVE ONCE LABELS EXIST. Every label file is (re)created EMPTY, and
    frames are re-assigned between train and val, so re-running this after any
    labelling throws away every box in the tree -- pre-labels, autolabel passes,
    and hand-drawn corrections alike -- with no warning and no undo. The boxes
    are the expensive part of this dataset; the images are not, since
    `dataset/<session>/clips/` holds the originals.

    So it now REFUSES when non-empty labels are present, and backs them up
    first when forced. To ADD a session to an existing tree, use
    `turret_host.ingest_session`, which leaves everything already there alone.
    """
    existing = [p for split in ("train", "val")
                for p in _glob.glob(_os.path.join(DATASET_DIR, "yolo", split,
                                                  "labels", "*.txt"))
                if _os.path.getsize(p) > 0]
    if existing and not force:
        print("REFUSING: dataset/yolo already holds %d non-empty label files."
              % len(existing))
        print("Rebuilding the split rewrites every label EMPTY and reshuffles")
        print("train/val, which would discard all of that work.")
        print()
        print("  to ADD a session, keeping what is there:")
        print("      python -m turret_host.ingest_session <session>")
        print("  to rebuild anyway (a backup is taken first):")
        print("      python -m turret_host.capture_dataset --split --force")
        return 1
    if existing:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dst = _os.path.join(DATASET_DIR, "_backups", stamp)
        for split in ("train", "val"):
            src = _os.path.join(DATASET_DIR, "yolo", split, "labels")
            if _os.path.isdir(src):
                shutil.copytree(src, _os.path.join(dst, split, "labels"))
        state = _os.path.join(DATASET_DIR, "review_state.json")
        if _os.path.exists(state):
            shutil.copy2(state, _os.path.join(dst, "review_state.json"))
        print("--force: backed up %d labels -> dataset/_backups/%s"
              % (len(existing), stamp))
    # Leading underscore means "not a capture session" -- backups, scratch,
    # anything a human parked here. Including a backup would silently duplicate
    # every frame in it, and duplicates across a train/val split are exactly the
    # failure this tool exists to prevent.
    sessions = ([session] if session else
                sorted(d for d in _os.listdir(DATASET_DIR)
                       if _os.path.isdir(_os.path.join(DATASET_DIR, d))
                       and d != "yolo" and not d.startswith("_")))
    if not sessions:
        print("no sessions in %s" % DATASET_DIR)
        return 1

    clips = []
    for s in sessions:
        session_dir = _os.path.join(DATASET_DIR, s)
        clips_dir = _os.path.join(session_dir, "clips")
        man_path = _os.path.join(session_dir, "manifest.json")
        listed = {}
        if _os.path.exists(man_path):
            for c in json.load(open(man_path)).get("clips", []):
                listed[c["clip_id"]] = c
        # Recover clips the manifest does not know about. Each clip writes its
        # own clip.json when it is stopped, so a session killed by a closed
        # window -- or one still running right now -- is still fully usable.
        if _os.path.isdir(clips_dir):
            for cid in sorted(_os.listdir(clips_dir)):
                if cid in listed:
                    continue
                cj = _os.path.join(clips_dir, cid, "clip.json")
                if _os.path.exists(cj):
                    rec = json.load(open(cj))
                    rec["_recovered"] = True
                    listed[cid] = rec
                    print("  recovered unlisted clip: %s/%s" % (s, cid))
        for c in listed.values():
            c["_session"] = s
            clips.append(c)
    if not clips:
        print("no clips recorded yet")
        return 1

    # Stratify by shot_key so val gets a spread of conditions, not four distance
    # clips and nothing else.
    rng = random.Random(seed)
    by_shot: Dict[str, List[dict]] = {}
    for c in clips:
        by_shot.setdefault(c["shot_key"], []).append(c)
    val_ids = set()
    for shot_key in sorted(by_shot):          # sorted: the split must be reproducible
        group = by_shot[shot_key]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction))) if len(group) > 1 else 0
        # KEY BY (session, clip_id), NOT clip_id. Numbering restarts in every
        # session, so a second room names its clips dist_3m_00, occl_under_00
        # ... exactly as the first one did. With a bare clip_id, holding out one
        # clip holds out EVERY session's clip of that name: measured here, that
        # sent 81% of all frames to validation while looking like a normal
        # 25% holdout. The failure is silent in the direction that matters --
        # it does not error, it just trains on a fraction of the data.
        val_ids.update((c["_session"], c["clip_id"]) for c in group[:n_val])

    out = _os.path.join(DATASET_DIR, "yolo")
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            _os.makedirs(_os.path.join(out, split, sub), exist_ok=True)

    # With one clip per shot, clip-wise holdout yields almost no validation set:
    # every group has len 1, so nothing can be held out without deleting that
    # condition from training entirely. Fall back to a temporal split, loudly.
    total_frames = sum(sum(c["frames"].values()) for c in clips)
    val_frames_clipwise = sum(sum(c["frames"].values())
                              for c in clips
                              if (c["_session"], c["clip_id"]) in val_ids)
    use_temporal = (val_mode == "temporal" or
                    (val_mode == "auto" and val_frames_clipwise < 0.10 * max(total_frames, 1)))
    if use_temporal:
        print("NOTE: clip-wise holdout would give only %d/%d frames to validation,"
              % (val_frames_clipwise, total_frames))
        print("      because there is ~1 clip per shot. Falling back to a TEMPORAL")
        print("      split: the last %d%% of each clip. This score is OPTIMISTIC --"
              % int(val_fraction * 100))
        print("      same room, same person, same grip. Shoot a second short session")
        print("      on another day for a validation set you can actually trust.\n")
        val_ids = set()

    counts = {"train": 0, "val": 0}
    for c in clips:
        clip_is_val = (c["_session"], c["clip_id"]) in val_ids
        src_dir = _os.path.join(DATASET_DIR, c["_session"], "clips", c["clip_id"])
        if not _os.path.isdir(src_dir):
            continue
        tail = _temporal_val_frames(src_dir, val_fraction) if use_temporal else set()
        for name in sorted(_os.listdir(src_dir)):
            if not name.endswith(".jpg"):
                continue
            split = "val" if (clip_is_val or name in tail) else "train"
            stem = "%s__%s__%s" % (c["_session"], c["clip_id"], name[:-4])
            shutil.copy2(_os.path.join(src_dir, name),
                         _os.path.join(out, split, "images", stem + ".jpg"))
            # A negative frame gets an EMPTY label file -- that is how YOLO is
            # told "this image contains no object", which is the whole point of
            # capturing negatives. Positives get an empty file too until they
            # are labelled; the labelling tool overwrites them.
            open(_os.path.join(out, split, "labels", stem + ".txt"), "w").close()
            counts[split] += 1

    yaml_path = _os.path.join(out, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write("path: %s\ntrain: train/images\nval: val/images\n\nnames:\n  0: drone\n"
                 % out.replace("\\", "/"))

    if use_temporal:
        print("split TEMPORALLY (last %d%% of each clip) -- optimistic, see note above"
              % int(val_fraction * 100))
    else:
        print("split by clip (never by frame), stratified by shot")
        print("  val clips: %s" % ", ".join("%s/%s" % sc for sc in sorted(val_ids)))
    print("  train %d frames   val %d frames" % (counts["train"], counts["val"]))
    print("  wrote %s" % yaml_path)
    print("\nPositives still need boxes. Upload %s to Roboflow/CVAT, pre-label with a" % out)
    print("drone checkpoint, correct, and export. Negatives are already correct as")
    print("empty label files -- do not let a labelling tool discard them.")
    return 0


def repair_session(session: str, dry_run: bool = False) -> int:
    """Merge key-repeat clip fragments back into one clip, and deduplicate.

    A held SPACE key can toggle recording many times a second, producing dozens
    or hundreds of ~2-frame clips for a single shot. Two things are wrong with
    that and they need different fixes:

      * as SPLIT UNITS they are useless, and because the split is stratified by
        shot they would swamp validation with one shot's fragments;
      * the frames are near-DUPLICATES, because every clip-start forces a keep
        (the difference history starts empty), which is exactly the thing the
        difference sampler exists to prevent.

    So: concatenate the fragments in order, then re-run the same difference test
    across the merged sequence and keep only frames that earn their place.
    """
    session_dir = _os.path.join(DATASET_DIR, session)
    clips_dir = _os.path.join(session_dir, "clips")
    man_path = _os.path.join(session_dir, "manifest.json")
    if not _os.path.exists(man_path):
        print("no manifest for %s" % session)
        return 1
    manifest = json.load(open(man_path))

    groups: Dict[str, List[dict]] = {}
    for c in manifest["clips"]:
        groups.setdefault(c["shot_key"], []).append(c)

    repaired: List[dict] = []
    for shot_key in sorted(groups):
        group = sorted(groups[shot_key], key=lambda c: c["clip_id"])
        # A fragment run: many clips for one shot, nearly all of them tiny.
        tiny = [c for c in group if sum(c["frames"].values()) <= 4]
        if len(group) < 5 or len(tiny) < len(group) * 0.8:
            repaired.extend(group)
            continue

        total_before = sum(sum(c["frames"].values()) for c in group)
        print("%s: %d fragment clips, %d frames -> merging"
              % (shot_key, len(group), total_before))
        if dry_run:
            repaired.extend(group)
            continue

        target_id = group[0]["clip_id"]
        target_dir = _os.path.join(clips_dir, target_id)
        kept: Dict[str, int] = {}
        last: Dict[str, Optional[np.ndarray]] = {}
        staging = _os.path.join(clips_dir, "_merge_tmp")
        shutil.rmtree(staging, ignore_errors=True)
        _os.makedirs(staging)

        for c in group:
            src = _os.path.join(clips_dir, c["clip_id"])
            if not _os.path.isdir(src):
                continue
            for name in sorted(_os.listdir(src)):
                if not name.endswith(".jpg"):
                    continue
                cam = name.split("_")[0]
                img = cv2.imread(_os.path.join(src, name))
                if img is None:
                    continue
                th = _thumb(img)
                if not _differs_enough(th, last.get(cam)):
                    continue          # a near-duplicate forced by a clip restart
                n = kept.get(cam, 0)
                cv2.imwrite(_os.path.join(staging, "%s_%04d.jpg" % (cam, n)), img,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                kept[cam] = n + 1
                last[cam] = th

        for c in group:
            shutil.rmtree(_os.path.join(clips_dir, c["clip_id"]), ignore_errors=True)
        _os.replace(staging, target_dir)

        merged = dict(group[0])
        merged["frames"] = kept
        merged["seconds"] = round(sum(c.get("seconds", 0.0) for c in group), 1)
        merged["_repaired_from"] = len(group)
        json.dump(merged, open(_os.path.join(target_dir, "clip.json"), "w"), indent=2)
        repaired.append(merged)
        print("  -> %s: %d frames kept, %d dropped as near-duplicates"
              % (target_id, sum(kept.values()), total_before - sum(kept.values())))

    if dry_run:
        print("\n(dry run -- nothing changed)")
        return 0

    manifest["clips"] = repaired
    tmp = man_path + ".tmp"
    json.dump(manifest, open(tmp, "w"), indent=2)
    _os.replace(tmp, man_path)
    print()
    _report(manifest)
    return 0


def show_stats() -> int:
    if not _os.path.isdir(DATASET_DIR):
        print("no dataset directory yet")
        return 1
    for s in sorted(_os.listdir(DATASET_DIR)):
        if s == "yolo" or s.startswith("_"):
            continue
        man = _os.path.join(DATASET_DIR, s, "manifest.json")
        if _os.path.exists(man):
            report = json.load(open(man))
            report["session"] = s          # the directory is the truth, not the field
            _report(report)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=None, help="name this session (default: timestamp)")
    ap.add_argument("--cameras", default="both", choices=["both", "narrow", "wide"])
    ap.add_argument("--wide-fast", action="store_true",
                    help="wide camera at 1280x720@60 (a CENTRE CROP, ~85 deg, not a downscale)")
    ap.add_argument("--split", action="store_true", help="build a YOLO train/val split")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even though labels exist (backs them up first)")
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--val-mode", default="auto", choices=["auto", "clip", "temporal"],
                    help="how to hold out validation. 'clip' is strict but needs "
                         "more than one clip per shot; 'auto' falls back to 'temporal'")
    ap.add_argument("--negatives-only", action="store_true",
                    help="capture ONLY the no-drone shots. See "
                         "negatives_only_shot_list() for why this deserves a "
                         "session of its own")
    ap.add_argument("--frames", type=int, default=500,
                    help="with --negatives-only: total frames across the "
                         "negative shots (default 500)")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print the shot list and exit; opens no cameras")
    ap.add_argument("--repair", action="store_true",
                    help="merge key-repeat clip fragments and drop the duplicates they forced")
    ap.add_argument("--dry-run", action="store_true", help="with --repair: report, change nothing")
    ap.add_argument("--auto", action="store_true",
                    help="hands-free: walk the shot list on a timer, with beeps. "
                         "No keys needed -- for capturing on your own")
    ap.add_argument("--ready-seconds", type=float, default=6.0,
                    help="with --auto: time to get into position before each shot")
    ap.add_argument("--shot-seconds", type=float, default=20.0,
                    help="with --auto: how long to record each shot (the shot "
                         "list asks for 20-30 s of MOVEMENT per shot)")
    args = ap.parse_args()

    if args.repair:
        if not args.session:
            print("--repair needs --session <name>")
            return 1
        return repair_session(args.session, args.dry_run)
    if args.list:
        print_shot_list()
        return 0
    if args.stats:
        return show_stats()
    if args.split:
        return build_split(args.session, args.val_fraction, val_mode=args.val_mode,
                           force=args.force)
    pilot = (AutoPilot(args.ready_seconds, args.shot_seconds) if args.auto else None)
    shots = (negatives_only_shot_list(args.frames) if args.negatives_only
             else None)
    if shots:
        print("NEGATIVES ONLY: %d shots, ~%d frames each, %d total target."
              % (len(shots), shots[0].target_frames,
                 sum(s.target_frames for s in shots)))
        print("NO DRONE IN FRAME AT ANY POINT -- not on a table, not in shot,")
        print("not in your hand. Every frame here is labelled empty, so a")
        print("drone that sneaks in teaches the model it is background.\n")
    return run_session(args.session, args.cameras, args.wide_fast, auto=pilot,
                       shots=shots)


if __name__ == "__main__":
    _sys.exit(main())
