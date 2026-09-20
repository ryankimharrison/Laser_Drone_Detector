"""Re-label every drone-present frame: upright detection + temporal Viterbi.

    python -m turret_host.autolabel                  # label, write, contact sheets
    python -m turret_host.autolabel --clips dist_3m_00,occl_under_00
    python -m turret_host.autolabel --dry-run        # score only, write nothing

TWO THINGS prelabel.py DID NOT DO, AND WHY EACH MATTERS
------------------------------------------------------
1. IT SCORED SIDEWAYS FRAMES. The C270 is mounted 90 degrees over and frames are
   stored as they come off the sensor, so every narrow frame Grounding DINO has
   ever been shown had a drone lying on its side. This is the same shape of bug
   as YuNet being fed a rotated frame and returning [] on an occupied room --
   a pretrained model asked to work outside the orientation it was trained on.
   Measured here on 12 frames: upright scored higher on 9, and the boxes land on
   the airframe rather than beside it. Detection runs upright; the box is mapped
   back so LABELS STAY IN STORED-FRAME COORDINATES, which is the convention
   confirm_labels.py and the training tree already use.

   The wide camera is not rotated, so it is scored as-is. The rotation is
   decided per frame from the camera in the filename, never globally.

2. IT CHOSE EACH FRAME INDEPENDENTLY. Per-frame argmax is why prelabel's own
   contact sheet showed boxes on a figurine and a laundry pile: those beat the
   real drone on a single frame, and nothing carried the knowledge that the
   drone was somewhere else a third of a second earlier. Frames within a clip
   are an ordered sequence of one object moving smoothly, so this runs VITERBI
   over each clip: the chosen trajectory maximises detection score AND temporal
   coherence together. A high-scoring distractor that appears for one frame in
   a place the drone cannot have reached loses to a lower-scoring box that fits
   the path.

THE NULL STATE, WHICH IS THE POINT
----------------------------------
Every frame also carries a NULL state -- "the drone is not visible here". Clips
are tagged has_drone per CLIP, not per frame, and the drone genuinely leaves the
shot: a frame in bg_cluttered_00 has none of it in view. Without NULL the
labeller is forced to box something, and what it boxes is furniture. Keeping it
means a frame can honestly come out empty, and an empty frame is a true negative
worth training on rather than a mislabelled positive.

WHAT THIS IS NOT
----------------
A suggestion machine, exactly like prelabel. It writes label files for a human
to confirm or reject; it does not make them correct. Contact sheets are written
next to the labels for that review, because looking at a sheet is the only thing
that has ever settled a labeller on this project.
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
import glob
import json
import math
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from turret_host import config

ROOT = _os.path.dirname(_pkg_dir)
DATASET_DIR = _os.path.join(ROOT, "dataset")
YOLO_DIR = _os.path.join(DATASET_DIR, "yolo")
SHEET_DIR = _os.path.join(DATASET_DIR, "autolabel_sheets")

GDINO_MODEL = "IDEA-Research/grounding-dino-base"
# NAME THE DISTRACTORS. With only the drone phrases in the prompt, every region
# must ground to "drone" or score nothing, so a white pillow on a bed comes back
# as a 0.45 drone. Giving the bed, the wall and the person their own phrases
# lets those regions ground where they belong and be dropped by name. Measured
# on the ten frames where the lock failed: the winning candidate changed from
# "a small white drone" (the pillow) to "a bed a blanket" on 6 of 10.
GDINO_PROMPT = ("a small white drone. a quadcopter. "
                "a ceiling fan. a light fixture. "
                "a pillow. a bed. a blanket. a person. a wall. a shelf.")
# Reject by name rather than accept by name: Grounding DINO's phrase grounding
# returns tokeniser debris for real hits ("a quadcoer", "##er", ""), and an
# accept-list throws those away. Anything not positively identified as a
# distractor stays a candidate and is judged on score, motion and trajectory.
#
# THE CEILING FAN IS THE WORST ONE IN THIS ROOM and it is not a joke: it is
# radially symmetric with four arms, so it is the best "quadcopter" a room can
# contain. It took every box in four wide tracks -- orient_nose_on, orient_top,
# orient_side, orient_tumbling -- 52 frames, all on the fan.
#
# It also defeats the motion cue, because a spinning fan moves. What gives it
# away is that its median box width was 520, 520, 538, 522 across the four
# tracks: a near-identical width in every track is a FIXED object, not a
# handheld one at varying range. Naming it is the fix; the width agreement is
# the check that should have caught it sooner.
DISTRACTOR_WORDS = ("pillow", "bed", "blanket", "person", "wall", "shelf",
                    "ceiling", "fan", "light fixture")
# Deliberately far below prelabel's 0.25. Low recall is fatal to Viterbi -- a
# frame whose true box was thresholded away can only be filled by NULL or by a
# distractor -- while precision is recovered by the path, not by the threshold.
CAND_THRESHOLD = 0.08
TEXT_THRESHOLD = 0.20
TOP_K = 8

# -- Viterbi costs. Additive; lower is better. ----------------------------
# BALANCE THESE AGAINST EACH OTHER, NOT IN ISOLATION. A NULL run is free
# (NULL->NULL has no transition cost) while a box run pays a transition every
# step, so the comparison per frame is:
#
#     NULL   : NULL_EMISSION
#     a box  : (1 - score) + W_MOVE * move + W_SIZE * size
#
# The first version of this used NULL_EMISSION = 0.62 with W_MOVE = 2.0 and
# every track came out entirely NULL: a 0.45-score box costs 0.55 in emission,
# leaving a transition budget of 0.07, which ordinary hand motion between
# sampled frames blows through immediately. The move penalty must be small
# enough to shape a trajectory without deciding whether there is one.
#
# With these numbers NULL wins when the best candidate scores below roughly
# 0.25 -- near prelabel's fixed threshold, but now a frame can keep a weaker
# box that fits the path and drop a stronger one that does not.
NULL_EMISSION = 0.85
W_MOVE = 0.60         # per unit of centre travel, normalised to frame diagonal
W_SIZE = 0.15         # per unit of |log(area ratio)|
NULL_SWITCH = 0.15    # entering or leaving NULL

# THE SMOOTHNESS PRIOR HAS A BIAS AND THIS IS THE CORRECTION.
# A static distractor is PERFECTLY temporally coherent -- zero travel, zero size
# change, zero transition cost -- so trajectory smoothing on its own actively
# prefers furniture to the target. That is not hypothetical: the first run of
# this module locked ten consecutive frames onto a pillow while the drone was
# held above the person's head.
#
# The drone is in a moving hand and the furniture is not, so frame-to-frame
# pixel difference separates them where score and coherence both fail. The
# score is the mean absolute difference inside the box over the mean across the
# whole frame, which normalises out exposure shifts and whole-scene motion.
# Measured on the ten failing frames: static candidates 0.2-0.7, drone 1.0-5.1.
W_MOTION = 0.25
MOTION_FLOOR = 0.05   # guards log() on a candidate in a perfectly still region
# Sampling is irregular (kept-on-difference, 0.25-3.0 s apart), so a two-frame
# gap permits more travel than a one-frame gap. Divide the move penalty by this.
def _gap_slack(gap: int) -> float:
    return math.sqrt(max(1, gap))


# ==========================================================================
#   frame enumeration
# ==========================================================================
def _clip_has_drone() -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for s in sorted(_os.listdir(DATASET_DIR)):
        if s == "yolo" or s.startswith("_") or s == "autolabel_sheets":
            continue
        sess = _os.path.join(DATASET_DIR, s)
        man = _os.path.join(sess, "manifest.json")
        if _os.path.exists(man):
            for c in json.load(open(man)).get("clips", []):
                out[c["clip_id"]] = bool(c["has_drone"])
        clips = _os.path.join(sess, "clips")
        if _os.path.isdir(clips):
            for cid in sorted(_os.listdir(clips)):
                cj = _os.path.join(clips, cid, "clip.json")
                if cid not in out and _os.path.exists(cj):
                    out[cid] = bool(json.load(open(cj))["has_drone"])
    return out


def _hard_examples() -> Dict[str, List[float]]:
    """stem -> box, from every clip's `hard_examples.json`.

    These are frames the LIVE detector missed while the drone was demonstrably
    in shot, boxed by interpolating between the confident hits either side.
    They outrank anything this module produces, for the obvious reason: a
    detector that already failed on a frame is the worst available witness to
    what is in it. Grounding DINO is a different model, but it is still a
    prediction about the frame; the sidecar is evidence from its neighbours.
    """
    out: Dict[str, List[float]] = {}
    for session in sorted(_os.listdir(DATASET_DIR)):
        clips = _os.path.join(DATASET_DIR, session, "clips")
        if not _os.path.isdir(clips):
            continue
        for clip in sorted(_os.listdir(clips)):
            p = _os.path.join(clips, clip, "hard_examples.json")
            if not _os.path.exists(p):
                continue
            try:
                frames = json.load(open(p)).get("frames", {})
            except (ValueError, OSError):
                continue
            for short, box in frames.items():
                out["%s__%s__%s" % (session, clip, short)] = box
    return out


def _tracks(only: Optional[Sequence[str]] = None):
    """(session, clip, camera) -> [(index, image_path, label_path), ...] sorted.

    Frames of one clip are split across train/ and val/, so both are swept and
    merged. The track is the CLIP, not the split -- temporal coherence does not
    care which side of a split a frame landed on.
    """
    has_drone = _clip_has_drone()
    groups: Dict[Tuple[str, str, str], List[Tuple[int, str, str]]] = defaultdict(list)
    for split in ("train", "val"):
        for ip in glob.glob(_os.path.join(YOLO_DIR, split, "images", "*.jpg")):
            stem = _os.path.basename(ip)[:-4]
            parts = stem.split("__")
            if len(parts) != 3:
                continue
            session, clip, tail = parts
            if not has_drone.get(clip, False):
                continue                      # negatives never go near the model
            if only and clip not in only:
                continue
            cam, _, idx = tail.rpartition("_")
            lp = _os.path.join(YOLO_DIR, split, "labels", stem + ".txt")
            groups[(session, clip, cam)].append((int(idx), ip, lp))
    for k in groups:
        groups[k].sort()
    return groups


# ==========================================================================
#   detection, upright
# ==========================================================================
class _Detector:
    """Grounding DINO, scored on an UPRIGHT frame, answering in stored coords."""

    def __init__(self, prompt: str = GDINO_PROMPT):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._torch = torch
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(GDINO_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_MODEL).to(self.device)
        self.model.eval()

    def __call__(self, bgr: np.ndarray, rotate: bool):
        """[(score, stored-frame box), ...] best first, distractors removed."""
        from PIL import Image
        h0 = bgr.shape[0]
        view = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE) if rotate else bgr
        img = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        inp = self.proc(images=img, text=self.prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inp)
        r = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=CAND_THRESHOLD,
            text_threshold=TEXT_THRESHOLD, target_sizes=[img.size[::-1]])[0]
        labels = r.get("text_labels", r.get("labels"))
        cands = []
        for b, s, lab in zip(r["boxes"], r["scores"], labels):
            text = str(lab).lower()
            if any(w in text for w in DISTRACTOR_WORDS):
                continue                  # grounded to furniture, by its own name
            x1, y1, x2, y2 = (float(v) for v in b)
            if rotate:
                # cv2.ROTATE_90_CLOCKWISE maps stored (x, y) -> (h0-1-y, x), so
                # the inverse of a rotated point (x', y') is (y', h0-1-x').
                xs = (y1, y2)
                ys = (h0 - 1 - x1, h0 - 1 - x2)
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
            cands.append((float(s), (x1, y1, x2, y2)))
        cands.sort(key=lambda c: -c[0])
        return cands[:TOP_K]


def motion_ratio(diff: np.ndarray, box) -> float:
    """Mean |frame difference| inside the box, over the mean across the frame.

    Above 1 means the box moved more than the scene did; below 1 means it is
    part of the furniture. Normalising by the frame mean is what makes this
    survive an exposure step or a whole-scene shift, both of which raise every
    pixel and would otherwise read as motion everywhere.
    """
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    roi = diff[y1:y2, x1:x2]
    if roi.size == 0:
        return MOTION_FLOOR
    return max(MOTION_FLOOR, float(roi.mean()) / max(1e-3, float(diff.mean())))


# ==========================================================================
#   Viterbi over a clip
# ==========================================================================
def _centre(b):
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def _area(b):
    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def _transition(a, b, gap: int, diag: float) -> float:
    ca, cb = _centre(a), _centre(b)
    move = math.hypot(cb[0] - ca[0], cb[1] - ca[1]) / diag
    size = abs(math.log(_area(b) / _area(a)))
    return W_MOVE * move / _gap_slack(gap) + W_SIZE * size


def emission(score: float, motion: float) -> float:
    """Cost of claiming this candidate is the drone.

    tanh of the log ratio so the motion term saturates: a box that moved 40x
    the scene is not 10x better evidence than one that moved 4x, and without
    the squash a single blurred frame would dominate an entire trajectory.
    """
    return (1.0 - score) - W_MOTION * math.tanh(math.log(max(MOTION_FLOOR, motion)))


def viterbi(frames: List[Tuple[int, List[Tuple[float, tuple, float]]]],
            diag: float) -> List[Optional[tuple]]:
    """Best trajectory through per-frame candidates, NULL always available.

    States are `None` plus this frame's candidates. Costs are additive and the
    whole thing is one forward pass with a backpointer, which is enough: the
    sequences are tens of frames, not thousands.
    """
    n = len(frames)
    if n == 0:
        return []
    states_at = [[None] + [b for _, b, _ in c] for _, c in frames]
    emit_at = [[NULL_EMISSION] + [emission(s, m) for s, _, m in c] for _, c in frames]

    prev_cost = list(emit_at[0])
    back: List[List[int]] = [[-1] * len(states_at[0])]

    for t in range(1, n):
        gap = max(1, frames[t][0] - frames[t - 1][0])
        cost = [math.inf] * len(states_at[t])
        bp = [0] * len(states_at[t])
        for j, sj in enumerate(states_at[t]):
            for i, si in enumerate(states_at[t - 1]):
                if si is None and sj is None:
                    step = 0.0
                elif si is None or sj is None:
                    step = NULL_SWITCH
                else:
                    step = _transition(si, sj, gap, diag)
                c = prev_cost[i] + step + emit_at[t][j]
                if c < cost[j]:
                    cost[j], bp[j] = c, i
        back.append(bp)
        prev_cost = cost

    j = int(np.argmin(prev_cost))
    path = [states_at[n - 1][j]]
    for t in range(n - 1, 0, -1):
        j = back[t][j]
        path.append(states_at[t - 1][j])
    path.reverse()
    return path


# ==========================================================================
#   writing
# ==========================================================================
def _to_yolo(b, w: int, h: int) -> str:
    x1, y1, x2, y2 = b
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    return "0 %.6f %.6f %.6f %.6f\n" % ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                        (x2 - x1) / w, (y2 - y1) / h)


def _sheet(tiles, path, cols=6):
    if not tiles:
        return
    hgt = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hgt - t.shape[0], 0, 4,
                                cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    rows = [cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, wmax - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
    cv2.imwrite(path, cv2.vconcat(rows))


def blank_tracks(names: Sequence[str]) -> int:
    """Empty every label in the named `<clip>__<camera>` tracks.

    `has_drone` is recorded PER CLIP, and the narrow camera's effective
    horizontal field is 28.8 degrees, so it routinely misses what the wide
    camera catches -- bg_cluttered_00 narrow is eighteen frames of a wall, a
    chair and a desk with no drone anywhere in them. Boxes there are furniture
    by definition, and the frames are worth keeping as true negatives rather
    than rejecting one at a time.

    An empty label file is a real claim ("no drone in this frame"), so this is
    only ever driven from a contact sheet somebody actually looked at.
    """
    wanted = set(names)
    n = 0
    for (session, clip, cam), items in sorted(_tracks().items()):
        if "%s__%s" % (clip, cam) not in wanted:
            continue
        for _, _, lp in items:
            with open(lp, "w"):
                pass
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", default="", help="comma-separated clip ids")
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--blank", default="",
                    help="comma-separated <clip>__<camera> tracks to empty "
                         "(reviewed as containing no drone at all)")
    a = ap.parse_args(argv)
    only = [c for c in a.clips.split(",") if c] or None

    if a.blank:
        n = blank_tracks([t for t in a.blank.split(",") if t])
        print("emptied %d labels across %d track(s)" % (n, len(a.blank.split(","))))
        return 0

    groups = _tracks(only)
    total = sum(len(v) for v in groups.values())
    print("%d tracks, %d drone-present frames" % (len(groups), total))
    if not total:
        return 1

    if not a.dry_run:
        backup = _os.path.join(DATASET_DIR, "labels_backup_before_autolabel")
        if not _os.path.isdir(backup):
            for split in ("train", "val"):
                dst = _os.path.join(backup, split)
                _os.makedirs(dst, exist_ok=True)
                for f in glob.glob(_os.path.join(YOLO_DIR, split, "labels", "*.txt")):
                    shutil.copy2(f, dst)
            print("backed up original labels -> %s" % _os.path.basename(backup))
        _os.makedirs(SHEET_DIR, exist_ok=True)

    hard = _hard_examples()
    if hard:
        print("%d mined hard examples will override the detector on their frames"
              % len(hard))

    det = _Detector()
    done = boxed = nulled = hard_used = 0
    for (session, clip, cam), items in sorted(groups.items()):
        rotate = (cam == "narrow") and config.NARROW_ROTATE_CLOCKWISE
        greys = [cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2GRAY) for _, ip, _ in items]
        h, w = greys[0].shape[:2]

        per_frame = []
        for k, (idx, ip, lp) in enumerate(items):
            # Difference against the neighbour in this track. The first frame
            # borrows the one after it, so every frame has a motion field and
            # none is silently scored as perfectly static.
            other = greys[k - 1] if k > 0 else greys[min(1, len(greys) - 1)]
            diff = cv2.absdiff(greys[k], other).astype(np.float32)
            cands = det(cv2.imread(ip), rotate)
            per_frame.append((idx, [(s, b, motion_ratio(diff, b)) for s, b in cands]))
            done += 1
        path = viterbi(per_frame, math.hypot(w, h))

        tiles = []
        for (idx, ip, lp), box in zip(items, path):
            stem = _os.path.basename(ip)[:-4]
            mined = hard.get(stem)
            if mined is not None:
                # Normalised cx,cy,w,h straight back to pixels, so the sheet
                # draws what will actually be written.
                cx, cy, bw, bh = mined
                box = ((cx - bw / 2) * w, (cy - bh / 2) * h,
                       (cx + bw / 2) * w, (cy + bh / 2) * h)
                hard_used += 1
            if box is None:
                nulled += 1
            else:
                boxed += 1
            if not a.dry_run:
                with open(lp, "w") as fh:
                    fh.write("" if box is None else _to_yolo(box, w, h))
            if a.no_sheets:
                continue
            vis = cv2.imread(ip)
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if rotate:
                vis = cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE)
            vis = cv2.resize(vis, (230, int(vis.shape[0] * 230 / vis.shape[1])))
            cv2.putText(vis, "%04d%s" % (idx, "" if box is not None else " NULL"),
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255) if box is not None else (0, 165, 255), 2)
            tiles.append(vis)

        # Best-candidate score per frame, so a track that comes out all-NULL is
        # diagnosable without a second run: weak scores mean the detector did
        # not see it, strong scores mean these costs are mistuned.
        tops = [c[0][0] if c else 0.0 for _, c in per_frame]
        print("   %-22s %-6s %3d frames  %3d boxed  %3d null   "
              "top score med %.2f max %.2f   [%d/%d]"
              % (clip, cam, len(items), sum(b is not None for b in path),
                 sum(b is None for b in path),
                 float(np.median(tops)), float(np.max(tops)), done, total))
        if not a.dry_run and not a.no_sheets:
            _sheet(tiles, _os.path.join(SHEET_DIR, "%s__%s.jpg" % (clip, cam)))

    print("\n%d boxed, %d null, %d total   (%d from mined hard examples)"
          % (boxed, nulled, boxed + nulled, hard_used))
    if not a.dry_run:
        print("contact sheets -> %s" % SHEET_DIR)
        print("REVIEW THEM before trusting any of this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
