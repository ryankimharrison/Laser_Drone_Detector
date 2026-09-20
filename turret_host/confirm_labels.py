"""Confirm or reject each pre-labelled box, one frame at a time. Both cameras.

    python -m turret_host.confirm_labels            # review
    python -m turret_host.confirm_labels --export   # write the clean dataset
    python -m turret_host.confirm_labels --stats    # what is left to do

    SPACE       the label shown is CORRECT
    BACKSPACE   the label shown is WRONG
    LEFT        back to the previous frame
    drag        draw the right box (turns a reject into a fix, optional)
    Q           save and quit            progress is saved continuously

WHY "WRONG" DOES NOT WRITE AN EMPTY LABEL
-----------------------------------------
The obvious implementation of "reject" is to delete the box. That is exactly the
failure already sitting in `dataset/yolo`: 421 of 858 drone-present frames carry
an empty label, so half the positive footage currently teaches the model that a
drone is background. Deleting a bad box converts "this box is wrong" into "there
is no drone here", which is a different and much more damaging claim.

So a reject is recorded as a VERDICT, not as a label edit. Rejected frames are
excluded from the exported dataset until someone draws a box. Nothing in this
module writes to `dataset/yolo`; `--export` builds a separate tree. The original
stays exactly as it was.

SPACE AND BACKSPACE ON A FRAME WITH NO BOX
------------------------------------------
They keep their meaning -- they judge the label as shown, and "no box" is a
label. SPACE confirms it (genuinely no drone in this frame: it left the shot, or
is unrecognisable) and the frame is kept as a true negative, which is worth
having. BACKSPACE rejects it (there IS a drone and it needs a box), and the
frame is held out until one is drawn.

THE 90-DEGREE MOUNT IS PER-CAMERA
---------------------------------
The C270 is mounted 90 deg over; the wide camera is not. Frames are stored as
they come off the sensor -- narrow 1280x720, wide 1920x1080 -- and labels are
normalised to that stored frame. Rotation therefore belongs to the DISPLAY and
the mouse mapping only, and it must be decided per frame from the camera in the
filename. `review_labels.py --include-wide` rotates every frame with the narrow
camera's constant, which shows wide frames sideways and maps drawn boxes to the
wrong place; that is why this module exists rather than a flag on that one.

ORDERING
--------
Worst-suggestion-first, reusing the ranking in `review_labels.py`: no box, then
implausible size, then odd aspect ratio, then the rest. Stopping early is a real
option -- what is left unreviewed is what was already most likely right.
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
import shutil
from typing import Dict, List, Optional, Tuple

import cv2

from turret_host import config

ROOT = _os.path.dirname(_pkg_dir)
DATASET_DIR = _os.path.join(ROOT, "dataset")
YOLO_DIR = _os.path.join(DATASET_DIR, "yolo")
EXPORT_DIR = _os.path.join(DATASET_DIR, "yolo_reviewed")
STATE_PATH = _os.path.join(DATASET_DIR, "review_state.json")

# Fit the window inside a laptop screen. A rotated narrow frame is portrait
# (720x1280) and a wide frame is landscape (1920x1080), so both dimensions are
# capped and the tighter constraint wins.
#
# THESE ARE DEFAULTS, overridable with --max-width/--max-height/--full. At
# 1400x900 a rotated narrow frame renders at 0.70 and a wide frame at 0.73 --
# both lose about 30% of their linear resolution, which matters when the call
# is "is that smear a drone or a hand". On a 1920x1080 display, --max-width
# 1920 --max-height 1080 puts WIDE at exactly 1:1 and narrow at 0.84.
#
# True 1:1 on NARROW is not reachable on a 1080-tall screen: rotated it is
# 1280 px tall. --full will do it and let the window overflow, because OpenCV
# windows do not scroll. If that is a problem, --no-rotate shows the narrow
# frame in its stored 1280x720 orientation, which IS 1:1 on a 1080p display --
# sideways, but every pixel present.
DISPLAY_H_MAX = 900
DISPLAY_W_MAX = 1400

CONFIRMED, REJECTED, EMPTY_OK, NEEDS_BOX = "confirmed", "rejected", "empty_ok", "needs_box"

# Verdicts whose frames are safe to train on.
KEEP = (CONFIRMED, EMPTY_OK)

# cv2.waitKeyEx codes. Arrow keys do NOT survive `waitKey() & 0xFF` on Windows
# (LEFT is 0x250000, and 0x250000 & 0xFF == 0), which is why the older reviewer's
# left arrow silently does nothing there. Several values per key: Windows first,
# then GTK/Qt, then a letter fallback.
K_SPACE = (32,)
K_BACKSPACE = (8,)
K_LEFT = (2424832, 65361, 81, ord('b'))
K_QUIT = (ord('q'), 27)
K_UNDO = (ord('r'),)


# ---------------------------------------------------------------------------
#   labels and frame identity
# ---------------------------------------------------------------------------
def _read_label(path: str) -> Optional[Tuple[float, float, float, float]]:
    """cx, cy, w, h normalised to the STORED (unrotated) frame, or None."""
    if not _os.path.exists(path) or _os.path.getsize(path) == 0:
        return None
    for line in open(path):
        p = line.split()
        if len(p) == 5:
            return tuple(float(v) for v in p[1:])
    return None


def _is_narrow(name: str) -> bool:
    """Which camera produced this frame. The rotation depends on it."""
    return "__narrow_" in name


def _suspicion(box) -> int:
    """Lower sorts first. Same ranking as review_labels.py."""
    if box is None:
        return 0
    _cx, _cy, w, h = box
    area = w * h
    if not (0.002 <= area <= 0.10):
        return 1
    if max(w, h) / max(min(w, h), 1e-6) > 3.0:
        return 2
    return 3


SUSPICION_NAMES = {0: "no box", 1: "implausible size",
                   2: "odd aspect ratio", 3: "looks reasonable"}


# ---------------------------------------------------------------------------
#   review state -- verdicts, never label edits
# ---------------------------------------------------------------------------
def load_state() -> Dict[str, dict]:
    if not _os.path.exists(STATE_PATH):
        return {}
    try:
        return json.load(open(STATE_PATH)).get("frames", {})
    except (ValueError, OSError):
        return {}


def save_state(frames: Dict[str, dict]) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"version": 1,
                   "updated_utc": datetime.datetime.now(
                       datetime.timezone.utc).isoformat(timespec="seconds"),
                   "frames": frames}, fh, indent=2)
    _os.replace(tmp, STATE_PATH)          # atomic: never a half-written state


# ---------------------------------------------------------------------------
#   collection
# ---------------------------------------------------------------------------
def collect(include_negatives: bool = False,
            skip_reviewed: bool = False) -> List[Tuple[str, str]]:
    """(image, label) for every frame worth a human verdict, worst first.

    Negative clips are skipped by default: `has_drone` was recorded at capture
    time, no model ever ran on them, and their empty label is correct by
    construction. There is nothing to confirm and it is 262 frames of clicking.
    """
    from turret_host.prelabel import _clip_has_drone, _clip_of
    has_drone = _clip_has_drone()
    # Already-judged frames are dropped BEFORE the sort, not skipped over during
    # it. The ordering is worst-first across the WHOLE tree, so a new session's
    # frames are interleaved with every previous session's by suspicion rank --
    # finishing the new "no box" frames lands you on old, already-confirmed ones
    # at the start of the next bucket. Starting at the first unreviewed index is
    # not enough, because "unreviewed" is scattered rather than contiguous.
    reviewed = set(load_state()) if skip_reviewed else set()
    out: List[Tuple[str, str]] = []
    for split in ("train", "val"):
        idir = _os.path.join(YOLO_DIR, split, "images")
        ldir = _os.path.join(YOLO_DIR, split, "labels")
        if not _os.path.isdir(idir):
            continue
        for n in sorted(_os.listdir(idir)):
            if not n.endswith(".jpg"):
                continue
            if n[:-4] in reviewed:
                continue
            if not include_negatives and not has_drone.get(_clip_of(n[:-4]), False):
                continue
            out.append((_os.path.join(idir, n),
                        _os.path.join(ldir, n[:-4] + ".txt")))
    out.sort(key=lambda it: (_suspicion(_read_label(it[1])), it[0]))
    return out


# ---------------------------------------------------------------------------
#   the reviewer
# ---------------------------------------------------------------------------
class Confirmer:
    """Display is rotated per camera; labels stay in stored-frame coordinates."""

    def __init__(self, items: List[Tuple[str, str]], frames: Dict[str, dict]):
        self.items = items
        self.frames = frames
        self.i = self._first_unreviewed()
        self.raw_w = self.raw_h = 0
        self.rotate = False
        self.scale = 1.0
        self.drag_from: Optional[Tuple[int, int]] = None
        self.drag_to: Optional[Tuple[int, int]] = None
        self.dirty = 0

    def _first_unreviewed(self) -> int:
        for n, (img, _l) in enumerate(self.items):
            if _os.path.basename(img)[:-4] not in self.frames:
                return n
        return 0

    # -- geometry ---------------------------------------------------------
    # cv2.ROTATE_90_CLOCKWISE on an (H, W) input gives a (W, H) output with
    #     out(row = x_in, col = H - 1 - y_in)
    # These are POINT mappings (pixel indices), which is what a mouse gives and
    # what a drawn corner is. Box edges are mapped as their two corner points and
    # re-min/maxed, since the rotation swaps which corner is which.
    def raw_to_disp(self, rx: float, ry: float) -> Tuple[int, int]:
        if not self.rotate:
            return int(rx * self.scale), int(ry * self.scale)
        if config.NARROW_ROTATE_CLOCKWISE:
            x, y = (self.raw_h - 1) - ry, rx
        else:
            x, y = ry, (self.raw_w - 1) - rx
        return int(x * self.scale), int(y * self.scale)

    def disp_to_raw(self, dx: int, dy: int) -> Tuple[float, float]:
        x, y = dx / self.scale, dy / self.scale
        if not self.rotate:
            return x, y
        if config.NARROW_ROTATE_CLOCKWISE:
            return y, (self.raw_h - 1) - x
        return (self.raw_w - 1) - y, x

    # -- mouse ------------------------------------------------------------
    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_from, self.drag_to = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_from:
            self.drag_to = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_from:
            self.drag_to = (x, y)
            self._commit_drag()
            self.drag_from = self.drag_to = None

    def _commit_drag(self) -> None:
        (ax, ay), (bx, by) = self.drag_from, self.drag_to
        if abs(ax - bx) < 6 or abs(ay - by) < 6:
            return                                    # a click, not a drag
        p1, p2 = self.disp_to_raw(ax, ay), self.disp_to_raw(bx, by)
        x1, x2 = sorted((p1[0], p2[0]))
        y1, y2 = sorted((p1[1], p2[1]))
        self._verdict(CONFIRMED, box=((x1 + x2) / 2.0 / self.raw_w,
                                      (y1 + y2) / 2.0 / self.raw_h,
                                      (x2 - x1) / self.raw_w,
                                      (y2 - y1) / self.raw_h), advance=False)

    # -- verdicts ---------------------------------------------------------
    def _stem(self) -> str:
        return _os.path.basename(self.items[self.i][0])[:-4]

    def _verdict(self, verdict: str, box=None, advance: bool = True) -> None:
        rec = {"verdict": verdict}
        prev = self.frames.get(self._stem())
        if box is not None:
            rec["box"] = list(box)                    # a human-drawn correction
        elif verdict == CONFIRMED:
            # A HUMAN BOX OUTRANKS THE LABEL FILE. Drawing records the box and
            # deliberately does NOT advance, so the natural next keystroke is
            # SPACE ("yes, that one") -- and re-reading the label file here
            # answered a different question: what the MODEL suggested, which on
            # a no-box frame is nothing. The drawn box was silently overwritten
            # with "no drone here", which is the exact claim this module exists
            # to stop anyone making by accident. This is the same rule the
            # display already follows: once drawn, the human box is the label.
            if prev and "box" in prev:
                rec["box"] = list(prev["box"])
            else:
                cur = _read_label(self.items[self.i][1])
                if cur:
                    rec["box"] = list(cur)            # freeze what was confirmed
        self.frames[self._stem()] = rec
        self.dirty += 1
        if self.dirty % 10 == 0:
            save_state(self.frames)
        if advance:
            self.i = min(self.i + 1, len(self.items) - 1)

    # -- loop -------------------------------------------------------------
    def run(self) -> int:
        win = "confirm labels"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)

        while True:
            img_path, lbl_path = self.items[self.i]
            name = _os.path.basename(img_path)
            raw = cv2.imread(img_path)
            if raw is None:
                self.i = min(self.i + 1, len(self.items) - 1)
                continue

            self.raw_h, self.raw_w = raw.shape[:2]
            self.rotate = _is_narrow(name)            # per camera, not global

            if self.rotate:
                disp = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE
                                  if config.NARROW_ROTATE_CLOCKWISE
                                  else cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                disp = raw.copy()
            # NEVER UPSCALE. Enlarging past 1:1 invents detail that is not in
            # the sensor data and makes a smeared 30 px drone look like a
            # confident one, which is the opposite of what a quality pass
            # wants. Shrink to fit, or show 1:1, never more.
            self.scale = min(DISPLAY_H_MAX / float(disp.shape[0]),
                             DISPLAY_W_MAX / float(disp.shape[1]), 1.0)
            if self.scale < 1.0:
                disp = cv2.resize(disp, (int(disp.shape[1] * self.scale),
                                         int(disp.shape[0] * self.scale)),
                                  interpolation=cv2.INTER_AREA)

            rec = self.frames.get(self._stem())
            # A human correction, once drawn, is what we show from then on.
            box = tuple(rec["box"]) if rec and "box" in rec else _read_label(lbl_path)
            if box:
                cx, cy, bw, bh = box
                a = self.raw_to_disp((cx - bw / 2) * self.raw_w, (cy - bh / 2) * self.raw_h)
                b = self.raw_to_disp((cx + bw / 2) * self.raw_w, (cy + bh / 2) * self.raw_h)
                colour = (0, 0, 255) if rec and rec["verdict"] == REJECTED else (0, 255, 0)
                cv2.rectangle(disp, (min(a[0], b[0]), min(a[1], b[1])),
                              (max(a[0], b[0]), max(a[1], b[1])), colour, 2)
            if self.drag_from and self.drag_to:
                cv2.rectangle(disp, self.drag_from, self.drag_to, (0, 200, 255), 2)

            self._draw_hud(disp, name, box, rec)
            cv2.imshow(win, disp)

            k = cv2.waitKeyEx(20)
            if k == -1:
                continue
            if k in K_QUIT:
                break
            elif k in K_SPACE:
                self._verdict(CONFIRMED if box else EMPTY_OK)
            elif k in K_BACKSPACE:
                self._verdict(REJECTED if box else NEEDS_BOX)
            elif k in K_LEFT:
                self.i = max(self.i - 1, 0)
            elif k in K_UNDO:
                self.frames.pop(self._stem(), None)
                self.dirty += 1

        save_state(self.frames)
        cv2.destroyAllWindows()
        self._summary()
        return 0

    def _draw_hud(self, disp, name: str, box, rec) -> None:
        done = len(self.frames)
        cam = "narrow" if self.rotate else "wide"
        session = name.split("__")[0]
        head = "%d/%d   reviewed %d   %s   %s   %s" % (
            self.i + 1, len(self.items), done, session, cam,
            "BOX" if box else "NO BOX")
        if rec:
            head += "   [%s]" % rec["verdict"]
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(disp, head, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if box else (0, 180, 255), 2)

        foot = ("SPACE=correct   BACKSPACE=wrong   LEFT=back   drag=fix   Q=quit"
                if box else
                "SPACE=no drone here   BACKSPACE=drone IS here   LEFT=back   drag=box")
        cv2.rectangle(disp, (0, disp.shape[0] - 30), (disp.shape[1], disp.shape[0]),
                      (0, 0, 0), -1)
        cv2.putText(disp, foot, (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(disp, name[:64], (10, disp.shape[0] - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)

    def _summary(self) -> None:
        tally: Dict[str, int] = {}
        for r in self.frames.values():
            tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        print("\nreviewed %d of %d frames" % (len(self.frames), len(self.items)))
        for v in (CONFIRMED, EMPTY_OK, REJECTED, NEEDS_BOX):
            if tally.get(v):
                print("   %5d  %s" % (tally[v], v))
        held = tally.get(REJECTED, 0) + tally.get(NEEDS_BOX, 0)
        if held:
            print("\n%d frames are held out of training until a box is drawn." % held)
        print("state -> %s" % STATE_PATH)


# ---------------------------------------------------------------------------
#   export
# ---------------------------------------------------------------------------
def export() -> int:
    """Write a clean YOLO tree: confirmed boxes and confirmed negatives only.

    Rejected, needs-box and UNREVIEWED frames are all left out. Unreviewed is
    deliberately excluded rather than passed through -- passing it through is how
    `dataset/yolo` ended up with 421 unexamined empty labels on drone-present
    frames, and a smaller honest set beats a larger set with that in it.
    """
    from turret_host.prelabel import _clip_has_drone, _clip_of
    frames = load_state()
    if not frames:
        print("no review state at %s -- nothing to export" % STATE_PATH)
        return 1
    has_drone = _clip_has_drone()

    if _os.path.isdir(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            _os.makedirs(_os.path.join(EXPORT_DIR, split, sub), exist_ok=True)

    counts = {"train": 0, "val": 0}
    boxed = negatives = held_boxless = 0
    for split in ("train", "val"):
        idir = _os.path.join(YOLO_DIR, split, "images")
        ldir = _os.path.join(YOLO_DIR, split, "labels")
        if not _os.path.isdir(idir):
            continue
        for n in sorted(_os.listdir(idir)):
            if not n.endswith(".jpg"):
                continue
            stem = n[:-4]
            rec = frames.get(stem)
            if rec is None:
                # Never reviewed. Negative clips are correct by construction and
                # were never shown to a human on purpose; everything else is
                # unexamined and stays out.
                if has_drone.get(_clip_of(stem), False):
                    continue
                box = None
            elif rec["verdict"] not in KEEP:
                continue
            elif rec["verdict"] == CONFIRMED and "box" not in rec:
                # CONFIRMED means "the box shown is right", so a record with no
                # box is self-contradictory -- it can only come from a drawn box
                # that was lost, or from state written by another tool. Emitting
                # it would write an EMPTY label on a drone-present frame, which
                # is the false negative this module exists to prevent. Hold it
                # out instead. Checked here rather than repaired in the state
                # file so it holds no matter who wrote that file last.
                held_boxless += 1
                continue
            else:
                box = tuple(rec["box"]) if "box" in rec else None

            shutil.copy2(_os.path.join(idir, n),
                         _os.path.join(EXPORT_DIR, split, "images", n))
            with open(_os.path.join(EXPORT_DIR, split, "labels", stem + ".txt"), "w") as fh:
                if box:
                    fh.write("0 %.6f %.6f %.6f %.6f\n" % box)
                    boxed += 1
                else:
                    negatives += 1
            counts[split] += 1
            _ = ldir                                   # labels come from state

    with open(_os.path.join(EXPORT_DIR, "data.yaml"), "w") as fh:
        fh.write("path: %s\ntrain: train/images\nval: val/images\n\nnames:\n  0: drone\n"
                 % EXPORT_DIR.replace("\\", "/"))

    print("exported -> %s" % EXPORT_DIR)
    print("   train %d   val %d" % (counts["train"], counts["val"]))
    print("   %d boxed   %d negatives" % (boxed, negatives))
    if held_boxless:
        print("   %d held out: verdict CONFIRMED but no box in the record"
              % held_boxless)
    print("\nNOTE: the split is TEMPORAL within each clip (capture_dataset fell back")
    print("to it because there is one clip per shot), so val shares clips with train.")
    print("Judge the model on live footage, not on this mAP.")
    return 0


def stats() -> int:
    items = collect()
    frames = load_state()
    by_susp: Dict[int, int] = {}
    todo = 0
    for img, lbl in items:
        stem = _os.path.basename(img)[:-4]
        if stem in frames:
            continue
        todo += 1
        s = _suspicion(_read_label(lbl))
        by_susp[s] = by_susp.get(s, 0) + 1
    print("%d drone-present frames, %d reviewed, %d to go" %
          (len(items), len(items) - todo, todo))
    for s in sorted(by_susp):
        print("   %5d  %s" % (by_susp[s], SUSPICION_NAMES[s]))
    tally: Dict[str, int] = {}
    for r in frames.values():
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    if tally:
        print("verdicts so far:")
        for v, c in sorted(tally.items()):
            print("   %5d  %s" % (c, v))
    return 0


def _frame_no(img: str) -> Optional[int]:
    stem = _os.path.basename(img)[:-4]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def stride_within_clips(items: List[Tuple[str, str]], stride: int,
                        reviewed=None) -> List[Tuple[str, str]]:
    """Offer only every Nth frame of each clip, counted IN TIME.

    A clip is one continuous capture at 30 fps, so consecutive frames are
    near-duplicates: the drone has moved a few pixels and nothing else has
    changed. Judging both costs two decisions and buys barely more than one.
    Measured on the current dataset -- of 370 frames left to review, all 370
    came from a single `follow` clip, which is why it feels like the same
    picture over and over. It nearly is.

    N = 15 is about 0.5 s at 30 fps.

    Strided IN TIME, then restored to the caller's order. `collect()` sorts
    worst-suggestion-first across the whole tree, so slicing that list
    directly would take every Nth frame BY SUSPICION RANK -- which is not a
    subsample of the clip at all, and would happily hand back fifteen
    consecutive frames that happened to score adjacently.
    """
    order = {img: i for i, (img, _l) in enumerate(items)}

    def clip_of(img: str) -> str:
        stem = _os.path.basename(img)[:-4]
        return stem.split("__narrow")[0].split("__wide")[0]

    # DROP ANYTHING SITTING NEXT TO A FRAME ALREADY JUDGED.
    #
    # Without this, a SECOND strided pass is worthless: the first pass removes
    # every 15th frame, so striding what REMAINS hands back their immediate
    # neighbours. Measured after one pass over a 370-frame clip, the next batch
    # sat a median of 2 frames -- 70 ms -- from something already judged. That
    # is the same picture, and the operator noticed before the tool did.
    #
    # The gap is half the stride: far enough that a new pick carries new
    # information, close enough that a second pass can still refine a clip.
    if reviewed:
        judged: Dict[str, set] = {}
        for stem in reviewed:
            clip = stem.split("__narrow")[0].split("__wide")[0]
            tail = stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                judged.setdefault(clip, set()).add(int(tail))
        gap = max(1, stride // 2)
        kept = []
        for it in items:
            n, seen = _frame_no(it[0]), judged.get(clip_of(it[0]))
            if n is None or not seen or all(abs(n - s) >= gap for s in seen):
                kept.append(it)
        if kept:
            items = kept

    by_clip: Dict[str, List[Tuple[str, str]]] = {}
    for it in items:
        by_clip.setdefault(clip_of(it[0]), []).append(it)

    kept: List[Tuple[str, str]] = []
    for clip in sorted(by_clip):
        # Sort by filename, which ends in a zero-padded frame index, so this
        # is chronological within the clip.
        chrono = sorted(by_clip[clip], key=lambda it: _os.path.basename(it[0]))
        kept.extend(chrono[::stride])

    kept.sort(key=lambda it: order[it[0]])
    print("stride %d: %d frames of %d, across %d clip(s). The %d skipped are "
          "near-duplicates in time, not lost -- re-run without --stride to "
          "reach them." % (stride, len(kept), len(items), len(by_clip),
                           len(items) - len(kept)))
    return kept


def main() -> int:
    # Declared up front: argparse reads these as defaults below, and a `global`
    # after a read is a SyntaxError, not a runtime one.
    global DISPLAY_W_MAX, DISPLAY_H_MAX
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", action="store_true",
                    help="write dataset/yolo_reviewed from the verdicts")
    ap.add_argument("--stats", action="store_true", help="what is left to do")
    ap.add_argument("--revisit", action="store_true",
                    help="show frames you have already judged too")
    ap.add_argument("--include-negatives", action="store_true",
                    help="also step through negative clips (normally pointless)")
    ap.add_argument("--max-width", type=int, default=DISPLAY_W_MAX,
                    metavar="PX", help="display cap; 1920 on a 1080p screen "
                                       "puts the WIDE camera at exactly 1:1")
    ap.add_argument("--max-height", type=int, default=DISPLAY_H_MAX,
                    metavar="PX")
    ap.add_argument("--full", action="store_true",
                    help="no downscaling at all -- 1:1 pixels. The rotated "
                         "narrow frame is 1280 px tall and WILL overflow a "
                         "1080p screen; OpenCV windows do not scroll")
    ap.add_argument("--stride", type=int, default=1, metavar="N",
                    help="within each clip, offer only every Nth frame in time. "
                         "15 is about 0.5 s at 30 fps -- see why below")
    args = ap.parse_args()

    if args.full:
        DISPLAY_W_MAX = DISPLAY_H_MAX = 10 ** 6
    else:
        DISPLAY_W_MAX, DISPLAY_H_MAX = args.max_width, args.max_height

    if args.export:
        return export()
    if args.stats:
        return stats()

    items = collect(args.include_negatives, skip_reviewed=not args.revisit)
    if args.stride > 1:
        items = stride_within_clips(items, args.stride, load_state())
    if not items:
        print("nothing to review -- run the pre-labeller first")
        return 1
    frames = load_state()
    print("%d frames, worst suggestion first. %d already reviewed." %
          (len(items), sum(1 for i, _l in items
                           if _os.path.basename(i)[:-4] in frames)))
    print(__doc__.split("    SPACE")[1].split("WHY")[0].rstrip())
    return Confirmer(items, frames).run()


if __name__ == "__main__":
    raise SystemExit(main())
