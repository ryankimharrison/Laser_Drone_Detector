"""Review and fix the pre-labelled drone boxes. Local, no upload, no account.

    python -m turret_host.review_labels

WHY NOT ROBOFLOW / CVAT
-----------------------
Roboflow's free tier makes datasets public, and this footage is a person in
their own bedroom. CVAT needs Docker. Label Studio needs a server and a project
set up. All of that to do one thing -- adjust a single box per image -- on a
dataset that must not leave the machine.

ORDERING IS THE FEATURE
-----------------------
Frames are sorted by how likely the suggestion is to be WRONG, worst first:

  1. no box at all            the model found nothing; draw one
  2. implausible size         too big or too small to be a drone at 2-5 m
  3. unusual aspect ratio     a drone is roughly square-ish; a tall thin box is
                              probably a figurine, a door frame, or a person
  4. everything else          most of these are already right

So if you run out of time, you have fixed the frames that mattered and the
remainder are the ones that were already good. Stopping early is a real option.

CONTROLS
  drag              draw a new box (replaces whatever is there)
  RIGHT / SPACE     save and go to the next frame
  LEFT              back
  D                 no drone visible here -> empty label (correct for a frame
                    where the drone left the frame or is unrecognisable)
  R                 undo my edits to this frame
  Q                 save and quit  (progress is saved as you go)
"""
from __future__ import annotations

import os as _os
import sys as _sys
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _pkg_dir:
    _sys.path[0] = _os.path.dirname(_pkg_dir)

import argparse
from typing import List, Optional, Tuple

import cv2

from turret_host import config

ROOT = _os.path.dirname(_pkg_dir)
YOLO_DIR = _os.path.join(ROOT, "dataset", "yolo")

DISPLAY_H = 960                 # tall window: the frames are portrait once rotated
MIN_AREA_FRAC, MAX_AREA_FRAC = 0.002, 0.10


def _read_label(path: str) -> Optional[Tuple[float, float, float, float]]:
    if not _os.path.exists(path) or _os.path.getsize(path) == 0:
        return None
    for line in open(path):
        p = line.split()
        if len(p) == 5:
            return tuple(float(v) for v in p[1:])       # cx, cy, w, h (normalised)
    return None


def _write_label(path: str, box: Optional[Tuple[float, float, float, float]]) -> None:
    with open(path, "w") as fh:
        if box:
            fh.write("0 %.6f %.6f %.6f %.6f\n" % box)


def _suspicion(box) -> int:
    """Lower sorts first. See ORDERING IS THE FEATURE."""
    if box is None:
        return 0
    _cx, _cy, w, h = box
    area = w * h
    if not (MIN_AREA_FRAC <= area <= MAX_AREA_FRAC):
        return 1
    aspect = max(w, h) / max(min(w, h), 1e-6)
    if aspect > 3.0:            # a drone is roughly square-ish from any angle
        return 2
    return 3


class Reviewer:
    """Rotates for display and maps clicks back, so boxes stay in raw coordinates.

    The C270 is mounted 90 degrees over. Reviewing sideways frames is miserable
    and error-prone, but the labels must stay in the raw frame the model trains
    on -- so the rotation lives here, in the display and the mouse mapping only.
    """

    def __init__(self, items: List[Tuple[str, str]]):
        self.items = items
        self.i = 0
        self.drag_from: Optional[Tuple[int, int]] = None
        self.drag_to: Optional[Tuple[int, int]] = None
        self.raw_w = self.raw_h = 0
        self.scale = 1.0
        self.edited = 0

    # -- geometry ---------------------------------------------------------
    def disp_to_raw(self, dx: int, dy: int) -> Tuple[float, float]:
        """Undo the display scale and the 90-degree rotation."""
        x = dx / self.scale
        y = dy / self.scale
        if config.NARROW_ROTATE_CLOCKWISE:
            # cv2.ROTATE_90_CLOCKWISE: out(row=y, col=x) = in(row=H-1-x, col=y)
            return y, (self.raw_h - 1) - x
        return (self.raw_w - 1) - y, x

    def raw_to_disp(self, rx: float, ry: float) -> Tuple[int, int]:
        if config.NARROW_ROTATE_CLOCKWISE:
            x = (self.raw_h - 1) - ry
            y = rx
        else:
            x = ry
            y = (self.raw_w - 1) - rx
        return int(x * self.scale), int(y * self.scale)

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_from, self.drag_to = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_from:
            self.drag_to = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_from:
            self.drag_to = (x, y)
            self.commit_drag()
            self.drag_from = self.drag_to = None

    def commit_drag(self):
        (ax, ay), (bx, by) = self.drag_from, self.drag_to
        if abs(ax - bx) < 6 or abs(ay - by) < 6:
            return                                   # a click, not a drag
        p1, p2 = self.disp_to_raw(ax, ay), self.disp_to_raw(bx, by)
        x1, x2 = sorted((p1[0], p2[0]))
        y1, y2 = sorted((p1[1], p2[1]))
        cx = (x1 + x2) / 2.0 / self.raw_w
        cy = (y1 + y2) / 2.0 / self.raw_h
        bw = (x2 - x1) / self.raw_w
        bh = (y2 - y1) / self.raw_h
        _img, lbl = self.items[self.i]
        _write_label(lbl, (cx, cy, bw, bh))
        self.edited += 1

    # -- loop -------------------------------------------------------------
    def run(self) -> int:
        cv2.namedWindow("review", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("review", self.on_mouse)
        originals = {}

        while True:
            img_path, lbl_path = self.items[self.i]
            raw = cv2.imread(img_path)
            if raw is None:
                self.i = (self.i + 1) % len(self.items)
                continue
            self.raw_h, self.raw_w = raw.shape[:2]
            if lbl_path not in originals:
                originals[lbl_path] = _read_label(lbl_path)

            disp = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE
                              if config.NARROW_ROTATE_CLOCKWISE
                              else cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.scale = DISPLAY_H / float(disp.shape[0])
            disp = cv2.resize(disp, (int(disp.shape[1] * self.scale), DISPLAY_H))

            box = _read_label(lbl_path)
            if box:
                cx, cy, bw, bh = box
                a = self.raw_to_disp((cx - bw / 2) * self.raw_w, (cy - bh / 2) * self.raw_h)
                b = self.raw_to_disp((cx + bw / 2) * self.raw_w, (cy + bh / 2) * self.raw_h)
                cv2.rectangle(disp, (min(a[0], b[0]), min(a[1], b[1])),
                              (max(a[0], b[0]), max(a[1], b[1])), (0, 255, 0), 2)
            if self.drag_from and self.drag_to:
                cv2.rectangle(disp, self.drag_from, self.drag_to, (0, 200, 255), 2)

            state = "BOX" if box else "NO BOX - draw one"
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(disp, "%d/%d   %s   edited %d" % (self.i + 1, len(self.items),
                                                          state, self.edited),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if box else (0, 180, 255), 2)
            cv2.putText(disp, "drag=box  D=no drone  R=undo  <-/->  Q=quit",
                        (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            cv2.imshow("review", disp)

            k = cv2.waitKey(20) & 0xFF
            if k in (ord(' '), 83, ord('n')):
                self.i = min(self.i + 1, len(self.items) - 1)
            elif k in (81, ord('b')):
                self.i = max(self.i - 1, 0)
            elif k == ord('d'):
                _write_label(lbl_path, None)
                self.edited += 1
                self.i = min(self.i + 1, len(self.items) - 1)
            elif k == ord('r'):
                _write_label(lbl_path, originals[lbl_path])
            elif k == ord('q'):
                break
        cv2.destroyAllWindows()
        print("reviewed up to %d/%d, %d edits saved" % (self.i + 1, len(self.items), self.edited))
        return 0


def collect(include_wide: bool = False) -> List[Tuple[str, str]]:
    """Positive frames only, worst-suggestion first.

    Negatives are skipped entirely: their empty label is correct by construction
    (we recorded has_drone per clip) and no model ever touched them, so there is
    nothing to review and 159 frames of nothing to click through.
    """
    from turret_host.prelabel import _clip_has_drone, _clip_of
    has_drone = _clip_has_drone()
    out = []
    for split in ("train", "val"):
        idir = _os.path.join(YOLO_DIR, split, "images")
        ldir = _os.path.join(YOLO_DIR, split, "labels")
        if not _os.path.isdir(idir):
            continue
        for n in sorted(_os.listdir(idir)):
            if not n.endswith(".jpg"):
                continue
            if not include_wide and "__wide_" in n:
                continue
            if not has_drone.get(_clip_of(n[:-4]), False):
                continue
            out.append((_os.path.join(idir, n), _os.path.join(ldir, n[:-4] + ".txt")))
    out.sort(key=lambda it: (_suspicion(_read_label(it[1])), it[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--include-wide", action="store_true")
    args = ap.parse_args()
    items = collect(args.include_wide)
    if not items:
        print("nothing to review -- run the pre-labeller first")
        return 1
    counts = {}
    for _i, lbl in items:
        s = _suspicion(_read_label(lbl))
        counts[s] = counts.get(s, 0) + 1
    print("%d frames to review, worst first:" % len(items))
    for s, name in ((0, "no box - draw one"), (1, "implausible size"),
                    (2, "odd aspect ratio"), (3, "looks reasonable")):
        if counts.get(s):
            print("   %4d  %s" % (counts[s], name))
    print()
    print(__doc__.split("CONTROLS")[1])
    return Reviewer(items).run()


if __name__ == "__main__":
    _sys.exit(main())
