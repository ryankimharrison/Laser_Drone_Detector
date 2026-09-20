"""Pre-label the captured drone frames so a human corrects boxes instead of drawing them.

    python -m turret_host.prelabel               # label dataset/yolo in place
    python -m turret_host.prelabel --review      # flip through the results

HOW NOT TO MEASURE A LABELLER -- twice burned on this project
-------------------------------------------------------------
Measured on this footage (152 drone frames, 23 conditions). Three columns,
because the first two both LIE:

                              any box   drone-SIZED   actually ON the drone
    COCO yolo11n                 0.7%        --                 --
    YOLO-World, any prompt       0.0%        --                 --
        (control prompt 'person' scored 89.5%, so the model was running)
    sapoepsilon YOLOv11s @0.05  73.7%       5.9%               ~0%
    TomSmail YOLOv8n    @0.05   92.1%      75.0%          a figurine, a
                                                          laundry pile, a face
    Grounding DINO (prompt)       --         --            ~17/24  <- USED

"Any box" flatters everything, because the characteristic failure is a
near-full-frame box around the PERSON holding the drone. Adding a size gate
fixed that and produced the SECOND trap: boxes that are drone-SHAPED but sitting
on a blue figurine on a shelf. Size plausibility says nothing about location.

Only looking at a contact sheet settled it. Do that before trusting any number
this module prints.

As a PRE-LABELLER an imprecise model is still right, because the two failure
modes cost differently: a missed box means a human draws one, a spurious box
means a human presses delete. So recall is what matters and precision barely
does -- provided the boxes that survive are actually on the target.

Two facts about this dataset make it better still, and both come free:

  * `has_drone` is recorded per clip, so NEGATIVE frames never go near the model
    and that 81% false-positive rate simply does not apply to them.
  * There is exactly ONE drone, so we keep only the highest-confidence box per
    frame. Extra boxes inside a positive frame are discarded before a human ever
    sees them.

WHAT THIS DOES NOT DO: it does not make the labels correct. Every box it writes
is a suggestion from a model that is wrong about a quarter of the time and
loose about where the airframe ends. Review them.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _pkg_dir:
    _sys.path[0] = _os.path.dirname(_pkg_dir)

import argparse
import json
from typing import Dict, List, Optional, Tuple

import cv2

from turret_host import config  # noqa: F401  (kept for path/consistency with the package)

ROOT = _os.path.dirname(_pkg_dir)
DATASET_DIR = _os.path.join(ROOT, "dataset")
YOLO_DIR = _os.path.join(DATASET_DIR, "yolo")
DEFAULT_WEIGHTS = _os.path.join(ROOT, "models", "drone_yolov8n.pt")


def newest_finetune() -> Optional[str]:
    """The most recent locally-trained checkpoint, or None if there is none.

    THE BOOTSTRAP. Grounding DINO was the right pre-labeller when nothing had
    been trained on this drone -- it reads "a small white drone" from text and
    needs no examples, which is why it beat every YOLO variant on the first
    pass. But it is a general model that has never seen THIS airframe in THIS
    room, and it tops out around 72% box rate here.

    A model fine-tuned on the frames a human already corrected has seen exactly
    that, so each round should pre-label better than the last: label -> correct
    -> train -> label the next batch with the result. Picking the newest
    checkpoint automatically means the loop closes without editing this file
    after every run.

    THE RISK, stated plainly: self-training compounds its own mistakes. If the
    model proposes and the human only ever presses SPACE, the model's biases
    become the ground truth and nobody notices. Two things keep it honest --
    worst-first ordering puts the frames it was least sure about in front of
    the human first, and BACKSPACE holds a frame out rather than accepting it.
    Neither works if the review is rubber-stamped.
    """
    import glob as _glob
    cands = (_glob.glob(_os.path.join(ROOT, "runs", "*", "weights", "best.pt"))
             + _glob.glob(_os.path.join(ROOT, "models", "drone_y11n*.pt")))
    if not cands:
        return None
    return max(cands, key=_os.path.getmtime)

# Deliberately low. See the module docstring: a missed box costs a human more
# than a spurious one, and only the best PLAUSIBLE box per frame survives.
PRELABEL_CONF = 0.05

# A geometric sanity gate, and it is doing real work. Measured on this footage,
# counting "did it emit a box" and "did it emit a DRONE-SIZED box" give wildly
# different answers, because the obvious failure is a near-full-frame box around
# the PERSON:
#
#                            any box   drone-sized box
#     sapoepsilon @0.05       73.7%          5.9%     <- boxes the person
#     TomSmail    @0.05       92.1%         75.0%     <- boxes the drone
#
# The drone is ~132 px across in a 1280 px frame at 3 m, so ~1% of frame area,
# and roughly 0.4%-4% over the 2-5 m envelope. These bounds are loose around
# that, wide enough for a close pass or an odd aspect, narrow enough that a box
# around a human torso can never survive.
MIN_AREA_FRAC = 0.002
MAX_AREA_FRAC = 0.10

# ---------------------------------------------------------------------------
#   BACKEND: Grounding DINO, and why it beat every YOLO variant here
# ---------------------------------------------------------------------------
# Open-vocabulary detection is a different problem from fixed-class detection,
# and Grounding DINO is built for it: a text phrase, grounded against image
# regions. The YOLO-family attempts all failed on this footage --
#
#     YOLO-World (any prompt)        0/152   the control prompt 'person'
#                                            scored 89.5%, so it was running
#     public drone checkpoints       boxed the person, a figurine, a laundry
#                                            pile -- 5.9% drone-sized
#     Grounding DINO                 ~17/24 boxes correctly ON the drone
#
# The target is a WHITE quadcopter indoors against a light wall. Every public
# drone checkpoint is trained on dark drones against bright sky, so the contrast
# polarity is inverted and the background is cluttered rather than empty. A
# model that reads "a small white drone" from text has no such prior to fight.
#
# The prompt is phrased for Grounding DINO's parser: lowercase, period-separated
# phrases. Two phrases beat one; naming the colour matters.
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
GDINO_PROMPT = "a small white drone. a quadcopter."
GDINO_BOX_THRESHOLD = 0.25
GDINO_TEXT_THRESHOLD = 0.20


def _clip_has_drone() -> Dict[str, bool]:
    """clip_id -> has_drone, from every session manifest and clip.json."""
    out: Dict[str, bool] = {}
    if not _os.path.isdir(DATASET_DIR):
        return out
    for s in sorted(_os.listdir(DATASET_DIR)):
        if s == "yolo" or s.startswith("_"):
            continue
        sess = _os.path.join(DATASET_DIR, s)
        man = _os.path.join(sess, "manifest.json")
        if _os.path.exists(man):
            for c in json.load(open(man)).get("clips", []):
                out[c["clip_id"]] = bool(c["has_drone"])
        clips_dir = _os.path.join(sess, "clips")
        if _os.path.isdir(clips_dir):
            for cid in _os.listdir(clips_dir):
                cj = _os.path.join(clips_dir, cid, "clip.json")
                if cid not in out and _os.path.exists(cj):
                    out[cid] = bool(json.load(open(cj))["has_drone"])
    return out


def _clip_of(stem: str) -> str:
    """Split image stems written by capture_dataset: <session>__<clip_id>__<cam>_NNNN."""
    parts = stem.split("__")
    return parts[1] if len(parts) >= 2 else ""


class _GroundingDino:
    """Text-prompted detector. Returns [(x1, y1, x2, y2, score), ...] per image."""

    def __init__(self, prompt: str = GDINO_PROMPT):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._torch = torch
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(GDINO_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_MODEL).to(self.device)

    def __call__(self, path: str):
        from PIL import Image
        image = Image.open(path).convert("RGB")
        inputs = self.proc(images=image, text=self.prompt,
                           return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]])[0]
        return [tuple(float(v) for v in box) + (float(score),)
                for box, score in zip(res["boxes"], res["scores"])]


class _UltralyticsBackend:
    def __init__(self, weights: str, conf: float):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, path: str):
        r = self.model.predict(path, conf=self.conf, verbose=False, device=0)[0]
        return [tuple(float(v) for v in b.xyxy[0]) + (float(b.conf),) for b in r.boxes]


def prelabel(weights: str = DEFAULT_WEIGHTS, conf: float = PRELABEL_CONF,
             narrow_only: bool = True, backend: str = "gdino",
             prompt: str = GDINO_PROMPT) -> int:
    if not _os.path.isdir(YOLO_DIR):
        print("no split yet -- run: python -m turret_host.capture_dataset --split")
        return 1

    if backend == "gdino":
        detect = _GroundingDino(prompt)
        label = "%s  prompt=%r" % (GDINO_MODEL, prompt)
    else:
        if not _os.path.exists(weights):
            print("weights not found: %s" % weights)
            return 1
        detect = _UltralyticsBackend(weights, conf)
        label = "%s @ conf %.2f" % (_os.path.basename(weights), conf)

    has_drone = _clip_has_drone()
    stats = {"pos": 0, "boxed": 0, "neg": 0, "unknown": 0, "wide_skipped": 0, "implausible": 0}

    for split in ("train", "val"):
        img_dir = _os.path.join(YOLO_DIR, split, "images")
        lbl_dir = _os.path.join(YOLO_DIR, split, "labels")
        if not _os.path.isdir(img_dir):
            continue
        names = sorted(n for n in _os.listdir(img_dir) if n.endswith(".jpg"))
        for i, name in enumerate(names):
            stem = name[:-4]
            clip = _clip_of(stem)

            # The wide camera only feeds the wide->narrow handoff, which is the
            # most droppable milestone. Labelling it doubles the human's work
            # for the least valuable half of the dataset.
            if narrow_only and "__wide_" in stem:
                stats["wide_skipped"] += 1
                continue

            lbl_path = _os.path.join(lbl_dir, stem + ".txt")
            if clip not in has_drone:
                stats["unknown"] += 1
                continue
            if not has_drone[clip]:
                # A negative. Empty file is the correct YOLO label; the model is
                # never consulted, so its false-positive rate is irrelevant here.
                open(lbl_path, "w").close()
                stats["neg"] += 1
                continue

            stats["pos"] += 1
            img_path = _os.path.join(img_dir, name)
            img_h, img_w = cv2.imread(img_path).shape[:2]
            boxes = detect(img_path)
            # Drop boxes that cannot be a drone at 2-5 m BEFORE choosing. Doing
            # it before the max() matters: a confident full-frame box around the
            # person would otherwise win over a correct one.
            plausible = [b for b in boxes
                         if MIN_AREA_FRAC <= ((b[2] - b[0]) * (b[3] - b[1]))
                         / float(img_w * img_h) <= MAX_AREA_FRAC]
            if not plausible:
                open(lbl_path, "w").close()      # a human draws this one
                stats["implausible"] += 1 if boxes else 0
                continue

            # Exactly one drone: keep the single most confident plausible box.
            x1, y1, x2, y2, _score = max(plausible, key=lambda b: b[4])
            w, h = img_w, img_h
            cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            with open(lbl_path, "w") as fh:
                fh.write("0 %.6f %.6f %.6f %.6f\n" % (cx, cy, bw, bh))
            stats["boxed"] += 1

            if (i + 1) % 100 == 0:
                print("  %s %d/%d" % (split, i + 1, len(names)))

    print("\npre-labelled with %s" % label)
    print("  positive frames      %d" % stats["pos"])
    print("  ...got a box         %d  (%.0f%%)  <- human CORRECTS these"
          % (stats["boxed"], 100.0 * stats["boxed"] / max(stats["pos"], 1)))
    print("  ...empty, needs one  %d          <- human DRAWS these"
          % (stats["pos"] - stats["boxed"]))
    print("  negatives (no model) %d          <- already correct, do not delete" % stats["neg"])
    if stats["wide_skipped"]:
        print("  wide frames skipped  %d          (--all-cameras to include)"
              % stats["wide_skipped"])
    if stats["unknown"]:
        print("  unknown clip         %d" % stats["unknown"])
    print("\nEvery box is a SUGGESTION from a model measured at ~74%% recall on this")
    print("footage, and it is loose about where the airframe ends. Review them.")
    return 0


def review(limit: int = 0) -> int:
    """Flip through pre-labelled frames. LEFT/RIGHT to move, Q to quit.

    Not an editor -- a sanity check, so you can see how good the suggestions are
    before committing hours to correcting them in a real tool.
    """
    items: List[Tuple[str, str]] = []
    for split in ("train", "val"):
        img_dir = _os.path.join(YOLO_DIR, split, "images")
        lbl_dir = _os.path.join(YOLO_DIR, split, "labels")
        if not _os.path.isdir(img_dir):
            continue
        for name in sorted(_os.listdir(img_dir)):
            if name.endswith(".jpg") and "__wide_" not in name:
                items.append((_os.path.join(img_dir, name),
                              _os.path.join(lbl_dir, name[:-4] + ".txt")))
    if not items:
        print("nothing to review")
        return 1
    if limit:
        step = max(1, len(items) // limit)
        items = items[::step]

    i = 0
    while True:
        img_path, lbl_path = items[i]
        img = cv2.imread(img_path)
        if img is None:
            i = (i + 1) % len(items)
            continue
        h, w = img.shape[:2]
        n_boxes = 0
        if _os.path.exists(lbl_path):
            for line in open(lbl_path):
                p = line.split()
                if len(p) != 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in p[1:])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 0), 2)
                n_boxes += 1
        if config.NARROW_ROTATE_CLOCKWISE:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (405, 720))
        tag = "%d/%d  %s  %s" % (i + 1, len(items), _os.path.basename(img_path)[:40],
                                 "BOX" if n_boxes else "no box")
        cv2.putText(img, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 230, 0) if n_boxes else (0, 160, 255), 1)
        cv2.imshow("prelabel review", img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
        i = (i + 1) % len(items) if k != ord('b') else (i - 1) % len(items)
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=None,
                    help="default: the newest locally-trained checkpoint")
    ap.add_argument("--conf", type=float, default=PRELABEL_CONF)
    ap.add_argument("--all-cameras", action="store_true",
                    help="also label wide-camera frames (doubles the human's work)")
    ap.add_argument("--backend", default=None, choices=["gdino", "yolo"],
                    help="default: yolo once a fine-tune exists, else gdino")
    ap.add_argument("--prompt", default=GDINO_PROMPT)
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="with --review: sample this many")
    args = ap.parse_args()
    # Prefer our own fine-tune over Grounding DINO once one exists.
    ft = newest_finetune()
    if args.backend is None:
        args.backend = 'yolo' if ft else 'gdino'
    if args.weights is None:
        args.weights = ft or DEFAULT_WEIGHTS
    if args.backend == 'yolo':
        print('pre-labelling with %s @ conf %.2f' % (args.weights, args.conf))
    if args.review:
        return review(args.limit)
    return prelabel(args.weights, args.conf, narrow_only=not args.all_cameras,
                    backend=args.backend, prompt=args.prompt)


if __name__ == "__main__":
    _sys.exit(main())
