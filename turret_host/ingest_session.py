"""Add ONE capture session to dataset/yolo without disturbing what is there.

    python -m turret_host.ingest_session 2026-09-18_195245
    python -m turret_host.ingest_session --list
    python -m turret_host.ingest_session <session> --dry-run

WHY NOT `capture_dataset --split`
---------------------------------
That rebuilds the whole tree, and it creates every label file with

    open(.../labels/<stem>.txt, "w").close()

i.e. EMPTY. Re-running it after any labelling has happened silently discards
every box in the dataset. It also re-shuffles which frames land in train vs val.

By the time a second session exists there is usually work on disk worth more
than the tree itself -- pre-labelled boxes, an autolabel pass, and a
`review_state.json` whose confirmed records point at those labels. So this adds
the new session's frames ALONGSIDE, touching no existing file, and pre-labels
only what it just added.

SPLIT RULE
----------
Matches what is already in the tree: the last 25% of each clip goes to val.
That is a TEMPORAL split, not a clip-wise one, and it is optimistic -- train and
val share clips, so neighbouring frames either side of the cut are near
duplicates. `capture_dataset` falls back to it because there is one clip per
shot and splitting by clip would drop whole conditions out of training. Keeping
the same rule at least keeps the two sessions comparable. Judge the model live.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _pkg_dir:
    _sys.path[0] = _os.path.dirname(_pkg_dir)

import argparse
import datetime
import glob
import shutil
from typing import Dict, List

import cv2

from turret_host import prelabel

ROOT = _os.path.dirname(_pkg_dir)
DATASET_DIR = _os.path.join(ROOT, "dataset")
YOLO_DIR = _os.path.join(DATASET_DIR, "yolo")
VAL_FRACTION = 0.25


def _sessions() -> List[str]:
    out = []
    for n in sorted(_os.listdir(DATASET_DIR)):
        p = _os.path.join(DATASET_DIR, n)
        if _os.path.isdir(p) and _os.path.isdir(_os.path.join(p, "clips")) \
                and not n.startswith("_") and n != "yolo":
            out.append(n)
    return out


def _existing_stems() -> set:
    have = set()
    for split in ("train", "val"):
        d = _os.path.join(YOLO_DIR, split, "images")
        if _os.path.isdir(d):
            have.update(n[:-4] for n in _os.listdir(d) if n.endswith(".jpg"))
    return have


def _backup() -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    st = _os.path.join(DATASET_DIR, "review_state.json")
    if _os.path.exists(st):
        dst = _os.path.join(DATASET_DIR, "review_state_%s.json" % stamp)
        shutil.copy2(st, dst)
        print("backed up review_state.json -> %s" % _os.path.basename(dst))
    for split in ("train", "val"):
        src = _os.path.join(YOLO_DIR, split, "labels")
        if _os.path.isdir(src):
            dst = _os.path.join(DATASET_DIR, "labels_backup_%s" % stamp, split)
            _os.makedirs(_os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
    print("backed up labels -> labels_backup_%s/" % stamp)


def ingest(session: str, dry_run: bool = False) -> int:
    sess_dir = _os.path.join(DATASET_DIR, session)
    clips_dir = _os.path.join(sess_dir, "clips")
    if not _os.path.isdir(clips_dir):
        print("no such session: %s" % session)
        return 1

    have = _existing_stems()
    added: Dict[str, List[str]] = {"train": [], "val": []}
    skipped = 0

    for clip_id in sorted(_os.listdir(clips_dir)):
        clip_dir = _os.path.join(clips_dir, clip_id)
        if not _os.path.isdir(clip_dir):
            continue
        for cam in ("narrow", "wide"):
            files = sorted(glob.glob(_os.path.join(clip_dir, "%s_*.jpg" % cam)))
            if not files:
                continue
            # Last VAL_FRACTION of THIS camera's frames in THIS clip -> val,
            # which is the rule already applied to the tree.
            cut = int(round(len(files) * (1.0 - VAL_FRACTION)))
            for i, src in enumerate(files):
                stem = "%s__%s__%s" % (session, clip_id,
                                       _os.path.basename(src)[:-4])
                if stem in have:
                    skipped += 1
                    continue
                added["val" if i >= cut else "train"].append(src)

    total = len(added["train"]) + len(added["val"])
    print("session %s" % session)
    print("   new frames: %d  (train %d, val %d)   already present: %d"
          % (total, len(added["train"]), len(added["val"]), skipped))
    if dry_run or total == 0:
        return 0

    _backup()
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            _os.makedirs(_os.path.join(YOLO_DIR, split, sub), exist_ok=True)
        for src in added[split]:
            clip_id = _os.path.basename(_os.path.dirname(src))
            stem = "%s__%s__%s" % (session, clip_id, _os.path.basename(src)[:-4])
            shutil.copy2(src, _os.path.join(YOLO_DIR, split, "images", stem + ".jpg"))
            open(_os.path.join(YOLO_DIR, split, "labels", stem + ".txt"), "w").close()
    print("copied %d frames into dataset/yolo" % total)
    return 0


def prelabel_session(session: str, prompt: str = prelabel.GDINO_PROMPT) -> int:
    """Grounding DINO over ONLY this session's positive frames, both cameras.

    Same gate and same choice as `prelabel.py`: drop boxes that cannot be a
    drone at this range BEFORE taking the best one, because a confident
    full-frame box around the person would otherwise win, and keep exactly one
    box since there is exactly one drone.
    """
    has_drone = prelabel._clip_has_drone()
    # Prefer our own fine-tune once one exists -- it has seen this drone in this
    # room, which Grounding DINO never has. Falls back to the text prompt on a
    # fresh checkout where nothing has been trained yet. See
    # prelabel.newest_finetune() for the bootstrap and its failure mode.
    ft = prelabel.newest_finetune()
    if ft:
        detect = prelabel._UltralyticsBackend(ft, prelabel.PRELABEL_CONF)
        print("pre-labelling with %s @ conf %.2f"
              % (_os.path.relpath(ft, ROOT), prelabel.PRELABEL_CONF))
    else:
        detect = prelabel._GroundingDino(prompt)
        print("no local fine-tune yet -- using Grounding DINO text prompt")
    stats = {"pos": 0, "boxed": 0, "neg": 0, "implausible": 0, "unknown": 0}

    for split in ("train", "val"):
        img_dir = _os.path.join(YOLO_DIR, split, "images")
        lbl_dir = _os.path.join(YOLO_DIR, split, "labels")
        if not _os.path.isdir(img_dir):
            continue
        names = sorted(n for n in _os.listdir(img_dir)
                       if n.endswith(".jpg") and n.startswith(session + "__"))
        for i, name in enumerate(names):
            stem = name[:-4]
            clip = prelabel._clip_of(stem)
            lbl_path = _os.path.join(lbl_dir, stem + ".txt")
            if clip not in has_drone:
                stats["unknown"] += 1
                continue
            if not has_drone[clip]:
                open(lbl_path, "w").close()
                stats["neg"] += 1
                continue
            stats["pos"] += 1
            img_path = _os.path.join(img_dir, name)
            img_h, img_w = cv2.imread(img_path).shape[:2]
            boxes = detect(img_path)
            plausible = [b for b in boxes
                         if prelabel.MIN_AREA_FRAC
                         <= ((b[2] - b[0]) * (b[3] - b[1])) / float(img_w * img_h)
                         <= prelabel.MAX_AREA_FRAC]
            if not plausible:
                open(lbl_path, "w").close()
                if boxes:
                    stats["implausible"] += 1
                continue
            x1, y1, x2, y2, _s = max(plausible, key=lambda b: b[4])
            cx, cy = (x1 + x2) / 2.0 / img_w, (y1 + y2) / 2.0 / img_h
            bw, bh = (x2 - x1) / float(img_w), (y2 - y1) / float(img_h)
            with open(lbl_path, "w") as fh:
                fh.write("0 %.6f %.6f %.6f %.6f\n" % (cx, cy, bw, bh))
            stats["boxed"] += 1
            if (i + 1) % 25 == 0:
                print("   %s %d/%d  boxed %d" % (split, i + 1, len(names), stats["boxed"]))

    print("\npre-label done for %s" % session)
    for k in ("pos", "boxed", "implausible", "neg", "unknown"):
        print("   %-12s %d" % (k, stats[k]))
    if stats["pos"]:
        print("   box rate on positives: %.0f%%" % (100.0 * stats["boxed"] / stats["pos"]))
    print("\nEvery box is a SUGGESTION. Review them.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-prelabel", action="store_true")
    ap.add_argument("--prompt", default=prelabel.GDINO_PROMPT)
    args = ap.parse_args()

    if args.list or not args.session:
        have = _existing_stems()
        print("sessions in dataset/:")
        for s in _sessions():
            n = len(glob.glob(_os.path.join(DATASET_DIR, s, "clips", "*", "*.jpg")))
            inn = sum(1 for st in have if st.startswith(s + "__"))
            print("   %-22s %4d frames on disk, %4d already in dataset/yolo" % (s, n, inn))
        return 0

    rc = ingest(args.session, args.dry_run)
    if rc or args.dry_run or args.no_prelabel:
        return rc
    print()
    return prelabel_session(args.session, args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
