"""Fine-tune YOLO11n on the reviewed drone frames.

    python -m turret_host.train_detector                 # train
    python -m turret_host.train_detector --check         # environment only

WHY SCALE AUGMENTATION IS TURNED UP
-----------------------------------
The captured footage is much CLOSER than the demo will be. With the narrow
camera's 28.8 deg effective horizontal field, a ~250 mm DJI Mini at 2 m should
cover about 2% of frame area. The reviewed boxes have a median of 4.2% and a
tail running to 38%, which puts a lot of the footage under ~1 m.

The demo is specified at 2-5 m. Training on close-ups and inferring at range is
a scale domain gap, and a detector that has never seen the target small will
miss it exactly when it matters.

`scale=0.9` makes Ultralytics sample each image between 0.1x and 1.9x instead of
the default 0.5x-1.5x, so the model sees the drone down to roughly a fifth of
its captured size -- which covers 5 m footage synthesised from 1 m footage. This
is a mitigation, not a substitute: real frames at real range would be better,
and if there is time before the showcase, capture some.

WHY NOT imgsz=1280
------------------
`config.YOLO_IMGSZ` is 1280 for INFERENCE, where the cost is one forward pass.
For training it multiplies every epoch by ~4x for a target that is already large
in frame. Train at 640 first to find out whether the approach works at all; if
recall at range is the problem, raise it then. The exported weights run at
whatever imgsz inference asks for.

READ THE VALIDATION NUMBER WITH SUSPICION
-----------------------------------------
`capture_dataset --split` fell back to a TEMPORAL split (last 25% of each clip)
because there is one clip per shot, so train and val share clips and the frames
either side of the cut are near-duplicates. mAP here measures memorisation as
much as generalisation. The number that counts is live footage.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _pkg_dir:
    _sys.path[0] = _os.path.dirname(_pkg_dir)

import argparse

ROOT = _os.path.dirname(_pkg_dir)
DATA = _os.path.join(ROOT, "dataset", "yolo_reviewed", "data.yaml")
WEIGHTS = _os.path.join(ROOT, "models", "yolo11n.pt")
RUNS = _os.path.join(ROOT, "runs")


def check() -> int:
    import torch
    print("torch %s  cuda %s" % (torch.__version__, torch.version.cuda))
    ok = torch.cuda.is_available()
    print("cuda available: %s" % ok)
    if ok:
        print("device: %s  capability %s" % (torch.cuda.get_device_name(0),
                                             torch.cuda.get_device_capability(0)))
        print("VRAM GB: %.1f" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
    import ultralytics
    print("ultralytics %s" % ultralytics.__version__)
    print("data: %s  exists %s" % (DATA, _os.path.exists(DATA)))
    print("weights: %s  exists %s" % (WEIGHTS, _os.path.exists(WEIGHTS)))
    return 0 if (ok and _os.path.exists(DATA) and _os.path.exists(WEIGHTS)) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.9,
                    help="see the module docstring -- this is the range-gap mitigation")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers. Defaults to 8 at imgsz<=640 and "
                         "2 above it -- see the train() call for the crash "
                         "that forces this")
    ap.add_argument("--name", default="drone_y11n")
    args = ap.parse_args()
    if args.workers is None:
        args.workers = 8 if args.imgsz <= 640 else 2

    if args.check:
        return check()

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        # WORKERS SCALE WITH IMAGE SIZE, and the default 8 does not survive
        # imgsz=1280 on this machine: each worker prefetches full-resolution
        # batches into system RAM, and the run died with "DataLoader worker
        # exited unexpectedly" during the first VALIDATION pass -- the val
        # loader spins up its own workers on top of the training ones. That
        # failure looks like a CUDA problem and is not; GPU memory was 3.0 of
        # 8.1 GB at the time.
        workers=args.workers,
        project=RUNS,
        name=args.name,
        exist_ok=True,
        patience=30,
        scale=args.scale,          # the range-gap mitigation
        fliplr=0.5,
        flipud=0.0,                # the drone is held upright-ish; do not flip vertically
        degrees=10.0,              # modest: the narrow camera's mount rotation is fixed
        mosaic=1.0,
        close_mosaic=15,
        hsv_v=0.5,                 # the room's lighting range varies a lot
        val=True,
        plots=True,
    )
    print("\nweights -> %s" % _os.path.join(RUNS, args.name, "weights", "best.pt"))
    print("Judge this on LIVE footage, not on the mAP above: the split is temporal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
