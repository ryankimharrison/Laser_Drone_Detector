"""Train or evaluate a detector against an audited, session-grouped dataset.

Use --freeze-test after reviewing test labels; --check audits without training.
Explicit weights/data/run names make camera-domain experiments reproducible.
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


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--freeze-test", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--data", default=DATA)
    from turret_host import config
    ap.add_argument("--weights", default=_os.path.join(ROOT, "models", config.YOLO_WEIGHTS))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.9)
    ap.add_argument("--degrees", type=float, default=10.)
    ap.add_argument("--hsv-h", type=float, default=.015)
    ap.add_argument("--hsv-s", type=float, default=.7)
    ap.add_argument("--hsv-v", type=float, default=.5)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="drone_flight")
    return ap


def main(argv=None):
    import json
    from pathlib import Path
    import re
    from turret_host.dataset_audit import audit, digest, freeze_test
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.workers is None:
        args.workers = 8 if args.imgsz <= 640 else 2
    if (min(args.epochs, args.imgsz, args.batch) <= 0 or args.workers < 0
            or not 0 <= args.scale <= 1 or not 0 <= args.degrees <= 180
            or not all(0 <= x <= 1 for x in (args.hsv_h, args.hsv_s, args.hsv_v))):
        ap.error("invalid training size, workers, or augmentation range")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        ap.error("run name must contain only letters, digits, underscore or hyphen")
    try:
        report = freeze_test(args.data) if args.freeze_test else audit(args.data)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    print("dataset", report['fingerprint'], report['counts'])
    if args.freeze_test:
        print("test images and labels frozen; no training started")
        return 0
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        ap.error("local weights not found: " + str(weights))
    import torch
    import ultralytics
    if args.device != "cpu" and not torch.cuda.is_available():
        ap.error("CUDA unavailable; use --device cpu for CPU evaluation")
    print("torch", torch.__version__, "ultralytics", ultralytics.__version__)
    if args.check:
        return 0
    run_dir = Path(RUNS) / args.name
    if run_dir.exists():
        ap.error("run already exists; select a new --name")
    run_dir.mkdir(parents=True)
    manifest = dict(arguments=vars(args), dataset=report,
                    weights=str(weights), weights_sha256=digest(weights),
                    torch=torch.__version__, ultralytics=ultralytics.__version__)
    (run_dir / 'experiment.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    from ultralytics import YOLO
    model = YOLO(str(weights))
    common = dict(data=str(Path(args.data).resolve()), imgsz=args.imgsz,
                  batch=args.batch, device=args.device, workers=args.workers,
                  project=RUNS, name=args.name, exist_ok=True)
    if args.evaluate:
        model.val(split=args.split, **common)
    else:
        model.train(epochs=args.epochs, seed=args.seed, deterministic=True,
                    patience=30, scale=args.scale, degrees=args.degrees,
                    fliplr=.5, flipud=0., hsv_h=args.hsv_h, hsv_s=args.hsv_s,
                    hsv_v=args.hsv_v, mosaic=1., close_mosaic=15,
                    val=True, plots=True, **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
