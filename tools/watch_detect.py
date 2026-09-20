"""Live detection monitor: hold the drone up and watch what each camera says.

Answers "the wide camera sees it but the narrow ignores it" with numbers. For
every frame it reports, per camera, whether the detector fired, at what
confidence, and how big the box was -- and it runs the NARROW frame at two
inference sizes at once so the imgsz effect is visible live rather than
inferred from a val sweep.

NOTHING MOVES. No serial port is opened, no motion is commanded, the laser is
never touched. This is cameras and the detector only, so it is safe to run
while standing in front of the turret.

    python tools/watch_detect.py --seconds 60
    python tools/watch_detect.py --seconds 30 --no-compare   # config imgsz only

Output, under diag/detect_<timestamp>/:
    detections.jsonl   one record per frame per camera
    frames/            annotated JPEGs, boxes drawn, saved on a hit and
                       periodically on a miss so both cases are reviewable
    session.json       config snapshot and the closing summary
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2                                               # noqa: E402
import numpy as np                                       # noqa: E402

from turret_host import config                           # noqa: E402
from turret_host.types import Frame                      # noqa: E402

BOX_HIT = (60, 220, 60)
BOX_ALT = (0, 165, 255)


def draw(img, dets, label, colour):
    out = img.copy()
    for d in dets:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, colour, 2)
        cv2.putText(out, "%s %.2f" % (d.label, d.conf),
                    (p1[0], max(14, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2)
    cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--compare", dest="compare", action="store_true", default=True,
                   help="also run the narrow frame at 1280 (default on)")
    p.add_argument("--no-compare", dest="compare", action="store_false")
    p.add_argument("--alt-imgsz", type=int, default=1280)
    p.add_argument("--save-every", type=float, default=1.0,
                   help="seconds between saved frames when nothing is detected")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out) if args.out else _ROOT / "diag" / ("detect_" + stamp)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    print("recording to %s\n" % out)

    from turret_host import cameras
    from turret_host.detector import Detector

    print("loading detector: %s at imgsz %d, conf %.2f"
          % (config.YOLO_WEIGHTS, config.YOLO_IMGSZ, config.YOLO_CONF))
    det = Detector().start()
    print("  -> %s, classes %s"
          % (det.weights_path.name, [det.model_names[i] for i in det.class_ids]))
    alt = None
    if args.compare and args.alt_imgsz != config.YOLO_IMGSZ:
        alt = Detector(imgsz=args.alt_imgsz).start()
        print("  -> comparison detector at imgsz %d" % args.alt_imgsz)

    ident = cameras.identify_cameras()
    for name, index, exposure, gain in (("narrow", ident.narrow_index, -6, 64),
                                        ("wide", ident.wide_index, -6, None)):
        cameras.lock_exposure(index, exposure, gain=gain)
    print("\nopening cameras...")
    narrow = cameras.narrow_thread(ident.narrow_index).start()
    wide = cameras.wide_thread(ident.wide_index).start()
    cams = [("narrow", narrow), ("wide", wide)]
    print("  narrow %.1f fps, wide %.1f fps"
          % (narrow.startup_fps, wide.startup_fps))

    print("\n" + "=" * 78)
    print("HOLD THE DRONE UP IN FRONT OF THE CAMERAS. %.0f s. Ctrl-C to stop."
          % args.seconds)
    print("=" * 78 + "\n")

    sink = (out / "detections.jsonl").open("w", encoding="utf-8")
    stats: Dict[str, Dict[str, list]] = {}
    last_seq: Dict[str, int] = {}
    last_save: Dict[str, float] = {}
    t0 = time.perf_counter()
    deadline = t0 + args.seconds

    def record(cam, tag, frame, dets, infer_ms):
        s = stats.setdefault(tag, {"frames": 0, "hits": 0, "conf": [], "size": []})
        s["frames"] += 1
        if dets:
            s["hits"] += 1
            best = max(dets, key=lambda d: d.conf)
            s["conf"].append(best.conf)
            s["size"].append(max(best.w, best.h))
        sink.write(json.dumps({
            "t": frame.t, "rel_t": frame.t - t0, "camera": cam, "tag": tag,
            "index": frame.index, "n": len(dets), "infer_ms": round(infer_ms, 2),
            "dets": [{"conf": round(d.conf, 4), "label": d.label,
                      "xyxy": [round(d.x1, 1), round(d.y1, 1),
                               round(d.x2, 1), round(d.y2, 1)],
                      "w": round(d.w, 1), "h": round(d.h, 1)} for d in dets],
        }) + "\n")

    try:
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            line = []
            saved_any = False
            for cam, thread in cams:
                frame, seq = thread.slot.get()
                if frame is None or last_seq.get(cam) == seq:
                    continue
                last_seq[cam] = seq

                r = det.detect(frame)
                record(cam, "%s@%d" % (cam, config.YOLO_IMGSZ), frame,
                       r.targets, r.infer_ms)
                best = max(r.targets, key=lambda d: d.conf) if r.targets else None
                line.append("%-6s@%-4d %d det %s"
                            % (cam, config.YOLO_IMGSZ, len(r.targets),
                               ("conf %.2f box %dx%d" % (best.conf, best.w, best.h))
                               if best else "--"))

                alt_targets = []
                if alt is not None and cam == "narrow":
                    ra = alt.detect(frame)
                    alt_targets = ra.targets
                    record(cam, "%s@%d" % (cam, args.alt_imgsz), frame,
                           alt_targets, ra.infer_ms)
                    ab = max(alt_targets, key=lambda d: d.conf) if alt_targets else None
                    line.append("%-6s@%-4d %d det %s"
                                % (cam, args.alt_imgsz, len(alt_targets),
                                   ("conf %.2f" % ab.conf) if ab else "--"))

                hit = bool(r.targets or alt_targets)
                if hit or (frame.t - last_save.get(cam, -1e9)) >= args.save_every:
                    last_save[cam] = frame.t
                    img = draw(frame.image, r.targets,
                               "%s @%d  t=%.2f" % (cam, config.YOLO_IMGSZ,
                                                   frame.t - t0), BOX_HIT)
                    if alt_targets:
                        img = draw(img, alt_targets,
                                   "%s @%d (orange) vs @%d (green)"
                                   % (cam, args.alt_imgsz, config.YOLO_IMGSZ),
                                   BOX_ALT)
                    cv2.imwrite(str(out / "frames" / ("%s_%06d_%s.jpg"
                                % (cam, frame.index, "HIT" if hit else "miss"))),
                                img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    saved_any = True

            if line:
                print("[%6.2f] %s%s" % (now - t0, "   |   ".join(line),
                                        "  *" if saved_any else ""))
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nstopped early")

    sink.close()
    for _n, c in cams:
        try:
            c.stop()
        except Exception:                                 # noqa: BLE001
            pass

    summary = {}
    print("\n" + "=" * 78)
    print("%-16s %-9s %-9s %-10s %-10s" %
          ("stream", "frames", "hits", "hit rate", "mean conf"))
    print("-" * 78)
    for tag, s in stats.items():
        rate = s["hits"] / s["frames"] if s["frames"] else 0.0
        mc = statistics.fmean(s["conf"]) if s["conf"] else 0.0
        ms = statistics.fmean(s["size"]) if s["size"] else 0.0
        summary[tag] = {"frames": s["frames"], "hits": s["hits"],
                        "hit_rate": round(rate, 3), "mean_conf": round(mc, 3),
                        "mean_box_px": round(ms, 1)}
        print("%-16s %-9d %-9d %-10.3f %-10.3f  box %.0f px"
              % (tag, s["frames"], s["hits"], rate, mc, ms))

    (out / "session.json").write_text(json.dumps({
        "weights": str(det.weights_path), "imgsz": config.YOLO_IMGSZ,
        "alt_imgsz": args.alt_imgsz if alt else None,
        "conf": config.YOLO_CONF, "summary": summary,
    }, indent=2), encoding="utf-8")
    print("\nwritten to %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
