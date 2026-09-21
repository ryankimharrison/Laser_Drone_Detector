"""Render the fused overlay from a RECORDED run, old model against new.

    python tools/preview_fused.py diag/flight/run_2026-09-20_151635
    python tools/preview_fused.py <run> --frames 6 --out /tmp/fused

Offline. Reads a recorded run's narrow and wide jpegs. No camera, no board,
no GUI -- so this can run while the rig is busy, and it answers the only
question that matters before touching the live panel: DOES THE OVERLAY
ACTUALLY LOOK BETTER.

WHY NOT JUST TRUST THE RMS
--------------------------
Because a residual number is computed on the calibration plane, over the
patch the projector happened to light, and the part of the overlay that looks
wrong is neither of those. tools/fit_wide_narrow_registration.py reports
3.491 -> 0.496 narrow px and a held-out extrapolation of 15.05 -> 0.60, which
is strong, but it is still arithmetic about a wall from 2026-09-19. This puts
the two models on the same real frame, side by side, and lets a person look.
The standing rule on this project is to open the image before scoring
anything, and that rule does not stop applying because the numbers are good.

WHAT IT DRAWS
-------------
For each sampled frame, one image with two panes:

    left    LEGACY   wide_narrow_homography.json, what the panel ships today
    right   FITTED   wide_narrow_registration.json, undistort + homography

Both are the 50/50 blend the live pane uses, so misregistration shows as a
double image in exactly the way it does on the panel -- that blend is not a
stylistic choice, it is the measurement. The narrow field's border is drawn
on each: a straight quadrilateral for the legacy model, which is what a
homography believes, and a sampled curve for the fitted one, which is what
the lens actually does. If the fit is right the right-hand pane is sharper,
and the difference is largest at the top and bottom of the overlaid patch.

THE RECORDED WIDE FRAMES ARE HALF SIZE, AND THAT IS A THIRD FRAME SIZE
----------------------------------------------------------------------
telemetry._wide_frame_writer saves wide at half resolution (INTER_AREA) with
`native=1920x1080` in the sidecar. So a recorded wide jpeg is 960x540 and is
a DOWNSCALE -- not the 1280x720 CENTRE CROP that --wide-fast delivers, and
not the 1920x1080 the registration was measured at. Three sizes, two of them
1.5 MP-ish, and the correct handling differs between them: a downscale needs
a SCALE, a crop needs an OFFSET.

registration.frame_offset() deliberately REFUSES 960x540 rather than guessing,
so this tool upscales the recorded frame back to native first, using the
`native=` field from the sidecar rather than assuming a factor of 2. That
keeps the one place that interprets frame sizes honest, and keeps this tool's
assumption where it belongs -- here, next to the recorder that created it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from turret_host import config                                  # noqa: E402
from turret_host.registration import WideNarrowRegistration     # noqa: E402

LEGACY_PATH = os.path.join(_ROOT, "turret_host", "calibration",
                           "wide_narrow_homography.json")

CYAN = (255, 214, 0)          # BGR
AMBER = (0, 170, 255)
WHITE = (255, 255, 255)


# ==========================================================================
#   the recording
# ==========================================================================
def wide_index(run: str) -> List[Tuple[float, str, Tuple[int, int]]]:
    """(capture time, jpeg path, native size) for every recorded wide frame."""
    out = []
    for jpg in sorted(glob.glob(os.path.join(run, "wide", "*.jpg"))):
        side = jpg[:-4] + ".txt"
        try:
            head = open(side, encoding="utf-8").readline()
            t = float(head.split("t=")[1].split()[0])
            native = head.split("native=")[1].split()[0]
            w, h = (int(v) for v in native.lower().split("x"))
        except (OSError, IndexError, ValueError):
            continue
        out.append((t, jpg, (w, h)))
    return out


def narrow_index(run: str) -> List[dict]:
    """Control rows that have a recorded narrow jpeg, with their frame time."""
    rows = []
    have = set()
    for jpg in glob.glob(os.path.join(run, "raw", "*.jpg")):
        try:
            have.add(int(os.path.basename(jpg)[:-4]))
        except ValueError:
            pass
    path = os.path.join(run, "control.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            i = r.get("frame_index")
            if i in have and r.get("frame_t"):
                rows.append({
                    "index": i, "t": float(r["frame_t"]),
                    "jpg": os.path.join(run, "raw", "%06d.jpg" % i),
                    "range_m": (r.get("est") or {}).get("range_m"),
                    "range_source": (r.get("est") or {}).get("range_source"),
                    "track": r.get("track"),
                })
    rows.sort(key=lambda r: r["t"])
    return rows


def pair(narrow_rows: Sequence[dict],
         wides: Sequence[Tuple[float, str, Tuple[int, int]]]) -> List[dict]:
    """Nearest-in-time wide frame for each narrow frame, with the skew."""
    if not wides:
        return []
    wt = np.array([w[0] for w in wides])
    out = []
    for r in narrow_rows:
        j = int(np.searchsorted(wt, r["t"]))
        best = None
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(wides):
                d = abs(wt[k] - r["t"])
                if best is None or d < best[0]:
                    best = (d, k)
        d, k = best
        out.append({**r, "skew_s": float(wt[k] - r["t"]),
                    "abs_skew_s": float(d), "wide_jpg": wides[k][1],
                    "wide_native": wides[k][2], "wide_t": float(wt[k])})
    return out


# ==========================================================================
#   the two models
# ==========================================================================
class LegacyModel:
    """The shipped plain homography on DISTORTED wide pixels."""

    name = "LEGACY  wide_narrow_homography.json"

    def __init__(self, path: str = LEGACY_PATH) -> None:
        with open(path, encoding="utf-8") as fh:
            b = json.load(fh)
        self.h = np.array(b["H_wide_to_narrow"], float)
        self.rms = float(b.get("rms_px", 0.0))
        self.narrow_size = tuple(config.NARROW_SIZE)

    def maps(self, display_size, native_size, _range_m):
        dw, dh = display_size
        fw, fh = native_size
        sx, sy = dw / float(fw), dh / float(fh)
        s_inv = np.array([[1.0 / sx, 0, 0], [0, 1.0 / sy, 0], [0, 0, 1.0]])
        m = self.h @ s_inv
        u = np.arange(dw, dtype=np.float64)
        v = np.arange(dh, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        p = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1) @ m.T
        w = np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])
        n = p[:, :2] / w
        mx = n[:, 0].reshape(dh, dw)
        my = n[:, 1].reshape(dh, dw)
        nw, nh = self.narrow_size
        inside = (mx >= 0) & (mx <= nw - 1) & (my >= 0) & (my <= nh - 1)
        return mx.astype(np.float32), my.astype(np.float32), inside

    def outline(self, native_size, _range_m):
        """Four corners. A homography believes this border is straight."""
        nw, nh = self.narrow_size
        c = np.array([[0, 0, 1], [nw - 1, 0, 1], [nw - 1, nh - 1, 1],
                      [0, nh - 1, 1]], float)
        q = c @ np.linalg.inv(self.h).T
        return q[:, :2] / q[:, 2:3]


class FittedModel:
    name = "FITTED  wide_narrow_registration.json"

    def __init__(self, reg: WideNarrowRegistration) -> None:
        self.reg = reg
        self.rms = reg.rms_px

    def maps(self, display_size, native_size, range_m):
        return self.reg.warp_maps(display_size, native_size, range_m)

    def outline(self, native_size, range_m):
        return self.reg.narrow_outline(native_size, range_m)


# ==========================================================================
#   rendering
# ==========================================================================
def render(model, narrow: np.ndarray, wide_native: np.ndarray,
           display_size: Tuple[int, int], native_size: Tuple[int, int],
           range_m: Optional[float], caption: str) -> Optional[np.ndarray]:
    dw, dh = display_size
    base = cv2.resize(wide_native, (dw, dh), interpolation=cv2.INTER_AREA)
    got = model.maps(display_size, native_size, range_m)
    if got is None:
        cv2.putText(base, "model refused this frame size", (12, dh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2, cv2.LINE_AA)
        return base
    mx, my, inside = got
    warped = cv2.remap(narrow, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    a = inside[:, :, None]
    # The same 50/50 the live pane uses. Misregistration has to show.
    out = np.where(a, (0.5 * base + 0.5 * warped).astype(np.uint8), base)

    poly = model.outline(native_size, range_m)
    if poly is not None:
        sx, sy = dw / float(native_size[0]), dh / float(native_size[1])
        pts = np.round(poly * (sx, sy)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, CYAN, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (dw - 1, 22), (0, 0, 0), -1)
    cv2.putText(out, caption, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                WHITE, 1, cv2.LINE_AA)
    return out


def side_by_side(left: np.ndarray, right: np.ndarray,
                 footer: str) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    gap = 8
    out = np.zeros((h + 26, left.shape[1] + gap + right.shape[1], 3), np.uint8)
    out[:left.shape[0], :left.shape[1]] = left
    out[:right.shape[0], left.shape[1] + gap:] = right
    cv2.putText(out, footer, (6, h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                WHITE, 1, cv2.LINE_AA)
    return out


# ==========================================================================
#   main
# ==========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="a diag/flight/run_* directory")
    ap.add_argument("--out", default=None,
                    help="directory for the comparison pngs "
                         "(default: <run>/fused_preview)")
    ap.add_argument("--frames", type=int, default=6,
                    help="how many frames to render, spread over the run")
    ap.add_argument("--max-skew-ms", type=float, default=25.0,
                    help="only render pairs closer together in time than "
                         "this -- a composite of two different moments tells "
                         "you nothing about registration")
    ap.add_argument("--width", type=int, default=1280,
                    help="pane width; the overlay's error scales with it")
    args = ap.parse_args(argv)

    run = args.run.rstrip("/\\")
    if not os.path.isdir(run):
        print("no such run directory: %s" % run)
        return 2
    out_dir = args.out or os.path.join(run, "fused_preview")

    reg = WideNarrowRegistration.load()
    if reg is None:
        print("no wide_narrow_registration.json -- run "
              "tools/fit_wide_narrow_registration.py --write first")
        return 1
    legacy, fitted = LegacyModel(), FittedModel(reg)

    wides = wide_index(run)
    narrows = narrow_index(run)
    pairs = pair(narrows, wides)
    if not pairs:
        print("no narrow/wide pairs in %s" % run)
        return 1

    # ---- the time skew, which is its own finding -----------------------
    skew_ms = np.array([p["abs_skew_s"] for p in pairs]) * 1000.0
    signed = np.array([p["skew_s"] for p in pairs]) * 1000.0
    print("=" * 70)
    print("NARROW <-> WIDE FRAME SKEW over %d narrow frames" % len(pairs))
    print("=" * 70)
    print("  |skew|  median %.1f  p90 %.1f  p99 %.1f  max %.1f ms"
          % (np.median(skew_ms), np.percentile(skew_ms, 90),
             np.percentile(skew_ms, 99), skew_ms.max()))
    print("  signed  mean %+.1f ms  (wide later than narrow if positive)"
          % signed.mean())
    close = int((skew_ms <= args.max_skew_ms).sum())
    print("  %d of %d pairs (%.1f%%) are within %.0f ms"
          % (close, len(pairs), 100.0 * close / len(pairs), args.max_skew_ms))
    # What that skew is worth in pixels is the thing worth knowing. 25 narrow
    # px per degree of platform motion (AGENT_HANDOFF, C270 as mounted).
    print("  at 30 deg/s of platform motion, the median skew alone displaces")
    print("  the two views by %.0f narrow px (25 px/deg as mounted)"
          % (np.median(skew_ms) / 1000.0 * 30.0 * 25.0))
    print()

    usable = [p for p in pairs if p["abs_skew_s"] * 1000.0 <= args.max_skew_ms]
    if not usable:
        print("no pair is within %.0f ms; raise --max-skew-ms to look anyway"
              % args.max_skew_ms)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    pick = [usable[i] for i in
            np.linspace(0, len(usable) - 1, min(args.frames, len(usable))).astype(int)]

    dw = args.width
    dh = int(round(dw * config.WIDE_SIZE[1] / config.WIDE_SIZE[0]))
    print("rendering %d frames at %dx%d into %s" % (len(pick), dw, dh, out_dir))
    print("%-9s %8s %9s %8s  %s" % ("frame", "skew ms", "range m", "track",
                                    "file"))

    written = 0
    for p in pick:
        narrow = cv2.imread(p["jpg"])
        wide_small = cv2.imread(p["wide_jpg"])
        if narrow is None or wide_small is None:
            continue
        native = p["wide_native"]
        # Back to capture coordinates. See the module docstring: the recorder
        # halves the wide frame, and 960x540 is a size registration.py will
        # (correctly) refuse, because it cannot know a downscale from a crop.
        wide = cv2.resize(wide_small, native, interpolation=cv2.INTER_LINEAR)

        r_m = p["range_m"]
        cap = "  |  skew %+.0f ms  range %s" % (
            p["skew_s"] * 1000.0,
            "%.2f m (%s)" % (r_m, p["range_source"]) if r_m else "unknown")
        a = render(legacy, narrow, wide, (dw, dh), native, r_m,
                   legacy.name + "   rms %.2f px" % legacy.rms + cap)
        b = render(fitted, narrow, wide, (dw, dh), native, r_m,
                   fitted.name + "   rms %.2f px" % fitted.rms + cap)
        if a is None or b is None:
            continue
        footer = ("frame %06d   both panes are a 50/50 blend: a double image "
                  "IS the misregistration.   parallax sign %s"
                  % (p["index"],
                     "OFF (never measured)" if reg.parallax_sign == 0
                     else "%+d" % reg.parallax_sign))
        path = os.path.join(out_dir, "fused_%06d.png" % p["index"])
        cv2.imwrite(path, side_by_side(a, b, footer))
        written += 1
        print("%-9d %8.1f %9s %8s  %s"
              % (p["index"], p["skew_s"] * 1000.0,
                 "%.2f" % r_m if r_m else "--", p["track"] or "--",
                 os.path.basename(path)))

    print("\nwrote %d comparisons -> %s" % (written, out_dir))
    if reg.extrapolating():
        print("NOTE: the overlay reaches wide radius %.0f px and the fit is "
              "held-out verified to %.0f. The outer band of the FITTED pane "
              "is the model extrapolating -- which is still far better than "
              "the legacy pane there, but it is not measured."
              % (reg.narrow_footprint_radius_px, reg.verified_radius_px))
    print("LOOK AT THEM. The numbers say the right pane is 7x better; the "
          "only thing that settles it is whether the double image closes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
