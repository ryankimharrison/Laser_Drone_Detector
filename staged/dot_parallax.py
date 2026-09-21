"""Where does the LASER actually land? Dot detection, parallax fit, true metric.

    python tools/dot_parallax.py --validate      # 20 annotated crops, look first
    python tools/dot_parallax.py                 # full pass + fit + re-score

STAGED; belongs at tools/. Offline on `raw/` frames. No hardware.

WHY
---
Every aim number so far is measured from `cmd.goal_u/goal_v` -- the goal PIXEL,
which is a model of where the beam lands, fitted from the dot at ONE range on
2026-09-18 with the ~30 mm laser-to-camera parallax deliberately unmodelled
(`goal_pixel.json` has c_u = c_v = 0). Geometry says that offset goes as
~42 px*m: 28 px at 1.5 m, 14 at 3 m, 8 at 5 m -- comparable to the whole error
budget, in a direction nobody has measured. So this measures the dot itself.

THE DOT IS NOT THE SMALL DOT THE CONFIG DESCRIBES
-------------------------------------------------
`config.DOT_AREA_PX` is (3, 40), tuned for a dot on a far wall. On the airframe
at 2 m the splash is a bloom: measured median area 1544 px, range 197-5107,
peak chroma 124. Reusing those bounds would find nothing. These are separate
constants on purpose -- the live dot-lock path is tuned for its own job and is
not touched here.

FRAGMENTATION IS NORMAL, SO "MORE THAN ONE BLOB" IS NOT A REJECTION
--------------------------------------------------------------------
The bloom breaks into several components across the rotor arms: of 25 sampled
firing frames only 8 had exactly one. Rejecting the rest would throw away most
of the data for no gain. Instead, components close to the brightest one are
MERGED (they are one splash seen through a gappy airframe), and a frame is
rejected only when a genuinely SEPARATE component carries a comparable weight
-- which is the reflection case the brief is actually worried about.

The laser also pulses at LASER_PULSE_HZ = 15 against ~30 fps, so a FIRING row
can legitimately have no dot because the beam was between pulses. Those are
reported as not-detected, never as a failure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLIGHT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "diag", "flight")

#: Green dominance (G - max(R,B)) above this is dot-coloured.
TAU = 40.0
#: A clipped core reads white, so brightness rescues it -- but only next to
#: green, never on its own.
BRIGHT = 235
#: Components smaller than this are speckle.
MIN_AREA = 40
#: Components whose centroid is within this of the brightest one are the same
#: splash seen through the gaps in the airframe.
MERGE_PX = 45.0
#: A separate component carrying at least this fraction of the main weight makes
#: the frame ambiguous -- two candidate dots, and we cannot say which is the beam.
AMBIGUOUS_FRAC = 0.35
#: Search this far outside the box: the dot can sit on the very edge.
BOX_PAD = 0.10


def load(run):
    path = run if os.path.isdir(run) else os.path.join(FLIGHT, run)
    with open(os.path.join(path, "control.jsonl"), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh], path


def find_dot(img, box):
    """(u, v, area, peak_gray, n_merged, mode) in full-frame px, or None.

    None means no dot THIS frame -- between pulses, or invisible. The caller
    must not read it as a miss.
    """
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = max(0, int(box[0] - BOX_PAD * bw))
    y1 = max(0, int(box[1] - BOX_PAD * bh))
    x2 = min(w, int(box[2] + BOX_PAD * bw))
    y2 = min(h, int(box[3] + BOX_PAD * bh))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    roi = img[y1:y2, x1:x2]
    b = roi.astype(np.int16)
    chroma = (b[:, :, 1] - np.maximum(b[:, :, 0], b[:, :, 2])).astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = (chroma > TAU) | ((gray > BRIGHT) & (chroma > TAU * 0.25))
    m8 = mask.astype(np.uint8) * 255
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m8, connectivity=8)
    comps = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < MIN_AREA:
            continue
        wt = float(np.where(lab == i, np.maximum(chroma, 0.0), 0.0).sum())
        comps.append((wt, a, i, (float(cen[i][0]), float(cen[i][1]))))
    if not comps:
        return None
    comps.sort(reverse=True)
    main = comps[0]
    near, far = [main], []
    for c in comps[1:]:
        d = math.hypot(c[3][0] - main[3][0], c[3][1] - main[3][1])
        (near if d <= MERGE_PX else far).append(c)
    for c in far:
        if c[0] >= AMBIGUOUS_FRAC * main[0]:
            return None                     # two real candidates; refuse to guess
    keep = np.zeros(lab.shape, np.uint8)
    for c in near:
        keep |= (lab == c[2]).astype(np.uint8)

    # WEIGHT BY BRIGHTNESS, NOT BY CHROMA, AND PREFER THE SATURATED CORE.
    #
    # Found by looking at the first 20 validation crops: a chroma-weighted
    # centroid lands in the middle of the green FLARE, which is not where the
    # beam hits. The impact point clips all three channels and reads WHITE, so
    # its chroma is near ZERO -- chroma weighting actively pushes the answer
    # away from the very pixel we are trying to find, and it did, consistently
    # up and to one side.
    #
    # So: if the blob has a saturated core, the dot is the centroid of that
    # core. Only when nothing is saturated (a dim or distant hit) does it fall
    # back to a brightness-weighted centroid over the whole blob.
    grayf = gray.astype(np.float32)
    yy, xx = np.indices(chroma.shape, dtype=np.float32)
    core = (keep > 0) & (grayf >= 245.0)
    if int(core.sum()) >= 4:
        wmap = np.where(core, grayf, 0.0)
        mode = "core"
    else:
        inblob = grayf[keep > 0]
        thr = float(np.percentile(inblob, 99.0)) if inblob.size else 255.0
        sel = (keep > 0) & (grayf >= thr)
        if int(sel.sum()) < 4:
            sel = keep > 0
        wmap = np.where(sel, grayf, 0.0)
        mode = "bright"
    tot = float(wmap.sum())
    if tot <= 0:
        return None
    return (x1 + float((wmap * xx).sum()) / tot,
            y1 + float((wmap * yy).sum()) / tot,
            int(sum(c[1] for c in near)), float(grayf.max()), len(near), mode)


def collect(run):
    rows, path = load(run)
    out = []
    for r in rows:
        if r.get("laser") != "FIRING":
            continue
        e = r.get("est") or {}
        c = r.get("cmd") or {}
        box = e.get("box")
        if not box or c.get("goal_u") is None:
            continue
        img = cv2.imread(os.path.join(path, "raw", "%06d.jpg" % r["frame_index"]))
        if img is None:
            continue
        d = find_dot(img, box)
        out.append({"frame": r["frame_index"], "row": r, "box": box,
                    "goal": (c["goal_u"], c["goal_v"]), "dot": d,
                    "range_m": e.get("range_m"), "range_source": e.get("range_source"),
                    "centre": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                    "span": min(box[2] - box[0], box[3] - box[1])})
    return out, path


def validate(run, n=20, outdir=None):
    got, path = collect(run)
    hits = [g for g in got if g["dot"]]
    print("%s: %d firing rows examined, dot found on %d (%.0f%%)"
          % (os.path.basename(path), len(got), len(hits),
             100.0 * len(hits) / max(len(got), 1)))
    step = max(1, len(hits) // n)
    sel = hits[::step][:n]
    tiles = []
    for g in sel:
        img = cv2.imread(os.path.join(path, "raw", "%06d.jpg" % g["frame"]))
        b = [int(v) for v in g["box"]]
        pad = 20
        x1, y1 = max(0, b[0] - pad), max(0, b[1] - pad)
        x2, y2 = min(img.shape[1], b[2] + pad), min(img.shape[0], b[3] + pad)
        crop = img[y1:y2, x1:x2].copy()
        sc = 300.0 / max(crop.shape[0], crop.shape[1])
        crop = cv2.resize(crop, None, fx=sc, fy=sc)
        du, dv = g["dot"][0] - x1, g["dot"][1] - y1
        cv2.circle(crop, (int(du * sc), int(dv * sc)), 14, (255, 0, 255), 2)
        cv2.drawMarker(crop, (int((g["goal"][0] - x1) * sc), int((g["goal"][1] - y1) * sc)),
                       (0, 200, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.drawMarker(crop, (int((g["centre"][0] - x1) * sc), int((g["centre"][1] - y1) * sc)),
                       (255, 255, 255), cv2.MARKER_SQUARE, 14, 2)
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, 15), (0, 0, 0), -1)
        cv2.putText(crop, "f%d a%d" % (g["frame"], g["dot"][2]), (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 255), 1)
        tiles.append(crop)
    if not tiles:
        return
    hgt = max(t.shape[0] for t in tiles)
    wid = max(t.shape[1] for t in tiles)
    cols = 5
    rowsn = (len(tiles) + cols - 1) // cols
    sheet = np.full((hgt * rowsn, wid * cols, 3), 25, np.uint8)
    for i, t in enumerate(tiles):
        r0, c0 = divmod(i, cols)
        sheet[r0 * hgt:r0 * hgt + t.shape[0], c0 * wid:c0 * wid + t.shape[1]] = t
    p = os.path.join(outdir or ".", "dot_validation_%s.png" % os.path.basename(path)[-6:])
    cv2.imwrite(p, sheet)
    print("  MAGENTA circle = detected dot, ORANGE cross = goal pixel, "
          "WHITE square = box centre")
    print("  -> %s" % p)


def fit_parallax(got):
    """Least squares  d = d_inf + c/R  per axis. Returns dict."""
    pts = [(g["dot"][0] - g["goal"][0], g["dot"][1] - g["goal"][1], g["range_m"])
           for g in got if g["dot"] and g["range_m"]]
    if len(pts) < 30:
        return None
    du = np.array([p[0] for p in pts])
    dv = np.array([p[1] for p in pts])
    inv = np.array([1.0 / p[2] for p in pts])
    A = np.vstack([np.ones_like(inv), inv]).T
    out = {"n": len(pts)}
    for name, d in (("u", du), ("v", dv)):
        sol, res, _, _ = np.linalg.lstsq(A, d, rcond=None)
        pred = A @ sol
        resid = d - pred
        ss_tot = float(((d - d.mean()) ** 2).sum())
        ss_res = float((resid ** 2).sum())
        out[name] = {"inf": float(sol[0]), "c": float(sol[1]),
                     "resid_med": float(np.median(np.abs(resid))),
                     "resid_p90": float(np.percentile(np.abs(resid), 90)),
                     "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                     "raw_med": float(np.median(d))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append",
                    default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args(argv)
    runs = a.run or ["run_2026-09-20_151635", "run_2026-09-20_145109"]
    if a.validate:
        validate(runs[0], outdir=a.outdir)
        return 0
    for run in runs:
        got, path = collect(run)
        hits = [g for g in got if g["dot"]]
        print("\n================ %s ================" % os.path.basename(path))
        print("  firing rows with a box: %d;  dot detected: %d (%.0f%%)"
              % (len(got), len(hits), 100.0 * len(hits) / max(len(got), 1)))
        if not hits:
            continue
        ar = sorted(g["dot"][2] for g in hits)
        print("  blob area px: median %d  p10 %d  p90 %d"
              % (ar[len(ar) // 2], ar[len(ar) // 10], ar[9 * len(ar) // 10]))
        src = {}
        for g in hits:
            src[g["range_source"]] = src.get(g["range_source"], 0) + 1
        print("  range_source on the fitted rows: %s" % src)

        d = [(g["dot"][0] - g["goal"][0], g["dot"][1] - g["goal"][1]) for g in hits]
        print("  DOT minus GOAL PIXEL: du median %+.1f px, dv median %+.1f px"
              % (statistics.median([x[0] for x in d]),
                 statistics.median([x[1] for x in d])))
        f = fit_parallax(hits)
        if f:
            print("  parallax fit  d = d_inf + c/R   (n=%d)" % f["n"])
            for ax in ("u", "v"):
                q = f[ax]
                print("    %s: d_inf %+7.2f px   c %+8.2f px*m   R2 %5.2f   "
                      "|resid| med %5.2f p90 %5.2f"
                      % (ax, q["inf"], q["c"], q["r2"], q["resid_med"], q["resid_p90"]))
            print("    geometry predicts |c| ~ 42 px*m on ONE axis; "
                  "measured |c_u| %.1f, |c_v| %.1f"
                  % (abs(f["u"]["c"]), abs(f["v"]["c"])))

        # THE METRIC, both ways.
        gp = sorted(math.hypot(g["goal"][0] - g["centre"][0],
                               g["goal"][1] - g["centre"][1]) for g in hits)
        dt = sorted(math.hypot(g["dot"][0] - g["centre"][0],
                               g["dot"][1] - g["centre"][1]) for g in hits)
        gs = sorted(math.hypot(g["goal"][0] - g["centre"][0],
                               g["goal"][1] - g["centre"][1]) / g["span"] for g in hits)
        ds = sorted(math.hypot(g["dot"][0] - g["centre"][0],
                               g["dot"][1] - g["centre"][1]) / g["span"] for g in hits)
        print("  BEAM-TO-CENTRE on the same %d rows:" % len(hits))
        print("    by GOAL PIXEL (what we have been reporting): median %5.1f px  "
              "p90 %5.1f   err/span %.3f" % (gp[len(gp) // 2], gp[9 * len(gp) // 10],
                                             gs[len(gs) // 2]))
        print("    by MEASURED DOT (the truth):                 median %5.1f px  "
              "p90 %5.1f   err/span %.3f" % (dt[len(dt) // 2], dt[9 * len(dt) // 10],
                                             ds[len(ds) // 2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
