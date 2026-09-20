"""Score a recorded run on LEAD ERROR -- the one number the goal reduces to.

    python tools/analyze_run.py diag/flight/run_2026-09-19_013900
    python tools/analyze_run.py --latency 0.060 diag/flight/run_*    # compare

THE METRIC
----------
"Keep the laser centred on the drone regardless of distance or speed" is a
statement about one quantity:

    lead error = || where the filter said the drone would be at time T
                   - where it actually was at T ||

The controller commits to an aim point LATENCY_S before the beam gets there,
so this is the distance between that commitment and the truth. Under
MAX_ERROR_TO_FIRE_PX the beam is on the drone; over it, nothing else matters.

It is computable from logs ALREADY ON DISK. `app.py` logs
`tracker.estimate_for_control(now)`, which is `predict_to(now + LATENCY_S)` --
a genuine forward prediction, not the filtered-at-t estimate. So each row
carries a prediction, its target time is `t + LATENCY_S`, and later rows carry
the detector boxes that say what actually happened.

PASS THE LATENCY THE RUN WAS RECORDED WITH, NOT TODAY'S
-------------------------------------------------------
`session.json` does not record the config (a gap worth closing), and
config.LATENCY_S changed from the 0.060 ESTIMATE to a MEASURED 0.066 on
2026-09-19. Scoring an older run against today's value asks what the filter
should have predicted rather than what it did. Default here is 0.060, which is
what every run currently on disk was recorded with; override with --latency.

WHAT "ACTUALLY WAS" MEANS, AND WHY IT IS NOT GROUND TRUTH
---------------------------------------------------------
The only witness on disk is the DETECTOR's own later box. So this measures
prediction against detection, and inherits detection's error. Two consequences
stated rather than buried:

  * a confident, well-centred box is a good witness; a box that has drifted
    onto a rotor is not, and nothing here can tell them apart. Hand annotation
    is what fixes this, which is why the annotation loop exists.
  * it can only be computed where a box EXISTS near T. In COAST and SEARCH
    there is none, so those frames drop out -- which biases the result toward
    the easy cases. Coverage is therefore reported next to every figure, and a
    lead error quoted without its coverage is meaningless.

THE CONTROL THAT MAKES IT MEAN SOMETHING
-----------------------------------------
A prediction is only worth having if it beats not predicting. So the same
error is computed for a NAIVE predictor -- "the drone is where it was last
seen" -- and reported alongside. If the filter does not beat that, the motion
model is adding nothing and the fix is elsewhere.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turret_host.tracker import world_up_in_image, _box_extent_along  # noqa: E402
from turret_host import config  # noqa: E402

#: The aim bias every run on disk was recorded with. `_aim_bias()` pushed the
#: aim point toward world-up by this fraction of the box extent, so est.u/v is
#: NOT the box centre and never was -- comparing the two measures a DESIGN
#: OFFSET and calls it tracking error.
#:
#: That mistake produced the headline "the filter is 20-31 px worse than not
#: predicting at all" on 2026-09-19. Measured: 44.2 px against the box centre,
#: 2.5 px against the biased centre. The bias explained 94% of it.
#:
#: config.AIM_BIAS_UP_FRAC is now 0.0, so runs recorded from 2026-09-19
#: onwards need no correction -- which is exactly why this has to be read
#: PER RUN from session.json rather than taken from the live config, or the
#: new baseline and the old one are not the same measurement.
LEGACY_AIM_BIAS = 0.15

#: Scored against this. config.MAX_ERROR_TO_FIRE_PX.
FIRE_PX = 25.0

#: Widest gap between the prediction's target time T and a bracketing
#: observation that still counts. Frames are ~33 ms apart, so anything past
#: this is extrapolating across a gap rather than interpolating within one.
MATCH_TOL_S = 0.050

#: Commands below this are not evidence of anything. The old sign-flip metric
#: counted a +/-0.5 step/s reversal the same as a +/-3000 one, which is why it
#: got WORSE (20% -> 26%) on the gain change that improved every error measure:
#: at lower gain more commands sit near zero. steps/s, against MAX_MOTOR_RATE
#: of 4000.
FLIP_MIN_RATE = 200.0


def _centre(box) -> Tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def load(run_dir: str) -> List[dict]:
    p = os.path.join(run_dir, "control.jsonl")
    if not os.path.exists(p):
        raise SystemExit("no control.jsonl in %s" % run_dir)
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass            # a truncated last line is normal after a kill
    return out


def run_config(run_dir: str) -> dict:
    """The config snapshot a run was recorded with, or {} for older runs.

    Added to session.json on 2026-09-19 precisely because LATENCY_S and
    AIM_BIAS_UP_FRAC both changed that day, and a log that does not say which
    values produced it cannot be re-scored later.
    """
    try:
        with open(os.path.join(run_dir, "session.json")) as fh:
            return json.load(fh).get("config") or {}
    except (OSError, ValueError):
        return {}


def observations(rows: List[dict]) -> List[Tuple[float, float, float, float]]:
    """Deduplicated (box_t, u, v, conf) -- what the detector actually saw.

    The control loop runs faster than the detector, so consecutive rows carry
    the SAME box. Keying on box_t collapses those; without it every repeat
    would be counted as an independent confirmation and the coverage figures
    would be inflated several-fold.
    """
    seen: Dict[float, Tuple[float, float, float, float]] = {}
    for r in rows:
        est = r.get("est") or {}
        b, bt = est.get("box"), est.get("box_t")
        if b and bt is not None and bt not in seen:
            u, v = _centre(b)
            seen[bt] = (bt, u, v, est.get("conf") or 0.0,
                        abs(b[2] - b[0]), abs(b[3] - b[1]))
    return [seen[k] for k in sorted(seen)]


def truth_at(obs, T: float) -> Optional[Tuple[float, float]]:
    """Where the detector says the target was at time T, by interpolation.

    Interpolated rather than nearest-neighbour: at 30 fps, nearest-neighbour
    quantises the truth to +/-16 ms, which at a plausible 500 px/s is +/-8 px
    of pure measurement artefact on a quantity being compared against a 25 px
    threshold. Only interpolates WITHIN a bracketing pair -- never extrapolates
    past the ends, where there is no evidence at all.
    """
    ts = obs[:, 0]
    i = int(np.searchsorted(ts, T))
    if i == 0 or i >= len(ts):
        return None
    t0, t1 = ts[i - 1], ts[i]
    if (T - t0) > MATCH_TOL_S or (t1 - T) > MATCH_TOL_S:
        return None
    w = 0.0 if t1 == t0 else (T - t0) / (t1 - t0)
    return (float(obs[i - 1, 1] + w * (obs[i, 1] - obs[i - 1, 1])),
            float(obs[i - 1, 2] + w * (obs[i, 2] - obs[i - 1, 2])))


def last_seen_before(obs, t: float) -> Optional[Tuple[float, float]]:
    """The naive predictor: wherever it was last actually observed."""
    i = int(np.searchsorted(obs[:, 0], t)) - 1
    return (float(obs[i, 1]), float(obs[i, 2])) if i >= 0 else None


def aim_point_at(obs, T: float, centre, aim_bias: float):
    """Where the tracker was TRYING to put the aim point at time T.

    Not the box centre: `_aim_bias()` pushes toward world-up by a fraction of
    the box's extent along that direction, so the aim point is a box-SIZE
    dependent offset from the centre. Reconstructed here from the logged box
    rather than assumed constant, because the offset scales with the target.
    """
    if aim_bias <= 0.0 or centre is None:
        return centre
    i = int(np.searchsorted(obs[:, 0], T))
    i = min(max(i, 0), len(obs) - 1)
    up = world_up_in_image(config.NARROW_ROTATION_DEG)
    extent = _box_extent_along((float(obs[i, 4]), float(obs[i, 5])), up)
    return (centre[0] + float(up[0]) * aim_bias * extent,
            centre[1] + float(up[1]) * aim_bias * extent)


def score(rows: List[dict], latency: float, aim_bias: float = 0.0) -> dict:
    obs_list = observations(rows)
    if len(obs_list) < 4:
        return {"n_obs": len(obs_list), "scored": 0}
    obs = np.array(obs_list, dtype=float)

    lead, naive, states, speeds, confs, srcs = [], [], [], [], [], []
    n_est = 0
    for r in rows:
        est = r.get("est") or {}
        if est.get("u") is None or r.get("t") is None:
            continue
        n_est += 1
        T = r["t"] + latency
        truth = aim_point_at(obs, T, truth_at(obs, T), aim_bias)
        if truth is None:
            continue
        lead.append(math.hypot(est["u"] - truth[0], est["v"] - truth[1]))
        states.append(r.get("track"))
        confs.append(est.get("conf") or 0.0)
        # Which camera the box came through. A wide-sourced box has been put
        # through wide_to_narrow before the tracker ever saw it, so it carries
        # that mapping's error on top of the detector's. Split on it, because
        # otherwise a bad mapping is indistinguishable from a bad prediction --
        # and the mapping was measured at 19.7 px RMS against a file claiming
        # 5.71. 1.0 = narrow, 0.0 = wide (control.py: box_source).
        srcs.append(((r.get("interlock") or {}).get("margins") or {})
                    .get("box_source"))

        seen = last_seen_before(obs, r["t"])
        naive.append(math.hypot(seen[0] - truth[0], seen[1] - truth[1])
                     if seen else float("nan"))

        # Target speed from the interpolated truth, one frame either side.
        a, b = truth_at(obs, T - 0.033), truth_at(obs, T + 0.033)
        speeds.append(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.066
                      if (a and b) else float("nan"))

    return {"n_obs": len(obs_list), "n_est": n_est, "scored": len(lead),
            "lead": np.array(lead), "naive": np.array(naive),
            "states": states, "speed": np.array(speeds),
            "conf": np.array(confs), "src": np.array(
                [np.nan if x is None else float(x) for x in srcs])}


def gate_audit(rows: List[dict]) -> dict:
    """Detected but not associated -- the 2026-09-19 open question.

    `n_targets` is what the detector found this pass; `est.has_box` is whether
    one of them survived association. The difference is the gate's reject rate,
    and it was never logged explicitly -- but it is recoverable from these two.
    """
    by_state: Dict[str, List[int]] = {}
    for r in rows:
        st = r.get("track") or "?"
        n = r.get("n_targets") or 0
        kept = bool((r.get("est") or {}).get("has_box"))
        d = by_state.setdefault(st, [0, 0, 0])
        d[0] += 1
        d[1] += 1 if n > 0 else 0
        d[2] += 1 if kept else 0
    return by_state


def sign_flips(rows: List[dict]) -> dict:
    """Magnitude-weighted command reversals. THE OLD METRIC WAS BROKEN.

    Counting every reversal made the K=2.0 -> 1.2 change look worse (20% ->
    26%) while every error measure improved, because at lower gain more
    commands sit near zero and a trivial jitter reversal counted as much as a
    full-scale one. Only reversals where BOTH commands are real movement are
    counted here, and the unweighted number is kept alongside so the two can
    be compared on the same run.
    """
    out = {}
    for key in ("rate_a", "rate_b"):
        vals = [(r.get("cmd") or {}).get(key) for r in rows]
        vals = [v for v in vals if v is not None]
        raw = sig = pairs = 0
        for p, q in zip(vals, vals[1:]):
            if p * q < 0:
                raw += 1
            if abs(p) >= FLIP_MIN_RATE and abs(q) >= FLIP_MIN_RATE:
                pairs += 1
                if p * q < 0:
                    sig += 1
        out[key] = {"n": len(vals), "raw": raw,
                    "raw_pct": 100.0 * raw / max(1, len(vals) - 1),
                    "sig": sig, "sig_pairs": pairs,
                    "sig_pct": 100.0 * sig / max(1, pairs)}
    return out


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def report(run_dir: str, latency: Optional[float] = None,
           aim_bias: Optional[float] = None) -> Optional[dict]:
    rows = load(run_dir)
    # PER-RUN, not from the live config. Both constants changed on
    # 2026-09-19, so taking today's values would score old runs against
    # settings they were never recorded under.
    cfg = run_config(run_dir)
    if latency is None:
        latency = float(cfg.get("LATENCY_S", 0.060))
    if aim_bias is None:
        aim_bias = float(cfg["AIM_BIAS_UP_FRAC"]) if "AIM_BIAS_UP_FRAC" in cfg             else LEGACY_AIM_BIAS
    s = score(rows, latency, aim_bias)
    print("=" * 74)
    print("%s   (%d control rows, LATENCY_S = %.3f, AIM_BIAS = %.2f%s)"
          % (os.path.basename(run_dir), len(rows), latency, aim_bias,
             "" if cfg else " assumed -- run predates the config snapshot"))
    if s.get("scored", 0) < 4:
        print("  not scorable: only %d distinct detector boxes in the whole run"
              % s.get("n_obs", 0))
        return None

    lead, naive = s["lead"], s["naive"]
    good = naive[~np.isnan(naive)]
    print("\n  LEAD ERROR  (prediction for t+L vs the detector's later box)")
    print("    scored          %d of %d rows with an estimate  (%.0f%% coverage)"
          % (s["scored"], s["n_est"], 100.0 * s["scored"] / max(1, s["n_est"])))
    print("    median          %7.1f px" % _pct(lead, 50))
    print("    p95             %7.1f px" % _pct(lead, 95))
    print("    under %.0f px     %6.1f%%   <- beam on the drone"
          % (FIRE_PX, 100.0 * float((lead < FIRE_PX).mean())))
    print("    under 60 px     %6.1f%%" % (100.0 * float((lead < 60).mean())))
    if len(good):
        print("\n  vs NAIVE 'it is where it was last seen'")
        print("    naive median    %7.1f px   (filter %.1f)"
              % (float(np.median(good)), _pct(lead, 50)))
        delta = float(np.median(good)) - _pct(lead, 50)
        if delta > 1.0:
            print("    -> the filter BEATS doing nothing by %.1f px" % delta)
        else:
            print("    -> THE FILTER IS NOT BEATING DOING NOTHING (%+.1f px)."
                  % delta)
            print("       The motion model is not earning its place; look at")
            print("       detection and association before tuning it.")

    sp = s["speed"][~np.isnan(s["speed"])]
    if len(sp) > 20:
        print("\n  LEAD ERROR vs TARGET SPEED   ('regardless of speed' = flat)")
        edges = np.percentile(sp, [0, 33, 66, 100])
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s["speed"] >= lo) & (s["speed"] < hi + 1e-9)
            if m.sum() > 3:
                print("    %6.0f-%-6.0f px/s   n=%-4d  median %6.1f px"
                      % (lo, hi, m.sum(), float(np.median(lead[m]))))

    src = s["src"]
    if np.isfinite(src).any():
        print("\n  LEAD ERROR BY CAMERA   (a wide box carries wide->narrow too)")
        for name, m in (("narrow", src == 1.0), ("wide  ", src == 0.0)):
            if m.sum() > 3:
                print("    %s  n=%-4d  median %6.1f px   under %.0f px %5.1f%%"
                      % (name, int(m.sum()), float(np.median(lead[m])), FIRE_PX,
                         100.0 * float((lead[m] < FIRE_PX).mean())))

    print("\n  ASSOCIATION GATE")
    print("    %-9s %6s %10s %10s" % ("state", "frames", "detected", "kept"))
    for st, (n, det, kept) in sorted(gate_audit(rows).items()):
        print("    %-9s %6d %9.1f%% %9.1f%%"
              % (st, n, 100.0 * det / max(1, n), 100.0 * kept / max(1, n)))

    print("\n  COMMAND REVERSALS   (>= %.0f steps/s on BOTH sides)" % FLIP_MIN_RATE)
    for k, f in sign_flips(rows).items():
        print("    %-7s raw %5.1f%%   magnitude-weighted %5.1f%%  (%d of %d pairs)"
              % (k, f["raw_pct"], f["sig_pct"], f["sig"], f["sig_pairs"]))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--latency", type=float, default=None,
                    help="override LATENCY_S; by default it is read per-run "
                         "from session.json, falling back to 0.060")
    ap.add_argument("--aim-bias", type=float, default=None,
                    help="override AIM_BIAS_UP_FRAC; by default read per-run "
                         "from session.json, falling back to %.2f. est.u/v is "
                         "the AIM POINT, not the box centre -- scoring it "
                         "against the centre measures the bias as error"
                         % LEGACY_AIM_BIAS)
    a = ap.parse_args(argv)

    dirs = []
    for pat in a.runs:
        dirs.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?")
                    else [pat])
    for d in dirs:
        if os.path.isdir(d):
            try:
                report(d, a.latency, a.aim_bias)
            except SystemExit as e:
                print("%s: %s" % (d, e))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
