"""Re-decide the laser interlock over a recorded run. STAGED; belongs at tools/.

    python tools/replay_gates.py                      # both runs, all variants
    python tools/replay_gates.py --run <dir>          # one run
    python tools/replay_gates.py --selftest           # validation only

WHY THIS IS A FAITHFUL REPLAY AND NOT A GUESS
---------------------------------------------
The interlock does not move the turret. `LaserInterlock.evaluate()` reads the
frame, the box, the error and the face margins and returns FIRING or an inhibit;
nothing downstream of it steers. So changing an interlock rule changes WHICH
ROWS FIRE and nothing else, and re-deciding the recorded rows gives the true
answer rather than an estimate.

That is only true while the change stays inside the interlock. Item 1 (holding
the last box across a detector miss) qualifies **because of a specific fact
about the tracker**: `TrackEstimate.box` is `_last_det`, and `_last_det` feeds
only `estimate.box` -- the range path uses `_stable_wh` and the aim bias uses
`_stable_wh` too (tracker.py `_range`, `_aim_bias`). So a held box cannot reach
the control law. Checked, not assumed; if that ever stops being true this
harness stops being faithful and the self-test below will not catch it.

Item 4 (the association gate) does NOT qualify: accepting a detection the gate
rejected changes the filter, the command and therefore every later frame. This
harness will not pretend otherwise -- it reports what the gate would have
ACCEPTED, and the frames have to be looked at.

THE DUTY CAP IS STATEFUL AND IS SIMULATED, NOT COPIED
-----------------------------------------------------
`duty` depends on the history of firing decisions: LASER_MAX_ON_MS of
continuous on-time forces the beam off and starts a LASER_COOLDOWN_S rest. Make
more rows fire and the cap is reached sooner, so the naive "these extra rows
would have fired" count is wrong. The state machine here is transcribed from
control.py (the `if on:` block after `on = all(...)`) and is re-run for every
variant. This is why the box-hold result is smaller than the row count suggests.

WHAT IT CANNOT SEE
------------------
Conditions this harness does not recompute are taken from the logged `failed`
list, which is correct only for rules the variant does not touch. It recomputes
`drone_lock`, `shape`, `error` and `duty`; everything else (face_clear,
heads_valid, settled, vel_fresh, armed, track, goal_calibrated) comes from the
log unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turret_host import config, control                   # noqa: E402
from turret_host.types import Detection                   # noqa: E402

FLIGHT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "diag", "flight")
RUNS = ("run_2026-09-20_144709", "run_2026-09-20_145109")

#: Conditions this harness re-decides. Everything else is read from the log.
RECOMPUTED = ("drone_lock", "shape", "error", "duty")


def load(run):
    path = run if os.path.isdir(run) else os.path.join(FLIGHT, run)
    with open(os.path.join(path, "control.jsonl"), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh], path


def box_of(row):
    e = row.get("est") or {}
    b = e.get("box")
    if not b:
        return None
    return Detection(b[0], b[1], b[2], b[3], float(e.get("conf") or 0.0),
                     "drone", "narrow")


def aim_target(row):
    c = row.get("cmd") or {}
    aim = (c.get("goal_u"), c.get("goal_v"))
    if aim[0] is None:
        return None, None
    tgt = (c["goal_u"] + c["error_u"], c["goal_v"] + c["error_v"])
    return aim, tgt


def replay(rows, hold_s=0.0, max_on_ms=None, cooldown_s=None,
           error_frac=None, aspect_gate=True):
    """Re-decide every row. Returns (fire_flags, notes)."""
    max_on_ms = config.LASER_MAX_ON_MS if max_on_ms is None else max_on_ms
    cooldown_s = (max_on_ms / 1000.0) if cooldown_s is None else cooldown_s

    on_since = None
    cooldown_until = 0.0
    held_box = None
    held_t = None
    fire = []
    held_used = 0
    duty_trips = 0
    lost_to_cooldown = 0

    for r in rows:
        t_now = r["t"]
        m = (r.get("interlock") or {}).get("margins") or {}
        logged_failed = set((r.get("interlock") or {}).get("failed") or [])
        # Conditions we do not touch, exactly as the live run found them.
        others_ok = not (logged_failed - set(RECOMPUTED))

        box = box_of(r)
        if box is not None:
            held_box, held_t = box, (r.get("est") or {}).get("box_t")
        use_box, use_t = box, (r.get("est") or {}).get("box_t")
        if box is None and hold_s > 0.0 and held_box is not None and held_t is not None:
            if 0.0 <= t_now - held_t <= hold_s:
                use_box, use_t = held_box, held_t
                held_used += 1

        aim, tgt = aim_target(r)
        if use_box is None or aim is None or use_t is None:
            age = float("inf")
            inside = float("-inf")
        else:
            age = t_now - use_t
            if not 0.0 <= age <= config.DRONE_BOX_MAX_AGE_S:
                inside = float("-inf")
            else:
                inside = control.beam_inside_box_px(use_box, aim, tgt)
        c_lock = inside > 0.0

        if use_box is None:
            aspect = float("nan")
        else:
            bh = use_box.y2 - use_box.y1
            aspect = (use_box.x2 - use_box.x1) / bh if bh > 0 else float("inf")
        c_shape = (config.FIRE_BOX_ASPECT_MIN <= aspect <= config.FIRE_BOX_ASPECT_MAX) \
            if aspect_gate else True

        err = abs(float((r.get("cmd") or {}).get("error_px") or 0.0))
        if error_frac is None:
            c_err = err < config.MAX_ERROR_TO_FIRE_PX
        else:
            if use_box is None:
                c_err = False
            else:
                lim = error_frac * min(use_box.x2 - use_box.x1,
                                       use_box.y2 - use_box.y1)
                c_err = err < max(lim, config.MAX_ERROR_TO_FIRE_PX)

        c_duty = t_now >= cooldown_until
        on = bool(others_ok and c_lock and c_shape and c_err and c_duty)

        # Duty accounting last, transcribed from control.py.
        if on:
            if on_since is None:
                on_since = t_now
            elif (t_now - on_since) * 1000.0 > max_on_ms:
                on = False
                on_since = None
                cooldown_until = t_now + cooldown_s
                duty_trips += 1
        else:
            on_since = None
        # Would this row have fired but for the cool-down? That is the cost of
        # the duty cap, and it is what makes the box hold a REGRESSION on its
        # own -- a longer burst reaches the cap sooner and then rests.
        if (not c_duty) and others_ok and c_lock and c_shape and c_err:
            lost_to_cooldown += 1
        fire.append(on)
    return fire, {"held_used": held_used, "duty_trips": duty_trips,
                  "lost_to_cooldown": lost_to_cooldown}


def stats(rows, fire):
    idx = [i for i, f in enumerate(fire) if f]
    if not idx:
        return {"rows": 0, "raw": 0, "merged": 0, "beam_s": 0.0, "longest": 0.0}
    dts = [rows[i + 1]["rel_t"] - rows[i]["rel_t"] for i in range(len(rows) - 1)]
    dt = statistics.median(dts)

    def group(maxgap):
        out, cur, last = [], None, None
        for i in idx:
            if cur is None:
                cur = [i, i]
            elif i - last - 1 <= maxgap:
                cur[1] = i
            else:
                out.append(cur)
                cur = [i, i]
            last = i
        if cur:
            out.append(cur)
        return out
    raw, merged = group(0), group(5)
    return {"rows": len(idx), "raw": len(raw), "merged": len(merged),
            "beam_s": len(idx) * dt,
            "longest": max(rows[b]["rel_t"] - rows[a]["rel_t"] for a, b in merged)}


def selftest(rows, label):
    """Baseline must reproduce the recorded FIRING decisions exactly.

    aspect_gate=False, deliberately: the shape gate was added AFTER these runs
    were recorded, so a harness that applies it cannot reproduce history. With
    it on, the only row that differs is f006835 -- the hand shot -- which is the
    gate doing exactly what it was added for. That single difference is itself a
    check on both the gate and this harness.
    """
    fire, _ = replay(rows, hold_s=0.0, aspect_gate=False)
    logged = [r["laser"] == "FIRING" for r in rows]
    diff = [i for i, (a, b) in enumerate(zip(fire, logged)) if a != b]
    print("  selftest %-22s %d/%d rows reproduced%s"
          % (label, len(rows) - len(diff), len(rows),
             "" if not diff else "   MISMATCHES: %s" % [rows[i]["frame_index"] for i in diff[:8]]))
    return not diff


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    runs = a.run or list(RUNS)

    print("VALIDATION -- the baseline variant must reproduce the run exactly")
    loaded = []
    allok = True
    for run in runs:
        rows, path = load(run)
        loaded.append((run, rows))
        allok &= selftest(rows, os.path.basename(path))
    if not allok:
        print("\nBASELINE DOES NOT REPRODUCE THE RUN -- every number below would be "
              "meaningless. Stopping.")
        return 2
    if a.selftest:
        return 0

    variants = [
        ("baseline (as recorded)", dict(hold_s=0.0, aspect_gate=False)),
        ("shape gate as shipped", dict(hold_s=0.0)),
        ("1  box hold 100 ms", dict(hold_s=0.100)),
        ("1  box hold 150 ms", dict(hold_s=0.150)),
        ("8  error < 0.35*min(w,h)", dict(error_frac=0.35)),
        ("1+8 hold 100 + box-rel err", dict(hold_s=0.100, error_frac=0.35)),
        ("2  duty 5000/500 ms", dict(max_on_ms=5000, cooldown_s=0.5)),
        ("1+2 hold 100 + duty 5000/500",
         dict(hold_s=0.100, max_on_ms=5000, cooldown_s=0.5)),
        ("1+2+8 all three",
         dict(hold_s=0.100, max_on_ms=5000, cooldown_s=0.5, error_frac=0.35)),
    ]
    for run, rows in loaded:
        print("\n=== %s ===" % run)
        print("  %-30s %7s %7s %8s %9s %9s %6s %8s" %
              ("variant", "rows", "raw", "merged", "beam s", "longest",
               "duty!", "lost2cd"))
        base = None
        for name, kw in variants:
            fire, notes = replay(rows, **kw)
            s = stats(rows, fire)
            if base is None:
                base = s
            d = "" if base is s else "  %+d rows" % (s["rows"] - base["rows"])
            print("  %-30s %7d %7d %8d %9.1f %9.2f %6d %8d%s"
                  % (name, s["rows"], s["raw"], s["merged"], s["beam_s"],
                     s["longest"], notes["duty_trips"], notes["lost_to_cooldown"], d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
