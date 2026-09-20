# Optimisation pass — staged, NOT APPLIED

    staged/tracker.py, staged/config.py  -> items 1 and 4
    staged/replay_gates.py               -> new file, belongs at tools/
    staged/host_2026-09-20_items1and4-tracker.patch

    patch --binary -p0 --dry-run < staged/host_2026-09-20_items1and4-tracker.patch

Applies clean and reproduces the staged files byte-for-byte. `turret_host/` is
untouched. COM5 not taken.

---

## The replay harness comes first, because without it none of the numbers mean anything

`staged/replay_gates.py` re-decides the interlock over a recorded run.

**It is a faithful replay, not an estimate, and here is why:** the interlock does
not move the turret. Changing an interlock rule changes which rows fire and
nothing else. That holds for item 1 too, because of a specific fact about the
tracker — `TrackEstimate.box` is `_last_det`, and `_last_det` feeds *only*
`estimate.box`; `_range()` and `_aim_bias()` both use `_stable_wh` instead. So a
held box cannot reach the control law. Checked in the source, not assumed.

**Validation:** the baseline variant reproduces the recorded FIRING decision on
**2557/2557** rows of 144709 and **2312/2312** rows of 145109. It also
reproduces the logged `inside_px` margin exactly on 2005/2170 rows; all 165
exceptions are `INHIBITED - face` rows and **none is a FIRING row**, so they
cannot touch any result here.

It does **not** cover item 4: accepting a detection the gate rejected changes
the filter, the command and every later frame. That one gets a counterfactual
and frames, never a replay number.

---

## THE HEADLINE: the duty cap is the binding constraint, and it inverts item 1

Replay of 145109. `duty!` is how many times `LASER_MAX_ON_MS` forced the beam
off; `lost2cd` is rows that passed every other gate and were refused by the
cool-down.

| variant | rows | raw bursts | beam s | longest | duty! | lost2cd |
|---|---|---|---|---|---|---|
| baseline (as recorded) | 638 | 62 | 21.0 | 6.05 | 1 | 52 |
| shape gate as shipped | 637 | 61 | 21.0 | 6.05 | 1 | 52 |
| **1  box hold 100 ms** | **567** | **40** | **18.7** | 2.79 | **3** | **159** |
| 1  box hold 150 ms | 576 | 35 | 19.0 | 2.00 | 3 | 158 |
| 8  error < 0.35·min(w,h) | 648 | 57 | 21.4 | 5.26 | 2 | 89 |
| 2  duty 5000/500 ms | 690 | 63 | 22.7 | 6.05 | 0 | 0 |
| **1+2 hold + duty** | **729** | **45** | **24.0** | 6.05 | **0** | **0** |
| 1+2+8 all three | 766 | 42 | 25.2 | 5.88 | 1 | 11 |

**Item 1 on its own is a regression: −71 firing rows and −2.3 s of beam.**

The hold does exactly what it was asked to do — raw bursts fall 62 → 40, so the
single-miss fragmentation is fixed. But stitching those bursts together builds
*continuous* on-time, which reaches the 2000 ms cap sooner, and each cap costs a
2000 ms rest. Duty trips go 1 → 3 and rows lost to cool-down 52 → 159. The cure
costs more than the disease.

With the duty cap relaxed the same hold is worth **+91 rows and +3.0 s**.

**So item 2 is not second, it is first.** It is also the single largest item on
the list by itself (+52 rows, +1.7 s, and it removes all 52 cool-down refusals),
it is one constant, and it is Ryan's call rather than mine. Recommend putting
`LASER_MAX_ON_MS 5000` / `LASER_COOLDOWN_S 500 ms` in front of him before
anything else is applied.

---

## Item 8 buys beam time by taking the beam off the drone. Do not ship as specified.

Box-relative error gate, `error < 0.35·min(w,h)`: +49 rows with the duty cap
held constant, and **0 rows lost to the gate itself** (the 28 it appears to lose
are all cool-down, confirmed by re-running with the cap disabled).

Then I looked at the frames. Beam point marked from `cmd.goal_u/goal_v`:

* **Newly permitted rows (60–77 px error): 5 of 6 put the beam on empty space
  inside the box** — f005263, f005304, f006328, f006181, f006146 all land on
  wall or background between the rotor arms. Only f005695 is on the airframe.
* **Baseline rows (0–60 px error): 8 of 9 are on the airframe** — f006565,
  f005351, f006591, f006574, f005197, f006553, f005391, f006107. Only f005187 is
  marginal.

The marker is independently validated: in every baseline frame the magenta beam
marker sits exactly on the visible green laser splash, so it is genuinely where
the light goes.

**Why:** `drone_lock` certifies the beam is inside the *box*, and a quadcopter's
box is mostly empty air between the arms — the body is a small central blob. The
60 px gate is the only thing keeping the beam on the body, and at this range it
sits almost exactly at the body's edge. Loosening it to 0.35·min(w,h) (63–80 px
here) pushes the beam off the airframe.

If a box-relative gate is still wanted, the fraction should be **tighter, not
looser** — the crossover in these frames is around 0.28·min(w,h), so ~0.25 with
the existing 60 px as a *ceiling* rather than a floor. That helps the genuinely
near-drone case the brief describes without buying wall time. It still needs the
aim-point evidence check before it is trustworthy.

---

## Item 9: no motion limit ever bound. Nothing to do.

| | 144709 | 145109 |
|---|---|---|
| peak \|rate\| | 1777 steps/s (44% of MAX_MOTOR_RATE) | 919 (23%) |
| rows above 90% of 4000 | 0 | 0 |
| `cmd.saturated` | 156 (6.1%) | 6 (0.3%) |
| median ramp time to the commanded rate | 4.9 ms | 2.2 ms |

Against a 33 ms command period, the VEL_ACCEL ramp costs 2–5 ms — the 100 ms
figure only applies near the 4000 steps/s ceiling, which was never approached.
`p_scale`/`authority` hit 0.0 on 142/31 rows, but that is `P_SCALE_DECAY`
refusing to act on stale evidence, not a rate limit.

**The turret is not limited by its motion constraints.** It is limited by the
control law and the sensing, which is where items 3, 5, 6 and 7 point. Leave
MAX_MOTOR_RATE, WIDE_MAX_MOTOR_RATE and VEL_ACCEL alone.

---

## Item 1, as staged

`FIRE_BOX_HOLD_S = 0.100` in config; tracker keeps `_held_det` / `_held_det_t`
separate from `_last_det`, which is unchanged. The held box carries its
**original** `box_t`, so `DRONE_BOX_MAX_AGE_S` still expires it on the real age
and containment still runs against the current aim point every frame. SEARCH
never offers a held box — a firing permission must not outlive its track.

Unit-tested against the real `PixelTracker` (no hardware): held for exactly 3
frames at 30 fps, reported age tracks elapsed time, a fresh hit takes over, the
box disappears on frame 4, and SEARCH offers nothing.

**Ship it only together with the duty change.**


---

## Item 4 — shape reject and the lone-detection rule

Two config constants and two small tracker changes, in the same patch as item 1.

### `ASSOC_BOX_ASPECT_MAX = 1.6`, and it is DELIBERATELY ONE-SIDED

The hand signature is **wide**: 1.67–2.22 across the 17 associated boxes of the
recorded hand approach. The drone held edge-on or folded is **tall and narrow**
— 0.39–0.48 on 17 frames across both runs, and I opened them: 145109 f004546,
f004638, f005092 and 144709 f002569, f003527 all show the real airframe against
the carrier's torso.

So there is **no low bound**. A symmetric window borrowed from
`FIRE_BOX_ASPECT_MIN` would throw those away and lose the track, which is the
failure item 4 exists to reduce. Refusing to *fire* on an odd box is free;
refusing to *associate* one is not.

Measured across both runs: of the 55 accepted boxes outside the firing window,
this rejects the 27 wide ones (25 of them the hand cluster) and keeps all 28
narrow ones.

### The filter goes in `_update()`, not in `_associate()` — found by testing

The first version put the check inside the gate. The counterfactual then showed
the staged tracker *still* adopting hand boxes, because **the gate is not the
only way a detection becomes the track**: `_update` has three seed paths that
bypass `_associate` entirely — the first frame ever, the SEARCH re-entry, and
the `RESEED_AFTER_MISSES` branch — and every one takes `max(dets, key=conf)`
directly. The reseed path was adopting the hand after six misses.

Moving the filter to the single point every path goes through closes all three.

### Counterfactual on the hand approach, f006700–f006840

Real logged detections fed through both trackers:

| | associated a box | of those, wider than 1.6 |
|---|---|---|
| live tracker | 18 frames | **18** |
| staged tracker | **0 frames** | 0 |

The staged tracker stays in SEARCH across the whole approach, so the aim point
never walks onto the hand. **This is a counterfactual, not a replay:** a tracker
that refused those boxes would have commanded differently and the later
detections would have landed elsewhere. What it settles is the question asked —
whether hand-shaped boxes are still taken as measurements.

### `ASSOC_LONE_PX = 250`, `ASSOC_LONE_CONF = 0.5`

A lone confident detection outside the covariance gate is accepted: with one box
in frame there is nothing to mistake it for, so a large innovation says the
*prediction* is stale. In 144709, 21 of 26 TRACK losses were the real drone at
conf 0.54–0.81, a median 241 px away; the rule would accept 46 of the 136 rows
that had a detection and associated nothing (6 of 13 in 145109). Shape-filtered
upstream and still capped by `GATE_MAX_PX`.

### Behavioural tests, no hardware

| case | result |
|---|---|
| normal square drone | TRACK |
| edge-on drone, aspect 0.40 | **TRACK** (must not be rejected) |
| hand-shaped, aspect 2.22 | **SEARCH**, `last_shape_dropped` 1 |
| lone 200 px jump, conf 0.7 | associated |
| lone 200 px jump, conf 0.3 | rejected (conf guard) |
| lone 280 px jump, conf 0.7 | rejected (beyond 250) |
| 200 px jump, two detections | rejected (not lone) |
| item 1 hold | still exactly 3 frames, original `box_t`, SEARCH offers none |

One caution for whoever reads the next run's logs: **item 1 changes what
`est.box is not None` means.** It is now "there is a usable measurement",
not "this frame associated". A fresh association is `box_t == frame_t`. My own
first test got this wrong.


---

## Item 6 (e) — optical flow as the velocity source. It works, and by a lot.

`staged/flow_velocity_study.py`, offline on the saved `raw/` frames. Four
sources scored on the only thing the feedforward needs them for: predicting
where the target is one loop-latency from now. Truth is the detector's actual
box centre at t+h.

**Median prediction error, px:**

| horizon | run | naive | boxdiff (current) | filter | **flow** |
|---|---|---|---|---|---|
| 33 ms | 144709 | 11.8 | 6.2 | 5.8 | **3.9** |
| **66 ms** | **144709** | **22.1** | **11.2** | **10.3** | **6.0** |
| 100 ms | 144709 | 32.7 | 17.1 | 15.1 | **8.9** |
| 200 ms | 144709 | 63.4 | 38.1 | 34.6 | **21.4** |
| 33 ms | 145109 | 4.5 | 4.3 | 3.5 | **2.8** |
| **66 ms** | **145109** | **7.4** | **7.3** | **5.8** | **4.1** |
| 100 ms | 145109 | 10.4 | 10.7 | 8.3 | **5.5** |
| 200 ms | 145109 | 18.4 | 21.7 | 16.3 | **10.5** |

Flow wins at every horizon on both runs, and **its margin is largest on the
fast run** — 66 ms, 144709: 6.0 px against the filter's 10.3 and naive's 22.1.
That is the case the whole optimisation is for.

**Frame-to-frame velocity jump — the quantity that made the feedforward flip
sign at ±150–300 px/s:**

| source | 144709 median / p90 | 145109 median / p90 |
|---|---|---|
| boxdiff (current) | 190.8 / 574.2 | 130.7 / 346.6 |
| filter | 84.9 / 249.7 | 39.0 / 142.8 |
| **flow** | **50.8 / 179.1** | **26.2 / 80.6** |

Roughly a 4x reduction against the source now feeding the term, on both runs.

### The current velocity source is worse than having none at all

On 145109 at 100 ms and 200 ms, **boxdiff is beaten by naive** (10.7 vs 10.4,
21.7 vs 18.4). Differencing box centres is not merely noisy at long horizons on
a slowly-carried target — it is actively worse than assuming the target does not
move. That is the strongest single argument for replacing it rather than
filtering it harder.

### Availability is the catch, and it must be designed for

Flow answered on **85%** of consecutive pairs in 144709 and **96%** in 145109
(median 113 and 118 surviving LK points). The gap tracks image speed — 144709's
carrier moved at a median 358 px/s against 131 — so the misses are motion blur
eating the corners, exactly when the velocity matters most. **A feedforward
built on this needs a defined fallback for the ~15% of fast frames where flow is
silent**, and that fallback must not be the box-differenced velocity it was
brought in to replace. Ramping `FEEDFORWARD_GAIN` off zero before that exists
would reintroduce the sign-flipping on precisely the frames that caused it.

### Two honest notes

* **I expected the laser splash to contaminate flow on firing frames** — it is a
  bright feature that moves with the BEAM, not the drone. It is not visible in
  the result: 145109 has 638 firing rows and the *better* availability of the
  two. 144709 has **zero** firing frames, so it is a splash-free control, and
  flow wins there by the larger margin. The concern is recorded as checked, not
  as confirmed.
* **The note in memory that "naive beats the filter at every horizon" does not
  reproduce on these two runs.** Here the filter beats naive everywhere. Either
  that earlier measurement was on different data or the filter has since
  improved. Flagged rather than quietly contradicted — it changes how much the
  filter is worth keeping as the fallback above.

### Guards, so a degenerate answer is silence rather than a confident zero

Forward-backward reprojection under 1.0 px, at least 6 surviving points, and a
coherence test (median absolute deviation of the surviving vectors under 3.0 px).
A blurred drone against a flat wall will otherwise return a confident number
built from nothing. Frames failing any guard are reported as NO MEASUREMENT and
excluded, which is why the flow column has a smaller n than the others.

`raw/` frames are used, never `frames/` — the latter has the box and the aim
cross burned in, and tracking the overlay would measure the estimate, which is
the thing under test. Verified by eye on f005394.
