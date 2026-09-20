# Session 2026-09-19, afternoon. Supersedes nothing; read with SESSION_2026-09-19.md

The morning session ended believing the tracking loop was the problem. It is
not. This session measured the loop, found it sound, and found the real
blocker.

---

## THE ONE FINDING THAT MATTERS

**The detector prefers a person to the drone.**

`run_2026-09-19_132817`, frame 003597: the tracker is in TRACK at **conf 0.74**
on the operator's hood/shoulder while the drone is plainly visible in their
other hand, unboxed. Frame 002724: TRACK on a **foot in a slipper**, conf 0.37.

Real drone locks in the same run scored **0.37-0.47**. So the false positive
OUTSCORED the true positives. Raising `YOLO_CONF` makes this worse, not
better: 0.55 keeps the hood and discards the drone.

This explains the symptom the whole morning chased. 18 TRACK segments,
**median 0.36 s**, 10.7 s of a 154 s run, nearly all exactly 10 frames long
(`TRACK_MISSES = 10`). The tracker was not losing the drone. **It was never on
it** -- it locked a transient false positive, centred *that*, and the lock died
when the false positive stopped firing.

It also drove the turret **63 deg from its datum in 7.7 s** chasing one, until
the attitude envelope guard stopped it.

### The safety inversion

`drone_lock` is the POSITIVE evidence the interlock requires before firing, and
a person satisfies it. The head-veto is what is supposed to stop the beam near
a face -- so permission and veto end up pointing at the same body. Measured: 66
frames had a face measured against the aim point, **40 of them with a box
locked**, 12 with the face inside 100 px. `face_clear` fired 36 times and
`heads_valid` failed closed 593 times, so the layering held -- **as the last
line, not the first.** The beam stays disarmed.

---

## THINGS I GOT WRONG TODAY, AND WHAT CAUGHT THEM

Recorded because the corrections came from the operator looking at pictures,
not from any number in the logs.

1. **"The filter is 20-31 px worse than not predicting."** FALSE. It was
   measuring `AIM_BIAS_UP_FRAC = 0.15`, a deliberate 41.7 px upward push, as
   though it were error. Against the box centre the estimate sits 44.2 px out;
   against the BIASED centre it sits **2.5 px** out. The bias explained 94% of
   it. Now `AIM_BIAS_UP_FRAC = 0.0` -- it existed to spare the fingers of
   whoever holds the drone, and it is a 5 mW demo pointer. See
   `aim-bias-not-error` in memory.

2. **"The association gate is too tight."** FALSE. What the gate rejects sits
   at **median 236 px in TRACK, 517 px in COAST** -- different objects, not
   near-misses. Target motion between detections is <=53 px at p95, well
   inside the 60 px gate. The adaptive gate and re-seed built for this are
   harmless (strictly additive, self-test clean) but were aimed at nothing.

3. **"Cleanly boxed."** The operator: *"that is not cleanly boxed, thats
   lagging behind."* Correct -- and the cause was mine: the recorder drew
   `est.box` (32-54 ms stale) onto the CURRENT frame. Fixed; frames now caption
   `BOXAGE=NNms` and draw the live aim point as a cross.

4. **The strided annotation batch** would have handed back frames a median of
   **2 frames (70 ms)** from ones already judged, because the stride ran over
   the *remaining* list rather than the timeline. Fixed with a minimum-gap
   filter against already-reviewed frames.

**The rule that came out of this, now in memory as `look-at-the-frames-rule`:**
open the images at track-START and track-LOSS before quoting any metric. A box
is not the drone until an image says so.

---

## MEASURED AND DONE -- THE CONTROL SIDE IS FINISHED

| what | value | how |
|---|---|---|
| loop latency | **66 ms** | laser flashed at a wall, n=16, 0 misses, 51.2 ms median + 15 ms inference |
| gain | K=1.2 = **0.63x** the 1/(8L) ceiling | correctly placed; stop tuning it |
| command reversals | **5-9%** magnitude-weighted | not oscillating |
| host<->board clock | **0.29 ms** residual, +4.3 ppm | verified 25/25 against a command the fit never saw |
| gyro | **200 -> 500 Hz**, DLPF 42 -> 188 Hz | ODR verified at 489 Hz by fresh-sample counting, not by trusting the register write |
| step integrity | payload returns to **0.0 px** | real 300/-260 microstep move and rewind, camera phase-correlation |
| filter velocity | tracks truth at **0.92-0.95** | it is not lagging |

---

## BUILT THIS SESSION

- `turret_host/clocksync.py` -- Cristian's algorithm, minimum-RTT filtered,
  fits offset AND skew. Validated against `imu fast`, a command it was not
  fitted on.
- `turret_host/step_integrity.py` + wired into `app.py` -- notes the datum
  after homing, rewinds every commanded microstep at shutdown, reports what
  did not come back. **Pitch from gravity, yaw from the magnetometer**: at
  level, gravity is completely blind to yaw, and on a differential that is the
  axis where ONE motor slipping shows up. Placement in `shutdown()` matters --
  after the threads join and `_safe_state` leaves velocity mode, before the
  port closes.
- `tools/step_check.py` -- the same check standalone, with `--capture` /
  `--verify`.
- `tools/analyze_run.py` -- scores a run on LEAD ERROR, with a naive
  "it is where it was last seen" control, split by source camera, plus a
  magnitude-weighted reversal metric.
- Firmware **iteration 13**: `clk` handshake, `ticks_ms` appended to `IMUF`
  (appended, so all three unanchored host parsers survive), gyro at 500 Hz /
  188 Hz, `imu burst` rate probe.
- Recorder: writes `raw/` (clean JPEG + YOLO sidecar label) beside the
  drawn-on `frames/`; logs association REJECTS with distances; snapshots 20
  config constants into `session.json`.
- `confirm_labels`: `--stride`, `--max-width/--max-height/--full`, no
  upscaling past 1:1.
- `capture_dataset --negatives-only --frames N`.

---

## DATASET AND MODELS

Operator reviewed **2511 frames** (1903 confirmed, 380 empty_ok, 228
rejected), then captured a 500-frame negatives-only session.

```
exported  train 2871  val 926   1896 boxed  1901 negatives (50%)
```

50% negatives is far above the usual 0-10% guidance. Deliberate -- the failure
is false positives on people -- but **it may cost recall, and that is
unmeasured.**

`drone_y11n_v4_1280` is training: imgsz **1280** to match inference (was 640;
the model learned 66 px drones and was shown 132 px ones), batch 4,
**workers 2**. Workers matter: the default 8 killed the run at imgsz 1280 with
`DataLoader worker exited unexpectedly` during the first VALIDATION pass. That
looks like a CUDA error and is not -- GPU was 3.0 of 8.1 GB.

`DRONE_WIDTH_M = 0.289` (DJI Mini 2, unfolded with props). Cross-checks
against this rig's own footage: 132 px at 3 m implies 283 mm; published 289.
**2% apart from independent sources.** Ranging now returns 1.35 m at 300 px
through 5.06 m at 80 px.

---

## OPEN, IN PRIORITY ORDER

1. **NO HELD-OUT TEST SET EXISTS.** Every frame ever captured is in training,
   and the val split is TEMPORAL within each clip, so val frames are
   near-duplicates of train frames a fraction of a second away. **v4's mAP
   cannot tell you whether v4 is better.** Two minutes of capture fixes this
   permanently; it must never be ingested.

2. **No unheld drone in the dataset, at all.** Every positive frame has it in
   a hand, so "drone" and "hand gripping an object" are perfectly correlated.
   The demo is a FLYING drone with no hand near it -- out of distribution for
   a model that only knows held ones. Needs: drone on a table, chair, box at
   heights, hung on thread.

3. **Goal pixel has no parallax term.** `c_u = c_v = 0`, `n_observations = 1`
   -- `g(R) = g_inf + c/R` needs two ranges. Worth ~17 px across 2-5 m against
   a 25 px firing threshold. Needs a flat target at two measured distances;
   the pure angle-sweep alternative fails because the perpendicular direction
   is unknown and a 15 deg error in it gives up to 40% range error.

4. **Two measurements of the laser dot disagree by 13 px**, same range, same
   wall: `goal_pixel.json` (635.08, 327.77) vs the structured-light session's
   (642.30, 316.65). Half the firing budget, unexplained. This caps aim
   accuracy regardless of the parallax term.

5. **Platform compensation is unbuilt.** The "interpolate between inference
   captures" half of the deliverable. Image motion = drone motion + turret
   motion; the filter cannot separate them, but the turret's share is
   COMMANDED and therefore known in advance -- better than the gyro, which
   only reports what already happened. The correction reduces to
   `J.pixel_rates(omega_now - omega_at_last_update) * L`: zero when the
   command is steady, nonzero exactly when it changes.

6. 216 frames of drone footage in the old set still carry EMPTY labels,
   teaching "drone = background".

7. **Face-interlock rotation sign still unproven.** `STATUS.md` item 1. Do not
   arm.

---

## SNAPSHOT

`snapshots/2026-09-19_151211/` (60 MB) -- source, firmware, all `.pt` weights,
`review_state.json`, every label file, session manifests, docs. Path in
`snapshots/LATEST`. Restore by copying directories back over the top. The
3.1 GB of dataset images and 494 MB of `diag` logs are deliberately excluded;
snapshot those too before anything that writes to `dataset/yolo`, in particular
`capture_dataset --split`, which wipes every label.
