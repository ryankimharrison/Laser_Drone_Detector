# Next stage: one clock, one log, a human in the loop

Planning document, written 2026-09-19 with no hardware attached. Nothing here
is measured yet; every number quoted from earlier work is marked as such.

---

## 0. THE GOAL, WRITTEN AS A NUMBER

> "Pretend the laser is always on. It stays centred in the drone's body
> regardless of distance or speed."

That is a statement about ONE quantity, and the whole stage should be built to
measure it:

```
LEAD ERROR  =  || where the filter said the drone would be at time t
                 - where the drone actually was at time t ||     (pixels)
```

Not "was the detection good". Not "did the loop look smooth". The controller
commits to an aim point L = 66 ms before the beam gets there (MEASURED
2026-09-19). Lead error is the distance between that commitment and the truth.
If lead error is under `MAX_ERROR_TO_FIRE_PX = 25`, the laser is on the drone.
If it is not, nothing else matters.

It is computable entirely offline from a log: at frame N the filter predicted
a position for `t_N + L`; some later frame actually observed that time. The
residual is the answer. **This is the metric the annotation loop, the clock
sync and the motion model all exist to improve**, and it is the one number
every test run should report.

Decompose it, because the three terms have different fixes:

```
lead error  =  detection error      (is the box on the drone?)      -> model, annotation
            +  platform term        (did MY motion move the image?) -> gyro, Phase 5
            +  target motion term   (did the DRONE manoeuvre?)      -> motion model
```

---

## 1. WHY A SHARED CLOCK IS THE PREREQUISITE

Every claim in the decomposition above is a claim about *simultaneity*: the
IMU read at the same instant as this frame, the motor command that was in
flight while that frame was exposing. Right now the recorder
(`turret_host/telemetry.py`) puts everything on one host `perf_counter`, which
is honest but records **when the host asked**, not **when the thing happened**.
Those differ by the serial round trip and the camera pipeline, and the camera
pipeline alone is most of the measured 51.2 ms.

Target, and it should be verified rather than assumed:

```
sync residual  < 2 ms     (negligible against L = 66 ms and a 33 ms frame)
sync residual  > 20 ms    would corrupt every cross-reference in this document
```

### 1.1 The protocol: Cristian's algorithm, minimum-RTT filtered

```
host:  t0 = perf_counter()
host -> pico:  "clk"
pico -> host:  "CLK <ticks_ms>"          # no I2C, no work, print and return
host:  t1 = perf_counter()

pico tick <ticks_ms> happened at host time (t0 + t1)/2,  uncertainty +/- (t1-t0)/2
```

Take ~20 samples, **keep only the lowest-RTT decile** and fit from those. This
is the standard trick and it matters here: a sample delayed by USB scheduling
or a busy firmware loop is biased in one direction only, so the minimum-RTT
samples are the least contaminated. Averaging all samples averages in the bias.

### 1.2 Fit offset AND skew, not just offset

Two independent crystals. A Pico's is typically 10-50 ppm from the PC's. At
30 ppm a ten-minute run drifts **18 ms** -- a quarter of the entire latency
budget, from a single sync at startup. So:

```
t_host = a * pico_ms + b
```

Least squares over sync samples collected throughout the run, with the
residual stored next to the coefficients. `a - 1` IS the relative crystal
error, in ppm, and it should come out in the tens. If it comes out in the
hundreds, something is wrong with the fit, not with the crystal.

Re-sync continuously, not once. Cheapest way: have the firmware stamp its tick
onto the reply of `imu fast` (which the attitude sampler already polls), so
every IMU poll is also a clock sample at no extra cost. Keep the dedicated
`clk` for the clean low-RTT samples the fit is built on.

### 1.3 The firmware question, answered honestly

**This needs exactly one flash, and then never again.** There is no way to
read an independent Pico clock without the Pico offering one. What the flash
buys is permanent:

- `clk` -> `CLK <ticks_ms>`, doing no other work
- `imu fast` reply gains the tick **at the moment of the I2C read**
- motor command ack gains the tick at the moment step generation started
- **a real gyro stream** -- see 5.0. `imu watch` already exists for this and
  is broken (exits on its own CRLF); fixing it belongs in this same flash,
  because a 10 Hz polled IMU makes the entire stabilisation plan unbuildable.

After that the protocol is fixed and the host side can change freely. Use
`tools/deploy.py` with a snapshot and iteration number, per the usual rule.

Use `ticks_ms`, not `ticks_us`: `ticks_us` wraps every ~17.9 minutes, which is
inside a test session. `ticks_ms` wraps at ~12.4 days. Parse with wrap-aware
arithmetic anyway -- a log that silently jumps backwards is worse than one
that refuses.

### 1.4 The validation, and it must be external

A sync verified against itself proves nothing. **Use the laser as a hardware
fiducial**, exactly as `measure_latency.py` already does:

1. Pico pulses the beam and reports the tick at which it did.
2. The dot appears in some camera frame at a known host time.
3. The clock model predicts which frame that should be.
4. Compare.

This is one test that checks the clock model, the frame-time offset and the
camera pipeline delay **together, end to end, against photons**. Run it at the
start and the end of every long session; the start-to-end difference is the
drift the skew fit was supposed to remove, so it also grades the fit.

It also *calibrates* something currently unmeasured: a frame's host receive
timestamp is not its capture time -- exposure, readout, USB and decode sit in
between. The fiducial measures that offset directly, and then every frame
timestamp in the log can be corrected by it instead of estimated.

---

## 2. THE LOG

Extend `turret_host/telemetry.py`; do not start over. It already writes
`control.jsonl`, `imu.jsonl` and annotated frames on one clock, with frame
writes on a worker thread behind a bounded queue so the control loop never
stalls on a JPEG (verified: 30 Hz held, 0 dropped). Keep all of that.

What each record gains:

| stream | add |
|---|---|
| every record | `pico_ms` where the Pico is the origin; `t_host` always |
| frames | `t_capture` = receive − measured pipeline delay (§1.4), not receive |
| detections | `t_frame` inherited, plus `t_infer_done`; conf, box, class |
| tracker | state, `q`, P trace, gate decision **and the rejects** |
| control | commanded rate, the `e` it acted on, saturation, authority |
| IMU | `pico_ms` of the sample; gyro at the highest rate the link allows |

Two things the current log does not capture and the analysis needs:

- **Association rejects.** Today a detection outside the gate vanishes. Log
  every rejected detection with its distance from the prediction and the gate
  radius that rejected it. This is the raw material for answering whether the
  gate is too tight (the open question from 2026-09-19: 79% seen, 0% kept in
  COAST) and it costs nothing.
- **The prediction itself.** Log what the filter predicted for `t + L`, with
  the `t + L` it was predicting for. Without this, lead error cannot be
  computed after the fact, and lead error is the goal metric.

Storage sanity: a 3-minute run at 30 fps is ~5400 frames per camera. JPEG at
quality 85, 1280x720 is roughly 150 kB, so ~800 MB per camera per run. Write
full frames only for sampled/locked frames and keep a downscaled proxy for the
rest, or the disk becomes the experiment's limiting factor within a day.

---

## 3. THE ANNOTATION LOOP

The instinct is right: **low-confidence detections are the high-information
ones**. That is uncertainty sampling and it is a real strategy. Three
additions, in descending order of how much they matter.

### 3.1 You cannot fix misses by labelling boxes the model drew

The detector finds the drone on 73.2% of frames (MEASURED). The other 26.7%
produce **no box at all** -- and misses are what break lock, so they are the
expensive failure. A sampler that only shows proposed detections will never
surface one, and the model will never learn from them.

The fix is free, because the tracker already knows where the drone should
have been: **sample frames where the filter held a confident track and the
detector produced nothing.** Those are candidate false negatives, pre-located.

### 3.2 Freeze a random test set NOW, before any of this

Actively-sampled data is, by construction, not representative. Metrics
computed on it are meaningless, and worse, they will look like they are
improving. So:

- Sample a **uniformly random** slice of every run into a test set.
- Label it, never train on it, never resample it.
- Every claim of "the model got better" is measured only here.

Do this before the first active-learning round or the baseline is gone for
good.

### 3.3 Present the crops blind

The loop's whole value is a human injecting information the model does not
have. Show the annotator the model's confidence and they anchor to it --
0.34 primes "probably not", and the label stops being independent. Show the
crop and its surroundings, nothing else. Record confidence in the log, not on
the screen.

### 3.4 The strata

Per run, with a minimum spacing of ~0.5 s (15 frames) between samples --
consecutive frames at 30 fps are near-duplicates and labelling both spends
human time for almost no information:

| stratum | what | why |
|---|---|---|
| A | uniform random | the frozen test set (§3.2) -- never trained on |
| B | conf in [0.20, 0.50] | uncertainty sampling; the original instinct |
| C | track confident, detector silent | false negatives (§3.1); breaks lock |
| D | detected but gate-rejected | is the gate too tight, or is that clutter? |
| E | conf > 0.80, small sample | catches confidently-wrong |

D is worth calling out: it answers tonight's open question with labels instead
of argument. If most gate rejects are the drone, widen the gate. If most are
clutter, the gate is right and the problem is elsewhere.

Respect the existing dataset rules: **label after splitting** (`--split` wipes
every label) and `has_drone` is per-clip, not per-camera.

### 3.5 The honest limit

A system trained on its own tracker's output converges on agreeing with
itself. The human label is the only new information entering the loop, so
**throughput is bounded by how many frames a person will look at**, not by
compute. That is why stratification matters more than volume: the job is
information per human-second. At ~2 s per confirm/deny, 150 frames is about
five minutes and ten runs gives ~1500 curated hard examples -- a real
increment on top of v2.

---

## 4. ANALYSIS, RUN THE SAME WAY EVERY TIME

One script, one report per run, so runs are comparable:

1. **Lead error** distribution -- the goal metric (§0). Median, p95, fraction
   under 25 px.
2. Lead error **vs drone speed** and **vs range**. "Regardless of distance or
   speed" is a claim about these two curves being flat. Plot them and find out.
3. **Lock survival**: how long locks last and what ends each one (miss streak,
   gate reject, envelope trip).
4. **Gate audit**: rejected detections, their distance from prediction, and
   (once labelled) how many were the drone.
5. **Platform vs target decomposition**: how much of the image motion was the
   turret moving, from the gyro, and how much was the drone.

Keep writing the hypothesis before the run, as in `diag/flight/HYPOTHESIS.md`.
Three of four hypotheses were denied last session and the denials are what
produced the fixes.

---

## 5. WHAT ACTUALLY ACHIEVES "CENTRED AT ANY SPEED"

Logging does not improve tracking; it tells you which of these to do. Listed
with the reasoning, to be confirmed or denied by §4.

### 5.0 THE IMU RATE IS THE BINDING CONSTRAINT  (found 2026-09-19)

`app.ATTITUDE_SAMPLE_HZ = 10.0`. At the image scale this system actually has:

```
NARROW_F_PX 1400  ->  24.4 px per degree
max slew  AXIS_STEP_DEG * MAX_MOTOR_RATE = 195 deg/s  =  4765 px/s
observed in flight logs                    100 deg/s  =  2443 px/s

sample gap        platform motion between samples, at 100 deg/s
IMU today 10 Hz     100 ms      244 px     <- COARSER THAN A FRAME
camera    30 Hz      33 ms       81 px
          100 Hz     10 ms       24 px
          200 Hz      5 ms       12 px
          500 Hz      2 ms        5 px
                                 25 px = MAX_ERROR_TO_FIRE_PX
```

**The IMU is currently a coarser clock than the camera**, so it cannot
interpolate between frames -- it is the thing that would need interpolating.
Everything in 5.1 below is unbuildable until this changes.

10 Hz was not a mistake: it was built for the attitude ENVELOPE GUARD, which
asks "has the payload tilted past 60 degrees" off the accelerometer, and 10 Hz
is ample for that. Two consumers, two rates. The envelope guard keeps its slow
poll; stabilisation needs a separate high-rate gyro path.

**Target 500 Hz gyro** (5 px at the observed slew, 10 px at full slew). 200 Hz
is the floor at which the platform term drops under the firing threshold.

#### Why polling cannot get there, and what replaces it

`imu fast` is request/response: host write, Pico parse, two I2C bursts, reply.
USB CDC schedules in 1 ms frames each way, so a round trip has a ~2 ms floor
before any work happens -- which is the entire budget at 500 Hz. And the link
is SHARED with the control loop. We have already broken it once this way: a
telemetry probe at an 80 ms timeout against a slowed `imu` left bytes in the
buffer and produced `no STATE json`. Scaling a poll 50x on that link is not a
tuning change, it is a new failure mode.

So the Pico must **stream**, unprompted, one-way, no round trip per sample:

- `imu stream on|off <hz>` -- Pico-side timer loop, each sample carrying its
  own `ticks_ms`. **`imu watch` already exists and is meant to do this**; it is
  broken because it exits on its own trailing CRLF via `key_pressed()`. Fixing
  that is part of the same single flash in 1.3, not an extra one.
- **Batch several samples per write.** At 1 ms USB frames, one line per sample
  caps out near 1 kHz with most of the bandwidth spent on framing.
- Compact or binary records. At 500 Hz a ~25 byte text line is 12.5 kB/s,
  which USB CDC handles trivially -- the real limit is MicroPython's string
  formatting, so measure it rather than assume.
- The stream must not be able to break what it monitors: a missing IMU must
  stop the stream, not disable the axes on every tick.

#### The trap: sample rate is not bandwidth

The ITG3205 has a configurable digital low-pass filter. If the DLPF is left at
a low bandwidth, sampling at 500 Hz returns 500 copies of a heavily smoothed
signal and buys nothing -- the transients at the start and end of each move,
which are exactly what needs capturing, are filtered out before the sample.
Set the DLPF to roughly half the sample rate (the part offers 256/188/98/42/
20/10/5 Hz; 188 Hz suits 500 Hz sampling) and **verify the noise floor rises**
when the bandwidth opens. If it does not, the register write did not take --
the same class of silent failure as the ADXL345 standby latch.

Also worth checking in the datasheet before writing the loop: if the part has
a FIFO, burst-reading it beats a per-sample poll by a wide margin.

### 5.1 Predict in a gyro-stabilised frame

The camera is on a moving platform, so image motion = drone motion + platform
motion. A pure image-plane filter cannot tell those apart, and the platform
term is the one that is **already known** -- commanded, and measurable by the
gyro at far above frame rate.

Subtract the platform's own rotation, predict the drone in a stabilised frame,
re-project. Two wins: the motion model stops being asked to explain the
turret's own slew, and the platform term can be propagated **between frames**
at gyro rate rather than waiting 33 ms for the next one.

This is the single change most aligned with "faster than the cameras can
capture", and it is why gyro logging is in Phase 2. Note the known constraint:
the IMU is off the rotation axis, so accelerometer tilt is contaminated by
`alpha * r` during motion -- **integrate the gyro, do not difference accel
tilt.**

### 5.2 Range from box size, for the aim point

Rotation-only pointing is range-independent for the *control* -- rotating the
turret moves the image by an angle whatever the distance. But the **laser is
offset from the camera**, so where the dot lands depends on range; that is
what the existing `g_inf + c/r` goal-pixel model is for, and it currently runs
on `ASSUMED_RANGE_M`.

With one known drone of known width, `range = f * W_real / w_px` is available
on every frame for free. That turns "regardless of distance" from an
assumption into a measurement. Log box size now; fit later.

### 5.3 A motion model that allows acceleration

The filter is constant-velocity. A manoeuvring drone is not. Constant-
acceleration, or an IMM switching between them, is the standard answer -- but
**do not do this before §4 says the target-motion term is actually the
dominant one.** It may well be the platform term, in which case §5.1 is the
whole fix and a fancier model just adds noise.

### 5.4 The unglamorous lever: sample faster

Prediction exists to cover the gap between samples. Halving the gap beats any
motion model. Two known and unused options:

- The wide camera does **720p60 if MJPEG is forced** -- 2x the frame rate,
  already established, currently unused.
- L is 66 ms of which inference is ~15 ms. Anything that cuts the pipeline
  raises the gain ceiling (`1/(8L)`) at the same time.

You cannot interpolate what you never sampled. Past a point the honest answer
is to sample more often, not to predict harder.

---

## 6. SEQUENCE

Each step verifiable before the next depends on it.

| # | step | done when |
|---|---|---|
| 1 | firmware: `clk`, ticks, **and the gyro stream (§5.0)** | one flash, snapshotted |
| 1b | measure the achievable stream rate and DLPF bandwidth | >= 200 Hz sustained, noise floor moves |
| 2 | host clock model + skew fit | residual < 2 ms, ppm plausible |
| 3 | laser fiducial validation | predicted frame == observed frame |
| 4 | extend the recorder (§2) incl. rejects and predictions | a run replays end to end |
| 5 | analysis report (§4), run on existing logs first | lead error has a baseline |
| 6 | sampler + frozen test set (§3) | first 150 crops ready to label |
| 7 | annotate, retrain, re-measure **on the frozen set only** | a number that can be trusted |
| 8 | §5 changes, in the order §4 says | lead error falls |

Step 5 before step 6 on purpose: the analysis runs on logs already on disk,
so there is a baseline before any new data is collected.

---

## 7. OPEN DECISIONS

- **One firmware flash** for the clock (§1.3). Unavoidable; confirm it is
  acceptable. Everything after is host-side.
- **Annotation budget per run.** How many frames are you willing to hand-label
  after each test? 150 at ~2 s each is about five minutes. Purely a question
  about your time; it sets the stratum sizes in §3.4, nothing else.
- **Frame retention.** DISK SPACE FOR THE LOGS -- unrelated to model input
  resolution, which stays at 1280 (live paired frames beat 640 in every size
  bucket; that is settled). A 3-minute run at 30 fps across both cameras is
  ~10,800 frames, ~1.6 GB at full JPEG quality, so ten runs is ~16 GB.
  Proposal: full quality only for frames that will actually be looked at
  (annotation samples and locked frames), downscaled proxies for the rest so
  the timeline still replays. Nothing about the model changes.
- Still open from the previous session and unchanged: **the face-interlock
  rotation sign has never passed** (`STATUS.md` item 1). None of the above
  requires the beam to be armed at a person. Keep it that way.
