# iter017 — IMU soft travel limits (crash guard)

**Status: written and exercised offline. NOT RUN ON HARDWARE.**
Nothing below has seen the rig. Treat every claim as "logic tested, wiring
unverified" until the calibration run in §3 has been done.

## 1. What this is

The payload has been hitting the frame. Every limit the firmware had was
quoted from the step counter, and the step counter is exactly what stops
being true in a collision: an A4988 raises no fault when a stepper skips
against a hard stop, the PIO keeps emitting pulses, and `position` keeps
counting them. On 2026-09-18 the counter read 57 deg wrong while the payload
sat against its own yoke.

This adds a limit built on gravity instead.

### New file: `guard.py`

| quantity | sensor | what it is |
|---|---|---|
| `tilt` | accel | `acos(ĝ·ẑ)` — angle between measured gravity and payload-up |
| `tilt_rate` | gyro | `ω · n̂`, `n̂ = (ĝ × ẑ)/\|ĝ × ẑ\|` — the exact derivative of `tilt` |
| `a_dev` | accel | `\|\|a\| − at-rest \|a\|\|` — the only witness a collision has |

**Both are yaw-invariant and neither has a calibrated sign.** Yaw rotates the
IMU *about* ẑ, so `tilt` is invariant by construction — no de-rotation, no
mounting convention, nothing to get wrong. `tilt_rate` falls out of
`ġ = −ω × ĝ` and needs no yaw either. This matters: `level`'s docstring
records four failed runs and one collision caused by a signed sensitivity
that inverted past 90 deg of yaw.

**The gyro is what makes the limit arrive in time.** At 195 deg/s with
`VEL_ACCEL = 40000` the payload needs ~100 ms to stop — about 10 deg of
travel — on top of ~40 ms of sampling latency. A limit that waits for `tilt`
to cross the line is ~27 deg too late. The guard trips on
`tilt + tilt_rate × 0.14 s` instead.

The one thing learned rather than derived is which sign of motor push
increases tilt. That is measured **online**, every sample, from the commanded
push against the measured rate (`PushSense`) — so a reversed motor or a
flipped `INVERT_DIR` cannot leave it stale. With no sign learned yet a trip
blocks both directions.

## 2. Wiring

- `stepper.VelocityLoop` gains `_blk_pos` / `_blk_neg` / `_blk_all`, plain
  booleans read by the 200 Hz ISR. **The ISR never touches I2C** — a
  transaction allocates and can raise, and neither is allowed in an
  interrupt. Measuring happens in ordinary context and publishes via
  `apply_guard()`.
- `cli.do_vel` / `do_pvel` sample once per host command (~30 Hz, ~1 ms
  against a ~45 ms round trip) and publish **before** commanding, so a
  refused rate never reaches the motors even for one 5 ms tick.
- `cli.do_payload` passes a guard callback into `coordinated_move`, which
  polls it every 64 steps (~16 ms). Position moves previously had **no** IMU
  protection at all — including homing's sweep and lash preload.
- Response is the same as the existing travel clamp: **zero both motors, do
  not steer.** Keeping the "yaw component" of a refused pitch demand is what
  drove the drone's box 270 px the wrong way on run_132627.
- Trip behaviour: latched, **auto-clears on retreat**. Release needs the tilt
  back inside a 6 deg hysteresis band *and* the payload actually slowed
  (`rate < 10 deg/s`, signed). Without the rate term the guard buzzes instead
  of stopping — trip, halt, release, re-accelerate, at the sample rate.
- `STATE` gains a `guard` block and `velocity.guard_hit` / `guard_stops`.
  `emit_state` does **not** sample — it reports the last motion-path
  measurement with an `age_ms` beside it, so a stale set cannot be mistaken
  for a live one.

## 3. Calibrating it — DO THIS FIRST

Until this is done the guard runs on a coarse 75 deg default
(`LEVEL_MAX_TILT_DEG`, the same claim `level` already makes). That is not an
absent guard, but it is not the real stop either.

```
level                  (or level by hand — see the roll note in the memories)
slimit                 sanity-check: it should read a tilt near 0
```

Then, **by hand, slowly**, drive the payload to just short of the frame on
one side and take the current value:

```
dmove 40 0             ... step toward the stop until it is close
slimit set             captures THIS tilt as that side's limit
```

Repeat on the other side (`dmove -40 0`, `slimit set`), then:

```
slimit save            persist to /slimit.json so it survives a reboot
slimit                 confirm both sides read what you expect
```

`slimit set` refuses a capture past 88 deg (payload already through the
frame) or below 2 deg (that would forbid all motion), and proves the
accelerometer is live first — a standby ADXL345 reads exactly 0.00 tilt,
which is inside every tolerance.

Other commands: `slimit watch` (live tilt / rate / predicted overrun),
`slimit reset` (clear a latched impact, then re-measure), `slimit on|off`.

## 4. What to watch on the first hardware run

1. **`slimit watch` while you push the payload by hand.** `tilt` should track
   what you see; `rate` should go positive as you move toward a stop and
   negative coming back. If the rate sign is inverted the predictor brakes
   the wrong way — that is the single most important thing to confirm, and
   it is a one-minute check.
2. **The push sign.** `slimit` prints `push->tilt sign ±N (confidence ...)`.
   It should settle within a few seconds of real motion and not flip.
3. **False impacts.** `IMPACT_G = 1.2` is set against measured ramps of
   ~0.5 g, but that figure is from `level`'s settle checks, not from a full
   slew. If normal tracking latches impacts, raise it — do not disable the
   guard.
4. **`guard_stops` climbing during tracking.** That means the host keeps
   commanding into a stop. A gate cannot rate-limit; the host loop is the
   right place to fix that.

## 5. Homing changes (host side, `turret_host/`)

- **The magnetometer sweep is now mandatory.** `--fast-home` is refused by
  `homing.py` and ignored-with-a-warning by `app.py`.
- **The stored yaw reference is loaded by default.**
  `turret_host/calibration/yaw_reference.json` has held a reference since
  2026-09-19 and was never read — the only way in was `--yaw-reference
  <float>` on the command line, which `app.py` defaulted to `None`. Every run
  since swept the motor field for 20 s and threw the datum away. A run that
  *establishes* a reference now also saves it, so the next home comes up
  absolute. A good stored reference is never overwritten.
- **The datum is verified against gravity after `sethome`.** The existing
  check ran *before* `dzero`/`sethome` and answered "did the preload come
  back", not "is the datum we wrote level". The new one catches the 5–7 deg
  roll settle, and names roll as irreducible (`level` corrects with pitch
  alone) rather than suggesting a re-home that cannot help. The
  standby-accelerometer signature is rejected by name.
- **Soft-limit margins are recorded against the new datum** and logged, so
  the usable envelope is a measured number rather than an assumption.
- **Re-home at close** (`app.shutdown` step 2c, after the step check, before
  the port closes; `--no-home-on-close` to skip). Budget 90 s. It clears
  `_platform_cancel` first — shutdown sets that early to kill a running
  homing task, and `homing._cmd` checks it before every command, so leaving
  it set would abort on the first line while looking like a board fault.

## 6. Offline test evidence

Three harnesses in the session scratchpad, all passing:

- `test_guard.py` — the physics. Confirms `tilt` is identical at 8 yaws,
  `tilt_rate` matches truth to 1e-14 deg/s across 15 tilt/yaw/push
  combinations, a full-slew approach stops at 42 deg against a 62 deg frame
  (and goes to 800 deg with the guard off), the push sign is learned and
  re-learned when the motor reverses, impact needs 2 consecutive samples, a
  dead bus holds the previous block rather than opening the gate, and a
  corrupt stored limit is refused.
- `test_callsites.py` — the seam. Imports the **real** `cli.py` and
  `stepper.py` behind hardware stubs and drives `dispatch()`. Confirms the
  ISR honours the flags, a pure-yaw command is untouched by a pitch block,
  `vel` still answers `ok` first on the line (the host tests `"ok" not in
  reply`), `pvel` keeps its `ok <pitch> <yaw>` shape, `STATE` stays valid
  JSON, and the turret is still drivable with no IMU.
- `test_homing.py` — the sequence, against a fake board.

**Two real bugs were found by these and fixed:** `slimit off` left stale
block flags published to the ISR (the payload would stay blocked by a guard
the operator had just switched off — indistinguishable from a seized axis),
and the release condition used `abs(rate)`, which refused to clear during a
brisk retreat.

None of this substitutes for the hardware run in §4.

---

# 7. Home that comes back (added after the §1–6 work)

**Status: written and exercised offline. NOT RUN ON HARDWARE.**

## The measurement that prompted it

```
limits ->  pitch now -0.000,  yaw now +0.000,  home pitch -0.000
imu    ->  tilt now -3.94 pitch, -1.97 roll      (4.40 deg from vertical)
```

Both are true. `dzero` wrote "this is zero" into a counter at whatever pose
the turret happened to be in — the `--skip-level` path — and that zero lives
in RAM and dies at reset. **Pitch has never returned to anything.** Yaw does
return, off the motor field. Pitch had no equivalent, even though the
reference it needs has been bolted to the payload the whole time.

## Why `level` was not already that reference

`level` drives to **vertical**. Home is not vertical; it is 4.40 deg off it.
So the one absolute reference on the machine was aimed at the wrong target.

Worse, `level` cannot reach its own target. ~2 deg of that 4.40 is **roll**,
and the mechanism has pitch and yaw only — there is no roll axis. It
minimises total tilt with pitch alone against a floor it cannot remove,
converges toward something unreachable, and settles out. That is the
documented hunting failure and why `--skip-level` became standing practice.

Reproduced in `test_datum.py` §1 against the rig's real numbers: `level`
drives pitch to 0.000 and is still left 1.97 deg from vertical, having never
satisfied its own 0.3 deg tolerance.

## What was built

**New file `datum.py`** — home stored as a gravity **vector** plus the yaw it
was captured at, persisted to `/datum.json`.

- `err = angle between measured g_hat and stored home g_hat`
- Storing the vector means the irreducible roll is **part of the target**, so
  the target becomes reachable and the same loop converges instead of
  hunting. That is a fix, not a workaround.

**`_converge_to()` in cli.py** — the levelling loop, extracted and shared.
`level` passes payload +Z (vertical); `datum go` passes the stored vector.
Only the setpoint is a parameter, so the probed direction, the per-iteration
"did that help", the envelope, the impact check and the step clamp are shared
rather than reimplemented — a fix to one is a fix to both.

**New commands:** `datum`, `datum set`, `datum go [tol]`, `datum clear`,
`datum save|load`. Plus a `datum` block in `STATE`.

## The one constraint

The error metric is **not yaw-invariant** — yaw rotates the IMU about payload
+Z, so the same pitch reads a different vector at a different yaw. The yaw at
capture is stored, and `datum go` **refuses** outside a 2 deg window rather
than let a pitch loop chase a yaw error it cannot reduce. Homing's sequence
already ends at the yaw datum, so this costs nothing — but pitch cannot be
homed first.

## Two things fixed in passing

- **`level` had no crash protection.** It called `platform.move_by()`
  directly — the one motion path with no guard — and `level` is precisely the
  routine that drove a payload into its frame on 2026-09-18. Now routed
  through `_guarded_pitch()`.
- **Homing's free kinematics check has been dead since iter012.** Homing
  parses `measured N deg of tilt per deg` with `_SENS_RE`; the 2026-09-19
  axis-agnostic rewrite dropped that sentence, so the host has matched
  nothing and silently skipped the check ever since. The quantity still
  existed; only the line had gone. Revived.

## Host side — delivered as a PATCH, not an edit

`staged/homing_datum_return.patch`, against the live file (which carries the
implementer's `LinkError` import and both catches). It:

- calls `datum go` after the yaw datum and before `dzero`/`sethome`;
- verifies the datum against **stored home** (`DATUM_VERIFY_TOL_DEG = 1.0`)
  instead of vertical. Against vertical this rig fails **every** run, and a
  warning that always fires is one nobody reads;
- degrades on firmware without `datum`, falling back to the vertical check
  and saying pitch will not come back;
- adds `HomingResult.datum_error_deg`.

**The attitude envelope in app.py is untouched and stays on true vertical**,
per the implementer's note — an 8–20 deg datum with an 85 deg limit would put
the trip past the mechanical stop. The firmware envelope
(`LEVEL_MAX_TILT_DEG`) likewise still measures from vertical inside
`_converge_to`, so a datum 70 deg over buys no extra travel toward the frame.
Both are asserted in `test_datum.py` §6.

## Bench procedure

```
level                  (or level by hand)
datum set              capture HERE as home; saves to /datum.json
datum                  confirm: stored vector, tilt from vertical, yaw
```
Then to prove it returns:
```
dmove 15 0             knock it off home
datum go               it should come back within ~0.3 deg
```

## Offline evidence

- `test_datum.py` — 30 checks against a simulated rig with an **unactuatable
  roll floor**, driving the real `cli.py`. Reproduces the hunting, then shows
  `datum go` returning from +14, −11 and +30 deg. Covers the yaw refusal, the
  standby signature (rejected three ways, including when `accel_live()`
  wrongly passes it), the envelope staying on vertical, and that `level`'s
  output strings homing parses are unchanged.
- `test_homing_patched.py` — 21 checks running the **patched** homing against
  a fake board whose `command()` **raises LinkError** the way the real
  `TurretLink` does, rather than returning error text. That seam is what cost
  a live run today.

Two bugs the tests caught: default arguments binding `STORE_PATH` at
definition time (so the path could not be redirected), and one assertion of
mine that passed for the wrong reason.

**None of this has touched hardware.**
