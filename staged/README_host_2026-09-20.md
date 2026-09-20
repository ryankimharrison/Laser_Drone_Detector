# Staged host + firmware changes — NOT APPLIED

`turret_host/`, `firmware/current/` and `tools/` are untouched. Every file
here is a full replacement for its counterpart; `staged/firmware/` is firmware.

    staged/link.py              -> turret_host/link.py
    staged/step_integrity.py    -> turret_host/step_integrity.py
    staged/app.py               -> turret_host/app.py
    staged/homing.py            -> turret_host/homing.py
    staged/firmware/stepper.py  -> firmware/current/stepper.py   (needs a deploy)
    staged/bench_pitch_delivery.py -> tools/bench_pitch_delivery.py  (new file)

Patch form, against the pre-change tree:

    patch --binary -p0 --dry-run < staged/host_2026-09-20_fix1-fix2.patch
    patch --binary -p0 --dry-run < staged/firmware_2026-09-20_vel-trip-latch.patch

**`--binary` is required** — the host files are CRLF and GNU patch otherwise
refuses with "different line endings". `staged/firmware/stepper.py` is LF, like
the rest of the firmware.

All five Python files pass `ast.parse`. The behavioural tests described below
were run against fake links; **nothing in here has touched the hardware.**

---

## Round 1 (earlier today, unchanged)

### (a) `step_integrity.verify()` no longer swallows the velmode-off failure
Was `except Exception: pass`. Now records the reason, calls `link.stop()` and
retries once, and if that fails skips the rewind and returns it through the
existing `error` channel.

### (b) `link.command()` says *which* failure the timeout was
`_transact` stashes what it collected in `_last_partial`; "nothing came back"
and "a reply came back but no prompt" are now reported differently.

### (c) `"move": (60.0, 5.0)` added to `_BLOCKING_TIMEOUTS`
There was no `move` entry, so it inherited a 1.0 s settle.

---

## Round 2 — FIX 1: the vel-mode / watchdog-latch shutdown trap

Three runs today (142026, 142809, 142953) ended with the board still in
velocity mode and the watchdog latch set; the next start was refused at
`homing._confirm_idle`. Four separate defects, and each one alone reproduces it.

### 1. `VelocityLoop.stop()` never cleared the trip latch — *firmware, root cause*

`staged/firmware/stepper.py`. `_tripped` is cleared by `start()` and by every
accepted `vel`, and **not** by `stop()`. The host stops sending `vel` at
shutdown, so the 400 ms watchdog fires on *every single run by design* — and
the flag then survived `velmode off` and every subsequent `state` until the
board was power-cycled. That is the whole of the "latched" behaviour, and it is
exactly why `velmode on` / `velmode off` clears it by hand: `start()` resets it.

`stop()` now clears it. The cumulative `trips` counter is deliberately **not**
touched — that is the real diagnostic record; `_tripped` is a per-session flag
and now has a per-session lifetime.

### 2. `homing._confirm_idle` clears the latch itself — *host, asked for*

`staged/homing.py`. New `_clear_vel_latch()` sends `velmode on` / `velmode off`
(the operator's own workaround), re-reads `state`, and only raises if the latch
*still* will not clear — with a message that says both were sent. Rationale in
the docstring: homing establishes position from **measurement**, so a stale
step counter is what it exists to replace. Worth keeping even once the firmware
fix is deployed, because a board on old firmware still hits it.

### 3. Shutdown leaves velocity mode *before* the step check — *host*

`staged/app.py`, new step **2a** and new `_leave_velocity_mode()`.

The comment on step 2b claimed `_safe_state()` had left velocity mode. **It had
not, and never did**: `_safe_state()` → `link.stop()` zeroes the *rates*;
leaving the *mode* is owed by `link.close()`, which is step 4. So the step check
has always run against a board in velocity mode and papered over it with a
`velmode off` of its own. It now happens in the shutdown sequence, which is the
only place that knows the order, and the result is read back from `state` and
logged.

### 4. `verify()` refuses to `move` unless the board *says* it is idle — *host*

`staged/step_integrity.py`, new `_vel_axes()`. `velmode off` returning a prompt
says the command was accepted, not that both axes left velocity mode — and a
`move` sharing the STEP pad with the velocity state machine loses steps
silently, which is the exact fault this module exists to detect. It now costs
one `state` round trip and refuses on `mode == "vel"`, on a running velocity
timer, or on a `state` it could not read.

### 5. Why the CLI appeared to go silent — *SOLVED, and it was not mine*

**The Implementer found it, and my first theory was wrong.** Corrected here so
the file does not preserve it: I argued the board had sent *zero bytes*. It had
not. It sent the echo, immediately, every time.

`_read_reply` ends a reply on a quiet gap longer than `settle` **once any bytes
have arrived** (`elif buf and ...`). A motion command echoes at once and then
prints *nothing at all* while the platform physically moves. So every failure
was simply a move that took longer than its own settle window:

| command | motion | settle it had | result |
|---|---|---|---|
| startup `dmove` 48.26 deg at 6 deg/s | 8.0 s | 5.0 s | cut off — run_144347 startup failed |
| step-check rewind `move pan -1666 800` | 1.07 s | 1.0 s (the default) | cut off |

Their fix, already applied to `turret_host/link.py`: `level` / `aim` / `dmove` /
`move` / `yaw` / `home` now use `settle = timeout`, i.e. for anything that moves,
wait for the prompt. **My `"move": (60.0, 5.0)` is dropped** — it would have
turned a 1.07 s failure into an 8 s one and fixed nothing. `staged/link.py` is
rebased onto their version.

Reproduced both ways against a fake port, so the regression is pinned:

| case | settle | prompt seen | elapsed |
|---|---|---|---|
| 8 s move, old 5 s settle | 5.0 | **no** (the bug) | 5.03 s |
| 8 s move, settle=timeout | 60.0 | yes | 8.00 s |
| 1.07 s rewind, old 1.0 s default | 1.0 | **no** (the bug) | 1.03 s |
| 1.07 s rewind, settle=timeout | 60.0 | yes | 1.08 s |

### What is still worth keeping from my side: the first-byte deadline

It addresses a **different** silence, and one their fix does not reach: *not one
byte, ever*. The settle gap has never bounded that case — it needs `buf` to be
non-empty — so an unanswering board always ran the full timeout, and
`settle = timeout` leaves that exactly as it was. A live console echoes before
it does any work (this module already depends on that: `_transact` strips the
echo off every reply), so "no byte in 2 s" is a console that is not reading its
input, not slow work.

| case | before | after |
|---|---|---|
| board never answers, `move` 60 s / settle 60 s | 60.0 s | **2.02 s** |
| board never answers, `level` 240 s / settle 240 s | 240.0 s | **2.02 s** |
| everything in the table above | unchanged | unchanged |

Round 1 (b) is also worth keeping and is carried through: `command()` now
distinguishes "nothing arrived" from "bytes arrived, no prompt", and the second
message explicitly suggests checking the command's settle against how long the
platform actually moves — which is the message that would have named this bug
on the first occurrence instead of the third.

### One accepted trade-off in the settle=timeout fix, for the record

A console that prints something and then dies **without** a prompt used to be
ended by the settle gap; with `settle = timeout` it now runs to the full
timeout — 240 s for `level`, on whichever thread issued it. The first-byte
deadline does not cover it (bytes did arrive). It needs a console that dies
mid-reply, which has not been observed, and waiting for the prompt is the right
default for a command that moves. Flagged, not argued.

---

## FIX 2: the attitude envelope is measured from vertical

`staged/app.py`. Rebased at 14:41 onto the Implementer's freeze
(`ATTITUDE_MAX_DEG` 60 -> 85 plus the reverse-on-trip branch). Their reverse
logic and `_cmd_hist` are carried through untouched; `ATTITUDE_MAX_DEG` is
still theirs and is **not** changed here. What changes is only *what the angle
is measured from*.

### This is no longer a tidy-up. At 85 it is the thing that keeps the trip
### on the right side of the mechanical stop.

The envelope was the angle from whatever attitude homing left behind. The
operator's own note gives the reason for raising it: *"the skip-level datum can
be several deg off"*. But while the angle is measured **from that datum**,
being several degrees off is precisely what spends the margin — and raising 60
to 85 spends nearly all of what is left:

| | datum at vertical | datum 8 deg off (run_142953) |
|---|---|---|
| old rule, 85 from the datum | trips at 85 deg from vertical | trips at up to **93 deg** from vertical |
| mechanical stop | 90 | 90 |

So at 60 the datum-relative rule tripped *too early* on one side — annoying,
and it is what ended run_142953 at 53.9 deg with 36 deg to spare. At 85 the
same rule trips *too late* on that side, and too late is the one that hits
metal. Measuring from vertical makes 85 mean 5 deg of margin wherever the datum
happens to sit.

### What changed

* new `tilt_from_vertical()` is what the envelope checks; `tilt_from_datum()`
  stays and is still reported, because "how far has it moved since homing" is
  the right question for the logs.
* the trip message now carries both numbers, how far off vertical the datum
  itself was, and how far short of the 90 deg stop the trip happened.
* **at READY** the guard logs how much envelope is actually left, and warns
  when the datum offset alone exceeds `90 - ATTITUDE_MAX_DEG` — the entire
  remaining margin to the stop, which at 85 is five degrees.

Reconstructing run_142953 with a datum at +6.7 deg (`_gravity_unit` /
`_angle_between`, roll 0), against the **old 60 deg** limit it ran under:

| payload pitch | from datum | from vertical | old | new |
|---|---|---|---|---|
| −40.0 | 46.70 | 40.00 | ok | ok |
| −50.0 | 56.70 | 50.00 | ok | ok |
| **−53.9** | **60.60** | **53.90** | **TRIP** | ok, 6.1 deg left |
| −60.0 | 66.70 | 60.00 | TRIP | at the limit |

The run ended 36 deg short of the mechanical stop on a limit that was never the
stated one. (The brief quotes 64.8 deg at the trip against my 60.6 — the real
datum and final sample carried roll that this reconstruction sets to zero.
Treat 60.6 as a model, not as a re-measurement.)

Also worth knowing:

* **It repairs the dead-reckoning cross-check for free.** That check argues
  gravity's total angle "bounds |true pitch| from BELOW" — true of tilt from
  vertical, *not* true of tilt from an arbitrary datum. It has been reasoning
  from the wrong reference on every skip-level run, which is the same run where
  the brief reports the pose being rejected repeatedly.
* The 2.5 deg roll floor adds 0.06 deg at 40 deg of pitch. Negligible.
* It is a safety-adjacent behaviour change either way, and it is Ryan's call.

## FIX 3: the bench is written, not run

`staged/bench_pitch_delivery.py`, and `--dry-run` works (no port opened):

    bench: 150 steps/s on BOTH motors = 7.31 deg/s of payload pitch;
           2.0 s burst = 14.6 deg per leg

Six legs: 0 / −20 / −40 deg, each driven **down and up**, one way only —
an out-and-back cancels exactly the asymmetry being looked for and hides lash
inside the reversal. All six end inside ±55 deg.

**Two deviations from the brief, both deliberate:**

1. **Pure pitch, not `vel` on tilt alone.** On the differential a single-motor
   `vel` is half pitch and half yaw; gravity loads the payload pitch axis, and
   pure pitch is `rate_a == rate_b`. It is also what the control law commands.
2. **The accelerometer, not a 500 Hz gyro stream.** *There is no firmware
   command that streams gyro samples.* `imu burst` measures how fast the board
   **can** read the part and reports an implied ODR; it does not emit samples.
   `imu watch` exits on its own CRLF. Polling `imu fast` costs ~0.21 s/sample,
   so a 2 s burst yields ~10 points — enough for a coarse within-burst profile,
   not enough to integrate.

   This is not a compromise for the ratio being asked for: delivered angle is
   the difference of two **absolute** tilt readings, which does not accumulate
   gyro bias, and a median of 15 samples against 0.76 deg single-sample noise
   gives ~19:1 on a 14.6 deg traverse. If the within-burst profile is wanted
   properly — separating lash take-up from steady-state slip — that needs a new
   firmware command to stream the gyro. Say so and it can be written.

Reading it, per the Implementer's framing (down and up at the same angle):

* `down ≈ up` → **scale**, not load. Symmetric under-delivery, and invisible to
  an out-and-back test.
* `down >> up` → **torque-limited climbing**, i.e. real gravity load.
* both ≈ 1.0 on the bench but 0.6 in flight → the **loop**, not the mechanism.

Preflight refuses to bench at all until it has excluded the three known faults
that would each produce a confident wrong answer: the dead tilt axis in
velocity mode (would read as 0% delivery), a `pads` MISMATCH, and the ADXL345
standby signature. It also runs a single-motor `vel` check on each motor first.

**Needs COM5 and an operator to place the payload at each angle. Not run.**

---

## Not mine, but found while looking

The Implementer applied `FEEDFORWARD_GAIN` 1.0 -> 0.0,
`MAX_ERROR_TO_FIRE_PX` 25 -> 60 and `SETTLED_RATE_DEG_S` 5 -> 40 directly to
`turret_host/config.py` for the demo take, so the detective session's request
that I make the feedforward change was already satisfied before it arrived.
Nothing staged here touches `config.py`.
