# Staged firmware iteration 15 — NOT FLASHED

Prepared 2026-09-20 by the Detective/intern session at the lead's request. **Nothing has been applied to `firmware/current/`.** Flash only when the lead says the port is ours.

| file | what it is |
|---|---|
| `iter015_vel-pad-and-clamp.patch` | unified diff against `firmware/current/` (10 hunks, 2 files) |
| `stepper.py`, `cli.py` | the full post-patch files, for eyeballing or straight copy |
| `../../tools/vel_pad_check.py` | the bench script that discriminates the pad hypothesis |

Apply with, from the project root:

```bash
patch -p0 --dry-run < firmware/staged/iter015_vel-pad-and-clamp.patch
```

Both files pass `ast.parse`. The patch applies cleanly at `-p0` from the project root.

---

## What is in it

### 1. The pad fix — the candidate root cause of "counter advances, shaft does not move"

**`stepper.py:373-391`** — `enter_vel_mode()` returned early when already in velocity mode, so it never re-claimed the STEP pad. It now re-claims unconditionally (one register write, harmless when already claimed).

**`stepper.py:905-920`** — `coordinated_move()` had **no velocity-mode guard**, unlike `move()` (`stepper.py:569`) and `single_step()` (`stepper.py:731`). Added. Its `finally` clause calls `_release_step_pin()` on **both** axes (`stepper.py:997`), handing the pad to SIO; if the axis was still in velocity mode, nothing ever took it back.

Together those explain the measured signature exactly: `vel_tick` keeps writing periods and keeps integrating `_pos_micro`, no pulse reaches the driver, per-axis, until reboot.

> **This is a mechanism I could not close from source alone.** `coordinated_move` moves *both* axes, so on its own it predicts both motors dying, and the bench saw only tilt. Something upstream must have put only one axis into that state. **`pads` (below) settles it in one command** — and the fix is written to be correct whichever path got the pad into SIO.

### 2. The clamp no longer steers

**`stepper.py:1297-1327`** — the pitch clamp kept `(ta - tb) // 2` and called it "the yaw component". On a nearly-pure-pitch demand that residual is noise: in `run_2026-09-20_132627` a `(-319, -514)` demand was executed as `(+97, -97)`, a 4.75 °/s yaw slew that moved the drone's box **270 px away from the goal** and took the pixel error from 145 to 360. It now zeroes both targets. Coming back off the stop still works — the next command whose push has the opposite sign fails the sign test and passes through untouched.

### 3. The clamp stops being invisible

**`cli.py:1189-1196`** — `limited` added to the STATE `velocity` dict. The property has existed since the clamp was written but was never serialised, so a host watching STATE could not tell "the firmware is refusing my pitch command" from "a motor is dead".

**`stepper.py:1110-1129`** — `_limited` cleared in `VelocityLoop.start()` so it means "during this run" rather than "at some point since boot".

### 4. The clamp frame is centred on the pose tracking started from

**`stepper.py:1094-1097, 1110-1129, 1297`** — `_arm_pitch_limit`'s docstring claims the frame is centred on the placed pose, but `_tick` compared the raw sum, and `vel_position` is `position + _pos_micro // 1e6` — `position` being the position-mode counter left over from homing, which `enter_vel_mode` does not reset. `start()` now records `_tot0` and `_tick` tests `tot - _tot0`.

### 5. `pads` — the diagnostic that would have saved the evening

**`cli.py:339-341, 1189-1209`, `stepper.py:821-834`** — a read-only command printing who owns each STEP pad (SIO / PIO0 / PIO1) against what the axis's mode requires. An axis in `vel` whose pad reads SIO prints `MISMATCH`. That is the fault, visible directly instead of inferred from a shaft that did not turn.

---

## What is deliberately NOT in it

- **`self.mode` is never set to `"move"`.** `stepper.py:183` initialises it, 393 sets `"vel"`, 409 sets `"idle"` — nothing sets `"move"`. So `enter_vel_mode`'s `if self.mode == "move": raise "a move is in progress"` (`stepper.py:376`) is **dead code** and always has been. Fixing it means adding `mode = "move"` / `finally: mode = "idle"` to both move paths. Real, but it is not implicated in tonight's failure and it widens the blast radius of a demo-night flash. Left for iteration 16.
- **The ~1.59× scale error.** `AXIS_STEP_DEG` is `360/(200×16×DIFFERENTIAL_N)` to 0.000 %, i.e. derived, not measured, so a wrong `DIFFERENTIAL_N` or `NARROW_F_PX` is invisible to every internal check. The camera leg reproduces 62.8 %. Not a demo blocker — a closed loop absorbs a gain error — but it means the clamp fires at a true ≈57° rather than 90°. Needs a **one-way** measurement; every test run so far has been an out-and-back, which cannot see scale.

---

## Bench script

```bash
.venv\Scripts\python.exe tools\vel_pad_check.py --axis tilt
```

**Reboot the board first** — leg 1 is only a baseline if nothing has taken the pad yet.

Three legs, gravity read as a **vector** before and after each (differencing pitch and roll separately measures where yaw happened to be, which is the arithmetic that produced the misleading 63 %):

1. **BASELINE** — velocity leg on the axis under test, straight after reboot. Expect motion.
2. **DISTURB** — one position-mode move on that same axis.
3. **RETEST** — the identical velocity leg.

Old firmware: leg 3 ≪ leg 1. Fixed: leg 3 ≈ leg 1. It prints `pads` at four points, so on iteration 15 the mechanism is visible rather than inferred, and it notices if the fixed firmware *refuses* the disturbing move (which is correct behaviour, but would make leg 3 prove nothing — so it drops out of velocity mode and redoes the move).
