# turret_host build spec

Python 3.12, Windows, RTX 5060 (Blackwell sm_120). Venv at `.venv/` already has
`opencv-python 5.0`, `torch 2.11+cu128`, `ultralytics`, `pyserial`, `numpy`, `pillow`.

**Read `turret_host/config.py` and `turret_host/types.py` first. They are the contract.**
Import constants from `config`; import dataclasses and `Slot` from `types`. Do not
redefine either. If you need a new shared type, it goes in `types.py`.

Every module must import cleanly with no hardware attached — guard hardware access
behind constructors/`start()`, never at import time. Tests run on a machine with no
camera and no board.

## Goal

A 5 mW green laser stays on a drone **held in a person's hand**, 2–5 m away, in a busy
room. It must never point at a face. Hand speeds: ≤1 m/s, ≤3 m/s².

## Architecture

```
capture-narrow ─┐
capture-wide  ──┼─> inference ─> tracker ─> control ─> link ─> Pico `vel <A> <B>`
                │                   │          │
                └───────────────────┴──────────┴──> SystemStatus ─> GUI
```

Hand-offs are `Slot`, never queues — a queue delivers stale frames the filter treats as
current.

## Modules

| file | owns |
|---|---|
| `cameras.py` | camera identification, capture threads, exposure lock |
| `detector.py` | YOLO target detection + YuNet face detection |
| `tracker.py` | pixel Kalman, adaptive Q, occlusion stabiliser, state machine |
| `control.py` | image Jacobian, control law, safety interlock |
| `link.py` | serial writer thread to the Pico |
| `calibrate.py` | Jacobian and goal-pixel calibration routines |
| `homing.py` | **auto-home at startup** — gyro cal, pitch datum, yaw datum, lash preload |
| `gui.py` | Tkinter spectator display + control panel |
| `app.py` | thread wiring, lifecycle, shutdown |

## Auto-homing (`homing.py`) — runs automatically at every startup

The system must come up on a **repeatable datum**, not "wherever it was left". Sequence,
with progress reported to the GUI (it takes tens of seconds and must not look hung):

1. **Confirm the link** and that the platform is idle. Abort loudly if not.
2. **`imu cal`** — gyro zero-rate bias, payload must be still. Uncalibrated, the X axis
   fabricates ~6 °/s, which looks exactly like mechanical drift.
3. **`level`** — drives to true level off gravity and sets the pitch datum. Repeatable
   across power cycles because gravity does not move. **Pitch only.**
4. **Yaw datum from the motor field** (below). Gravity cannot see yaw, so this is the only
   absolute yaw reference the machine has.
5. **Lash preload** — approach the final datum from one consistent direction, overshooting
   by more than `BACKLASH_DEG` and coming back. There is no `preload` firmware command
   (the doc lists one; it was never built), so do it with `dmove`/`yaw`. Always approach
   the datum the same way, every time.
6. **`sethome`** and report the established datum.

### Yaw datum from the motors' magnetic field

The magnetometer cannot give a compass heading — it is inches from two steppers whose
field swamps the earth's. But that is the signal, not the problem: **the motors' permanent-
magnet field is fixed in the base frame and rotates in the IMU frame as the payload yaws.**

Measured previously: a coarse monotonic trend of about **−4.96 LSB/deg on `my`**, plus a
resolver-like ripple of ~4 LSB at a **3.07°** period — against 3.12° predicted from the
rotor (7.2° of motor shaft ÷ 2.3077 belt ratio), a 1.7 % match.

Use `imu mag`, which replies `MAG <mx> <my> <mz> <pitch> <yaw>` — field and pose in one
round trip, so the platform cannot move between the two readings.

- Sweep yaw slowly across a modest range (±10–15° is enough), sampling at rest at each step.
  Settle before sampling; do not sample while moving.
- Fit the coarse trend and take the datum from it. Sweep in **one direction only** for the
  fit, or backlash contaminates it — the same effect that makes the `level` sensitivity
  measurement read 1.17 instead of 1.0 on a 2° probe.
- Field magnitude is ~2000 LSB against a ±2048 full scale, so **check for saturation** and
  reduce gain rather than fitting clipped data.
- The ripple period is an independent measurement of where the mechanism really is.
  Comparing its phase against the commanded position is a **direct lost-step detector** —
  expose that as a health check the GUI can show. It caps at 75 Hz so it is useless as
  tracking feedback; it is a datum and a health signal, not a servo input.
- This is the *coarse* datum. If the fit is poor, say so and fall back to using the current
  pose as yaw zero — but report clearly that yaw is unhomed.

## Hard-won constraints — violating these reproduces bugs that already cost days

**Cameras**
- **Use `cv2.CAP_MSMF`.** `CAP_DSHOW` silently delivers 10 fps at 720p while reporting 30.
- **Force MJPEG** (`CAP_PROP_FOURCC` = `MJPG`). Uncompressed YUYV modes exist at the same
  resolutions and are 3–6× slower. MSMF *refuses* the FOURCC set and returns False — that
  is normal, it negotiates the compressed mode itself. Do not "fix" it by using DSHOW.
- **MSMF silently ignores exposure writes** (`set()` returns True, value never changes).
  DSHOW honours them. To lock exposure: open briefly on DSHOW, set exposure, release,
  reopen on MSMF. UVC controls persist on the device across handles. **Verify it took.**
- **Open cameras sequentially, never concurrently.** MSMF's source reader does not
  tolerate being raced; a concurrent second open never becomes ready.
- **Grab-loop pattern is mandatory**: `cap.grab(); t = perf_counter(); cap.retrieve()` in a
  tight thread, publishing to a Slot. `CAP_PROP_BUFFERSIZE` is not settable on Windows, so
  this is the only way to defeat the driver's hidden 2–5 frame buffer.
- **Always `release()`**, including on exception and Ctrl-C. A leaked handle costs the next
  run: `isOpened()` returns True and every `read()` fails.
- **Verify delivered fps at startup** over `FPS_PROBE_FRAMES` and refuse below
  `MIN_ACCEPTABLE_FPS`. Never trust `CAP_PROP_FPS`.
- Identify the **narrow camera by serial** (`C8258920`). The max-width fingerprint cannot
  distinguish it from the built-in HP webcam, which also caps at 1280. The **wide** camera
  has no serial — use VID:PID plus the 1920 fingerprint.
- Worst-case frame gap is ~50 ms against a 33 ms median. One late frame is not a lost track.

**Detector**
- The C270's **IR-cut filter is removed**. COCO weights were trained on IR-cut imagery, so
  expect degraded confidence. **Nothing may gate on colour.** Shape and motion still work.
- Faces come from **YuNet**, not a person detector — a person detector fires on the hand
  holding the drone and the laser would never fire.
- Run the target detector on the **full 1280×720 narrow frame** at `imgsz=1280`.
- **THE FACE DETECTOR IS THE ONE PLACE THAT MUST ROTATE THE FRAME.** YuNet is a
  single-shot **upright** face detector: its anchors and its training set are upright
  faces. The narrow C270 is mounted 90° over (`NARROW_ROTATION_DEG`), so on the raw
  buffer a real face in the room is a sideways face and YuNet's recall against it
  collapses — *measured: 0 of 1 faces found on the raw buffer, 1 of 1 (conf 0.88) on the
  same frame rotated upright.* A blind face detector returns an empty list, which
  `control.py` cannot distinguish from a room with no faces in it, so the interlock would
  be **structurally blind** and the demo's whole safety story would do nothing.
  Therefore: rotate the frame fed to YuNet so faces are upright, and **map the returned
  boxes back into raw-frame coordinates** so `nearest_face_px` and the GUI overlay are
  unchanged. Rotate a **downscaled** copy (`max_side`) — a face at the 60–80 cm demo
  range is ~320 px across a raw narrow frame, so half resolution still leaves ~160 px
  against YuNet's ~15 px floor, and the downscale more than pays for the rotation
  (measured 33 ms → 7.6 ms). This does **not** apply to the drone detector: YOLO is
  trained with rotation augmentation and a drone has no canonical up, so there the
  geometry-carries-the-rotation rule stands.

**Control**
- **Feedforward is the whole game.** With ~60 ms latency the proportional term is only
  stable to ~2 Hz and cannot chase a hand. The velocity term commands the turret at the
  target's angular rate continuously.
- **The control law, with its signs, which this spec previously left undefined:**
  `omega = -J_inv @ (K*e + predicted_pixel_velocity)`, where `e = target_px - goal_px`
  and `J = d(image position of a world-fixed point)/d(motor step)` — which is how
  `calibrate.py` measures it (point at a **static** scene, step a motor, see where the
  feature went). The camera rides the payload, so a static feature moves *opposite* to the
  aim direction and that inversion is already inside the measured `J`; hence the leading
  minus. **Omitting it makes both terms positive feedback** — the P term drives the target
  away from the goal, the feedforward doubles the crossing rate — and the loop saturates
  and slews into a travel limit on the first frame it acquires. The minus belongs in the
  law, not in the stored matrix: `calibrate_jacobian()` re-derives `J` from measurement
  every run and would overwrite a hand-negated file.
- Send `vel` **every frame**, right after the detection lands. The Pico holds the rate
  between frames — that is what keeps the beam moving smoothly at 30 fps.
- `vel` takes **signed motor rates**, not payload angles. Use `vel`, not `pvel`: the
  Jacobian already maps pixel error to motor rates, and going through payload angles just
  adds a conversion that can carry a sign error.
- **Never poll `state` while servoing.** Two requests contend for one serial link; feedback
  latency roughly doubles and the watchdog parks the motors in the gap. One request, one
  answer.
- Firmware watchdog is **400 ms**. Command period must stay well under it.
- If the turret runs away from the target, a **column of `J` has the wrong sign**. Fix it
  in `J`, never in the wiring.
- Send `vel 0 0` on any transition out of TRACK.

**Safety — not tuning knobs**
- Laser on **only** when: TRACK confirmed, no face within the inhibit margin, platform
  settled, `|e| < MAX_ERROR_TO_FIRE_PX`, a `vel` went out in the last 100 ms, **and the
  goal pixel is calibrated** (see below). Any one failing → laser off **before anything
  else happens**.
- **The goal pixel must be CALIBRATED before the laser may fire.** Face clearance is
  measured from the goal pixel, i.e. from where the beam is believed to land. With no
  `goal_pixel.json` the goal falls back to frame centre, which is where the *camera*
  points; the laser's boresight offset from it has never been measured. At
  `NARROW_F_PX = 1400` one degree of offset is 24 px against a 120 px margin, and the
  sign is unknown, so it can **eat** the margin — the interlock would report 130 px of
  clearance with the beam 30 px from a face. `MAX_ERROR_TO_FIRE_PX` does not bound this:
  it certifies the target is near the goal pixel, not that the goal pixel is near the
  beam. Frame centre stays a legitimate fallback for **aiming**; it is never a fallback
  that may emit light.
- **"No data" is never "all clear."** A stale face report, a stalled frame pipeline, a
  dead camera thread and a dead serial writer are all interlock **failures**, not neutral
  states: laser off, leave the servo path, and say so on the panel. The interlock's own
  `vel_fresh` and duty-cap checks live *inside* `evaluate()`, so a loop that stops calling
  `evaluate()` has stopped enforcing them — the control loop must therefore evaluate
  liveness on **every** pass, including the passes where no new estimate arrived, not only
  when fresh data lands.
- `LASER_ENABLED` defaults False and the GUI arm control must be explicit and reversible.
- The laser goes **off immediately** on entering COAST — do not wait for the decay.

## GUI (`gui.py`)

Tkinter + PIL (both available; no new dependencies). Runs in the main thread, reads the
published `SystemStatus` and display frames from Slots. Must never touch an inference
buffer — draw on a `DISPLAY_SIZE` copy.

- Narrow view: bbox, predicted point, goal pixel, laser dot if seen.
- Wide view: all detections and every face box.
- **Large state banner**: `SEARCH` / `ACQUIRE` / `TRACK` / `INHIBITED — face` / `FIRING`.
  The inhibit banner is the demo moment; make it unmissable.
- Adaptive-Q level as a bar, so a jink is visible.
- Telemetry: per-camera fps, loop Hz, inference ms, serial RTT, pitch/yaw, pixel error.
- Controls: Start/Stop tracking, **Arm/Disarm laser** (disarmed on launch, and on every
  stop), **E-STOP** (sends `vel 0 0`, disarms, disables), and calibration triggers.
- Rotate the narrow view for display only — `NARROW_ROTATION_DEG`. For the **drone**
  detector, never rotate frames for processing; carry the rotation in the geometry instead.
  A full-frame copy per frame buys nothing. **But see the face-detector exception below —
  this rule was originally written without one, and that was a spec bug.**
- The GUI must stay responsive while tracking, and closing it must shut down cleanly:
  stop threads, `vel 0 0`, release cameras.

## Laser arming — amended 2026-09-16 after the integration review

These are behaviours the code now enforces. They are stricter than what this spec
originally described, deliberately. See `STATUS.md` for the evidence behind each.

- **A safety action latches the beam off until a human re-arms.** `link.safety_veto(latch=True)`
  — used by E-STOP, operator STOP, disarm, a dead worker thread, and shutdown — makes
  `set_laser(True)` refuse outright, independent of the safety epoch. Only `on_arm()` clears
  it. The epoch alone was not enough: a frame that read the counter *after* the safety action
  saw a consistent world and was allowed through, measured at 60/60 trials leaving the board
  with the beam on. A generation counter cannot fix that class; a latch can, because there is
  no value a racing frame can read that makes `laser on` acceptable again.
- **The control loop reads the safety epoch as the FIRST thing in a pass**, ahead of the
  tracking gate and ahead of `estimate_for_control()` (which takes the tracker lock and is
  therefore a preemption point). Anything that reads an input before reading the epoch
  reopens the race above.
- **`on_arm()` refuses until the face interlock has been proven against a real face.**
  `verify_face_interlock()` must pass in the current process. Skipped only for a simulated
  link, where nothing can emit light. This is the enforcement of pre-arm checklist item 1.
- **A refused `laser on` is abandoned, not retried.** The firmware answers a refusal by
  safe-stating every axis, so an unlatched retry loop is 55 safe_state() calls a second with
  the vel stream re-enabling the motors in between. Three attempts, then sticky refusal until
  ARM. An `off` is never abandoned.
- **The laser transact is bounded at 80 ms and sits in front of the `vel` stream**, so one
  writer pass is at most 80 + 250 = 330 ms against the firmware's 400 ms velocity watchdog.
  A 1.0 s laser timeout there made the command period exceed the watchdog outright.

### Face-detector exception — amended

- The narrow face pass **rotates its input** (`NARROW_ROTATION_DEG`) because YuNet is an
  upright-face detector and the C270 is mounted a quarter turn over. Fed the raw buffer it is
  structurally blind, and a blind face detector returns `[]`, which `control.py` cannot
  distinguish from an empty room.
- It runs at **native resolution**. A 640 downscale was tried and reverted: measured, it
  halved the range at which a face is detected at all (2.80 m → 1.40 m), which is a bystander
  standing behind the drone holder going unseen. Native costs 19.4 ms against a 33 ms frame
  period on a dedicated thread — affordable.
- **`config.py` lines 98–99 still say "never rotate frames on the processing path".** That is
  false for the face path and `config.py` is a fixed contract file, so this paragraph is the
  carve-out. Do not "fix" the code to match that comment.

## Style

Match the existing firmware's voice: comments explain *why*, especially where a choice
looks wrong but is deliberate. No defensive try/except that swallows errors. Prefer
explicit failure at startup over silent degradation at runtime.
