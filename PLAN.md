# Laser Turret — status and plan

**Showcase: 22 September 2026.** Written 2026-09-17.

Where this disagrees with `AGENT_HANDOFF.md` or `DETECTION_BRIEF.md`, this file
wins — several of their claims were measured and disproved on this machine.

---

## DONE

### Hardware, verified against the actual board
- **PCB audited against copper.** `tools/verify_against_kicad.py`, **44/44 checks**,
  walks pad→net→pad and never trusts a net name. Wired as a deploy gate.
- Both documented traps confirmed in the copper: **Finding #1** (tilt STEP/DIR net
  labels are swapped — the net called "Step" lands on the driver's DIR pin) and
  **Finding #7** (Q1/Q2/Q4 are D,G,S but a TO-220AB N-MOSFET is G,D,S).
- Reworks confirmed by the builder: loom re-pinned, MOSFETs crossed, laser works.
  RST/SLP is on **5 V** not 3V3 — out of spec for an A4988, accepted knowingly.
- Supply: **12 V 2 A** (can go 24 V), onboard 5 V buck at 1.5 A.

### Firmware — was already flashed; now backed up and extended
- `firmware/backups/iter001_*` is the **pristine as-found capture** off the board.
  `tools/deploy.py` gates on the copper audit, snapshots, bumps `firmware/ITERATION`,
  uploads only changed files, then resets and confirms the console returns.
- **iter002** — yaw travel limit `None → ±225°` (it was genuinely unlimited);
  `MOSFET_PINOUT_FIXED → True`; `RST_SLP_TIED_TO` corrected to `"5V"`.
- **iter003** — added **`imu mag`**, which did not exist. Replies
  `MAG <mx> <my> <mz> <pitch> <yaw>` — field *and* pose in one round trip, so the
  platform cannot move between the two readings.
- **iter004** — `LASER_ENABLED → True` for the static wall test. **Still armed.**

### Platform — measured, not assumed
- **IMU calibrated.** Gyro X bias 87.8 LSB = 6.11 °/s, drift now 0.01 °/s.
  (Handoff recorded 85.8 / 5.97 — reproduced within 2.3 %.)
- **Datum set at TRUE LEVEL** via gravity, repeatable across power cycles.
  Was 9.28° off. Motion confirmed working.
- Levelling measured **1.171°** of tilt per degree commanded (expected ~1.0).
  Probably backlash: 0.57° of lash against a 2.0° probe is 28 %. Re-measure with a
  ~10° probe if `DIFFERENTIAL_N` is ever in doubt.

### Cameras — several handoff claims disproved
- **Wide camera identified**: SVPRO USBFHD01M, **OV2710 + 2.1 mm M12**.
  Derived from real sensor geometry: **f = 700 px**, H FOV **107.9°**, ~12.2 px/deg
  at the axis. Handoff's "f ≈ 730, ~105°" were estimates.
- **720p60 is real.** The handoff calls 720p "a trap" delivering 9 fps. That 9 fps is
  the **YUYV** mode; forcing **MJPEG** gives a measured **59 fps**. It is a *centre
  crop* (~108° → ~85°), same angular resolution, double the rate.
- **USB rule corrected.** Not "separate native ports" — **one camera per xHCI root
  port**. Hubs are fine; chained hubs are fine. Proven: both cameras currently behind
  hubs (the C270 two deep) streaming concurrently at full rate. Lowering resolution
  does **not** rescue a shared root port (tested at 800×600 and 640×480 MJPEG).
  The handoff blames a transaction translator; TTs only bridge full/low-speed
  devices, so a "multi-TT" hub would not fix anything.
- **MSMF vs DSHOW, both measured**: MSMF delivers full frame rate but **silently
  ignores exposure writes** (`set()` returns True, value never changes); DSHOW
  honours exposure but collapses the C270 to 10 fps. Lock exposure through a short
  DSHOW open, release, reopen on MSMF.
- **C270 rotation resolved**: **90° clockwise**, confirmed from scene geometry
  (floor on the right edge → bottom after rotation), not just eyeball. The handoff
  left this sign explicitly open.
- First frame after open is **pure black** — needs ~30 frames of warmup.

### Host stack — built, reviewed twice, running
`turret_host/`, 12 modules. Verified: **30.1 Hz control loop**, both cameras 30 fps,
14.7 ms inference, clean shutdown, exit 0. `compileall` + `import` clean, pyflakes
clean.

**Nine defects found by adversarial review and closed**, each with a reproduction:

| defect | evidence closed |
|---|---|
| Stalled pipeline latched the beam ON (found by 2 lenses) | beam off 66 ms after a wedge |
| Face clearance measured from an *uncalibrated* goal pixel | `goal_calibrated` now an interlock condition |
| Camera + serial death detected then ignored | beam off 31 ms after link loss |
| E-STOP zeros overwritten by an in-flight `vel` | **12/60 → 0/60** beam-on-at-end |
| `shutdown()` blocked the Tk thread 240 s | closes in **2.01 s** |
| **Control law sign-inverted** — would have been runaway | P term −120 where −120 required |
| **`DEFAULT_AXIS_MIX` transposed** — pitch travel limit would not engage | matches firmware convention |
| **YuNet fed a 90°-rotated frame** | **0 faces → 1 face** (conf 0.88) |
| Face range halved by a `max_side=640` downscale | 1.40 m → **2.80 m** |

The YuNet one was a **spec bug of mine**: BUILD_SPEC said "never rotate frames for
processing." Right for the drone detector, wrong for an upright-face detector. The
face interlock was structurally blind — YuNet returned `[]` every frame, which
`control.py` cannot distinguish from an empty room.

---

## NOT DONE

### 1. A detector that fires on the drone — THE critical path
Nothing detects a drone today. COCO has **no `drone` class**, so `TARGET_CLASSES`
can never match. A held quadcopter reads to COCO as `kite`, `bird`, or nothing.

**We have the drone and can shoot in a demo-like room. That is the unblocker.**

Order:
1. **YOLO-World zero-shot, 10 minutes.** `yolov8s-worldv2.pt`, `set_classes(["drone","quadcopter"])`.
   Whether it fires at all restructures the plan. Untested, and cheap to settle.
2. **Capture footage** (below). Only input nobody can substitute for.
3. **Pre-label with the best available model, correct, fine-tune YOLO11n.**

Pretrained drone checkpoints (`sapoepsilon/yolov11s-drone-detector`,
`TomSmail/drone-yolo-v1`) are **label bootstrappers and fine-tune starting points**,
not final models: every public drone dataset is small drones against sky, and this
target is ~132 px — 10 % of frame width — indoors, hand-occluded, through a camera
with no IR-cut filter. Note their metrics are not comparable: 74.1 % AP was measured
on held-out Anti-UAV, 91.3 % on its own easy split.

### 2. Capture protocol
Shoot **through the same MJPEG/MSMF path the live system uses** — not H.264 to disk,
or you train on H.264 artifacts and infer on MJPEG.

Vary deliberately (this matters more than frame count):
distance 2/3/4/5 m · **occlusion — different grips covering different parts** ·
orientation incl. nose-on · cluttered *and* plain background · people moving behind ·
motion blur (record while moving) · the demo room's lighting range.

**Capture negatives**: people holding nothing, and holding a phone, bottle, mug.
Without them the model learns "person ⇒ drone" and fails in a room of spectators.

Use **both cameras** — 49° vs 105°, different distortion, one without IR-cut.

Sample **1–2 Hz**, or better, keep a frame only when it differs enough from the last
kept one. **Split by clip, never by frame** — near-duplicates across the split make
validation measure memorisation, and it fails silently.

Do **not** build a labelling tool. Roboflow / CVAT / Label Studio already do
model-assisted pre-labelling and YOLO export.

### 3. Safety — blocks arming
- **YuNet against a real face at 60–80 cm.** Enforced: `on_arm()` refuses until
  `verify_face_interlock()` passes in-process. **Turn a light on** — the room is dark
  enough to make a failure ambiguous between "wrong rotation" and "too dark".
- **Fail-closed interlock redesign.** Current logic is face-veto-only = fail-open: a
  dropped frame or a blind detector permits the beam. Change to **drone-box
  containment grants permission, head region vetoes**.
- **Heads, not faces.** Union `YOLO11n-pose` + COCO `person` + YuNet. Someone turned
  away is invisible to YuNet and can turn round in ~0.3 s. Inhibit on the **head
  region only** — a whole-person inhibit means the laser never fires, because the
  drone is in their hands.
- **Laser IR leakage.** Cheap 532 nm DPSS modules often omit the IR blocking filter
  and leak 1064/808 nm. Invisible ⇒ no blink reflex ⇒ no natural protection, and head
  detection does nothing about it. **Measurable today**: the IR-cut-removed C270 is an
  IR detector — compare it against the filtered HP webcam on the same beam.

### 4. Calibration — automate it, do not skip it
Skipping is not free: an uncalibrated Jacobian raises `NotCalibratedError` instead of
commanding motion, and `goal_calibrated` is an interlock condition, so **no beam**.

Both can run unattended inside startup homing, which already moves the platform:
- **Jacobian** — known motor move, measure the pixel shift with `cv2.phaseCorrelate`
  on the whole frame. Works on any textured scene: no tape, no clicking. ~10 s.
- **Goal pixel** — pulse the laser at 15 Hz (half the frame rate), difference
  consecutive frames so the static scene cancels, locate the dot. ~5 s. Single-point
  is enough: the residual across 2–5 m is ≤7 px against a 132 px target.

An analytic Jacobian is the wrong shortcut — four independent sign choices, and a
wrong sign is runaway.

### 5. Never run against hardware
- Camera identification, exposure lock, fps floor on the real devices
  (`_start_cameras` is the only startup path never executed).
- **Auto-homing end to end** — the full ~40 s sequence including the magnetometer
  yaw fit. Gravity cannot see yaw, so the motors' own field is the only absolute yaw
  datum: coarse trend ~−4.96 LSB/deg on `my`, plus a rotor-period ripple (~3.07°)
  that doubles as a lost-step detector.
- **Thermal throttle test** — 5 minutes flat out watching `nvidia-smi`. It is a
  laptop; a detector that holds 30 fps then decays looks exactly like a software bug.

### 6. Droppable
Drone bounding-box width → apparent-size ranging (costs ≤7 px to skip; `DRONE_WIDTH_M`
is `None` and ranging is guarded off). Stereo calibration, camera intrinsics, and the
LED latency measurement — all already deferred.

---

## Machine — surveyed, and not what the docs assume

Both handoff and brief assume an **RTX 4070**. It is an **HP Victus laptop, RTX 5060
Laptop 8 GB, sm_120 (Blackwell)**, 15.6 GB RAM. So `cu124` is wrong — the venv has
**torch 2.11+cu128**, verified computing at 6.5 TFLOP/s. Being a laptop makes the
throttling check real rather than hypothetical.

**Pico is on COM5** — always resolve by VID/PID `2e8a:0005`, never by COM number.

---

## Laser state right now

| gate | state |
|---|---|
| firmware `LASER_ENABLED` | **True** (iter004) — the board accepts `laser on force` today |
| host `turret_host` `LASER_ENABLED` | **False** — the stack cannot fire |

Do not treat the firmware as a backstop. Disarm with iteration 5 once the wall test
is done.
