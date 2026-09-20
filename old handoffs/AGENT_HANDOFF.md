# Laser Turret — agent handoff

You are building the tracking stack for a laser turret: the perception layer on
an RTX 4070 host and the velocity-mode extension to the existing Pico firmware.
This document is self-contained. Everything you need is either inline or in a
file listed in §0.3. You should not need to ask the human anything before
reaching milestone M5.

**The one requirement.** A 5 mW green laser stays smoothly on a drone held in a
person's hand, 2–5 m away, in a busy classroom, regardless of how the hand
moves. Every decision below serves that. If you find a conflict between this
document and that requirement, the requirement wins.

**Demo reality.** The drone is *held*, not flown. Someone's hand is on it,
partially covering it, and their face is 60–80 cm away. The system must fire at
the drone and never at the face. Speeds are hand speeds: ≤1 m/s, ≤3 m/s².

---

## 0. Orientation

### 0.1 What is done and verified

- Custom RP2040 PCB (rev **3/27/2026 prod v1**) designed, fabricated, brought
  up. Every pin in this document was derived from the PCB copper and is
  machine-checked (`turret_test/tests/verify_against_kicad.py`, 41 checks).
- Six hardware findings diagnosed and fixed. Condensed in §1.3. You do not
  need to rediscover them.
- MicroPython bring-up firmware on the Pico with a serial CLI, a browser GUI
  bridge, 100 host-side tests, an on-board no-motion test suite. Passes clean.
- Motors verified moving. Step generation measured at **0.1 % timing error
  from 200 to 20 000 steps/s**.
- Motion platform CAD complete; SLS nylon parts being printed.

### 0.2 What you are building

1. **Firmware:** a velocity mode for the existing MicroPython suite (§4). Not a
   rewrite. About 150 lines.
2. **Host:** capture → detect → filter → control → serial, plus a spectator
   display (§5).
3. **Calibration tooling** for the four empirical constants the loop needs (§6).

### 0.3 Files you should have

```
turret_test/                 the working firmware + tools. READ pinmap.py FIRST.
  pinmap.py                  hardware truth. Do not edit.
  config.py                  tunables. You will edit this.
  stepper.py                 PIO step generation, Axis, coordinated_move().
  kinematics.py              the differential transform + Platform. READ THIS.
  cli.py                     serial console. You will add commands here.
  diagnostics.py, endstop.py, peripherals.py, main.py
  gui_server.py, gui.html    browser control panel (holds the COM port when running)
  release_board.py           returns the board to the REPL so mpremote can deploy
  deploy.ps1 / TURRET.bat    deploy with pin-map audit + host tests as gates
  tests/                     host_test.py, smoke_test.py, verify_against_kicad.py
  hardware/turret.kicad_pcb  the ground-truth copper
  HARDWARE_NOTES.md          the six findings, in full
  BRINGUP.md                 what happened on first power-up
turret_vision/geom.py        reference maths (camera model, triangulation). Optional.
TRACKING_ARCHITECTURE_v2.md  the reasoning behind §3. Read if something here seems odd.
```

If `turret_test/` is missing, stop and ask for it. Nothing else is essential.

### 0.4 Machines and ports

- **Host:** Windows, RTX 4070. Cameras and Pico all on USB.
- **Pico:** enumerates as **VID 2e8a PID 0005** when running MicroPython. Find
  it by VID/PID, never by COM number — it moved from COM3 to COM4 on the dev
  machine after a reflash and will be something else on yours.
- Only one process may hold the port. `gui_server.py` holds it while running.
  `mpremote` needs the console released first: run `python release_board.py`.

---

## 1. Hardware — facts you can rely on

### 1.1 Pin map (from copper, verified)

Two stepper drivers on Pololu-16 carriers. **The firmware calls them `pan` and
`tilt`. Those names are wrong for this platform** — they are Motor A and
Motor B of a differential (§1.4). Neither is payload pan or tilt on its own.

| firmware name | this doc | driver ref | STEP | DIR | EN | MS1/2/3 | pad 2 (VDD) | coils | RST/SLP hdr |
|---|---|---|---|---|---|---|---|---|---|
| `pan` | **Motor A** | A3 | GPIO8 | GPIO7 | GPIO12 | GPIO11/10/9 | GPIO13 | J3 | J4 |
| `tilt` | **Motor B** | A2 | GPIO1 | GPIO0 | GPIO5 | GPIO4/3/2 | GPIO6 | J2 | J1 |

EN is **active low**. Pad 2 is **VDD** on these A4988 carriers and is driven
high by the GPIO as the chip's logic supply (§1.3, finding 2).

| function | GPIO | notes |
|---|---|---|
| Laser (Q1 gate) | 17 | HIGH = on. 100 kΩ gate pulldown → off when MCU dead |
| IR LEDs (Q2) | 14 | same structure |
| Fan (Q4) | 15 | same structure |
| Endstop 1 (J14 pin 2) | 16 | switch to GND, internal pull-up, active low |
| Endstop 2 (J13 pin 2) | 19 | same |
| I2C1 SDA / SCL (J11) | 26 / 27 | **no pull-ups on the board**; IMU module supplies them |
| Ultrasonic trig / echo (J5) | 20 / 21 | echo is 5 V — locked out in config; not used |
| Analog in (J6) | 28 / ADC2 | not used |
| Free | 18, 22 | |

Pico VSYS is **unconnected**: the board cannot power the Pico. USB only.

### 1.2 Drivers

A4988 carriers, sense resistor 0.068 Ω (`SENSE_RESISTOR`), 1/16 microstepping
(MS1/2/3 = 1,1,1). Current limit set by trimpot; Vref = I × 0.544. Target
~0.7–1.0 A/phase → **0.38–0.54 V**. RST and SLP are jumpered together and tied
to **3.3 V** (not 5 V — see finding 4).

### 1.3 The six findings, condensed

You will not hit these; they are fixed. You need them to avoid undoing them.

| # | what | state |
|---|---|---|
| 1 | Schematic's tilt STEP/DIR *labels* are swapped. Copper is right; `pinmap.py` follows copper. | handled; audit enforces it |
| 2 | Driver pad 2 is FAULT on DRV8825 but **VDD** on A4988. Firmware drives it HIGH as logic supply (`A4988_VDD_SOURCE="gpio"`). | handled |
| 3 | J5/J6/J13/J14 carry 5 V; RP2040 is not 5 V-tolerant. Only bare switches on endstops. | locked out in config |
| 4 | RST/SLP must be 3.3 V for A4988 (abs max is VDD+0.3). Was 5 V; reworked. | done |
| 5 | J2 and J3 have **mirrored** pin orders; the motor loom was interleaved. Loom re-pinned (BLK BLU GRN RED). | done |
| 6 | No local VMOT decoupling, no ground pour. Chopper hiss. Bodge caps recommended. | cosmetic |

Two consequences for you: **the two axes will turn opposite ways for the same
command sign** (finding 5 — handle with the sign constants in §4, not by
re-crimping), and **never plug or unplug a motor with VMOT live**.

### 1.4 Motion platform — the differential

A bevel-gear differential wrist. Two steppers off the moving assembly drive
input bevels through 26T->60T belts (**2.3077:1**), miters 1:1. Cameras and laser sit
on the output gear.

- Both motors turning the **same** way → carrier rotates → **pitch** (up/down)
- Both motors turning **opposite** ways → output gear spins → **yaw** (left/right)
- Any single motor moving alone → equal parts pitch and yaw

**Pitch is the outer joint.** The carrier pitches about the fixed axis; the
output gear yaws about an axis *carried by* the carrier, which is tilted 11°
off nominal in the CAD's rest pose. So the yaw axis moves with pitch.

**`kinematics.py` already implements all of it** — `Platform.move_by(pitch,
yaw)` drives both motors together, and `stepper.coordinated_move()` puts them
on one shared timeline so they start and finish on the same tick. Use it for
homing, calibration and any manual positioning.

**The tracking loop still does not need it.** The control loop uses an
empirically calibrated 2×2 matrix (§6.3) that maps pixel error directly to motor step
rates, and that matrix contains the belt ratio, the differential, the joint
order and the axis tilt. The analytic form is in Appendix A for understanding
and for the pitch deck.

| quantity | value |
|---|---|
| resolution, both motors stepping | **0.04875° per axis-step** at 1/16 |
| resolution, one motor stepping | 0.024375° in *each* axis (diagonal) |
| beam travel per axis-step at 3 m | 2.55 mm |
| max payload rate at 4000 steps/s | ~195 °/s (indoor need: <40 °/s) |
| payload accel at `ACCEL=20000` | ~975 °/s² |
| gear backlash, estimated | 0.16–0.46° (SLS PA12 tooth clearance); belts negligible |

### 1.5 Optics and sensors

| item | part | key facts |
|---|---|---|
| narrow camera | Logitech C270, **IR-cut filter removed**, **mounted rotated 90°** | 1280×720 **30 fps** (MSMF), rolling shutter, f ≈ 1400 px. 49° across the sensor's long axis → **28.8° horizontal as mounted**. Refocus after the filter removal. |
| wide camera | generic UVC module, 120° M12 — **`32E4:9230`, "HD USB Camera"** | **1920×1080 max** (measured), rolling shutter, ~105° horizontal, **f ≈ 730 px**, heavy barrel distortion |
| laser | **532 nm green, 5 mW**, Class 3R | on the payload between the cameras |
| IMU | GY-85 (ITG3205 + ADXL345 + HMC5883L) | I2C on J11: 0x68 / 0x53 / 0x1E |
| target | black quadcopter, ~283 mm diagonal, ~90 mm body, held by hand | |

At 3 m the C270 puts **~132 px on the drone, ~37 px on the battery.** That is
enough. Keep native resolution — do not downscale the narrow camera, ever.

#### The wide camera is not the 5 MP module this document used to claim

Measured, with the module physically connected and the frame visually
confirmed (index 1, `32E4:9230`):

- **1920×1080 is the ceiling.** Requests for 2592×1944, 2048×1536 and 4K all
  clamp to 1080p on *both* DSHOW and MSMF. Earlier revisions of this document
  specified a 2592×1944 IMX335; that was wrong.
- **It reports no serial** — only a location id (`6&112b2957&0&3`) that changes
  when you move the USB plug. See §1.5a below; this breaks the resolve-by-serial
  plan for this camera specifically.
- Generic descriptor ("HD USB Camera", empty vendor string), no microphone.
- **1920×1080 @ 30.3 fps on MSMF, and 720p is a trap** — it delivers 9 fps.
  Full table in §5.2. Run it at native resolution.

Consequences, both of which matter:

**Pixel budget.** ~105° across 1920 px is **~18 px/deg**, not the ~29 px/deg a
2592-wide sensor would give. Angular resolution on the wide camera is down
~38 % from the old assumption. It is still only a finder — it hands off to the
narrow camera — so this is survivable, but the hand-off happens later and the
minimum detectable target is correspondingly bigger. Re-check the acquisition
range budget against 18 px/deg before trusting it.

**Barrel distortion is severe and must be corrected.** The 120° M12 lens bows
straight lines visibly across the whole frame. A pixel error near the edge does
**not** correspond to the same angle as the same pixel error at centre, so the
2×2 image Jacobian is *not* constant across the field — which is exactly the
assumption §6 currently makes. Either:

1. undistort (`cv2.initUndistortRectifyMap` once, then `remap` per frame — a
   few hundred µs on GPU) and servo on undistorted pixels, or
2. calibrate the Jacobian at a 3×3 grid of field positions and bilinearly
   interpolate.

Option 1 is less work and less to get wrong. Either way, **run a chessboard
calibration on the wide camera** (`cv2.calibrateCamera`, ~20 views) to get real
intrinsics; the f ≈ 730 px above is derived from the FOV spec, not measured,
and on a fisheye-ish M12 lens that estimate is soft. Consider `cv2.fisheye` if
the standard model's reprojection error exceeds ~1 px.

The narrow C270 is mildly distorted by comparison, but calibrate it too — it is
the camera the laser actually aims through.

#### Offsets — MEASURED, no longer TBD

Read out of `sterolaserview.step` with a CAD kernel and confirmed by the
builder. All three features lie on one line, `y = 0`:

```
     wide (120°)            laser             narrow (60°, C270)
     x = -30.000 mm      x = 0.000 mm            x = +30.000 mm
```

| constant | value |
|---|---|
| `LASER_TO_NARROW_MM` | **+30.000** |
| `LASER_TO_WIDE_MM` | **−30.000** |
| `STEREO_BASELINE_MM` | **60.000** |
| vertical offset, any pair | **0.000** |

They are in `turret_vision/head_geometry.py` with the parallax maths; import
them, do not retype them. Identification was confirmed by bore radius and boss
height — narrow 8.500 / 1.700 mm, wide 8.000 / 1.400 mm; the narrow mount is
the taller boss.

**No vertical offset is a real simplification — with one catch.** Parallax
between a camera and the beam is purely horizontal *in the payload frame*. The
image sensors sit at the same height (builder confirmed), so this holds
physically. But **the C270 is mounted rotated 90°**, so in *its* frame that
horizontal offset lands on the sensor's vertical axis: its parallax correction
goes on the goal **row**, the wide camera's on the goal **column**. Apply the
simplification per camera, not globally.

**The lenses sit at different depths** — the M12 barrel protrudes far more than
the C270's. The sensors are level with each other, so this is an axial offset
only. For parallax it is negligible (a 20 mm axial offset changes the
correction by <0.1 px at 3 m). For stereo it is not: the pair is not
fronto-parallel, so rectification will rotate and crop. Let `stereoCalibrate`
solve it; never hand-write a rectification assuming coplanar centres.

#### The C270 is mounted rotated 90° — confirmed

| | horizontal FOV | px per horizontal degree |
|---|---|---|
| as specced | 49.0° | 26.1 |
| **as mounted** | **28.8°** | 25.0 |

Angular resolution is unchanged — same lens — but the **horizontal field shrinks
by 41 %**, and the vertical grows to 49°, which is the wrong axis for a target
that mostly moves side to side. The tracking window is narrower than every
earlier estimate in this document assumed; combined with the wide camera's
18 px/deg, **the acquisition envelope is now the thinnest part of the design.**
Re-check it before tuning anything else.

The sign (90° vs 270°) is **not yet verified** and cannot be read from a single
frame. Getting it backwards inverts the parallax correction rather than merely
misplacing it — the dot lands on the wrong side and doubles the error. Confirm
against a known target before trusting it.

**Do not rotate every frame to "fix" this.** That is a full-frame copy per
frame for no benefit. Carry the rotation in the geometry, where pixel error is
converted to a correction, and rotate only the spectator display.
`turret_vision/head_geometry.py` holds `NARROW_ROTATION_DEG`.

**Parallax is not optional.** At 3 m the beam lands 14 px from where the narrow
camera says the target centre is — 0.573°, or **11.8 turret steps**. At 2 m it
is 21 px. Uncorrected, the dot sits beside the drone at close range and the
error *shrinks* as the target retreats, which reads exactly like a mis-tuned
gain and is not one.

#### The cameras are not guaranteed parallel — and stereo is brutally sensitive to it

They sit in a 3D-printed shell. With only a 60 mm baseline the true disparity
is small (28 px at 3 m), so a small relative rotation swamps it:

| relative yaw | stereo range reported at a true 3 m |
|---|---|
| 0.10° | 2.76 m (−8 %) |
| 0.25° | 2.46 m (−18 %) |
| 0.50° | 2.09 m (−30 %) |
| 1.00° | 1.60 m (**−47 %**) |

One degree of print skew nearly halves the reported range. Stereo range good to
10 % at 3 m needs the relative rotation known to **~0.1°**, which no printed
part gives you by construction.

That is what `cv2.stereoCalibrate` is *for* — it measures the rotation instead
of assuming it — so this is a calibration requirement, not a design flaw. But:

1. **Stereo range is unusable until a stereo calibration has been run.** Do not
   ship a hard-coded rectification.
2. The calibration holds only while the shell does. Anything that flexes or
   reseats a camera invalidates it **silently** — the range just goes wrong.
3. **Cross-check stereo against apparent-size range every frame.** A
   persistent, range-dependent disagreement means the extrinsics moved. This is
   the only warning you will get, and it is nearly free.
4. Seed the calibration with the known 60.000 mm baseline and treat a solved
   baseline far from it as a *failed calibration*, not a discovery.

**Do not let this block the tracking loop.** Assuming a flat 3 m across the
whole 2–5 m envelope costs at most **7 px** of aim error, against a drone that
is 132 px wide at 3 m:

| true range | 2.0 m | 2.5 m | 3.0 m | 4.0 m | 5.0 m |
|---|---|---|---|---|---|
| residual | −7.0 px | −2.8 px | 0 | +3.5 px | +5.6 px |

So aiming at the drone *body* works with **no range estimate at all**. Build
the loop against `ASSUMED_RANGE_M = 3.0`, get it tracking, then add ranging to
tighten it. Ranging earns its keep for aiming at something small like the
battery (37 px at 3 m), not for hitting the airframe.

**Prefer apparent-size range over stereo when the box is clean.** A 2 px error
on a 132 px box is ~1.5 %; a 1 px stereo match error is ~3.6 %, and matching
across a 49° and a 105° camera makes 1 px optimistic. The catch is the demo:
a hand partially covers the drone, and occlusion shrinks the apparent size and
biases size-range **long**. Stereo does not fail that way. Use size when the box
looks clean, stereo when it does not, and treat a standing disagreement as
evidence the box is clipped.

#### 1.5a Resolving camera indices — the two cameras need different methods

OpenCV indices are not stable. `turret_vision/identify_cameras.py` records the
USB identity of each camera; resolve against it at startup. But the two cameras
do not offer the same handle:

| camera | VID:PID | serial | resolve by |
|---|---|---|---|
| narrow (C270) | `046D:0825` | `C8258920` (real) | **serial** — fully stable |
| wide | `32E4:9230` | none | **VID:PID + max-resolution fingerprint (1920×1080)** |

The wide camera has no stable identity of its own. Do not buy a second
identical module, and if a camera is ever swapped, re-run `identify_cameras.py`
and update the record.

**Read the Status column in that tool's output.** Windows keeps a registry
entry for every camera ever plugged into the machine, and a stale entry looks
identical to a live one except for Status. During bring-up this caused a wrong
identification: a disconnected module (`32E6:9221`, Shenzhen Icspring — which
*does* have a mic and a real serial) was read as the connected wide camera.
That is a different device and is not part of this build.

**The cameras may be replaced.** If a global-shutter ≥90 fps USB 3 camera
(AR0234 class) lands in the narrow slot, every "30 fps" note in this document
relaxes and nothing else changes. Design for 30 fps; do not hard-code it.

**The IMU:** good for finding level (non-contact homing, ~0.5°), detecting a
bump, and measuring backlash. **Useless for yaw** — the magnetometer cannot
work next to stepper motors. Not accurate enough to replace step counting.
Optional for M1–M6.

### 1.6 Safety rules — not negotiable

1. **The only real e-stop is cutting motor power (VIN).** The software e-stop
   works only while firmware is alive. Put a physical switch on VIN.
2. Laser on **only** when: a track is confirmed, no face within the inhibit
   margin, the platform is settled, and a `vel` was received in the last
   100 ms. Any one failing → laser off, before anything else happens.
3. `LASER_MAX_ON_MS` hard cap stays. `LASER_ENABLED=False` stays until the
   beam path is safe and the face interlock is tested.
4. Never plug/unplug a motor with VMOT applied.
5. Do not run motion tests with anyone in front of the platform until M5 passes.

---

## 2. What exists: the MicroPython suite

### 2.1 Connect

```
python -m mpremote connect list          # find the 2e8a:0005 port
python release_board.py                  # if the console is running
python -m mpremote connect COMx repl     # interactive
```

`main.py` runs a console at boot with prompt `turret> `. Ctrl-C is the panic
stop (parks everything, stays in the console). Two Ctrl-C within 1 s exit to
the REPL. `quit` also exits.

### 2.2 Commands you will use

```
state                       one line: STATE {json}

  -- differential (payload axes) --
diff                        resolution, position, ratio, signs
pitch <deg> [dps]           pure pitch: both motors together
yaw <deg> [dps]             pure yaw: both motors opposed
dmove <dpitch> <dyaw> [dps] relative payload move, both at once
aim <pitch> <yaw> [dps]     absolute payload move
dzero                       set this pose as payload zero
dchar [steps]               characterise the differential (MANUAL)
dsign pitch|yaw|A|B         flip a sign for this session

  -- motors (neither is a payload axis) --
enable|disable <pan|tilt|both>
move <axis> <steps> [rate]  counted trapezoidal move, blocking
deg <axis> <deg> [rate]
ms <axis> <1|2|4|8|16>
zero <axis>
home <axis>                 3-phase homing on the assigned endstop
laser on|off|pulse <ms>     refused unless LASER_ENABLED
stop                        disable all, outputs off
safe                        the no-motion test suite
wiring                      coil connector reference
```

`state` JSON:

```json
{"payload":{"pitch":0.0,"yaw":0.0,"axis_step_deg":0.04875,"N":2.3077},
 "axes":{"pan":{"enabled":false,"microstep":8,"position":0,"degrees":0.0,
  "steps_per_rev":1600,"fault":null,"pad2":"vdd-from-gpio","has_endstop":true},
  "tilt":{...}},
 "endstops":{"pan":{"gpio":16,"triggered":false,"raw":1},"tilt":{...}},
 "outputs":{"laser":false,"ir":false,"fan":false},
 "driver":"A4988","laser_enabled":false,"rst_slp":"3V3",
 "max_rate":{"pan":4000,"tilt":4000}}
```

`position` is in microsteps of the *current* `microstep` setting.

### 2.3 Deploy

```
powershell -ExecutionPolicy Bypass -File deploy.ps1          # gates + copy
python tests/verify_against_kicad.py                        # must pass
python tests/host_test.py                                   # must pass
```

`deploy.ps1` refuses if the pin map disagrees with the copper. Keep it that way.

### 2.4 How step generation works today (you will add a second mode)

Each axis owns one PIO state machine (`pan`→sm 0, `tilt`→sm 4) running:

```
out(x, 32)          ; one FIFO word per step = extra delay cycles
set(pins,1) [7]     ; 8-cycle high pulse
set(pins,0)
hold: jmp(x_dec, hold)
```

at 1 MHz, overhead 11 cycles. `Axis.move()` streams a trapezoidal profile of
delay words into the FIFO. Two register-level helpers exist and you will reuse
them: `_set_funcsel(gpio, PIO0|PIO1|SIO)` hands the STEP pad between PIO and
SIO (MicroPython does not do this for you), and `_clear_fifos(sm_id)`.

`Axis._claim_step_pin()` / `_release_step_pin()` wrap those. **Call
`_claim_step_pin()` before activating any SM on that pin**, or the pad stays
under SIO and no pulses come out — this bit us once already.

---

## 3. The tracking design

Close the loop on the **pixel**, not on angles. The cameras ride on the payload
with the laser, so a detection in the narrow camera *is* the pointing error.
The laser dot should sit at a known pixel `g` (depends weakly on range); the
target is at pixel `p`; drive `p` onto `g`.

```
C270 frame (native res)
   │
   ▼
detector → bbox → aim-point stabiliser (§5.5)
   │
   ▼
pixel Kalman  x = [u, v, u̇, v̇]   adaptive Q   → predict to (t_now + L)
   │
   ▼
goal g = g(R)                                   §6.4 — or the detected dot, §5.6
   │
   ▼
e = p̂ − g            (px)
ω = J⁻¹ · (K·e + p̂̇)  (motor steps/s, per motor)     ← feedforward + proportional
   │
   ▼
serial:  vel <ωA> <ωB>          every frame
   │
   ▼
Pico: accel-limit, hold rate in PIO, watchdog, dead-reckon position
```

**Why feedforward is the whole game.** With ~50–70 ms of loop latency the
proportional term is only stable up to ~2 Hz. It cannot chase a hand. The
`p̂̇` term commands the turret at the *target's* angular rate continuously, so
between frames the laser is already moving with the drone; `K·e` only trims.

**Why this survives sloppy calibration.** In an angle-based chain every
calibration error is a miss on the target. Here they all live inside `J` and
`g`, and the loop tolerates `J` being 20 % wrong — it becomes a gain error the
P-term absorbs.

**What is empirical, and only these four:** `L` (latency), lash (steps), `J`
(2×2), `g(R)` (two constants per axis). §6 has the procedures.

---

## 4. Firmware: add velocity mode

Extend `stepper.py` and `cli.py`. Do not touch `pinmap.py`. Keep every
existing command working — `move`, `home`, `safe` and the tests are your
regression suite.

### 4.1 The PIO program

A **second** state machine per axis (`pan`→sm 1, `tilt`→sm 5), same `set_base`
pin, running a period-register program at **5 MHz**:

```python
@rp2.asm_pio(set_init=rp2.PIO.OUT_LOW)
def _vel_program():
    label("top")
    pull(noblock)          # FIFO empty -> OSR <- X  (keeps last period)
    mov(x, osr)            # X = period, persists across iterations
    jmp(not_x, "top")      # period 0 = idle, no pulses, keep polling
    mov(y, x)              # Y = loop counter (X must survive)
    set(pins, 1) [7]       # 8 cycles = 1.6 us high  (A4988 min 1 us)
    set(pins, 0)
    label("hold")
    jmp(y_dec, "hold")     # Y+1 cycles
    jmp("top")
```

Cycles per step = **15 + Y**. So for a rate `r` steps/s:

```
period = round(5_000_000 / r)
Y      = max(0, period - 15)
```

Rate quantisation at 5 MHz is ≤0.006 % at 300 steps/s — no fractional
accumulator needed.

`pull(noblock)` on an empty FIFO copies X into OSR (RP2040 datasheet §3.4.7).
That is what makes the period persist without the CPU feeding anything.

### 4.2 Updating the period without FIFO build-up

Do **not** just `sm.put()` every tick — at low step rates the 4-deep FIFO fills
with stale periods and the rate lags by four steps. Instead:

```python
def _set_period(self, y):
    sm = self.vel_sm
    sm.put(y)
    sm.exec("pull(noblock)")   # our word is there -> into OSR
    sm.exec("mov(x, osr)")     # X = new period, immediately
```

X is only read at the top of each step (`mov(y,x)`), so clobbering it
mid-hold-loop is harmless. Takes effect within one step at any rate. FIFO ends
empty every time.

### 4.3 The velocity loop (200 Hz `machine.Timer`)

Per motor, state: `target_rate`, `current_rate`, `dir`, `pos_f` (float, dead
reckoned), `t_last_vel_ms`.

```
every 5 ms:
  if now - t_last_vel_ms > 100:            # watchdog
      target_rate = 0; laser off (direct GPIO write, not via CLI)
  dr = target_rate - current_rate
  step = min(|dr|, ACCEL * 0.005)          # accel limit, steps/s per tick
  current_rate += sign(dr) * step
  if sign(current_rate) != dir and current_rate != 0:
      _set_period(0)                       # stop
      sleep_us(300)
      set DIR pin
      dir = sign(current_rate)
  _set_period(y_for(|current_rate|))       # 0 if current_rate == 0
  pos_f += current_rate * 0.005
```

Keep the timer callback allocation-free: pre-compute everything, use ints where
you can, no string formatting inside it. MicroPython timer callbacks cannot
allocate.

`ACCEL` here is per-motor microsteps/s². The existing `config.ACCEL = 20000`
is a sane start.

### 4.4 Mode switching

`Axis` gains `mode ∈ {"idle","move","vel"}`. `move()` and `single_step()`
refuse when `mode=="vel"`; `vel` command refuses when a `move` is in progress.
Switching into vel: `_claim_step_pin()`, `move_sm.active(0)`, `vel_sm.active(1)`,
`_set_period(0)`. Switching out: `_set_period(0)`, wait 2 ms, `vel_sm.active(0)`,
`_release_step_pin()`, fold `pos_f` into `position`.

### 4.5 New commands

```
vel <A_steps_per_s> <B_steps_per_s>   signed. Enters vel mode on both if needed.
                                       Resets the watchdog. Replies "ok".
vel 0 0                                stop (still in vel mode, rate 0)
velmode off                            leave vel mode, back to move mode
preload <steps>                        move A +steps, B -steps (counted), then zero
                                       both. Takes up gear lash. See §6.2.
lash?                                  prints the stored lash constants
signs <±1> <±1>                        stored sign flip per motor (config)
```

**Motor A ≡ `pan` ≡ A3. Motor B ≡ `tilt` ≡ A2.** The `vel` arguments are
per-motor step rates after the sign constants — the host's `J` already
accounts for the differential, so the Pico never needs to know pitch from yaw.

Extend `state` with `"mode"`, per-motor `"rate"`, and `"vel_age_ms"`.

### 4.6 Position while in vel mode

Dead-reckoned from commanded rate × time. Exact to within one step per rate
change because the PIO runs the commanded period exactly. Good enough: the
closed loop corrects any residual every frame, and absolute position only
matters for homing, which uses `move`.

*(Optional, later: count real steps by DMA from the PIO's RX FIFO with a `push`
added after each pulse. Not needed for the demo.)*

### 4.7 Tests to add

To `tests/host_test.py`, against the existing `fake_hw` stubs: period math,
accel limiting, direction-change sequencing, watchdog trip, and that `move`
refuses in vel mode. Extend `fake_hw.StateMachine` with `exec()`.

On hardware: `vel 200 200` for 10 s, then `state` — dead-reckoned position
should read ~2000 ± 10 on each motor. `vel 0 0` — motors silent within
`2000/20000 = 100 ms`. Let the watchdog trip — verify rate decays and the
laser GPIO reads low.

---

## 5. Host software

Python 3.11+, Windows, RTX 4070. Packages: `opencv-python`, `numpy`, `torch`
(CUDA), `ultralytics`, `pyserial`. TensorRT export via ultralytics.

### 5.1 Process and thread layout

```
process: tracker
  thread capture-narrow   MSMF, grab loop, newest-wins slot, timestamps
  thread capture-wide     same
  thread inference        owns CUDA; detector on narrow (every frame) + wide (search)
  thread control          Kalman, gating, control law, serial vel
  main thread             spectator display at ≤30 fps from a 640×360 copy
```

**Slots, not queues.** Each hand-off is a lock + one variable; the consumer
takes the newest and the producer overwrites. A queue would deliver stale
frames the filter treats as current.

Runtime setup, once, after models load:

```python
gc.freeze(); gc.disable()          # 35 ms collect pause measured -> 0 after freeze
sys.setswitchinterval(0.001)       # default 5 ms GIL hold -> 1 ms
```

Run `gc.collect()` manually only when no track is active. Preallocate arrays
in the control loop.

**Display:** draw on a 640×360 copy, never the inference buffer. Measured cost
~1.3 ms per frame; at 30 fps that is ~4 % GIL duty — acceptable. If the
control-loop period jitter exceeds 5 ms with the display on, move the display
to a separate process fed by shared memory. Measure before deciding.

### 5.2 Capture

**Use MSMF, not DSHOW. This is measured, and an earlier version of this
document said the opposite.** On the C270 at 1280×720:

| backend | `set(FOURCC, MJPG)` | reported FOURCC | `CAP_PROP_FPS` | **delivered fps** |
|---|---|---|---|---|
| `CAP_DSHOW` | returns `True` | `YUY2` | 30.0 | **10.0** |
| `CAP_MSMF` | returns `False` | `""` (empty) | 30.0 | **30.4** |

DirectShow accepts the FOURCC request, silently ignores it, stays in
uncompressed YUY2 — which cannot fit 720p30 through USB 2 — and *still* reports
`CAP_PROP_FPS = 30`. Every signal it gives you is wrong, and you lose two
thirds of your frame rate without a single error. Media Foundation negotiates
the compressed mode itself, so it **refuses** `set(FOURCC)` and reports an
empty FOURCC (it hands back already-decoded BGR). Do not read that `False` as
a failure and do not "fix" it by switching back to DSHOW.

```python
cap = cv2.VideoCapture(index, cv2.CAP_MSMF)      # MSMF -- see the table above
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)        # manual; convention differs by backend
cap.set(cv2.CAP_PROP_EXPOSURE, -6)               # tune: short, no bloom on the dot
cap.set(cv2.CAP_PROP_GAIN, ...)                  # tune
# Never trust CAP_PROP_FPS. Time 40 frames at startup and assert >= 25.
```

**Verify the rate at startup, per camera, every run.** Time 40 `read()` calls
and refuse to start if the delivered rate is below 25 fps. A silent drop to 10
fps triples the control-loop latency and the tracking will look like a tuning
problem for hours before anyone checks the capture.

Full measured mode table, C270 on this machine (`probe_camera.py`, 40-frame
bursts, written to `turret_vision/camera_modes.json`):

| backend | size | delivered fps | median gap | worst gap |
|---|---|---|---|---|
| MSMF | **1280×720** | **30.14** | 32.5 ms | 50.0 ms |
| MSMF | 640×480 | 30.07 | 32.0 ms | 48.2 ms |
| DSHOW | 1280×720 | 10.00 | 98.5 ms | 121.6 ms |
| DSHOW | 640×480 | 30.11 | 32.1 ms | 48.5 ms |

1920×1080 and above are rejected on both backends — 1280×720 is the camera's
ceiling, as expected. Note DSHOW is fine at 640×480: the bandwidth only runs
out at 720p, which is why this is easy to miss if you test at VGA.

Two things confirmed by the same run and worth having in hand:

- `CAP_PROP_BUFFERSIZE` is **not settable** (`set()` returns False). The
  continuous-grab thread is mandatory, not an optimisation.
- Manual exposure **does** take, via DSHOW. If MSMF's exposure control turns
  out to be inadequate on the bench, the fallback is to set exposure through a
  short DSHOW open, release it, then reopen on MSMF for streaming — UVC
  controls persist on the device across handles. Verify that on the bench
  before relying on it.
- Worst-case frame gap at 720p/MSMF is 50 ms against a 33 ms median, so the
  predictor has to tolerate an occasional missed frame. Do not treat a single
  late frame as a lost track.

**MSMF opens are slow and unreliable. Treat startup as a failure-prone step,
not a formality.** Observed repeatedly on this machine:

- `VideoCapture(1, CAP_MSMF)` on the integrated webcam (which has a paired IR
  sensor) blocked over three minutes with no error.
- The same open on the C270 took under a second on one run and over three
  minutes on the next, with nothing changed.
- **Opening two MSMF captures concurrently from two threads does not work** —
  the second never became ready within 20 s. Media Foundation's source-reader
  startup does not tolerate being raced.
- A camera whose process was killed without `release()` stays claimed, and the
  next open either blocks or returns a capture that delivers no frames. The
  symptom is `isOpened() == True` with every `read()` failing.

So:

1. **Open cameras sequentially, never concurrently.** Only the steady-state
   capture is parallel. `dual_capture_test.py` does it this way for exactly
   this reason.
2. Open in a worker thread with a ~10 s deadline; fall back to DSHOW on
   timeout, degraded rate and all, rather than letting startup wedge.
3. **Verify frames, not just `isOpened()`.** Require N successful `read()`
   calls before declaring a camera ready.
4. **Always `release()`**, including on exception and on Ctrl-C. Wrap the
   capture in a context manager or `try/finally`. A leaked handle costs you the
   next run.
5. Resolve the index by VID/PID first (`identify_cameras.py`) so you never open
   the wrong device to begin with.

Budget real time for camera startup on demo day — it is not instant, and it is
the least reliable part of the host stack.

The grab thread runs `cap.grab(); t = perf_counter(); cap.retrieve()` in a
tight loop and publishes `(frame, t)` to the slot. **This is the only way to
defeat the driver's hidden 2–5 frame buffer** — `CAP_PROP_BUFFERSIZE` is not
honoured on Windows. Timestamp at `grab()` return; that is the best available.

Lock exposure and gain. Auto-exposure changes frame timing (unpredictable
latency) and blooms the green dot into a blob.

For the wide camera, decode MJPEG on the GPU: `torchvision.io.decode_jpeg`
with `device="cuda"`. At 1920×1080 CPU decode is ~6–12 ms per frame and holds
the GIL — worth moving off the CPU, but measure first; it may fit.

**Run the wide camera at 1920×1080. Do NOT downscale it to 720p.** An earlier
revision of this document suggested 720p to save bandwidth. That is backwards
on this module — measured:

| backend | size | delivered fps | median gap | worst gap |
|---|---|---|---|---|
| MSMF | **1920×1080** | **30.31** | 32.5 ms | 46.1 ms |
| MSMF | 1280×720 | **9.01** | 107 ms | **277.8 ms** |
| MSMF | 640×480 | 29.77 | 32.1 ms | 48.5 ms |
| DSHOW | 1920×1080 | 4.97 | 204.8 ms | 244.0 ms |
| DSHOW | 1280×720 | 9.93 | 97.5 ms | 115.2 ms |

1080p is this module's native compressed mode. 720p apparently is not in its
MJPEG list, so it falls back to uncompressed and starves — **a 70 % frame-rate
loss for asking for fewer pixels**, with a worst-case gap of 278 ms, nearly
nine frame periods. If you genuinely need to cut pixels, drop straight to
640×480 (29.8 fps), which is a real compressed mode. Never assume a smaller
frame is a faster one; probe it.

Same rule as the C270: **verify the delivered rate at startup and refuse to
run below 25 fps.**

#### USB: cameras on separate NATIVE ports. This is not advisory.

Cameras on **separate USB root ports**; Pico on a third if possible. **Never
run both cameras through one hub or USB-C splitter.**

This was hit for real during bring-up. On the dev laptop both cameras were
behind a USB-C → 3× USB-A splitter — a single hub, so every port shares one
upstream link. Result: each camera worked perfectly alone (narrow 1280×720,
wide 1920×1080), but the second one to start **failed outright**, with
`ERROR_DEVICE_NOT_CONNECTED` (`0x8007048F`) from MSMF.

That error is a liar. The device is present, enumerated, and works alone. A
UVC camera reserves isochronous bandwidth for its whole stream at start-up, and
when there is none left the start is simply refused. Expect to waste an hour on
this if you do not recognise it.

**Diagnosis:** compare `LocationInfo` strings (`identify_cameras.py`, or Device
Manager → View → Devices by connection). Two cameras sharing a path prefix
share an upstream link. On the dev laptop both read `...0014.0000.003.x` —
same branch. The integrated webcam was on `...006`, a different one.

**Fix:** separate native ports. The GPU host has native USB-A, so this specific
failure probably does not reproduce there — **but verify it, do not assume it.**
Run `turret_vision/dual_capture_test.py` on the GPU host before anything else;
it opens both cameras, measures them together, compares against the solo rates
in `camera_modes.json`, and on failure walks a fallback ladder to report what
*does* coexist. A laptop result does not transfer.

If you are ever forced onto a hub, drop the **wide** camera to 640×480 — it is
the finder, so the cost is acquisition range, not tracking. Never put it at
1280×720, which on that module is both slower and uncompressed.

### 5.3 Detector

- **Model:** Ultralytics YOLO, nano or small (YOLO11n / YOLOv8n). Start from
  COCO weights (already knows `bird`, `airplane`, `person`).
- **Classes:** `drone`, `bird`, `plane`. Faces come from a separate detector
  (below) so you do not need face labels in your training set.
- **Input:** the **full 1280×720 narrow frame at `imgsz=1280`**. At 30 fps a
  4070 runs a nano model at that size in a few ms. No ROI logic needed for the
  demo; it stays native resolution. (Add ROI cropping later only if you
  upgrade to a fast camera and inference becomes the limit.)
- **Wide camera:** `imgsz=1280` on a downscaled frame is fine — it only
  searches. Run it at 10 fps while a track is active, 30 fps when searching.
- **Export:** `model.export(format="engine", half=True, imgsz=1280)`. Load the
  `.engine`. 2–3× over PyTorch.
- **Confidence:** 0.35 to *propose*, then the gate (§5.4) decides.

**Face detector:** OpenCV's built-in **YuNet** (`cv2.FaceDetectorYN`). No
training, fast, runs on the wide camera every frame. Its output drives the
laser inhibit. Do not substitute a person detector — a person detector fires
on the hand holding the drone and the laser never fires.

#### The C270 has had its IR-cut filter removed — this changes the detector plan

Confirmed by the builder, invisible in any spec sheet, and it undercuts two
assumptions above.

- **COCO weights were trained on IR-cut imagery.** With the filter gone the
  sensor responds well past 700 nm, near-IR bleeds into all three channels, and
  colour stops being photometrically meaningful. This is a genuine domain
  shift, not a nuisance — expect degraded confidence straight out of the box.
  **Fine-tuning on frames from this camera moves from "most valuable" to
  "required".** Item 1 below is now the critical path.
- **Nothing may gate on colour.** Drone-vs-bird colour cues and green-channel
  laser-dot detection both get much weaker. Shape and motion still work.
- **Re-focus the C270 and verify sharpness at 2–5 m before any calibration.**
  Removing ~1 mm of filter glass shortens the optical path and shifts the focal
  plane. The lens holder is accessible (it has been opened already), so this is
  adjustable. It matters beyond looking nice: a soft image inflates bounding
  boxes, which biases apparent-size range **short** and silently corrupts the
  parallax correction. Do this first — every pixel measurement downstream
  depends on it.
- **Lock exposure and gain.** Room lighting and daylight are NIR-rich, so
  highlights bloom and auto-exposure hunts. Already required for latency
  reasons; now doubly so.

**The upside is real.** The board's IR LEDs (GPIO14) now do something. Active
NIR illumination lights a black drone in a dim room, is invisible to
spectators, and carries no laser-safety burden. If the demo room is dim or the
drone is low-contrast against the crowd, this is the lever to pull — and it is
already wired. Capture the training set **with the IR LEDs in whatever state
the demo will use**, or the fine-tune is on the wrong domain again.

**Training data, in order of value:**
1. **200–500 frames of the actual drone, held by a hand, in the actual room,
   through the filter-removed C270.** Label with any box tool; bootstrap boxes
   from COCO weights and correct them. This matters more than everything below
   combined, and the filter removal is why.
2. Drone-vs-Bird Detection Challenge dataset — exists for exactly this
   confusion.
3. Anti-UAV / Anti-UAV410; Roboflow Universe drone sets.
4. Augment with random erasing (hand occlusion) and motion blur.

### 5.4 Track and gate

Single target. Associate the detection nearest the Kalman prediction within a
gate of 3σ (start: 60 px). No ByteTrack needed for one target — add it only if
multiple drones ever appear.

| state | enter | leave |
|---|---|---|
| SEARCH | start; TRACK lost | drone detected in wide → slew toward it |
| ACQUIRE | drone detected in narrow | 5 consecutive frames → TRACK; 3 misses → SEARCH |
| TRACK | confirmed | 10 consecutive misses → COAST |
| COAST | track lost | hold last velocity 150 ms, decay to 0 over 300 ms; **laser off immediately** |

Laser may be on only in TRACK, and only when the face inhibit is clear and
`|e| < 25 px`.

### 5.5 Pixel Kalman with adaptive Q

State `x = [u, v, u̇, v̇]`, measurement `z = [u, v]` = bbox centre.

```
F(dt) = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
H     = [[1,0,0,0],[0,1,0,0]]
Q(dt) = q · [[dt⁴/4, 0, dt³/2, 0],[0, dt⁴/4, 0, dt³/2],[dt³/2, 0, dt², 0],[0, dt³/2, 0, dt²]]
R     = σ_m² · I₂         σ_m = 3 px to start
```

`dt` from the capture timestamps, never assumed. Predict for the control law
to `t_now + L` (§6.1).

**Adaptive Q:** innovation `ν = z − H·x⁻`, `S = H·P⁻·Hᵀ + R`,
`NIS = νᵀ S⁻¹ ν`. If `NIS > 9.21` (χ², 2 dof, 99 %): `q ← min(q·8, q_max)`.
Every frame: `q ← q_base + 0.85·(q − q_base)`. Start `q_base` so that
`√q ≈ 3 m/s²` expressed in px/s² (`3 · f / R` ≈ 1400 px/s² at 3 m), `q_max`
= 30× that. Tune on a recording of someone jinking the drone by hand until
the innovation sequence looks white.

**Aim-point stabiliser (occlusion by the hand):** track bbox area alongside
the centre. If area drops >30 % in one frame while the centre shifts toward
the surviving edge, hold the previous centre's offset from that edge instead
of re-centring on the remnant. Release when area recovers within 20 %. Also
**bias the aim point up by 15 % of box height** — the hand is below the drone.

**Rolling-shutter bias:** the C270 reads out over ~20 ms, so a moving target's
centroid is displaced in the direction of motion by roughly `k·v̂` with
`k ≈ 0.010 s`. Subtract it from each measurement. Calibrate `k` once by
panning the turret past a static object at a known step rate and measuring the
apparent shift. Drop this if the camera is upgraded to global shutter.

### 5.6 The laser dot as the goal

532 nm on a black airframe is high contrast. Each frame, in a 60×60 window
around `g(R)`:

```
mask = (G − max(R,B)) > τ     with locked exposure, τ ≈ 60 to start
```

Accept a blob of 3–40 px area with roundness > 0.6. If found, **use its
centroid as `g`** for this frame — the loop then measures its own miss and
every modelling error cancels. If not found, use `g(R)`. Never gate anything
on the dot being visible; it is opportunistic, as the human said.

**This mask is chroma-based, and the C270 has no IR-cut filter.** 532 nm is
still strongly G-dominant, so the test should survive — but near-IR passes
every Bayer filter roughly equally, so NIR-lit surfaces go grey and the whole
scene desaturates, shrinking the margin. Two consequences:

- **Retune `τ` on real frames.** The τ ≈ 60 starting point assumes an IR-cut
  sensor; it is a guess here.
- **The IR LEDs make this worse, not better.** Under NIR illumination the black
  airframe goes bright grey, and `G − max(R,B)` on a bright grey background is
  small — exactly where the dot has to be found. If you use the IR LEDs for
  detection, either pulse them off on alternate frames for dot-finding, or add
  a local-brightness-peak term so the test does not rest on chroma alone. The
  dot is both the greenest *and* the brightest thing in a 60×60 window; use
  both.

Since the dot is opportunistic, none of this is fatal — worst case you fall
back to `g(R)` more often. Just do not assume the mask works because it works
on a normal webcam.

### 5.7 Control law

```
p̂  = predicted target pixel at t_now + L
p̂̇  = predicted pixel velocity
e   = p̂ − g
if |e| < 1 px: e = 0                           # deadband on the P term only
ω   = J_inv @ (K·e + p̂̇)                      # ω = [ωA, ωB] motor steps/s
ω   = clip(ω, ±MAX_RATE)
send f"vel {ωA:.0f} {ωB:.0f}"
```

`K` in 1/s. Start **K = 2.0**. Raise until the response to a step (jog the
turret, watch the error decay) starts to overshoot, then back off 30 %. With
`L ≈ 60 ms` expect to land around K = 3–4.

Send `vel` on every frame, immediately after the detection lands. The Pico
holds the rate between frames — that is what keeps the beam moving with the
drone at 30 fps. Send `vel 0 0` on any transition out of TRACK; the watchdog
covers a host crash.

### 5.8 Serial

`pyserial` at any baud (USB CDC). One writer thread, one line per command,
read the `ok` reply. Keep the writer isolated so a USB stall cannot block the
control thread — the control thread writes into a 1-deep slot, the serial
thread drains it. Measure `vel`→`ok` round trip; expect 1–3 ms.

### 5.9 Spectator display

Read from the control thread's published state. Show: the narrow view with
bbox, predicted point, goal pixel, the dot if seen; the wide view with all
detections and faces; a large laser state indicator — `SEARCH` / `ACQUIRE` /
`TRACK` / **`INHIBITED — face`** / `FIRING`; and the adaptive-Q level as a bar
so a jink is visible. The `INHIBITED — face` banner is the demo moment. Keep
it to 30 fps and 640×360.

---

## 6. Calibration — the four empirical constants

Do these in order. Each takes minutes. Record the numbers in `config.py`.

### 6.1 Latency `L`

Wire an LED to a spare Pico GPIO (18 or 22) and point it at the narrow camera.
Add a `blink` command. From the host: send `blink`, timestamp; find the first
frame whose ROI brightness jumps, take its capture timestamp. `L` = the
difference, averaged over 50 trials; also record the spread.

Expect 50–70 ms with the grab thread; 90–160 ms without it. If you see the
latter, the grab thread is not doing its job. Repeat with the display on and
off; the difference is your display cost.

### 6.2 Backlash and preload

Two-move test from vel mode is easiest:

1. `vel 100 100` for 2 s (both motors, same sign → pitch), then `vel -100 -100`.
2. In the narrow camera, count frames from the reversal until a static scene
   feature begins moving the other way. `lash_steps = 100 · Δt`.
3. Repeat opposite-sign (yaw). Repeat both twice. Take the worst.

Set `PRELOAD_STEPS = round(1.2 · lash_steps)`. On startup, after homing:
`preload <PRELOAD_STEPS>`. Re-run the two-move test; residual lash should be
0–2 steps. If it is still >3, enable reversal compensation in the 200 Hz loop:
on a sign change, add `residual` extra steps at `MAX_RATE` before resuming.

Measure on the day. PA12 swells with humidity.

### 6.3 The image Jacobian `J`

Static scene (or the drone held still). Note a distinct feature's pixel
position `p₀` in the narrow camera.

1. `move pan 200` (Motor A). Measure `p₁`. `move pan -200`.
2. `move tilt 200` (Motor B). Measure `p₂`. `move tilt -200`.

```
J = [[ (p₁−p₀).u/200, (p₂−p₀).u/200 ],
     [ (p₁−p₀).v/200, (p₂−p₀).v/200 ]]        px per step
J_inv = inv(J)
```

Use 200 steps so the shift is tens of pixels but the feature stays in frame.
This `J` already contains the belt, the differential, the joint order, the 11°
axis tilt, the camera's focal length and the sign of everything. If the demo's
yaw range exceeds ±25°, repeat at the extremes and interpolate; otherwise one
`J` at the centre is fine.

**Signs:** if the turret runs away from the target instead of toward it, a
column of `J` has the wrong sign. That is finding 5 showing up. Fix it in `J`,
not in the loom.

### 6.4 Goal pixel `g(R)`

Point at a flat wall. Enable the laser. At two measured distances `R₁`, `R₂`
(say 2.0 m and 4.0 m), find the dot's pixel `g₁`, `g₂` in the narrow camera.
Fit per axis:

```
g(R) = g_∞ + c / R          two points → g_∞ and c
```

That is the parallax correction, calibrated without knowing `f` or the
offsets.

**This is also why the C270's 90° rotation and its unknown sign do not block
you.** Fitting `g(R)` per axis from two measured points determines the
direction and magnitude of the parallax empirically — rotation, sign, axis swap
and lens depth all fall out of the fit. The sign ambiguity only bites if
someone hard-codes the correction from the CAD numbers instead of calibrating.
**Do the calibration; do not derive `g(R)` analytically.**

Range `R` at runtime from **apparent size**:
`R = f · W_real / w_px` with `W_real` = the drone's real width and `f` from
a single measurement (known-width object at known distance). A 20 % range
error costs ~3–4 px at 3 m; apparent size is better than that.

Stereo range via the wide camera is an upgrade, not a requirement.

---

## 7. Build order with acceptance tests

| M | build | passes when |
|---|---|---|
| **M0** | Deploy the existing suite; `safe` passes; `move pan 400` / `move tilt 400` move the right physical motors | `safe` all PASS/INFO/WARN, no FAIL |
| **M1** | Firmware vel mode (§4) | hardware test in §4.7 passes; host tests pass |
| **M2** | Latency `L` (§6.1), backlash + preload (§6.2) | `L` known with spread; residual lash ≤2 steps |
| **M3** | Capture threads + display; **`J` calibrated** (§6.3) | 30 fps sustained on both cameras; `L` unchanged with display on |
| **M4** | Control loop with a **hand-clicked pixel** as the target (no detector) | click a static point → laser lands on it within 2 frames and stays; move the *turret* by hand → it returns |
| **M5** | Same, with the target moved by hand (click follows a template tracker) | laser follows a hand-moved marker at 0.5 m/s with <15 mm error, no visible stepping |
| **M6** | Detector fine-tuned; face detector; gate | tracks the real drone, held; face inhibit fires reliably, hand does not trip it |
| **M7** | Adaptive Q, occlusion stabiliser, `g(R)` (§6.4), dot override | a jink re-acquires within 5 frames; laser stays on the airframe with a hand over a third of it |
| **M8** | Wide→narrow handoff; rehearsal; hard-negative retrain | end-to-end demo from cold start |

**M4 and M5 are the ones to protect.** They prove the loop with nothing
learned in it. If M5 does not hold, no detector will save M6. Do not skip to
M6 because it is more interesting.

---

## 8. Tuning notes

- **Jitter, not lag, is what looks bad.** If the beam wanders while the target
  is still, lower `K` or raise `R` (measurement noise) before touching `q`.
- **Lag on a fast move** → `L` is under-estimated, or `q_base` is too low.
  Re-measure `L` first.
- **Overshoot on a jink** → `q_max` too high or decay too slow. Halve `q_max`.
- **Laser lands beside the target consistently** → `g(R)` is off. Re-do §6.4,
  or the range estimate is wrong (check `W_real`).
- **Stair-stepping / visible ticks** at low speed → `ACCEL` too high for the
  step rate, or the Pico timer is missing ticks. Check for allocation in the
  callback.
- **Dead zone after reversals** → preload lost. Re-run `preload`. It drifts
  slowly if the two motors' rates differ for long periods.
- **Runaway** → a sign in `J` is wrong. Stop, fix, redo §6.3.

---

## 9. What would surprise you if I did not say it

- `pan`/`tilt` in the firmware are **not** pan and tilt. They are Motor A and
  Motor B of a differential. Both move for any single-axis motion.
- The two motors turn **opposite directions** for the same command sign. `J`
  absorbs it. Do not "fix" the wiring.
- The C270 through OpenCV with `CAP_DSHOW` silently delivers 10 fps instead of
  30 at 720p while reporting 30 — use `CAP_MSMF` and verify the real rate.
- The C270 through OpenCV silently adds 2–5 frames of latency unless you use
  the grab-loop pattern.
- A person detector will keep the laser off during the demo. Use faces.
- `gc.collect()` measured **35.6 ms** on a modest heap. Freeze it.
- The Pico's STEP pad must be claimed for PIO every time an SM is activated;
  MicroPython does it only at construction.
- MicroPython cannot build an exception class from two built-in bases — the
  existing code uses a `usage` flag instead; keep that pattern.
- The magnetometer is decoration. The gyro and accelerometer are useful.
- Never plug or unplug a motor with VMOT live.

---

## Appendix A — kinematics, for understanding only

With motor angles `θ_A, θ_B` and belt reduction `N = 2.3077`:

```
pitch = (θ_A + θ_B) / 2N          θ_A = N·(pitch + yaw)
yaw   = (θ_A − θ_B) / 2N          θ_B = N·(pitch − yaw)
```

Signs to be confirmed empirically (the IMU three-move test in
`VISION_BUILD_PLAN.md` §2). Pitch is the outer joint; the yaw axis rides on
the carrier and is 11° off nominal +Y in the CAD rest pose. The two axes
intersect at CAD (0, 28, 0). For a pitch-outer wrist with boresight along +x
the closed-form pointing inverse is `θ_pitch = atan2(−u_z, u_x)`,
`θ_yaw = atan2(u_y, √(u_x²+u_z²))`. None of this is used by the servo loop.

## Appendix B — if the narrow camera is upgraded

Target: global shutter, ≥90 fps at native resolution, USB 3 uncompressed, UVC,
manual exposure, ≥1280 px wide, colour. **AR0234** class (Arducam / ELP /
e-con See3CAM_24CUG). Then: drop the rolling-shutter bias term; run inference
on an ROI crop only if it becomes the bottleneck; expect manoeuvre response to
improve ~3× and the honest pitch to include evasive targets.

## Appendix C — the sim

`https://claude.ai/code/artifact/3c0bc01c-0d6b-4e6c-ab59-a40f6d2a129b` — the
wrist's real CAD geometry with the differential kinematics live. Drive the
motor sliders to see which motion is pitch and which is yaw.
