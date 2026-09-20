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
  stepper.py                 PIO step generation, Axis class. You will extend this.
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
input bevels through 20T→60T belts (**3:1**), miters 1:1. Cameras and laser sit
on the output gear.

- Both motors turning the **same** way → carrier rotates → **pitch** (up/down)
- Both motors turning **opposite** ways → output gear spins → **yaw** (left/right)
- Any single motor moving alone → equal parts pitch and yaw

**Pitch is the outer joint.** The carrier pitches about the fixed axis; the
output gear yaws about an axis *carried by* the carrier, which is tilted 11°
off nominal in the CAD's rest pose. So the yaw axis moves with pitch.

**You do not need any of that in code.** The control loop uses an empirically
calibrated 2×2 matrix (§6.3) that maps pixel error directly to motor step
rates, and that matrix contains the belt ratio, the differential, the joint
order and the axis tilt. The analytic form is in Appendix A for understanding
and for the pitch deck.

| quantity | value |
|---|---|
| resolution, both motors stepping | **0.0375° per axis-step** at 1/16 |
| resolution, one motor stepping | 0.0187° in *each* axis (diagonal) |
| beam travel per axis-step at 3 m | 2.0 mm |
| max payload rate at 4000 steps/s | ~150 °/s (indoor need: <40 °/s) |
| payload accel at `ACCEL=20000` | ~750 °/s² |
| gear backlash, estimated | 0.16–0.46° (SLS PA12 tooth clearance); belts negligible |

### 1.5 Optics and sensors

| item | part | key facts |
|---|---|---|
| narrow camera | Logitech C270 | 1280×720 **30 fps**, rolling shutter, USB 2 MJPEG, fixed focus, ~49° horizontal, f ≈ 1400 px |
| wide camera | SVPRO IMX335 5 MP, 120° M12 | 2592×1944 **30 fps**, rolling shutter, USB 2 MJPEG, ~105° horizontal, f ≈ 990 px |
| laser | **532 nm green, 5 mW**, Class 3R | on the payload between the cameras |
| IMU | GY-85 (ITG3205 + ADXL345 + HMC5883L) | I2C on J11: 0x68 / 0x53 / 0x1E |
| target | black quadcopter, ~283 mm diagonal, ~90 mm body, held by hand | |

At 3 m the C270 puts **~132 px on the drone, ~37 px on the battery.** That is
enough. Keep native resolution — do not downscale the narrow camera, ever.

**Offsets — measure and put in config:**

```
LASER_TO_NARROW_MM    laser aperture centre → C270 lens centre (lateral, and vertical if any)
LASER_TO_WIDE_MM      laser aperture centre → IMX335 lens centre
STEREO_BASELINE_MM    = sum (cameras on opposite sides of the laser)
```

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
{"axes":{"pan":{"enabled":false,"microstep":8,"position":0,"degrees":0.0,
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
  thread capture-narrow   DSHOW, grab loop, newest-wins slot, timestamps
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

```python
cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)     # DSHOW, not MSMF, on Windows
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)        # manual (DSHOW convention)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)               # tune: short, no bloom on the dot
cap.set(cv2.CAP_PROP_GAIN, ...)                  # tune
```

The grab thread runs `cap.grab(); t = perf_counter(); cap.retrieve()` in a
tight loop and publishes `(frame, t)` to the slot. **This is the only way to
defeat the driver's hidden 2–5 frame buffer** — `CAP_PROP_BUFFERSIZE` is not
honoured on Windows. Timestamp at `grab()` return; that is the best available.

Lock exposure and gain. Auto-exposure changes frame timing (unpredictable
latency) and blooms the green dot into a blob.

For the 5 MP wide camera, decode MJPEG on the GPU: `torchvision.io.decode_jpeg`
with `device="cuda"`. CPU decode is 15–30 ms per frame and holds the GIL. If
that is awkward, run the wide camera at 1280×720 — it only finds things.

Cameras on **separate USB root ports**; Pico on a third if possible.

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

**Training data, in order of value:**
1. **200–500 frames of the actual drone, held by a hand, in the actual room.**
   Label with any box tool; bootstrap boxes from COCO weights and correct
   them. This matters more than everything below combined.
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
offsets. Range `R` at runtime from **apparent size**:
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

With motor angles `θ_A, θ_B` and belt reduction `N = 3`:

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
