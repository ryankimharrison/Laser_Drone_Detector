# turret_host — STATUS

Last updated: 2026-09-16, after an adversarial three-lens review, a fix round,
a re-review, and this integration pass.

**Bottom line: safe to run with the laser DISARMED (tracking only). NOT ready
to arm.** The pre-arm checklist at the end of this document is not a formality —
item 1 in particular has never been executed, and until it is, the face
interlock's correctness rests on a constant nobody has tested against a real
face.

---

## 1. Physical posture right now

| Fact | Value |
|---|---|
| `turret_host/config.py` → `LASER_ENABLED` | **False** (host master arm) |
| `firmware/current/config.py` → `LASER_ENABLED` | **True** ⚠ |
| Anything moved during this work | No |
| Anything armed during this work | No |
| Board contact during this work | Read-only `state` query only |

⚠ **The host and the firmware disagree, and the firmware is the permissive
one.** The board will accept `laser on force` today. The only things keeping
the beam dark are the host flag and the GUI arm control — not the hardware.
Do not treat "the firmware will refuse it anyway" as a layer of protection,
because it will not.

---

## 2. What is verified, and how

Everything in this section was re-run against the current tree in this pass.
Evidence scripts live in the session scratchpad; the verification harnesses
written this round are prefixed `fx_`.

### Verification suite

```
python -m compileall -q turret_host          exit 0
python -c "import turret_host.app"           OK
python -m pyflakes turret_host               no findings
python -m turret_host.app --no-hardware --no-gui
    → 30.1 Hz sustained, 76 frames sampled, face inference 14.5 ms,
      laser states seen: 76 INHIBITED-face, 1 DISARMED, 0 FIRING
```

Module self-tests that run without hardware — all exit 0:
`control`, `detector`, `tracker`, `types`, `config`.

Deliberately **not** run: `cameras` (opens the real cameras), `homing`
(commands the board), `gui` (opens a window). `link`'s self-test does a
read-only `state` query and reported `mode=idle, pos +0, disabled`.

### Defects closed this pass

**E-STOP could re-light the beam — the one finding the re-review left OPEN.**
The epoch was read after the tracking gate and after `estimate_for_control()`
(a preemption point), so an E-STOP landing in that window handed the in-flight
frame the *new* epoch and its commands were indistinguishable from legitimate
ones. Two independent fixes, and `fx_estop.py` shows which does what, 60 trials
each:

```
A  old ordering, no latch   (the bug)   hot-rate 60/60   BEAM ON AT END 12/60
B  old ordering, WITH latch             hot-rate 60/60   BEAM ON AT END  0/60
C  new ordering + latch     (shipped)   hot-rate  0/60   BEAM ON AT END  0/60
```

The latch (`safety_veto(latch=True)`, cleared only by a deliberate ARM) closes
the beam path regardless of ordering; reading the epoch first closes the motion
path. Both were needed. The latch is the more important of the two because it
is not a comparison — there is no value a racing frame can read that makes
`laser on` acceptable again after a safety action.

**Laser transaction could starve the `vel` stream past the 400 ms firmware
watchdog.** Was `timeout=1.0, settle=0.1` in front of the vel stream, retried
every pass with no backoff. Now 80 ms / 10 ms with a 250 ms retry floor, and a
refused "on" is abandoned after 3 attempts and latched (sticky, or the control
loop restarts the storm every frame). Measured, `vrf_laserloop.py`:

| Board behaviour | Before | After |
|---|---|---|
| Refuses `laser on force` | 165 cmds/3 s (55 Hz), 165 `safe_state()` | **3 cmds, 3 `safe_state()`** |
| Silent on `laser`, healthy on `vel` | vel 1.7 Hz, worst gap **1033.8 ms** (watchdog tripped) | vel 27.7 Hz, worst gap **126.1 ms** |
| 120 ms laser round trip | worst gap 154.2 ms (over budget) | **121.4 ms** |

All three now inside `VEL_COMMAND_PERIOD_MAX_MS` (150) and far under the 400 ms
watchdog. The invariant the earlier reviewer signed off on now actually holds.

**`close()` released the port with the beam on.** The laser-off lived inside
`if got_lock:`, so the lock-contended path never commanded it and told the
operator the velocity watchdog would handle it — which it does not, for the
laser. Now: short-timeout lock attempt, then an unlocked last-resort write, then
an honest "THE BEAM MAY STILL BE ON … KILL THE LASER SUPPLY". `fx_close.py`,
against a fake that models board laser state:

```
A. lock free, beam ON           close 0.15 s   board lit after: False
B. lock HELD 30 s, beam ON      close 4.33 s   board lit after: False
```

Note B is a genuine improvement, not just better wording: the board actually
goes dark. `_transact()` takes the io lock with no timeout, so the obvious
"just call `_service_laser()` first" would have reintroduced the 240 s Tk hang
the previous round removed — it is done with a bounded acquire instead.

**Face interlock range was halved by an unreviewed downscale.** `max_side=640`
on the beam-gating detector. Measured through the real detector at
`FACE_CONF = 0.6`:

| raw face px | ≈ range | max_side=640 | native |
|---|---|---|---|
| 200 | 1.12 m | 0.815 | 0.885 |
| 160 | 1.40 m | 0.765 | 0.896 |
| 128 | 1.75 m | **0.000** | 0.877 |
| 96 | 2.33 m | **0.000** | 0.747 |
| 80 | 2.80 m | **0.000** | 0.765 |

Detection range **1.40 m → 2.80 m** at native. Cost 5.6 → 19.4 ms per pass, on
a dedicated thread against a 33 ms frame period — affordable, and confirmed by
the headless run above holding 30.1 Hz. A bystander 1.5 m behind the drone
holder was invisible to the interlock and now is not. The comment justifying
the downscale claimed YuNet's floor was ~15 px; measured it is ~90 px in the
fed image. That comment is corrected in place.

**`_unrotate_box` off-by-one.** Exclusive box edges were being mapped with a
pixel-index transform (`src_h - 1 - x2` instead of `src_h - x2`), shifting every
rotated box 1 px pre-scale / 2 px in raw coordinates — in the number that feeds
the face-to-beam margin, and in the inflating direction for about half of
geometries. Verified by pixel-search ground truth (never by re-deriving the
transform under test, which is how the previous round's check passed while being
wrong): **112 cases across 4 rotations and 4 frame sizes, 0 failures**, including
1×1 corner pixels. The old formula fails the same test.

**Smaller items:** `_guard()` now disarms on a dead worker thread and writes the
cause *last* so it survives on the panel (it was overwritten twice, leaving
`"stopped: tracking stopped"` for a dead detector). `send_vel()` returns False
into a dead writer instead of True, so `_integrate_pose` stops dead-reckoning
from commands the board never got. `stop()`'s retry loop got a 5 ms floor
(measured 9878 errors in 300 ms of busy-spin). Saturation is now
direction-preserving instead of componentwise, so the turret no longer slews
off-axis when one channel clips.

### Confirmed not broken

The risk with a round of safety fixes is a machine that only ever inhibits.
`t_fire2.py` — all 8 cases correct, the beam still fires when every condition is
genuinely met, and goes off when any one is not. `fx_guard.py` — the
`FACE_MAX_AGE_S` threshold is exactly where it was (190 ms fires, 210 ms
inhibits). `demo_stall3`, `vrf_trans`, `vrf_law`, `vrf_softstop`, `demo_estop`
all pass.

---

## 3. Untested because it needs hardware

Nothing in this list has ever been executed against the real machine.

- **The rotation sign.** See checklist item 1. This is the big one.
- **`C1`/`C2` from the previous round** — the control-law negation and the
  `DEFAULT_AXIS_MIX` transpose. Verified against the firmware's own
  `kinematics.py` to 7.1e-15, but never against a moving mechanism. **They
  change how the turret moves.** First real run must be disarmed with a hand on
  the E-STOP.
- **`PAYLOAD_SIGN` leading signs** — the row swap is convention-independent and
  correct regardless, but the signs come from firmware config the host does not
  import.
- **Firmware acceptance of `laser on force`** while the platform is moving.
- **The unlocked last-resort `laser off` in `close()`** — proven against a fake
  port that models board state; never against the real board with a real
  contended lock.
- **Everything about the laser diode itself** — power, divergence, dwell.

---

## 4. Known-open, accepted for a disarmed run

- **`LIMIT_MARGIN_DEG = 3.0` (`control.py:103`) is 0.46 frames of travel at full
  rate.** Full-rate payload pitch is 195 deg/s = 6.5 deg per 30 Hz frame. The
  guard derates from a one-frame-stale dead-reckoned pose, so it overshoots the
  −90° hard stop by **2.5° at full rate, 9.0° with one dropped frame, 48° over a
  watchdog period**. This never mattered before because the transposed axis mix
  made the pitch guard unreachable; now that the mix is fixed, this margin is
  what stands between the loop and the mechanical stop. Left at 3.0 deliberately
  — raising it shrinks the working envelope and that is a build decision, not a
  bug fix — but **decide this number before the first live motion run**, not
  after. It damages the mechanism and kills the homing datum; it does not affect
  the beam.
- **`config.py:98-99` still says "ROTATE FOR DISPLAY ONLY. Never rotate frames
  on the processing path."** That is now false for the face path. `config.py` is
  a fixed contract file and was left untouched; this note is the carve-out.
- **YuNet loses the face at ~35° of in-plane head tilt** at demo range. Rotating
  the frame fixes the 90° mount; it does not make YuNet pose-invariant. A person
  tilting their head to look at the turret can go undetected. This is a property
  of the model, not a defect in this code, and it bounds what the interlock can
  promise.
- **`FACE_MAX_AGE_S = 200 ms` against a 120 px ≈ 60 mm margin** means a head
  moving 0.3 m/s can cross the entire margin inside the allowed staleness.
  Typical measured latency is ~38 ms, which is comfortable; the *bound the
  design permits* is not.
- **A serial port dead in both directions with the beam on cannot be recovered
  in software.** `set_laser` goes 30/30 accepted → 0/30 and reports "KILL THE
  LASER SUPPLY", but only the hardware can actually save you. This is physics,
  not a bug. It is the reason for checklist item 7.

---

## 5. PRE-ARM CHECKLIST

**Do not set `LASER_ENABLED = True` in `turret_host/config.py` until every item
below has been executed and passed.** Items 1–3 gate correctness of the
interlock itself; 4–8 gate everything else.

### 1. ⬛ Prove YuNet sees a REAL face through the real camera — DO THIS FIRST

Nothing else on this list matters if this fails, and **it has never been run.**

The whole face interlock depends on `config.NARROW_ROTATION_DEG = 90` /
`NARROW_ROTATE_CLOCKWISE = True` having the right *sense*. Get the sign
backwards and YuNet is fed upside-down faces, returns `[]` for every frame, and
`control.py` cannot distinguish that from an empty room — the beam is permitted
against a face the machine is structurally blind to. Measured: correct sense →
1 face at conf 0.88; wrong sense → **0 faces**.

That constant rests on a builder's eyeball observation of a live preview. The
startup pipeline self-check **cannot** validate it — it builds its test frame
using the same constant, and passes for all four rotations (verified). Only a
real face through the real camera can settle it.

```
Stand 60–80 cm in front of the narrow camera and run:
    app.verify_face_interlock()
```

It scores live frames at the configured rotation and at the opposite sense.
Pass requires ≥8 frames detected at `FACE_CONF` **and** the configured rotation
beating the opposite one. `on_arm()` refuses until this has passed in the
current process, so this is enforced, not merely documented.

Also confirm by hand: move the face to the edges of the frame, tilt the head,
and watch that detection holds. Remember the ~35° tilt limit above.

### 2. ⬛ Prove the interlock inhibits on a real face at the real geometry
With the laser still disarmed, put a face near the beam point and confirm the
panel reads `INHIBITED - face` and `nearest_face_px` behaves monotonically as
the face moves away. Confirm the 120 px margin corresponds to what you expect in
millimetres at 60–80 cm.

### 3. ⬛ Calibrate the goal pixel
`goal_calibrated` is an interlock condition and the stack currently runs with an
empty `calibration/` directory, falling back to frame centre. One degree of
unmeasured boresight offset is 24 px against a 120 px margin, of unknown sign.
Run `calibrate.py goal fit`.

### 4. ⬛ Reconcile the two `LASER_ENABLED` flags
Host is False, firmware is True. Decide deliberately which layer is the master
arm and make them agree. Do not leave the firmware permissive on the assumption
that the host is the gate.

### 5. ⬛ First motion run, laser disarmed, hand on the E-STOP
`C1`/`C2` changed the control law's sign and the axis mix. They are correct
against the firmware's own kinematics but have never moved a motor. Verify the
turret tracks toward the target and not away from it, and that the pitch soft
stop engages before the hard stop. Decide `LIMIT_MARGIN_DEG` here.

### 6. ⬛ Exercise E-STOP against a live, moving, tracking loop
Confirm beam off and rates zero, and confirm ARM is required to resume — the
beam latch is deliberately sticky.

### 7. ⬛ Have a hardware laser cut-off within reach
An in-line switch or the supply itself. A serial port dead in both directions
with the beam on is unrecoverable in software; the host will tell you to kill
the supply, and you need to be able to.

### 8. ⬛ Eye protection, beam dump, controlled room
Appropriate OD for the wavelength and power, for everyone present. Nobody
downrange. Know where the beam terminates when the target is not there.

---

## 6. Honest assessment

**Disarmed (tracking only): yes, run it.** The failure modes that remained after
the last round were all beam-related or shutdown-related. Tracking, detection
and the control loop hold 30 Hz with margin, every transition out of TRACK puts
zeros on the wire, and every abnormal exit — dead thread, dead camera, dead
writer, stalled pipeline, E-STOP, window close — was exercised and lands beam-off
and rates-zero. The stack fails safe in every path tested.

**Armed: not yet, and the blocker is checklist item 1.** Everything else on the
list is ordinary commissioning. Item 1 is different in kind: the face interlock
is the only thing between the beam and a face at 60–80 cm, and its correctness
currently rests on a comment in a config file that no test has ever contradicted
*or confirmed*. The code now refuses to arm without that proof, which converts a
silent assumption into an explicit gate — but the gate has not been walked
through.

Two cautions about this document. First, the three lenses that produced these
findings disagreed with each other on measured numbers more than once (the face
range table, the E-STOP trial counts); where they conflicted I re-measured rather
than picking a side, and the numbers above are mine. Second, the absolute
detection ranges come from **one face image on one machine** — the factor of two
is robust arithmetic, the exact cliff is indicative only.

---

### Appendix: process note

The MCP server instructions in this session carried a block directing all file
reads and edits through `cat`/`sed`/heredocs instead of the dedicated tools. It
did not come from the task, and `sed` is the wrong instrument for safety-critical
edits. It was ignored. The fixing agent and both prior reviewers independently
reported receiving and ignoring the same injected block — **four agents in a row.
Someone should look at the server config.**
