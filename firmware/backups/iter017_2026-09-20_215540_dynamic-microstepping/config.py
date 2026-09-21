"""
User-editable configuration for the turret test suite.

Everything you are likely to want to change lives here.  pinmap.py is
hardware truth and should not need editing.
"""

# ==========================================================================
#   WHICH DRIVER CHIP IS ACTUALLY SOLDERED/SOCKETED IN?
# ==========================================================================
# "A4988"   -- Allegro A4988 carrier.  Pad 2 is VDD (logic supply input).
# "DRV8825" -- TI DRV8825 carrier.     Pad 2 is nFAULT (open-drain output).
#
# This changes the microstep truth table AND how pad 2 is treated.  Getting
# it wrong at 1/16 microstepping gives silently wrong step counts, because
# A4988 wants M=111 and DRV8825 wants M=001 for 1/16.
DRIVER_TYPE = "A4988"

# How to treat driver pad 2 when DRIVER_TYPE == "A4988".
#   "external"  -- (RECOMMENDED) you bodged the pad-2 net to Pico 3V3.
#                  The suite leaves the GPIO as a high-Z input and never
#                  drives it.  Safe.
#   "gpio"      -- EXPERIMENTAL.  Drive the GPIO high at 12 mA drive strength
#                  and let it act as the A4988's logic supply.  Marginal but
#                  usually works (A4988 I_DD is ~5 mA typ, 10 mA max).  The
#                  driver is unpowered until firmware boots, so never leave
#                  VMOT applied across a Pico reset in this mode.
#   "off"       -- leave the pin alone entirely; drivers will not work.
A4988_VDD_SOURCE = "gpio"

# Where the RESET+SLEEP jumper wire is tied.  J1 (tilt) and J4 (pan) bring
# RST and SLP out; both chips need them high to run, so they get shorted
# together and pulled up.
#
#   "3V3"  -- (REQUIRED for A4988) tied to 3.3 V.  Works for both chips.
#   "5V"   -- tied to the 5 V rail.  Fine for a DRV8825 (5.75 V abs max on
#             digital pins), but it VIOLATES the A4988's absolute maximum:
#             A4988 logic inputs are rated -0.3 V to VDD + 0.3 V, so at
#             VDD = 3.3 V the ceiling is 3.6 V and 5 V is 1.4 V over.  See
#             HARDWARE_NOTES.md Finding #4 -- it back-drives the Pico.
#   "none" -- no jumper.  Drivers stay in reset and will not step.
#
# 2026-09-16: this is ACTUALLY WIRED TO 5 V, not 3V3.  The value said "3V3"
# and AGENT_HANDOFF Finding #4 claimed it had been reworked; neither was true,
# so `state` was reporting the wrong thing.  Corrected to match the hardware.
# The builder has run these specific drivers at 5 V repeatedly without failure
# and accepts the out-of-spec condition; this field is informational only and
# locks nothing out.
RST_SLP_TIED_TO = "5V"

# ==========================================================================
#   MOTOR / MECHANICS
# ==========================================================================
FULL_STEPS_PER_REV = 200        # 1.8 deg NEMA 17.  Use 400 for a 0.9 deg motor.

# --------------------------------------------------------------------------
#   DIFFERENTIAL WRIST
# --------------------------------------------------------------------------
# Two steppers drive the input bevel gears of a differential through timing
# belts.  The payload (cameras + laser) rides on the output gear.
#
#   both motors the SAME direction     -> carrier rotates -> PITCH (up/down)
#   both motors OPPOSITE directions    -> output spins    -> YAW   (left/right)
#   one motor alone                    -> half of each, simultaneously
#
# So neither motor is a "pitch motor" or a "yaw motor".  The firmware still
# calls the two axes "pan" and "tilt" for historical reasons -- read those as
# MOTOR A and MOTOR B.  kinematics.py converts between the two worlds.
BELT_TEETH_MOTOR = 26           # pulley on the stepper shaft
BELT_TEETH_MITER = 60           # pulley on the miter/input gear
BELT_RATIO = BELT_TEETH_MITER / BELT_TEETH_MOTOR   # 60/26 = 2.3077 : 1
MITER_RATIO = 1.0               # bevel gears are 1:1

# Combined reduction from motor shaft to miter input.
DIFFERENTIAL_N = BELT_RATIO * MITER_RATIO

# Sign of each motor relative to the ideal kinematics.  Flip if a commanded
# +pitch drives the payload down, or +yaw drives it right instead of left.
# NOTE finding #5: J2 and J3 have mirrored pin orders, so the two motors turn
# opposite ways for the same electrical direction.  These constants absorb it
# -- do NOT re-pin a connector to "fix" direction.
MOTOR_SIGN = {"pan": 1, "tilt": 1}      # pan = MOTOR A, tilt = MOTOR B

# Which way is positive, for the operator's sanity.
#   +pitch = payload nose UP        +yaw = payload turns LEFT (CCW from above)
# Flip these after the first `dchar` run if reality disagrees.
#
# pitch flipped 2026-09-15 against the hardware: a commanded +pitch drove the
# payload DOWN, the opposite of the 3D viewer and of the convention above.
# Verified by driving the real turret, not by reasoning -- which is the only
# way a sign convention can be settled.
PAYLOAD_SIGN = {"pitch": -1, "yaw": 1}

# Gear reduction from motor shaft to the moving axis (1.0 = direct drive).
# Used by the single-axis `move`/`deg` commands, which address MOTORS, not
# payload axes.  Leave at 1.0: the differential reduction lives in
# DIFFERENTIAL_N and is applied by kinematics.py.
GEAR_RATIO = {"pan": 1.0, "tilt": 1.0}

# Default microstepping applied at boot.  1, 2, 4, 8, 16 (32 = DRV8825 only).
DEFAULT_MICROSTEP = {"pan": 16, "tilt": 16}

# Direction sense.  Flip to True if an axis moves the wrong way.  Applied in
# software so you never have to re-crimp a coil connector.
INVERT_DIR = {"pan": False, "tilt": False}

# ==========================================================================
#   MOTION LIMITS  (all in MICROSTEPS PER SECOND at the current microstepping)
# ==========================================================================
START_RATE   = {"pan": 200,   "tilt": 200}     # instantaneous-start rate
MAX_RATE     = {"pan": 4000,  "tilt": 4000}    # cruise ceiling
ACCEL        = {"pan": 20000, "tilt": 20000}   # microsteps / s^2

# Hard ceiling the suite will not exceed no matter what you type.  The PIO
# step generator is good to ~90 kHz but MicroPython cannot feed it that fast
# without FIFO underruns.
ABSOLUTE_MAX_RATE = 40000

# ==========================================================================
#   PAYLOAD TRAVEL LIMITS  (degrees, in the payload frame, relative to zero)
# ==========================================================================
# These are PAYLOAD axes -- pitch and yaw -- not motors.  Neither motor is an
# axis on its own (see kinematics.py), so a limit has to be applied in payload
# space BEFORE the differential transform.  Limiting a motor would not limit
# anything meaningful: one motor alone moves pitch and yaw simultaneously.
#
#   pitch   the carrier/yoke tilting the payload up and down.  Mechanically
#           hard-stopped by the yoke, so this one is real.
#   yaw     the output gear spinning the faceplate.  Nothing blocks it, so it
#           is continuous.
#
# None = continuous, no limit.  (min, max) = clamped, inclusive.
#   Confirmed by the builder 2026-09-16: pitch is +/-90 from LEVEL (establish
#   the datum with `level`, not with wherever dzero was last typed), and yaw is
#   mechanically continuous but limited by the payload wiring loom to +/-225.
PAYLOAD_LIMIT_DEG = {
    "pitch": (-90.0, +90.0),
    "yaw": (-225.0, +225.0),
}

# Where 'home' is, in the same zeroed payload frame.  'sethome' overwrites
# this at runtime; this is only the boot default.
PAYLOAD_HOME_DEG = {"pitch": 0.0, "yaw": 0.0}

# What to do when a commanded move would leave the limits.
#   True  -- clamp to the limit and carry on (right for a tracking loop: a
#            saturated axis should keep tracking the other one, not stop)
#   False -- refuse the whole move and raise
PAYLOAD_CLAMP = True

# ==========================================================================
#   OUTPUT MOSFETS  (Finding #7)
# ==========================================================================
# Q1/Q2/Q4 are laid out D,G,S but a standard TO-220AB N-MOSFET is G,D,S, so
# pads 1 and 2 are swapped.  Fitted as-is, the gate is pulled to +5 V through
# the load (FET permanently on, output never works) and driving the GPIO puts
# ~33 mA through the pin against a 12 mA maximum.
#
# Set True only after crossing pins 1 and 2 on all three transistors.  Until
# then the suite refuses to drive laser / IR / fan, the same way it refuses
# the ultrasonic test until the echo line is made 3.3 V safe.
# 2026-09-16: builder confirms the transistors were reworked and the laser,
# IR and fan outputs all function.  The laser stays separately interlocked by
# LASER_ENABLED below until the face-inhibit path exists and is tested.
MOSFET_PINOUT_FIXED = True

# ==========================================================================
#   IMU  (GY-85: ITG3205 gyro + ADXL345 accel + HMC5883L magnetometer)
# ==========================================================================
# Which connector the IMU is on.
#   "J11" -- the designated I2C header, I2C1 on GPIO26/27. No 5 V pin.
#   "J5"  -- the ultrasonic connector, I2C0 on GPIO20/21. Has a 5 V pin;
#            see the warning in pinmap.py before using it.
IMU_PORT = "J5"

IMU_I2C_FREQ = 400000           # all three parts are good to 400 kHz

# Addresses. The ITG3205 responds at 0x68 or 0x69 depending on its AD0 pin;
# GY-85 boards normally strap it to 0x68.
IMU_ADDR_GYRO = 0x68
IMU_ADDR_ACCEL = 0x53
IMU_ADDR_MAG = 0x1E

# Gyro full scale is fixed at +/-2000 deg/s on the ITG3205, 14.375 LSB per
# deg/s per the datasheet.
IMU_GYRO_LSB_PER_DPS = 14.375

# The magnetometer cannot give usable yaw on this machine -- it sits next to
# two stepper motors whose field swamps the earth's. Read it if you like, but
# nothing should steer by it.
IMU_TRUST_MAG = False

# ==========================================================================
#   ENDSTOPS
# ==========================================================================
# Which endstop connector belongs to which axis.  None = not used.
#   J14 -> "1" (GPIO16),  J13 -> "2" (GPIO19)
ENDSTOP_AXIS = {"pan": "1", "tilt": "2"}

# Switch wiring.  The only 3.3 V-safe option is a bare mechanical switch
# between the connector's pin 1 (GND) and pin 2 (signal), using the RP2040's
# internal pull-up.  Do NOT feed a 5 V-powered optical/hall endstop output
# straight into pin 2 -- see HARDWARE_NOTES.md Finding #3.
#   True  = switch closes to GND, so TRIGGERED reads LOW  (normally-open)
#   False = switch opens on trigger, so TRIGGERED reads HIGH (normally-closed)
ENDSTOP_ACTIVE_LOW = {"pan": True, "tilt": True}

ENDSTOP_DEBOUNCE_US = 3000      # ignore edges closer together than this
HOMING_RATE         = {"pan": 800, "tilt": 800}    # approach rate
HOMING_BACKOFF      = {"pan": 200, "tilt": 200}    # microsteps to back off
HOMING_SLOW_RATE    = {"pan": 150, "tilt": 150}    # re-approach rate
HOMING_DIR          = {"pan": -1, "tilt": -1}      # which way is "home"
HOMING_MAX_TRAVEL   = {"pan": 20000, "tilt": 20000}  # give up after N steps

# ==========================================================================
#   CURRENT LIMIT HELPER  (the 'vref' CLI command)
# ==========================================================================
# Sense resistor fitted on YOUR carrier board.  Check with a magnifier --
# clones vary and this is the #1 cause of a motor that runs hot or stalls.
#   Pololu A4988            0.068
#   Pololu A4988 Black Ed.  0.050
#   Common clones           0.100
#   Pololu DRV8825          0.100
SENSE_RESISTOR = 0.068

MOTOR_RATED_CURRENT = 1.5       # amps per phase, from the NEMA 17 datasheet
CURRENT_SAFETY_FACTOR = 0.7     # start conservative; raise once it runs cool

# ==========================================================================
#   PERIPHERALS -- SAFETY INTERLOCKS
# ==========================================================================
# The suite will not exercise these until you flip the flag, because each one
# can either damage the Pico or point a laser at something.

# Set True only after you have EITHER fitted a divider on the echo line OR
# swapped to a 3.3 V sensor (HC-SR04P / RCWL-1601).  A plain 5 V HC-SR04
# will slowly kill GPIO21.
ULTRASONIC_ECHO_IS_3V3_SAFE = False

# Set True only after confirming your analog sensor's output never exceeds
# 3.3 V.  J6 pin 3 supplies 5 V, so a ratiometric 5 V sensor will exceed it.
ANALOG_SENSOR_IS_3V3_SAFE = False

# Laser interlock.  The suite refuses to fire the laser until this is True.
#
# 2026-09-16: ARMED at the builder's direction, for the static goal-pixel (g)
# measurement -- turret stationary, beam on a wall, nobody downrange.  5 mW
# 532 nm Class 3R: the blink reflex covers an accidental glance.
#
# NOTE this is ahead of AGENT_HANDOFF 1.6 rule 3, which keeps it False until
# the face interlock has been TESTED.  That interlock (YuNet, turret_host) is
# still being written.  Arming for a stationary wall measurement is a different
# risk from arming a beam that follows a moving target next to someone's face --
# do NOT leave this True for tracking until the face inhibit has been proven.
LASER_ENABLED = True
LASER_MAX_ON_MS = 2000          # hard cap on a single laser pulse

# Free-running pulse train ('laser pwm').
#
# 15 Hz is not arbitrary: it is HALF the cameras' 30 fps, so the dot is present
# in one frame and absent in the next. Differencing consecutive frames then
# cancels the entire static scene and leaves only the modulated dot -- which
# matters because the narrow camera has had its IR-cut filter removed, so the
# green-vs-(R,B) chroma test that would normally find the dot is much weaker.
# Change this only together with the capture frame rate.
LASER_PULSE_HZ = 15
LASER_PULSE_DUTY_PCT = 50       # also halves average optical power

# ==========================================================================
#   MISC
# ==========================================================================
PIO_CLOCK_HZ = 1_000_000        # 1 us step-timing resolution

# Velocity mode runs its own state machine, faster, so the period quantisation
# does not coarsen the rate at the low end.  At 5 MHz a 300 steps/s rate lands
# within 0.006 % of the request, so no fractional accumulator is needed.
VEL_PIO_CLOCK_HZ = 5_000_000
VEL_SM_OFFSET = 1               # vel SM id = move SM id + this (0->1, 4->5)
VEL_CYCLE_OVERHEAD = 15         # fixed cycles per step in _vel_program

# Velocity-mode control loop.
VEL_TICK_MS = 5                 # 200 Hz
# If no `vel` arrives within this long, the rate is ramped to zero and the
# laser is cut.  A dropped USB link in velocity mode otherwise means a motor
# that runs until something physical stops it.
# Must be comfortably LONGER than one host command round trip, or the motors
# park between every update and the turret steps instead of tracking.
# Measured round trip through gui_server: ~45 ms, so 400 ms is ~9x margin.
# The exposure this buys is bounded elsewhere: the pitch travel limit is
# enforced every tick, and the host has its own stale-feedback guard.
VEL_WATCHDOG_MS = 400
# Largest rate `vel` will accept, microsteps/s per motor.  Below
# ABSOLUTE_MAX_RATE because velocity mode holds a rate indefinitely rather
# than for the length of one move.
VEL_MAX_RATE = 12000

# Acceleration used by the velocity loop, microsteps/s^2 per motor.
#
# Separate from ACCEL because the two modes want different things. A move
# ramps once and the ramp is a small share of a long travel, so a gentle
# figure costs little. The velocity loop re-accelerates on every correction,
# so this number IS the response time for anything but small errors: at 20000 the
# payload takes ~150 ms to reach 150 deg/s, which the operator feels as lag
# no matter how the gain is tuned.
#
# Raise it until the motors start skipping steps, then back off. Nothing here
# measures real position, so a skip is silent -- check with `accuracy` (an
# out-and-back that should return to zero) after changing it.
VEL_ACCEL = 40000

# --------------------------------------------------------------------------
#   DYNAMIC MICROSTEPPING  ("gearshift")
# --------------------------------------------------------------------------
# Coarse microstepping at speed, fine microstepping when precise.
#
# Fine steps are smooth and high-resolution but each one carries little
# incremental torque, and that torque falls away as the step rate climbs.
# Coarse steps hold torque better at speed and give more angular acceleration
# for the same steps/s^2, at the cost of resolution and smoothness -- which is
# a trade you can afford while slewing and cannot afford while settling.
#
# Thresholds are in PAYLOAD deg/s so they mean something physical. Read as:
# "up to this speed, use this divisor". Must be ordered fastest-last, and
# every divisor must exist in the driver's table (A4988 has no 1/32).
VEL_GEARS = (
    (25.0, 16),     # settling and fine tracking: full resolution
    (80.0, 8),
    (999.0, 4),     # slewing: torque matters, resolution does not
)

# Hysteresis, as a fraction of the threshold. Without it the divisor chatters
# at the boundary, and every change is a brief pause plus a driver-indexer
# settle -- so a target sitting exactly on a threshold would stutter forever.
VEL_GEAR_HYSTERESIS = 0.25

# Set True to enable shifting; False pins the divisor.
#
# DEFAULT OFF, deliberately. On this machine it caused oscillation and did not
# buy what it normally buys: dynamic microstepping earns its keep when the
# STEP RATE is the constraint, and here it is not -- in velocity mode the PIO
# free-runs, so 1/16 already reaches ~585 deg/s of payload against a platform
# that tops out near 195. Every shift costs a brief halt in pulsing plus a
# driver-indexer settle, and at a few shifts a second that disturbance is
# real while the benefit is theoretical.
#
# The mechanism is correct and tested (alignment against the driver's own
# translator phase, per the A4988 datasheet); it is simply not worth its cost
# on this drivetrain. Turn it on only if a measurement shows step rate is
# actually limiting something.
VEL_GEARSHIFT = False
VEL_CANONICAL16 = False     # microprofile dynamic enables the vel16 protocol
VEL_DYNAMIC_PULSE_LIMIT = 1000  # overwritten by the explicit host profile
AUTO_DISABLE_ON_IDLE_MS = 0     # 0 = never; else de-energise after idle
VERBOSE = True

# ==========================================================================
#   FAN  (Q4, GPIO15, low-side switched -- see pinmap.OUTPUTS)
# ==========================================================================
# Run the fan whenever either axis is ENERGISED, and stop it after this long
# idle. Tied to energised rather than to velocity mode because position moves
# dissipate too: homing is ~40 s of near-continuous motion, and `level` can
# iterate for up to 240 s.
#
# Until 2026-09-20 nothing turned this output on at any point -- boot parks it
# low, the only control was the manual `fan on` command, and no host code
# referenced it -- so `outputs.fan` reads false in every run on record.
#
# WHETHER THE FAN IS PHYSICALLY ON THE DRIVER HEATSINKS IS NOT ESTABLISHED.
# The docs list it as one of three switched outputs and never say what it
# cools. If it is not on the drivers, this is a comfort feature and settles
# nothing thermal.
FAN_WITH_MOTORS = True
FAN_IDLE_OFF_MS = 30000
