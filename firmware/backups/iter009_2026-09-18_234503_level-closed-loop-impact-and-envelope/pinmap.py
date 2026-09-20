"""
Pin map for the Laser Turret controller board -- rev "3/27/2026 prod v1".

SINGLE SOURCE OF TRUTH. Every GPIO number below was extracted from the
KiCad schematic AND cross-checked against the copper in turret.kicad_pcb
(pad-to-net mapping), not from the net *labels*.

Board topology
--------------
  A1  Raspberry Pi Pico (RP2040)
  A2  stepper driver, TILT axis   (footprint Module:Pololu_Breakout-16)
  A3  stepper driver, PAN axis    (footprint Module:Pololu_Breakout-16)
  U1  XL1509-5.0 buck -> 5V rail
  Q1/Q2/Q4  low-side N-MOSFETs (laser / IR LEDs / fan)

!! READ HARDWARE_NOTES.md BEFORE POWERING THE BOARD !!
Three findings are encoded in this file; the comments mark each one.
"""

# --------------------------------------------------------------------------
# FINDING #1 -- TILT STEP/DIR NET LABELS ARE SWAPPED IN THE SCHEMATIC
# --------------------------------------------------------------------------
# The schematic net named "/Tilt Step" lands on A2 pad 16, which is the
# driver's DIR input.  The net named "/Tilt Dir" lands on A2 pad 15, which
# is the driver's STEP input.  Verified on the copper:
#
#     A2 pad 15 (STEP) <- net "/Tilt Dir"  <- Pico pad 2  = GPIO1
#     A2 pad 16 (DIR)  <- net "/Tilt Step" <- Pico pad 1  = GPIO0
#
# The BOARD IS FINE -- only the labels lie.  The values below are physical
# truth.  If you ever fix the labels in KiCad, this file does not change.
#
# The PAN axis (A3) is labelled correctly: pad 15 STEP <- GPIO8, pad 16 DIR
# <- GPIO7.  Only tilt is affected.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# FINDING #2 -- DRIVER PAD 2 IS "FAULT" ON A DRV8825 BUT "VDD" ON AN A4988
# --------------------------------------------------------------------------
# The schematic symbol used is Driver_Motor:Pololu_Breakout_DRV8825, so pad 2
# is wired to a Pico GPIO and named "/Pan Fault" / "/Tilt Fault".
#
# On a Pololu A4988 carrier that same physical pad is VDD -- the chip's LOGIC
# SUPPLY INPUT.  A DRV8825 makes its own logic rail from VMOT and therefore
# has a spare pad there; an A4988 does not.
#
# So with A4988s installed, the drivers have NO LOGIC SUPPLY and will not
# respond to STEP/DIR at all.  See HARDWARE_NOTES.md for the recommended
# bodge (pad 2 net -> Pico 3V3, pad 36) and why it must be 3.3 V, not 5 V.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# FINDING #3 -- 5 V SENSOR RETURNS INTO NON-5V-TOLERANT RP2040 GPIOs
# --------------------------------------------------------------------------
# J5 (ultrasonic), J6 (analog sensor) and J13/J14 (endstops) all carry 5 V on
# one pin and route a sensor return straight to a Pico GPIO with no divider
# or clamp.  RP2040 I/O is NOT 5 V tolerant.  The suite refuses to run those
# tests until you acknowledge the wiring -- see peripherals.py.
# --------------------------------------------------------------------------

BOARD_REV = "3/27/2026 prod v1"

# ==========================================================================
#   STEPPER AXES
# ==========================================================================

PAN = {
    "name":  "pan",
    "ref":   "A3",
    # Pico GPIO      driver pad   schematic net
    "step":  8,      # pad 15     /Pan Step      (label correct)
    "dir":   7,      # pad 16     /Pan Dir       (label correct)
    "en":    12,     # pad 9      /Pan En        (active LOW)
    "m0":    11,     # pad 10     /Pan M0        (A4988: MS1)
    "m1":    10,     # pad 11     /Pan M1        (A4988: MS2)
    "m2":    9,      # pad 12     /Pan M2        (A4988: MS3)
    "fault": 13,     # pad 2      /Pan Fault  <-- A4988: this is VDD! Finding #2
    "coils": "J3",   # A3 A2->J3.1  A1->J3.2  B1->J3.3  B2->J3.4
    "rst_slp_header": "J4",   # 2-pin: pad13 RST, pad14 SLP -- needs a shunt
}

TILT = {
    "name":  "tilt",
    "ref":   "A2",
    # Pico GPIO      driver pad   schematic net
    "step":  1,      # pad 15     /Tilt Dir   <-- MISLABELLED, see Finding #1
    "dir":   0,      # pad 16     /Tilt Step  <-- MISLABELLED, see Finding #1
    "en":    5,      # pad 9      /Tilt En       (active LOW)
    "m0":    4,      # pad 10     /Tilt M0       (A4988: MS1)
    "m1":    3,      # pad 11     /Tilt M1       (A4988: MS2)
    "m2":    2,      # pad 12     /Tilt M2       (A4988: MS3)
    "fault": 6,      # pad 2      /Tilt Fault <-- A4988: this is VDD! Finding #2
    "coils": "J2",   # A2 B2->J2.1  B1->J2.2  A1->J2.3  A2->J2.4
    "rst_slp_header": "J1",   # 2-pin: pad13 RST, pad14 SLP -- needs a shunt
}

AXES = {"pan": PAN, "tilt": TILT}

# ==========================================================================
#   FINDING #5 -- THE TWO COIL CONNECTORS ARE WIRED IN OPPOSITE ORDER
# ==========================================================================
# Derived from the copper.  Labels are the A4988 carrier's own silkscreen
# (1A/1B are one coil, 2A/2B are the other):
#
#     J2  TILT :  pin1=2B   pin2=2A   pin3=1A   pin4=1B
#     J3  PAN  :  pin1=1B   pin2=1A   pin3=2A   pin4=2B
#
# That is an exact mirror.  Both connectors do keep each coil on an ADJACENT
# pin pair -- (1,2) and (3,4) -- so a motor loom whose coils are also on
# adjacent pairs will work in both, just turning opposite ways.
#
# The danger is a loom wired INTERLEAVED, i.e. coil pairs on (1,3) and (2,4).
# Plug that in and each driver output pair straddles two different coils:
# the motor cannot make rotating torque, so it buzzes and sits still, and
# the driver sees something close to a short -- which is exactly the
# "buzzes, will not turn, driver gets very hot" failure.
#
# ALWAYS ohm out the loom before plugging it in.  The two ends of one coil
# read a few ohms to each other; wires from different coils read open.
# ==========================================================================

COIL_PINOUT = {
    "tilt": {"connector": "J2", "pins": ("2B", "2A", "1A", "1B")},
    "pan":  {"connector": "J3", "pins": ("1B", "1A", "2A", "2B")},
}


def describe_wiring():
    """Coil connector reference -- the 'wiring' command prints this."""
    out = ["Coil connectors (labels are the A4988 carrier's silkscreen).",
           "1A/1B are ONE coil.  2A/2B are the OTHER coil.",
           ""]
    for axis in ("pan", "tilt"):
        c = COIL_PINOUT[axis]
        out.append("  %-4s %s :  %s" % (
            axis.upper(), c["connector"],
            "   ".join("pin%d=%-3s" % (i + 1, p)
                       for i, p in enumerate(c["pins"]))))
    out += [
        "",
        "  NOTE (Finding #5): those two are in OPPOSITE order.  The same loom",
        "  in both connectors gives the two axes different coil assignment,",
        "  so they will turn opposite ways.  Fix that in software with",
        "  INVERT_DIR in config.py -- do not re-crimp.",
        "",
        "  Each connector keeps a coil on an adjacent pin pair: (1,2) is one",
        "  coil, (3,4) is the other.  If your motor loom is INTERLEAVED",
        "  instead -- coils on (1,3) and (2,4) -- the driver drives across",
        "  two different coils.  Symptom: buzzes, will not rotate, driver",
        "  gets very hot.  That is a wiring fault, not a firmware one.",
        "",
        "  CHECK WITH A MULTIMETER, motor unplugged from the board:",
        "    the two wires of one coil    -> a few ohms (typically 1-5)",
        "    wires from different coils   -> open circuit",
        "  Whichever two wires read a few ohms MUST land on pins 1&2",
        "  together, and the other pair on pins 3&4.",
        "",
        "  ---- THE LOOM SHIPPED WITH THIS BUILD (M20-1060400) ----",
        "  As supplied its 4-pin end is INTERLEAVED and must be re-pinned:",
        "",
        "      pin1 BLK A+ \\__ coil A        pin2 GRN B+ \\__ coil B",
        "      pin3 BLU A- /                 pin4 RED B- /",
        "",
        "  Coils land on (1,3) and (2,4), so each driver output pair gets one",
        "  wire from each coil.  That is the buzz-and-overheat fault above.",
        "",
        "  FIX: swap pins 2 and 3 -- move GRN and BLU.  Nothing else.",
        "       corrected order:  pin1 BLK   pin2 BLU   pin3 GRN   pin4 RED",
        "                         \\___coil A___/         \\___coil B___/",
        "",
        "  That one swap is correct for BOTH J2 and J3.  The axes will then",
        "  turn opposite ways (Finding #5) -- fix that with INVERT_DIR, not",
        "  by re-pinning one of them differently.",
        "",
        "  Verify before plugging in:  BLK-BLU a few ohms, GRN-RED a few ohms,",
        "  BLK-GRN open.",
    ]
    return "\n".join(out)


# ==========================================================================
#   ENDSTOPS  (J13 / J14, 3-pin: pin1 GND, pin2 signal, pin3 +5V)
# ==========================================================================
# The schematic does not say which axis each belongs to -- assign in config.py.

ENDSTOP_1 = 16   # J14 pin 2   net "/Endstop 1"
ENDSTOP_2 = 19   # J13 pin 2   net "/Endstop 2"

ENDSTOPS = {"1": ENDSTOP_1, "2": ENDSTOP_2}

# ==========================================================================
#   LOW-SIDE SWITCHED OUTPUTS  (GPIO -> 100R -> MOSFET gate, 100k pulldown)
# ==========================================================================
# Load connects between the connector's 5 V pin and the MOSFET drain.
# GPIO HIGH = load ON.  The 100k gate pulldown means a floating GPIO is safe.

LASER   = 17   # Q1 gate via R1 100R, pulldown R4 100k -> load on J10
IR_LEDS = 14   # Q2 gate via R2 100R, pulldown R3 100k -> load on J9
FAN     = 15   # Q4 gate via R7 100R, pulldown R8 100k -> load on J7

OUTPUTS = {"laser": LASER, "ir": IR_LEDS, "fan": FAN}

# ==========================================================================
#   SENSORS
# ==========================================================================

ULTRASONIC_TRIG = 20   # J5 pin 3   (output to sensor -- safe)
ULTRASONIC_ECHO = 21   # J5 pin 2   (INPUT -- 5 V on HC-SR04!  Finding #3)

I2C_SDA = 26   # J11 pin 2   -> I2C1 SDA
I2C_SCL = 27   # J11 pin 1   -> I2C1 SCL
I2C_ID  = 1    # GPIO26/27 are the I2C1 peripheral on RP2040

# The ultrasonic connector doubles as a SECOND, independent I2C bus.
#
# Verified against the RP2040 datasheet, Table 2 (GPIO Bank 0 Functions):
#   GPIO20  F3 = I2C0 SDA        GPIO21  F3 = I2C0 SCL
# so J5's trig/echo pair is a genuine I2C0 bus, on a different peripheral
# instance from J11's I2C1. Both can run at the same time.
#
# WARNING -- J5 pin 4 is on the 5 V rail (Finding #3). I2C idles HIGH through
# its pull-ups, so a module whose pull-ups tie to VCC_IN rather than to its own
# regulated 3V3 will hold BOTH GPIO20 and GPIO21 at 5 V continuously. RP2040
# absolute maximum is IOVDD + 0.5 V = 3.8 V. That is not a transient overshoot,
# it is a permanent over-voltage on two pins. Measure SDA/SCL at idle before
# connecting the Pico.
I2C0_SDA = 20  # J5 pin 3 (ultrasonic TRIG)  -> I2C0 SDA
I2C0_SCL = 21  # J5 pin 2 (ultrasonic ECHO)  -> I2C0 SCL
I2C0_ID  = 0
# NOTE: no bus pull-up resistors on this board.  Your module must have them.

ANALOG_SENSOR = 28   # J6 pin 2 -> GPIO28 / ADC2   (J6 pin 3 is 5 V)
ANALOG_ADC_CH = 2

# ==========================================================================
#   UNUSED / AVAILABLE
# ==========================================================================
FREE_GPIO = (18, 22)   # Pico pads 24 and 29, routed to nothing

# Pico pads left unconnected by the board:
#   36 3V3_OUT, 37 3V3_EN, 35 ADC_VREF, 30 RUN, 39 VSYS, 40 VBUS
#
# VSYS UNCONNECTED means the board's 5 V rail does NOT power the Pico.
# The Pico is USB-powered only.  That is fine for bench testing (this suite
# talks over USB anyway) but the board cannot run standalone as built.
# GND is common between USB and the board, so mixed powering is safe.

PICO_SELF_POWERED_FROM_BOARD = False


def describe():
    """Human-readable dump of the map -- used by the CLI 'pins' command."""
    lines = ["Laser Turret board rev %s" % BOARD_REV, ""]
    for ax in (PAN, TILT):
        lines.append("%s axis (%s):" % (ax["name"].upper(), ax["ref"]))
        for k in ("step", "dir", "en", "m0", "m1", "m2", "fault"):
            note = ""
            if ax is TILT and k in ("step", "dir"):
                note = "   <- schematic label is swapped (Finding #1)"
            if k == "fault":
                note = "   <- VDD if an A4988 is fitted (Finding #2)"
            lines.append("   %-6s GPIO%-3d%s" % (k.upper(), ax[k], note))
        lines.append("   coils on %s, RST/SLP shunt header %s"
                     % (ax["coils"], ax["rst_slp_header"]))
        lines.append("")
    lines.append("Endstop 1 GPIO%d (J14)    Endstop 2 GPIO%d (J13)"
                 % (ENDSTOP_1, ENDSTOP_2))
    lines.append("Laser GPIO%d   IR GPIO%d   Fan GPIO%d"
                 % (LASER, IR_LEDS, FAN))
    lines.append("Ultrasonic trig GPIO%d echo GPIO%d (echo is 5 V! Finding #3)"
                 % (ULTRASONIC_TRIG, ULTRASONIC_ECHO))
    lines.append("I2C%d sda GPIO%d scl GPIO%d (no on-board pull-ups)"
                 % (I2C_ID, I2C_SDA, I2C_SCL))
    lines.append("Analog sensor GPIO%d (ADC%d)" % (ANALOG_SENSOR, ANALOG_ADC_CH))
    lines.append("Free: GPIO%d, GPIO%d" % FREE_GPIO)
    return "\n".join(lines)
