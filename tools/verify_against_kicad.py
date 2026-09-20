"""Audit firmware/current/pinmap.py against the PCB copper.

pinmap.py is the firmware's single source of truth for hardware; this proves it
still matches turret.kicad_pcb. Net *labels* are untrustworthy (Finding #1), so
every check walks pad -> net -> pad and never trusts a name.

Exit code 0 = all checks pass. Run before any deploy.
"""
import os
import sys

BS = chr(92)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCB = os.path.join(ROOT, "turret-2026-03-27_134819", "turret.kicad_pcb")
sys.path.insert(0, os.path.join(ROOT, "firmware", "current"))
import pinmap  # noqa: E402

# Raspberry Pi Pico: physical pad number -> GPIO number.
PICO_PAD_TO_GPIO = {
    1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7, 11: 8, 12: 9,
    14: 10, 15: 11, 16: 12, 17: 13, 19: 14, 20: 15, 21: 16, 22: 17,
    24: 18, 25: 19, 26: 20, 27: 21, 29: 22, 31: 26, 32: 27, 34: 28,
}
GPIO_TO_PICO_PAD = {g: p for p, g in PICO_PAD_TO_GPIO.items()}

# Pololu Breakout-16 carrier pad -> A4988 signal.
A4988_PAD = {
    "vdd": "2", "en": "9", "m0": "10", "m1": "11", "m2": "12",
    "rst": "13", "slp": "14", "step": "15", "dir": "16",
}


# ---------------------------------------------------------------- s-expression
def tokenize(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in "()":
            out.append(c); i += 1
        elif c == '"':
            j, buf = i + 1, []
            while s[j] != '"':
                if s[j] == BS:
                    buf.append(s[j + 1]); j += 2
                else:
                    buf.append(s[j]); j += 1
            out.append(("str", "".join(buf))); i = j + 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < n and not s[j].isspace() and s[j] not in '()"':
                j += 1
            out.append(("sym", s[i:j])); i = j
    return out


def parse(toks):
    stack = [[]]
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            stack[-2].append(stack.pop())
        else:
            stack[-1].append(t[1])
    return stack[0]


def kids(node, name):
    return [x for x in node if isinstance(x, list) and x and x[0] == name]


def load_pcb():
    root = parse(tokenize(open(PCB, encoding="utf-8").read()))[0]
    fps = {}
    for fp in kids(root, "footprint"):
        ref = next((p[2] for p in kids(fp, "property")
                    if len(p) >= 3 and p[1] == "Reference"), None)
        if not ref:
            continue
        pads = {}
        for pad in kids(fp, "pad"):
            net = kids(pad, "net")
            pads[pad[1]] = net[0][2] if net and len(net[0]) >= 3 else None
        fps[ref] = pads
    return fps


# ------------------------------------------------------------------- the audit
class Audit:
    def __init__(self, fps):
        self.fps = fps
        self.passed = 0
        self.failures = []

    def net_of(self, ref, pad):
        return self.fps.get(ref, {}).get(str(pad))

    def check(self, desc, got, want):
        if got == want:
            self.passed += 1
        else:
            self.failures.append("%s\n      expected %r\n      got      %r"
                                 % (desc, want, got))

    def connected(self, desc, ref_a, pad_a, ref_b, pad_b):
        """Assert two pads share a net, whatever that net is called."""
        na, nb = self.net_of(ref_a, pad_a), self.net_of(ref_b, pad_b)
        if na is not None and na == nb:
            self.passed += 1
        else:
            self.failures.append(
                "%s\n      %s pad %s -> %r\n      %s pad %s -> %r"
                % (desc, ref_a, pad_a, na, ref_b, pad_b, nb))

    def gpio_to(self, desc, gpio, ref, pad):
        pico_pad = GPIO_TO_PICO_PAD.get(gpio)
        if pico_pad is None:
            self.failures.append("%s: GPIO%s is not a Pico pad" % (desc, gpio))
            return
        self.connected(desc, "A1", pico_pad, ref, pad)

    def unconnected(self, desc, ref, pad):
        net = self.net_of(ref, pad)
        if net is None or net.startswith("unconnected"):
            self.passed += 1
        else:
            self.failures.append("%s: expected no connection, found %r" % (desc, net))


def main():
    if not os.path.exists(PCB):
        print("PCB not found: %s" % PCB)
        return 2
    a = Audit(load_pcb())

    # --- stepper axes: every logic pin, pad-to-pad -------------------------
    for axis_name, axis in pinmap.AXES.items():
        ref = axis["ref"]
        for sig in ("step", "dir", "en", "m0", "m1", "m2"):
            a.gpio_to("%s.%s GPIO%d -> %s pad %s"
                      % (axis_name, sig, axis[sig], ref, A4988_PAD[sig]),
                      axis[sig], ref, A4988_PAD[sig])
        # Finding #2: pad 2 is VDD on an A4988, driven by the "fault" GPIO.
        a.gpio_to("%s.fault GPIO%d -> %s pad 2 (A4988 VDD)"
                  % (axis_name, axis["fault"], ref), axis["fault"], ref, "2")

    # --- Finding #1: tilt STEP/DIR really are crossed relative to labels ---
    a.check("Finding #1: tilt STEP pad 15 carries the net LABELLED '/Tilt Dir'",
            a.net_of("A2", 15), "/Tilt Dir")
    a.check("Finding #1: tilt DIR pad 16 carries the net LABELLED '/Tilt Step'",
            a.net_of("A2", 16), "/Tilt Step")
    a.check("pan is NOT crossed: pad 15 carries '/Pan Step'",
            a.net_of("A3", 15), "/Pan Step")

    # --- switched outputs: GPIO -> gate resistor -> MOSFET gate -----------
    for name, gpio, res, fet in (("laser", pinmap.LASER, "R1", "Q1"),
                                 ("ir", pinmap.IR_LEDS, "R2", "Q2"),
                                 ("fan", pinmap.FAN, "R7", "Q4")):
        a.gpio_to("%s GPIO%d -> %s (gate resistor)" % (name, gpio, res),
                  gpio, res, "1")
        a.connected("%s: %s -> %s gate (pad 2)" % (name, res, fet), res, "2", fet, "2")

    # --- Finding #7: footprint is D,G,S but a TO-220AB N-FET is G,D,S -----
    a.check("Finding #7: Q1 pad 2 is the GATE net (part expects DRAIN there)",
            a.net_of("Q1", "2"), "Net-(Q1-G)")
    a.check("Finding #7: Q1 pad 1 is the load/drain net",
            a.net_of("Q1", "1"), "/Laser out")

    # --- endstops, I2C, sensors ------------------------------------------
    a.gpio_to("endstop 1 GPIO%d -> J14 pin 2" % pinmap.ENDSTOP_1,
              pinmap.ENDSTOP_1, "J14", "2")
    a.gpio_to("endstop 2 GPIO%d -> J13 pin 2" % pinmap.ENDSTOP_2,
              pinmap.ENDSTOP_2, "J13", "2")
    a.gpio_to("I2C1 SDA GPIO%d -> J11 pin 2" % pinmap.I2C_SDA,
              pinmap.I2C_SDA, "J11", "2")
    a.gpio_to("I2C1 SCL GPIO%d -> J11 pin 1" % pinmap.I2C_SCL,
              pinmap.I2C_SCL, "J11", "1")
    a.gpio_to("ultrasonic TRIG GPIO%d -> J5 pin 3" % pinmap.ULTRASONIC_TRIG,
              pinmap.ULTRASONIC_TRIG, "J5", "3")
    a.gpio_to("ultrasonic ECHO GPIO%d -> J5 pin 2" % pinmap.ULTRASONIC_ECHO,
              pinmap.ULTRASONIC_ECHO, "J5", "2")
    a.gpio_to("analog GPIO%d -> J6 pin 2" % pinmap.ANALOG_SENSOR,
              pinmap.ANALOG_SENSOR, "J6", "2")

    # --- Finding #3: these connectors really do carry 5 V ------------------
    for j, pin in (("J5", "4"), ("J6", "3"), ("J13", "3"), ("J14", "3")):
        a.check("Finding #3: %s pin %s is 5 V" % (j, pin), a.net_of(j, pin), "/5V")

    # --- Finding #5: coil connectors are mirrored --------------------------
    a.check("Finding #5: J2 pin order (tilt)",
            [a.net_of("J2", p) for p in (1, 2, 3, 4)],
            ["Net-(A2-B2)", "Net-(A2-B1)", "Net-(A2-A1)", "Net-(A2-A2)"])
    a.check("Finding #5: J3 pin order (pan) is the mirror",
            [a.net_of("J3", p) for p in (1, 2, 3, 4)],
            ["Net-(A3-A2)", "Net-(A3-A1)", "Net-(A3-B1)", "Net-(A3-B2)"])

    # --- things that must NOT be connected ---------------------------------
    for gpio in pinmap.FREE_GPIO:
        a.unconnected("GPIO%d must be free" % gpio, "A1", GPIO_TO_PICO_PAD[gpio])
    a.unconnected("VSYS must be unconnected (board cannot power the Pico)", "A1", 39)
    a.unconnected("Pico 3V3 out must be unconnected", "A1", 36)

    # --- rst/slp headers exist and are not driven by the board -------------
    a.connected("J4 pin 1 -> A3 RST (pad 13)", "J4", "1", "A3", "13")
    a.connected("J1 pin 1 -> A2 RST (pad 13)", "J1", "1", "A2", "13")

    total = a.passed + len(a.failures)
    print("pinmap.py  <->  %s" % os.path.basename(PCB))
    print("board rev declared in pinmap: %s" % pinmap.BOARD_REV)
    print()
    if a.failures:
        for f in a.failures:
            print("  FAIL  %s" % f)
        print()
    print("  %d/%d checks passed" % (a.passed, total))
    return 1 if a.failures else 0


if __name__ == "__main__":
    sys.exit(main())
