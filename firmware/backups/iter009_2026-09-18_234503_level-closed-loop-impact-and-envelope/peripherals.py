"""
Non-stepper hardware on the Laser Turret board: the three MOSFET-switched
outputs, the ultrasonic connector, the analog sensor input and the I2C bus.

Every input on this board sits behind a connector that also carries 5 V,
with no divider or clamp in between.  RP2040 GPIOs are not 5 V tolerant,
so the read paths here are gated behind explicit config.py acknowledgements
rather than just running and hoping.
"""

import time

from machine import ADC, I2C, PWM, Pin

import config
import pinmap


class SafetyError(Exception):
    pass


# ==========================================================================
#   LOW-SIDE SWITCHED OUTPUTS
# ==========================================================================
class SwitchedOutput:
    """
    GPIO -> 100R -> N-MOSFET gate, with a 100k gate pulldown.

    The load sits between the connector's 5 V pin and the drain, so the
    MOSFET sinks it to ground.  GPIO HIGH = load ON.  The gate pulldown
    means the load is off whenever the Pico is unpowered or the pin is
    floating, which is the right failure mode for a laser.

    THAT IS THE INTENDED CIRCUIT, AND THE BOARD DOES NOT BUILD IT.  Finding #7:
    the footprints are laid out D,G,S while a real TO-220AB N-MOSFET is G,D,S,
    so pads 1 and 2 are swapped.  As fitted, the gate is pulled to +5 V through
    the load (so the FET is permanently on and the load never conducts) and
    driving the GPIO sinks ~33 mA through it -- roughly three times the
    RP2040's maximum drive setting.  The pulldown ends up on the drain, so the
    unpowered-is-off property described above does not hold either.

    Until pins 1 and 2 are crossed on Q1/Q2/Q4, `MOSFET_PINOUT_FIXED` stays
    False and turning an output on raises instead.
    """

    def __init__(self, name, gpio):
        self.name = name
        self.gpio = gpio
        self.pin = Pin(gpio, Pin.OUT, value=0)
        self._on = False

    def on(self, force=False):
        if force:
            # Deliberate override, used to TEST the Finding #7 analysis: if the
            # load lights while the pinout is still unfixed, then the fitted
            # transistor is not a standard G,D,S TO-220 and the finding does
            # not apply to this build. Expect ~20-30 mA out of the GPIO for as
            # long as this is asserted, so use it briefly.
            self.pin.value(1)
            self._on = True
            return
        if not getattr(config, "MOSFET_PINOUT_FIXED", False):
            raise SafetyError(
                "%s is blocked: Q1/Q2/Q4 have gate and drain swapped "
                "(Finding #7). Driving GPIO%d would sink ~33 mA through the "
                "pin and the output cannot work anyway. Cross pins 1 and 2 "
                "on the transistor, then set MOSFET_PINOUT_FIXED = True."
                % (self.name, self.gpio))
        self.pin.value(1)
        self._on = True

    def off(self):
        self.pin.value(0)
        self._on = False

    def set(self, state):
        self.on() if state else self.off()

    def is_on(self):
        return self._on

    def pulse(self, ms):
        self.on()
        time.sleep_ms(int(ms))
        self.off()

    def blink(self, count=5, on_ms=200, off_ms=200):
        for _ in range(count):
            self.on()
            time.sleep_ms(on_ms)
            self.off()
            time.sleep_ms(off_ms)


class Laser(SwitchedOutput):
    """Same hardware as the others, but interlocked."""

    def __init__(self, name, gpio):
        SwitchedOutput.__init__(self, name, gpio)
        self._pwm = None

    def on(self, force=False):
        if not config.LASER_ENABLED:
            raise SafetyError(
                "Laser is interlocked.  Set LASER_ENABLED = True in config.py "
                "only when the beam path is safe and eyes are protected.")
        SwitchedOutput.on(self, force=force)

    def pulse(self, ms):
        ms = int(ms)
        if ms > config.LASER_MAX_ON_MS:
            raise SafetyError(
                "Requested %d ms exceeds LASER_MAX_ON_MS (%d ms)"
                % (ms, config.LASER_MAX_ON_MS))
        SwitchedOutput.pulse(self, ms)

    # ------------------------------------------------------------------
    def pwm_on(self, hz=None, duty_pct=None):
        """
        Free-running pulse train, generated in hardware.

        Hardware PWM rather than a timer loop, for two reasons. The velocity
        loop already owns a 200 Hz timer ISR, and a software blink would jitter
        against it -- which matters here because the point of pulsing is that
        the timing is KNOWN. And PWM keeps running untouched while the CPU is
        busy stepping.

        Why pulse at all: a modulated dot can be pulled out of a cluttered
        scene by differencing frames, which is worth far more than it sounds.
        The narrow camera has had its IR-cut filter removed, so near-IR bleeds
        into every channel and the green-vs-(R,B) chroma test that finds the
        dot is much weaker than it would be on a normal sensor. A dot that is
        present in one frame and absent in the next survives that, because the
        background subtracts away and only the modulated thing remains.

        So the useful frequency is not arbitrary: at 30 fps capture, pulse at
        HALF the frame rate so the dot alternates on/off frame by frame.
        LASER_PULSE_HZ defaults to 15 for that reason.

        Duty below 100 % also cuts average optical power for the same apparent
        brightness, which is a safety gain, not a cost.
        """
        if not config.LASER_ENABLED:
            raise SafetyError(
                "Laser is interlocked.  Set LASER_ENABLED = True in config.py "
                "only when the beam path is safe and eyes are protected.")
        if not getattr(config, "MOSFET_PINOUT_FIXED", False):
            raise SafetyError(
                "laser is blocked: Q1/Q2/Q4 have gate and drain swapped "
                "(Finding #7). Cross pins 1 and 2 on the transistor, then set "
                "MOSFET_PINOUT_FIXED = True.")
        hz = int(hz or config.LASER_PULSE_HZ)
        duty = float(duty_pct if duty_pct is not None
                     else config.LASER_PULSE_DUTY_PCT)
        if hz < 1 or hz > 20000:
            raise SafetyError("pulse rate must be 1..20000 Hz")
        if not (0 < duty <= 100):
            raise SafetyError("duty must be 0..100 %")
        on_ms = duty / 100.0 * 1000.0 / hz
        if on_ms > config.LASER_MAX_ON_MS:
            raise SafetyError(
                "%.1f ms on-time per cycle exceeds LASER_MAX_ON_MS (%d ms)"
                % (on_ms, config.LASER_MAX_ON_MS))
        if self._pwm is None:
            self._pwm = PWM(self.pin)
        self._pwm.freq(hz)
        self._pwm.duty_u16(int(duty / 100.0 * 65535))
        self._on = True
        return hz, duty, on_ms

    def pwm_off(self):
        if self._pwm is not None:
            self._pwm.duty_u16(0)
            self._pwm.deinit()
            self._pwm = None
        # Re-assert the pin as a driven LOW. A deinit()ed PWM leaves the pad
        # in whatever state it stopped in, and a laser is not something to
        # leave to chance.
        self.pin = Pin(self.gpio, Pin.OUT, value=0)
        self._on = False

    def off(self):
        if self._pwm is not None:
            self.pwm_off()
        else:
            SwitchedOutput.off(self)


def build_outputs():
    return {
        "laser": Laser("laser", pinmap.LASER),
        "ir": SwitchedOutput("ir", pinmap.IR_LEDS),
        "fan": SwitchedOutput("fan", pinmap.FAN),
    }


# ==========================================================================
#   ULTRASONIC  (J5: 1 GND, 2 ECHO, 3 TRIG, 4 +5V)
# ==========================================================================
class Ultrasonic:
    """
    HC-SR04-style ranger.

    A stock HC-SR04 runs at 5 V and returns a 5 V ECHO pulse straight into
    GPIO21.  That is out of spec for the RP2040 and will degrade the pad.
    Fix it either way before setting ULTRASONIC_ECHO_IS_3V3_SAFE:
      * divider on ECHO: 1k in series, 2k to GND, at the connector; or
      * fit an HC-SR04P / RCWL-1601 (3.3 V native) and feed J5 pin 4 from
        3V3 rather than the 5 V rail.
    """

    SPEED_OF_SOUND_CM_PER_US = 0.0343

    def __init__(self):
        self.trig = Pin(pinmap.ULTRASONIC_TRIG, Pin.OUT, value=0)
        self.echo = None

    def _arm(self):
        if not config.ULTRASONIC_ECHO_IS_3V3_SAFE:
            raise SafetyError(
                "ECHO on GPIO%d is driven by a 5 V sensor and the RP2040 is "
                "not 5 V tolerant.  Fit a divider or a 3.3 V sensor, then set "
                "ULTRASONIC_ECHO_IS_3V3_SAFE = True in config.py."
                % pinmap.ULTRASONIC_ECHO)
        if self.echo is None:
            self.echo = Pin(pinmap.ULTRASONIC_ECHO, Pin.IN)

    def read_cm(self, timeout_us=30000):
        """Distance in cm, or None on timeout (nothing in range)."""
        self._arm()
        self.trig.value(0)
        time.sleep_us(5)
        self.trig.value(1)
        time.sleep_us(10)
        self.trig.value(0)

        t0 = time.ticks_us()
        while self.echo.value() == 0:
            if time.ticks_diff(time.ticks_us(), t0) > timeout_us:
                return None
        start = time.ticks_us()
        while self.echo.value() == 1:
            if time.ticks_diff(time.ticks_us(), start) > timeout_us:
                return None
        width = time.ticks_diff(time.ticks_us(), start)
        return (width * self.SPEED_OF_SOUND_CM_PER_US) / 2.0

    def read_avg(self, samples=5, gap_ms=60):
        vals = []
        for _ in range(samples):
            v = self.read_cm()
            if v is not None:
                vals.append(v)
            time.sleep_ms(gap_ms)
        if not vals:
            return None
        return sum(vals) / len(vals)


# ==========================================================================
#   ANALOG SENSOR  (J6: 1 GND, 2 signal -> GPIO28/ADC2, 3 +5V)
# ==========================================================================
class AnalogSensor:
    def __init__(self):
        self.adc = None

    def _arm(self):
        if not config.ANALOG_SENSOR_IS_3V3_SAFE:
            raise SafetyError(
                "GPIO%d (ADC%d) has no divider and J6 pin 3 supplies 5 V.  "
                "Confirm your sensor never exceeds 3.3 V, then set "
                "ANALOG_SENSOR_IS_3V3_SAFE = True in config.py."
                % (pinmap.ANALOG_SENSOR, pinmap.ANALOG_ADC_CH))
        if self.adc is None:
            self.adc = ADC(pinmap.ANALOG_ADC_CH)

    def read_raw(self):
        self._arm()
        return self.adc.read_u16()

    def read_volts(self, vref=3.3):
        return self.read_raw() * vref / 65535.0

    def read_avg(self, samples=16):
        self._arm()
        total = 0
        for _ in range(samples):
            total += self.adc.read_u16()
            time.sleep_us(200)
        return total / samples


# ==========================================================================
#   I2C  (J11: 1 SCL -> GPIO27, 2 SDA -> GPIO26)
# ==========================================================================
def i2c_bus(freq=100000):
    """
    There are NO pull-up resistors on this board.  I2C needs them (typically
    4.7k to 3.3 V on each line).  If a scan returns nothing, that is the
    first thing to check -- most breakout modules include their own, but
    a bare sensor will not.
    """
    return I2C(pinmap.I2C_ID,
               scl=Pin(pinmap.I2C_SCL),
               sda=Pin(pinmap.I2C_SDA),
               freq=freq)


def i2c_scan(freq=100000):
    try:
        bus = i2c_bus(freq)
        found = bus.scan()
    except Exception as e:
        return {"ok": False, "error": str(e), "devices": []}
    return {
        "ok": True,
        "devices": [(a, hex(a)) for a in found],
        "count": len(found),
    }


def _pullup_probe(gpio):
    """
    Is this line actively pulled up, or just floating?

    Simply reading an idle input tells you nothing: a floating pad holds
    whatever charge it last had and will happily read either level, so a
    passive read reports 'pull-ups present' one minute and 'no pull-ups'
    the next on the very same board.

    Instead, force the line LOW, release it, and see whether anything pulls
    it back up.  A real bus pull-up (typically 4.7k) restores it in well
    under a microsecond.  A floating line just stays where it was left.
    """
    p = Pin(gpio, Pin.OUT)
    p.value(0)
    time.sleep_us(200)
    Pin(gpio, Pin.IN, None)           # release
    time.sleep_us(500)
    recovered = Pin(gpio, Pin.IN, None).value()
    time.sleep_ms(2)
    recovered_slow = Pin(gpio, Pin.IN, None).value()
    idle = Pin(gpio, Pin.IN, None).value()
    return {"recovered": recovered, "recovered_slow": recovered_slow,
            "idle": idle, "pulled_up": bool(recovered or recovered_slow)}


def i2c_line_check():
    """
    Tell 'no device' apart from 'no pull-ups' on SDA/SCL.

    Uses an active probe rather than a passive read -- see _pullup_probe.
    """
    sda = _pullup_probe(pinmap.I2C_SDA)
    scl = _pullup_probe(pinmap.I2C_SCL)

    if sda["pulled_up"] and scl["pulled_up"]:
        verdict = "both lines recover after being pulled low -- pull-ups present"
    elif not sda["pulled_up"] and not scl["pulled_up"]:
        verdict = ("NEITHER line recovers after being pulled low -- there are "
                   "no pull-up resistors on the bus. This board has none of "
                   "its own; add 4.7k to 3V3 on each line, or use a module "
                   "that includes them.")
    elif not sda["pulled_up"]:
        verdict = "SDA has no pull-up (SCL does) -- check that line"
    else:
        verdict = "SCL has no pull-up (SDA does) -- check that line"

    return {"sda": sda["idle"], "scl": scl["idle"],
            "sda_pulled_up": sda["pulled_up"],
            "scl_pulled_up": scl["pulled_up"],
            "verdict": verdict}


def build_sensors():
    return {"ultrasonic": Ultrasonic(), "analog": AnalogSensor()}
