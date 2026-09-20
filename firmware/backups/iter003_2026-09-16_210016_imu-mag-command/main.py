"""
Entry point.  MicroPython runs main.py automatically at boot.

Boot order matters here: every pin is parked in its safe state BEFORE
anything else happens, so that a reset mid-move cannot leave a driver
enabled or the laser on.
"""

import sys

import config
import pinmap
from machine import Pin


def park_all_pins():
    """
    Force everything to its safe resting state.

    - driver EN high  (active low, so high = coils released)
    - STEP/DIR low
    - all three MOSFET gates low (laser, IR, fan off)
    - driver pad 2 left as a high-Z input until stepper.py claims it,
      so we never drive into an external 3V3 bodge
    """
    for axis in (pinmap.PAN, pinmap.TILT):
        Pin(axis["en"], Pin.OUT, value=1)
        Pin(axis["step"], Pin.OUT, value=0)
        Pin(axis["dir"], Pin.OUT, value=0)
        for k in ("m0", "m1", "m2"):
            Pin(axis[k], Pin.OUT, value=0)
        Pin(axis["fault"], Pin.IN, None)

    for gpio in pinmap.OUTPUTS.values():
        Pin(gpio, Pin.OUT, value=0)

    Pin(pinmap.ULTRASONIC_TRIG, Pin.OUT, value=0)


def main():
    park_all_pins()
    import cli
    cli.main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        park_all_pins()
        print("\ninterrupted -- pins parked safe.  You are at the REPL.")
    except Exception as e:
        park_all_pins()
        sys.print_exception(e)
        print("\npins parked safe.  You are at the REPL.")
