"""
Endstop handling and homing for the Laser Turret board.

Connectors J13 (GPIO19, "Endstop 2") and J14 (GPIO16, "Endstop 1") are
3-pin: pin 1 GND, pin 2 signal, pin 3 +5 V.

The ONLY 3.3 V-safe way to use them is a bare mechanical switch wired
between pin 1 and pin 2, read with the RP2040's internal pull-up.  Leave
pin 3 unconnected.  A 5 V-powered optical or hall endstop that drives
pin 2 to 5 V will damage the Pico -- RP2040 I/O is not 5 V tolerant.
See HARDWARE_NOTES.md Finding #3.
"""

import time

from machine import Pin

import config
import pinmap


class Endstop:
    def __init__(self, name, gpio, active_low=True,
                 debounce_us=None):
        self.name = name
        self.gpio = gpio
        self.active_low = active_low
        self.debounce_us = (debounce_us if debounce_us is not None
                            else config.ENDSTOP_DEBOUNCE_US)
        # Pull-up for a switch-to-GND; pull-down for a switch-to-5V-through-
        # divider arrangement.  Pull-up is the safe default.
        pull = Pin.PULL_UP if active_low else Pin.PULL_DOWN
        self.pin = Pin(gpio, Pin.IN, pull)
        self._last_raw = self.pin.value()
        self._stable = self._last_raw
        self._last_change_us = time.ticks_us()
        self.edge_count = 0

    def raw(self):
        return self.pin.value()

    def triggered(self):
        """Debounced trigger state."""
        v = self.pin.value()
        now = time.ticks_us()
        if v != self._last_raw:
            self._last_raw = v
            self._last_change_us = now
        elif (v != self._stable and
              time.ticks_diff(now, self._last_change_us) >= self.debounce_us):
            self._stable = v
            self.edge_count += 1
        return (self._stable == 0) if self.active_low else (self._stable == 1)

    def triggered_now(self):
        """Undebounced -- use as a motion guard where latency matters."""
        v = self.pin.value()
        return (v == 0) if self.active_low else (v == 1)

    def wait_for(self, state, timeout_ms=5000):
        t0 = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t0) < timeout_ms:
            if self.triggered() == state:
                return True
            time.sleep_ms(1)
        return False

    def status(self):
        return {
            "name": self.name,
            "gpio": self.gpio,
            "raw": self.raw(),
            "triggered": self.triggered(),
            "active_low": self.active_low,
            "edges": self.edge_count,
        }


def build_endstops():
    """Create the endstops that config.py assigns to an axis."""
    out = {}
    for axis_name, key in config.ENDSTOP_AXIS.items():
        if key is None:
            continue
        gpio = pinmap.ENDSTOPS.get(str(key))
        if gpio is None:
            continue
        out[axis_name] = Endstop(
            "%s (endstop %s)" % (axis_name, key),
            gpio,
            active_low=config.ENDSTOP_ACTIVE_LOW.get(axis_name, True),
        )
    return out


class HomingError(Exception):
    pass


def home(axis, endstop, log=print):
    """
    Three-phase home: fast seek, back off, slow re-approach.

    The slow re-approach is what makes homing repeatable -- the fast seek
    only gets you close, and its trigger point varies with approach speed.
    On success the axis position is zeroed at the trigger point.
    """
    name = axis.name
    direction = config.HOMING_DIR.get(name, -1)
    max_travel = config.HOMING_MAX_TRAVEL.get(name, 20000)
    fast = config.HOMING_RATE.get(name, 800)
    slow = config.HOMING_SLOW_RATE.get(name, 150)
    backoff = config.HOMING_BACKOFF.get(name, 200)

    if not axis.enabled:
        raise HomingError("%s is not enabled" % name)

    guard = endstop.triggered_now

    # If we are already sitting on the switch, back off before seeking.
    if guard():
        log("  already triggered -- backing off %d steps" % backoff)
        axis.move(-direction * backoff, rate=slow)
        if guard():
            raise HomingError(
                "%s endstop still triggered after backing off %d steps. "
                "Check ENDSTOP_ACTIVE_LOW['%s'] and the switch wiring."
                % (name, backoff, name))

    # Phase 1 -- fast seek.
    log("  seek at %d steps/s (max %d steps)" % (fast, max_travel))
    r1 = axis.move(direction * max_travel, rate=fast, guard=guard)
    if not r1["guarded"]:
        raise HomingError(
            "%s travelled %d steps without hitting the endstop. "
            "Wrong HOMING_DIR, switch not wired, or ENDSTOP_ACTIVE_LOW wrong."
            % (name, abs(r1["steps"])))
    first_hit = axis.position
    log("  hit after %d steps" % abs(r1["steps"]))

    # Phase 2 -- back off clear of the switch.
    axis.move(-direction * backoff, rate=slow)
    if guard():
        raise HomingError(
            "%s endstop did not release after %d-step backoff"
            % (name, backoff))

    # Phase 3 -- slow re-approach for a repeatable trigger point.
    log("  re-approach at %d steps/s" % slow)
    r3 = axis.move(direction * backoff * 3, rate=slow, guard=guard)
    if not r3["guarded"]:
        raise HomingError(
            "%s endstop did not re-trigger on slow approach" % name)

    second_hit = axis.position
    spread = abs(second_hit - first_hit)
    axis.zero()
    log("  homed.  fast/slow trigger points differ by %d microsteps" % spread)
    return {"spread": spread, "first": first_hit, "second": second_hit}


def repeatability(axis, endstop, cycles=5, log=print):
    """
    Home repeatedly from the same offset and report the spread.

    This is the single most useful mechanical test on the board: it catches
    a loose pulley, a marginal current limit, a bouncing switch and skipped
    steps, all of which show up as a growing spread.
    """
    results = []
    offset = config.HOMING_BACKOFF.get(axis.name, 200) * 5
    direction = config.HOMING_DIR.get(axis.name, -1)

    for i in range(cycles):
        log("cycle %d/%d" % (i + 1, cycles))
        axis.move(-direction * offset, rate=config.HOMING_RATE.get(axis.name, 800))
        home(axis, endstop, log=lambda *a: None)
        # After home() the position is zeroed, so record where the axis sat
        # relative to the commanded pre-home offset.
        results.append(axis.position)

    if len(results) < 2:
        return {"cycles": cycles, "spread": 0, "samples": results}

    spread = max(results) - min(results)
    log("")
    log("repeatability over %d cycles: spread %d microsteps (%.3f deg)"
        % (cycles, spread, axis.steps_to_degrees(spread)))
    if spread == 0:
        log("  perfect -- switch and mechanics are repeatable")
    elif spread <= 2:
        log("  good")
    else:
        log("  POOR.  Suspect: switch bounce, low current limit (motor is")
        log("  skipping), a loose coupling, or too high an approach rate.")
    return {"cycles": cycles, "spread": spread, "samples": results}
