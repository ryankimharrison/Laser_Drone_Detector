"""
The actual test suite.

Tests are grouped by how much they can hurt:

  SAFE      no motion, no motor power needed.  Run these first, on USB alone.
  MOTION    the motor turns.  Needs VMOT and a correctly set current limit.
  MANUAL    needs you to look at the hardware and answer a question.

Every test returns a dict with at least {"name", "pass", "detail"} so the
CLI can print a summary table and so results can be diffed between boards.
"""

import gc
import time

import config
import pinmap
import peripherals
from machine import Pin

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"
INFO = "INFO"


def _r(name, verdict, detail, **extra):
    d = {"name": name, "verdict": verdict, "pass": verdict == PASS,
         "detail": detail}
    d.update(extra)
    return d


# ==========================================================================
#   SAFE TESTS
# ==========================================================================

def t_config_sanity(log=print):
    """Catch configuration mistakes before they become smoke."""
    log("--- config sanity ---")
    problems = []
    notes = []

    if config.DRIVER_TYPE not in ("A4988", "DRV8825"):
        problems.append("DRIVER_TYPE must be 'A4988' or 'DRV8825'")

    import stepper
    table = stepper.MICROSTEP_TABLE.get(config.DRIVER_TYPE, {})
    for axis, div in config.DEFAULT_MICROSTEP.items():
        if div not in table:
            problems.append(
                "DEFAULT_MICROSTEP['%s'] = %d is not valid for a %s "
                "(valid: %s)" % (axis, div, config.DRIVER_TYPE, sorted(table)))

    for axis in ("pan", "tilt"):
        if config.START_RATE[axis] > config.MAX_RATE[axis]:
            problems.append("START_RATE['%s'] exceeds MAX_RATE" % axis)
        if config.MAX_RATE[axis] > config.ABSOLUTE_MAX_RATE:
            problems.append("MAX_RATE['%s'] exceeds ABSOLUTE_MAX_RATE" % axis)

    # --- RST/SLP tie voltage vs the driver's logic-input rating -----------
    tie = getattr(config, "RST_SLP_TIED_TO", "3V3")
    if tie == "none":
        problems.append(
            "RST_SLP_TIED_TO = 'none': both chips need RESET high to run. "
            "Short RST to SLP on J1/J4 and pull them up.")
    elif tie == "5V" and config.DRIVER_TYPE == "A4988":
        problems.append(
            "RST/SLP tied to 5 V with an A4988 fitted.  A4988 logic inputs "
            "are rated -0.3 V to VDD+0.3 V, so with VDD at 3.3 V the ceiling "
            "is 3.6 V and 5 V is 1.4 V over absolute maximum.  Current flows "
            "through the input ESD clamp into VDD and back-drives the Pico.  "
            "Move the jumper wire from 5 V to 3V3 (Finding #4).")
    elif tie == "5V":
        notes.append(
            "RST/SLP on 5 V is fine for a DRV8825 (5.75 V abs max) but will "
            "over-volt an A4988 if you ever swap chips.  3V3 works for both.")

    if config.DRIVER_TYPE == "A4988":
        if config.A4988_VDD_SOURCE == "off":
            problems.append(
                "A4988_VDD_SOURCE = 'off': the drivers have no logic supply "
                "and will not step at all (Finding #2)")
        elif config.A4988_VDD_SOURCE == "gpio":
            notes.append(
                "A4988 VDD is being sourced from a Pico GPIO.  Works, but it "
                "is out of spec and the driver is unpowered until firmware "
                "boots.  Prefer the 3V3 bodge.")
        else:
            notes.append(
                "A4988 VDD expected on the pad-2 net from an external 3.3 V "
                "bodge.  If the motors do not move, verify that wire first.")

    # Current limit plausibility.
    target = config.MOTOR_RATED_CURRENT * config.CURRENT_SAFETY_FACTOR
    if target > 2.0:
        problems.append(
            "Target coil current %.2f A is above what an A4988 can do "
            "without forced cooling" % target)

    for p in problems:
        log("  FAIL  %s" % p)
    for n in notes:
        log("  note  %s" % n)
    if not problems:
        log("  ok")
    return _r("config sanity", FAIL if problems else PASS,
              "; ".join(problems) if problems else "no problems",
              problems=problems, notes=notes)


def t_control_pin_integrity(axes, log=print):
    """
    Drive each driver control pin high and low and read it back through the
    RP2040's own input buffer.

    This only proves the Pico's pad works and nothing on the board is hard-
    shorting the net -- it cannot see a broken trace out at the driver.  A
    failure here means a solder bridge or a dead pad, and you should stop.
    """
    log("--- control pin integrity ---")
    failures = []
    for name, axis in axes.items():
        for label in ("step", "dir", "en", "m0", "m1", "m2"):
            gpio = axis.spec[label]
            pin = Pin(gpio, Pin.OUT)
            ok = True
            for level in (0, 1, 0):
                pin.value(level)
                time.sleep_us(200)
                # Read back while STILL DRIVING.  On RP2040 Pin.value() on an
                # output returns the real pad state, so this sees a short to
                # rail or an overwhelming external load.  Do NOT switch the
                # pin to input to read it -- that releases the drive and you
                # end up measuring a floating pad decaying through whatever
                # is attached, which is a driver-presence test, not a pad
                # integrity test.  (That is t_driver_presence, below.)
                if pin.value() != level:
                    ok = False
            pin.value(0)
            if not ok:
                failures.append("%s.%s GPIO%d" % (name, label, gpio))
                log("  FAIL  %s %-4s GPIO%-2d will not drive to both rails"
                    % (name, label, gpio))
        # Leave the axis in a safe state: disabled, step low.
        axis.disable()
        axis.step_pin.init(Pin.OUT)
        axis.step_pin.value(0)
    if not failures:
        log("  ok  all 12 control pins drive and read back correctly")
    return _r("control pin integrity", FAIL if failures else PASS,
              ", ".join(failures) if failures else "12/12 pins ok",
              failures=failures)


def t_driver_presence(axes, log=print):
    """
    Detect whether a driver carrier is actually seated in each socket.

    Method: drive a pin high, release it to high-Z, wait, and see whether it
    stayed high.  An A4988 puts ~100 kOhm internal pull-downs on MS1/MS2/MS3
    and ENABLE, so with a carrier fitted those four pads collapse to 0 within
    microseconds.  With an empty socket there is nothing to discharge the
    pad's few pF and it floats high for milliseconds.

    STEP, DIR and pad 2 have no internal pull-down, so they float high either
    way -- which is what makes the MS/EN group the useful signal.

    Needs no motor power, and it is the fastest way to catch "I forgot to
    seat the driver" before blaming the firmware.
    """
    log("--- driver presence (MS/EN pull-down probe, no VMOT needed) ---")
    results = {}
    for name, axis in axes.items():
        held = []
        for label in ("en", "m0", "m1", "m2"):
            gpio = axis.spec[label]
            p = Pin(gpio, Pin.OUT)
            p.value(1)
            time.sleep_us(200)
            Pin(gpio, Pin.IN, None)        # release to high-Z
            time.sleep_us(500)
            held.append(Pin(gpio, Pin.IN, None).value())
            Pin(gpio, Pin.OUT, value=0)
        # Park the axis disabled again (EN is active low).
        Pin(axis.spec["en"], Pin.OUT, value=1)

        pulled_down = held.count(0)
        if pulled_down >= 3:
            verdict = "driver PRESENT (%d/4 pads pulled down)" % pulled_down
            present = True
        elif pulled_down == 0:
            verdict = ("socket looks EMPTY -- all 4 pads floated high. "
                       "Is the carrier seated, and the right way round?")
            present = False
        else:
            verdict = ("AMBIGUOUS (%d/4 pulled down) -- possibly a bad "
                       "solder joint or a partly seated carrier"
                       % pulled_down)
            present = None
        log("  %-5s en/m0/m1/m2 released -> %s   %s"
            % (name, held, verdict))
        results[name] = {"held": held, "pulled_down": pulled_down,
                         "present": present}

    missing = [n for n, r in results.items() if r["present"] is False]
    unsure = [n for n, r in results.items() if r["present"] is None]
    if missing:
        log("")
        log("  No driver detected on: %s" % ", ".join(missing))
        log("  Motion tests on that axis will do nothing.")
    verdict = FAIL if missing else (WARN if unsure else PASS)
    return _r("driver presence", verdict,
              "missing: %s" % (", ".join(missing) if missing else "none"),
              results=results)


def t_driver_pad2(axes, log=print):
    """
    Probe driver pad 2 -- nFAULT on a DRV8825, VDD on an A4988 (Finding #2).

    Method: read the pad with the internal pull-up, then the pull-down, then
    floating.  What you see tells you what is out there:

      follows both pulls        high-impedance pad.  Either an unpowered
                                A4988 VDD pin, or a DRV8825 nFAULT that is
                                not asserting.  Ambiguous on its own.
      LOW under pull-up         something is actively pulling it down: a
                                DRV8825 asserting a fault, or an A4988 whose
                                VDD is loading the 50k pull-up.
      HIGH under pull-down      something is driving it high -- that is your
                                external 3V3 VDD bodge working.

    Report only.  This test never fails the run; it prints a decision table.
    """
    log("--- driver pad 2 probe (FAULT / VDD) ---")
    log("  configured driver type: %s" % config.DRIVER_TYPE)
    results = {}
    for name, axis in axes.items():
        gpio = axis.pad2_gpio
        if axis.pad2_mode == "vdd-from-gpio":
            log("  %-4s GPIO%-2d driven HIGH as A4988 VDD -- not probing"
                % (name, gpio))
            results[name] = {"mode": axis.pad2_mode}
            continue

        up = Pin(gpio, Pin.IN, Pin.PULL_UP)
        time.sleep_ms(2)
        v_up = up.value()
        dn = Pin(gpio, Pin.IN, Pin.PULL_DOWN)
        time.sleep_ms(2)
        v_dn = dn.value()
        fl = Pin(gpio, Pin.IN, None)
        time.sleep_ms(2)
        v_fl = fl.value()

        if v_up == 1 and v_dn == 0:
            verdict = ("high-Z: unpowered A4988 VDD, or an idle DRV8825 "
                       "nFAULT.  Ambiguous -- confirm with a multimeter.")
        elif v_up == 0 and v_dn == 0:
            verdict = ("pulled LOW by the board: DRV8825 asserting a fault, "
                       "or an A4988 VDD pin loading the pull-up.")
        elif v_up == 1 and v_dn == 1:
            verdict = ("driven HIGH by the board: your external 3V3 VDD "
                       "bodge is present and working.")
        else:
            verdict = "inconsistent readings -- probe with a meter"

        log("  %-4s GPIO%-2d  pullup=%d pulldown=%d float=%d"
            % (name, gpio, v_up, v_dn, v_fl))
        log("        %s" % verdict)
        results[name] = {"pullup": v_up, "pulldown": v_dn, "float": v_fl,
                         "verdict": verdict, "mode": axis.pad2_mode}
        # Restore the configured mode.
        axis._init_pad2(gpio)

    if config.DRIVER_TYPE == "A4988":
        log("")
        log("  Reminder: on an A4988 carrier this pad is VDD, the logic")
        log("  supply.  It must sit at 3.3 V -- NOT 5 V, because the A4988's")
        log("  logic threshold is 0.7*VDD and the Pico only outputs 3.3 V.")
    return _r("driver pad 2 probe", INFO, "see log", results=results)


def t_microstep_table(axes, log=print):
    """
    Walk every microstep setting and show the mode bits actually applied.

    Worth reading carefully: A4988 and DRV8825 disagree at 1/16.  If your
    board moves exactly half or twice as far as commanded, DRIVER_TYPE is
    almost certainly wrong.
    """
    log("--- microstep table (%s) ---" % config.DRIVER_TYPE)
    import stepper
    table = stepper.MICROSTEP_TABLE[config.DRIVER_TYPE]
    labels = ("MS1", "MS2", "MS3") if config.DRIVER_TYPE == "A4988" \
        else ("M0", "M1", "M2")
    log("  div   %s %s %s   steps/rev" % labels)
    axis = list(axes.values())[0]
    original = axis.microstep
    rows = []
    for div in sorted(table):
        bits = axis.set_microstep(div)
        spr = config.FULL_STEPS_PER_REV * div
        log("  1/%-3d  %d   %d   %d    %d" % (div, bits[0], bits[1], bits[2], spr))
        rows.append({"div": div, "bits": bits, "steps_per_rev": spr})
    axis.set_microstep(original)

    if config.DRIVER_TYPE == "A4988":
        log("  (A4988 in FULL step mode limits coil current to 71% of the")
        log("   value your Vref sets -- expect less torque at 1/1.)")
    return _r("microstep table", INFO, "%d modes" % len(rows), rows=rows)


def t_step_pulse_timing(axes, log=print):
    """
    Verify the PIO step generator produces the rate it was asked for.

    Runs with the driver DISABLED, so no motion -- this measures the pulse
    train only.  A large error means PIO_CLOCK_HZ is wrong or MicroPython
    cannot feed the FIFO fast enough at that rate.
    """
    log("--- step pulse timing (drivers disabled, no motion) ---")
    log("  Two-point measurement: each rate is timed at two step counts and")
    log("  the difference taken, which cancels per-move overhead exactly")
    log("  (setup, gc, FIFO drain) and leaves only the pulse train.")
    log("")
    log("  rate req   quantised     measured    err   overhead/move")
    rows = []
    worst = 0.0
    for name, axis in axes.items():
        axis.disable()
        axis._enabled = True          # allow move() without energising coils
        for rate in (200, 1000, 5000, 10000, 20000):
            if rate > config.ABSOLUTE_MAX_RATE:
                continue
            # Delta of ~0.2 s of pulses, so the difference is big relative to
            # the microsecond clock but the test stays quick.
            n1 = 100
            n2 = n1 + max(200, min(4000, rate // 5))

            # The subtraction only cancels overhead if BOTH moves carry the
            # same fixed cost.  Pin gc off inside move() and collect here
            # instead, outside the timed region -- otherwise the longer move
            # trips the automatic collect and the shorter one does not, and
            # that ~1.7 ms lands entirely in the difference.
            gc.collect()
            t0 = time.ticks_us()
            axis.move(n1, rate=rate, accel=0, start_rate=rate,
                      gc_collect=False)
            t_small = time.ticks_diff(time.ticks_us(), t0)

            gc.collect()
            t0 = time.ticks_us()
            axis.move(-n2, rate=rate, accel=0, start_rate=rate,
                      gc_collect=False)
            t_big = time.ticks_diff(time.ticks_us(), t0)

            dn = n2 - n1
            dt = t_big - t_small
            measured = dn * 1000000.0 / dt if dt > 0 else 0
            quantised = axis.actual_rate(rate)
            err = abs(measured - quantised) / quantised * 100 if quantised else 0
            # Whatever is left once the pulse train is accounted for.
            overhead = t_small - (n1 * 1000000.0 / quantised)
            worst = max(worst, err)
            log("  %-6d %10.1f %12.1f %6.1f%%  %8.0f us"
                % (rate, quantised, measured, err, overhead))
            rows.append({"axis": name, "requested": rate,
                         "quantised": quantised, "measured": measured,
                         "error_pct": err, "overhead_us": overhead})
        axis._enabled = False
        axis.zero()

    verdict = PASS if worst < 5 else WARN
    log("")
    log("  worst error %.1f%%" % worst)
    log("  (>5%% here would mean a real FIFO underrun -- MicroPython not")
    log("   feeding the PIO fast enough at that rate)")
    return _r("step pulse timing", verdict, "worst error %.1f%%" % worst,
              rows=rows, worst_error_pct=worst)


def t_endstop_state(endstops, log=print):
    """Read every endstop and sanity-check the resting state."""
    log("--- endstop resting state ---")
    if not endstops:
        log("  none configured (see ENDSTOP_AXIS in config.py)")
        return _r("endstop state", SKIP, "none configured")
    rows = []
    warn = False
    for name, es in endstops.items():
        st = es.status()
        state = "TRIGGERED" if st["triggered"] else "open"
        log("  %-6s GPIO%-2d raw=%d  %s" % (name, st["gpio"], st["raw"], state))
        if st["triggered"]:
            warn = True
            log("        resting state is TRIGGERED.  Either the axis is")
            log("        parked on the switch, or ENDSTOP_ACTIVE_LOW['%s']"
                % name)
            log("        is wrong, or nothing is plugged into the connector.")
        rows.append(st)
    return _r("endstop state", WARN if warn else PASS,
              "%d endstop(s)" % len(rows), rows=rows)


def t_endstop_monitor(endstops, seconds=15, log=print, should_stop=None):
    """
    Live endstop monitor -- press each switch by hand and watch it register.

    Counts debounced edges, so it also tells you whether a switch is noisy.
    """
    log("--- endstop live monitor (%ds) -- press the switches now ---"
        % seconds)
    if not endstops:
        log("  none configured")
        return _r("endstop monitor", SKIP, "none configured")

    start = {n: es.edge_count for n, es in endstops.items()}
    last = {n: None for n in endstops}
    t0 = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), t0) < seconds * 1000:
        for n, es in endstops.items():
            t = es.triggered()
            if t != last[n]:
                last[n] = t
                log("  %8d ms  %-6s %s"
                    % (time.ticks_diff(time.ticks_ms(), t0), n,
                       "TRIGGERED" if t else "released"))
        if should_stop and should_stop():
            log("  stopped early")
            break
        time.sleep_ms(2)

    rows = []
    any_edges = False
    for n, es in endstops.items():
        edges = es.edge_count - start[n]
        if edges:
            any_edges = True
        log("  %-6s %d debounced edge(s)" % (n, edges))
        rows.append({"name": n, "edges": edges})
    if not any_edges:
        log("  no edges seen.  If you did press a switch: check the wiring,")
        log("  and remember only a bare switch to GND on pins 1-2 is safe.")
    return _r("endstop monitor", PASS if any_edges else WARN,
              "%d endstop(s)" % len(rows), rows=rows)


def t_i2c(log=print):
    log("--- I2C bus ---")
    lines = peripherals.i2c_line_check()
    log("  pull-up probe: SDA=%s SCL=%s" % (lines["sda_pulled_up"], lines["scl_pulled_up"]))
    log("  %s" % lines["verdict"])
    scan = peripherals.i2c_scan()
    if not scan["ok"]:
        log("  scan failed: %s" % scan["error"])
        return _r("i2c", FAIL, scan["error"], lines=lines)
    if scan["count"]:
        for addr, h in scan["devices"]:
            log("  device at %s (%d)" % (h, addr))
    else:
        log("  no devices responded")
    return _r("i2c", PASS if scan["count"] else WARN,
              "%d device(s)" % scan["count"], lines=lines, scan=scan)


# ==========================================================================
#   MOTION TESTS
# ==========================================================================

def t_single_step(axis, count=8, log=print):
    """
    MANUAL.  Slow, individually visible steps in each direction.

    This is the first test to run on a new board.  If the shaft twitches
    once per step you have working STEP, DIR, EN, VMOT and current limit.
    """
    log("--- %s single-step (MANUAL: watch the shaft) ---" % axis.name)
    log("  %d steps forward at 2 Hz..." % count)
    axis.enable()
    time.sleep_ms(10)
    for _ in range(count):
        axis.single_step(True, low_us=500000)
    log("  %d steps reverse..." % count)
    for _ in range(count):
        axis.single_step(False, low_us=500000)
    log("  net position should be 0: %d" % axis.position)
    return _r("%s single-step" % axis.name, INFO,
              "position %d" % axis.position, position=axis.position)


def t_direction(axis, steps=None, log=print):
    """MANUAL.  Half a turn each way, so 'forward' can be confirmed."""
    steps = steps or int(axis.steps_per_rev() // 2)
    log("--- %s direction (MANUAL) ---" % axis.name)
    axis.enable()
    log("  +%d microsteps (half a rev, nominal FORWARD)" % steps)
    axis.move(steps)
    time.sleep_ms(400)
    log("  -%d microsteps (back to start)" % steps)
    axis.move(-steps)
    log("  If 'forward' was the wrong way, set INVERT_DIR['%s'] = True"
        % axis.name)
    return _r("%s direction" % axis.name, INFO, "position %d" % axis.position)


def t_step_accuracy(axis, revs=2, log=print):
    """
    Out and back N revolutions, then report the closing error.

    The software position always closes to zero -- that proves nothing.  The
    real check is mechanical: mark the shaft and see whether it returns to
    the mark.  A drift means lost steps: current limit too low, rate too
    high, or acceleration too aggressive.
    """
    log("--- %s step accuracy: %d rev out and back (MANUAL) ---"
        % (axis.name, revs))
    log("  Mark the shaft/coupler before you continue.")
    spr = int(axis.steps_per_rev())
    n = spr * revs
    axis.enable()
    axis.zero()
    r1 = axis.move(n)
    time.sleep_ms(300)
    r2 = axis.move(-n)
    log("  commanded +%d then -%d, software position now %d"
        % (n, n, axis.position))
    log("  peak rate %d steps/s, %d microsteps/rev at 1/%d"
        % (config.MAX_RATE[axis.name], spr, axis.microstep))
    log("  Does the mark line up?  If not, the motor lost steps.")
    return _r("%s step accuracy" % axis.name, INFO,
              "closing position %d" % axis.position,
              out=r1, back=r2, position=axis.position)


def t_max_rate(axis, start=500, stop=None, step=500, dwell_steps=400,
               log=print, should_stop=None):
    """
    Ramp the step rate up until the motor stalls, to find its usable ceiling.

    Stalling is loud and it will lose position, so hold the axis clear of
    anything it can hit.  Watch and listen: the stall point is where the
    motor stops turning and starts buzzing.  Note the last good rate and put
    ~70% of it in MAX_RATE.
    """
    stop = stop or min(config.ABSOLUTE_MAX_RATE, 20000)
    log("--- %s max-rate sweep %d..%d steps/s (MANUAL: listen for stall) ---"
        % (axis.name, start, stop))
    log("  Press Ctrl-C, or send any character, to stop early.")
    axis.enable()
    rows = []
    for rate in range(start, stop + 1, step):
        if should_stop and should_stop():
            log("  stopped by user at %d steps/s" % rate)
            break
        rpm = rate * 60.0 / axis.steps_per_rev()
        log("  %6d steps/s  (%.0f rpm at shaft)" % (rate, rpm))
        axis.move(dwell_steps, rate=rate, start_rate=min(start, rate))
        axis.move(-dwell_steps, rate=rate, start_rate=min(start, rate))
        rows.append({"rate": rate, "rpm": rpm})
        time.sleep_ms(120)
    log("  Sweep done.  Set MAX_RATE['%s'] to about 70%% of the last rate"
        % axis.name)
    log("  that still turned smoothly.")
    return _r("%s max rate" % axis.name, INFO,
              "swept %d points" % len(rows), rows=rows)


def t_resonance(axis, low=200, high=3000, points=20, dwell_steps=300,
                log=print, should_stop=None):
    """
    Step through a band of rates looking for the resonance that makes a
    stepper growl and skip.

    Steppers have a mechanical resonance, usually somewhere between 100 and
    900 full-steps/s, where torque collapses.  Microstepping suppresses it;
    if you find a bad band, either avoid it or raise the microstepping.
    """
    log("--- %s resonance sweep %d..%d steps/s (MANUAL: listen) ---"
        % (axis.name, low, high))
    axis.enable()
    rows = []
    inc = max(1, (high - low) // max(1, points - 1))
    for rate in range(low, high + 1, inc):
        if should_stop and should_stop():
            log("  stopped early")
            break
        full_sps = rate / float(axis.microstep)
        log("  %6d microsteps/s  (%.0f full-steps/s)" % (rate, full_sps))
        axis.move(dwell_steps, rate=rate, start_rate=min(low, rate))
        axis.move(-dwell_steps, rate=rate, start_rate=min(low, rate))
        rows.append({"rate": rate, "full_steps_per_sec": full_sps})
        time.sleep_ms(150)
    log("  Note any rate where it growled or lost position.")
    return _r("%s resonance" % axis.name, INFO,
              "swept %d points" % len(rows), rows=rows)


def t_microstep_consistency(axis, log=print):
    """
    Command exactly one revolution at each microstep setting.

    Every pass should land in the same place.  If one setting overshoots or
    undershoots by a clean factor of two, DRIVER_TYPE is wrong -- that is
    the A4988-vs-DRV8825 1/16 disagreement showing up mechanically.
    """
    log("--- %s microstep consistency (MANUAL: mark the shaft) ---"
        % axis.name)
    import stepper
    table = stepper.MICROSTEP_TABLE[config.DRIVER_TYPE]
    original = axis.microstep
    axis.enable()
    rows = []
    for div in sorted(table):
        axis.set_microstep(div)
        spr = int(axis.steps_per_rev())
        rate = min(config.MAX_RATE[axis.name], 400 * div)
        log("  1/%-3d : %d microsteps = 1 rev, at %d steps/s" % (div, spr, rate))
        axis.move(spr, rate=rate)
        time.sleep_ms(500)
        axis.move(-spr, rate=rate)
        time.sleep_ms(300)
        rows.append({"div": div, "steps_per_rev": spr, "rate": rate})
    axis.set_microstep(original)
    log("  Each pass should have returned to the same mark.")
    return _r("%s microstep consistency" % axis.name, INFO,
              "%d modes" % len(rows), rows=rows)


def t_holding_torque(axis, log=print):
    """MANUAL.  Energise and hold, so you can feel the holding torque."""
    log("--- %s holding torque (MANUAL) ---" % axis.name)
    axis.enable()
    log("  Coils energised.  Try to turn the shaft by hand.")
    log("  Weak or no resistance -> current limit (Vref) is too low, or the")
    log("  A4988 has no VDD, or a coil pair is miswired at the connector.")
    log("  Buzzing/heating with no torque -> the coil pairs are split")
    log("  across the wrong connector pins.")
    time.sleep(5)
    axis.disable()
    log("  Coils released -- the shaft should now spin freely.")
    return _r("%s holding torque" % axis.name, INFO, "manual observation")


# ==========================================================================
#   HELPERS
# ==========================================================================

def vref_table(log=print):
    """
    Current-limit reference for the trimpot on each carrier.

    A4988   : I_limit = Vref / (8 * Rsense)
    DRV8825 : I_limit = Vref / (5 * Rsense)

    Measure Vref between the trimpot wiper and GND with the motor
    DISCONNECTED and VMOT applied.
    """
    rs = config.SENSE_RESISTOR
    k = 8.0 * rs if config.DRIVER_TYPE == "A4988" else 5.0 * rs
    log("--- current limit / Vref helper ---")
    log("  driver         %s" % config.DRIVER_TYPE)
    log("  sense resistor %.3f ohm  (CHECK THIS -- clones vary)" % rs)
    log("  formula        I = Vref / %.3f" % k)
    log("")
    log("   target I    set Vref to")
    rows = []
    i = 0.4
    while i <= 2.01:
        v = i * k
        rows.append({"current": i, "vref": v})
        log("   %.2f A       %.3f V" % (i, v))
        i += 0.2
    target = config.MOTOR_RATED_CURRENT * config.CURRENT_SAFETY_FACTOR
    log("")
    log("  Your config: %.2f A rated * %.2f safety = %.2f A -> Vref %.3f V"
        % (config.MOTOR_RATED_CURRENT, config.CURRENT_SAFETY_FACTOR,
           target, target * k))
    log("  Measure with the motor UNPLUGGED and VMOT applied.")
    if config.DRIVER_TYPE == "A4988":
        log("  Note: in FULL-step mode the A4988 only reaches 71% of this.")
    return _r("vref", INFO, "Vref %.3f V" % (target * k),
              rows=rows, target_current=target, target_vref=target * k)


# ==========================================================================
#   SUITES
# ==========================================================================

def run_safe(axes, endstops, log=print):
    """Everything that needs no motor power and moves nothing."""
    log("=" * 66)
    log("SAFE SUITE -- no motion, VMOT not required")
    log("=" * 66)
    results = [
        t_config_sanity(log),
        t_control_pin_integrity(axes, log),
        t_driver_presence(axes, log),
        t_driver_pad2(axes, log),
        t_microstep_table(axes, log),
        t_step_pulse_timing(axes, log),
        t_endstop_state(endstops, log),
        t_i2c(log),
    ]
    summarise(results, log)
    return results


def run_motion(axes, log=print, should_stop=None):
    """Motion tests.  VMOT must be on and the current limit set."""
    log("=" * 66)
    log("MOTION SUITE -- the motors will move.  Clear the mechanism.")
    log("=" * 66)
    results = []
    for name, axis in axes.items():
        results.append(t_single_step(axis, log=log))
        results.append(t_direction(axis, log=log))
        results.append(t_step_accuracy(axis, log=log))
        axis.disable()
    summarise(results, log)
    return results


def summarise(results, log=print):
    log("")
    log("-" * 66)
    log("%-34s %s" % ("TEST", "RESULT"))
    log("-" * 66)
    counts = {}
    for r in results:
        v = r["verdict"]
        counts[v] = counts.get(v, 0) + 1
        log("%-34s %-5s %s" % (r["name"][:34], v, r["detail"][:22]))
    log("-" * 66)
    log("  " + "  ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())))
    return counts
