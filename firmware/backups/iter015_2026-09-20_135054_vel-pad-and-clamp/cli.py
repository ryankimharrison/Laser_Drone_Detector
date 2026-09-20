"""
Interactive serial console for the turret test suite.

Connect with any serial terminal at any baud (USB CDC ignores baud):
    mpremote connect COM5 repl
    ...or Thonny's REPL, or PuTTY.

Type 'help' for the command list.  During a long test, sending any
character aborts it.
"""

import select
import sys
import time

import config
import diagnostics
import endstop as endstop_mod
import kinematics
import peripherals
import pinmap
import stepper

# select.poll() is the portable way to check stdin without blocking on the
# RP2040 port, but it does not exist everywhere (notably Windows CPython,
# where the host-side tests run).  Degrade to "no abort key" rather than
# failing to import.
_poll = None
try:
    _poll = select.poll()
    _poll.register(sys.stdin, select.POLLIN)
except (AttributeError, OSError, ValueError):
    _poll = None


def key_pressed():
    """True if the user has sent anything -- used to abort long tests."""
    if _poll is None:
        return False
    try:
        if _poll.poll(0):
            sys.stdin.read(1)
            return True
    except Exception:
        pass
    return False


# ==========================================================================
#   LEVELLING SAFETY ENVELOPE
# ==========================================================================
# `level` drives the payload open-loop on a sensitivity measured once, before
# any correction. On 2026-09-18 that measurement came out -0.454 deg/deg with
# the payload already pitched near its stop; the correction is -t/k, so every
# iteration drove further down, 20 deg at a time, until the payload hit its
# own frame. It finished 57.6 deg from level and set the datum there.
#
# These bound the blast radius of a bad measurement. Gravity is available on
# every iteration, so none of them has to be inferred from the step counter --
# which was itself 57 deg wrong at the time, having been zeroed at the bogus
# datum a previous run had set.

#: Refuse to level beyond this. The mechanism reaches its own frame before the
#: +/-90 deg the step counter believes in, and at a large tilt the counter is
#: exactly what cannot be trusted to notice.
LEVEL_MAX_TILT_DEG = 75.0

#: Largest single correction. Was 20.0 -- the size of the jumps that did the
#: damage. Levelling is not time-critical; a smaller step costs one extra
#: iteration and bounds how far one wrong command can travel.
LEVEL_MAX_STEP_DEG = 12.0

#: Slack when asking "did the tilt get worse?". The accelerometer's own noise
#: is ~0.76 deg/sample (MEASURED), and settle_tilt averages 12, so the mean is
#: good to ~0.22 deg. 1.0 deg is comfortably outside that and well inside any
#: real divergence, which grows by whole degrees per iteration.
LEVEL_NOISE_DEG = 1.0

#: Departure from the MEASURED at-rest |a| that counts as an impact, checked
#: after LEVEL_SETTLE_MS. A settled payload reads a constant magnitude at any
#: attitude, so a persistent departure is pure acceleration -- something is
#: still moving when it should not be. An A4988 raises no fault when a stepper
#: skips against a hard stop, and the step counter keeps counting, so the
#: accelerometer is the ONLY witness that a collision happened at all.
#:
#: Checked against the measured baseline, not 1.000 g: this part reads
#: 0.89-0.99 g stationary. At 0.35 against 1.0 it rejected a perfectly normal
#: 11 deg move and killed a homing run.
LEVEL_IMPACT_G = 0.45

#: Settle time after a levelling move before the payload is called still.
#: settle_tilt() uses 250 ms for the same reason: an 11 deg move rings, and a
#: sample taken during the ring is not a measurement of anything.
LEVEL_SETTLE_MS = 250

BANNER = r"""
+------------------------------------------------------------------+
|  Laser Turret -- stepper test & debug suite                      |
|  board rev 3/27/2026 prod v1                                     |
+------------------------------------------------------------------+
"""

HELP = """
SUITES
  safe                     run every no-motion test (start here)
  motion                   run the motion tests on both axes
  vref                     current-limit / Vref table for your driver

INFO
  pins                     full pin map, with the known board findings
  status                   axis state, position, fault line
  state                    same, as one line of JSON (used by the GUI)
  pads                     who owns each STEP pad (SIO/PIO0/PIO1) vs mode.
                           An axis in `vel` whose pad reads SIO counts steps
                           it never emits -- that is the fault to look for.
  cfg                      current configuration
  findings                 the hardware findings, short form
  wiring                   coil connector pinouts -- read this if a motor
                           buzzes but will not turn

DIFFERENTIAL          (payload axes -- what you actually want)
  diff                     resolution, position, gear ratio, signs
  pitch <deg> [dps]        pure pitch: both motors together
  yaw <deg> [dps]          pure yaw: both motors opposed
  dmove <dpitch> <dyaw>    relative payload move, both at once
  aim <pitch> <yaw> [dps]  absolute payload move
  dzero                    set this pose as payload zero
  dchar                    characterise the differential (MANUAL: watch it)
  dsign pitch|yaw|A|B      flip a sign for this session

IMU                   (GY-85 on J5 -- the only sensor that sees the mechanism)
  imu                      status, live rate and tilt
  imu cal                  gyro zero-rate calibration -- payload MUST be still
  imu watch                stream rate + tilt
  imu fast                 machine-readable, cheap: IMUF gx gy gz pitch roll ms
  imu burst [n]            measure the real gyro read rate and implied ODR
  clk                      CLK <ticks_ms> -- host/board clock handshake
  imu mag                  "MAG mx my mz pitch yaw" -- field AND pose in one
                           reply, for host-side yaw homing off the motor field
  level [tol]              drive to TRUE LEVEL using gravity, then set the
                           datum. Pitch only: gravity cannot see yaw.

MOTOR CONTROL         (axis = pan | tilt | both)
  NOTE: 'pan' is MOTOR A, 'tilt' is MOTOR B. Neither is a payload axis --
  driving one alone moves the payload diagonally. Use the commands above.
  enable <axis>            energise coils
  disable <axis>           release coils
  ms <axis> <div>          set microstepping (1 2 4 8 16 [32 on DRV8825])
  invert <axis>            flip the direction sense for this session
  zero <axis>              set current position as zero

MOTION
  move <axis> <steps> [rate]     move N microsteps (negative = reverse)
  deg <axis> <degrees> [rate]    move by angle
  step <axis> [n]                n slow single steps you can watch
  hold <axis>                    energise and hold, to feel the torque
  accuracy <axis> [revs]         out-and-back, check for lost steps
  maxrate <axis>                 sweep rate upward to find the stall point
  resonance <axis>               sweep for the resonance growl
  msweep <axis>                  one rev at every microstep setting

ENDSTOPS
  endstops                 live monitor -- press the switches
  home <axis>              3-phase homing routine
  rep <axis> [cycles]      homing repeatability (the best mechanical test)

PERIPHERALS
  laser on|off|pulse <ms>  interlocked -- see LASER_ENABLED in config.py
  ir on|off|blink
  fan on|off
  ping                     ultrasonic distance
  adc                      analog sensor input
  i2c                      bus scan + pull-up check

  stop                     disable everything, drop all outputs
  quit                     leave the console (back to the REPL)
"""


class Console:
    def __init__(self):
        self.axes = stepper.build_axes()
        # The two motors as one 2-DOF payload. Neither motor is a payload
        # axis on its own -- see kinematics.py.
        self.platform = kinematics.Platform(self.axes)
        self.endstops = endstop_mod.build_endstops()
        self.outputs = peripherals.build_outputs()
        self.sensors = peripherals.build_sensors()
        # The velocity loop needs the laser pin directly, not through the
        # Output wrapper: if the watchdog trips it is because something in the
        # normal path has stopped responding, so the cut must not depend on it.
        self._imu = None
        self.velocity = stepper.VelocityLoop(
            self.axes, laser_pin=self.outputs["laser"].pin)
        self.running = True

    # ------------------------------------------------------------------
    def resolve(self, name):
        """'pan' | 'tilt' | 'both' -> list of Axis."""
        if name in ("both", "all", None, ""):
            return list(self.axes.values())
        ax = self.axes.get(name)
        if ax is None:
            raise ValueError("unknown axis '%s' (pan, tilt or both)" % name)
        return [ax]

    def safe_state(self):
        for ax in self.axes.values():
            try:
                ax.abort()
                ax.disable()
            except Exception:
                pass
        for o in self.outputs.values():
            try:
                o.off()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def run(self):
        print(BANNER)
        print(pinmap.describe())
        print("")
        self.warn_findings()
        print("Type 'help' for commands, 'safe' to run the no-motion suite.")
        last_interrupt = -10000
        while self.running:
            try:
                line = input("turret> ").strip()
            except EOFError:
                print("")
                break
            except KeyboardInterrupt:
                # Ctrl-C always parks the hardware first -- it is the panic
                # button and must be safe to hit at any moment, including at
                # an idle prompt, which is when the GUI's E-STOP sends it if
                # nothing happens to be moving.
                self.safe_state()
                now = time.ticks_ms()
                if time.ticks_diff(now, last_interrupt) < 1000:
                    # Two in quick succession means "get out of the way":
                    # that is exactly what mpremote sends to reach the raw
                    # REPL, and without this the console would hold the port
                    # forever and no file could ever be copied to the board.
                    print("\n[second interrupt -- leaving the console]")
                    break
                last_interrupt = now
                print("\n[interrupt -- everything disabled]")
                continue
            if not line:
                continue
            self.execute(line)
        self.safe_state()
        print("everything disabled.")

    def execute(self, line):
        """
        Run one command line, converting any failure into a printed message.

        A usage mistake (bad or missing argument) only prints -- it must not
        drop the motors, because on a tilt axis with gravity on it, losing
        holding torque over a typo means the barrel falls.  Anything else is
        treated as a real fault and everything is parked safe.
        """
        try:
            self.dispatch(line.split())
            return True
        except KeyboardInterrupt:
            self.safe_state()
            print("\n[interrupted -- everything disabled]")
        except Exception as e:
            # A usage mistake is either a plain ValueError/IndexError from
            # argument parsing, or a StepperUsageError flagged with .usage.
            if isinstance(e, (ValueError, IndexError)) or getattr(e, "usage", False):
                print("usage error: %s" % e)
            else:
                self.safe_state()
                print("error: %s: %s   [everything disabled]"
                      % (type(e).__name__, e))
        return False

    def warn_findings(self):
        print("Board findings encoded in this suite:")
        print("  1. Tilt STEP/DIR net LABELS are swapped in the schematic.")
        print("     Physical truth: GPIO0=DIR, GPIO1=STEP.  Handled here.")
        if config.DRIVER_TYPE == "A4988":
            print("  2. Driver pad 2 is VDD on an A4988, not FAULT.")
            print("     A4988_VDD_SOURCE = '%s'" % config.A4988_VDD_SOURCE)
            if config.A4988_VDD_SOURCE == "external":
                print("     -> expecting your 3V3 bodge on that net.")
            elif config.A4988_VDD_SOURCE == "gpio":
                print("     -> sourcing VDD from a GPIO (out of spec).")
        else:
            print("  2. DRIVER_TYPE=DRV8825, so pad 2 is a real FAULT input.")
        unsafe = []
        if not config.ULTRASONIC_ECHO_IS_3V3_SAFE:
            unsafe.append("ultrasonic echo")
        if not config.ANALOG_SENSOR_IS_3V3_SAFE:
            unsafe.append("analog sensor")
        if unsafe:
            print("  3. 5 V sensor returns are locked out: %s"
                  % ", ".join(unsafe))
        print("")

    # ------------------------------------------------------------------
    def dispatch(self, argv):
        cmd = argv[0].lower()
        a = argv[1:]

        def arg(i, default=None, cast=str):
            if i < len(a):
                return cast(a[i])
            if default is None and cast is not str:
                raise ValueError("missing argument %d for '%s'" % (i + 1, cmd))
            return default

        # -------- clock --------
        # FIRST in the chain, and it does nothing but read the counter and
        # print it. This is one half of a Cristian's-algorithm handshake: the
        # host brackets it with two perf_counter() reads and places this tick
        # at the midpoint, so every microsecond of work done in here is error
        # added to the shared timebase. No argument parsing, no I2C, no lock.
        if cmd == "clk":
            print("CLK %d" % time.ticks_ms())

        # -------- info --------
        elif cmd in ("help", "?"):
            print(HELP)
        elif cmd == "pins":
            print(pinmap.describe())
        elif cmd == "findings":
            self.warn_findings()
        elif cmd == "wiring":
            print(pinmap.describe_wiring())
        elif cmd == "cfg":
            self.show_config()
        elif cmd == "status":
            self.show_status()
        elif cmd == "state":
            self.emit_state()
        elif cmd == "pads":
            self.show_pads()

        # -------- differential (payload axes) --------
        elif cmd == "diff":
            print(self.platform.describe())
        elif cmd == "pitch":
            self.do_payload(float(a[0]), 0.0, arg(1, cast=float) if len(a) > 1 else None)
        elif cmd == "yaw":
            self.do_payload(0.0, float(a[0]), arg(1, cast=float) if len(a) > 1 else None)
        elif cmd == "dmove":
            self.do_payload(float(a[0]), float(a[1]),
                            float(a[2]) if len(a) > 2 else None)
        elif cmd == "aim":
            cp, cy = self.platform.position()
            self.do_payload(float(a[0]) - cp, float(a[1]) - cy,
                            float(a[2]) if len(a) > 2 else None)
        elif cmd == "dzero":
            self.platform.zero()
            print("payload zeroed: pitch 0.000  yaw 0.000")
        elif cmd == "limits":
            self.do_limits()
        elif cmd == "sethome":
            self.do_sethome(a)
        elif cmd == "home2":
            self.do_gohome(float(a[0]) if a else None)
        elif cmd == "dsign":
            self.do_dsign(a[0] if a else "")
        elif cmd == "dchar":
            self.do_dchar(int(a[0]) if a else 400)

        # -------- velocity mode (tracking) --------
        elif cmd == "vel":
            self.do_vel(a)
        elif cmd == "pvel":
            self.do_pvel(a)
        elif cmd == "level":
            self.do_level(a)
        elif cmd == "imu":
            self.do_imu(a)
        elif cmd == "velmode":
            self.do_velmode(a[0] if a else "")

        # -------- suites --------
        elif cmd == "safe":
            diagnostics.run_safe(self.axes, self.endstops)
        elif cmd == "motion":
            if not self.confirm("Motors will move. Mechanism clear?"):
                return
            diagnostics.run_motion(self.axes, should_stop=key_pressed)
        elif cmd == "vref":
            diagnostics.vref_table()

        # -------- axis control --------
        elif cmd == "enable":
            for ax in self.resolve(arg(0, "both")):
                ax.enable()
                print("%s enabled" % ax.name)
        elif cmd == "disable":
            for ax in self.resolve(arg(0, "both")):
                ax.disable()
                print("%s disabled" % ax.name)
        elif cmd == "ms":
            div = arg(1, cast=int)
            for ax in self.resolve(arg(0, "both")):
                bits = ax.set_microstep(div)
                print("%s: 1/%d  bits=%s  %d microsteps/rev"
                      % (ax.name, div, bits, int(ax.steps_per_rev())))
        elif cmd == "invert":
            for ax in self.resolve(arg(0, "both")):
                cur = config.INVERT_DIR.get(ax.name, False)
                config.INVERT_DIR[ax.name] = not cur
                print("%s INVERT_DIR now %s (session only -- edit config.py "
                      "to persist)" % (ax.name, not cur))
        elif cmd == "zero":
            for ax in self.resolve(arg(0, "both")):
                ax.zero()
                print("%s zeroed" % ax.name)

        # -------- motion --------
        elif cmd == "move":
            ax = self.resolve(arg(0))[0]
            n = arg(1, cast=int)
            rate = arg(2, cast=int) if len(a) > 2 else None
            self.do_move(ax, n, rate)
        elif cmd == "deg":
            ax = self.resolve(arg(0))[0]
            deg = float(a[1])
            rate = arg(2, cast=int) if len(a) > 2 else None
            self.do_move(ax, ax.degrees_to_steps(deg), rate)
        elif cmd == "step":
            ax = self.resolve(arg(0))[0]
            n = int(a[1]) if len(a) > 1 else 8
            diagnostics.t_single_step(ax, n)
        elif cmd == "hold":
            diagnostics.t_holding_torque(self.resolve(arg(0))[0])
        elif cmd == "accuracy":
            ax = self.resolve(arg(0))[0]
            revs = int(a[1]) if len(a) > 1 else 2
            diagnostics.t_step_accuracy(ax, revs)
        elif cmd == "maxrate":
            ax = self.resolve(arg(0))[0]
            if not self.confirm("This will stall the motor. Axis clear?"):
                return
            diagnostics.t_max_rate(ax, should_stop=key_pressed)
            ax.disable()
        elif cmd == "resonance":
            ax = self.resolve(arg(0))[0]
            diagnostics.t_resonance(ax, should_stop=key_pressed)
            ax.disable()
        elif cmd == "msweep":
            ax = self.resolve(arg(0))[0]
            diagnostics.t_microstep_consistency(ax)
            ax.disable()

        # -------- endstops --------
        elif cmd == "endstops":
            diagnostics.t_endstop_monitor(self.endstops,
                                          should_stop=key_pressed)
        elif cmd == "home":
            name = arg(0)
            ax = self.resolve(name)[0]
            es = self.endstops.get(ax.name)
            if es is None:
                print("no endstop assigned to %s (see ENDSTOP_AXIS)" % ax.name)
                return
            ax.enable()
            print("homing %s..." % ax.name)
            print(endstop_mod.home(ax, es))
        elif cmd == "rep":
            ax = self.resolve(arg(0))[0]
            es = self.endstops.get(ax.name)
            if es is None:
                print("no endstop assigned to %s" % ax.name)
                return
            cycles = int(a[1]) if len(a) > 1 else 5
            ax.enable()
            endstop_mod.repeatability(ax, es, cycles)
            ax.disable()

        # -------- peripherals --------
        elif cmd in ("laser", "ir", "fan"):
            self.do_output(cmd, a)
        elif cmd == "ping":
            d = self.sensors["ultrasonic"].read_avg()
            print("distance: %s" % ("timeout / out of range" if d is None
                                    else "%.1f cm" % d))
        elif cmd == "adc":
            s = self.sensors["analog"]
            raw = s.read_avg()
            print("ADC%d raw %.0f / 65535  = %.3f V"
                  % (pinmap.ANALOG_ADC_CH, raw, raw * 3.3 / 65535.0))
        elif cmd == "i2c":
            diagnostics.t_i2c()

        # -------- control --------
        elif cmd == "stop":
            self.safe_state()
            print("all axes disabled, all outputs off")
        elif cmd in ("quit", "exit"):
            self.running = False
        else:
            print("unknown command '%s' -- try 'help'" % cmd)

    # ------------------------------------------------------------------
    def do_payload(self, dpitch, dyaw, rate=None):
        """Relative payload move. Both motors run together."""
        if not self.platform.enabled:
            self.platform.enable()
            print("(both motors auto-enabled)")
        r = self.platform.move_by(dpitch, dyaw, rate=rate)
        sa, sb = r["steps"]
        if r.get("clamped"):
            print("  LIMITED on %s -- asked pitch %+.3f yaw %+.3f, "
                  "travel allows less" % (", ".join(r["clamped"]),
                                          dpitch, dyaw))
        print("pitch %+.3f  yaw %+.3f   (motor A %+d, B %+d)  in %d us"
              % (r["pitch"], r["yaw"], sa, sb, r["elapsed_us"]))
        p, y = self.platform.position()
        print("  now at pitch %+.3f  yaw %+.3f" % (p, y))
        if r.get("guarded"):
            print("  STOPPED EARLY: endstop triggered")

    def do_vel(self, a):
        """vel <A_steps_per_s> <B_steps_per_s> -- signed motor rates."""
        if len(a) < 2:
            raise ValueError("vel takes two signed motor rates, A and B")
        ra, rb = float(a[0]), float(a[1])
        if not self.platform.enabled:
            self.platform.enable()
        self.velocity.command(ra, rb)
        print("ok")

    def do_pvel(self, a):
        """
        pvel <pitch_deg_s> <yaw_deg_s> -- payload rates, for humans.

        The host tracking loop should use `vel` and do its own transform: its
        image Jacobian already maps pixel error straight to motor rates, so
        routing through payload angles just adds a conversion that can carry a
        sign error. This exists for driving the thing by hand.
        """
        if len(a) < 2:
            raise ValueError("pvel takes pitch and yaw rates in deg/s")
        dp, dy = float(a[0]), float(a[1])
        ra, rb = kinematics.payload_rate_to_motor(dp, dy,
                                                  self.platform.microstep)
        if not self.platform.enabled:
            self.platform.enable()
        self.velocity.command(ra, rb)
        # Report the pose IN THE REPLY. A host closing a loop otherwise has to
        # poll `state` as a second round trip, and the two commands contend
        # for the one serial link -- each blocking the other, doubling the
        # feedback latency and letting the watchdog park the motors in the
        # gap. One request, one answer, everything the loop needs.
        p, y = self.platform.position()
        print("ok %.4f %.4f" % (p, y))

    def do_velmode(self, what):
        w = (what or "").lower()
        if w == "off":
            self.velocity.stop()
            print("velocity mode off")
        elif w in ("on", ""):
            self.velocity.start()
            print("velocity mode on -- send 'vel A B'; watchdog %d ms"
                  % config.VEL_WATCHDOG_MS)
        elif w == "status":
            v = self.velocity
            print("running   %s" % v.running)
            print("ticks     %d   watchdog trips %d   gearshifts %d"
                  % (v.ticks, v.trips, v.shifts))
            print("gear      1/%d  (%s)"
                  % (self.axes["pan"].microstep,
                     "dynamic" if getattr(config, "VEL_GEARSHIFT", False)
                     else "pinned"))
            print("last vel  %d ms ago%s"
                  % (v.age_ms(), "  TRIPPED" if v.tripped else ""))
            for n, ax in self.axes.items():
                print("  %-5s mode=%-4s target %+8.1f  current %+8.1f steps/s"
                      % (n, ax.mode, ax.target_rate, ax.current_rate))
        else:
            raise ValueError("velmode takes: on | off | status")

    def do_imu(self, a):
        """imu | imu cal | imu watch | imu mag -- the payload IMU.

        `imu mag` exists for YAW HOMING. Gravity cannot see yaw, so `level`
        gives a pitch datum only. But the motors' permanent-magnet field is
        fixed in the base frame and rotates in the IMU frame as the payload
        yaws, which makes it the only absolute yaw reference on the machine:
        a coarse monotonic trend (~-4.96 LSB/deg on my) plus a resolver-like
        ripple at the rotor period (~3.1 deg of payload). The host sweeps yaw,
        samples this, and fits the datum -- so the algorithm lives on the host
        and the firmware just reports numbers.

        Machine-readable on purpose: "MAG <mx> <my> <mz> <pitch> <yaw>".
        Yaw is included so one round trip carries both the field and the pose
        it was measured at; sampling them with two commands would let the
        platform move between the two and silently corrupt the fit.
        """
        import imu as imu_mod
        what = (a[0].lower() if a else "")
        # Only cache the object once begin() has actually succeeded. Assigning
        # first means a failed probe leaves a half-initialised object behind,
        # and every later call skips begin() and reports a stale "NOT FOUND"
        # instead of retrying -- which is exactly what it did while the module
        # was unpowered.
        if self._imu is None:
            d = imu_mod.GY85()
            d.begin()
            self._imu = d
        d = self._imu
        if what == "cal":
            print("hold still...")
            b = d.calibrate_gyro()
            print("gyro bias %.1f, %.1f, %.1f LSB  (%.2f, %.2f, %.2f deg/s)"
                  % (b[0], b[1], b[2],
                     b[0] / config.IMU_GYRO_LSB_PER_DPS,
                     b[1] / config.IMU_GYRO_LSB_PER_DPS,
                     b[2] / config.IMU_GYRO_LSB_PER_DPS))
        elif what == "mag":
            if not d.have_mag:
                raise ValueError("no magnetometer on the bus")
            mx, my, mz = d.mag_raw()
            p, y = self.platform.position()
            print("MAG %d %d %d %.4f %.4f" % (mx, my, mz, p, y))
        elif what == "fast":
            # Machine-readable, and CHEAP: two I2C burst reads, ~1 ms, no
            # describe() and no accel_live(). The plain `imu` command now
            # costs 80-160 ms because describe() proves liveness over 16
            # samples, which is fine for a human and far too slow for a
            # telemetry sampler -- at 10 Hz it would hold the io lock most of
            # the time and starve the vel stream it is supposed to be
            # observing. It also must not RECOVER the part as a side effect:
            # a sampler silently reconfiguring the accelerometer would hide
            # the very fault we are sampling to catch.
            gx, gy, gz = d.gyro_dps() if d.have_gyro else (0.0, 0.0, 0.0)
            if d.have_accel:
                pitch, roll = d.tilt_deg()
            else:
                pitch = roll = 0.0
            # The tick is APPENDED, not prefixed. Three host parsers
            # (app.py, telemetry.py, calibrate_wide_to_narrow.py) match the
            # first five fields with re.search and are unanchored, so adding a
            # sixth field at the end cannot break them. Prefixing would break
            # all three at once, on a board that is otherwise fine.
            #
            # Taken AFTER the reads, so it timestamps when the sample was in
            # hand rather than when the command was parsed.
            print("IMUF %.4f %.4f %.4f %.4f %.4f %d"
                  % (gx, gy, gz, pitch, roll, time.ticks_ms()))
        elif what == "burst":
            self._imu_burst(d, int(a[1]) if len(a) > 1 else 200)
        elif what == "watch":
            print("Ctrl-C or any key to stop")
            while not key_pressed():
                gx, gy, gz = d.gyro_dps() if d.have_gyro else (0, 0, 0)
                if d.have_accel:
                    pitch, roll = d.tilt_deg()
                else:
                    pitch = roll = 0.0
                print("gyro %+8.2f %+8.2f %+8.2f deg/s   tilt %+7.2f %+7.2f"
                      % (gx, gy, gz, pitch, roll))
                time.sleep_ms(150)
        else:
            print(d.describe())
            if d.have_gyro:
                print("  rate now  %+.2f, %+.2f, %+.2f deg/s" % d.gyro_dps())
            if d.have_accel:
                print("  tilt now  %+.2f pitch, %+.2f roll" % d.tilt_deg())

    def _imu_burst(self, d, n):
        """imu burst [n] -- how fast can this board ACTUALLY read the gyro?

        Measure before architecting. The plan is to stream the gyro at 500 Hz
        so the host can subtract the turret's own motion from the image, but
        nothing so far has established that this board can sustain that rate,
        and the answer decides whether the stream can live in the command loop
        or needs its own core. Guessing it would mean designing around a
        number nobody has.

        Three timings, because they separate the costs:
          1. the read alone, into a caller-owned buffer, no allocation
          2. plus the RAW_DATA_RDY check, which is one extra byte off the bus
          3. free-running, counting how many reads returned a NEW sample

        (3) is also the honest test of whether the ODR change took. Poll well
        above the configured 500 Hz and the fresh fraction gives the sensor's
        real output rate directly -- no trusting a register write we cannot
        read back meaningfully. This board has form here: the ADXL345 sat in
        standby returning a perfectly plausible all-zero "level" for an entire
        session. An implied ODR near 200 means DLPF_FS/SMPLRT_DIV did not take
        and the part is still on its old configuration.
        """
        if not d.have_gyro:
            raise ValueError("no gyro on the bus")
        n = max(10, min(int(n), 2000))
        buf = bytearray(8)

        t0 = time.ticks_us()
        for _ in range(n):
            d.gyro_burst_into(buf)
        raw_us = time.ticks_diff(time.ticks_us(), t0) / n

        t0 = time.ticks_us()
        for _ in range(n):
            d.gyro_ready()
            d.gyro_burst_into(buf)
        rdy_us = time.ticks_diff(time.ticks_us(), t0) / n

        fresh = 0
        t0 = time.ticks_us()
        for _ in range(n):
            if d.gyro_ready():
                fresh += 1
            d.gyro_burst_into(buf)
        span_s = time.ticks_diff(time.ticks_us(), t0) / 1e6

        print("IMUBURST n=%d" % n)
        print("  read only         %7.1f us  -> %6.0f Hz ceiling"
              % (raw_us, 1e6 / raw_us if raw_us else 0))
        print("  read + ready bit  %7.1f us  -> %6.0f Hz"
              % (rdy_us, 1e6 / rdy_us if rdy_us else 0))
        print("  fresh             %d of %d in %.1f ms" % (fresh, n, span_s * 1e3))
        print("  IMPLIED ODR       %.0f Hz   (configured 500; ~200 means the "
              "register write did not take)" % (fresh / span_s if span_s else 0))

    def do_level(self, a):
        """
        level [tol_deg] -- drive the payload to true level using the IMU.

        Until now "home" meant "wherever it was when someone typed dzero" --
        an arbitrary pose that changes every session and cannot be recovered
        after a power cycle. Gravity does not move, so the accelerometer gives
        a datum that is the same tomorrow as today.

        Yaw is not observable from gravity, so this sets a PITCH datum only.

        THE ERROR IS TOTAL TILT FROM VERTICAL, NOT THE PITCH COMPONENT.
        --------------------------------------------------------------
        This used to minimise tilt_deg()[0], which assumes the IMU's pitch axis
        is the mechanism's pitch axis. That holds near yaw 0 and nowhere else:
        the IMU turns with the payload, so a commanded pitch arrives in the
        IMU's pitch and roll in a ratio set by yaw. The measured sensitivity
        then scales by roughly cos(yaw) and INVERTS past 90 deg. Measured on
        this machine:

            yaw ~0     k = +1.16      levels fine
            yaw ~79    k = +0.19      "the payload barely moved" -- refused
            yaw ~115   k = -0.45      drove AWAY from level, into its own frame
            yaw ~180   k = -0.98      refused

        Four failed runs and one collision, all the same cause. Total tilt from
        vertical -- the angle between measured gravity and the gravity of a
        level payload -- needs no axis assignment, so it is correct at every
        yaw. A pitch component can read near zero while the payload is 70 deg
        over, if the tilt has landed on roll.

        THE DIRECTION IS PROBED, NOT PREDICTED.
        ---------------------------------------
        Rather than a signed sensitivity, this commands a small move and keeps
        whichever direction REDUCED the total tilt. That cannot invert, because
        it is not a sign -- it is an observation. The gain is taken from what
        the probe actually achieved, floored so a near-zero response cannot
        produce an enormous step.
        """
        import imu as imu_mod
        tol = float(a[0]) if a else 0.3
        if self._imu is None:
            d = imu_mod.GY85()
            d.begin()
            self._imu = d
        d = self._imu
        if not d.have_accel:
            raise ValueError("no accelerometer -- cannot level (%s)"
                             % d.accel_note)

        # PROVE THE SENSOR BEFORE TRUSTING A ZERO.
        #
        # A dead or standby ADXL345 reads exactly (-0.00, +0.00) tilt. That is
        # not a detectable error value -- it is the single most plausible
        # reading this routine can get, and it is inside every tolerance the
        # fast path below tests against. So `level` printed "already within
        # 0.30 deg", skipped the sensitivity probe, and called zero()/set_home()
        # at whatever pose the turret happened to be in. The datum this command
        # exists to make repeatable became arbitrary, silently, and the board
        # then reported "level" while the step counter read pitch +24 deg.
        #
        # The `abs(k) < 0.2` guard further down DOES catch a dead IMU, but it
        # lives in the else branch, so a reading of exactly 0.00 can never
        # reach it. Structurally, the fast path was the only path a dead
        # sensor could take, and it was the one path with no check on it.
        live, why = d.accel_live()
        if not live:
            raise ValueError(
                "accelerometer is not measuring (%s). REFUSING to set a datum: "
                "a dead accelerometer reads exactly 0.00 tilt, which this "
                "command would otherwise accept as 'already level' at ANY "
                "pose." % why)
        print("accelerometer %s" % why)

        # At-rest |a| for the impact check, measured AFTER accel_live().
        #
        # The order is the whole point. accel_live() may have had to RECOVER
        # the part from standby, and a standby ADXL345 reads (0,0,0) -- so a
        # baseline taken before it captured 0.00 g, and every subsequent
        # reading was then "0.91 g from at-rest", which aborted the run on the
        # first move. Measure the sensor only once it is known to be measuring.
        #
        # Measured rather than assumed to be 1.000 because this part reads
        # 0.89-0.99 g stationary, and a threshold against a textbook 1.0
        # spends most of its budget on that scale error.
        acc = 0.0
        for _ in range(16):
            g = d.accel_g()
            acc += (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) ** 0.5
            time.sleep_ms(5)
        base_g = acc / 16.0
        print("at-rest |a| = %.3f g" % base_g)

        if not self.platform.enabled:
            self.platform.enable()
            print("(both motors auto-enabled)")

        def settle_tilt(n=12, wait_ms=90):
            """Total angle from vertical, degrees. Axis-agnostic -- see above.

            Averaged with the payload stopped: the accelerometer reads gravity
            PLUS any real acceleration, so a sample taken while the thing is
            still ringing is not a tilt at all. That matters more here than it
            looks, because the IMU is mounted OFF the rotation axis and picks
            up a lever-arm term whenever the platform accelerates.
            """
            import math
            time.sleep_ms(250)
            acc = 0.0
            for _ in range(n):
                g = d.accel_g()
                mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) ** 0.5
                if mag <= 0.0:
                    continue
                # Angle between measured gravity and the payload's own "up".
                c = g[2] / mag
                if c > 1.0:
                    c = 1.0
                elif c < -1.0:
                    c = -1.0
                acc += math.degrees(math.acos(c))
                time.sleep_ms(wait_ms)
            return acc / n

        t0 = settle_tilt()
        print("tilt now %.2f deg from vertical (total, not a component)" % t0)
        if t0 < tol:
            print("already within %.2f deg" % tol)
        else:
            probe = 2.0 if t0 > 2.0 else 1.0
            print("probing with a %+.1f deg pitch move..." % probe)
            self.platform.move_by(probe, 0.0)
            t1 = settle_tilt()

            # DIRECTION IS OBSERVED, NOT DERIVED.
            #
            # `sign` is which way to go, taken from which way actually helped.
            # It cannot invert the way a signed sensitivity can, because it is
            # not a sign -- it is a measurement of the total tilt before and
            # after a move. At yaw ~115 the old signed k came out -0.454 and
            # drove the payload into its own frame; this cannot, because a
            # direction that increases the tilt is simply not kept.
            moved = abs(t1 - t0)
            k = moved / probe              # MAGNITUDE only, deg per deg
            sign = +1.0 if t1 < t0 else -1.0
            print("  tilt %.2f -> %.2f, so %+d is the direction that helps "
                  "(response %.3f deg/deg)" % (t0, t1, int(sign), k))
            if k < 0.2:
                raise ValueError(
                    "the payload barely moved (%.3f deg/deg of TOTAL tilt). "
                    "This is now axis-agnostic, so a yaw offset cannot cause "
                    "it -- the mechanism is slipping, is against a stop, or "
                    "the IMU is not on the payload." % k)
            # No sign check any more -- there is no sign to get wrong. What
            # remains is CONVERGENCE, and it is not symmetric. The step is
            # t/k against a true gain near 1.0, so the tilt left after one
            # iteration is t*(1 - 1/k):
            #
            #   k too HIGH  -> undershoots, always converges, just slower.
            #                  k = 1.357 leaves 26% per step. Harmless.
            #   k too LOW   -> overshoots. Below k_true/2 the overshoot
            #                  exceeds the error and it RUNS AWAY.
            #
            # A low k used to be produced by a yaw offset, which is why this
            # refusal cost four runs tonight. Now that the error is total tilt,
            # a low k really does mean the mechanism did not move.
            if k < 0.5:
                raise ValueError(
                    "response %.3f deg/deg is too LOW to drive on. Refusing: "
                    "the step is t/k, so below about half the true gain the "
                    "overshoot exceeds the error and it diverges. Check that "
                    "the payload is clear of its stops and the belt is not "
                    "slipping." % k)
            if k > 1.5:
                print("  NOTE: %.3f is well above 1.0. Safe -- it only "
                      "undershoots -- but it means the payload moved further "
                      "than commanded, so check DIFFERENTIAL_N, the microstep "
                      "setting, and the B-path pulley grub screw." % k)

            # CLOSED LOOP, NOT OPEN LOOP ON k.
            #
            # Everything above is a one-shot measurement taken before any
            # correction, and a one-shot measurement can be wrong -- that is
            # how this routine drove a payload into its own frame. Gravity is
            # available on EVERY iteration, so the loop no longer has to trust
            # k: it checks that each move actually reduced the tilt, and stops
            # the moment one does not. That catches a bad k, a slipping belt,
            # a seized axis and a collision identically, without having to
            # tell them apart first.
            prev = None
            step = 0.0
            reversals = 0
            for i in range(10):
                t = settle_tilt()
                if t < tol:
                    break

                # (a) ABSOLUTE ENVELOPE, from gravity rather than the step
                # counter. The counter was 57 deg wrong during the crash
                # because it had been zeroed at a bogus datum; gravity cannot
                # drift and cannot be zeroed wrong.
                if t > LEVEL_MAX_TILT_DEG:
                    raise ValueError(
                        "tilt %.2f deg is beyond the %.0f deg levelling "
                        "envelope. Refusing to continue -- at this attitude "
                        "the payload is near its own frame, and the step "
                        "counter cannot be trusted to say so."
                        % (t, LEVEL_MAX_TILT_DEG))

                # (b) DID THE LAST MOVE HELP? Total tilt must fall every
                # iteration. If it rose, the probed direction was wrong for
                # this part of the travel -- reverse once and carry on rather
                # than giving up, since the direction is an observation and
                # observations can be reversed. Two reversals in a row is a
                # mechanism problem, not a direction problem.
                if prev is not None and t > prev + LEVEL_NOISE_DEG:
                    sign = -sign
                    reversals = reversals + 1
                    print("  tilt rose %.2f -> %.2f; reversing to %+d"
                          % (prev, t, int(sign)))
                    if reversals >= 2:
                        raise ValueError(
                            "tilt rose again after reversing (%.2f -> %.2f "
                            "deg). Neither direction reduces it, so this is "
                            "not a sign problem: the mechanism is slipping, "
                            "jammed, or against a stop." % (prev, t))
                else:
                    reversals = 0
                prev = t

                # Total tilt is unsigned, so the direction comes from `sign`.
                step = sign * t / k
                if step > LEVEL_MAX_STEP_DEG:
                    step = LEVEL_MAX_STEP_DEG
                elif step < -LEVEL_MAX_STEP_DEG:
                    step = -LEVEL_MAX_STEP_DEG
                print("  iter %d: tilt %+.2f -> commanding %+.2f deg"
                      % (i + 1, t, step))
                self.platform.move_by(step, 0.0)

                # (c) DID SOMETHING HIT? A payload at rest reads 1 g at ANY
                # attitude -- gravity's magnitude does not care which way up
                # it is -- so any departure from 1 g is acceleration, and
                # right after a completed move it is either ringing or an
                # impact. This is the only evidence available that a stepper
                # skipped against a hard stop: an A4988 reports no fault for
                # that, and the step counter goes on counting steps the
                # mechanism never took.
                #
                # Sampled over ~80 ms and taking the WORST reading, because a
                # collision is a decaying transient: one instantaneous sample
                # lands wherever it lands in the ring.
                # Against the MEASURED at-rest magnitude, not 1.000 g: this
                # part reads 0.89-0.99 g stationary, so a textbook 1.0 spends
                # a third of the budget on scale error before any motion. And
                # AFTER a settle, because an 11 deg move rings for ~200 ms and
                # sampling immediately flags every normal move as an impact --
                # which it did, costing a homing run.
                time.sleep_ms(LEVEL_SETTLE_MS)
                worst = 0.0
                for _ in range(8):
                    g = self._imu.accel_g()
                    mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) ** 0.5
                    if abs(mag - base_g) > worst:
                        worst = abs(mag - base_g)
                    time.sleep_ms(10)
                if worst > LEVEL_IMPACT_G:
                    raise ValueError(
                        "|a| is still %.2f g from its at-rest %.2f g, %d ms "
                        "after a %+.2f deg move -- a settled payload does not "
                        "do that. Something was struck, or an axis is jammed. "
                        "Stopping before the next move."
                        % (worst, base_g, LEVEL_SETTLE_MS, step))
            t = settle_tilt()
            print("settled at %.2f deg from vertical" % t)

        before = self.platform.position()
        self.platform.zero()
        home, _hit = self.platform.set_home()
        print("")
        print("datum set at TRUE LEVEL (pitch), not an arbitrary pose.")
        print("  step counter moved %+.3f deg during levelling"
              % (before[0]))
        print("  this datum is repeatable: gravity is the reference.")

    def do_limits(self):
        """Show the payload travel limits and where the payload sits in them."""
        p, y = self.platform.position()
        print("payload travel limits (degrees, payload frame, from zero)")
        for name, now in (("pitch", p), ("yaw", y)):
            lim = self.platform.limit_for(name)
            if lim is None:
                print("  %-5s  continuous (no limit)      now %+8.3f"
                      % (name, now))
            else:
                lo, hi = lim
                room_lo, room_hi = now - lo, hi - now
                flag = ""
                if now < lo or now > hi:
                    flag = "   *** OUTSIDE ***"
                elif min(room_lo, room_hi) < 5.0:
                    flag = "   (near limit)"
                print("  %-5s  %+8.2f .. %+8.2f      now %+8.3f%s"
                      % (name, lo, hi, now, flag))
        print("  home   pitch %+.3f  yaw %+.3f"
              % (self.platform.home["pitch"], self.platform.home["yaw"]))
        print("  policy %s" % ("clamp to limit"
                               if config.PAYLOAD_CLAMP else "refuse the move"))

    def do_sethome(self, a):
        """sethome            -- home = where the payload is now
           sethome P Y        -- home = this explicit pose"""
        if len(a) >= 2:
            home, hit = self.platform.set_home(float(a[0]), float(a[1]))
        else:
            home, hit = self.platform.set_home()
        print("home set: pitch %+.3f  yaw %+.3f" % (home["pitch"], home["yaw"]))
        if hit:
            print("  (clamped on %s to stay inside travel)" % ", ".join(hit))
        print("  NOTE: home is a datum you chose, not a mechanical reference.")
        print("  There is no absolute encoder -- it is lost on reset.")

    def do_gohome(self, rate=None):
        if not self.platform.enabled:
            self.platform.enable()
            print("(both motors auto-enabled)")
        h = self.platform.home
        print("moving to home: pitch %+.3f  yaw %+.3f" % (h["pitch"], h["yaw"]))
        r = self.platform.go_home(rate=rate)
        p, y = self.platform.position()
        print("  arrived at pitch %+.3f  yaw %+.3f  in %d us"
              % (p, y, r["elapsed_us"]))
        if r.get("guarded"):
            print("  STOPPED EARLY: endstop triggered")

    def do_dsign(self, what):
        w = what.lower()
        if w in ("pitch", "yaw"):
            config.PAYLOAD_SIGN[w] = -config.PAYLOAD_SIGN[w]
            print("PAYLOAD_SIGN[%s] now %+d (session only -- edit config.py "
                  "to persist)" % (w, config.PAYLOAD_SIGN[w]))
        elif w in ("a", "b"):
            key = "pan" if w == "a" else "tilt"
            config.MOTOR_SIGN[key] = -config.MOTOR_SIGN[key]
            print("MOTOR_SIGN[%s] (motor %s) now %+d (session only)"
                  % (key, w.upper(), config.MOTOR_SIGN[key]))
        else:
            raise ValueError("dsign takes: pitch | yaw | A | B")

    def do_dchar(self, steps=400):
        """
        MANUAL. Drive each differential mode in turn so you can watch which
        physical motion it produces and confirm the signs.

        This is the test that catches a wrong MOTOR_SIGN before it becomes a
        tracking loop that runs away from the target.
        """
        pf = self.platform
        ms = pf.microstep
        axis_deg = kinematics.axis_step_deg(ms)
        move_deg = steps * axis_deg

        print("--- differential characterisation (MANUAL: watch the payload) ---")
        print("  1/%d microstepping, %.5f deg per axis-step" % (ms, axis_deg))
        print("  each move is %d steps = %.2f deg on a cardinal axis" % (steps, move_deg))
        print("")
        pf.enable()

        def leg(label, dp, dy, expect):
            print("  %s  -> expect %s" % (label, expect))
            before = pf.position()
            r = pf.move_by(dp, dy)
            sa, sb = r["steps"]
            after = pf.position()
            print("     motor A %+5d, B %+5d   ->  d(pitch) %+.3f  d(yaw) %+.3f"
                  % (sa, sb, after[0] - before[0], after[1] - before[1]))
            time.sleep_ms(700)

        leg("PURE PITCH  +", move_deg, 0.0, "nose UP, motors same direction")
        leg("PURE PITCH  -", -move_deg, 0.0, "back to start")
        leg("PURE YAW    +", 0.0, move_deg, "turn LEFT, motors opposed")
        leg("PURE YAW    -", 0.0, -move_deg, "back to start")

        print("  MOTOR A ALONE -> expect a DIAGONAL: half pitch and half yaw")
        before = pf.position()
        self.axes["pan"].move(steps)
        after = pf.position()
        print("     d(pitch) %+.3f  d(yaw) %+.3f   (each should be %.3f)"
              % (after[0] - before[0], after[1] - before[1], move_deg / 2))
        self.axes["pan"].move(-steps)
        time.sleep_ms(400)

        print("")
        print("  If a direction was wrong, fix it with 'dsign pitch|yaw|A|B'")
        print("  and then set it permanently in config.py. Do NOT re-pin a")
        print("  motor connector -- see HARDWARE_NOTES.md finding #5.")
        pf.disable()

    def do_move(self, ax, steps, rate):
        if not ax.enabled:
            ax.enable()
            print("(%s auto-enabled)" % ax.name)
        es = self.endstops.get(ax.name)
        guard = es.triggered_now if es else None
        if guard and guard():
            print("refusing to move: %s endstop is already triggered"
                  % ax.name)
            return
        r = ax.move(steps, rate=rate, guard=guard)
        print("moved %+d microsteps (%.2f deg) in %d us -> position %d"
              % (r["steps"], ax.steps_to_degrees(r["steps"]),
                 r["elapsed_us"], r["position"]))
        if r["guarded"]:
            print("  STOPPED EARLY: endstop triggered")
        f = ax.fault()
        if f:
            print("  DRIVER FAULT asserted on %s" % ax.name)

    def do_output(self, which, a):
        out = self.outputs[which]
        action = a[0] if a else "status"
        if action == "on":
            force = len(a) > 1 and a[1].lower() == "force"
            if force:
                out.on(force=True)
                print("%s ON  (FORCED past Finding #7)" % which)
                print("  GPIO%d is sinking ~20-30 mA while this is on."
                      % out.gpio)
                print("  If the load does NOT light, the copper analysis is "
                      "right and the pinout really is swapped.")
            else:
                out.on()
                print("%s ON" % which)
        elif action == "off":
            out.off()
            print("%s off" % which)
        elif action == "pulse":
            ms = int(a[1]) if len(a) > 1 else 200
            out.pulse(ms)
            print("%s pulsed %d ms" % (which, ms))
        elif action == "blink":
            out.blink()
            print("%s blinked" % which)
        elif action == "pwm":
            if not hasattr(out, "pwm_on"):
                raise ValueError("%s has no pwm mode" % which)
            if len(a) > 1 and a[1].lower() in ("off", "stop"):
                out.pwm_off()
                print("%s pulse train off" % which)
            else:
                hz = float(a[1]) if len(a) > 1 else None
                duty = float(a[2]) if len(a) > 2 else None
                f, d, on_ms = out.pwm_on(hz, duty)
                print("%s pulsing at %d Hz, %.0f%% duty (%.1f ms on per cycle)"
                      % (which, f, d, on_ms))
                print("  at 30 fps capture this alternates on/off frame by "
                      "frame -- difference consecutive frames to isolate the "
                      "dot")
        else:
            print("%s is %s (GPIO%d)"
                  % (which, "ON" if out.is_on() else "off", out.gpio))

    def show_pads(self):
        """Who owns each STEP pad right now: SIO, PIO0 or PIO1.

        The question this answers is "is the velocity state machine actually
        connected to the STEP pin?". An axis whose mode is `vel` but whose pad
        reads SIO counts every step it is asked for and emits none of them --
        silently, until reboot. One line per axis; `MISMATCH` is the fault.
        """
        names = {5: "SIO", 6: "PIO0", 7: "PIO1"}
        print("PADS")
        for name, ax in sorted(self.axes.items()):
            gpio = ax.spec["step"]
            fs = stepper._get_funcsel(gpio)
            want = 6 if (ax.vel_sm_id if ax.mode == "vel" else ax.sm_id) < 4 \
                else 7
            owner = names.get(fs, "?%s" % fs)
            ok = "ok" if (ax.mode != "vel" or fs == want) else "MISMATCH"
            print("  %-5s STEP=GPIO%-2d mode=%-5s funcsel=%-4s expect=%-4s %s"
                  % (name, gpio, ax.mode, owner, names.get(want, "?"), ok))

    def emit_state(self):
        """
        One line of JSON describing everything, for the desktop GUI.

        Deliberately machine-readable: the GUI polls this a couple of times a
        second, and parsing the human-formatted `status` table would break
        the moment someone adjusted a column width.
        """
        import json
        axes = {}
        for name, ax in self.axes.items():
            s = ax.status()
            # In velocity mode the integer counter is stale: the distance
            # covered lives in the dead-reckoning accumulator until the mode
            # is left. Reporting s["position"] there shows a motionless axis
            # while the motor is visibly turning.
            pos = ax.vel_position if ax.mode == "vel" else s["position"]
            axes[name] = {
                "enabled": s["enabled"],
                "microstep": s["microstep"],
                "position": pos,
                "degrees": s["degrees"],
                "steps_per_rev": int(s["steps_per_rev"]),
                "fault": s["fault"],
                "pad2": s["pad2_mode"],
                "has_endstop": name in self.endstops,
                "mode": ax.mode,
                "rate": ax.current_rate,
                "target_rate": ax.target_rate,
            }
        es = {}
        for name, e in self.endstops.items():
            st = e.status()
            es[name] = {"gpio": st["gpio"], "triggered": st["triggered"],
                        "raw": st["raw"]}
        out = {name: o.is_on() for name, o in self.outputs.items()}
        vel = {
            "running": self.velocity.running,
            "tripped": self.velocity.tripped,
            # THE CLAMP IS OTHERWISE SILENT. The property has existed since the
            # pitch limit was written but was never serialised, so a host
            # watching STATE could not tell "the firmware is refusing my pitch
            # command" from "a motor is dead" -- which is exactly the hour
            # run_2026-09-20_132627 cost. Latched within a run; cleared by
            # VelocityLoop.start().
            "limited": self.velocity.limited,
            "age_ms": self.velocity.age_ms(),
            "ticks": self.velocity.ticks,
            "trips": self.velocity.trips,
        }
        try:
            pp, py = self.platform.position()
            payload = {"pitch": round(pp, 4), "yaw": round(py, 4),
                       "axis_step_deg": round(
                           kinematics.axis_step_deg(self.platform.microstep), 5),
                       "N": round(config.DIFFERENTIAL_N, 4)}
        except Exception as e:
            payload = {"error": str(e)}
        print("STATE " + json.dumps({
            "payload": payload,
            "velocity": vel,
            "axes": axes,
            "endstops": es,
            "outputs": out,
            "driver": config.DRIVER_TYPE,
            "laser_enabled": config.LASER_ENABLED,
            "rst_slp": getattr(config, "RST_SLP_TIED_TO", "?"),
            "max_rate": config.MAX_RATE,
        }))

    def show_status(self):
        for name, ax in self.axes.items():
            s = ax.status()
            print("%-5s %-8s 1/%-3d pos %+7d (%+8.2f deg)  pad2=%s fault=%s"
                  % (s["name"],
                     "ENABLED" if s["enabled"] else "disabled",
                     s["microstep"], s["position"], s["degrees"],
                     s["pad2_mode"],
                     "n/a" if s["fault"] is None else s["fault"]))
        for name, es in self.endstops.items():
            st = es.status()
            print("endstop %-5s GPIO%-2d %s"
                  % (name, st["gpio"],
                     "TRIGGERED" if st["triggered"] else "open"))
        for name, o in self.outputs.items():
            print("%-6s %s" % (name, "ON" if o.is_on() else "off"))

    def show_config(self):
        print("driver type        %s" % config.DRIVER_TYPE)
        if config.DRIVER_TYPE == "A4988":
            print("A4988 VDD source   %s" % config.A4988_VDD_SOURCE)
        print("full steps/rev     %d" % config.FULL_STEPS_PER_REV)
        print("sense resistor     %.3f ohm" % config.SENSE_RESISTOR)
        print("motor rated        %.2f A  (x%.2f safety)"
              % (config.MOTOR_RATED_CURRENT, config.CURRENT_SAFETY_FACTOR))
        for ax in ("pan", "tilt"):
            print("%-5s ms=1/%-3d start=%-5d max=%-6d accel=%-6d invert=%s"
                  % (ax, config.DEFAULT_MICROSTEP[ax], config.START_RATE[ax],
                     config.MAX_RATE[ax], config.ACCEL[ax],
                     config.INVERT_DIR[ax]))
        print("laser enabled      %s" % config.LASER_ENABLED)
        print("echo 3v3 safe      %s" % config.ULTRASONIC_ECHO_IS_3V3_SAFE)
        print("analog 3v3 safe    %s" % config.ANALOG_SENSOR_IS_3V3_SAFE)

    def confirm(self, question):
        try:
            ans = input("%s [y/N] " % question).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes")


def main():
    Console().run()
