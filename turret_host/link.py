"""Serial link to the Pico: a dedicated writer thread and a 1-deep slot.

WHY A THREAD AT ALL
-------------------
The control loop must never block on USB. A Windows USB CDC write can stall for
tens of milliseconds when the host controller is busy, and the control thread is
also the thread that decides whether the laser may be on. So the control thread
only ever does `slot.put(...)` -- a lock, an assignment, an event set -- and a
separate serial thread drains the slot and does the blocking I/O. Newest-wins:
if the serial thread is behind, the stale rate is overwritten and never sent.
Sending a rate that is two frames old is worse than not sending it.

THE ONE RULE THAT COST DAYS
---------------------------
**Never poll `state` while servoing.** Two outstanding requests contend for one
serial link, each blocking the other; the feedback latency roughly doubles and
the firmware's 400 ms watchdog parks the motors in the gap. The symptom is a
turret that steps between poses instead of tracking, and it looks exactly like
a tuning problem. One request, one answer. Every blocking helper in this module
therefore refuses to run while the servo path is active -- see `_require_idle`.

WHY THERE IS NO KEEPALIVE
-------------------------
This module deliberately does NOT re-send the last rate when the control loop
goes quiet. The firmware watchdog parking the motors after 400 ms of silence is
the safety behaviour we want; a link-level keepalive would defeat it and leave a
motor holding its last rate forever after the control thread died.

WHY `vel` AND NOT `pvel`
------------------------
`vel` takes signed MOTOR rates. The image Jacobian already maps pixel error
straight to motor rates, so routing through payload angles only adds a
conversion that can carry a sign error.
"""
# Running this file as a script (`python turret_host/link.py`) puts
# turret_host/ at the FRONT of sys.path instead of the project root. That
# breaks the absolute imports below -- and worse, `turret_host/types.py` then
# SHADOWS THE STDLIB `types` MODULE, so the very next `import threading` dies
# inside enum/functools with a baffling circular-import error. So drop the
# script directory and put the project root in its place, which is exactly the
# path layout `python -m turret_host.link` produces.
#
# Guarded on __package__: an ordinary `import turret_host.link` never runs this
# and never touches sys.path.
if __package__ in (None, ""):                                    # noqa: E402
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path = [p for p in _sys.path
                 if _os.path.abspath(p or _os.getcwd()) != _here]
    _sys.path.insert(0, _os.path.dirname(_here))

import math
import re
import threading
import time

import serial
import serial.tools.list_ports

from turret_host import config
from turret_host.types import ControlOutput, LinkStatus, Slot


PROMPT = b"turret>"

# Host-side I/O policy. These are this module's own timing, not platform
# constants -- nothing here duplicates config.py.
#
# `timeout` is the hard ceiling on one reply. `settle` is the quiet gap that
# ends a reply when the prompt has not arrived yet: the console prints progress
# lines with long silent gaps while the platform is physically moving, and a
# short settle would cut the reply off mid-move and leave the rest of the
# output to be misread as the *next* command's reply.
_BLOCKING_TIMEOUTS = {
    # (timeout_s, settle_s)
    "imu cal": (25.0, 2.0),      # averages gyro samples for ~10 s
    # 2026-09-20: motion commands print nothing while the platform moves, and a
    # move longer than the settle window returned "no reply" three times today
    # (dmove 48 deg at 6 deg/s = 8 s of silence against a 5 s settle; the
    # step-check rewind's 1.07 s move against the 1.0 s default). For anything
    # that moves, the quiet gap is the full timeout: wait for the prompt.
    "level": (240.0, 240.0),
    "aim": (60.0, 60.0),
    "dmove": (60.0, 60.0),
    "move": (60.0, 60.0),
    "yaw": (60.0, 60.0),
    "sethome": (10.0, 1.0),
    "home": (60.0, 60.0),
    "imu mag": (6.0, 0.5),
    "state": (6.0, 0.5),
    "stop": (6.0, 0.5),
}
_DEFAULT_BLOCKING_TIMEOUT = (10.0, 1.0)

# The vel round trip is "ok\r\nturret> " and nothing else. If it has not come
# back inside this, the link is in trouble and we want to know now, not after
# the watchdog has already parked the motors.
_VEL_TIMEOUT_S = 0.25
_VEL_SETTLE_S = 0.05
#: DIR resync before the first real rate of a link's life -- see
#: TurretLink._resync_vel_dir. 20 steps/s for ~12 ms is a quarter of a
#: microstep: no visible motion, but the firmware's velocity tick writes the
#: DIR pin on each sign it sees.
_DIR_RESYNC_RATE = 20.0
_DIR_RESYNC_DWELL_S = 0.012

# The laser transact sits IN FRONT OF the vel stream on the writer thread (an
# "off" must never queue behind a rate update), so its timeout is part of the
# command period the firmware's 400 ms velocity watchdog is counting.
#
# It used to be timeout=1.0, settle=0.1 -- 2.5x the watchdog, in front of the
# stream, retried on every pass with no backoff. Measured against a board that
# answered `vel` but not `laser`: vel fell to 1.7 Hz with a worst gap of
# 1033 ms, tripping the watchdog; a single healthy 120 ms laser round trip
# already blew the 150 ms VEL_COMMAND_PERIOD_MAX_MS budget.
#
# The budget now: _LASER_TIMEOUT_S + _VEL_TIMEOUT_S = 80 + 250 = 330 ms worst
# case for one pass, which is under the 400 ms watchdog. A laser transition the
# board does not answer inside 80 ms is not going to be saved by waiting a
# second for it -- what it needs is for the vel stream to keep running while we
# retry, and for the operator to be told.
_LASER_TIMEOUT_S = 0.08
_LASER_SETTLE_S = 0.01

# Minimum spacing between retries of a laser transition the board rejected.
# Without it a refusal is retried at writer-loop rate: measured at 165 commands
# and 165 errors in 3 s (55 Hz). On the real firmware a refused `laser on force`
# makes cli.py call safe_state() -- aborting and disabling every axis -- so that
# storm is 55 safe_state() calls a second with the vel stream re-enabling the
# motors in between, not a harmless log flood.
_LASER_RETRY_S = 0.25

# How many times a REFUSED "on" is retried before it is abandoned and latched
# off. An "on" the board will not accept is not a safety problem -- the beam is
# dark, which is the direction we want -- so it is given up on. An "off" is
# never abandoned: it is retried for as long as the writer lives.
_LASER_ON_ATTEMPTS = 3

# After `vel 0 0` the firmware RAMPS to zero rather than dropping the rate, so
# leaving velocity mode immediately would abandon a still-moving motor and fold
# a stale count back into the position counter. Worst case is MAX_MOTOR_RATE
# (4000 steps/s) over the firmware's VEL_ACCEL of 40000 steps/s^2 = 100 ms; 150
# gives margin without making shutdown feel hung.
_VEL_RAMP_DOWN_S = 0.15

# Serial read timeout on the persistent handle. Kept small so `read(1)` wakes
# within a fraction of the frame period instead of busy-polling `in_waiting`.
_READ_TIMEOUT_S = 0.02

_RTT_EMA_ALPHA = 0.2             # published rtt_ms is smoothed; GUI readability

# How long close() will wait for the io lock before giving up on parking the
# motors and just releasing the handle. close() runs on the Tk MAIN THREAD, so
# this is a bound on how long the window takes to shut -- it is deliberately
# far below the 240 s `level` timeout a platform task can be holding, and above
# a normal `vel` round trip (~10 ms) plus the ramp.
_CLOSE_IO_TIMEOUT_S = 2.0

# How long close() waits for the io lock for the BEAM-OFF alone, before the
# longer wait for the motor park. Short on purpose: it runs on the Tk main
# thread, and if a platform command is holding the port for its full 240 s we
# want to get to the unlocked last-resort write, not sit here.
_CLOSE_LASER_LOCK_S = 0.3


class LinkError(RuntimeError):
    """The board is not where we left it, or did not answer."""


class LinkBusy(LinkError):
    """A blocking command was attempted while the servo path is live.

    This is the `state`-while-servoing bug made impossible rather than
    documented. Stop the servo first.
    """


# ==========================================================================
#   Port discovery and REPL recovery  (patterns from tools/send.py)
# ==========================================================================
def find_port():
    """The Pico's COM port, resolved by VID/PID. COM numbers move; VID/PID
    does not."""
    for p in serial.tools.list_ports.comports():
        if p.vid == config.PICO_VID and p.pid == config.PICO_PID:
            return p.device
    return None


def _read_reply(ser, timeout, settle):
    """Read until the prompt returns, or until the board goes quiet.

    Returns (text, saw_prompt). The echoed command line is still in `text`;
    the caller strips it because only the caller knows what it sent.
    """
    buf = bytearray()
    deadline = time.monotonic() + timeout
    last = time.monotonic()
    saw_prompt = False
    while time.monotonic() < deadline:
        n = ser.in_waiting
        # read(1) blocks up to the port timeout and returns the instant a byte
        # lands, so this costs no CPU and adds well under a millisecond to the
        # measured RTT. A sleep-poll loop would quantise every round trip to
        # the sleep interval.
        chunk = ser.read(n if n else 1)
        if chunk:
            buf += chunk
            last = time.monotonic()
            if buf.rstrip().endswith(PROMPT):
                saw_prompt = True
                break
        elif buf and time.monotonic() - last > settle:
            break
    text = buf.decode("utf-8", "replace")
    lines = [ln for ln in text.splitlines() if ln.strip() != PROMPT.decode()]
    return "\n".join(lines).strip(), saw_prompt


def _at_repl(port):
    """True if the board sits at the bare MicroPython REPL, not the console."""
    with serial.Serial(port, config.SERIAL_BAUD, timeout=0.3) as ser:
        time.sleep(0.25)
        ser.reset_input_buffer()
        ser.write(b"\r\n")
        time.sleep(0.4)
        banner = ser.read(ser.in_waiting or 1).decode("utf-8", "replace")
    if PROMPT.decode() in banner:
        return False
    return ">>>" in banner


def _reset_into_console(port, report=None):
    """Reboot the board so main.py runs and starts the console.

    main.py is __main__-guarded, so importing it is not enough -- only a real
    reset starts the CLI. machine.reset() re-enumerates the USB CDC device, so
    the handle dies and the COM port disappears and comes back under a
    possibly different name; that is why we re-resolve by VID/PID in the wait
    loop instead of reopening the name we had. Safe: main.py parks every pin
    before the CLI starts.
    """
    if report:
        report("board at REPL -- resetting into the console")
    try:
        with serial.Serial(port, config.SERIAL_BAUD, timeout=0.3) as ser:
            ser.write(b"import machine; machine.reset()\r\n")
            ser.flush()
            time.sleep(0.2)
    except serial.SerialException:
        # The reset tears the CDC device down mid-write. This specific failure
        # is the expected outcome of the command succeeding, not an error to
        # report -- the loop below is what decides whether it worked.
        pass

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        time.sleep(0.4)
        p = find_port()
        if not p:
            continue
        try:
            with serial.Serial(p, config.SERIAL_BAUD, timeout=0.3) as ser:
                time.sleep(0.3)
                ser.reset_input_buffer()
                ser.write(b"\r\n")
                time.sleep(0.5)
                if PROMPT in ser.read(ser.in_waiting or 1):
                    return p
        except serial.SerialException:
            continue                       # still re-enumerating; keep waiting
    raise LinkError("board did not return a turret> prompt after reset")


# ==========================================================================
#   MAG parsing
# ==========================================================================
def parse_mag(reply):
    """Parse the `imu mag` reply: "MAG <mx> <my> <mz> <pitch> <yaw>".

    Returns (mx, my, mz, pitch_deg, yaw_deg) -- field as ints in LSB, pose in
    degrees. Field and pose come from ONE round trip on purpose: sampling them
    with two commands lets the platform move between the two readings and
    silently corrupts the yaw-datum fit.

    Saturation is not checked here. Full scale and what to do about a clipped
    axis are homing's business; this function reports what the board said.
    """
    for line in reply.splitlines():
        line = line.strip()
        if not line.startswith("MAG "):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise LinkError("malformed MAG reply (want 5 fields): %r" % line)
        return (int(parts[1]), int(parts[2]), int(parts[3]),
                float(parts[4]), float(parts[5]))
    raise LinkError("no MAG line in reply: %r" % reply)


def _parse_pose(reply):
    """Pull "pitch <p>  yaw <y>" out of a human-formatted reply, if present.

    The move/home replies all carry the resulting pose. Scraping it costs
    nothing and keeps LinkStatus fresh without a second round trip -- which is
    the whole point of this module.
    """
    pose = None
    for line in reply.splitlines():
        parts = line.replace(",", " ").split()
        for i in range(len(parts) - 3):
            if parts[i] == "pitch" and parts[i + 2] == "yaw":
                try:
                    pose = (float(parts[i + 1]), float(parts[i + 3]))
                except ValueError:
                    continue               # "pitch <deg> yaw" in help text
    return pose                            # last one wins: it is the newest


# ==========================================================================
#   TurretLink
# ==========================================================================
class TurretLink:
    """The serial link. Construct freely; `start()` touches the hardware.

    Threading contract:
      * control thread  -> submit()/send_vel()/stop()   (never blocks on I/O)
      * serial thread   -> owns the port while servoing
      * setup code      -> the blocking helpers, only when NOT servoing
    """

    def __init__(self, port=None, on_message=None):
        # Nothing here opens a port: this module must import and construct on
        # a machine with no board attached.
        self._forced_port = port
        self._on_message = on_message      # optional progress sink for the GUI
        self.last_gear_report = None       # last ACK, includes its own timestamp

        self._ser = None
        self._port = ""
        # Exactly one transaction at a time. RLock, not Lock, so close() can
        # take it with a TIMEOUT and still run the park sequence through
        # _transact() on the same thread -- see close(). Cross-thread mutual
        # exclusion is unchanged.
        self._io_lock = threading.RLock()

        self.vel_slot = Slot()             # control -> serial, newest wins
        self.status_slot = Slot()          # link -> GUI

        self._writer = None
        self._stop_event = threading.Event()

        self._stat_lock = threading.Lock()
        self._connected = False
        self._sent = 0
        self._errors = 0
        self._last_error = ""
        self._rtt_ms = 0.0                 # EMA, for a readable GUI number
        self._rtt_ms_last = 0.0            # the most recent measurement
        self._pitch = 0.0
        self._yaw = 0.0
        self._last_vel_ok_t = None         # perf_counter of the last good vel
        self._vel_seq_sent = 0             # slot seq the writer has completed
        self._vel_progress = threading.Event()
        # The rates the board last ACKNOWLEDGED, not the ones we queued, and
        # the safety epoch they were queued under. stop() confirms on both,
        # because a sequence number can be satisfied by an item that overwrote
        # the zeros, and zeros acknowledged before this stop() say nothing
        # about a non-zero rate that is being written right now -- see stop().
        self._last_rates_sent = None
        self._last_rate_epoch = -1
        # Bumped by every safety action (stop/E-STOP/disarm/close). A caller
        # that decided "laser on" or "rate X" against an older epoch has
        # decided against a world that no longer exists, so its command is
        # dropped rather than applied. This is what stops an in-flight vel from
        # overwriting E-STOP's zeros, and an in-flight FIRING from outliving a
        # disarm by one frame.
        self._safety_epoch = 0
        # Hard beam latch, set by safety_veto(latch=True) -- E-STOP, operator
        # STOP, disarm, a dead worker thread, shutdown. While it is set,
        # set_laser(True) is REFUSED outright, regardless of epoch, and only an
        # explicit clear_beam_latch() from the operator's ARM path clears it.
        #
        # This is deliberately not the epoch mechanism. The epoch is a
        # generation counter, and every counter-based scheme has the same
        # shape of hole: a caller that reads the counter after the safety
        # action sees a consistent-looking world and is allowed through. Moving
        # the read to the top of the control pass closes today's instance of
        # that hole; this latch closes the CLASS, because it is not a
        # comparison -- there is no value a racing frame can read that makes
        # `laser on` acceptable again. An E-STOP must be undone by a human.
        #
        # stop() deliberately does NOT latch: it runs on every TRACK->SEARCH
        # transition, and latching there would mean the beam never returns when
        # a target is reacquired, which is an inhibit-everything failure
        # dressed up as safety.
        self._beam_latched = False
        # Set once the writer thread has exited for good (port lost, or stopped
        # and joined). A laser request with nothing to carry it is an error,
        # not a success -- see set_laser().
        self._writer_gone = False
        # One report per link for the undeliverable-laser-request condition:
        # the control loop asks 30 times a second.
        self._laser_path_reported = False

        # True between the first send_vel() and the next stop(). Blocking
        # helpers refuse while it is set.
        self._servo_active = False
        # Two separate facts, because the firmware's `stop` settles one and
        # not the other:
        #   _motors_live      we enabled the coils by sending `vel`, so they
        #                     must be zeroed before we let go of the port.
        #   _vel_mode_entered the 200 Hz velocity loop is running, so it must
        #                     be left (which folds the dead-reckoning
        #                     accumulator back into the step counter).
        # `vel` auto-enables the motors, so neither may be assumed: sending
        # `vel 0 0` at shutdown when we never servoed would energise the coils
        # on the way out the door.
        self._motors_live = False
        self._vel_mode_entered = False

        # Laser state, driven from the writer thread -- see set_laser().
        # _want is what the interlock asked for; _is is what the board was
        # actually told. They differ only for the few ms it takes the writer
        # to notice. Both start False: nothing here ever arms by default.
        self._laser_want = False
        self._laser_is = False
        # Retry accounting for a laser transition the board rejects. See
        # _service_laser(): rejections are spaced by _LASER_RETRY_S so they
        # cannot starve the vel stream, and a rejected "on" is abandoned after
        # _LASER_ON_ATTEMPTS instead of being hammered forever.
        self._laser_next_try = 0.0
        self._laser_on_tries = 0
        # Set when an "on" was abandoned, so the control loop can tell the
        # operator the beam is NOT lit despite the interlock permitting it.
        self._laser_on_failed = False
        # Wakes the writer immediately on a new rate or a laser transition,
        # instead of waiting out the slot's poll timeout.
        self._wake = threading.Event()

    # ------------------------------------------------------------------
    #   lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Resolve the board, recover it from the REPL if needed, open the
        port and start the writer thread. Raises rather than degrading."""
        if self._ser is not None:
            raise LinkError("already started")

        port = self._forced_port or find_port()
        if not port:
            raise LinkError(
                "no MicroPython board %04x:%04x found -- check USB, and that "
                "nothing else holds the port (only one process may)"
                % (config.PICO_VID, config.PICO_PID))

        if _at_repl(port):
            port = _reset_into_console(port, report=self._report)

        ser = serial.Serial(port, config.SERIAL_BAUD, timeout=_READ_TIMEOUT_S)
        self._ser = ser
        self._port = port
        # Let the CDC endpoint settle, then throw away whatever banner or
        # half-line was sitting in the buffer. Anything left here would be
        # read as the first command's reply.
        time.sleep(0.25)
        ser.reset_input_buffer()

        reply = self._transact("", timeout=3.0, settle=0.4)
        if reply is None:
            raise LinkError("board on %s did not answer with a turret> prompt"
                            % port)

        with self._stat_lock:
            self._connected = True
        self._publish()
        self._report("link up on %s" % port)

        self._stop_event.clear()
        self._writer = threading.Thread(target=self._writer_loop,
                                        name="turret-link-writer", daemon=True)
        self._writer.start()
        return self

    def close(self, io_timeout=_CLOSE_IO_TIMEOUT_S):
        """Dark, stopped, closed -- in that order, on every exit path.

        BOUNDED. This is called from `gui._on_close` on the Tk main thread, so
        it may not block for longer than a person will wait. A platform task
        (`level` is a 240 s blocking command) can be holding the io lock, and
        the old code queued behind it for the remainder of that timeout with
        the window half-closed and the cameras already released. Now the park
        sequence is attempted under `_io_lock.acquire(timeout=io_timeout)` and
        SKIPPED WITH A LOUD REPORT if the lock cannot be had -- the handle is
        always released, which is what the next run depends on.
        """
        self._stop_event.set()
        self._wake.set()
        writer_orphaned = False
        if self._writer is not None:
            self._writer.join(timeout=2.0)
            if self._writer.is_alive():
                # Almost certainly queued on _io_lock behind a platform task.
                # Do NOT null it and do NOT null _ser below: the writer's
                # _transact() checks `self._ser is None` BEFORE taking the
                # lock, so clearing the handle under a live writer turns its
                # next pass into `None.write(...)` -- an AttributeError that
                # _writer_loop_inner does not catch, killing the thread with a
                # traceback after "shutdown complete".
                writer_orphaned = True
                self._note_error("close(): the serial writer did not exit in "
                                 "2 s (queued on the io lock behind a platform "
                                 "command?)")
                self._report("WARNING: the serial writer did not exit in 2 s; "
                             "closing the port anyway.")
            else:
                self._writer = None

        if self._ser is None:
            return

        # BEAM OFF FIRST, BEFORE THE LOCK TEST.
        #
        # This used to live inside `if got_lock:` below, so the "could not take
        # the io lock" path released the handle and returned WITHOUT EVER
        # COMMANDING THE BEAM OFF -- reproduced: beam on, lock held, close()
        # returned in 4 s with the board still holding the beam, and the only
        # thing the operator was told was that the velocity watchdog would park
        # the motors. That message is worse than nothing here: this module's own
        # writer-exit path says plainly that the velocity watchdog does NOT
        # cover the laser.
        #
        # NOT via a bare _service_laser(): _transact() takes the io lock with
        # `with self._io_lock`, which has NO timeout, so calling it here would
        # block this thread -- the Tk MAIN THREAD -- for as long as the platform
        # command holds the port (up to 240 s for `level`). That is precisely
        # the hang the last fix round removed, and it must not come back.
        #
        # So: take the lock with a SHORT timeout for the beam-off alone. If the
        # holder lets go quickly we send it properly. If it does not, fall back
        # to _force_laser_off_unlocked() below.
        beam_off_attempted = False
        with self._stat_lock:
            beam_live = self._laser_is or self._laser_want
        if beam_live:
            beam_off_attempted = True
            with self._stat_lock:
                self._laser_want = False
            if self._io_lock.acquire(timeout=_CLOSE_LASER_LOCK_S):
                try:
                    self._service_laser(force=True)
                except (serial.SerialException, LinkError) as exc:
                    self._note_error("close(): laser off before the lock test "
                                     "raised: %s" % exc)
                finally:
                    self._io_lock.release()

        got_lock = self._io_lock.acquire(timeout=float(io_timeout))
        try:
            if not got_lock:
                with self._stat_lock:
                    still_on = self._laser_is
                self._note_error(
                    "close(): could not take the io lock within %.1f s -- the "
                    "motors were NOT parked and velocity mode was NOT left. A "
                    "blocking platform command is still holding the port."
                    % float(io_timeout))
                self._report(
                    "WARNING: could not park the motors at shutdown -- the io "
                    "lock is held by a platform command still in flight. The "
                    "firmware velocity watchdog (%d ms) will park the motors."
                    % config.VEL_WATCHDOG_MS)
                if still_on:
                    # Last resort. We are about to release the handle with the
                    # board holding the beam, so garbling whatever command the
                    # lock-holder has in flight costs nothing we still want --
                    # that command is being abandoned either way -- and the
                    # alternative is walking away from a lit laser.
                    sent = self._force_laser_off_unlocked()
                    # Say the thing that matters, and do NOT point at the
                    # velocity watchdog, which does not cover the laser.
                    self._note_error(
                        "close(): THE BOARD MAY STILL BE HOLDING THE BEAM ON "
                        "-- `laser off` was written unlocked=%s, unconfirmed"
                        % sent)
                    self._report(
                        "WARNING: THE BEAM MAY STILL BE ON. `laser off` was "
                        "%s but NOT confirmed by the board, and the firmware "
                        "velocity watchdog does NOT cover the laser. "
                        "KILL THE LASER SUPPLY."
                        % ("written directly to the port"
                           if sent else "NOT able to be written"))
                elif beam_off_attempted:
                    self._report("beam commanded off before the port was "
                                 "released.")
                return
            try:
                # Laser off BEFORE the motors are parked: the beam must not
                # outlive the loop that was aiming it, even by the 150 ms of
                # ramp-down below. The writer thread is already joined, so
                # this runs here, on the closing thread. Retried here even if
                # the attempt above ran: that one may have been refused, and
                # under the lock we know we have the port.
                if self._laser_is or self._laser_want:
                    self._laser_want = False
                    self._service_laser(force=True)
                if self._motors_live:
                    self._send_vel_blocking(0.0, 0.0)
                    time.sleep(_VEL_RAMP_DOWN_S)
                if self._vel_mode_entered:
                    # Leaving velocity mode folds the dead-reckoning
                    # accumulator back into the integer step counter. Skip it
                    # and `state` reports a stale position to whatever homes
                    # the machine next -- and the 200 Hz timer keeps running
                    # after we have gone.
                    self._transact("velmode off", timeout=3.0, settle=0.5)
                    self._vel_mode_entered = False
            except (serial.SerialException, LinkError) as exc:
                # The only caught write in this module, and it is caught
                # solely so the `finally` still closes the handle: if the
                # board is already gone there is nothing to zero, and a
                # leaked handle costs the NEXT run. The failure is recorded
                # and surfaced, never swallowed into silence.
                self._note_error("parking the motors at shutdown: %s" % exc)
                self._report("WARNING: could not park the motors: %s" % exc)
        finally:
            if got_lock:
                self._io_lock.release()
            with self._stat_lock:
                self._connected = False
                self._servo_active = False
            # The handle is released unconditionally, orphaned writer or not: a
            # leaked handle makes the NEXT run open a port that reports open
            # and fails every write. Safe to do under a live writer because
            # _transact() re-checks `_ser` INSIDE the io lock and raises
            # LinkError rather than dereferencing None.
            ser, self._ser = self._ser, None
            if ser is not None:
                try:
                    ser.close()
                except Exception as exc:          # pragma: no cover
                    self._note_error("closing the port: %s" % exc)
            if writer_orphaned:
                self._writer = None
            self._publish()

    def _force_laser_off_unlocked(self):
        """Write `laser off` to the port WITHOUT the io lock. Returns True if
        the bytes went out.

        Called from exactly one place: close(), when the io lock could not be
        taken and the board is still holding the beam. Every other path in this
        module serialises on _io_lock and must keep doing so -- two commands in
        flight at once is the contention trap the whole design avoids.

        The justification for breaking that rule here, and only here: the
        handle is about to be released with a lit laser on the other end. The
        in-flight command whose byte stream this interleaves with is being
        abandoned by this very close(). A garbled command is recoverable; a
        laser left on after the host has exited is not.

        No reply is read: reading would mean waiting, and the lock-holder is
        going to consume the reply anyway. This is a best-effort shout into the
        port, reported honestly as unconfirmed by the caller.
        """
        ser = self._ser
        if ser is None:
            return False
        try:
            ser.write(b"\r\nlaser off\r\n")
            ser.flush()
            return True
        except Exception as exc:              # pragma: no cover - port is dying
            self._note_error("close(): unlocked `laser off` write failed: %s"
                             % exc)
            return False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    #   control-thread side -- must never block on I/O
    # ------------------------------------------------------------------
    def submit(self, out, epoch=None):
        """Hand a ControlOutput to the serial thread. Call once per frame.

        Returns immediately. The slot is one deep: if the serial thread has
        not drained the previous command, this overwrites it. That is correct
        -- an older rate is strictly worse than the newest one.

        `epoch` is the safety epoch the caller read BEFORE it computed these
        rates (see safety_epoch). Pass it and a stop() that landed mid-frame
        wins; omit it and this rate overwrites the zeros.
        """
        if not isinstance(out, ControlOutput):
            raise TypeError("submit() takes a ControlOutput; got %r"
                            % type(out).__name__)
        return self.send_vel(out.rate_a, out.rate_b, epoch=epoch)

    def send_vel(self, rate_a, rate_b, epoch=None):
        """Queue signed motor rates (steps/s). Non-blocking.

        Returns True if the rates were queued, False if they were dropped as
        stale (a safety action overtook the caller -- see safety_epoch).
        """
        ra = self._check_rate(rate_a, "A")
        rb = self._check_rate(rate_b, "B")
        # Symmetry with set_laser(): a command nothing can carry is a FAILED
        # command, not an accepted one. This used to return True regardless, so
        # app._integrate_pose() dead-reckoned the payload pose from a `vel` the
        # board never received -- feeding the travel-limit guard a fiction.
        # _hardware_fault() caught it on the next pass, so it was bounded to one
        # frame, but the asymmetry was accidental rather than deliberate.
        live = self._writer_alive()
        with self._stat_lock:
            if not (live and self._connected):
                return False
            now_epoch = self._safety_epoch
            if epoch is not None and epoch != now_epoch:
                # A stop()/E-STOP landed while the caller was computing these.
                # Dropping it here is the whole point: send_vel() also re-arms
                # _servo_active, so letting it through would resurrect the
                # servo path the safety action just left.
                return False
            self._servo_active = True
        self.vel_slot.put((ra, rb, now_epoch))
        self._wake.set()
        return True

    # -- safety epoch ---------------------------------------------------
    @property
    def safety_epoch(self):
        """Read this BEFORE deciding a rate or a laser state, pass it back in.

        The control loop and the Tk thread both command this link. Without a
        generation counter, a frame that decided FIRING/rate-X at t can apply
        it AFTER an E-STOP at t+200us and silently undo it.
        """
        with self._stat_lock:
            return self._safety_epoch

    def safety_veto(self, latch=False):
        """Beam off now, and invalidate every decision already in flight.

        `latch=True` additionally sets the hard beam latch: set_laser(True) is
        refused until clear_beam_latch() is called. Use it for every operator
        or fault-driven safety action (E-STOP, STOP, disarm, dead thread,
        shutdown). Leave it False for routine transitions like stop() on
        TRACK->SEARCH, which must not require a human to undo.
        """
        with self._stat_lock:
            self._safety_epoch += 1
            self._laser_want = False
            if latch:
                self._beam_latched = True
        self._wake.set()

    @property
    def beam_latched(self):
        """True while a safety action is holding the beam off until re-arm."""
        with self._stat_lock:
            return self._beam_latched

    def clear_beam_latch(self):
        """Release the hard beam latch. ONLY from a deliberate operator ARM.

        Never call this from the control loop, from a fault-recovery path, or
        from anything that runs automatically. The latch exists precisely so
        that a machine which has taken a safety action cannot re-light itself.
        """
        with self._stat_lock:
            was = self._beam_latched or self._laser_on_failed
            self._beam_latched = False
            # Also give the abandoned-"on" state a fresh budget: ARM is exactly
            # the moment the operator asserts the board should now accept the
            # beam (typically after fixing the firmware's LASER_ENABLED).
            self._laser_on_failed = False
            self._laser_on_tries = 0
            self._laser_next_try = 0.0
        return was

    @property
    def laser_on_failed(self):
        """True when a requested `laser on` was refused by the board and given
        up on. The beam is dark; the operator is being told the interlock
        permitted something the hardware would not do."""
        with self._stat_lock:
            return self._laser_on_failed

    def set_laser(self, on, epoch=None):
        """Request the laser on or off. Non-blocking, control-thread safe.

        INTEGRATION GAP this closes: the interlock decides the laser state
        every frame WHILE SERVOING, and every other console command in this
        module goes through `_require_idle()` and would be refused.  So the
        laser is not a blocking helper -- it is a second desired-state input to
        the writer thread, which owns the port.  The writer issues it only on a
        transition (the firmware holds the state), so a steady FIRING costs
        nothing extra, and it is serialised behind the same `_io_lock` as the
        vel stream: still one command in flight, ever.

        The writer checks the laser BEFORE the vel each pass, so an "off"
        never queues behind a rate update. `vel` is still what the watchdog
        counts, and a laser round trip costs ~3 ms against a 33 ms frame.

        Returns True if the request was accepted, False if it could not be --
        no live writer to carry it, or a safety action has overtaken the
        caller's `epoch`. A request that cannot be delivered is RECORDED AS AN
        ERROR rather than silently accepted, because "we asked for off and
        nobody was listening" is precisely the failure that leaves a board
        holding the beam on. The control loop turns that False into an
        interlock failure via laser_path_ok.

        NON-BLOCKING, ALWAYS. This is called from the control thread once per
        frame, so it never touches the port itself -- not even to force an
        "off" when the writer is gone. The beam-off attempts live where
        blocking is already the contract: the writer thread's own exit path,
        stop(), and close().
        """
        want = bool(on)
        live = self._writer_alive()
        with self._stat_lock:
            if want and self._beam_latched:
                # A safety action is holding the beam off until a human re-arms.
                # Checked BEFORE the epoch, and with no epoch involved, because
                # this is the check that does not care what the caller read or
                # when it read it.
                return False
            if want and self._laser_on_failed:
                # The board refused this transition _LASER_ON_ATTEMPTS times and
                # the writer abandoned it. STICKY, and deliberately so: without
                # this, the control loop's next frame sets _laser_want True
                # again and the retry storm restarts at frame rate. The beam is
                # dark; the control loop turns this False into an interlock
                # failure so the operator is told, and ARM clears it.
                return False
            if want and epoch is not None and epoch != self._safety_epoch:
                # A stop()/E-STOP/disarm overtook the frame that decided this.
                return False
            deliverable = live and self._connected
            if not deliverable:
                # Nothing can carry this request. Record the ASK as off (never
                # latch an undeliverable "on"), and report the first time only
                # -- the control loop asks 30 times a second and a flood of
                # identical lines buries the one that matters.
                self._laser_want = False
                board_on = self._laser_is
                first = not self._laser_path_reported
                self._laser_path_reported = True
            elif want == self._laser_want:
                return True
            else:
                self._laser_want = want
        if not deliverable:
            if first:
                self._note_error(
                    "set_laser(%s) cannot be delivered: connected=%s, "
                    "writer_alive=%s; the board was last told laser=%s"
                    % (want, self._connected, live, board_on))
                if board_on:
                    self._report(
                        "SERIAL LINK LOST WITH THE BEAM ON -- the host can no "
                        "longer command `laser off`. KILL THE LASER SUPPLY.")
                else:
                    self._report("laser requests cannot reach the board: the "
                                 "serial writer is gone.")
            return False
        self._wake.set()
        return True

    def _writer_alive(self):
        w = self._writer
        return w is not None and w.is_alive()

    @property
    def laser_path_ok(self):
        """True when a laser request can actually reach the board.

        The control loop treats False as an interlock failure: a link that
        cannot carry `laser off` is not a link the beam may be lit over.
        """
        with self._stat_lock:
            connected = self._connected
            gone = self._writer_gone
        return bool(connected and not gone and self._writer_alive())

    @property
    def laser_on(self):
        """What the board was last told, not what it was last asked for."""
        with self._stat_lock:
            return self._laser_is

    def stop(self):
        """Zero both rates and leave the servo path.

        Waits briefly for the zeros to actually go out -- unlike send_vel,
        which is fire-and-forget. This is a transition, not a frame, so a few
        milliseconds here is free, and "stop" that has not reached the board
        is not a stop.
        """
        # Leaving the servo path means leaving TRACK, and the laser is not
        # permitted outside TRACK. Requested before the zeros so the beam is
        # never the last thing still doing its old job.
        #
        # safety_veto() FIRST, and it is what turns the beam off: it bumps the
        # epoch, so a control-loop frame that is mid-flight with a full-rate
        # `vel` and a FIRING decision loses both. Without it, that frame's
        # submit() overwrote these zeros in the slot and stop() still reported
        # success -- an E-STOP that left the mechanism slewing.
        self.safety_veto()
        self.set_laser(False)
        with self._stat_lock:
            ever_servoed = (self._servo_active or self._motors_live
                            or self._vel_mode_entered)
            self._servo_active = False
            epoch = self._safety_epoch
        if not ever_servoed:
            # Never entered the servo path, so there is nothing to stop -- and
            # `vel` AUTO-ENABLES THE MOTORS, so a reflexive `vel 0 0` here
            # would energise the coils and enter velocity mode on the way out
            # of a run that never commanded anything. Shutdown after a homing
            # run hits this every time.
            return
        if not self._writer_alive():
            # No writer to drain the slot -- send it ourselves, under the lock.
            # force=True: there is no next pass to defer a backed-off retry to.
            self._service_laser(force=True)
            self._send_vel_blocking(0.0, 0.0)
            return
        # Confirm on the RATES the board acknowledged, never on a slot sequence
        # number: the slot is newest-wins, so a racing submit() can satisfy any
        # sequence target with a full-rate `vel` and make this report a stop
        # that never happened. Re-put the zeros on every pass so a racing
        # submit() loses instead of winning -- combined with the epoch bump
        # above, the racing submit() is dropped anyway, and this is the
        # backstop for a caller that passed no epoch.
        deadline = time.monotonic() + 0.3
        while True:
            self.vel_slot.put((0.0, 0.0, epoch))
            self._wake.set()
            with self._stat_lock:
                if (self._last_rates_sent == (0.0, 0.0)
                        and self._last_rate_epoch >= epoch):
                    return
            if time.monotonic() >= deadline:
                break
            # A FLOOR on the retry spacing, not just a wait. _vel_progress is
            # set even on a FAILED send, so against a board that rejects every
            # `vel` this loop and the writer ping-pong flat out: measured at
            # 9878 errors in 300 ms, which buries the real fault in the log and
            # burns a core at shutdown. 5 ms still gives ~60 attempts inside the
            # 300 ms budget, far more than the handful a healthy board needs.
            self._vel_progress.wait(0.01)
            self._vel_progress.clear()
            time.sleep(0.005)
        with self._stat_lock:
            last = self._last_rates_sent
        self._note_error("stop(): zero rates NOT confirmed within 300 ms; the "
                         "board was last acknowledged at %r" % (last,))
        self._report("WARNING: stop() could not confirm zero rates; last "
                     "acknowledged rates %r" % (last,))

    @staticmethod
    def _check_rate(rate, which):
        r = float(rate)
        if not math.isfinite(r):
            # A NaN would reach the firmware as the literal "nan", raise there
            # and park the motors. Catch it on this side, where the traceback
            # names the control law that produced it.
            raise ValueError("motor %s rate is not finite: %r" % (which, rate))
        limit = float(config.MAX_MOTOR_RATE)
        return max(-limit, min(limit, r))

    # ------------------------------------------------------------------
    #   watchdog accounting
    # ------------------------------------------------------------------
    def ms_since_vel(self):
        """Milliseconds since the last vel the board acknowledged.

        inf if none has ever succeeded. This is the number that matters: the
        firmware watchdog counts from the command it actually received, not
        from the one we queued.
        """
        with self._stat_lock:
            t = self._last_vel_ok_t
        if t is None:
            return float("inf")
        return (time.perf_counter() - t) * 1000.0

    def vel_overdue(self):
        """True once the command period has exceeded the host budget.

        config.VEL_COMMAND_PERIOD_MAX_MS (150) is well under the firmware's
        400 ms watchdog on purpose: this trips while there is still margin, so
        the GUI can say so before the motors park.
        """
        return self.ms_since_vel() > config.VEL_COMMAND_PERIOD_MAX_MS

    def watchdog_tripped(self):
        """True once the firmware watchdog has certainly parked the motors."""
        return self.ms_since_vel() > config.VEL_WATCHDOG_MS

    def watchdog_margin_ms(self):
        """Milliseconds of slack left before the firmware parks the motors."""
        return config.VEL_WATCHDOG_MS - self.ms_since_vel()

    # ------------------------------------------------------------------
    #   status
    # ------------------------------------------------------------------
    def status(self):
        """A fresh LinkStatus snapshot. Safe to call from any thread."""
        with self._stat_lock:
            st = LinkStatus(
                connected=self._connected,
                port=self._port,
                rtt_ms=self._rtt_ms,
                sent=self._sent,
                errors=self._errors,
                last_error=self._last_error,
                pitch_deg=self._pitch,
                yaw_deg=self._yaw,
            )
        if st.connected and self.servoing and self.vel_overdue():
            # Surfaced through last_error because that is the field the GUI
            # already shows. An overdue command period is not a cosmetic
            # metric: it is the failure that parks the motors mid-track.
            st.last_error = ("vel period %.0f ms > %d ms budget"
                             % (self.ms_since_vel(),
                                config.VEL_COMMAND_PERIOD_MAX_MS))
        return st

    @property
    def rtt_ms_last(self):
        """The most recent round trip, unsmoothed."""
        with self._stat_lock:
            return self._rtt_ms_last

    @property
    def connected(self):
        with self._stat_lock:
            return self._connected

    @property
    def servoing(self):
        with self._stat_lock:
            return self._servo_active

    @property
    def port(self):
        return self._port

    def _publish(self):
        self.status_slot.put(self.status())

    def _report(self, msg):
        if self._on_message:
            self._on_message(msg)

    def _note_error(self, msg):
        with self._stat_lock:
            self._errors += 1
            self._last_error = msg

    def _note_pose(self, reply):
        pose = _parse_pose(reply)
        if pose is not None:
            with self._stat_lock:
                self._pitch, self._yaw = pose

    # ------------------------------------------------------------------
    #   the serial thread
    # ------------------------------------------------------------------
    def _writer_loop(self):
        try:
            self._writer_loop_inner()
        except BaseException as exc:
            # Recorded, reported, and NOT re-raised -- because the `finally`
            # below is the beam's last chance to be told to go off, and a
            # traceback out of a daemon thread would skip nothing but would
            # also tell nobody. laser_path_ok goes False either way, which is
            # what makes the control loop disarm.
            self._note_error("serial writer thread died: %r" % (exc,))
            self._report("serial writer thread died: %r" % (exc,))
        finally:
            # THE BEAM MUST NOT OUTLIVE THE THREAD THAT COMMANDS IT.
            # Whatever ends this thread -- a lost port, stop_event, or an
            # unexpected exception -- it is the last thing that can put `laser
            # off` on the wire from here, so it does that on the way out. After
            # this, set_laser() reports an error instead of silently accepting
            # requests nothing will carry (see _writer_gone).
            with self._stat_lock:
                self._laser_want = False
                board_on = self._laser_is
            if board_on:
                try:
                    # _service_laser() swallows SerialException itself and
                    # returns False, so success is judged on the RESULTING
                    # STATE below, not on whether this raised. force=True: this
                    # is the writer's LAST act, so the retry backoff and the
                    # abandon budget must not be allowed to skip it.
                    self._service_laser(force=True)
                except (serial.SerialException, LinkError) as exc:
                    self._note_error("laser off on the closing writer thread "
                                     "raised: %s" % exc)
                with self._stat_lock:
                    still_on = self._laser_is
                if still_on:
                    self._note_error(
                        "THE WRITER THREAD EXITED WITH THE BEAM ON and could "
                        "not command `laser off`: the port is gone in both "
                        "directions")
                    self._report(
                        "SERIAL LINK LOST WITH THE BEAM ON -- `laser off` "
                        "could not be sent. The firmware velocity watchdog "
                        "does NOT cover the laser. KILL THE LASER SUPPLY.")
                else:
                    self._report("serial writer exiting: beam commanded off "
                                 "first.")
            with self._stat_lock:
                self._writer_gone = True
            self._publish()

    def _writer_loop_inner(self):
        last_seq = 0
        while not self._stop_event.is_set():
            # Woken by send_vel()/set_laser(); the timeout is only a backstop
            # so the loop still notices _stop_event with no traffic at all.
            self._wake.wait(0.02)
            self._wake.clear()

            # Laser FIRST, every pass. An "off" must never wait behind a rate
            # update: the whole interlock exists to make the beam stop before
            # anything else happens.
            if not self._service_laser():
                return

            item, seq = self.vel_slot.get()
            if seq == last_seq or item is None:
                continue                   # nothing new; the slot is not a queue
            last_seq = seq
            item_epoch = None
            if isinstance(item, ControlOutput):
                ra, rb = item.rate_a, item.rate_b
            elif len(item) == 3:
                ra, rb, item_epoch = item
            else:
                ra, rb = item
            if item_epoch is not None:
                with self._stat_lock:
                    stale = item_epoch != self._safety_epoch
                if stale:
                    # A safety action (stop/E-STOP) overtook this rate between
                    # the frame that computed it and now. Sending it would undo
                    # the stop. The zeros stop() re-puts are the newest item.
                    continue
            try:
                self._send_vel_blocking(ra, rb, epoch=item_epoch)
                with self._stat_lock:
                    # ONLY on success: a rejected zero is not a sent zero, and
                    # stop() must not be able to confirm on one.
                    self._vel_seq_sent = seq
            except (serial.SerialException, LinkError) as exc:
                # Recorded and surfaced, not retried: a retry would stack a
                # second command behind the next frame's and reproduce the
                # contention this whole design exists to avoid. The control
                # loop sends again in ~33 ms anyway.
                self._note_error(str(exc))
                if isinstance(exc, serial.SerialException):
                    # The port itself is gone. Stop pretending we have a link.
                    with self._stat_lock:
                        self._connected = False
                    self._publish()
                    self._report("serial link lost: %s" % exc)
                    return
            self._vel_progress.set()
            self._publish()

    def _service_laser(self, force=False):
        """Push one pending laser transition. Returns False if the port died.

        `force=True` bypasses the retry backoff and the abandon budget. Used by
        the shutdown paths -- stop(), close(), the writer's exit -- where this
        is the ONE attempt that will ever be made and there is no next pass to
        defer to. Without it the backoff added for the retry storm would
        silently skip the beam-off at shutdown, which is the exact failure this
        module exists to prevent.

        `laser on force` because the firmware's plain `laser on` refuses while
        the platform is moving, and the interlock has already established that
        the beam may be lit -- with seven conditions that the firmware cannot
        see. Refusing here would mean the laser never fires on a tracked
        target, which is the entire point of the machine.
        """
        with self._stat_lock:
            want, have = self._laser_want, self._laser_is
            if want == have:
                # Settled. Clear the retry accounting so the NEXT transition
                # starts with a full budget rather than inheriting the last
                # one's exhausted count.
                self._laser_next_try = 0.0
                self._laser_on_tries = 0
                return True
            now = time.monotonic()
            if not force and now < self._laser_next_try:
                # Backing off from a rejection. Returning True here is what
                # keeps the vel stream flowing at full rate while we wait --
                # the whole point of the spacing.
                return True
            # Space the NEXT attempt before making this one, so a transact that
            # itself takes time cannot produce back-to-back retries.
            self._laser_next_try = now + _LASER_RETRY_S
            if want:
                self._laser_on_tries += 1
                tries = self._laser_on_tries
            else:
                tries = 0
        cmd = "laser on force" if want else "laser off"
        try:
            reply = self._transact(cmd, timeout=_LASER_TIMEOUT_S,
                                   settle=_LASER_SETTLE_S)
        except serial.SerialException as exc:
            self._note_error("laser: %s" % exc)
            with self._stat_lock:
                self._connected = False
            self._publish()
            self._report("serial link lost while setting the laser: %s" % exc)
            return False
        if reply is None or reply.lower().startswith(("error", "usage error")) \
                or "unknown command" in reply.lower():
            # Do NOT latch _laser_is on a failed command. Leaving them
            # different means the next pass retries -- and for an "off" that
            # retry is exactly what we want.
            detail = "no reply" if reply is None else reply.strip()
            if want and not force and tries >= _LASER_ON_ATTEMPTS:
                # Give up on the "on". The board is dark, which is the safe
                # direction, and hammering it is actively harmful: the firmware
                # answers a refused `laser on force` with safe_state(), which
                # aborts and disables every axis. Latching _laser_want False
                # also stops the control loop's next set_laser(True) from
                # restarting the storm, because set_laser() only wakes the
                # writer on a CHANGE of _laser_want.
                with self._stat_lock:
                    self._laser_want = False
                    self._laser_on_failed = True
                    self._laser_on_tries = 0
                self._note_error(
                    "board refused %r %d times: %s -- ABANDONED, the beam is "
                    "NOT lit" % (cmd, tries, detail))
                self._report(
                    "the board refused `laser on force` %d times (%s). The beam "
                    "is NOT lit. If LASER_ENABLED is False in the FIRMWARE "
                    "config this is expected -- the firmware refuses even with "
                    "`force`, and answers by safe-stating every axis."
                    % (tries, detail))
                self._publish()
                return True
            self._note_error("board refused %r: %s" % (cmd, detail))
            return True
        with self._stat_lock:
            self._laser_is = want
            if want:
                self._laser_on_failed = False
        self._publish()
        return True

    def _send_vel_blocking(self, rate_a, rate_b, epoch=None):
        """One vel round trip. Runs on the serial thread (or at shutdown)."""
        # %.1f, not %.4f: the line is echoed back over the same link, so every
        # byte is paid for twice, and a tenth of a step/s is far below the
        # firmware's own quantisation.
        if (getattr(config, "VEL_DIR_RESYNC_ON_ENTRY", True)
                and not self._vel_mode_entered
                and (rate_a != 0.0 or rate_b != 0.0)):
            self._resync_vel_dir()
        verb = "vel16" if config.DYNAMIC_MICROSTEPPING else "vel"
        cmd = "%s %.1f %.1f" % (verb, rate_a, rate_b)
        t0 = time.perf_counter()
        reply = self._transact(cmd, timeout=_VEL_TIMEOUT_S,
                               settle=_VEL_SETTLE_S)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        if reply is None:
            raise LinkError("no prompt after %r within %.0f ms"
                            % (cmd, _VEL_TIMEOUT_S * 1000.0))
        if "ok" not in reply:
            raise LinkError("board rejected %r: %s" % (cmd, reply.strip()))
        if config.DYNAMIC_MICROSTEPPING:
            gear = re.search(r"\bMS=(8|16) SH=(\d+) GC=([01])\b", reply)
            if gear is None:
                raise LinkError("dynamic velocity ACK missing gear metadata")
            self.last_gear_report = {
                "ack_t": time.perf_counter(), "microstep": int(gear[1]),
                "shifts": int(gear[2]), "pulse_capped": bool(int(gear[3])),
                "wire_divisor": 16, "rate_a16": float(rate_a), "rate_b16": float(rate_b)}
        with self._stat_lock:
            self._vel_mode_entered = True
            self._motors_live = True
            # What the board ACKNOWLEDGED, for stop() to confirm against. Set
            # only past the two raises above, so a rejected command never
            # counts as a sent one.
            self._last_rates_sent = (float(rate_a), float(rate_b))
            self._last_rate_epoch = (self._safety_epoch if epoch is None
                                     else int(epoch))
            self._sent += 1
            self._rtt_ms_last = rtt_ms
            self._rtt_ms = (rtt_ms if self._rtt_ms == 0.0 else
                            (1.0 - _RTT_EMA_ALPHA) * self._rtt_ms
                            + _RTT_EMA_ALPHA * rtt_ms)
            self._last_vel_ok_t = time.perf_counter()

    def _resync_vel_dir(self):
        """Force the board to write both DIR pins before the first real rate.

        Firmware iterations <= 13 cache the velocity-mode direction in
        `_vel_dir` and write the DIR pin only when a rate's sign differs from
        the cache; position moves (homing's coordinated moves) set the pin
        without touching the cache. After boot + homing, motor A's pin says
        reverse while the cache says forward, so a first POSITIVE rate on A
        ran the motor BACKWARDS until the loop happened to ask for a negative
        one. Measured 2026-09-20: run_124411 (whole run) and run_122343 (first
        lunge) -- the "start tracking slews down-right" failure. Iteration 14
        fixes the cache; this is the host-side guard for a board that has not
        been flashed. One negative then one positive tick-sized rate makes the
        firmware see a sign on each motor, from a standstill, once per link.
        """
        for ra, rb in ((-_DIR_RESYNC_RATE, -_DIR_RESYNC_RATE),
                       (_DIR_RESYNC_RATE, _DIR_RESYNC_RATE)):
            verb = "vel16" if config.DYNAMIC_MICROSTEPPING else "vel"
            cmd = "%s %.1f %.1f" % (verb, ra, rb)
            reply = self._transact(cmd, timeout=_VEL_TIMEOUT_S,
                                   settle=_VEL_SETTLE_S)
            if reply is None:
                raise LinkError("DIR resync: no prompt after %r" % cmd)
            if "ok" not in reply:
                raise LinkError("DIR resync: board rejected %r: %s"
                                % (cmd, reply.strip()))
            time.sleep(_DIR_RESYNC_DWELL_S)

    def _transact(self, cmd, timeout, settle):
        """Write one line, read one reply. Returns None if no prompt came back.

        The io lock is the single serialisation point for the port: the writer
        thread and the blocking helpers both go through here, so two commands
        can never be in flight at once.
        """
        if self._ser is None:
            raise LinkError("link is not open")
        with self._io_lock:
            ser = self._ser
            if ser is None:
                # close() ran while we were queued on the lock. Re-checked
                # INSIDE the lock on purpose: the check above is racy, and
                # `None.write(...)` would be an AttributeError that no caller
                # in this module catches.
                raise LinkError("link was closed while the command was queued")
            ser.write((cmd + "\r\n").encode())
            ser.flush()
            text, saw_prompt = _read_reply(ser, timeout, settle)
        if cmd and text.startswith(cmd):
            text = text[len(cmd):].strip()     # drop the console's echo
        if not saw_prompt:
            return None
        return text

    # ------------------------------------------------------------------
    #   non-blocking probe -- safe to call WHILE servoing
    # ------------------------------------------------------------------
    #: Read-only commands `try_probe` will send. Anything not listed is
    #: refused: this path exists to observe a servoing machine, and a command
    #: that changes state has no business jumping the vel stream.
    #: "imu mag" added 2026-09-20. VERIFIED read-only against the firmware
    #: (cli.py, `imu mag`): one mag_raw() burst, a platform.position() read and
    #: a print. No describe(), no accel_live(), nothing that can reconfigure a
    #: part -- so it meets this list's own stated criterion, and it is strictly
    #: LIGHTER than "imu", which is already here and does run describe() at
    #: 80-160 ms. Needed because step_integrity reads magnetometer yaw, which
    #: is the ONLY witness to one motor slipping: on a differential, pitch is
    #: the SUM of the two motor positions and yaw the DIFFERENCE, and gravity
    #: is blind to yaw. Without this the step check loses that axis entirely.
    PROBE_COMMANDS = ("imu fast", "imu", "imu mag", "state", "velmode status")

    _probe_ok = 0
    _probe_dropped = 0

    def try_probe(self, cmd, timeout=0.25, settle=0.008, lock_wait=0.010):
        """One read-only query that YIELDS rather than starving the vel stream.

        `command()` refuses outright while servoing, and for good reason: the
        firmware parks the motors if a `vel` is more than VEL_WATCHDOG_MS late,
        and `_transact` takes the io lock with NO timeout, so a query queued
        behind a slow reply delays every vel behind it. This board has already
        recorded 94 watchdog trips.

        So this does not queue. It tries the lock for `lock_wait` and gives up
        if the writer holds it, counting the miss in `probe_drops`. A dropped
        telemetry sample costs nothing; a starved vel stream parks the turret
        mid-track, which is the exact failure we would be trying to observe.

        Returns the reply text, or None if the lock was busy / no prompt came
        back. Callers must treat None as "no sample", never as "no data".
        """
        base = cmd.strip().lower()
        if base not in self.PROBE_COMMANDS:
            raise ValueError(
                "try_probe is read-only and jumps the vel stream; %r is not in "
                "%s. Use command() with the servo stopped." % (cmd, list(self.PROBE_COMMANDS)))
        if self._ser is None:
            return None
        if not self._io_lock.acquire(timeout=lock_wait):
            self._probe_dropped += 1
            return None
        text, saw_prompt = "", False
        try:
            ser = self._ser
            if ser is None:
                return None
            ser.reset_input_buffer()
            ser.write((cmd + "\r\n").encode())
            ser.flush()
            text, saw_prompt = _read_reply(ser, timeout, settle)
        except Exception:                                  # noqa: BLE001
            saw_prompt = False
        finally:
            # LEAVE THE PORT CLEAN, ALWAYS, AND DO IT INSIDE THE LOCK.
            #
            # If this probe timed out, the board is still writing the rest of
            # its reply. `_transact` does NOT flush before reading, so those
            # bytes become the front of the NEXT reader's buffer -- and the
            # next reader is the control path. Observed exactly once and it
            # cost a run: an `imu` probe timed out at 80 ms against a command
            # that now takes ~160 ms, and startup died with
            #     no STATE json in the reply to 'state'
            # which reads like a board fault and is entirely self-inflicted.
            try:
                if not saw_prompt and self._ser is not None:
                    self._ser.reset_input_buffer()
            except Exception:                              # noqa: BLE001
                pass
            self._io_lock.release()

        if not saw_prompt:
            self._probe_dropped += 1
            return None
        if cmd and text.startswith(cmd):
            text = text[len(cmd):].strip()
        self._probe_ok += 1
        return text

    @property
    def probe_stats(self):
        """(delivered, dropped) -- dropped is lock contention, not an error."""
        return (self._probe_ok, self._probe_dropped)

    # ------------------------------------------------------------------
    #   blocking helpers -- setup only, NEVER while servoing
    # ------------------------------------------------------------------
    def _require_idle(self, what):
        if self._ser is None:
            raise LinkError("link is not open")
        with self._stat_lock:
            active = self._servo_active
        if active:
            raise LinkBusy(
                "refusing %r while servoing: a second request contends with "
                "the vel stream, doubles feedback latency and lets the %d ms "
                "watchdog park the motors. Call stop() first."
                % (what, config.VEL_WATCHDOG_MS))

    def command(self, cmd, timeout=None, settle=None):
        """Send one console command and return its reply text. Blocks.

        For homing and calibration, which run with the servo path stopped.
        """
        self._require_idle(cmd)
        base = cmd.strip().lower()
        default = _BLOCKING_TIMEOUTS.get(base)
        if default is None:
            default = _BLOCKING_TIMEOUTS.get(base.split()[0] if base else "",
                                             _DEFAULT_BLOCKING_TIMEOUT)
        reply = self._transact(cmd,
                               timeout=timeout if timeout is not None
                               else default[0],
                               settle=settle if settle is not None
                               else default[1])
        if reply is None:
            raise LinkError("no reply to %r within the timeout" % cmd)
        low = reply.lower()
        if low.startswith("usage error") or low.startswith("error:") \
                or "unknown command" in low:
            raise LinkError("%r failed: %s" % (cmd, reply.strip()))
        self._note_pose(reply)
        self._publish()
        return reply

    # -- the setup-time command set -------------------------------------
    def level(self, tol_deg=None):
        """Drive to TRUE LEVEL off gravity and set the pitch datum.

        Pitch only: yaw rotates about the gravity vector, so the accelerometer
        cannot see it at all. Takes tens of seconds and moves the platform.
        """
        return self.command("level" if tol_deg is None
                            else "level %.3f" % tol_deg)

    def imu_cal(self):
        """Gyro zero-rate calibration. The payload MUST be still.

        Uncalibrated, the X axis fabricates ~6 deg/s, which reads exactly like
        mechanical drift.
        """
        return self.command("imu cal")

    def imu_mag(self):
        """One `imu mag` round trip -> (mx, my, mz, pitch_deg, yaw_deg)."""
        reply = self.command("imu mag")
        mx, my, mz, pitch, yaw = parse_mag(reply)
        with self._stat_lock:
            self._pitch, self._yaw = pitch, yaw
        self._publish()
        return mx, my, mz, pitch, yaw

    def sethome(self, pitch_deg=None, yaw_deg=None):
        """Set the datum: current pose, or an explicit one."""
        if pitch_deg is None and yaw_deg is None:
            return self.command("sethome")
        if pitch_deg is None or yaw_deg is None:
            raise ValueError("sethome takes both pitch and yaw, or neither")
        return self.command("sethome %.4f %.4f" % (pitch_deg, yaw_deg))

    def aim(self, pitch_deg, yaw_deg, dps=None):
        """Absolute payload move. Accelerates from rest and stops -- never use
        this to track; that is what `vel` is for."""
        cmd = "aim %.4f %.4f" % (pitch_deg, yaw_deg)
        if dps is not None:
            cmd += " %.4f" % dps
        return self.command(cmd)

    def dmove(self, dpitch_deg, dyaw_deg, dps=None):
        """Relative payload move, both axes at once."""
        cmd = "dmove %.4f %.4f" % (dpitch_deg, dyaw_deg)
        if dps is not None:
            cmd += " %.4f" % dps
        return self.command(cmd)

    def yaw(self, deg, dps=None):
        """Pure yaw: both motors opposed."""
        cmd = "yaw %.4f" % deg
        if dps is not None:
            cmd += " %.4f" % dps
        return self.command(cmd)

    def cmd_stop(self):
        """The firmware `stop`: disable everything, drop all outputs.

        This is NOT the tracking stop -- it releases the coils, so a loaded
        tilt axis will fall. `stop()` (zero rates) is what ends a track.
        """
        reply = self.command("stop")
        with self._stat_lock:
            # The firmware's `stop` aborts and disables the axes, so there is
            # nothing left to zero -- and re-sending `vel` at close() would
            # silently re-enable the coils this command just dropped. It does
            # NOT stop the velocity timer, though, so _vel_mode_entered stands
            # and close() still owes a `velmode off`.
            self._motors_live = False
        return reply

    def state(self):
        """Parse the `state` JSON. SETUP ONLY.

        There is no version of this that is safe while servoing, which is why
        it goes through _require_idle like everything else. The tracking loop
        needs no board state: `vel` is fire-and-acknowledge and the pose comes
        from the camera.
        """
        import json
        reply = self.command("state")
        for line in reply.splitlines():
            if line.startswith("STATE "):
                data = json.loads(line[len("STATE "):])
                payload = data.get("payload", {})
                if "pitch" in payload and "yaw" in payload:
                    with self._stat_lock:
                        self._pitch = float(payload["pitch"])
                        self._yaw = float(payload["yaw"])
                    self._publish()
                return data
        raise LinkError("no STATE line in reply: %r" % reply)


# ==========================================================================
#   Connect, report, exit -- without moving anything.
# ==========================================================================
if __name__ == "__main__":
    import sys

    link = TurretLink(on_message=lambda m: print("[link] %s" % m))
    link.start()
    try:
        st = link.state()
        payload = st.get("payload", {})
        vel = st.get("velocity", {})
        print("port        %s" % link.port)
        print("payload     pitch %+.3f  yaw %+.3f"
              % (payload.get("pitch", float("nan")),
                 payload.get("yaw", float("nan"))))
        print("velocity    running=%s tripped=%s age=%s ms trips=%s"
              % (vel.get("running"), vel.get("tripped"),
                 vel.get("age_ms"), vel.get("trips")))
        for name, ax in sorted(st.get("axes", {}).items()):
            print("axis %-5s  %-8s 1/%-2d  pos %+8d  mode=%s"
                  % (name, "ENABLED" if ax.get("enabled") else "disabled",
                     ax.get("microstep", 0), ax.get("position", 0),
                     ax.get("mode")))
        print("driver      %s   max_rate %s   laser_enabled %s"
              % (st.get("driver"), st.get("max_rate"),
                 st.get("laser_enabled")))

        # Round-trip timing without commanding motion: `state` is the only
        # command here, and this whole block runs with the servo path idle.
        t0 = time.perf_counter()
        link.command("cfg")
        print("rtt(cfg)    %.1f ms" % ((time.perf_counter() - t0) * 1000.0))
        print("status      %s" % (link.status(),))
    finally:
        # No vel was ever sent, so close() will not energise the motors.
        link.close()
    sys.exit(0)
