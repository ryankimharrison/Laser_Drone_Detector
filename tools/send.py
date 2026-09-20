"""Send commands to the turret console and print replies.

    python tools/send.py state
    python tools/send.py "imu cal" imu
    python tools/send.py --timeout 12 "imu cal"

Resolves the board by VID/PID (2e8a:0005) -- never by COM number, which moves.
Only one process may hold the port, so nothing else may be attached.
"""
import argparse
import sys
import time

import serial
import serial.tools.list_ports

VID, PID = 0x2E8A, 0x0005
PROMPT = b"turret>"


def find_port():
    for p in serial.tools.list_ports.comports():
        if p.vid == VID and p.pid == PID:
            return p.device
    return None


def read_reply(ser, timeout, settle=0.4):
    """Read until the prompt returns, or until the board goes quiet."""
    buf, deadline, last = bytearray(), time.time() + timeout, time.time()
    while time.time() < deadline:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            last = time.time()
            if buf.rstrip().endswith(PROMPT):
                break
        else:
            if buf and time.time() - last > settle:
                break
            time.sleep(0.02)
    text = buf.decode("utf-8", "replace")
    # Drop the echoed command line and the trailing prompt.
    lines = [ln for ln in text.splitlines() if ln.strip() != "turret>"]
    return "\n".join(lines).strip()


def at_repl(port):
    """True if the board is at the bare REPL rather than the turret console."""
    with serial.Serial(port, 115200, timeout=0.3) as ser:
        time.sleep(0.25)
        ser.reset_input_buffer()
        ser.write(b"\r\n")
        time.sleep(0.4)
        banner = ser.read(ser.in_waiting or 1).decode("utf-8", "replace")
    if PROMPT.decode() in banner:
        return False
    return ">>>" in banner


def reset_into_console(port, quiet=False):
    """Reboot the board so main.py runs and starts the console.

    main.py is __main__-guarded, so importing it is not enough -- only a real
    reset starts the CLI. machine.reset() re-enumerates the USB CDC device, so
    the handle dies and the COM port disappears and comes back; reconnect after.
    Safe: main.py parks every pin before the CLI starts.
    """
    if not quiet:
        print("[board at REPL -- resetting into the console]")
    try:
        with serial.Serial(port, 115200, timeout=0.3) as ser:
            ser.write(b"import machine; machine.reset()\r\n")
            ser.flush()
            time.sleep(0.2)
    except serial.SerialException:
        pass  # the reset kills the handle -- expected

    deadline = time.time() + 15
    while time.time() < deadline:
        time.sleep(0.4)
        p = find_port()
        if not p:
            continue
        try:
            with serial.Serial(p, 115200, timeout=0.3) as ser:
                time.sleep(0.3)
                ser.reset_input_buffer()
                ser.write(b"\r\n")
                time.sleep(0.5)
                if PROMPT in ser.read(ser.in_waiting or 1):
                    return p
        except serial.SerialException:
            continue
    raise SystemExit("board did not return a turret> prompt after reset")


def send(commands, timeout=6.0, port=None, quiet=False, settle=0.4):
    port = port or find_port()
    if not port:
        raise SystemExit("no MicroPython board (2e8a:0005) found")
    if at_repl(port):
        port = reset_into_console(port, quiet=quiet)
    out = []
    with serial.Serial(port, 115200, timeout=0.3) as ser:
        time.sleep(0.25)
        ser.reset_input_buffer()
        for cmd in commands:
            ser.write((cmd + "\r\n").encode())
            ser.flush()
            reply = read_reply(ser, timeout, settle=settle)
            if reply.startswith(cmd):
                reply = reply[len(cmd):].strip()
            out.append((cmd, reply))
            if not quiet:
                print("=== %s ===" % cmd)
                print(reply if reply else "(no output)")
                print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commands", nargs="+")
    ap.add_argument("--timeout", type=float, default=6.0,
                    help="seconds to wait for each reply (imu cal needs ~10)")
    ap.add_argument("--settle", type=float, default=0.4,
                    help="quiet gap that ends a reply; raise it for commands "
                         "that pause while the platform moves (level, home)")
    ap.add_argument("--port")
    args = ap.parse_args()
    send(args.commands, timeout=args.timeout, port=args.port, settle=args.settle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
