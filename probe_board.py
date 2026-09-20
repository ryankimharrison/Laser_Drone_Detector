"""Read-only probe of the turret board. Sends only informational commands."""
import sys
import time

import serial
import serial.tools.list_ports

SAFE_COMMANDS = [
    "",  # wake / get a prompt
    "state",
    "velmode status",
    "limits",
    "diff",
    "cfg",
    "findings",
    "imu",
]


def find_port():
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x2E8A and p.pid == 0x0005:
            return p.device
    return None


def drain(ser, settle=0.45):
    """Read until the board goes quiet."""
    chunks, last = [], time.time()
    while time.time() - last < settle:
        n = ser.in_waiting
        if n:
            chunks.append(ser.read(n))
            last = time.time()
        else:
            time.sleep(0.02)
    return b"".join(chunks).decode("utf-8", "replace")


def main():
    port = find_port()
    if not port:
        print("No 2e8a:0005 device found.")
        return 1
    print("port: %s\n" % port)

    with serial.Serial(port, 115200, timeout=0.3) as ser:
        time.sleep(0.3)
        banner = drain(ser, 0.8)
        if banner.strip():
            print("=== unsolicited output on connect ===")
            print(banner.strip())
            print()

        for cmd in SAFE_COMMANDS:
            ser.reset_input_buffer()
            ser.write((cmd + "\r\n").encode())
            ser.flush()
            reply = drain(ser)
            label = cmd if cmd else "<newline>"
            print("=== %s ===" % label)
            print(reply.strip() if reply.strip() else "(no response)")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
