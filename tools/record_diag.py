"""Synchronised diagnostic recorder: board telemetry + camera frames + timestamps.

READ-ONLY with respect to the board. It sends exactly two commands, `imu` and
`state`, both of which are informational. It never commands motion, never homes,
and never touches the laser. Safe to run with the turret powered and armed --
though there is no reason to have it armed for this.

Everything is stamped with time.perf_counter(), which is the same clock
cameras.CameraThread puts in Frame.t, so board samples and frames land on one
timeline and can be joined directly. session.json records the perf_counter ->
wall-clock anchor so the recording can also be placed in real time.

Outputs, under --out:

  session.json     metadata, clock anchor, config snapshot, closing summary
  telemetry.jsonl  one record per board poll (gyro, tilt, positions, watchdog)
  frames.jsonl     one record per frame THIS SCRIPT OBSERVED (see below)
  frames/          JPEGs, subsampled to --save-fps

DO NOT read frame loss out of frames.jsonl. Frames are handed over through a
newest-wins Slot, so a sampler that stalls -- and writing a 1920x1080 JPEG
stalls it for ~15 ms -- silently misses frames that the camera delivered
perfectly. Measured: a poll loop like this one reports 26 fps and "36% of
frames lost" against a camera that CameraThread's own counters show delivering
a clean 30.00 fps with zero failed grabs.

The authoritative delivery numbers are CameraThread.frames / .fps /
.failed_grabs / .max_gap_ms, sampled into telemetry.jsonl as cmd="camera" and
summarised under summary.cameras[*].thread_counters. Trust those.

Usage:
    python tools/record_diag.py --seconds 60
    python tools/record_diag.py --seconds 30 --save-fps 2 --no-board
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import serial                                            # noqa: E402
import serial.tools.list_ports                           # noqa: E402

from turret_host import config                           # noqa: E402

# Only informational commands are ever written to the port. Keep it that way:
# this script is meant to be safe to run without reading it first.
SAFE_COMMANDS = ("imu", "state")

RATE_RE = re.compile(r"rate now\s+([-+\d.]+),\s*([-+\d.]+),\s*([-+\d.]+)")
TILT_RE = re.compile(r"tilt now\s+([-+\d.]+) pitch,\s*([-+\d.]+) roll")
BIAS_RE = re.compile(r"gyro bias\s+([-+\d.]+),\s*([-+\d.]+),\s*([-+\d.]+)")
FOUND_RE = re.compile(r"(gyro|accel|mag)\s+\S+\s+0x[0-9A-Fa-f]{2}\s+(ok|NOT FOUND|not found)")


# ==========================================================================
#   board
# ==========================================================================
def find_port() -> Optional[str]:
    for p in serial.tools.list_ports.comports():
        if p.vid == config.PICO_VID and p.pid == config.PICO_PID:
            return p.device
    return None


def read_until_prompt(ser: serial.Serial, timeout: float = 1.5) -> str:
    """Read until the `turret>` prompt comes back, or timeout.

    Much faster than a fixed settle: the board answers `imu` in a few ms and a
    quiet-period heuristic would spend most of the budget waiting to be sure.
    """
    buf = bytearray()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            if b"turret>" in buf:
                break
        else:
            time.sleep(0.002)
    return buf.decode("utf-8", "replace")


def parse_imu(text: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    m = RATE_RE.search(text)
    if m:
        out["gyro_dps"] = [float(g) for g in m.groups()]
    m = TILT_RE.search(text)
    if m:
        out["tilt_deg"] = [float(g) for g in m.groups()]
    m = BIAS_RE.search(text)
    if m:
        out["gyro_bias_lsb"] = [float(g) for g in m.groups()]
    present = {k: v for k, v in FOUND_RE.findall(text)}
    if present:
        out["present"] = {k: (v == "ok") for k, v in present.items()}
    return out


def parse_state(text: str) -> Dict[str, object]:
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return {}
    try:
        return json.loads(text[i:j + 1])
    except ValueError:
        return {}


class BoardPoller(threading.Thread):
    """Alternates `imu` and `state`, appending one record per poll."""

    def __init__(self, port: str, sink: Path, echo: bool = True):
        super().__init__(name="board-poll", daemon=True)
        self.port = port
        self.sink = sink
        self.echo = echo
        self.records: List[dict] = []
        self.error: Optional[str] = None
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        try:
            with serial.Serial(self.port, config.SERIAL_BAUD, timeout=0.3) as ser, \
                    self.sink.open("w", encoding="utf-8") as fh:
                time.sleep(0.3)
                ser.reset_input_buffer()
                ser.write(b"\r\n")
                read_until_prompt(ser)

                i = 0
                while not self._stop_evt.is_set():
                    cmd = SAFE_COMMANDS[i % len(SAFE_COMMANDS)]
                    i += 1
                    ser.reset_input_buffer()
                    t = time.perf_counter()
                    ser.write((cmd + "\r\n").encode())
                    ser.flush()
                    reply = read_until_prompt(ser)
                    rec: Dict[str, object] = {"t": t, "cmd": cmd,
                                              "rtt_ms": (time.perf_counter() - t) * 1000.0}

                    if cmd == "imu":
                        rec.update(parse_imu(reply))
                    else:
                        st = parse_state(reply)
                        if st:
                            pay = st.get("payload", {})
                            vel = st.get("velocity", {})
                            axes = st.get("axes", {})
                            rec["payload"] = pay
                            rec["velocity"] = vel
                            rec["laser"] = st.get("outputs", {}).get("laser")
                            rec["laser_enabled"] = st.get("laser_enabled")
                            rec["endstops"] = st.get("endstops")
                            rec["axis_pos"] = {k: axes.get(k, {}).get("position")
                                               for k in axes}
                            rec["axis_fault"] = {k: axes.get(k, {}).get("fault")
                                                 for k in axes}
                        else:
                            rec["parse_error"] = reply[-200:]

                    self.records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    if self.echo:
                        _echo(rec)
                    self._stop_evt.wait(0.05)
        except Exception as exc:                          # noqa: BLE001
            self.error = "%s: %s" % (type(exc).__name__, exc)


def _echo(rec: dict) -> None:
    """One compact live line per poll, so the stream is watchable as it records."""
    t = rec["t"]
    if rec["cmd"] == "imu":
        g = rec.get("gyro_dps")
        tilt = rec.get("tilt_deg")
        print("[%8.3f] imu   gyro %s   tilt %s"
              % (t,
                 "%+7.2f %+7.2f %+7.2f" % tuple(g) if g else "    ?",
                 "%+6.2f %+6.2f" % tuple(tilt) if tilt else "    ?"))
    else:
        pay = rec.get("payload") or {}
        vel = rec.get("velocity") or {}
        print("[%8.3f] state pitch %+8.3f yaw %+9.3f   laser %-5s  "
              "vel_running %-5s trips %s"
              % (t, pay.get("pitch", float("nan")), pay.get("yaw", float("nan")),
                 rec.get("laser"), vel.get("running"), vel.get("trips")))


# ==========================================================================
#   cameras
# ==========================================================================
class FrameRecorder:
    """Records every frame's timing; writes a subsample of the pixels."""

    def __init__(self, out_dir: Path, save_fps: float, quality: int = 85):
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.sink = (out_dir / "frames.jsonl").open("w", encoding="utf-8")
        self.save_period = (1.0 / save_fps) if save_fps > 0 else float("inf")
        self.quality = quality
        self.counts: Dict[str, int] = {}
        self.saved: Dict[str, int] = {}
        self.times: Dict[str, List[float]] = {}
        self._last_save: Dict[str, float] = {}
        self._last_seq: Dict[str, int] = {}

    def offer(self, cam_name: str, frame, seq: int) -> None:
        """Record `frame` if it is one we have not seen (slot may repeat)."""
        if frame is None or self._last_seq.get(cam_name) == seq:
            return
        self._last_seq[cam_name] = seq

        import cv2
        self.counts[cam_name] = self.counts.get(cam_name, 0) + 1
        self.times.setdefault(cam_name, []).append(frame.t)

        rec = {"t": frame.t, "camera": cam_name, "index": frame.index,
               "seq": seq, "shape": list(frame.image.shape), "file": None}

        last = self._last_save.get(cam_name, float("-inf"))
        if frame.t - last >= self.save_period:
            self._last_save[cam_name] = frame.t
            name = "%s_%06d_%.6f.jpg" % (cam_name, frame.index, frame.t)
            cv2.imwrite(str(self.frames_dir / name), frame.image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            rec["file"] = "frames/" + name
            self.saved[cam_name] = self.saved.get(cam_name, 0) + 1

        self.sink.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self.sink.close()

    def summary(self) -> Dict[str, object]:
        """Observed-by-this-sampler timing. NOT a frame-loss measurement.

        Every field here is downstream of a newest-wins Slot and of this
        script's own stalls. `gaps_over_*` counts how stale the newest frame
        got from the SAMPLER's point of view, which is a useful thing to know
        about a consumer and a useless thing to say about a camera.
        """
        out: Dict[str, object] = {"_warning": "sampler-observed, not camera truth "
                                              "-- see thread_counters"}
        for cam, ts in self.times.items():
            gaps = [(b - a) * 1000.0 for a, b in zip(ts, ts[1:])]
            span = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            out[cam] = {
                "frames": self.counts.get(cam, 0),
                "saved": self.saved.get(cam, 0),
                "span_s": round(span, 3),
                "mean_fps": round((len(ts) - 1) / span, 2) if span > 0 else 0.0,
                "gap_ms_mean": round(statistics.fmean(gaps), 2) if gaps else 0.0,
                "gap_ms_max": round(max(gaps), 2) if gaps else 0.0,
                "gap_ms_p95": (round(sorted(gaps)[int(len(gaps) * 0.95)], 2)
                               if len(gaps) >= 20 else None),
                "gaps_over_watchdog": sum(1 for g in gaps if g > config.VEL_WATCHDOG_MS),
                "gaps_over_cmd_budget": sum(1 for g in gaps
                                            if g > config.VEL_COMMAND_PERIOD_MAX_MS),
            }
        return out


# ==========================================================================
#   main
# ==========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--save-fps", type=float, default=4.0,
                   help="JPEG write rate per camera; 0 saves none. "
                        "Timestamps are recorded for EVERY frame regardless.")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--no-board", action="store_true")
    p.add_argument("--no-cameras", action="store_true")
    p.add_argument("--wide-fast", action="store_true",
                   help="wide camera at 1280x720@60 (centre crop)")
    p.add_argument("--quiet", action="store_true", help="no live echo")
    args = p.parse_args(argv)

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out) if args.out else _ROOT / "diag" / stamp
    out.mkdir(parents=True, exist_ok=True)
    print("recording to %s\n" % out)

    t_perf0, t_wall0 = time.perf_counter(), time.time()
    session: Dict[str, object] = {
        "started_wall": t_wall0,
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clock_anchor": {"perf_counter": t_perf0, "wall": t_wall0,
                         "note": "wall = t_wall0 + (t_perf - t_perf0)"},
        "args": vars(args),
        "config": {
            "NARROW_SIZE": list(config.NARROW_SIZE), "NARROW_FPS": config.NARROW_FPS,
            "WIDE_SIZE": list(config.WIDE_SIZE), "WIDE_FPS": config.WIDE_FPS,
            "VEL_WATCHDOG_MS": config.VEL_WATCHDOG_MS,
            "VEL_COMMAND_PERIOD_MAX_MS": config.VEL_COMMAND_PERIOD_MAX_MS,
            "LASER_ENABLED_host": config.LASER_ENABLED,
            "NARROW_ROTATION_DEG": config.NARROW_ROTATION_DEG,
            "NARROW_ROTATE_CLOCKWISE": config.NARROW_ROTATE_CLOCKWISE,
        },
        "commands_sent_to_board": list(SAFE_COMMANDS),
    }

    poller: Optional[BoardPoller] = None
    if not args.no_board:
        port = find_port()
        if not port:
            print("WARNING: no board at %04X:%04X -- continuing without telemetry"
                  % (config.PICO_VID, config.PICO_PID))
        else:
            print("board on %s (read-only: %s)\n" % (port, ", ".join(SAFE_COMMANDS)))
            session["port"] = port
            poller = BoardPoller(port, out / "telemetry.jsonl", echo=not args.quiet)
            poller.start()

    cams = []
    recorder: Optional[FrameRecorder] = None
    if not args.no_cameras:
        from turret_host import cameras
        print("identifying cameras...")
        ident = cameras.identify_cameras()
        print(ident.report())
        session["cameras"] = {"narrow_index": ident.narrow_index,
                              "wide_index": ident.wide_index}

        for name, index, exposure, gain in (
                ("narrow", ident.narrow_index, -6, 64),
                ("wide", ident.wide_index, -6, None)):
            res = cameras.lock_exposure(index, exposure, gain=gain)
            print("  %s exposure lock: %s" % (name, res.message or
                                              ("ok" if res.ok else "did not take")))

        # Sequentially. MSMF does not tolerate a raced second open.
        print("opening narrow (index %d)..." % ident.narrow_index)
        narrow = cameras.narrow_thread(ident.narrow_index).start()
        print("  narrow: %.1f fps at startup" % narrow.startup_fps)
        print("opening wide (index %d)..." % ident.wide_index)
        wide = cameras.wide_thread(ident.wide_index, fast=args.wide_fast).start()
        print("  wide:   %.1f fps at startup" % wide.startup_fps)
        cams = [("narrow", narrow), ("wide", wide)]
        session["startup_fps"] = {n: c.startup_fps for n, c in cams}
        recorder = FrameRecorder(out, args.save_fps)
        # CameraThread.frames counts from thread start, which includes the
        # startup fps probe and whatever ran while the other camera was being
        # opened. Only the delta over the recording window is a rate.
        cam_baseline = {n: (c.frames, c.failed_grabs) for n, c in cams}
        cam_t0 = time.perf_counter()

    print("\nrecording for %.0f s -- Ctrl-C to stop early\n" % args.seconds)
    deadline = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < deadline:
            if recorder is not None:
                for name, cam in cams:
                    frame, seq = cam.slot.get()
                    recorder.offer(name, frame, seq)
                time.sleep(0.004)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\ninterrupted -- closing out cleanly")

    # -- shutdown ---------------------------------------------------------
    # Counters BEFORE stop(): .fps is a rolling window that decays to 0 once
    # the grab loop exits, so reading it after stopping reports 0.0 for a
    # camera that was running perfectly.
    cam_final = {}
    if cams:
        cam_dt = time.perf_counter() - cam_t0
        for name, cam in cams:
            f0, g0 = cam_baseline[name]
            cam_final[name] = {
                "frames_in_window": cam.frames - f0,
                "fps_in_window": round((cam.frames - f0) / cam_dt, 2) if cam_dt > 0 else 0.0,
                "fps_rolling": round(cam.fps, 2),
                "startup_fps": round(cam.startup_fps, 2),
                "failed_grabs_in_window": cam.failed_grabs - g0,
                "max_gap_ms": round(cam.max_gap_ms, 1),
                "error": cam.error,
            }

    if poller is not None:
        poller.stop()
        poller.join(timeout=3.0)
        if poller.error:
            print("board poller error: %s" % poller.error)
            session["board_error"] = poller.error
    for _name, cam in cams:
        try:
            cam.stop()
        except Exception:                                 # noqa: BLE001
            pass

    summary: Dict[str, object] = {}
    if recorder is not None:
        recorder.close()
        observed = recorder.summary()
        # The authoritative numbers: CameraThread counts every grab in its own
        # loop, with nothing of this script in the path.
        for name, counters in cam_final.items():
            observed.setdefault(name, {})["thread_counters"] = counters
        summary["cameras"] = observed
    if poller is not None:
        summary["board"] = _board_summary(poller.records)
    session["summary"] = summary
    (out / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("\nwritten to %s" % out)
    return 0


def _board_summary(records: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {"polls": len(records)}
    imu = [r for r in records if r["cmd"] == "imu"]
    state = [r for r in records if r["cmd"] == "state"]
    out["imu_polls"] = len(imu)
    out["state_polls"] = len(state)

    for i, axis in enumerate(("gyro_x", "gyro_y", "gyro_z")):
        col = [r["gyro_dps"][i] for r in imu if r.get("gyro_dps")]
        if col:
            out[axis] = {"min": round(min(col), 3), "max": round(max(col), 3),
                         "mean": round(statistics.fmean(col), 3),
                         "stdev": round(statistics.pstdev(col), 4) if len(col) > 1 else 0.0,
                         "distinct": len(set(col))}
    for i, axis in enumerate(("tilt_pitch", "tilt_roll")):
        col = [r["tilt_deg"][i] for r in imu if r.get("tilt_deg")]
        if col:
            sd = statistics.pstdev(col) if len(col) > 1 else 0.0
            out[axis] = {"min": round(min(col), 3), "max": round(max(col), 3),
                         "mean": round(statistics.fmean(col), 3),
                         "stdev": round(sd, 6), "distinct": len(set(col))}
            # The whole reason this script exists. A live accelerometer cannot
            # produce one repeated value; a dead one reads exactly 0.00 forever.
            if len(col) > 5 and sd == 0.0:
                out[axis]["VERDICT"] = (
                    "ZERO VARIANCE over %d samples at %.2f -- not live data"
                    % (len(col), col[0]))

    trips = [r["velocity"].get("trips") for r in state
             if r.get("velocity") and r["velocity"].get("trips") is not None]
    if trips:
        out["watchdog_trips"] = {"first": trips[0], "last": trips[-1],
                                 "delta": trips[-1] - trips[0]}
    pitches = [r["payload"].get("pitch") for r in state
               if r.get("payload") and r["payload"].get("pitch") is not None]
    if pitches:
        out["payload_pitch"] = {"first": pitches[0], "last": pitches[-1]}
    lasers = [r.get("laser") for r in state if "laser" in r]
    if lasers:
        out["laser_seen_on"] = any(bool(x) for x in lasers)
    rtts = [r["rtt_ms"] for r in records]
    if rtts:
        out["rtt_ms"] = {"mean": round(statistics.fmean(rtts), 2),
                         "max": round(max(rtts), 2)}
    return out


if __name__ == "__main__":
    sys.exit(main())
