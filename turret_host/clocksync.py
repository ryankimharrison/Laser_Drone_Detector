"""One timeline for the host and the board.

    python -m turret_host.clocksync --port COM5                # 90 s fit
    python -m turret_host.clocksync --port COM5 --duration 300
    python -m turret_host.clocksync --quick                    # offset only

WHY THIS EXISTS
---------------
Every cross-reference the next stage depends on is a claim about SIMULTANEITY:
this IMU sample belongs to that frame, this motor command was in flight while
that frame was exposing. The flight recorder currently timestamps everything
on one host `perf_counter`, which is honest but records WHEN THE HOST ASKED,
not when the thing happened. Between those two sits the serial round trip.

So the board is made to report its own `ticks_ms` (firmware iteration 13, the
`clk` command), and this fits the map from that counter onto host time.

CRISTIAN'S ALGORITHM, MINIMUM-RTT FILTERED
------------------------------------------
    t0 = perf_counter()
    -> "clk"
    <- "CLK <ticks_ms>"
    t1 = perf_counter()

    that tick happened at host time (t0+t1)/2, give or take rtt/2

The midpoint is only fair if the request and the reply took equally long. They
do not: USB CDC schedules in 1 ms frames, the board may be mid-print, and every
one of those delays is ONE-SIDED. It can only make a sample later, never
earlier. Averaging all samples therefore averages in the bias.

The lowest-RTT samples are the ones least contaminated, so the fit uses only
those. This is the standard NTP trick and it is worth a lot here: the spread
between the fastest and slowest round trip is typically several milliseconds,
and the whole error budget for the fit is 2 ms.

WHY A STRAIGHT LINE AND NOT A NUMBER
------------------------------------
Two independent crystals. A single offset measured at startup decays at the
rate of their difference, which for cheap parts is tens of ppm:

    30 ppm over a 10 minute run = 18 ms of drift

against an end-to-end control latency of 66 ms. That is not a rounding error,
it is a quarter of the budget, and it grows the longer the session runs -- so
the fit is `t_host = a*pico_ms + b`, and `a - 1` in ppm is the crystal error.
Collecting samples across the whole duration rather than in one burst at the
start is what makes `a` observable at all.

WHAT THIS DOES NOT PROVE
------------------------
A clock model verified against its own samples is circular. It says the
handshake is self-consistent, not that a frame timestamp means anything.
The external check is the laser fiducial in `measure_latency.py`: flash the
beam at a known tick and confirm the dot lands in the frame this model
predicts. Run that after this.
"""
from __future__ import annotations

# turret_host/types.py shadows the stdlib `types` module for anything run as a
# script from inside this directory. Fix the path before any other import.
import os as _os
import sys as _sys
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _pkg_dir:
    _sys.path[0] = _os.path.dirname(_pkg_dir)

import argparse
import json
import re
import time
from typing import List, NamedTuple, Optional

import numpy as np

from turret_host import calibrate

CAL_DIR = _os.path.join(_pkg_dir, "calibration")
CAL_PATH = _os.path.join(CAL_DIR, "clock.json")

_CLK_RE = re.compile(r"CLK\s+(\d+)")

#: Fraction of samples kept, best round trip first. 0.25 is a compromise: the
#: bias falls as this shrinks, but so does the time baseline the skew is fitted
#: over, and a skew fitted on a handful of clustered points is worse than a
#: slightly biased one fitted on many.
KEEP_FRAC = 0.25

#: ticks_ms is a 30-bit counter on MicroPython and wraps at 2**30 ms, about
#: 12.4 days. Nothing here runs that long, but a log that silently jumps
#: backwards is worse than one that refuses, so the unwrap is explicit.
TICKS_MODULO = 1 << 30


class Sample(NamedTuple):
    pico_ms: int          # board counter, unwrapped
    host_mid: float       # host perf_counter at the midpoint of the exchange
    rtt: float            # seconds; the uncertainty is +/- rtt/2


class ClockModel(NamedTuple):
    a: float              # host seconds per pico millisecond
    b: float              # host seconds at pico_ms == 0
    residual_ms: float    # RMS of the kept samples about the line
    ppm: float            # (a*1000 - 1) * 1e6 -- relative crystal error
    n_kept: int
    n_total: int
    rtt_min_ms: float
    rtt_med_ms: float
    span_s: float

    def to_host(self, pico_ms) -> float:
        """Board ticks -> host perf_counter seconds."""
        return self.a * np.asarray(pico_ms, dtype=float) + self.b

    def to_pico_ms(self, t_host) -> float:
        return (np.asarray(t_host, dtype=float) - self.b) / self.a


def _unwrap(raw: List[int]) -> List[int]:
    """Undo the 2**30 ms wrap. Monotonic in, monotonic out."""
    out, bump, prev = [], 0, None
    for v in raw:
        if prev is not None and v < prev - TICKS_MODULO // 2:
            bump += TICKS_MODULO
        out.append(v + bump)
        prev = v
    return out


def one_sample(ser, timeout: float = 0.25) -> Optional[Sample]:
    """A single bracketed `clk` exchange, or None if it did not come back."""
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(b"clk\r\n")
    ser.flush()

    deadline = t0 + timeout
    chunks = []
    while time.perf_counter() < deadline:
        n = ser.in_waiting
        if n:
            chunks.append(ser.read(n))
            m = _CLK_RE.search(b"".join(chunks).decode("ascii", "replace"))
            if m:
                t1 = time.perf_counter()
                return Sample(int(m.group(1)), 0.5 * (t0 + t1), t1 - t0)
        else:
            time.sleep(0.0005)
    return None


def collect(ser, duration_s: float, burst: int = 12,
            verbose: bool = True) -> List[Sample]:
    """Bursts of samples spread across `duration_s`.

    Bursts rather than a steady trickle: within a burst the samples are close
    enough in time that the fastest one is a good local estimate, and spreading
    the bursts is what gives the skew a long enough baseline to be real.
    """
    out: List[Sample] = []
    t_end = time.perf_counter() + duration_s
    # At least 4 bursts even on a short run, or there is no baseline to fit.
    n_bursts = max(4, int(duration_s / 10.0))
    gap = duration_s / n_bursts
    for i in range(n_bursts):
        got = 0
        for _ in range(burst):
            s = one_sample(ser)
            if s is not None:
                out.append(s)
                got += 1
            time.sleep(0.004)
        if verbose:
            print("  burst %2d/%d  %d samples  best rtt %.2f ms"
                  % (i + 1, n_bursts, got,
                     1e3 * min((s.rtt for s in out[-got:]), default=float("nan"))))
        if i < n_bursts - 1:
            time.sleep(max(0.0, min(gap, t_end - time.perf_counter())))
    return out


def fit(samples: List[Sample], keep_frac: float = KEEP_FRAC) -> ClockModel:
    if len(samples) < 8:
        raise RuntimeError("only %d clock samples; need at least 8 to fit a "
                           "line with any meaning" % len(samples))
    pico = np.array(_unwrap([s.pico_ms for s in samples]), dtype=float)
    host = np.array([s.host_mid for s in samples], dtype=float)
    rtt = np.array([s.rtt for s in samples], dtype=float)

    # Keep the least-contaminated samples. Guard the count so a short run still
    # has enough points either side of the baseline to define a slope.
    k = max(6, int(round(len(samples) * keep_frac)))
    idx = np.argsort(rtt)[:k]
    idx.sort()
    a, b = np.polyfit(pico[idx], host[idx], 1)

    resid = host[idx] - (a * pico[idx] + b)
    return ClockModel(
        a=float(a), b=float(b),
        residual_ms=float(np.sqrt(np.mean(resid ** 2)) * 1e3),
        ppm=float((a * 1000.0 - 1.0) * 1e6),
        n_kept=int(k), n_total=len(samples),
        rtt_min_ms=float(rtt.min() * 1e3),
        rtt_med_ms=float(np.median(rtt) * 1e3),
        span_s=float((pico.max() - pico.min()) / 1e3),
    )


def save(model: ClockModel, path: str = CAL_PATH) -> str:
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({
            "comment": "t_host_perf_counter = a * pico_ticks_ms + b. "
                       "Valid only within one host process -- perf_counter has "
                       "no epoch, so b is meaningless across restarts. Re-run "
                       "clocksync at the start of every session.",
            "a": model.a, "b": model.b,
            "residual_ms": model.residual_ms, "ppm": model.ppm,
            "n_kept": model.n_kept, "n_total": model.n_total,
            "rtt_min_ms": model.rtt_min_ms, "rtt_med_ms": model.rtt_med_ms,
            "span_s": model.span_s,
        }, fh, indent=2)
    return path


def report(m: ClockModel) -> None:
    print("\n%s" % ("-" * 66))
    print("  samples        %d kept of %d, over %.0f s" % (m.n_kept, m.n_total, m.span_s))
    print("  round trip     %.2f ms best, %.2f ms median" % (m.rtt_min_ms, m.rtt_med_ms))
    print("  residual       %.3f ms RMS about the line" % m.residual_ms)
    print("  crystal error  %+.1f ppm" % m.ppm)
    drift_10min = abs(m.ppm) * 1e-6 * 600 * 1e3
    print("  -> a single startup offset would drift %.1f ms over 10 minutes"
          % drift_10min)

    print()
    if m.residual_ms < 2.0:
        print("  GOOD: residual is inside the 2 ms the cross-referencing needs.")
    else:
        print("  RESIDUAL TOO HIGH (%.2f ms, want < 2). The usual cause is USB"
              % m.residual_ms)
        print("  scheduling: if the Pico shares a root hub with a camera, the")
        print("  cameras' isochronous transfers reserve slots in the same 1 ms")
        print("  frames. Move it to a port with no camera on it and re-run.")
    if abs(m.ppm) > 200:
        print("  SUSPICIOUS: %+.0f ppm is far outside what a crystal does."
              % m.ppm)
        print("  Suspect the fit, not the hardware -- too short a baseline, or")
        print("  the round trips are so noisy the slope is fitting jitter.")


_IMUF_TICK_RE = re.compile(r"IMUF\s+(?:[-+\d.]+\s+){5}(\d+)")


def verify(ser, model: ClockModel, n: int = 25) -> dict:
    """Check the model against a command it was NOT fitted on.

    The fit used `clk`, which is deliberately the cheapest thing the board can
    do. `imu fast` is the opposite: it stamps its tick AFTER two I2C burst
    reads. So bracketing it with host timestamps gives two independent facts to
    check, and the second is the one that makes this a real test rather than a
    restatement:

      1. the mapped tick must land INSIDE the host bracket -- if the model is
         wrong by more than the round trip, it will not.

      2. it must land LATE in that bracket, not in the middle. The I2C work
         happens before the stamp, so the tick is closer to the reply than to
         the request. A model that were merely fitting the midpoint of every
         exchange would put it at 50%. Seeing it sit well past that is
         evidence the mapping tracks real board time, not an artefact of how
         the samples were collected.

    Reported as a position in the bracket: 0.0 at the write, 1.0 at the reply.
    """
    inside, pos, off = 0, [], []
    for _ in range(n):
        ser.reset_input_buffer()
        t0 = time.perf_counter()
        ser.write(b"imu fast\r\n")
        ser.flush()
        buf, tick, t1 = [], None, None
        deadline = t0 + 1.0
        while time.perf_counter() < deadline:
            k = ser.in_waiting
            if k:
                buf.append(ser.read(k))
                m = _IMUF_TICK_RE.search(b"".join(buf).decode("ascii", "replace"))
                if m:
                    t1 = time.perf_counter()
                    tick = int(m.group(1))
                    break
            else:
                time.sleep(0.0005)
        if tick is None or t1 is None:
            continue
        t_mapped = float(model.to_host(tick))
        span = t1 - t0
        if t0 <= t_mapped <= t1:
            inside += 1
        pos.append((t_mapped - t0) / span if span > 0 else float("nan"))
        # Signed distance outside the bracket; 0 when inside.
        off.append(0.0 if t0 <= t_mapped <= t1
                   else (t_mapped - t1 if t_mapped > t1 else t_mapped - t0))
    return {"n": len(pos), "inside": inside,
            "pos_med": float(np.median(pos)) if pos else float("nan"),
            "worst_ms": float(np.max(np.abs(off)) * 1e3) if off else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--duration", type=float, default=90.0,
                    help="seconds to spread samples over; the skew needs a "
                         "long baseline to be observable (default 90)")
    ap.add_argument("--quick", action="store_true",
                    help="8 s, offset only -- do not trust the ppm from this")
    a = ap.parse_args(argv)

    dur = 8.0 if a.quick else a.duration
    link = (calibrate.CalibrationLink(port=a.port) if a.port
            else calibrate.CalibrationLink()).open()
    try:
        print("collecting clock samples over %.0f s..." % dur)
        samples = collect(link.ser, dur)
        if not samples:
            print("\nNO REPLIES. `clk` arrived in firmware iteration 13 -- an")
            print("older board silently does nothing. Check `imu fast` ends in")
            print("a tick; if it does not, deploy the firmware first.")
            return 1
        model = fit(samples)
    finally:
        link.close()

    report(model)

    # Re-open only for the verify pass; the fit above closed the link in its
    # finally, and leaving it open across the report would hold the port for
    # no reason.
    link = (calibrate.CalibrationLink(port=a.port) if a.port
            else calibrate.CalibrationLink()).open()
    try:
        v = verify(link.ser, model)
    finally:
        link.close()
    print("\nVERIFY against `imu fast` (a command the fit never saw)")
    print("  mapped tick inside the host bracket   %d/%d" % (v["inside"], v["n"]))
    print("  position in bracket (median)          %.2f   [0=write, 1=reply]"
          % v["pos_med"])
    if v["inside"] < v["n"]:
        print("  worst excursion outside                %.2f ms" % v["worst_ms"])
    if v["inside"] == v["n"] and v["pos_med"] > 0.5:
        print("  GOOD: every tick lands inside, and late in the bracket -- the")
        print("  I2C reads happen before the stamp, which is what should show.")
    elif v["inside"] < v["n"]:
        print("  FAILED: a mapped tick outside the bracket is impossible if the")
        print("  model is right. Re-run the fit; if it persists the board's")
        print("  tick and the host clock are not linearly related.")

    if not a.quick:
        print("\nsaved %s" % save(model))
    else:
        print("\n--quick: NOT saved. The ppm from an 8 s baseline is noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
