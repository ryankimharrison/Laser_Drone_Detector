"""Exercise the shuttle sweep against a fake link. No port, no hardware.

A fake that is too obliging proves nothing (see fake-link-hides-the-real-error-type),
so this one models the thing being measured -- it reports a gyro magnitude equal
to the commanded pitch rate scaled by a per-rate delivery function -- and it
refuses anything the real TurretLink would refuse mid-servo.
"""
import importlib.util
import math
import sys
import time

ROOT = r"C:\Users\ryank\Desktop\Laser Turret"
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "bench", ROOT + r"\staged\bench_pitch_delivery.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

# Shrink the sweep so the test runs in seconds.
bench.PLATEAU_TARGET_S = 1.0
bench.REST_S = 0.0
bench.STALL_RATES = (1200, 2400)
bench.SETTLE_S = 0.0
bench.TILT_SAMPLES = 2

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print("%-58s %s %s" % (name, "PASS" if cond else "FAIL", detail))


class FakeLink:
    PROBE = ("imu fast", "imu", "imu mag", "state", "velmode status")

    def __init__(self, deliver):
        self.deliver = deliver
        self.a = self.b = 0.0
        self.sent = 0
        self.refused = []
        self.mismatched = []
        self.t0 = time.perf_counter()

    def send_vel(self, a, b, epoch=None):
        self.a, self.b = float(a), float(b)
        self.sent += 1
        if self.a != self.b:
            self.mismatched.append((self.a, self.b))

    def stop(self):
        self.a = self.b = 0.0

    def command(self, cmd, timeout=5.0):
        return "pads A step=PIO B step=PIO OK"

    def try_probe(self, cmd, timeout=0.25, **kw):
        parts = cmd.split()
        base = parts[0] + (" " + parts[1] if len(parts) > 1 else "")
        if base not in self.PROBE:
            self.refused.append(cmd)
            raise RuntimeError("refused during servo: %s" % cmd)
        if cmd == "imu mag":
            return "MAG 100 200 300 0.0 12.5"
        rate = abs(self.a)
        dps = rate * 0.04875 * self.deliver(rate, time.perf_counter() - self.t0)
        # Pure pitch: the whole rotation on one gyro axis.
        return "IMUF %+.2f %+.2f %+.2f %+.2f %+.2f" % (dps, 0.0, 0.0, 3.0, 0.5)


def leg(deliver, rate, sign=-1.0, start=0.0):
    link = FakeLink(deliver)
    return link, bench.shuttle_leg(link, float(rate), sign, start, lambda s: None)


healthy = lambda r, t: 1.0                                        # noqa: E731

# 1. A leg is ONE traverse: it must end at the envelope, not run to the ceiling.
link, r = leg(healthy, 2400)
check("a leg is one traverse, ending at the envelope",
      not r["timed_out"] and abs(abs(r["end_deg"]) - bench.SHUTTLE_ENVELOPE_DEG) < 3.0,
      "%.1f deg in %.2f s" % (r["end_deg"], r["held_s"]))

# 2. Its duration must match the traverse, not the 4 s ceiling.
want_s = bench.SHUTTLE_ENVELOPE_DEG / (2400 * 0.04875)
check("leg duration matches the commanded traverse",
      abs(r["held_s"] - want_s) < 0.25, "%.2f s vs %.2f expected" % (r["held_s"], want_s))

# 3. MATCHED MOTORS. Anything else is not pure pitch and the metric is invalid.
check("both motors always commanded the same rate", not link.mismatched,
      str(link.mismatched[:2]))

# 4. The ramp is excluded from the plateau.
check("the VEL_ACCEL ramp is blanked out of the plateau",
      r["ramp"] and all(x[0] < bench.RAMP_BLANK_S for x in r["ramp"])
      and all(x[0] >= bench.RAMP_BLANK_S for x in r["plateau"]),
      "%d ramp / %d plateau samples" % (len(r["ramp"]), len(r["plateau"])))

# 5. A stalled axis must read ~0 delivery. Note what this does NOT test: the
#    open-loop integral ends the leg on schedule whatever the mechanism does,
#    so the leg does not time out here. The stall shows up in the GYRO, which
#    is the only instrument in this bench that is not the command itself.
link_s, rs = leg(lambda r_, t: 0.02, 2400)
mags = sorted(x[1] for x in rs["plateau"])
check("a stalled axis reads ~0 delivery, not ~1",
      mags and mags[len(mags) // 2] / (2400 * 0.04875) < 0.05,
      "%.3f" % (mags[len(mags) // 2] / (2400 * 0.04875)) if mags else "no samples")

# 6. Alternating legs return: two legs must end near the start.
l1, r1 = leg(healthy, 1200, sign=-1.0, start=0.0)
l2, r2 = leg(healthy, 1200, sign=+1.0, start=r1["end_deg"])
check("alternating legs are self-unwinding",
      abs(r2["end_deg"]) <= bench.SHUTTLE_ENVELOPE_DEG + 3.0,
      "%.1f -> %.1f -> %.1f deg" % (0.0, r1["end_deg"], r2["end_deg"]))

# 7. Never a command the real link refuses mid-servo.
check("no command the real link would refuse while servoing",
      not link.refused, str(link.refused))

# 8. The watchdog cannot fire.
gap_ms = 1000.0 * r["held_s"] / max(1, link.sent)
check("vel resend faster than the watchdog",
      gap_ms < bench.config.VEL_WATCHDOG_MS,
      "%.0f ms mean gap vs %d ms" % (gap_ms, bench.config.VEL_WATCHDOG_MS))

# 9. A silent IMU must yield no plateau, and _score must return None -- never a
#    confident 0.00 that would read as a stall.
class Deaf(FakeLink):
    def try_probe(self, cmd, timeout=0.25, **kw):
        return None
deaf_leg = bench.shuttle_leg(Deaf(healthy), 1200.0, -1.0, 0.0, lambda s: None)
assert not deaf_leg["plateau"] and not deaf_leg["ramp"]
check("no gyro -> no plateau, and _score refuses to score it",
      bench._score(1200, [], [], 1, 0.0) is None)

# 10. THE WHOLE SWEEP, end to end: two rates, two passes, the rest, the
#     unwinds and the per-rate bookkeeping.
fade = lambda r_, t: max(0.15, 1.0 - 0.25 * t)                    # noqa: E731
link_f = FakeLink(fade)
rows = bench.stall_sweep(link_f, lambda s: None)
check("the sweep runs end to end and scores every rate",
      len(rows) == 2 * len(bench.STALL_RATES),
      "%d rows for %d rates x 2 passes" % (len(rows), len(bench.STALL_RATES)))
if rows:
    check("yaw is recorded at both ends of every rate",
          all(r_["yaw_before"] is not None and r_["yaw_after"] is not None
              for r_ in rows))
    check("pads is recorded for every rate", all(r_["pads"] for r_ in rows))

# 10b. THE THERMAL SIGNATURE, asked of the scorer directly. The end-to-end run
#      above cannot ask this cleanly -- one fake decays across the whole sweep,
#      so by the later rates it has already bottomed out and there is nothing
#      left to fade. This is the claim that actually matters: given samples that
#      fade, does _score report it?
want = 2400 * bench.config.AXIS_STEP_DEG
fading = [(i * 0.05, want * (1.0 - 0.006 * i), [want, 0.0, 0.0])
          for i in range(100)]
sc = bench._score(2400, fading, [], 5, 5.0)
check("_score reports a fading axis as 2nd half below 1st",
      sc["last_half"] < sc["first_half"] - 0.1,
      "first %.2f -> last %.2f" % (sc["first_half"], sc["last_half"]))

flat = [(i * 0.05, want, [want, 0.0, 0.0]) for i in range(100)]
sc2 = bench._score(2400, flat, [], 5, 5.0)
check("_score reports a steady axis as flat and ~1.0",
      abs(sc2["delivery"] - 1.0) < 1e-6
      and abs(sc2["first_half"] - sc2["last_half"]) < 1e-6,
      "delivery %.3f, halves %.3f/%.3f"
      % (sc2["delivery"], sc2["first_half"], sc2["last_half"]))

# 11. The report must not crash on the rows the sweep produces.
try:
    bench.report_stall(rows, lambda s: None)
    ok = True
except Exception as exc:                                          # noqa: BLE001
    ok = False
    print("   report_stall raised: %r" % (exc,))
check("report_stall renders the sweep's own rows", ok)

print("\nALL PASS" if all(PASS) else "\n%d FAILURE(S)" % PASS.count(False))
sys.exit(0 if all(PASS) else 1)
