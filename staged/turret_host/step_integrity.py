"""Did the motors go where they were told? Note the datum, then rewind to it.

Called by `app.py`: `capture()` once homing has established a datum, and
`verify()` on shutdown. Nothing here ever runs during tracking.

WHAT IT IS FOR
--------------
Every number downstream of the Jacobian assumes a commanded microstep becomes
a degree of payload. Nothing in the system can see the one failure that breaks
that: an A4988 raises no fault when a stepper skips, the step counter keeps
counting regardless, and the error is permanent, silent and cumulative.

So: record where the board says it is at the datum, run the session, then
command EXACTLY the negative of the microsteps it emitted. If nothing slipped,
the payload returns to the pose it started in, and whatever it is off by is
the accumulated loss.

PITCH FROM GRAVITY, YAW FROM THE MAGNETOMETER -- THEY ARE NOT INTERCHANGEABLE
-----------------------------------------------------------------------------
Gravity gives an absolute pitch datum and is **completely blind to yaw**: at
level, rotating in yaw does not move the gravity vector at all. On a
differential that is exactly the wrong blind spot, because pitch is the SUM of
the two motor positions and yaw is their DIFFERENCE -- so BOTH motors losing
steps together shows up in gravity, and ONE motor losing steps does not. One
driver running hot or one belt slipping is the more likely fault, and it is
the one gravity cannot see.

Yaw therefore comes from the magnetometer, against the datum in
`calibration/yaw_reference.json` (slope -3.276 LSB/deg, fitted off the
steppers' own permanent-magnet field). It is coarse -- about 1 deg -- and it
is only trustworthy with the motors idle, which at shutdown they are.

BACKLASH IS NOT STEP LOSS
-------------------------
Measured lash on this mechanism is 0.57 deg, so arriving at the datum from the
opposite side leaves a real offset that no step was lost to. The rewind
overshoots and comes back, approaching from the same direction it left.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Dict, Optional

from turret_host import config

_IMUF = re.compile(r"IMUF\s+\S+\s+\S+\s+\S+\s+([-+\d.]+)\s+([-+\d.]+)")
_MAG = re.compile(r"MAG\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")

YAW_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration", "yaw_reference.json")

#: Overshoot before returning, so lash is taken up in the outbound direction.
LASH_STEPS = 40
#: An 11 deg move rings; a sample taken during the ring measures the ring.
SETTLE_S = 0.5
#: Accelerometer single-sample noise is a MEASURED 0.76 deg. Samples 10 ms
#: apart are NOT independent (the part filters internally), so the mean
#: improves more slowly than sqrt(n) -- do not quote 0.22 deg from 12 of them.
TILT_SAMPLES = 12

#: Tolerances. Gravity's is loose on purpose: a real rewind measured 1.51 deg
#: on gravity while the CAMERA read 0.0 px on the same return, so the
#: accelerometer path is noisier than its per-sample figure suggests.
PITCH_TOL_DEG = 1.8
YAW_TOL_DEG = 3.0


def _gravity_unit(pitch_deg: float, roll_deg: float):
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    return (-math.sin(p), math.sin(r) * math.cos(p), math.cos(r) * math.cos(p))


def _angle_between(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))


def _yaw_slope() -> Optional[float]:
    try:
        with open(YAW_REF_PATH) as fh:
            return float(json.load(fh)["slope_lsb_per_deg"])
    except (OSError, ValueError, KeyError):
        return None


def _read_tilt(link, samples: int = TILT_SAMPLES):
    """(pitch, roll) median, or None.

    None means NO MEASUREMENT, never "level". A standby-latched ADXL345 reads
    exactly -0.00/+0.00, which is a plausible answer and is how a dead sensor
    previously passed as a successful datum.
    """
    ps, rs = [], []
    for _ in range(samples):
        m = _IMUF.search(link.try_probe("imu fast", timeout=0.5) or "")
        if m:
            ps.append(float(m.group(1)))
            rs.append(float(m.group(2)))
        time.sleep(0.01)
    if len(ps) < 3:
        return None
    ps.sort()
    rs.sort()
    return ps[len(ps) // 2], rs[len(rs) // 2]


def _read_mag_y(link) -> Optional[float]:
    vals = []
    for _ in range(5):
        m = _MAG.search(link.try_probe("imu mag", timeout=0.8) or "")
        if m:
            vals.append(float(m.group(2)))     # my -- the yaw-sensitive axis
        time.sleep(0.02)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def _positions(link) -> Optional[Dict[str, int]]:
    try:
        st = link.state()
        ax = st["axes"]
        return {k: int(ax[k]["position"]) for k in ("pan", "tilt")}
    except Exception:                                   # noqa: BLE001
        return None


def _vel_axes(link):
    """Axis names the board still reports in velocity mode, or None.

    None means NO ANSWER -- the same distinction `_read_tilt` makes, and for
    the same reason: "the board did not say" must never be read as "the board
    said no".
    """
    try:
        st = link.state()
    except Exception:                                      # noqa: BLE001
        return None
    axes = st.get("axes") or {}
    if not axes:
        return None
    bad = [n for n, ax in axes.items() if ax.get("mode") == "vel"]
    if (st.get("velocity") or {}).get("running") and not bad:
        # The 200 Hz timer is live even though neither axis admits to it.
        # Still not a state to command a `move` from.
        bad = ["velocity timer"]
    return bad


def capture(link, level_ok: Optional[bool] = None) -> Optional[dict]:
    """Note the datum. Call AFTER homing, with the platform stationary.

    `level_ok` is homing's own gravity verdict (`HomingResult.pitch_datum_ok`).
    It is stored rather than checked here, because a datum taken at a pose that
    is NOT level is still a usable relative reference -- the rewind can still
    say whether the payload came back. What it cannot do is say the payload is
    at zero, and `describe()` refuses to claim that when this is False.
    """
    pos = _positions(link)
    if pos is None:
        return None
    return {"positions": pos, "tilt": _read_tilt(link),
            "mag_y": _read_mag_y(link), "t": time.time(),
            "level_ok": level_ok}


def verify(link, datum: Optional[dict], rate: int = 800) -> Optional[dict]:
    """Rewind to the datum and measure what came back. Returns a report."""
    if not datum:
        return None
    p0 = datum["positions"]
    p1 = _positions(link)
    if p1 is None:
        return {"error": "board did not report its position"}
    delta = {k: p1[k] - p0[k] for k in p0}
    if all(v == 0 for v in delta.values()):
        return {"delta": delta, "moved": False}

    # LEAVE VELOCITY MODE FIRST, HERE, NOT IN THE CALLER.
    #
    # The rewind is a `move`, and the firmware refuses one outright while the
    # axis is in velocity mode: "StepperError: pan is in velocity mode --
    # 'velmode off' before moving". MEASURED on the GUI shutdown path, where
    # the operator closed the window while tracking: _safe_state() zeroes the
    # rates via link.stop() but does NOT issue `velmode off` -- that is owed by
    # link.close(), which runs AFTER this check. So the ordering differs
    # between the headless and GUI shutdown paths, and relying on the caller to
    # have left velocity mode is relying on something that is not true on every
    # path. This function needs move mode to do its job, so it asks for it.
    #
    # Idempotent: firmware do_velmode("off") is velocity.stop() plus a print,
    # with no error when it is already stopped. Also worth knowing, per
    # link.py: leaving velocity mode FOLDS the dead-reckoning accumulator back
    # into the integer step counter, so doing it before the rewind is what
    # makes the counter self-consistent for the residual reported below.
    # DO NOT SWALLOW THIS FAILURE.
    #
    # It used to be `except Exception: pass`, on the reasoning that the moves
    # below would report their own failure. They do not report it usefully:
    # they come back as "no reply ... within the timeout" with nothing about
    # velocity mode, and the report says nothing at all about why the rewind
    # did not happen. Measured on run_2026-09-20_142026.
    #
    # There is a concrete way for this call to fail here: link.command() goes
    # through _require_idle(), which raises LinkBusy while _servo_active is
    # still set -- and on the GUI shutdown path it can still be set at this
    # point. link.stop() is what clears it, so that is the retry.
    velmode_note = None
    try:
        link.command("velmode off", timeout=3.0)
    except Exception as exc:                               # noqa: BLE001
        velmode_note = "velmode off refused (%s)" % exc
        try:
            link.stop()
            link.command("velmode off", timeout=3.0)
            velmode_note += "; recovered after link.stop()"
        except Exception as exc2:                          # noqa: BLE001
            # Still in velocity mode. The firmware will refuse every move
            # below, so running them would only produce a misleading
            # "residual". Skip the rewind and SAY SO -- a check that silently
            # did not run is worse than one that reports it could not.
            return {"error": "could not leave velocity mode, rewind SKIPPED "
                             "-- %s; retry after stop() also failed (%s)"
                             % (velmode_note, exc2),
                    "delta": delta, "moved": False}

    # ASK THE BOARD, DO NOT ASSUME. `velmode off` returning a prompt says the
    # command was accepted; it does not say both axes left velocity mode, and
    # this function is about to issue a `move`, which is the one command that
    # must not be sent in that state -- two step generators on one STEP pad
    # produce a garbled pulse train that loses steps silently, which is the
    # exact fault this whole module exists to detect. Reading it back costs one
    # `state` round trip on a path that is already seconds long.
    still_vel = _vel_axes(link)
    if still_vel is None:
        return {"error": "could not read `state` to confirm the axes left "
                         "velocity mode; rewind SKIPPED rather than issue a "
                         "`move` into an unknown mode",
                "delta": delta, "moved": False,
                "velmode_note": velmode_note}
    if still_vel:
        return {"error": "still in velocity mode after `velmode off` (%s); "
                         "rewind SKIPPED -- a move sharing the STEP pad with "
                         "the velocity state machine loses steps silently"
                         % ", ".join(sorted(still_vel)),
                "delta": delta, "moved": False,
                "velmode_note": velmode_note}

    for k in ("pan", "tilt"):
        if delta[k] == 0:
            continue
        back = -delta[k]
        over = LASH_STEPS if back > 0 else -LASH_STEPS
        link.command("move %s %d %d" % (k, back + over, rate), timeout=60.0)
        time.sleep(0.15)
        link.command("move %s %d %d" % (k, -over, rate), timeout=60.0)
    time.sleep(SETTLE_S)

    rep = {"delta": delta, "moved": True,
           "counter_residual": {k: (_positions(link) or p1)[k] - p0[k]
                                for k in p0}}
    if velmode_note:
        # The rewind DID run, but not on the first ask. Worth recording: it
        # means the caller handed this function a link that still thought it
        # was servoing, which is a bug in the shutdown ordering even though
        # the check recovered from it.
        rep["velmode_note"] = velmode_note

    t0, t2 = datum.get("tilt"), _read_tilt(link)
    rep["level_ok"] = datum.get("level_ok")
    if t0 and t2:
        rep["pitch_off_deg"] = _angle_between(_gravity_unit(*t0),
                                              _gravity_unit(*t2))
        # TWO DIFFERENT QUESTIONS, AND ONLY THE SECOND IS THE ONE ASKED.
        #
        # pitch_off_deg is RELATIVE: did the payload come back to the pose it
        # started in? That is the step-loss measurement.
        #
        # pitch_from_zero_deg is ABSOLUTE: is the payload at true level now?
        # These differ whenever homing did not actually reach zero, and in
        # that case a run can pass the relative check while dead reckoning
        # has been wrong since the first frame. Both get reported; neither
        # substitutes for the other.
        rep["datum_pitch_deg"] = float(t0[0])
        rep["datum_roll_deg"] = float(t0[1])
        rep["final_pitch_deg"] = float(t2[0])
        rep["final_roll_deg"] = float(t2[1])
        rep["pitch_from_zero_deg"] = abs(float(t2[0]))
    m0, m2, slope = datum.get("mag_y"), _read_mag_y(link), _yaw_slope()
    if m0 is not None and m2 is not None and slope:
        rep["yaw_off_deg"] = abs((m2 - m0) / slope)
    return rep


def describe(rep: Optional[dict]) -> list:
    """Lines for the operator. Never raises."""
    if rep is None:
        return []
    if "error" in rep:
        return ["step check: %s" % rep["error"]]
    if not rep.get("moved"):
        return ["step check: the platform never moved; nothing to check."]

    d = rep["delta"]
    out = ["step check: rewound pan %+d, tilt %+d microsteps to the datum"
           % (-d["pan"], -d["tilt"])]
    if rep.get("velmode_note"):
        out.append("  NOTE %s -- the caller left velocity mode on; the "
                   "rewind ran only after a retry" % rep["velmode_note"])
    res = rep.get("counter_residual") or {}
    if any(res.values()):
        out.append("  COUNTER residual %+d/%+d -- the rewind itself was short, "
                   "so the numbers below are not a step-loss measurement"
                   % (res.get("pan", 0), res.get("tilt", 0)))

    bad = False
    measured = 0
    p, y = rep.get("pitch_off_deg"), rep.get("yaw_off_deg")
    if p is None:
        out.append("  PITCH  no IMU reading -- not a pass, just no measurement")
    else:
        measured += 1
        bad |= p > PITCH_TOL_DEG
        out.append("  PITCH  %.2f deg from the datum (gravity, tol %.1f) "
                   "~%.0f microsteps" % (p, PITCH_TOL_DEG,
                                         p / config.AXIS_STEP_DEG))
    if y is None:
        out.append("  YAW    no magnetometer reference -- the axis where ONE "
                   "motor slipping would show is UNCHECKED")
    else:
        measured += 1
        bad |= y > YAW_TOL_DEG
        out.append("  YAW    %.2f deg from the datum (magnetometer, tol %.1f) "
                   "~%.0f microsteps" % (y, YAW_TOL_DEG,
                                         y / config.AXIS_STEP_DEG))

    # ABSOLUTE: where is the payload actually pointing, against true level?
    # Reported separately from the relative result above, because a rewind
    # that returns perfectly to a datum that was never at zero is a PASS on
    # the relative test and a wrong machine.
    z = rep.get("pitch_from_zero_deg")
    if z is not None:
        dat = rep.get("datum_pitch_deg")
        if rep.get("level_ok") is False:
            out.append("  ZERO   payload now %+.2f deg from true level; the "
                       "DATUM ITSELF was %+.2f deg off, so this run never had "
                       "a valid zero and no step-loss claim can be made from it"
                       % (rep.get("final_pitch_deg", 0.0), dat or 0.0))
            bad = True
        else:
            drifted = z > PITCH_TOL_DEG
            bad |= drifted
            out.append("  ZERO   payload now %+.2f deg from true level "
                       "(datum was %+.2f, tol %.1f)%s"
                       % (rep.get("final_pitch_deg", 0.0), dat or 0.0,
                          PITCH_TOL_DEG,
                          "  *** DRIFTED ***" if drifted else ""))

    # NOTHING MEASURED IS NOT A PASS. The per-axis lines above already say so,
    # but the summary line used to contradict them and announce "the payload
    # returned" on a run where no sensor answered -- and `verdict` is now read
    # by scripts, which would take that as a clean bill of health. A dead
    # GY-85 is the single most likely reason both reads return None, and it is
    # precisely when a step-loss claim is worthless.
    if bad:
        out.append("  STEPS LOST -- dead reckoning drifted by the above and "
                   "it accumulates")
        rep["verdict"] = "STEPS LOST"
    elif measured == 0:
        out.append("  UNMEASURED -- no sensor answered, so this run says "
                   "NOTHING about step loss. Not a pass.")
        rep["verdict"] = "UNMEASURED"
    else:
        if measured < 2:
            out.append("  PARTIAL -- only one axis was measured; the other is "
                       "unchecked, not clean.")
        out.append("  the payload returned. No net step loss above the noise.")
        rep["verdict"] = "OK" if measured == 2 else "PARTIAL"
    return out


def write_report(rep: Optional[dict], lines: list, run_dir) -> Optional[str]:
    """Persist the check beside control.jsonl / imu.jsonl. Never raises.

    The report used to exist only as GUI log lines, which meant it could not
    be cross-referenced against the run it describes -- and the standing rule
    for this project is that the frames, the commands and the IMU get read
    together, on one clock. A verdict that evaporates when the window closes
    is not part of that.
    """
    if rep is None or run_dir is None:
        return None
    try:
        path = os.path.join(str(run_dir), "step_check.json")
        blob = dict(rep)
        blob["lines"] = list(lines)
        blob["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        blob["pitch_tol_deg"] = PITCH_TOL_DEG
        blob["yaw_tol_deg"] = YAW_TOL_DEG
        blob["axis_step_deg"] = config.AXIS_STEP_DEG
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True, default=str)
        return path
    except Exception:                                      # noqa: BLE001
        return None
