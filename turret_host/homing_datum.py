"""A stored homing datum, and the cheap check that decides whether to trust it.

    from turret_host.homing_datum import HomingDatum, verify
    d = HomingDatum.load()
    ok, why, obs = verify(d, pitch_deg, yaw_deg, mag)

WHY THIS IS ALLOWED TO EXIST
----------------------------
`homing.py` rebuilds the datum from physics on every boot because there is no
absolute encoder, and that is the right default. But rebuilding and VERIFYING
are different costs: the rebuild is a gyro calibration, an iterative `level`,
and a 33-point magnetometer sweep -- about 36 s. The verification is ONE
`imu mag`, which iter003 added precisely because it returns the field AND the
pose in a single round trip, so the platform cannot move between the two
readings. That is ~43 ms.

So: check first, rebuild only when the check fails.

WHAT THE CHECK ACTUALLY PROVES, AND WHAT IT CANNOT
--------------------------------------------------
Two independent physical witnesses, neither of which the host can fake:

  pitch   gravity, via the IMU tilt. Catches the platform being nudged,
          re-mounted, or left on a different surface.
  yaw     `my` against the stored reference. The magnetometer is inches from
          two steppers and sees their permanent-magnet field, which is FIXED
          IN THE BASE FRAME and rotates in the IMU frame as the payload yaws.
          That is what makes it a yaw witness at all.

It CANNOT detect a change that moves neither: a pure translation of the whole
rig, or a yaw rotation of the base with the payload moving with it. Both leave
the datum physically correct in the payload frame, which is the frame the
control law works in, so neither invalidates it.

EVERY AMBIGUOUS ANSWER MEANS REHOME. No stored datum, an unreadable file, a
missing yaw reference, a disagreeing witness, a stale record, a firmware
iteration bump -- all return False. The expensive path is the safe one, so it
is the default for anything this module is not certain about.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _pkg_dir:
    _sys.path[0] = _os.path.dirname(_pkg_dir)

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

DATUM_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "calibration", "homing_datum.json")

# -- tolerances ------------------------------------------------------------
# Pitch: PITCH_VERIFY_TOL_DEG in homing.py is what a FRESH home is allowed to
# read after its own lash preload, so a stored datum cannot reasonably be held
# to anything tighter. Matched deliberately rather than invented.
PITCH_TOL_DEG = 0.75
# Yaw: the measured trend is ~4.96 LSB/deg on `my`, so 5 LSB is about one
# degree of yaw. Tight enough that a knock is caught, loose enough to survive
# the sensor noise and the rotor-period ripple (~3.07 deg) riding on the trend.
YAW_TOL_LSB = 5.0
# A datum older than this is not trusted. Nothing physical expires, but a
# record that has outlived the session that made it has usually outlived the
# assumptions too -- and rehoming costs 36 s, not an afternoon.
MAX_AGE_S = 12 * 3600.0


@dataclass
class HomingDatum:
    """What a successful home established, and what it takes to re-verify it."""
    pitch_tilt_deg: float             # gravity tilt AT the datum, not the counter
    yaw_reference_lsb: float          # the `my` value that DEFINES yaw zero
    mag_xyz_lsb: Tuple[float, float, float]
    field_magnitude_lsb: float
    gyro_bias_dps: Tuple[float, float, float]
    firmware_iteration: int
    saved_monotonic: float            # time.time() at save
    note: str = ""
    messages: list = field(default_factory=list)

    # -- persistence -------------------------------------------------------
    def save(self, path: str = DATUM_PATH) -> None:
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        _os.replace(tmp, path)        # atomic: never a half-written datum

    @staticmethod
    def load(path: str = DATUM_PATH) -> Optional["HomingDatum"]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            return None
        try:
            return HomingDatum(
                pitch_tilt_deg=float(blob["pitch_tilt_deg"]),
                yaw_reference_lsb=float(blob["yaw_reference_lsb"]),
                mag_xyz_lsb=tuple(float(v) for v in blob["mag_xyz_lsb"]),
                field_magnitude_lsb=float(blob["field_magnitude_lsb"]),
                gyro_bias_dps=tuple(float(v) for v in blob["gyro_bias_dps"]),
                firmware_iteration=int(blob["firmware_iteration"]),
                saved_monotonic=float(blob["saved_monotonic"]),
                note=str(blob.get("note", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None               # malformed -> rehome, never guess


def current_iteration() -> int:
    """firmware/ITERATION, or -1 when it cannot be read.

    A firmware change can move the datum under us -- iter002 alone altered the
    yaw travel limit and the MOSFET flag. -1 never equals a stored value, so an
    unreadable ITERATION forces a rehome rather than trusting a stale record.
    """
    p = _os.path.join(_os.path.dirname(_pkg_dir), "firmware", "ITERATION")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return int(fh.read().strip().split()[0])
    except (OSError, ValueError, IndexError):
        return -1


def verify(datum: Optional[HomingDatum],
           tilt_deg: float,
           mag_my_lsb: float,
           now_s: float,
           iteration: Optional[int] = None) -> Tuple[bool, str, dict]:
    """(trust_it, human reason, observations). False means run a full home.

    `tilt_deg` is the IMU's gravity tilt and `mag_my_lsb` the magnetometer's
    y component -- both from ONE `imu mag`, which is the whole point: read
    separately, the platform could move between them and the two witnesses
    would describe different poses.
    """
    obs = {"tilt_deg": tilt_deg, "my_lsb": mag_my_lsb}
    if datum is None:
        return False, "no stored datum", obs

    it = current_iteration() if iteration is None else iteration
    if it != datum.firmware_iteration:
        return False, ("firmware is iteration %s, datum was taken on %s"
                       % (it, datum.firmware_iteration)), obs

    age = now_s - datum.saved_monotonic
    if age < 0:
        return False, "datum is stamped in the future (clock changed)", obs
    if age > MAX_AGE_S:
        return False, "datum is %.1f h old (limit %.0f h)" % (age / 3600.0,
                                                              MAX_AGE_S / 3600.0), obs
    obs["age_s"] = age

    d_pitch = abs(tilt_deg - datum.pitch_tilt_deg)
    obs["d_pitch_deg"] = d_pitch
    if d_pitch > PITCH_TOL_DEG:
        return False, ("gravity disagrees: tilt %+.2f vs datum %+.2f "
                       "(%.2f > %.2f deg)"
                       % (tilt_deg, datum.pitch_tilt_deg, d_pitch,
                          PITCH_TOL_DEG)), obs

    d_yaw = abs(mag_my_lsb - datum.yaw_reference_lsb)
    obs["d_yaw_lsb"] = d_yaw
    obs["d_yaw_deg_approx"] = d_yaw / 4.96
    if d_yaw > YAW_TOL_LSB:
        return False, ("motor field disagrees: my %+.1f vs datum %+.1f "
                       "(%.1f LSB ~ %.1f deg of yaw)"
                       % (mag_my_lsb, datum.yaw_reference_lsb, d_yaw,
                          d_yaw / 4.96)), obs

    return True, ("datum holds: pitch within %.2f deg, yaw within %.1f LSB "
                  "(~%.2f deg), age %.0f min"
                  % (d_pitch, d_yaw, d_yaw / 4.96, age / 60.0)), obs


def from_result(result, iteration: Optional[int] = None,
                now_s: float = 0.0) -> Optional[HomingDatum]:
    """Build a storable datum from a HomingResult, or None if it is not storable.

    A fast home is deliberately refused: it reports `yaw_homed=False` and its
    yaw datum is "wherever it was pointing", so storing it would let a later
    boot verify against an arbitrary pose and believe it was homed.
    """
    if not getattr(result, "ok", False):
        return None
    if not getattr(result, "yaw_homed", False):
        return None
    ref = getattr(result, "yaw_reference_lsb", None)
    if ref is None or not math.isfinite(float(ref)):
        return None
    samples = getattr(result, "samples", None) or []
    last = samples[-1] if samples else None
    mag_xyz = (float(getattr(last, "mx", 0.0)), float(getattr(last, "my", ref)),
               float(getattr(last, "mz", 0.0))) if last is not None \
        else (0.0, float(ref), 0.0)
    return HomingDatum(
        pitch_tilt_deg=float(result.pitch_tilt_deg),
        yaw_reference_lsb=float(ref),
        mag_xyz_lsb=mag_xyz,
        field_magnitude_lsb=float(getattr(result, "field_magnitude_lsb", 0.0)),
        gyro_bias_dps=tuple(float(v) for v in result.gyro_bias_dps),
        firmware_iteration=current_iteration() if iteration is None else iteration,
        saved_monotonic=now_s,
        note="Written by a FULL home with a real yaw datum. Verified on the "
             "next boot with one `imu mag`; any disagreement rehomes.",
    )
