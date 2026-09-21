"""Fixed and dynamic profiles. Select once, before constructing the application."""
from turret_host import config

_BASE = {name: getattr(config, name) for name in (
    "AXIS_STEP_DEG", "PRELOAD_STEPS", "MAX_MOTOR_RATE",
    "WIDE_MAX_MOTOR_RATE", "MOTOR_MAX_RATE")}


def configure(divisor, dynamic=False):
    if dynamic and divisor != 16:
        raise ValueError("dynamic mode uses canonical 1/16 host units")
    if divisor not in (8, 16):
        raise ValueError("microstep must be 8 or 16")
    config.MICROSTEP_DIVISOR = divisor
    config.DYNAMIC_MICROSTEPPING = bool(dynamic)
    config.AXIS_STEP_DEG = _BASE["AXIS_STEP_DEG"] * 16 / divisor
    for name in ("PRELOAD_STEPS", "MAX_MOTOR_RATE", "WIDE_MAX_MOTOR_RATE",
                 "MOTOR_MAX_RATE"):
        setattr(config, name, _BASE[name] * divisor / 16)
    if dynamic:
        # Same pulse ceiling at 1/8 buys twice the angular speed. Wide-only
        # acquisition retains its old cap; its detections are older.
        config.MAX_MOTOR_RATE = _BASE["MAX_MOTOR_RATE"] * 2


def scale_jacobian(matrix, source_divisor=16):
    import numpy as np
    if source_divisor not in (8, 16):
        raise ValueError("unsupported Jacobian microstep divisor")
    return np.asarray(matrix, dtype=float) * source_divisor / config.MICROSTEP_DIVISOR


def apply_to_board(link, divisor, dynamic=False):
    """Firmware owns paired switching and acceleration. Refuse old firmware."""
    import json
    command = ("microprofile dynamic %d" % int(_BASE["MAX_MOTOR_RATE"])
               if dynamic else "microprofile %d" % divisor)
    reply = link.command(command)
    lines = [line[13:] for line in reply.splitlines()
             if line.startswith("MICROPROFILE ")]
    if len(lines) != 1:
        raise RuntimeError("firmware lacks microprofile support; deploy updated firmware first")
    profile = json.loads(lines[0])
    accel = profile.get("canonical_vel_accel", 0)
    tick = profile.get("tick_ms", 0)
    expected = max(1, int(accel * tick / 1000) * divisor // 16)
    if (profile.get("divisor") != divisor or profile.get("gearshift") is not bool(dynamic)
            or accel <= 0 or tick <= 0
            or profile.get("accel_per_tick") != [expected] * 2):
        raise RuntimeError("firmware microstep/acceleration profile verification failed")
    if dynamic and (profile.get("wire_divisor") != 16
                    or profile.get("pulse_limit") != int(_BASE["MAX_MOTOR_RATE"])
                    or profile.get("protocol") != 2):
        raise RuntimeError("dynamic firmware protocol or pulse limit mismatch")
    state = link.state()
    if any(state.get("axes", {}).get(n, {}).get("microstep") != divisor
           for n in ("pan", "tilt")):
        raise RuntimeError("board axes disagree with requested microstep profile")
    if abs(state.get("payload", {}).get("axis_step_deg", 0) - config.AXIS_STEP_DEG) > 1e-4:
        raise RuntimeError("board and host angular step sizes disagree")
    return profile
