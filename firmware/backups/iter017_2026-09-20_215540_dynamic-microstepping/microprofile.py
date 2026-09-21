"""Select fixed or dynamic microstepping at idle; preserve angular acceleration."""
import config

_BASE = {name: dict(getattr(config, name))
         for name in ("START_RATE", "MAX_RATE", "ACCEL")}
_VEL_MAX = config.VEL_MAX_RATE


def apply(console, divisor, pulse_limit=1000):
    dynamic = divisor == "dynamic"
    if dynamic:
        divisor = 16
        if not 100 <= pulse_limit <= 4000:
            raise ValueError("dynamic pulse limit must be 100..4000")
    if divisor not in (8, 16):
        raise ValueError("microprofile requires 8 or 16")
    axes = [console.axes[n] for n in ("pan", "tilt")]
    if console.velocity.timer is not None or any(
            ax.mode != "idle" or ax.current_rate or ax.target_rate for ax in axes):
        raise ValueError("microprofile requires idle axes; stop tracking first")
    laser = console.outputs.get("laser")
    if laser is not None and laser.is_on():
        raise ValueError("microprofile requires laser off")
    if any(ax.microstep not in (8, 16) for ax in axes):
        raise ValueError("microprofile requires starting divisors 8 or 16")
    # Align to the common translator phase before coarsening. At most one
    # sixteenth-step per motor; homing follows this command on the host.
    aligned = []
    for ax in axes:
        n = 1 if ax.microstep == 16 and divisor == 8 and ax.phase16 % 2 else 0
        if n:
            result = ax.move(1, rate=100, start_rate=100)
            if result["steps"] != 1 or ax.phase16 % 2:
                raise ValueError("microprofile phase alignment failed")
        aligned.append(n)
    config.VEL_GEARSHIFT = False
    config.VEL_CANONICAL16 = False
    for ax in axes:
        ax.set_microstep(divisor)
        # set_microstep already scales VEL_ACCEL from its canonical 1/16
        # units. Do NOT also halve config.VEL_ACCEL (that would scale twice).
        ax._accel_per_tick = max(1, int(config.VEL_ACCEL * config.VEL_TICK_MS / 1000)
                                * divisor // 16)
    for name, values in _BASE.items():
        for axis, value in values.items():
            getattr(config, name)[axis] = value * divisor // 16
    config.VEL_MAX_RATE = _VEL_MAX * divisor // 16
    if dynamic:
        config.VEL_CANONICAL16 = True
        config.VEL_DYNAMIC_PULSE_LIMIT = pulse_limit
        config.VEL_GEARSHIFT = True
    console.velocity._arm_pitch_limit()
    return {"divisor": divisor, "gearshift": dynamic,
            "wire_divisor": 16 if dynamic else divisor, "protocol": 2,
            "pulse_limit": pulse_limit if dynamic else None,
            "canonical_vel_accel": config.VEL_ACCEL, "tick_ms": config.VEL_TICK_MS,
            "accel_per_tick": [ax._accel_per_tick for ax in axes],
            "alignment_steps16": aligned}
