# Dynamic microstepping and fixed-mode comparison

## Faster tracking: dynamic mode

`--dynamic-microstep` now enables automatic 1/16 <-> 1/8 switching and doubles
the narrow tracking angular-speed ceiling. This is different from the fixed
A/B comparison below, which deliberately preserves angular speed.

The implementation is local and offline-tested; it has NOT been deployed or
validated on the mechanism. Deploy the updated firmware first (see below).

```powershell
./.venv/Scripts/python.exe -m turret_host.app --dynamic-microstep --tracking-only --skip-level --record diag/microstep_dynamic
```

Return to baseline with a separate launch after closing the first one:

```powershell
./.venv/Scripts/python.exe -m turret_host.app --microstep 16 --tracking-only --skip-level --record diag/microstep_baseline
```

No new camera/Jacobian/goal-pixel calibration is required solely for this
mode. The host retains fixed 1/16 units and the existing Jacobian. The new
`vel16` firmware command converts EVERY request to the current physical gear,
including commands after a shift. Calibration files are not rewritten.

With the current 1,000 pulse/s baseline:

| Behavior | Fine 1/16 | Coarse 1/8 |
|---|---:|---:|
| Physical pulse cap per motor | 1,000/s | 1,000/s |
| Nominal pure-axis angular cap | 48.75 deg/s | 97.5 deg/s |
| Angular acceleration | unchanged | unchanged |

These are commanded ceilings, not a claim that the loaded motors achieve
them. Wide-only acquisition keeps the old angular cap because its evidence
arrives more slowly. This version does not use 1/4 steps.

Shifting policy is based on the faster motor's demand in 1/16-equivalent
units: request coarse above 700/s for 100 ms; request fine below 400/s for
300 ms, and only once the ramped commanded rate is also below 400/s. There
must be at least 500 ms between shifts. Coarsening is deferred when the
existing phase estimate is unaligned; it is never forced after a timeout.
While deferred, the fine pulse cap remains in effect.

Both driver modes change with the timer ISR excluded and both PIO generators
paused. Travel limits use fixed units, counters retain fractional progress,
and `velmode off` restores 1/16 before homing/rewinding. The existing phase
estimate is dead reckoning, NOT a hardware pulse counter: software tests do
not establish physical translator alignment or absence of shift-induced
motion. First hardware runs must check for jolts, oscillation and lost steps.

For the first comparison, record baseline -> dynamic -> baseline with the
same target motions and range, starting slowly and increasing movement speed.
Use `--tracking-only` during these tests. Judge target retention and error
together with IMU motion and the shutdown return check. A higher requested
speed without delivered motion is not an improvement.

`control.jsonl` now includes `gear_ack`: the last acknowledged physical
divisor, cumulative shifts, firmware pulse-cap flag, canonical requested
rates and `ack_t`. The timestamp matters: it is a previous serial ACK, not
necessarily the command on the same control row. `session.json` records
`DYNAMIC_MICROSTEPPING`, `MICROSTEP_DIVISOR` and the actual host caps.

The host refuses dynamic firmware without protocol version 2. The firmware
refuses legacy `vel` commands while dynamic mode is enabled; use the host's
new protocol, or select `microprofile 16` while idle to leave dynamic mode.

## Fixed-speed comparison

Implemented locally; hardware validation is pending. Use `--microstep 16` or
`--microstep 8` on each application launch. Close the application completely
between runs so only one process owns the serial port.

## Firmware prerequisite

The board must have the updated `cli.py` and new `microprofile.py`. An older
board is rejected before homing; the application never silently falls back to
`ms both 8`. The normal deployment tool is:

```powershell
./.venv/Scripts/python.exe tools/deploy.py --dry-run
./.venv/Scripts/python.exe tools/deploy.py --label fixed-microstep-profiles
```

The deploy tool uploads all changed files in `firmware/current`, including any
pre-existing changes there, and resets the board. Review the dry-run list first.
No firmware was deployed as part of implementing this feature.

## A/B runs

If the old watchdog latch prevents startup, with the app CLOSED:

```powershell
./.venv/Scripts/python.exe tools/send.py "laser off" "velmode on" "velmode off" "state"
```

This clears the velocity latch; it does not test motor delivery. Confirm the
returned state has velocity stopped, both axes idle, and laser off. The
microprofile command refuses a moving or illuminated rig.

Run A:

```powershell
./.venv/Scripts/python.exe -m turret_host.app --microstep 16 --tracking-only --skip-level --record diag/microstep16
```

Run B, after A has fully closed:

```powershell
./.venv/Scripts/python.exe -m turret_host.app --microstep 8 --tracking-only --skip-level --record diag/microstep8
```

`--skip-level` retains the project's existing startup choice. It still runs
the rest of homing; it is not a way to skip setup or verify the starting pose.
Place the mechanism at the same sensible starting pose for both runs.
`--tracking-only` refuses ARM for the entire process. Press START in the GUI.
The recorder creates a timestamped run subdirectory under each requested path.

Use the same lighting, distance and target in each run:

1. Hold the drone stationary for 10 seconds.
2. Make repeatable horizontal passes, then vertical passes, then reversals.
3. Repeat increasingly fast passes, including briefly leaving the narrow view.
4. Close normally and retain the shutdown/step-integrity report.

Two runs give an initial comparison. A third 1/16 run (A/B/A), or a reversed
order repeat, helps distinguish microstepping from warming motors/drivers and
differences in hand movement. Compare visible target retention and reacquisition,
pixel error, saturation, and measured IMU motion versus commanded motion.
Exclude homing, actual travel-limit stops, and false target locks from motor
delivery comparisons. Convert pulse rates using each run's
`session.json` config `AXIS_STEP_DEG`; raw pulse rates are not comparable.

## What is preserved

The initial experiment holds angular limits constant: with the current 1,000
pulse/s baseline, 1/8 uses 500 pulse/s, yielding the same nominal 48.75 deg/s
pure-axis cap. Firmware position rates and accelerations are scaled too;
velocity acceleration is scaled exactly once. This tests whether the motor
follows more reliably/smoothly, not whether requesting twice the speed helps.
A speed-ceiling sweep is a separate hardware experiment after this passes.

The host doubles the loaded Jacobian at 1/8, halves step-based caps/preload,
and updates the angular conversion. Existing untagged Jacobian files are
interpreted as 1/16. New calibration saves record `microstep_divisor`, allowing
either mode to load them correctly. Launching alone never writes calibration.
The GUI's Jacobian calibration uses scaled probe travel. Camera intrinsics,
registration and goal-pixel files are not changed.

Switching occurs only before homing, never during tracking. Before coarsening,
the firmware may move each motor by ONE 1/16 microstep to reach a common driver
translator phase. Homing then establishes the datum. Automatic gearshifting
stays off. The selected mode remains on the board until changed or rebooted;
use `--microstep 16` to return to baseline. Launching without a profile while
the board remains at 1/8 is rejected by the existing homing geometry check.

## Offline verification

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p '*microstepping.py' -v
./.venv/Scripts/python.exe -m turret_host.app --no-hardware --no-gui --microstep 8 --tracking-only --run-seconds 3
```

These exercise scaling and application wiring, not physical motor delivery.
