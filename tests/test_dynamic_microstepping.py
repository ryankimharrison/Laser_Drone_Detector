"""Offline tests execute the real firmware methods behind hardware stubs.

They verify software units/transitions; they cannot verify PIO pulse timing
or the real driver's translator phase.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from turret_host import config, control
from turret_host.microstepping import configure, apply_to_board

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    now = 0
    def ticks_ms(self): return self.now
    def ticks_diff(self, a, b): return a - b
    def sleep_us(self, us): pass


class SM:
    def __init__(self): self.enabled = True
    def active(self, value): self.enabled = bool(value)
    def exec(self, instruction): pass


class Axis:
    def __init__(self):
        self.microstep = 16
        self.mode = 'idle'
        self.position = 0
        self._pos_micro = 0
        self.phase16 = 0
        self.target_rate = self.current_rate = 0
        self._accel_per_tick = 200
        self.vel_sm = SM()
        self.period = 0

    @property
    def vel_position(self): return self.position + self._pos_micro // 1000000
    def enter_vel_mode(self): self.mode = 'vel'
    def leave_vel_mode(self):
        self.position += self._pos_micro // 1000000
        self._pos_micro = 0
        self.target_rate = self.current_rate = 0
        self.mode = 'idle'
    def set_microstep(self, div):
        old = self.microstep
        self.position = self.position * div // old
        self._pos_micro = self._pos_micro * div // old
        self.current_rate = self.current_rate * div // old
        self.target_rate = self.target_rate * div // old
        self.microstep = div
        self._accel_per_tick = 200 * div // 16
    def _set_period(self, period): self.period = period
    def _period_for(self, rate): return rate
    def set_target_rate(self, rate): self.target_rate = int(rate)
    def vel_tick(self, dt):
        d = self.target_rate - self.current_rate
        self.current_rate += max(-self._accel_per_tick, min(self._accel_per_tick, d))
        self._pos_micro += self.current_rate * dt


class Platform:
    enabled = True
    def __init__(self, axes): self.axes = axes
    @property
    def microstep(self): return self.axes['pan'].microstep


class FirmwareDynamic(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.irq = []
        self.cfg = types.SimpleNamespace(
            VEL_TICK_MS=5, VEL_CANONICAL16=True, VEL_GEARSHIFT=True,
            VEL_DYNAMIC_PULSE_LIMIT=1000, VEL_WATCHDOG_MS=400,
            FULL_STEPS_PER_REV=200, DIFFERENTIAL_N=2.3077,
            PAYLOAD_LIMIT_DEG={'pitch': (-90, 90)})
        timer = type('Timer', (), {'PERIODIC': 1, 'init': lambda s, **kw: None, 'deinit': lambda s: None})
        env = {'config': self.cfg, 'time': self.clock, 'Timer': timer, '_VEL_SET_LOW': 0,
               'StepperUsageError': ValueError,
               'machine': types.SimpleNamespace(
                   disable_irq=lambda: self.irq.append('off'),
                   enable_irq=lambda token: self.irq.append('on'))}
        tree = ast.parse((ROOT / 'firmware/current/stepper.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VelocityLoop')
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'stepper.py', 'exec'), env)
        self.axes = {n: Axis() for n in ('pan', 'tilt')}
        self.laser = types.SimpleNamespace(value=lambda x: setattr(self, 'laser_value', x))
        self.loop = env['VelocityLoop'](self.axes, self.laser)
        tree = ast.parse((ROOT / 'firmware/current/cli.py').read_text())
        console = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                       and any(isinstance(m, ast.FunctionDef) and m.name == 'do_vel' for m in n.body))
        methods = [m for m in console.body if isinstance(m, ast.FunctionDef)
                   and m.name in ('do_vel', 'dispatch')]
        exec(compile(ast.Module(body=methods, type_ignores=[]), 'cli.py', 'exec'), env)
        self.c = types.SimpleNamespace(
            axes=self.axes, velocity=self.loop,
            platform=Platform(self.axes),
            guard_vel=lambda a, b: '', vel_flags=lambda: '')
        self.c.do_vel = types.MethodType(env['do_vel'], self.c)
        self.c.dispatch = types.MethodType(env['dispatch'], self.c)

    def command16(self, a, b):
        with patch('builtins.print') as out:
            self.c.dispatch(['vel16', str(a), str(b)])
        return out.call_args.args[0]

    def test_repeated_wire_commands_do_not_double_after_shift(self):
        self.command16(1800, 900)
        self.assertEqual(self.axes['pan'].target_rate, 1000)  # cap while fine
        self.clock.now = 110
        ack = self.command16(1800, 900)
        self.assertIn('MS=8', ack)
        self.assertEqual(self.axes['pan'].target_rate, 900)
        for _ in range(8):
            self.clock.now += 33
            self.command16(1800, 900)
            self.assertEqual(self.axes['pan'].target_rate * 16 / self.axes['pan'].microstep, 1800)
            self.assertEqual(self.axes['tilt'].target_rate, 450)

    def test_legacy_wire_command_refused_in_dynamic_mode(self):
        with self.assertRaisesRegex(ValueError, 'vel16'):
            self.c.dispatch(['vel', '1000', '1000'])
        self.assertIsNone(self.loop.timer)

    def test_hysteresis_dwell_and_slow_return(self):
        self.command16(900, 0)
        self.clock.now = 110
        self.command16(900, 0)
        self.assertEqual(self.axes['pan'].microstep, 8)
        for t, rate in [(200, 650), (400, 710), (600, 500)]:
            self.clock.now = t
            self.command16(rate, 0)
        self.assertEqual(self.loop.shifts, 1)
        self.clock.now = 700
        self.command16(100, 0)
        self.clock.now = 999
        self.command16(100, 0)
        self.assertEqual(self.axes['pan'].microstep, 8)
        self.clock.now = 1001
        self.command16(100, 0)
        self.assertEqual(self.axes['pan'].microstep, 16)
        self.assertEqual(self.axes['pan']._accel_per_tick, 200)

    def test_no_forced_unaligned_shift_after_many_commands(self):
        self.axes['pan'].phase16 = 1
        for t in range(0, 5000, 33):
            self.clock.now = t
            self.command16(2000, 1000)
        self.assertEqual(self.loop.shifts, 0)
        self.assertEqual(self.axes['pan'].target_rate, 1000)

    def test_pair_shift_preserves_fraction_and_travel_origin(self):
        for ax in self.axes.values():
            ax.position, ax._pos_micro = 101, 500000
            ax.current_rate = ax.target_rate = 800
        self.loop._tot0 = 150
        self.loop._arm_pitch_limit()
        limit = self.loop._pitch_sum_limit
        for div in (8, 16, 8, 16):
            self.loop._do_shift(div)
            ax = self.axes['pan']
            self.assertEqual((ax.position * 1000000 + ax._pos_micro) * 16 // div, 101500000)
            self.assertEqual(ax.current_rate * 16 // div, 800)
            self.assertEqual(self.loop._tot0, 150)
            self.assertEqual(self.loop._pitch_sum_limit, limit)
        self.assertEqual(self.irq, ['off', 'on'] * 4)

    def test_watchdog_and_guard_still_stop_coarse_motion(self):
        self.command16(1800, 0)
        self.clock.now = 110
        self.command16(1800, 0)
        self.loop._blk_all = True
        self.loop._tick(None)
        self.assertEqual(self.axes['pan'].target_rate, 0)
        self.assertEqual(self.laser_value, 0)
        self.loop._blk_all = False
        self.command16(1800, 0)
        self.clock.now = 600
        self.loop._tick(None)
        self.assertTrue(self.loop.tripped)
        self.assertEqual(self.axes['pan'].target_rate, 0)

    def test_travel_limit_after_shift_with_nonzero_origin(self):
        self.command16(2000, 2000)
        self.clock.now = 110
        self.command16(2000, 2000)
        self.loop._tot0 = 4000
        # Just beyond 90 degrees relative to a nonzero fine-step origin.
        total16 = self.loop._tot0 + self.loop._pitch_sum_limit + 100
        self.axes['pan'].position = total16 // 2
        self.axes['tilt'].position = 0
        self.loop._tick(None)
        self.assertTrue(self.loop.limited)
        self.assertEqual(self.axes['pan'].target_rate, 0)
        self.assertEqual(self.axes['tilt'].target_rate, 0)

    def test_stop_returns_to_fine_before_position_commands(self):
        self.command16(1800, 0)
        self.clock.now = 110
        self.command16(1800, 0)
        self.axes['pan'].position = 50
        self.axes['pan']._pos_micro = 500000
        self.loop.stop()
        self.assertEqual(self.axes['pan'].microstep, 16)
        self.assertEqual(self.axes['pan'].position, 101)
        self.assertEqual(self.axes['pan'].mode, 'idle')


class HostDynamic(unittest.TestCase):
    def tearDown(self): configure(16)

    def test_double_speed_ceiling_same_calibration_and_acceleration_units(self):
        configure(16)
        path = ROOT / 'turret_host/calibration/jacobian.json'
        j = control.Jacobian.load(path).J
        cap, wide = config.MAX_MOTOR_RATE, config.WIDE_MAX_MOTOR_RATE
        configure(16, dynamic=True)
        self.assertEqual(config.MAX_MOTOR_RATE, 2 * cap)
        self.assertEqual(config.WIDE_MAX_MOTOR_RATE, wide)
        self.assertEqual(config.AXIS_STEP_DEG, .04875)
        np.testing.assert_array_equal(control.Jacobian.load(path).J, j)

    def test_cli_excludes_conflicting_modes(self):
        from turret_host.app import build_parser
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            build_parser().parse_args(['--microstep', '8', '--dynamic-microstep'])

    def test_wire_units_and_gear_ack_across_shifts(self):
        from turret_host.link import TurretLink, LinkError
        configure(16, dynamic=True)
        link = TurretLink()
        link._vel_mode_entered = True
        with patch.object(link, '_transact', return_value='ok MS=8 SH=1 GC=0') as tx:
            link._send_vel_blocking(1800, -900)
            self.assertEqual(tx.call_args.args[0], 'vel16 1800.0 -900.0')
            self.assertEqual(link.last_gear_report['microstep'], 8)
            self.assertEqual(link.last_gear_report['rate_a16'], 1800)
            tx.return_value = 'ok MS=16 SH=2 GC=1'
            link._send_vel_blocking(1800, -900)
            self.assertEqual(tx.call_args.args[0], 'vel16 1800.0 -900.0')
            self.assertTrue(link.last_gear_report['pulse_capped'])
            tx.return_value = 'ok'
            with self.assertRaisesRegex(LinkError, 'metadata'):
                link._send_vel_blocking(1800, -900)

    def test_dynamic_handshake_and_old_protocol_rejection(self):
        configure(16, dynamic=True)
        profile = dict(divisor=16, gearshift=True, wire_divisor=16, protocol=2,
                       pulse_limit=config.MAX_MOTOR_RATE // 2,
                       canonical_vel_accel=40000, tick_ms=5, accel_per_tick=[200, 200])
        commands = []
        def command(c):
            commands.append(c)
            return 'MICROPROFILE ' + json.dumps(profile)
        link = types.SimpleNamespace(command=command, state=lambda: {
            'axes': {n: {'microstep': 16} for n in ('pan', 'tilt')},
            'payload': {'axis_step_deg': .04875}})
        apply_to_board(link, 16, dynamic=True)
        self.assertEqual(commands, ['microprofile dynamic %d' % (config.MAX_MOTOR_RATE // 2)])
        profile['protocol'] = 1
        with self.assertRaisesRegex(RuntimeError, 'protocol'):
            apply_to_board(link, 16, dynamic=True)

    def test_dynamic_rewind_refuses_unrestored_board(self):
        from turret_host.step_integrity import verify
        configure(16, dynamic=True)
        calls = []
        link = types.SimpleNamespace(command=lambda c, **k: calls.append(c),
                                     state=lambda: {'axes': {'pan': {'microstep': 8}}})
        report = verify(link, {'positions': {'pan': 0, 'tilt': 0}})
        self.assertIn('error', report)
        self.assertEqual(calls, ['velmode off'])


if __name__ == '__main__': unittest.main()
