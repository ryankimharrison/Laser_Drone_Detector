import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from turret_host import config, control
from turret_host.microstepping import configure, apply_to_board

ROOT = Path(__file__).resolve().parents[1]


class HostProfiles(unittest.TestCase):
    def tearDown(self):
        configure(16)

    def test_equal_physical_motion_and_unchanged_calibration(self):
        path = ROOT / 'turret_host/calibration/jacobian.json'
        original = path.read_bytes()
        configure(16)
        j16 = control.Jacobian.load(path)
        omega16 = j16.motor_rates([200, -100])
        angular16 = control.TravelLimits().axis_rates(omega16)
        cap16 = config.MAX_MOTOR_RATE * config.AXIS_STEP_DEG
        for div in (8, 8, 16, 8):
            configure(div)
            j = control.Jacobian.load(path)
            omega = j.motor_rates([200, -100])
            np.testing.assert_allclose(omega, omega16 * div / 16)
            np.testing.assert_allclose(control.TravelLimits().axis_rates(omega), angular16)
            self.assertAlmostEqual(config.MAX_MOTOR_RATE * config.AXIS_STEP_DEG, cap16)
            self.assertAlmostEqual(config.PRELOAD_STEPS * config.AXIS_STEP_DEG, 14 * .04875)
        self.assertEqual(path.read_bytes(), original)

    def test_saved_eighth_calibration_loads_in_both_modes(self):
        configure(8)
        j = control.Jacobian([[1.0, .8], [-.9, .7]])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'j.json'
            j.save(p)
            np.testing.assert_allclose(control.Jacobian.load(p).J, j.J)
            configure(16)
            np.testing.assert_allclose(control.Jacobian.load(p).J, j.J / 2)

    def test_old_or_mismatched_board_refused(self):
        configure(8)
        link = types.SimpleNamespace(command=lambda c: 'unknown command')
        with self.assertRaisesRegex(RuntimeError, 'deploy'):
            apply_to_board(link, 8)
        profile = {'divisor': 8, 'gearshift': False, 'accel_per_tick': [100, 100],
                   'canonical_vel_accel': 40000, 'tick_ms': 5}
        link.command = lambda c: 'MICROPROFILE ' + json.dumps(profile)
        link.state = lambda: {'axes': {'pan': {'microstep': 8}, 'tilt': {'microstep': 16}}}
        with self.assertRaisesRegex(RuntimeError, 'axes disagree'):
            apply_to_board(link, 8)
        link.state = lambda: {'axes': {n: {'microstep': 8} for n in ('pan', 'tilt')},
                              'payload': {'axis_step_deg': .0975}}
        self.assertEqual(apply_to_board(link, 8), profile)

    def test_tracking_only_blocks_arm(self):
        from turret_host.app import TurretApp, build_parser
        args = build_parser().parse_args(['--microstep', '8', '--tracking-only'])
        fake = types.SimpleNamespace(args=args, log=lambda *a: None)
        TurretApp.on_arm(fake)  # must return before touching interlock or link

    def test_normal_startup_selects_fixed_sixteenth_profile(self):
        from turret_host.app import build_parser
        args = build_parser().parse_args([])
        self.assertEqual(args.microstep, 16)
        self.assertFalse(args.dynamic_microstep)


class Axis:
    def __init__(self):
        self.microstep = 16
        self.phase16 = 1
        self.mode = 'idle'
        self.current_rate = self.target_rate = 0
        self._accel_per_tick = 200
        self.moves = []

    def move(self, n, **kwargs):
        self.moves.append(n)
        self.phase16 += n
        return {'steps': n}

    def set_microstep(self, div):
        self.microstep = div


class FirmwareProfiles(unittest.TestCase):
    def setUp(self):
        cfg = types.ModuleType('config')
        for key, value in [('START_RATE', 200), ('MAX_RATE', 4000), ('ACCEL', 20000)]:
            setattr(cfg, key, dict(pan=value, tilt=value))
        cfg.VEL_MAX_RATE, cfg.VEL_ACCEL, cfg.VEL_TICK_MS = 12000, 40000, 5
        cfg.VEL_GEARSHIFT = True
        spec = importlib.util.spec_from_file_location('profile_under_test', ROOT / 'firmware/current/microprofile.py')
        self.mod = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'config': cfg}):
            spec.loader.exec_module(self.mod)
        self.cfg = cfg
        self.c = types.SimpleNamespace(
            axes={n: Axis() for n in ('pan', 'tilt')}, outputs={},
            velocity=types.SimpleNamespace(timer=None, _arm_pitch_limit=lambda: None))

    def test_roundtrip_preserves_physical_acceleration_and_alignment(self):
        for div in (8, 8, 16, 8, 16):
            p = self.mod.apply(self.c, div)
            self.assertEqual(p['accel_per_tick'], [200 * div // 16] * 2)
            self.assertEqual(self.cfg.MAX_RATE['pan'], 4000 * div // 16)
            self.assertEqual(self.cfg.ACCEL['pan'], 20000 * div // 16)
            self.assertEqual(self.cfg.VEL_ACCEL, 40000)  # canonical, not double-scaled
            self.assertFalse(self.cfg.VEL_GEARSHIFT)
        self.assertEqual(self.c.axes['pan'].moves, [1])

    def test_refuses_live_motion_and_laser_before_mutation(self):
        self.c.velocity.timer = object()
        with self.assertRaises(ValueError):
            self.mod.apply(self.c, 8)
        self.c.velocity.timer = None
        self.c.outputs['laser'] = types.SimpleNamespace(is_on=lambda: True)
        with self.assertRaises(ValueError):
            self.mod.apply(self.c, 8)
        self.assertEqual(self.c.axes['pan'].moves, [])
        self.assertEqual(self.c.axes['pan'].microstep, 16)

    def test_dynamic_then_fixed_profile_restores_settings(self):
        self.mod.apply(self.c, 8)
        report = self.mod.apply(self.c, 'dynamic', 1000)
        self.assertTrue(report['gearshift'])
        self.assertEqual(report['wire_divisor'], 16)
        self.assertEqual(report['protocol'], 2)
        self.assertEqual(report['accel_per_tick'], [200, 200])
        self.assertEqual(self.cfg.MAX_RATE['pan'], 4000)
        self.assertEqual(self.cfg.VEL_DYNAMIC_PULSE_LIMIT, 1000)
        self.assertTrue(self.cfg.VEL_CANONICAL16)
        self.mod.apply(self.c, 16)
        self.assertFalse(self.cfg.VEL_CANONICAL16)
        self.assertFalse(self.cfg.VEL_GEARSHIFT)


if __name__ == '__main__':
    unittest.main()
