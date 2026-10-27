from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from turret_host import config, calibrate, control
from turret_host.camera_profile import OV9281, C270, WIDE
from turret_host.calibration_context import CalibrationMismatchError, stamp, validate
from turret_host.camera_fusion import WideToNarrow
from turret_host.registration import WideNarrowRegistration


class CalibrationContext(unittest.TestCase):
    def setUp(self):
        self.profile = replace(OV9281, name='test-module', vid_pid=(123, 456),
                               lens_id='test-lens', mount_id='test-mount')
        p = patch.object(config, 'CAMERA_PROFILES', {'narrow': self.profile, 'wide': WIDE})
        p.start()
        self.addCleanup(p.stop)
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_bound_jacobian_roundtrip_and_mismatched_mode(self):
        path = self.root / 'j.json'
        j = control.Jacobian([[1., .8], [-.9, .7]])
        j.save(path)
        np.testing.assert_allclose(control.Jacobian.load(path).J, j.J)
        for changed in (replace(self.profile, size=(640, 400)),
                        replace(self.profile, lens_id='other'),
                        replace(self.profile, rotation_deg=0),
                        replace(self.profile, serial='other')):
            with patch.dict(config.CAMERA_PROFILES, narrow=changed):
                with self.assertRaises(CalibrationMismatchError):
                    control.Jacobian.load(path)

    def test_legacy_allowed_only_on_exact_original_profile(self):
        with self.assertRaises(CalibrationMismatchError):
            validate({})
        with patch.dict(config.CAMERA_PROFILES, narrow=C270):
            validate({})
        with patch.dict(config.CAMERA_PROFILES, narrow=replace(C270, mount_id='moved')):
            with self.assertRaises(CalibrationMismatchError):
                validate({})

    def test_uncommissioned_profile_cannot_stamp_measurements(self):
        with patch.dict(config.CAMERA_PROFILES, narrow=OV9281):
            with self.assertRaisesRegex(CalibrationMismatchError, 'Commission'):
                stamp({})

    def test_generic_writer_and_goal_reader_share_context(self):
        with patch.object(calibrate, 'CALIB_DIR', self.root):
            path = calibrate.save_calibration('goal_pixel',
                                              dict(u_inf=500., c_u=0., v_inf=300., c_v=10.))
            self.assertTrue(control.GoalModel.load(path).is_calibrated)
            self.assertIn('camera_context', calibrate.load_goal_pixel())
            with patch.dict(config.CAMERA_PROFILES, narrow=replace(self.profile, mount_id='moved')):
                with self.assertRaises(CalibrationMismatchError):
                    calibrate.load_goal_pixel()
        validate(stamp({}, ('wide',)), ('wide',))

    def test_fusion_and_registration_do_not_hide_incompatible_fits(self):
        for loader in (WideToNarrow.load, WideNarrowRegistration.load):
            p = self.root / 'old.json'
            p.write_text(json.dumps({'scale': 2.}))
            with self.assertRaises(CalibrationMismatchError):
                loader(str(p))
        p = str(self.root / 'fit.json')
        WideToNarrow.derived().save(p)
        self.assertIn('measured', WideToNarrow.load(p).provenance)

    def test_exposure_and_fps_do_not_invalidate_geometry(self):
        blob = stamp({})
        with patch.dict(config.CAMERA_PROFILES,
                        narrow=replace(self.profile, exposure=-7, fps=60)):
            validate(blob)


if __name__ == '__main__':
    unittest.main()
