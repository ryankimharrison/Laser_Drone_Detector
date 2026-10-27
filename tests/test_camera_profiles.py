from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
from turret_host import cameras, config
from turret_host.camera_profile import load_profiles, OV9281


class CameraProfiles(unittest.TestCase):
    def test_local_profile_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'camera.json'
            p.write_text(json.dumps({'narrow': {'vid_pid': [123, 456],
                                              'fourcc': 'YUY2', 'rotation_deg': 0}}))
            n = load_profiles(p, 'ov9281')['narrow']
            self.assertEqual(n.vid_pid, (123, 456))
            self.assertEqual(n.fourcc, 'YUY2')
            self.assertEqual(n.rolling_shutter_s, 0)
            with self.assertRaises(ValueError):
                replace(n, pipeline_delay_s=-1)
            with self.assertRaises(ValueError):
                replace(n, rolling_shutter_s=.01)

    def test_gray_bgra_and_invalid_input(self):
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        for value in (gray, gray[..., None]):
            actual = cameras.bgr_image(value)
            self.assertEqual(actual.shape, (3, 4, 3))
            for i in range(3):
                np.testing.assert_array_equal(actual[..., i], gray)
        self.assertEqual(cameras.bgr_image(np.zeros((3, 4, 4), np.uint8)).shape, (3, 4, 3))
        with self.assertRaises(cameras.CameraError):
            cameras.bgr_image(gray.astype(np.uint16))

    def test_uncommissioned_identity_is_not_guessed(self):
        device = Mock()
        device.describe.return_value = 'USB camera'
        with patch.object(cameras, '_pnp_video_devices', return_value=[device]), \
                patch.object(config, 'NARROW_VID_PID', None):
            with self.assertRaisesRegex(cameras.CameraIdentificationError, 'not commissioned'):
                cameras.identify_cameras()

    def test_cache_signature_changes_with_geometry(self):
        with patch.object(config, 'CAMERA_PROFILES', {'narrow': OV9281}):
            first = cameras._usb_signature([])
        with patch.object(config, 'CAMERA_PROFILES', {'narrow': replace(OV9281, size=(640, 400))}):
            self.assertNotEqual(first, cameras._usb_signature([]))

    def test_fps_counts_distinct_images_and_checks_decode(self):
        cap = Mock()
        a = np.zeros((2, 2), np.uint8)
        cap.retrieve.side_effect = [(True, a), (True, a.copy()), (True, a + 1)]
        with patch.object(cameras.time, 'perf_counter', side_effect=[0., 1.]):
            self.assertEqual(cameras.measure_delivered_fps(cap, frames=3, warmup=0), 2.)
        cap.retrieve.side_effect = [(False, None)]
        with self.assertRaises(cameras.CameraError):
            cameras.measure_delivered_fps(cap, frames=1, warmup=0)

    def test_null_exposure_does_not_open_camera(self):
        with patch.object(cameras.cv2, 'VideoCapture') as cap:
            result = cameras.lock_exposure(0, None)
        cap.assert_not_called()
        self.assertFalse(result.ok)


if __name__ == '__main__':
    unittest.main()
