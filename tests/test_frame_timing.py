from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from turret_host import cameras, config
from turret_host.app import wide_evidence_fresh
from turret_host.camera_profile import OV9281
from turret_host.detector import Detector
from turret_host.telemetry import FlightRecorder
from turret_host.tracker import PixelTracker
from turret_host.types import Detection, TrackState, LaserState


class FrameTiming(unittest.TestCase):
    def frame(self):
        profile = replace(OV9281, pipeline_delay_s=.04)
        with patch.dict(config.CAMERA_PROFILES, narrow=profile):
            return cameras.received_frame(np.zeros((4, 6, 3), np.uint8), 10., 7, 'narrow')

    def test_capture_offset_applied_once_and_inference_preserves_provenance(self):
        frame = self.frame()
        self.assertAlmostEqual(frame.t, 9.96)
        detector = Detector()
        detector._model = Mock()
        detector._model.predict.return_value = [SimpleNamespace(boxes=None, speed={})]
        with patch.object(detector, '_predict_kwargs', return_value={}):
            result = detector.detect(frame)
        self.assertEqual(result.frame_t, frame.t)
        self.assertEqual(result.received_t, 10.)
        self.assertEqual(result.time_source, 'calibrated_offset')
        self.assertIsNotNone(result.infer_done_t)
        self.assertEqual(result.pipeline_delay_s, .04)

    def test_prediction_advances_to_now_plus_only_command_lead(self):
        tr = PixelTracker()
        tr._begin_track(Detection(500., 300., 540., 380., .9, 'drone'), 9.96)
        tr._t = 9.96
        tr._x[2:] = [100., 0.]
        with patch.object(config, 'CONTROL_LEAD_S', .01):
            est = tr.estimate_for_control(10.02)
        self.assertAlmostEqual(est.predicted_to_t, 10.03)
        self.assertAlmostEqual(est.u, 520. + 100. * .07)

    def test_two_stale_frames_or_future_evidence_cannot_authorize_wide_fallback(self):
        self.assertFalse(wide_evidence_fresh(10., 9., 9.01))
        self.assertFalse(wide_evidence_fresh(10., 9.98, 10.01))
        self.assertTrue(wide_evidence_fresh(10., 9.98, 9.95))

    def test_telemetry_records_actual_target_time_and_structured_camera_profiles(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            rec = FlightRecorder(tmp, imu_hz=0, save_frames=False)
            tr = PixelTracker()
            est = tr.update([Detection(500., 300., 540., 380., .9, 'drone')], 9.96)
            est = tr.predict_to(10.03)
            rec.log_control(t=10.02, state=TrackState.ACQUIRE, est=est, out=None,
                            laser=LaserState.DISARMED, interlock=None, link=None,
                            frame_t=9.96, received_t=10., infer_done_t=10.01,
                            time_source='calibrated_offset', pipeline_delay_s=.04)
            self.assertIsNone(rec.failed)
            rec.close()
            row = json.loads((Path(tmp) / 'control.jsonl').read_text())
            self.assertAlmostEqual(row['observation_age_ms'], 60.)
            self.assertEqual(row['est']['predicted_to_t'], 10.03)
            session = json.loads((Path(tmp) / 'session.json').read_text())
            self.assertEqual(session['schema_version'], 2)
            self.assertIsInstance(session['config']['CAMERA_PROFILES']['narrow'], dict)

    def test_legacy_laser_measurement_refuses_mono_before_hardware_access(self):
        from turret_host import measure_latency
        with patch.object(cameras, 'identify_cameras') as identify, \
                patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                measure_latency.main([])
        identify.assert_not_called()

    def test_analysis_uses_recorded_prediction_time_instead_of_legacy_latency(self):
        from tools.analyze_run import score
        rows = []
        for i in range(5):
            t = 10. + .02 * i
            x = 100. + 100. * i
            rows.append({'t': t, 'track': 'TRACK', 'est': {
                'box': [x - 5., 95., x + 5., 105.], 'box_t': t, 'conf': .9,
                'u': x + 50., 'v': 100., 'predicted_to_t': t + .01}})
        result = score(rows, latency=.9)
        self.assertGreater(result['scored'], 0)
        np.testing.assert_allclose(result['lead'], 0., atol=1e-9)


if __name__ == '__main__':
    unittest.main()
