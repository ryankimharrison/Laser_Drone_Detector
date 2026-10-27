import unittest
from unittest.mock import patch
import numpy as np

from turret_host import config
from turret_host.tracker import PixelTracker
from turret_host.types import Detection, TrackState


def box(source='narrow', width=80., height=40.):
    return Detection(600., 300., 600. + width, 300. + height, .9, 'drone', source)


class FlightTracker(unittest.TestCase):
    def test_global_shutter_does_not_shift_a_moving_detection(self):
        d = box('wide->narrow')
        shifted = PixelTracker._rolling_shutter_corrected(d, np.array([1000., -500.]))
        self.assertEqual(shifted.source, d.source)
        self.assertEqual((shifted.cx, shifted.cy), (d.cx, d.cy))
        with patch.object(config, 'ROLLING_SHUTTER_K_S', .01):
            shifted = PixelTracker._rolling_shutter_corrected(box(), np.array([1000., -500.]))
        self.assertAlmostEqual(shifted.cx, box().cx - 10.)
        self.assertAlmostEqual(shifted.cy, box().cy + 5.)

    def test_shape_filter_can_be_disabled_without_accepting_invalid_boxes(self):
        with patch.object(config, 'ASSOC_BOX_ASPECT_MAX', None):
            self.assertTrue(PixelTracker._shape_ok(box(width=120., height=20.)))
            self.assertFalse(PixelTracker._shape_ok(box(width=-1.)))
            self.assertFalse(PixelTracker._shape_ok(box(width=float('nan'))))
        with patch.object(config, 'ASSOC_BOX_ASPECT_MAX', 1.6):
            self.assertFalse(PixelTracker._shape_ok(box(width=120., height=20.)))

    def test_confirmation_and_loss_have_equal_durations_at_different_rates(self):
        for hz in (30, 60, 120):
            tr = PixelTracker()
            tr.update([box(width=40., height=80.)], 0.)
            acquired = None
            for i in range(1, hz):
                state = tr.update([box(width=40., height=80.)], i / hz).state
                if state is TrackState.TRACK:
                    acquired = i / hz
                    break
            self.assertIsNotNone(acquired)
            self.assertAlmostEqual(acquired, config.ACQUIRE_CONFIRM_S)
            for i in range(1, hz):
                state = tr.update([], acquired + i / hz).state
                if state is TrackState.COAST:
                    self.assertAlmostEqual(i / hz, config.TRACK_LOSS_S)
                    break
            else:
                self.fail('tracker never coasted')

    def test_miss_resets_confirmation_and_reacquisition_requires_confirmation(self):
        tr = PixelTracker()
        tr.update([box()], 0.)
        tr.update([], .02)
        self.assertEqual(tr.update([box()], .04).state, TrackState.ACQUIRE)
        self.assertEqual(tr.update([box()], .15).state, TrackState.ACQUIRE)
        self.assertEqual(tr.update([box()], .18).state, TrackState.TRACK)
        self.assertEqual(tr.update([], .52).state, TrackState.COAST)
        self.assertEqual(tr.update([box()], .53).state, TrackState.ACQUIRE)

    def test_range_at_ten_meters_is_not_clamped_to_six(self):
        tr = PixelTracker(rotation_deg=0)
        width = config.NARROW_F_PX * config.DRONE_WIDTH_M / 10.
        with patch.object(config, 'ASSOC_BOX_ASPECT_MAX', None):
            est = tr.update([box(width=width, height=20.)], 0.)
        self.assertEqual(est.range_source, 'size')
        self.assertAlmostEqual(est.range_m, 10.)


if __name__ == '__main__':
    unittest.main()
