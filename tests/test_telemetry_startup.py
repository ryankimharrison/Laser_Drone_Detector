import tempfile
import threading
import time
import unittest

from turret_host.telemetry import FlightRecorder


class FakeLink:
    probe_stats = (0, 0)

    def __init__(self):
        self.probes = 0

    def try_probe(self, command):
        self.probes += 1
        return "IMUF 0 0 0 0 0 1"


class TelemetryStartupTests(unittest.TestCase):
    def test_board_probes_are_deferred_until_explicit_enable(self):
        link = FakeLink()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = FlightRecorder(tmp, link=None, imu_hz=100,
                                      save_frames=False).start()
            time.sleep(0.04)
            self.assertEqual(link.probes, 0)
            self.assertIsNone(recorder._thread)
            recorder.enable_imu(link)
            deadline = time.time() + 1.0
            while link.probes == 0 and time.time() < deadline:
                time.sleep(0.005)
            self.assertGreater(link.probes, 0)
            first = recorder._thread
            recorder.enable_imu(link)
            self.assertIs(recorder._thread, first)
            recorder.close()


if __name__ == "__main__":
    unittest.main()
