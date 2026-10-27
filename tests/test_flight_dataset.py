import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

from turret_host import ingest_session, train_detector
from turret_host.dataset_audit import audit, freeze_test


class FlightDataset(unittest.TestCase):
    def setUp(self):
        quiet = patch('sys.stdout', new_callable=io.StringIO)
        quiet.start()
        self.addCleanup(quiet.stop)
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'source'
        self.out = self.root / 'yolo'
        self.source.mkdir()
        for name, value in [('DATASET_DIR', str(self.source)), ('YOLO_DIR', str(self.out))]:
            p = patch.object(ingest_session, name, value)
            p.start()
            self.addCleanup(p.stop)

    def capture(self, session, content=None):
        clip = self.source / session / 'clips' / 'pass'
        clip.mkdir(parents=True)
        for i, cam in enumerate(('narrow', 'wide')):
            (clip / (cam + '_001.jpg')).write_bytes(content or (session + cam).encode())

    def populate(self):
        for split in ('train', 'val', 'test'):
            self.capture(split + '_flight')
            ingest_session.ingest(split + '_flight', split=split)
        return self.out / 'data.yaml'

    def test_both_cameras_stay_in_session_partition_and_labels_survive(self):
        self.capture('flight')
        self.assertEqual(ingest_session.ingest('flight', split='val'), 0)
        self.assertEqual(len(list((self.out / 'val/images').glob('*.jpg'))), 2)
        label = next((self.out / 'val/labels').glob('*.txt'))
        label.write_text('0 .5 .5 .2 .1\n')
        original = label.read_bytes()
        ingest_session.ingest('flight', split='val')
        self.assertEqual(label.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            ingest_session.ingest('flight', split='train')

    def test_explicit_assignment_and_dry_run_do_not_write(self):
        self.capture('flight')
        with self.assertRaisesRegex(ValueError, '--split'):
            ingest_session.ingest('flight')
        ingest_session.ingest('flight', dry_run=True, split='test')
        self.assertFalse(self.out.exists())

    def test_freeze_detects_label_edits_and_blocks_new_test_frames(self):
        data = self.populate()
        with self.assertRaisesRegex(ValueError, 'not frozen'):
            audit(data)
        first = freeze_test(data)
        self.assertEqual(audit(data)['fingerprint'], first['fingerprint'])
        with self.assertRaisesRegex(ValueError, 'frozen'):
            ingest_session.ingest('test_flight', split='test')
        label = next((self.out / 'test/labels').glob('*.txt'))
        label.write_text('0 .5 .5 .2 .1')
        with self.assertRaisesRegex(ValueError, 'changed'):
            audit(data)
        with self.assertRaisesRegex(ValueError, 'changed'):
            freeze_test(data)

    def test_session_leakage_and_content_duplicates_rejected(self):
        data = self.populate()
        original = next((self.out / 'train/images').glob('*.jpg'))
        extra = self.out / 'val/images' / original.name
        extra.write_bytes(b'different pixels')
        (self.out / 'val/labels' / extra.with_suffix('.txt').name).touch()
        with self.assertRaisesRegex(ValueError, 'leaks'):
            audit(data, require_frozen=False)
        extra.unlink()
        target = next((self.out / 'val/images').glob('*.jpg'))
        target.write_bytes(original.read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit(data, require_frozen=False)

    def test_training_freeze_mode_does_not_load_models(self):
        data = self.populate()
        self.assertEqual(train_detector.main(['--data', str(data), '--freeze-test']), 0)
        args = train_detector.build_parser().parse_args(['--weights', 'candidate.pt', '--seed', '42'])
        self.assertEqual(args.weights, 'candidate.pt')
        self.assertEqual(args.seed, 42)

    def test_temporally_split_legacy_session_is_not_silently_extended(self):
        self.capture('flight')
        for part in ('train', 'val'):
            dest = self.out / part / 'images'
            dest.mkdir(parents=True)
            (dest / 'flight__pass__narrow_001.jpg').write_bytes(b'old')
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            ingest_session.ingest('flight', split='train')

    def test_training_and_test_evaluation_use_explicit_inputs_without_overwrite(self):
        data = self.populate()
        freeze_test(data)
        weights = self.root / 'current.pt'
        weights.write_bytes(b'local checkpoint')
        model = Mock()
        yolo = Mock(return_value=model)
        fake_torch = SimpleNamespace(__version__='test')
        fake_ultra = SimpleNamespace(__version__='test', YOLO=yolo)
        args = ['--data', str(data), '--weights', str(weights), '--device', 'cpu',
                '--seed', '42', '--imgsz', '1280', '--degrees', '25']
        with patch.dict('sys.modules', torch=fake_torch, ultralytics=fake_ultra), \
                patch.object(train_detector, 'RUNS', str(self.root / 'runs')):
            self.assertEqual(train_detector.main(args + ['--name', 'candidate']), 0)
            kw = model.train.call_args.kwargs
            self.assertEqual(kw['seed'], 42)
            self.assertEqual(kw['degrees'], 25.)
            self.assertEqual(kw['workers'], 2)
            yolo.assert_called_with(str(weights.resolve()))
            manifest = json.loads((self.root / 'runs/candidate/experiment.json').read_text())
            self.assertEqual(manifest['dataset']['fingerprint'], audit(data)['fingerprint'])
            with self.assertRaises(SystemExit), patch('sys.stderr', new_callable=io.StringIO):
                train_detector.main(args + ['--name', 'candidate'])
            self.assertEqual(train_detector.main(args + ['--evaluate', '--split', 'test',
                                                         '--name', 'baseline_test']), 0)
            self.assertEqual(model.val.call_args.kwargs['split'], 'test')


if __name__ == '__main__':
    unittest.main()
