"""Audit session-grouped YOLO datasets and freeze independent test evidence."""
import hashlib
import json
import math
from pathlib import Path

import yaml

SPLITS = ('train', 'val', 'test')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(data, require_frozen=True):
    """Require directory partitions, reviewed-label files and independent sessions.

    Empty label files are valid negatives. Human review cannot be inferred from
    the files; it remains a separate dataset preparation step.
    """
    data = Path(data).resolve()
    spec = yaml.safe_load(data.read_text(encoding='utf-8'))
    if not isinstance(spec, dict) or spec.get('names') not in ({0: 'drone'}, ['drone']):
        raise ValueError('data.yaml must define the single class names: {0: drone}')
    root = Path(spec.get('path', '.'))
    root = (data.parent / root).resolve() if not root.is_absolute() else root.resolve()
    groups, hashes, records, counts = {}, {}, {}, {}
    for split in SPLITS:
        entry = spec.get(split)
        if not isinstance(entry, str):
            raise ValueError(f'data.yaml must define a {split} image directory')
        images = (root / entry).resolve()
        if images.name != 'images' or not images.is_dir():
            raise ValueError(f'{split} must point to an existing partition/images directory')
        files = sorted(p for p in images.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f'{split} has no images; collect an independent session')
        counts[split] = len(files)
        records[split] = {}
        for path in files:
            parts = path.stem.split('__')
            if len(parts) != 3:
                raise ValueError(f'{path.name}: expected session__clip__camera_frame naming')
            group = parts[0]
            if group in groups and groups[group] != split:
                raise ValueError(f'session {group} leaks between {groups[group]} and {split}')
            groups[group] = split
            image_hash = digest(path)
            if image_hash in hashes and hashes[image_hash] != split:
                raise ValueError(f'duplicate image content crosses splits: {path}')
            hashes[image_hash] = split
            label = images.parent / 'labels' / path.relative_to(images).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'missing label file: {label}')
            for line in label.read_text().splitlines():
                values = line.split()
                if len(values) != 5 or values[0] != '0':
                    raise ValueError(f'invalid single-class YOLO label: {label}')
                xywh = [float(x) for x in values[1:]]
                if (not all(math.isfinite(x) and 0 <= x <= 1 for x in xywh)
                        or min(xywh[2:]) <= 0):
                    raise ValueError(f'invalid normalized box: {label}')
            records[split][path.relative_to(images).as_posix()] = [image_hash, digest(label)]
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    lock_path = root / 'test_lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != records['test']:
            raise ValueError('frozen test images/labels changed; create a new dataset version')
    elif require_frozen:
        raise ValueError('test set not frozen; review its labels then use --freeze-test')
    return dict(root=str(root), counts=counts, sessions=groups, fingerprint=fingerprint,
                test_snapshot=records['test'])


def freeze_test(data):
    result = audit(data, require_frozen=False)
    path = Path(result['root']) / 'test_lock.json'
    if not path.exists():
        with path.open('x', encoding='utf-8') as f:
            json.dump(result['test_snapshot'], f, indent=2, sort_keys=True)
    return result
