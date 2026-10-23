"""Synthetic saved-audit review only; no model, training or real-data rebuild."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from clean_grouped_v2 import prepare
from review_dataset_v2 import review_saved_audit, validate, extract_conflict_sources


class SavedReviewTests(unittest.TestCase):
    def fixture(self, base):
        root = base / 'data'; root.mkdir()
        rows = []; rng = np.random.default_rng(42)
        first = None
        for label in ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']:
            for i in range(5):
                pixels = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
                if first is None: first = pixels
                path = f'{label}_{i}.png'
                Image.fromarray(pixels).save(root / path)
                rows.append(dict(path=path, label=label, pixel_sha256=hashlib.sha256(
                    str((48,48)).encode()+b'|'+pixels.tobytes()).hexdigest()))
        Image.fromarray(first).save(root / 'conflict.png')
        rows.append(dict(path='conflict.png',label='Fear',pixel_sha256=rows[0]['pixel_sha256']))
        audit = base / 'audit'
        frame, summary = prepare(pd.DataFrame(rows), root, audit)
        return root, audit, summary['split_sha256']

    def test_review_preserves_inputs_and_creates_large_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root, audit, locked = self.fixture(base)
            before = {str(p): p.read_bytes() for p in base.rglob('*') if p.is_file()}
            out = base / 'review'
            checks = review_saved_audit(audit, root, out, locked)
            self.assertEqual(checks['conflicting_groups'], 1)
            self.assertEqual(checks['conflict_images_verified'], 2)
            self.assertTrue(checks['audit_files_unchanged'])
            self.assertFalse(checks['visual_review_complete'])
            for path, contents in before.items(): self.assertEqual(Path(path).read_bytes(), contents)
            self.assertEqual(len(pd.read_csv(out / 'conflict_members_review.csv')), 2)
            sheet = next((out / 'conflict_contact_sheets').glob('*.png'))
            with Image.open(sheet) as image: self.assertEqual(image.width, 1050)
            self.assertEqual(pd.read_csv(out / 'conflict_group_review.csv').iloc[0].reason,
                             'exact_pixel_label_conflict')
            with self.assertRaises(FileExistsError): review_saved_audit(audit, root, out, locked)

    def test_rejects_modified_split_and_missing_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root, audit, locked = self.fixture(base)
            path = audit / 'split_manifest.csv'; original = path.read_bytes()
            path.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256'): validate(audit, locked)
            path.write_bytes(original)
            manifest = pd.read_csv(audit / 'decision_manifest.csv')
            manifest[manifest.path != 'conflict.png'].to_csv(audit / 'decision_manifest.csv', index=False)
            with self.assertRaises(ValueError): validate(audit, locked)

    def test_selective_zip_extraction_and_changed_image_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root, audit, locked = self.fixture(base)
            zip_path = base / 'dataset.zip'
            with zipfile.ZipFile(zip_path, 'w') as z:
                for path in root.iterdir(): z.write(path, 'Data/' + path.name)
            summary_path = audit / 'summary.json'; summary=json.loads(summary_path.read_text())
            summary['archive_sha256'] = hashlib.sha256(zip_path.read_bytes()).hexdigest()
            summary_path.write_text(json.dumps(summary))
            extracted = extract_conflict_sources(audit, zip_path, locked)
            self.assertEqual(len(list(extracted.iterdir())), 2)
            Image.new('RGB', (48,48), 'red').save(extracted / 'conflict.png')
            checks = review_saved_audit(audit, extracted, base / 'review', locked)
            self.assertEqual(checks['conflict_images_unavailable_or_changed'], 1)
            self.assertEqual(checks['conflict_images_verified'], 1)


if __name__ == '__main__': unittest.main()
