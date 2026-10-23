"""CPU-only synthetic checks; no real dataset or model access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from clean_grouped_v2 import CLASSES, Components, prepare, assign_splits


class ProtocolTests(unittest.TestCase):
    def test_components_transitive(self):
        c = Components(4)
        c.union(2, 3); c.union(1, 2)
        self.assertEqual(c.find(1), c.find(3))
        self.assertNotEqual(c.find(0), c.find(3))

    def test_reproducible_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'data'; root.mkdir()
            rows = []
            def save(path, label, pixels):
                Image.fromarray(pixels).save(root / path)
                rows.append(dict(path=path, label=label, pixel_sha256=hashlib.sha256(
                    str((48, 48)).encode() + b'|' + pixels.tobytes()).hexdigest()))
            rng = np.random.default_rng(88)
            originals = {}
            for label in CLASSES:
                for i in range(12):
                    pixels = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
                    save(f'{label}_{i}.png', label, pixels)
                    originals[label, i] = pixels
            save('exact.png', 'Happy', originals['Happy', 0])
            save('conflict.png', 'Sad', originals['Fear', 0])
            # A same-label near-copy and a cross-label high-confidence edge.
            near = originals['Angry', 0].copy(); near[0, 0, 0] ^= 1
            save('near.png', 'Angry', near)
            near_conflict = originals['Surprise', 0].copy(); near_conflict[0, 0, 0] ^= 1
            save('near_conflict.png', 'Happy', near_conflict)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            manifest = pd.DataFrame(rows)
            a, sa = prepare(manifest, root, base / 'a')
            b, sb = prepare(manifest.sample(frac=1, random_state=2), root, base / 'b')
            self.assertEqual(sa, sb)
            self.assertEqual((base / 'a/split_manifest.csv').read_bytes(),
                             (base / 'b/split_manifest.csv').read_bytes())
            self.assertEqual(sa['proposed_conflict_exclusions'], 4)
            self.assertEqual(sa['exact_redundant'], 1)
            self.assertGreaterEqual(sa['high_confidence_pairs'], 2)
            near_rows = a[a.path.isin(['Angry_0.png', 'near.png'])]
            self.assertEqual(near_rows.group_id.nunique(), 1)
            self.assertEqual(near_rows.split.nunique(), 1)
            self.assertEqual(set(a.label), set(CLASSES))
            self.assertTrue((a.groupby('group_id').split.nunique() == 1).all())
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            integrity = json.loads((base / 'a/split_integrity.json').read_text())
            self.assertTrue(integrity['passed'])
            self.assertEqual(set(integrity['crossing_groups'].values()), {0})
            self.assertEqual(integrity['assigned_images'], len(a))
            self.assertEqual(len(pd.read_csv(base / 'a/conflicting_groups_manifest.csv')), 4)
            self.assertTrue((base / 'a/cross_label_candidates.csv').is_file())
            self.assertFalse(sa['final_test_evaluation_enabled'])
            with self.assertRaises(FileExistsError):
                prepare(manifest, root, base / 'a')

    def test_infeasible_split_fails(self):
        frame = pd.DataFrame([dict(label=c, group_id=c, path=c, reliable_group_id=c,
                                  pixel_sha256=c, target=i) for i,c in enumerate(CLASSES)])
        with self.assertRaises(ValueError):
            assign_splits(frame)




class NotebookIntegrationTests(unittest.TestCase):
    def test_defaults_syntax_and_pinned_module(self):
        import ast
        notebook = json.loads(Path('CNN_resnet18.ipynb').read_text())
        cells = [''.join(c['source']) for c in notebook['cells']]
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                ast.parse(cells[i])
        config = ast.parse(cells[4])
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in config.body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name)
                     and isinstance(node.value, ast.Constant)}
        self.assertFalse(constants['RUN_TRAINING'])
        self.assertIsNone(constants['RESUME_RUN_DIR'])
        self.assertIsNone(constants['REVIEWED_V2_SPLIT_SHA256'])
        self.assertIn(constants['DATASET_PROTOCOL'], ('baseline', 'clean_grouped_v2'))
        self.assertEqual(constants['V2_MODULE_SHA256'], hashlib.sha256(
            Path('clean_grouped_v2.py').read_bytes()).hexdigest())
        self.assertIn('RUN_FINAL_TEST = False', cells[32])
        for i in [16, 18, 20, 22, 24, 26, 28, 30, 32, 34]:
            self.assertIn("globals().get('V2_TRAINING_APPROVED', False)", cells[i])

    def test_loader_branch_hash_and_no_stale_import(self):
        import ast
        import importlib.util
        import io
        import re
        import tempfile
        from unittest.mock import Mock
        notebook = json.loads(Path('CNN_resnet18.ipynb').read_text())
        tree = ast.parse(''.join(notebook['cells'][12]['source']))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                        and node.name == 'load_v2_module')
        payload = Path('clean_grouped_v2.py').read_bytes()
        expected = hashlib.sha256(payload).hexdigest()
        commit = 'a' * 40
        fake_open = Mock(side_effect=[io.BytesIO(json.dumps({'sha': commit}).encode()), io.BytesIO(payload)])
        from urllib.request import Request
        env = dict(Path=Path, re=re, hashlib=hashlib, json=json, tempfile=tempfile,
                   importlib=__import__('importlib'), Request=Request, urlopen=fake_open)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<loader>', 'exec'), env)
        module, provenance = env['load_v2_module'](expected)
        self.assertTrue(callable(module.prepare))
        self.assertEqual(provenance['branch'], 'salma')
        self.assertEqual(provenance['commit'], commit)
        urls = [call.args[0].full_url for call in fake_open.call_args_list]
        self.assertTrue(urls[0].endswith('/commits/salma'))
        self.assertIn('/' + commit + '/clean_grouped_v2.py', urls[1])
        self.assertNotIn('/main/', ''.join(urls))
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'module.py'; file.write_bytes(payload)
            loaded, _ = env['load_v2_module'](expected, file)
            self.assertTrue(callable(loaded.prepare))
            file.write_bytes(b'raise RuntimeError("must never execute")')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                env['load_v2_module'](expected, file)
        fake_open.side_effect = OSError('offline')
        with self.assertRaisesRegex(RuntimeError, 'No fallback to main'):
            env['load_v2_module'](expected)


if __name__ == '__main__':
    unittest.main()
