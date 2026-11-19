"""Algorithm selection and manifest export without production training."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

import world_models as wm


NOTEBOOK = Path(__file__).resolve().parents[1] / 'world_model' / 'training.ipynb'


def cell_source(fragment):
    cells = json.loads(NOTEBOOK.read_text())['cells']
    return next(''.join(c['source']) for c in cells if fragment in ''.join(c['source']))


def function_source(name):
    tree = ast.parse(cell_source(f'def {name}('))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec')


class RunControlTests(unittest.TestCase):
    def test_unselected_cells_do_not_construct_train_or_save(self):
        backend = Mock()
        for algorithm in wm.REQUIRES:
            if algorithm == 'tcn':
                continue
            with self.subTest(algorithm=algorithm):
                namespace = {'CONFIG': {'algorithms': ['tcn']}, 'RESULTS': [], 'wm': backend}
                exec(function_source('show_model_config'), namespace)
                output = StringIO()
                with redirect_stdout(output):
                    exec(cell_source(f"ALGORITHM = '{algorithm}'"), namespace)
                self.assertIn(f'{algorithm}: not selected; skipping.', output.getvalue())
                self.assertEqual(namespace['RESULTS'][0]['status'], 'not_selected')
        self.assertEqual(backend.mock_calls, [])

    def test_neural_wrapper_rejects_unselected_algorithm(self):
        backend = Mock()
        namespace = {'CONFIG': {'algorithms': ['tcn']}, 'wm': backend}
        exec(function_source('train_neural_configured'), namespace)
        with self.assertRaisesRegex(ValueError, 'not selected'):
            namespace['train_neural_configured']('pets_mlp')
        backend.train_neural.assert_not_called()

    def test_neural_wrapper_runs_selected_algorithm(self):
        backend = Mock()
        namespace = dict(CONFIG={'algorithms': ['tcn']}, wm=backend, window_index=object(),
                         validation=object(), preprocessing=object(), DEVICE='cpu',
                         RESUME_STATE_DICTS={}, RESUME_SOURCES={})
        exec(function_source('train_neural_configured'), namespace)
        self.assertIs(namespace['train_neural_configured']('tcn'), backend.train_neural.return_value)
        self.assertEqual(backend.train_neural.call_args.args[0], 'tcn')
        self.assertEqual(backend.train_neural.call_count, 1)

    def test_final_notebook_cell_serializes_timestamp_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'prefix': 'test', 'run_id': 'manifest', 'algorithms': ['tcn'],
                      'split': {'train': [11], 'validation': [14], 'test_reserved': []},
                      'source_manifest': [{'start': pd.Timestamp('2026-10-23 10:00:05'),
                                           'rows': np.int64(42), 'path': Path('K11.csv')}],
                      'cleanup_feature_cache': False}
            namespace = dict(MODEL_DIR=Path(directory), CONFIG=config, wm=wm, json=json,
                             RESULTS=[{'algorithm': 'tcn', 'status': 'completed', 'metric': np.float64(np.nan)}],
                             METADATA_PATHS=[Path('test_tcn_manifest.json')])
            with redirect_stdout(StringIO()):
                exec(cell_source('manifest_path.write_text('), namespace)
            manifest = json.loads(namespace['manifest_path'].read_text())
            self.assertEqual(manifest['config']['source_manifest'][0]['start'], '2026-10-23 10:00:05')
            self.assertEqual(manifest['config']['source_manifest'][0]['rows'], 42)
            self.assertIsNone(manifest['results'][0]['metric'])
            self.assertIsInstance(config['source_manifest'][0]['start'], pd.Timestamp)


if __name__ == '__main__':
    unittest.main()
