"""Notebook imports must refresh stale helpers and resolve this project's files."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / 'dev' / 'world_model'
HELPERS = ('world_model_protocol', 'world_models', 'world_model_plots')


class NotebookImportTests(unittest.TestCase):
    def check_imports(self, cached_from_project):
        for name in ('training', 'evaluation'):
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / 'dev' / 'world_model' / f'{name}.ipynb').read_text())
                source = ''.join(next(c['source'] for c in notebook['cells']
                                      if c['cell_type'] == 'code' and 'def locate_root' in ''.join(c['source'])))
                tree = ast.parse(source)
                start = next(i for i, node in enumerate(tree.body)
                             if isinstance(node, ast.Assign) and any(
                                 isinstance(target, ast.Name) and target.id == 'PROJECT_ROOT'
                                 for target in node.targets))
                # Run the exact import bootstrap without unrelated plotting/package displays.
                stop = next(i for i, node in enumerate(tree.body[start:], start)
                            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'display')
                bootstrap = ast.Module(body=tree.body[start:stop], type_ignores=[])
                stale = {}
                for helper in HELPERS:
                    module = ModuleType(helper)
                    module.__file__ = str((SHARED if cached_from_project else Path('/tmp/other_project')) / f'{helper}.py')
                    stale[helper] = module
                namespace = {'Path': Path, 'sys': sys, 'locate_root': lambda: ROOT,
                             'importlib': __import__('importlib')}
                with patch.dict(sys.modules, stale), patch.object(sys, 'path', list(sys.path)):
                    with redirect_stdout(StringIO()):
                        exec(compile(bootstrap, str(ROOT / name), 'exec'), namespace)
                    wm, wp = namespace['wm'], namespace['wp']
                    self.assertEqual(Path(wm.__file__).resolve(), SHARED / 'world_models.py')
                    self.assertEqual(Path(wp.__file__).resolve(), SHARED / 'world_model_protocol.py')
                    self.assertIs(wm.wp, wp)
                    self.assertEqual(len(wm.CONTROL_TARGETS), 12)
                    self.assertEqual(wp.PRIORITIES, ['1_ph', '1_cod(mg/L)', '1_bod(mg/L)',
                                                     '1_do(mg/L)', '1_nh3(mg/L)'])
                    self.assertTrue(callable(wp.training_starts))
                    self.assertTrue(callable(wm.train_neural))
                    if name == 'evaluation':
                        plots = namespace['refresh_local_helper']('world_model_plots')
                        self.assertTrue(callable(plots.priority_comparison))
                    self.assertNotIn('wm.wp.', ''.join(''.join(cell['source']) for cell in notebook['cells']))

    def test_cached_older_local_helpers_are_refreshed(self):
        self.check_imports(cached_from_project=True)

    def test_cached_helpers_from_another_project_are_replaced(self):
        self.check_imports(cached_from_project=False)


if __name__ == '__main__':
    unittest.main()
