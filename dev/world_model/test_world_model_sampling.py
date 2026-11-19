"""Notebook seed separation and legacy checkpoint-cohort reconstruction."""

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import world_models as wm
import world_model_protocol as wp
from test_world_model_protocol import example_frame, settings


ROOT = Path(__file__).resolve().parents[2]


def notebook_source(name, marker):
    """Resolve a code cell by an identifying snippet so indices need not be pinned."""
    notebook = json.loads((ROOT / 'dev/world_model' / f'{name}.ipynb').read_text())
    if isinstance(marker, int):
        return ''.join(notebook['cells'][marker]['source'])
    matches = [c for c in notebook['cells']
               if c['cell_type'] == 'code' and marker in ''.join(c['source'])]
    if len(matches) != 1:
        raise ValueError(f'{name}.ipynb: {len(matches)} cells match {marker!r}; expected exactly one.')
    return ''.join(matches[0]['source'])


def execute_nodes(source, namespace, names):
    selected = []
    assigned_names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            assigned = {part.id for target in node.targets for part in ast.walk(target)
                        if isinstance(part, ast.Name)}
            if (assigned & names) - assigned_names:
                selected.append(node)
                assigned_names.update(assigned)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<notebook sampling>', 'exec'), namespace)


class NotebookSamplingTests(unittest.TestCase):
    def prepare(self, seed):
        config = {**settings(), 'seed': seed, 'sampling_seed': 42, 'algorithms': ['tcn'],
                  'training_windows_per_cultivation': 16, 'validation_windows_per_cultivation': 12,
                  'treatment_diagnostics': {'windows_per_cultivation': 12, 'post_event_minutes': 1}}
        namespace = {'wm': wm, 'wp': wp, 'np': np, 'CONFIG': config,
                     'TIMING': wm.time_settings(config), 'SAMPLING_INTERVAL_SECONDS': 5,
                     'train_frames': {11: example_frame(400)},
                     'validation_frames': {14: example_frame(400, cultivation=14)}}
        execute_nodes(notebook_source('training', 'def eligible_for_diagnostics(frame)'), namespace,
                      {'eligible_for_diagnostics', 'sampling_config', 'window_index',
                       'raw_validation', 'treatment_anchors'})
        return namespace

    def test_training_checkpoint_and_audit_starts_stay_fixed_across_training_seeds(self):
        runs = [self.prepare(seed) for seed in [42, 43, 44]]
        for namespace in runs:
            self.assertEqual(namespace['sampling_config']['seed'], 42)
            np.testing.assert_array_equal(namespace['window_index'].entries, runs[0]['window_index'].entries)
            self.assertEqual(namespace['raw_validation']['anchors'], runs[0]['raw_validation']['anchors'])
            for cultivation, starts in namespace['treatment_anchors'].items():
                np.testing.assert_array_equal(starts, runs[0]['treatment_anchors'][cultivation])
        self.assertEqual([run['CONFIG']['seed'] for run in runs], [42, 43, 44])

    def test_neural_wrapper_still_passes_original_training_seed(self):
        namespace = self.prepare(43)
        namespace.update(validation={}, preprocessing={}, DEVICE='cpu',
                         RESUME_STATE_DICTS={}, RESUME_SOURCES={})
        execute_nodes(notebook_source('training', 'def train_neural_configured(algorithm)'), namespace, {'train_neural_configured'})
        with patch.object(wm, 'train_neural', return_value={}) as trainer:
            namespace['train_neural_configured']('tcn')
        self.assertIs(trainer.call_args.args[4], namespace['CONFIG'])
        self.assertEqual(trainer.call_args.args[4]['seed'], 43)

    def test_checkpoint_reconstruction_supports_new_and_legacy_metadata(self):
        namespace = self.prepare(43)
        for legacy in [False, True]:
            config = dict(namespace['CONFIG'])
            if legacy:
                config.pop('sampling_seed')
            sampling = {**config, 'seed': config.get('sampling_seed', config['seed'])}
            raw, _ = wm.raw_windows(namespace['validation_frames'], sampling, 12)
            preprocessing = wm.fit_preprocessing(namespace['window_index'], namespace['train_frames'])
            evaluation = {'wm': wm, 'reference': {'config': config, 'preprocessing': preprocessing,
                                                  'validation_anchors': raw['anchors']},
                          'frames': namespace['validation_frames'],
                          'EVAL_CONFIG': {'split': 'validation', 'evaluation_working_budget_gib': 1,
                                          'assessment_seed': 42, 'test_windows_per_cultivation': 12}}
            execute_nodes(notebook_source('evaluation', 'def prepare_evaluation()'), evaluation, {'prepare_evaluation'})
            restored, _, _ = evaluation['prepare_evaluation']()
            self.assertEqual(restored['anchors'], raw['anchors'])

    def test_primary_and_recursive_evaluation_ignore_training_seed(self):
        previous = None
        for seed in [42, 43, 44]:
            namespace = self.prepare(seed)
            preprocessing = wm.fit_preprocessing(namespace['window_index'], namespace['train_frames'])
            evaluation = {'wm': wm, 'reference': {'config': namespace['CONFIG'], 'preprocessing': preprocessing},
                          'frames': namespace['validation_frames'],
                          'EVAL_CONFIG': {'split': 'validation', 'evaluation_working_budget_gib': 1,
                                          'assessment_seed': 42, 'assessment_windows_per_cultivation': 12,
                                          'rollout_starts': 4, 'horizons_minutes': [1, 5], 'progress_bars': False}}
            execute_nodes(notebook_source('evaluation', 'def prepare_evaluation()'), evaluation, {'prepare_assessment'})
            raw, _, _ = evaluation['prepare_assessment']()
            execute_nodes(notebook_source('evaluation', 'checkpoint_raw, checkpoint_prepared'), evaluation, {'rollout_config', 'rollout_anchors'})
            current = (raw['anchors'], evaluation['rollout_anchors'])
            self.assertEqual(evaluation['rollout_config']['seed'], 42)
            if previous is not None:
                self.assertEqual(previous, current)
            previous = current


if __name__ == '__main__':
    unittest.main()
