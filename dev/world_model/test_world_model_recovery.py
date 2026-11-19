"""Recovery equivalence and selected-checkpoint diagnostic causality."""

from pathlib import Path
import ast
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import world_models as wm
import world_model_recovery as recovery
import world_model_training_diagnostics as diagnostics
import test_world_model_rollout_training as rollout_fixtures
from test_world_model_protocol import example_frame


class InterruptedTraining(Exception):
    pass


@unittest.skipIf(wm.torch is None, 'PyTorch unavailable')
class RecoveryTests(unittest.TestCase):
    def fixture(self, directory, members=1):
        config, frame, index, prep, validation = rollout_fixtures.RolloutTrainingTests().fixture()
        config['neural']['tcn'].update(epochs=4, patience=5, members=members, head_dropout=.2,
                                     optimizer='adamw', weight_decay=.0001, lr_scheduler=True,
                                     scheduler_factor=.5, scheduler_patience=0, min_learning_rate=1e-6, min_delta=.05)
        config['priority_targets'] = wm.wp.PRIORITIES.copy()
        config['recovery'] = {'enabled': True, 'every_epochs': 1,
                              'directory': directory, 'resume_by_algorithm': {}}
        return config, frame, index, prep, validation

    def test_recovery_matches_uninterrupted_dropout_bootstrap_and_curriculum(self):
        for members, interrupted_member, rollout in [(1, 0, 1), (1, 0, 3), (2, 1, 3)]:
            with self.subTest(members=members), tempfile.TemporaryDirectory() as directory:
                config, _, index, prep, validation = self.fixture(directory, members)
                if rollout == 1:
                    config['neural']['tcn'].update(rollout_steps=1, rollout_curriculum=[])
                config['run_id'] = 'uninterrupted'
                reference = wm.train_neural('tcn', index, validation, prep, config, 'cpu')
                config['run_id'] = 'interrupted'
                original_save = recovery.atomic_save

                def interrupt_after_epoch(torch, payload, path):
                    original_save(torch, payload, path)
                    active = payload['active']
                    if active and active['member'] == interrupted_member and active['epoch'] == 2:
                        raise InterruptedTraining()

                with patch.object(recovery, 'atomic_save', side_effect=interrupt_after_epoch):
                    with self.assertRaises(InterruptedTraining):
                        wm.train_neural('tcn', index, validation, prep, config, 'cpu')
                source = Path(directory) / 'test_tcn_interrupted.recovery.pt'
                config['run_id'] = 'recovered'
                config['recovery']['resume_by_algorithm'] = {'tcn': str(source)}
                resumed = wm.train_neural('tcn', index, validation, prep, config, 'cpu')
                self.assertTrue(resumed['optimizer_state_restored'])
                self.assertLess(resumed['training_curve'][-1]['learning_rate'], .001)
                self.assertEqual(resumed['training_curve'], reference['training_curve'])
                self.assertEqual(resumed['selected_checkpoints'], reference['selected_checkpoints'])
                for first, second in zip(reference['models'], resumed['models']):
                    for key, value in first.state_dict().items():
                        wm.torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
                # A fully trained recovery file can still export the selected members without retraining.
                config['recovery']['resume_by_algorithm']['tcn'] = str(Path(directory) / 'test_tcn_recovered.recovery.pt')
                config['run_id'] = 'export_again'
                finished = wm.train_neural('tcn', index, validation, prep, config, 'cpu')
                self.assertEqual(finished['training_curve'], reference['training_curve'])

    def test_recovery_rejects_changed_recipe_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, index, prep, validation = self.fixture(directory)
            wm.train_neural('tcn', index, validation, prep, config, 'cpu')
            config['recovery']['resume_by_algorithm']['tcn'] = str(Path(directory) / 'test_tcn_rollout.recovery.pt')
            config['run_id'] = 'changed'
            config['neural']['tcn']['learning_rate'] /= 2
            with self.assertRaisesRegex(ValueError, 'Recovery mismatch'):
                wm.train_neural('tcn', index, validation, prep, config, 'cpu')

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recovery.pt'
            recovery.atomic_save(wm.torch, {'value': 1}, path)
            with patch.object(wm.torch, 'save', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    recovery.atomic_save(wm.torch, {'value': 2}, path)
            self.assertEqual(wm.torch.load(path, weights_only=True)['value'], 1)
            self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_validity_is_per_channel_with_ph_and_temperature_rules(self):
        targets = ['1_cod(mg/L)', '1_ph', '1_temp_air(C)']
        rows = diagnostics.validity_rows(np.array([[-2., 15., -5.], [np.nan, 7., 1.]]), targets, 'event', 3, 5)
        self.assertEqual(rows[0]['negative'], 1)
        self.assertEqual(rows[0]['maximum_negative_magnitude'], 2.)
        self.assertEqual(rows[0]['nonfinite'], 1)
        self.assertEqual(rows[1]['ph_outside_0_14'], 1)
        self.assertEqual(rows[2]['negative'], 0)
        self.assertEqual(rows[0]['horizon_seconds'], 15)

    def test_event_forecast_uses_predicted_feedback_and_persists_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, index, prep, _ = self.fixture(directory)
            config['response_diagnostics'] = {'enabled': True, 'horizons_minutes': [.25, .5],
                                              'events_per_action': 1, 'no_event_starts': 1}
            frame = example_frame(100, cultivation=14)
            fitted = {'algorithm': 'tcn', 'preprocessing': prep}
            calls, predictions = [], []

            def probe(_, prepared):
                history = prepared['sequence'] * np.array(prep['input_scale']) + np.array(prep['input_mean'])
                calls.append(history)
                predicted = np.ones((len(history), len(wm.CONTROL_TARGETS)), np.float32)
                predicted[:, :9] = history[:, -1, :9] + 1
                predictions.append(predicted.copy())
                return predicted, None

            with patch.object(wm, 'predict', side_effect=probe):
                response, coverage, validity = diagnostics.event_response_diagnostics(wm, fitted, {14: frame}, config)
            self.assertFalse(response.empty)
            self.assertFalse(validity.empty)
            for step in range(1, len(calls)):
                np.testing.assert_allclose(calls[step][:, -1, :9], predictions[step-1][:, :9], atol=1e-5)
                np.testing.assert_allclose(calls[step][:, -1, 9:15], calls[0][:, -1, 9:15], atol=1e-5)
            self.assertTrue((coverage.selected_origins <= 1).all())

    def test_notebook_selected_checkpoint_report_and_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, index, prep, validation = self.fixture(directory)
            config['response_diagnostics'] = {'enabled': True, 'horizons_minutes': [.25],
                                              'events_per_action': 1, 'no_event_starts': 1}
            fitted = wm.train_neural('tcn', index, validation, prep, config, 'cpu')
            notebook = json.loads((Path(__file__).parents[1] / 'world_model/training.ipynb').read_text())
            source = ''.join(next(c['source'] for c in notebook['cells']
                                  if c['cell_type'] == 'code' and 'def validate_fitted' in ''.join(c['source'])))
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'validate_fitted')
            import pandas as pd
            namespace = {'wm': wm, 'wd': diagnostics, 'wp': wm.wp, 'np': np, 'pd': pd,
                         'CONFIG': config, 'validation': validation, 'raw_validation': validation['raw'],
                         'preprocessing': prep, 'validation_frames': {14: example_frame(100, cultivation=14)},
                         'display': lambda *args: None, 'Markdown': str}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook report>', 'exec'), namespace)
            table, _ = namespace['validate_fitted'](fitted)
            self.assertTrue(fitted['validation_prediction_validity'])
            self.assertTrue(fitted['validation_event_response'])
            metadata = wm.save_model(fitted, directory, config, prep, [], index, validation, table)
            stored = json.loads(metadata.read_text())
            self.assertTrue(stored['validation_event_response'])
            restored, _ = wm.load_model(metadata, 'cpu')
            np.testing.assert_allclose(wm.predict(restored, validation)[0], wm.predict(fitted, validation)[0])


if __name__ == '__main__':
    unittest.main()
