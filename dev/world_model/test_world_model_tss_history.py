"""Measured TSS inputs with unchanged level targets, causal feedback and new-schema loading."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import world_models as wm
from test_world_model_protocol import example_frame, settings
import test_world_model_rollout_training as rollout_fixtures


@unittest.skipIf(wm.torch is None, 'PyTorch unavailable')
class TSSHistoryTests(unittest.TestCase):
    def fixture(self):
        config, frame, _, _, _ = rollout_fixtures.RolloutTrainingTests().fixture()
        config['features'] = wm.FEATURES.copy()
        config['residual_targets'] = wm.TARGETS.copy()
        config['neural']['tcn'].update(rollout_steps=1, rollout_curriculum=[])
        index = wm.WindowIndex({11: frame}, config, 8)
        prep = wm.fit_preprocessing(index, {11: frame})
        raw, _ = wm.raw_windows({14: example_frame(64, cultivation=14)}, config, 4)
        return config, frame, index, prep, wm.transform(raw, prep)

    def test_required_tss_history_preserves_level_targets_and_effective_loss_weights(self):
        config, frame, index, prep, _ = self.fixture()
        self.assertEqual(len(wm.FEATURES), 21)
        self.assertEqual(wm.make_architecture('tcn', config)['input_channels'], 21)
        self.assertEqual(prep['targets'], wm.CONTROL_TARGETS)
        boundaries = index.boundary_rows()
        baseline = np.zeros_like(boundaries['truth'])
        baseline[:, :9] = boundaries['latest'][:, :9]
        changes = (boundaries['truth'] - baseline).astype(np.float64)
        np.testing.assert_array_equal(prep['delta_mean'], changes.mean(axis=0))
        np.testing.assert_array_equal(prep['delta_scale'], changes.std(axis=0))
        expected_weights = np.array(prep['delta_scale']) / np.array(prep['training_level_std'])
        expected_weights /= np.sqrt(np.mean(expected_weights**2))
        np.testing.assert_allclose(wm.point_loss_weights(prep), expected_weights, rtol=1e-6)
        seq, extra, y = wm.LazyWindows(index, prep)[0]
        raw = index.raw_batch([0])
        prepared = wm.transform(raw, prep)
        np.testing.assert_allclose(seq, prepared['sequence'][0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(y, prepared['y'][0], rtol=1e-6, atol=1e-6)
        j = prep['targets'].index('1_tss(mg/L)')
        start = index.entries[0, 1]
        expected = (frame['1_tss(mg/L)'].iloc[start] - prep['delta_mean'][j])/prep['delta_scale'][j]
        self.assertAlmostEqual(float(y[j]), expected, places=4)
        self.assertEqual(wm.target_baseline(raw['latest'], prep)[0, j], 0.)
        np.testing.assert_array_equal(raw['sequence'][0, :, prep['features'].index('1_tss(mg/L)')],
                                      frame['1_tss(mg/L)'].iloc[start-index.history:start].to_numpy(np.float32))
        table, _ = wm.point_metrics(raw['truth'], raw['truth'], raw['latest'], prep)
        self.assertEqual(table.loc[table.target.eq('1_tss(mg/L)'), 'baseline_method'].item(), 'persistence')
        without_tss = {**config, 'features': [c for c in config['features'] if c != '1_tss(mg/L)']}
        with self.assertRaisesRegex(ValueError, '21-channel'):
            wm.validate_schema(without_tss)
        frame.loc[30, '1_tss(mg/L)'] = np.nan
        missing = wm.WindowIndex({11: frame}, config)
        self.assertTrue(set(missing.entries[:, 1]).isdisjoint(range(30, 43)))


    def test_future_tss_labels_do_not_enter_multistep_feedback(self):
        config, _, index, prep, _ = self.fixture()
        history, extras, truth = [x[None] for x in wm.LazyRolloutWindows(index, [0], 3)[0]]
        calls = []

        class Probe(wm.nn.Module):
            def forward(self, sequence, extra):
                calls.append(sequence.detach().clone())
                return wm.torch.zeros(len(sequence), 12), wm.torch.zeros(len(sequence), 12)

        spec = {**config['neural']['tcn'], 'rollout_steps': 3, 'rollout_checkpointing': False}
        wm.neural_rollout_loss(Probe(), history, extras, truth, prep, spec)
        first = calls.copy()
        calls.clear()
        wm.neural_rollout_loss(Probe(), history, extras, truth+10000, prep, spec)
        for original, changed in zip(first, calls):
            wm.torch.testing.assert_close(original, changed)
        column = prep['features'].index('1_tss(mg/L)')
        native = first[1][0, -1, column]*prep['input_scale'][column]+prep['input_mean'][column]
        self.assertAlmostEqual(float(native), prep['delta_mean'][-1], places=5)
        # A constant level prediction is not added to the previous TSS level.
        native_again = first[2][0, -1, column]*prep['input_scale'][column]+prep['input_mean'][column]
        self.assertAlmostEqual(float(native_again), float(native), places=5)

    def test_recursive_evaluation_feeds_predicted_tss_and_persists_context(self):
        config, _, _, prep, validation = self.fixture()
        config['rollout_starts'] = 2
        frame = example_frame(100, cultivation=14)
        calls, predictions = [], []

        def probe(_, prepared):
            native = prepared['sequence']*np.array(prep['input_scale'])+np.array(prep['input_mean'])
            calls.append(native)
            predicted = np.ones((len(native),12), np.float32)
            predicted[:, :9] = native[:, -1, :9]
            predicted[:, -1] = native[:, -1, prep['features'].index('1_tss(mg/L)')]+1
            predictions.append(predicted.copy())
            return predicted, None

        with patch.object(wm, 'predict', side_effect=probe):
            wm.recursive_metrics({'algorithm': 'tcn', 'preprocessing': prep}, {14: frame}, validation['raw'], config, [.25])
        for step in range(1, len(calls)):
            np.testing.assert_allclose(calls[step][:,-1,prep['features'].index('1_tss(mg/L)')], predictions[step-1][:,-1], atol=1e-5)
            for context in wm.CONTEXT:
                column = prep['features'].index(context)
                np.testing.assert_allclose(calls[step][:,-1,column], calls[0][:,-1,column], atol=1e-5)

    def test_training_save_reload_and_schema_rejection(self):
        config, _, index, prep, validation = self.fixture()
        fitted = wm.train_neural('tcn', index, validation, prep, config, 'cpu')
        predicted, _ = wm.predict(fitted, validation)
        table, _ = wm.point_metrics(validation['raw']['truth'], predicted, validation['raw']['latest'], prep)
        with tempfile.TemporaryDirectory() as directory:
            path = wm.save_model(fitted, directory, config, prep, [], index, validation, table)
            stored = json.loads(path.read_text())
            self.assertEqual(stored['schema_version'], 'world_model_v5')
            self.assertEqual(len(stored['input_order']), 21)
            loaded, _ = wm.load_model(path)
            np.testing.assert_allclose(wm.predict(loaded, validation)[0], predicted)
            bad = copy.deepcopy(stored)
            bad['preprocessing']['residual_targets'].append('1_tss(mg/L)')
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                wm.load_model(path)
            with self.assertRaisesRegex(ValueError, 'input features'):
                invalid_config = copy.deepcopy(config)
                invalid_config['features'].remove('1_tss(mg/L)')
                wm.load_resume_checkpoint(path, 'tcn', invalid_config, prep)


if __name__ == '__main__':
    unittest.main()
