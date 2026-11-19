"""Training rollout causality, gradients, cohort boundaries, and artifact compatibility."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import world_models as wm
from test_world_model_protocol import example_frame, settings


@unittest.skipIf(wm.torch is None, 'PyTorch unavailable')
class RolloutTrainingTests(unittest.TestCase):
    def setUp(self):
        wm.torch.set_num_threads(1)

    def fixture(self, interval=5):
        config = settings(interval)
        neural = {'epochs': 2, 'patience': 2, 'batch_size': 4, 'learning_rate': .001,
                  'hidden_size': 4, 'dilations': [1, 2], 'head_dropout': 0.,
                  'loss': 'normalized_mse_plus_nll', 'nll_weight': .01,
                  'rollout_steps': 3, 'rollout_checkpointing': True,
                  'rollout_curriculum': [{'start_epoch': 1, 'steps': 1}, {'start_epoch': 2, 'steps': 3}],
                  'include_initial_checkpoint': True}
        config.update(neural={name: dict(neural) for name in ['tcn', 'gru', 'lstm', 'pets_mlp']},
                      cpu_threads=1, loader_workers=0, pin_memory=False, validation_batch_size=4,
                      training_track='capped_all_models', prefix='test', run_id='rollout',
                      split={'train': [11], 'validation': [14], 'test_reserved': [16]})
        frame = example_frame(64, interval)
        index = wm.WindowIndex({11: frame}, config, 8)
        prep = wm.fit_preprocessing(index, {11: frame})
        raw, _ = wm.raw_windows({14: example_frame(64, interval, 14)}, config, 4)
        return config, frame, index, prep, wm.transform(raw, prep)

    def test_complete_segments_exclude_gaps_missing_rows_labels_and_episode_end(self):
        config, _, _, _, _ = self.fixture()
        frame = example_frame(100)
        frame.loc[30:, 'timestamp'] += pd.Timedelta(minutes=1)
        frame.loc[70, '1_bod(mg/L)'] = np.nan
        frame.loc[50, 'status'] = 'BAD'
        frame['status'] = frame['status'].fillna('OK')
        index = wm.WindowIndex({11: frame}, config)
        positions = wm.rollout_training_positions(index, 3)
        starts = set(index.entries[positions, 1])
        self.assertTrue({27, 47, 65, 67, 73, 97}.issubset(starts))
        self.assertTrue(starts.isdisjoint({28, 29, 48, 49, 68, 69, 70, 98, 99}))
        self.assertTrue(all(index.entries[positions, 0] == 11))

    def test_future_labels_never_enter_inputs_and_actions_align(self):
        config, _, index, prep, _ = self.fixture()
        positions = wm.rollout_training_positions(index, 3)
        dataset = wm.LazyRolloutWindows(index, positions[:1], 3)
        history, extras, truth = [value[None] for value in dataset[0]]
        start = index.entries[positions[0], 1]
        np.testing.assert_array_equal(extras[0, :, :5], index.stores[11]['values'][start:start+3, -5:])

        class Probe(wm.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = wm.nn.Parameter(wm.torch.zeros(12))
                self.calls = []

            def forward(self, sequence, extra):
                self.calls.append((sequence.detach().clone(), extra.detach().clone()))
                mean = self.offset.expand(len(sequence), -1)
                return mean, wm.torch.zeros_like(mean)

        model = Probe()
        spec = {**config['neural']['tcn'], 'rollout_checkpointing': False}
        original = history.clone()
        wm.neural_rollout_loss(model, history, extras, truth, prep, spec)
        calls = model.calls.copy()
        model.calls.clear()
        wm.neural_rollout_loss(model, history, extras, truth + 10000, prep, spec)
        for (seq, ext), (again_seq, again_ext) in zip(calls, model.calls):
            wm.torch.testing.assert_close(seq, again_seq)
            wm.torch.testing.assert_close(ext, again_ext)
        wm.torch.testing.assert_close(history, original)
        native_next = (calls[1][0][:, -1] * wm.torch.tensor(prep['input_scale']) +
                       wm.torch.tensor(prep['input_mean']))
        np.testing.assert_allclose(native_next[:, -5:], extras[:, 0, :5], atol=1e-5)
        np.testing.assert_allclose(native_next[:, 9:15], original[:, -1, 9:15], atol=1e-5)

    def test_later_errors_backpropagate_through_predicted_history(self):
        class ConstantDelta(wm.nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = wm.nn.Parameter(wm.torch.tensor(.2))

            def forward(self, sequence, extra):
                mean = self.theta.expand(len(sequence), 1)
                return mean, wm.torch.zeros_like(mean)

        prep = {'targets': wm.TARGETS[:1], 'target_transform': 'delta',
                'input_mean': [0.]*len(wm.FEATURES), 'input_scale': [1.]*len(wm.FEATURES),
                'delta_mean': [0.], 'delta_scale': [1.], 'training_level_std': [1.]}
        for checkpointing in [False, True]:
            model = ConstantDelta()
            history = wm.torch.ones(1, 4, len(wm.FEATURES))
            loss, _, _ = wm.neural_rollout_loss(
                model, history, wm.torch.zeros(1, 3, 8), wm.torch.zeros(1, 3, 1), prep,
                {'rollout_steps': 3, 'rollout_checkpointing': checkpointing,
                 'loss': 'normalized_mse_plus_nll', 'nll_weight': 0.})
            loss.backward()
            self.assertAlmostEqual(model.theta.grad.item(), 2 * (1.2 + 2*1.4 + 3*1.6) / 3, places=5)

    def test_one_step_loss_matches_existing_objective_for_all_transforms_and_filter(self):
        for mode in ['delta', 'mixed', 'level']:
            config, _, index, prep, _ = self.fixture()
            if mode == 'delta':
                config['targets'] = wm.TARGETS + ['1_tss(mg/L)']
                config['target_provenance'] = {t: 'test' for t in config['targets']}
            config['target_transform'] = mode
            config['input_filter'] = {'mode': 'trailing_median', 'window_seconds': 20, 'features': wm.TARGETS[:2]}
            frame = example_frame(64)
            index = wm.WindowIndex({11: frame}, config, 8)
            prep = wm.fit_preprocessing(index, {11: frame})
            model = wm.DynamicsNetwork(wm.make_architecture('tcn', config)).eval()
            sequence, extra, target = wm.LazyWindows(index, prep)[0]
            mean, logvar = model(sequence[None], extra[None])
            nll = .5 * (np.log(2*np.pi) + logvar + (target[None] - mean).square() * (-logvar).exp())
            point = ((mean - target[None]) * wm.torch.tensor(wm.point_loss_weights(prep))).square().mean()
            expected = point + .01 * nll.mean()
            history, extras, truth = [value[None] for value in wm.LazyRolloutWindows(index, [0], 1)[0]]
            actual, _, _ = wm.neural_rollout_loss(model, history, extras, truth, prep, config['neural']['tcn'])
            wm.torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_curriculum_validation(self):
        with self.assertRaises(ValueError): wm.rollout_settings({'rollout_steps': 0})
        with self.assertRaises(ValueError): wm.rollout_settings({'rollout_steps': 1.5})
        with self.assertRaises(ValueError): wm.rollout_settings({'rollout_discount': 0})
        with self.assertRaises(ValueError):
            wm.rollout_settings({'epochs': 2, 'rollout_steps': 3,
                                 'rollout_curriculum': [{'start_epoch': 1, 'steps': 1}]})

    def test_activation_checkpointing_preserves_dropout_gradients(self):
        config, _, index, prep, _ = self.fixture()
        config['neural']['tcn']['head_dropout'] = .2
        model = wm.DynamicsNetwork(wm.make_architecture('tcn', config)).train()
        history, extras, truth = [value[None] for value in wm.LazyRolloutWindows(index, [0], 3)[0]]
        results = []
        for enabled in [False, True]:
            wm.torch.manual_seed(19)
            model.zero_grad(set_to_none=True)
            loss, _, _ = wm.neural_rollout_loss(
                model, history, extras, truth, prep,
                {**config['neural']['tcn'], 'rollout_checkpointing': enabled})
            loss.backward()
            results.append((loss.detach(), [param.grad.clone() for param in model.parameters()]))
        wm.torch.testing.assert_close(results[0][0], results[1][0])
        for first, second in zip(results[0][1], results[1][1]):
            wm.torch.testing.assert_close(first, second)

    def test_cpu_training_curriculum_and_saved_model_reload(self):
        config, _, index, prep, validation = self.fixture()
        fitted = wm.train_neural('tcn', index, validation, prep, config, 'cpu')
        self.assertEqual([row['rollout_steps'] for row in fitted['training_curve']], [0, 1, 3])
        self.assertEqual(fitted['training_rollout']['steps'], 3)
        self.assertIsNone(fitted['peak_cuda_allocated_gib'])
        predicted, _ = wm.predict(fitted, validation)
        metrics, _ = wm.point_metrics(validation['raw']['truth'], predicted,
                                       validation['raw']['latest'], prep)
        with tempfile.TemporaryDirectory() as directory:
            path = wm.save_model(fitted, directory, config, prep, [], index, validation, metrics)
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata['horizon_seconds'], 5)
            self.assertEqual(metadata['training_rollout']['horizon_seconds'], 15)
            restored, _ = wm.load_model(path)
            np.testing.assert_allclose(wm.predict(restored, validation)[0], predicted)
        legacy_path = Path(__file__).parents[1] / 'model/world_model/wm_5s_12targets_proxy_tcn_5STESTFULLTRAINBATCH64.json'
        if legacy_path.exists():
            with self.assertRaisesRegex(ValueError, 'world_model_v5'):
                wm.load_model(legacy_path)

    def test_other_architectures_and_minute_cadence_train_on_rollouts(self):
        for algorithm, interval in [('gru', 5), ('lstm', 5), ('pets_mlp', 60)]:
            with self.subTest(algorithm=algorithm, interval=interval):
                config, _, index, prep, validation = self.fixture(interval)
                if algorithm == 'pets_mlp':
                    config['neural'][algorithm]['members'] = 2
                fitted = wm.train_neural(algorithm, index, validation, prep, config, 'cpu')
                self.assertTrue(np.isfinite(wm.predict(fitted, validation)[0]).all())
                self.assertEqual(fitted['training_rollout']['horizon_seconds'], 3 * interval)

    def test_twelve_and_sixty_step_feedback_gradients(self):
        class ConstantDelta(wm.nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = wm.nn.Parameter(wm.torch.tensor(.01))

            def forward(self, sequence, extra):
                mean = self.theta.expand(len(sequence), 1)
                return mean, wm.torch.zeros_like(mean)

        prep = {'targets': wm.TARGETS[:1], 'target_transform': 'delta',
                'input_mean': [0.]*len(wm.FEATURES), 'input_scale': [1.]*len(wm.FEATURES),
                'delta_mean': [0.], 'delta_scale': [1.], 'training_level_std': [1.]}
        for steps in [12, 60]:
            model = ConstantDelta()
            loss, _, _ = wm.neural_rollout_loss(
                model, wm.torch.ones(1, 4, len(wm.FEATURES)), wm.torch.zeros(1, steps, 8), wm.torch.zeros(1, steps, 1),
                prep, {'rollout_steps': steps, 'loss': 'normalized_mse_plus_nll', 'nll_weight': 0.})
            loss.backward()
            times = np.arange(1, steps + 1)
            self.assertAlmostEqual(model.theta.grad.item(), float(2 * np.mean(times * (1 + times*.01))), places=3)


if __name__ == '__main__':
    unittest.main()
