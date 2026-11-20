"""Derived outputs, headline selection metric, and loss-weight bounds."""

import numpy as np
import unittest

import world_models as wm


TARGETS = list(wm.CONTROL_TARGETS)
MEASURED = ['1_tds(ppm)', '1_do(mg/L)', '1_ph', '1_cod(mg/L)', '1_toc(mg/L)',
            '1_turbidity(NTU)', '1_nh3(mg/L)', '1_temp_air(C)', '2_light(Lux)']
TSS = '1_tss(mg/L)'
TURBIDITY = '1_turbidity(NTU)'


def settings(derived=True, headline=True, bounds=None):
    config = {
        'targets': TARGETS, 'features': list(wm.FEATURES), 'actions': list(wm.ACTIONS),
        'target_transform': 'mixed', 'residual_targets': list(wm.TARGETS),
        'label_mode': 'proxy_prototype',
        'target_provenance': {name: 'test' for name in TARGETS},
    }
    if derived:
        config['derived_outputs'] = {TSS: {'source': TURBIDITY, 'scale': 1.5, 'offset': 0.0}}
    if headline:
        config['headline_targets'] = list(MEASURED)
    if bounds is not None:
        config['point_loss_weight_bounds'] = bounds
    return config


def preprocessing(config):
    count = len(config['targets'])
    return {**config,
            'delta_mean': [0.0] * count, 'delta_scale': [1.0] * count,
            'training_level_std': [1.0] * count,
            'derived_outputs': config.get('derived_outputs') or {},
            'headline_targets': list(config.get('headline_targets') or []),
            'point_loss_weight_bounds': config.get('point_loss_weight_bounds')}


class DerivedOutputTests(unittest.TestCase):
    def test_schema_accepts_declared_derivation(self):
        self.assertEqual(wm.validate_schema(settings()), TARGETS)

    def test_derived_target_may_not_be_headline(self):
        config = settings()
        config['headline_targets'] = list(MEASURED) + [TSS]
        with self.assertRaises(ValueError):
            wm.validate_schema(config)

    def test_source_must_be_a_predicted_input_channel(self):
        for broken in [{'source': 'not_a_target', 'scale': 1.},
                       {'source': 'biomassa_estimasi(g/L)', 'scale': 1.},   # target but not in FEATURES
                       {'source': TSS, 'scale': 1.}]:                       # self reference
            config = settings()
            config['derived_outputs'] = {TSS: broken}
            with self.subTest(source=broken['source']):
                with self.assertRaises(ValueError):
                    wm.derived_output_specs(config)

    def test_restore_levels_enforces_the_relation(self):
        config = settings()
        prep = preprocessing(config)
        rng = np.random.default_rng(0)
        latest = np.abs(rng.normal(10, 1, (6, len(TARGETS))))
        delta = rng.normal(0, .1, (6, len(TARGETS)))
        levels = wm.restore_levels(delta, latest, prep)
        tss, turbidity = TARGETS.index(TSS), TARGETS.index(TURBIDITY)
        expected = 1.5 * (latest[:, turbidity] + delta[:, turbidity])
        np.testing.assert_allclose(levels[:, tss], expected, rtol=0, atol=1e-9)

    def test_restore_levels_is_unchanged_without_derivation(self):
        config = settings(derived=False)
        prep = preprocessing(config)
        rng = np.random.default_rng(1)
        latest = rng.normal(0, 1, (4, len(TARGETS)))
        delta = rng.normal(0, 1, (4, len(TARGETS)))
        levels = wm.restore_levels(delta, latest, prep)
        # A mixed transform persists residual targets and uses a zero baseline for
        # output-only targets; with no derivation declared this is the plain formula.
        residual = set(wm.residual_target_order(prep))
        baseline = np.where([[name in residual for name in TARGETS]], latest, 0.0)
        np.testing.assert_allclose(levels, baseline + delta, rtol=0, atol=1e-9)

    def test_loss_mask_zeroes_only_derived_columns(self):
        mask = wm.derived_loss_mask(preprocessing(settings()))
        self.assertEqual(mask[TARGETS.index(TSS)], 0.0)
        self.assertEqual(float(mask.sum()), float(len(TARGETS) - 1))

    def test_all_targets_derived_is_rejected(self):
        config = settings()
        config['derived_outputs'] = {name: {'source': TURBIDITY, 'scale': 1.0}
                                     for name in TARGETS}
        with self.assertRaises(ValueError):
            wm.derived_loss_mask(preprocessing(config))


class HeadlineMetricTests(unittest.TestCase):
    def test_headline_score_excludes_derived_and_proxy_targets(self):
        prep = preprocessing(settings())
        rng = np.random.default_rng(2)
        truth = np.abs(rng.normal(10, 1, (8, len(TARGETS))))
        predicted = truth + rng.normal(0, .1, truth.shape)
        latest = truth.copy()
        table, score = wm.point_metrics(truth, predicted, latest, prep)
        self.assertEqual(len(table.attrs['headline_targets']), len(MEASURED))
        self.assertNotIn(TSS, table.attrs['headline_targets'])
        self.assertEqual(score, table.attrs['macro_NRMSE_headline'])
        self.assertGreaterEqual(table.attrs['macro_NRMSE_all'], 0.0)
        self.assertTrue(table.set_index('target').loc[TSS, 'is_derived'])

    def test_headline_score_is_the_reported_selection_value(self):
        prep = preprocessing(settings())
        rng = np.random.default_rng(3)
        truth = np.abs(rng.normal(5, 1, (5, len(TARGETS))))
        table, score = wm.point_metrics(truth, truth, truth, prep)
        # Perfect prediction gives zero error on every scored channel.
        self.assertAlmostEqual(score, 0.0, places=12)

    def test_without_declaration_headline_falls_back_to_measured(self):
        prep = preprocessing(settings(headline=False))
        names = wm.headline_target_names(prep)
        self.assertNotIn(TSS, names)
        self.assertIn(TURBIDITY, names)


class PointLossWeightTests(unittest.TestCase):
    def test_bounds_clip_extreme_weights(self):
        prep = preprocessing(settings())
        prep['training_level_std'] = [1.0] + [1e6] * (len(TARGETS) - 1)
        unbounded = wm.point_loss_weights({**prep, 'point_loss_weight_bounds': None})
        bounded = wm.point_loss_weights({**prep, 'point_loss_weight_bounds': [0.5, 2.0]})
        self.assertGreater(float(unbounded.max()), 2.0)
        self.assertLessEqual(float(bounded.max()), 2.0 + 1e-6)
        self.assertGreaterEqual(float(bounded.min()), 0.5 - 1e-6)

    def test_bounds_are_validated(self):
        prep = preprocessing(settings(bounds=[2.0, 0.5]))
        with self.assertRaises(ValueError):
            wm.point_loss_weights(prep)


class SilentlyIgnoredSettingTests(unittest.TestCase):
    def test_latest_sensor_to_head_is_rejected_outside_tcn(self):
        config = {'targets': TARGETS, 'features': list(wm.FEATURES), 'actions': list(wm.ACTIONS),
                  'history_minutes': 60, 'sampling_interval_seconds': 60,
                  'neural': {'pg_ac_tcn': {'hidden_size': 8, 'latest_sensor_to_head': True}}}
        with self.assertRaises(ValueError):
            wm.make_architecture('pg_ac_tcn', config)

    def test_dedicated_heads_are_rejected_outside_the_tcn_family(self):
        config = {'targets': TARGETS, 'features': list(wm.FEATURES), 'actions': list(wm.ACTIONS),
                  'history_minutes': 60, 'sampling_interval_seconds': 60,
                  'neural': {'gru': {'hidden_size': 8, 'head_layout': 'dedicated'}}}
        with self.assertRaises(ValueError):
            wm.make_architecture('gru', config)


class FoldSplitTests(unittest.TestCase):
    """The LOCO fold helper must never produce overlapping splits."""

    @staticmethod
    def fold_split():
        import ast, json
        from pathlib import Path
        notebook = json.loads((Path(__file__).parent / 'training.ipynb').read_text())
        source = ''.join(next(c['source'] for c in notebook['cells']
                              if c['cell_type'] == 'code' and 'def fold_split' in ''.join(c['source'])))
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'fold_split')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<fold_split>', 'exec'), namespace)
        return namespace['fold_split']

    def test_each_fold_holds_out_exactly_one_cultivation(self):
        fold_split = self.fold_split()
        for holdout in [11, 12, 13, 14, 16]:
            split = fold_split([11, 12, 13, 14, 16], holdout)
            with self.subTest(holdout=holdout):
                self.assertEqual(split['validation'], [holdout])
                self.assertEqual(len(split['train']), 4)
                self.assertNotIn(holdout, split['train'])
                self.assertEqual(split['test_reserved'], [])

    def test_default_does_not_overlap_training_and_validation(self):
        fold_split = self.fold_split()
        split = fold_split([11, 12, 13, 14, 16], None)
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertEqual(len(split['validation']), 1)

    def test_unknown_or_degenerate_input_is_rejected(self):
        fold_split = self.fold_split()
        with self.assertRaises(ValueError):
            fold_split([11, 12], 99)
        with self.assertRaises(ValueError):
            fold_split([11], None)
        with self.assertRaises(ValueError):
            fold_split([11, 11, 12], 11)


class SelectionPolicyTests(unittest.TestCase):
    """An all-data model may not early-stop on a cohort it trained on."""

    @staticmethod
    def settings(**over):
        base = {'epochs': 20, 'selection': 'validation', 'fixed_epochs': None,
                'include_initial_checkpoint': False}
        base.update(over)
        return base

    def test_default_is_validation_and_needs_a_cohort(self):
        with self.assertRaises(ValueError):
            wm.selection_policy(self.settings(), None)
        with self.assertRaises(ValueError):
            wm.selection_policy(self.settings(), {})

    def test_validation_mode_is_accepted_with_a_cohort(self):
        policy = wm.selection_policy(self.settings(), {'y': np.zeros((3, 12))})
        self.assertEqual(policy, {'mode': 'validation', 'fixed_epochs': None})

    def test_fixed_epochs_needs_no_cohort(self):
        policy = wm.selection_policy(self.settings(selection='fixed_epochs', fixed_epochs=12), {})
        self.assertEqual(policy['mode'], 'fixed_epochs')
        self.assertEqual(policy['fixed_epochs'], 12)

    def test_fixed_epochs_budget_is_validated(self):
        for bad in [None, 0, -1, 21, True, 3.5]:
            with self.subTest(budget=bad):
                with self.assertRaises(ValueError):
                    wm.selection_policy(
                        self.settings(selection='fixed_epochs', fixed_epochs=bad), {})

    def test_fixed_epochs_rejects_initial_checkpoint(self):
        with self.assertRaises(ValueError):
            wm.selection_policy(self.settings(selection='fixed_epochs', fixed_epochs=5,
                                              include_initial_checkpoint=True), {})

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            wm.selection_policy(self.settings(selection='oracle'), {})

    def test_validation_available_detects_empty_cohorts(self):
        self.assertFalse(wm.validation_available(None))
        self.assertFalse(wm.validation_available({}))
        self.assertFalse(wm.validation_available({'y': np.zeros((0, 12))}))
        self.assertTrue(wm.validation_available({'y': np.zeros((1, 12))}))


class NeuralForwardShapeTests(unittest.TestCase):
    """Every architecture must produce a (batch, outputs) prediction.

    dlinear previously raised 'mat1 and mat2 shapes cannot be multiplied' because its
    projections were applied after an extra transpose, so this pins the output shape
    for every registered architecture.
    """

    TARGETS = list(wm.CONTROL_TARGETS)

    def setUp(self):
        if wm.torch is None:
            self.skipTest('PyTorch is unavailable.')

    def architecture(self, algorithm, steps=60):
        return {'algorithm': algorithm, 'history_minutes': 60, 'history_steps': steps,
                'sampling_interval_seconds': 60, 'hidden_size': 8,
                'dilations': [1, 2, 4, 8], 'head_dropout': 0.0,
                'head_layout': 'shared',
                'head_groups': {'shared': list(range(len(self.TARGETS)))},
                'attention_heads': 2, 'latent_size': 4,
                'sensor_channels': len(wm.FEATURES) - len(wm.ACTIONS),
                'action_channels': len(wm.ACTIONS),
                'moving_average_kernel': 5, 'pool_scales': [1, 4, 16],
                'latest_sensor_to_head': False, 'latest_sensor_names': [],
                'latest_sensor_indices': [],
                'input_channels': len(wm.FEATURES), 'outputs': len(self.TARGETS),
                'target_order': self.TARGETS, 'log_variance_bounds': [-8, 2]}

    def test_every_architecture_returns_batch_by_outputs(self):
        torch = wm.torch
        batch, steps = 4, 60
        sequence = torch.zeros(batch, steps, len(wm.FEATURES))
        extra = torch.zeros(batch, len(wm.ACTIONS) + 3)
        for algorithm in sorted(wm.REQUIRES):
            if algorithm in ['ridge', 'random_forest', 'hrp_xgb', 'lightgbm',
                             'xgboost', 'gpr', 'grey_box']:
                continue
            with self.subTest(algorithm=algorithm):
                model = wm.DynamicsNetwork(self.architecture(algorithm, steps))
                model.eval()
                with torch.no_grad():
                    mean, log_variance = model(sequence, extra)
                self.assertEqual(tuple(mean.shape), (batch, len(self.TARGETS)))
                self.assertEqual(tuple(log_variance.shape), (batch, len(self.TARGETS)))

    def test_dlinear_accepts_a_short_history(self):
        """The projection length must follow history_steps, not the channel count."""
        torch = wm.torch
        for steps in [12, 60, 300]:
            with self.subTest(steps=steps):
                model = wm.DynamicsNetwork(self.architecture('dlinear', steps))
                model.eval()
                with torch.no_grad():
                    mean, _ = model(torch.zeros(3, steps, len(wm.FEATURES)),
                                    torch.zeros(3, len(wm.ACTIONS) + 3))
                self.assertEqual(tuple(mean.shape), (3, len(self.TARGETS)))


class OfflineReLEmsembleTests(unittest.TestCase):
    """MOPO, ORPO and MOUP uncertainty aggregation."""

    def setUp(self):
        self.means = np.array([[[1.0, 2.0]], [[3.0, 6.0]], [[5.0, 10.0]]])  # (3,1,2)
        self.vars = np.array([[[1.0, 4.0]], [[1.0, 4.0]], [[1.0, 4.0]]])

    def test_max_std_takes_the_most_pessimistic_member(self):
        u = wm.model_uncertainty(self.means, self.vars, 'max_std')
        np.testing.assert_allclose(u, [[1.0, 2.0]])

    def test_ensemble_std_is_dispersion_of_means(self):
        u = wm.model_uncertainty(self.means, self.vars, 'ensemble_std')
        np.testing.assert_allclose(u, self.means.std(axis=0))

    def test_mc_dropout_adds_epistemic_and_aleatoric(self):
        u = wm.model_uncertainty(self.means, self.vars, 'mc_dropout')
        epistemic = np.sqrt(self.means.var(axis=0))
        aleatoric = np.sqrt(self.vars.mean(axis=0))
        np.testing.assert_allclose(u, np.sqrt(epistemic ** 2 + aleatoric ** 2))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            wm.model_uncertainty(self.means, self.vars, 'vibes')

    def test_aggregation_that_needs_disagreement_rejects_one_member(self):
        single = self.means[:1]
        for mode in ['max_std', 'ensemble_std']:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    wm.model_uncertainty(single, self.vars[:1], mode)

    def test_registry_declares_one_mode_per_algorithm(self):
        self.assertEqual(set(wm.ENSEMBLE_ALGORITHMS), {'mopo_ensemble', 'moup_mlp'})
        modes = {a: s['uncertainty_mode'] for a, s in wm.ENSEMBLE_ALGORITHMS.items()}
        self.assertEqual(modes['mopo_ensemble'], 'max_std')
        self.assertEqual(modes['moup_mlp'], 'mc_dropout')

    def test_any_multi_member_model_exposes_dispersion(self):
        """A TCN with several members has real disagreement, so it exposes u(s,a)."""
        self.assertIsNone(wm.resolved_uncertainty_mode('tcn', 1))
        self.assertEqual(wm.resolved_uncertainty_mode('tcn', 5), 'ensemble_std')
        self.assertEqual(wm.resolved_uncertainty_mode('gru', 7), 'ensemble_std')

    def test_declared_mode_wins_over_the_member_count(self):
        self.assertEqual(wm.resolved_uncertainty_mode('mopo_ensemble', 5), 'max_std')
        self.assertEqual(wm.resolved_uncertainty_mode('moup_mlp', 5), 'mc_dropout')

    def test_single_member_is_rejected_by_the_architecture(self):
        config = {'targets': list(wm.CONTROL_TARGETS), 'features': list(wm.FEATURES),
                  'actions': list(wm.ACTIONS), 'history_minutes': 60,
                  'sampling_interval_seconds': 60,
                  'neural': {'mopo_ensemble': {'hidden_size': 8, 'members': 1}}}
        with self.assertRaises(ValueError):
            wm.make_architecture('mopo_ensemble', config)

    def test_mc_dropout_is_required_only_by_moup(self):
        base = {'targets': list(wm.CONTROL_TARGETS), 'features': list(wm.FEATURES),
                'actions': list(wm.ACTIONS), 'history_minutes': 60,
                'sampling_interval_seconds': 60}
        with self.assertRaises(ValueError):   # MOUP without dropout has no epistemic term
            wm.make_architecture('moup_mlp', {**base,
                'neural': {'moup_mlp': {'hidden_size': 8, 'members': 2, 'mc_dropout': 0.}}})
        with self.assertRaises(ValueError):   # MOPO must not silently carry dropout
            wm.make_architecture('mopo_ensemble', {**base,
                'neural': {'mopo_ensemble': {'hidden_size': 8, 'members': 2, 'mc_dropout': .1}}})


class RolloutSelectionTests(unittest.TestCase):
    """Checkpoint selection must follow the configured training rollout length."""

    def test_metric_label_tracks_rollout_steps(self):
        def label(steps):
            return ('one-step validation macro NRMSE' if steps == 1 else
                    f'{steps}-step recursive validation macro NRMSE')
        self.assertEqual(label(1), 'one-step validation macro NRMSE')
        self.assertEqual(label(15), '15-step recursive validation macro NRMSE')

    def test_cohort_rejects_nonpositive_steps(self):
        if wm.torch is None:
            self.skipTest('PyTorch unavailable')
        with self.assertRaises(ValueError):
            wm.rollout_validation_cohort({}, {}, {'history_minutes': 60,
                                                  'sampling_interval_seconds': 60}, 0)

    def test_score_rollout_rejects_steps_beyond_the_cohort(self):
        if wm.torch is None:
            self.skipTest('PyTorch unavailable')
        cohort = {'steps_available': 5}
        with self.assertRaises(ValueError):
            wm.score_rollout(None, cohort, {}, 6)

    def test_train_neural_accepts_the_new_keyword(self):
        import inspect
        signature = inspect.signature(wm.train_neural)
        self.assertIn('validation_frames', signature.parameters)
        self.assertIsNone(signature.parameters['validation_frames'].default)


class ActionResponseGateTests(unittest.TestCase):
    """A model may only be handed to an RL policy when it shows a graded action response."""

    def frame(self, rows):
        import pandas as pd
        return pd.DataFrame(rows)

    def test_coin_flip_direction_fails_the_gate(self):
        import world_model_training_diagnostics as wd
        response = self.frame([
            {'group': 'Pump1_NaHCO3 (mL)', 'direction_correct': True, 'horizon_minutes': 15.0}] * 5
            + [{'group': 'Pump1_NaHCO3 (mL)', 'direction_correct': False, 'horizon_minutes': 15.0}] * 5
            + [{'group': 'no_event', 'direction_correct': True, 'horizon_minutes': 15.0}] * 4)
        coverage = self.frame([{'group': 'Pump1_NaHCO3 (mL)', 'eligible_origins': 100,
                                'selected_origins': 10, 'status': 'available'},
                               {'group': 'no_event', 'eligible_origins': 999,
                                'selected_origins': 4, 'status': 'available'}])
        gate = wd.action_response_gate(response, coverage,
                                       {'action_response_gate': {'min_direction_agreement': .6,
                                                                 'min_events_per_action': 5}})
        self.assertFalse(gate['rl_eligible'])
        self.assertEqual(gate['gate_status'], 'fail')
        self.assertEqual(gate['per_action'][0]['status'], 'weak_response')

    def test_strong_consistent_response_passes(self):
        import world_model_training_diagnostics as wd
        response = self.frame([
            {'group': 'Pump3_Microalgae (mL)', 'direction_correct': True, 'horizon_minutes': 15.0}] * 9
            + [{'group': 'no_event', 'direction_correct': True, 'horizon_minutes': 15.0}] * 4)
        coverage = self.frame([{'group': 'Pump3_Microalgae (mL)', 'eligible_origins': 40,
                                'selected_origins': 9, 'status': 'available'},
                               {'group': 'no_event', 'eligible_origins': 999,
                                'selected_origins': 4, 'status': 'available'}])
        gate = wd.action_response_gate(response, coverage,
                                       {'action_response_gate': {'min_direction_agreement': .6,
                                                                 'min_events_per_action': 5}})
        self.assertTrue(gate['rl_eligible'])
        self.assertEqual(gate['usable_actions'], ['Pump3_Microalgae (mL)'])

    def test_unavailable_action_is_reported_not_silently_passed(self):
        import world_model_training_diagnostics as wd
        response = self.frame([
            {'group': 'Aerator_RU (L/s)', 'direction_correct': None, 'horizon_minutes': 15.0}])
        coverage = self.frame([{'group': 'Aerator_RU (L/s)', 'eligible_origins': 0,
                                'selected_origins': 0,
                                'status': 'unavailable; no complete eligible segment'}])
        gate = wd.action_response_gate(response, coverage, {'action_response_gate': {}})
        self.assertFalse(gate['rl_eligible'])
        self.assertIn('Aerator_RU', gate['reason'])

    def test_no_diagnostics_blocks_eligibility(self):
        import world_model_training_diagnostics as wd
        gate = wd.action_response_gate(None, None, {'action_response_gate': {}})
        self.assertFalse(gate['rl_eligible'])
        self.assertEqual(gate['gate_status'], 'no_response_diagnostics')

    def test_coverage_limitations_record_zero_events(self):
        import world_model_training_diagnostics as wd
        coverage = self.frame([
            {'group': 'Aerator_RU (L/s)', 'eligible_origins': 0, 'selected_origins': 0,
             'status': 'unavailable'},
            {'group': 'Pump1_NaHCO3 (mL)', 'eligible_origins': 256, 'selected_origins': 2,
             'status': 'available'}])
        rows = {r['action']: r for r in wd.action_coverage_limitations(coverage)}
        self.assertFalse(rows['Aerator_RU (L/s)']['gradeable'])
        self.assertTrue(rows['Pump1_NaHCO3 (mL)']['gradeable'])


class ActionNoveltyTests(unittest.TestCase):
    """Action novelty answers "has this control input been tried?"."""

    def test_identical_action_has_zero_novelty(self):
        """A logged action must score exactly zero novelty, not a scaled artefact."""
        reference = np.array([[0., 0., 0., 2., 90.]] * 10, dtype=np.float32)
        novelty = wm.action_novelty(np.array([[0., 0., 0., 2., 90.]]), reference, k=1)
        self.assertAlmostEqual(float(novelty[0]), 0.0, places=6)

    def test_explicit_and_derived_scaling_agree(self):
        """Passing the training mean/scale must equal letting the reference supply them."""
        rng = np.random.default_rng(3)
        raw = rng.normal(0, 4, (50, 5)).astype(np.float32)
        mean, scale = raw.mean(0), raw.std(0)
        query = raw[7:8]
        explicit = wm.action_novelty(query, raw, k=1, mean=mean, scale=scale)
        derived = wm.action_novelty(query, raw, k=1)
        self.assertAlmostEqual(float(explicit[0]), float(derived[0]), places=5)

    def test_distance_is_in_standardised_units(self):
        """A far action must land at a plausible multi-sigma distance, not a raw-unit value."""
        rng = np.random.default_rng(4)
        reference = rng.normal(0, 1, (400, 5)).astype(np.float32)
        far = wm.action_novelty(np.full((1, 5), 10.0), reference, k=1)
        self.assertLess(float(far[0]), 40.0)
        self.assertGreater(float(far[0]), 5.0)

    def test_unseen_action_is_more_novel(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, (200, 5)).astype(np.float32)
        near = wm.action_novelty(np.zeros((1, 5)), reference, k=1)
        far = wm.action_novelty(np.full((1, 5), 6.0), reference, k=1)
        self.assertGreater(float(far[0]), float(near[0]))

    def test_k_is_clamped_to_the_reference_size(self):
        reference = np.zeros((3, 5), dtype=np.float32)
        novelty = wm.action_novelty(np.zeros((2, 5)), reference, k=99)
        self.assertEqual(novelty.shape, (2,))
        self.assertTrue(np.isfinite(novelty).all())

    def test_empty_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            wm.action_novelty(np.zeros((1, 5)), np.zeros((0, 5)), k=1)

    def test_constant_reference_channel_does_not_divide_by_zero(self):
        reference = np.tile(np.array([1., 2., 3., 4., 5.], np.float32), (20, 1))
        novelty = wm.action_novelty(np.array([[1., 2., 3., 4., 5.]]), reference, k=1)
        self.assertTrue(np.isfinite(novelty).all())

    def test_reference_is_built_from_training_windows_only(self):
        """fit_action_reference must use the indexed training entries, not every row."""
        class Store(dict):
            pass

        class Index:
            stores = {11: {'values': np.arange(60, dtype=np.float32).reshape(10, 6)}}
            entries = np.array([[11, 5], [11, 8]], dtype=np.int64)

        reference = wm.fit_action_reference(Index(), {})
        self.assertEqual(len(reference['reference']), 2)
        self.assertEqual(len(reference['mean']), len(wm.ACTIONS))


class McDropoutDeterminismTests(unittest.TestCase):
    """MC dropout must be reproducible per seed without losing its stochastic spread."""

    def setUp(self):
        if wm.torch is None:
            self.skipTest('PyTorch unavailable')
        torch = wm.torch
        torch.manual_seed(0)
        architecture = {'algorithm': 'moup_mlp', 'history_minutes': 60, 'history_steps': 24,
                        'sampling_interval_seconds': 60, 'hidden_size': 8, 'dilations': [],
                        'head_dropout': 0.0, 'head_layout': 'shared', 'attention_heads': 2,
                        'latent_size': 4, 'moving_average_kernel': 5, 'pool_scales': [1, 4],
                        'sensor_channels': len(wm.FEATURES) - len(wm.ACTIONS),
                        'action_channels': len(wm.ACTIONS), 'mc_dropout': .1,
                        'mc_passes': 6, 'mc_seed': 44, 'uncertainty_mode': 'mc_dropout',
                        'head_groups': {'shared': list(range(len(wm.CONTROL_TARGETS)))},
                        'latest_sensor_to_head': False, 'latest_sensor_names': [],
                        'latest_sensor_indices': [], 'input_channels': len(wm.FEATURES),
                        'outputs': len(wm.CONTROL_TARGETS),
                        'target_order': list(wm.CONTROL_TARGETS), 'log_variance_bounds': [-8, 2]}
        self.model = wm.DynamicsNetwork(architecture).eval()
        self.prepared = {'sequence': torch.randn(30, 24, len(wm.FEATURES)).numpy(),
                         'extra': torch.zeros(30, len(wm.ACTIONS) + 3).numpy()}

    def passes(self, seed):
        return wm.neural_components([self.model], self.prepared, 'cpu', 16,
                                    mc_passes=6, mc_seed=seed)[0]

    def test_same_seed_reproduces_exactly(self):
        np.testing.assert_array_equal(self.passes(44), self.passes(44))

    def test_passes_within_a_call_still_differ(self):
        """Reseeding must not collapse the dropout spread the estimator relies on."""
        components = self.passes(44)
        self.assertFalse(np.array_equal(components[0], components[1]))
        self.assertGreater(float(components.std(axis=0).mean()), 0.0)

    def test_different_seeds_give_different_draws(self):
        self.assertFalse(np.array_equal(self.passes(44), self.passes(45)))

    def test_rng_state_is_restored_after_seeded_prediction(self):
        torch = wm.torch
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        self.passes(44)
        after = torch.get_rng_state()
        self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
