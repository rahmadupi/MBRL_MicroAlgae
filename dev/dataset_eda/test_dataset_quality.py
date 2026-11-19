"""Regression checks for descriptive quality screens, independent of source datasets."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import dataset_quality as dq


class DatasetQualityTests(unittest.TestCase):
    def setUp(self):
        self.targets = list(dq.LABELS)
        self.config = {
            'priority_targets': ['1_ph', '1_cod(mg/L)', '1_bod(mg/L)', '1_do(mg/L)', '1_nh3(mg/L)'],
            'median_window_seconds': 180, 'mad_multiplier': 6., 'spike_return_fraction': .25,
            'flatline_review_minutes': 10., 'spike_review_pct': .1, 'within_variance_review_pct': 20.,
            'detail_minutes': 30, 'max_event_examples_per_target': 5, 'spectrum_min_samples': 20,
            'spectrum_max_samples': 500, 'spectrum_nperseg': 64, 'fast_period_seconds': 60.,
            'ph_screen': [0., 14.], 'nonnegative_targets': [t for t in self.targets if t != '1_ph'],
            'rounding_tolerance': {}, 'default_rounding_tolerance': .0051,
        }
        rng = np.random.default_rng(45)
        self.frame = pd.DataFrame({t: 10 + rng.normal(0, .02, 240) for t in self.targets})
        self.frame['1_ph'] = 7 + rng.normal(0, .01, 240)
        self.frame['timestamp'] = pd.date_range('2026-01-01', periods=240, freq='5s')
        self.frame['Pump1_NaHCO3 (mL)'] = 0.

    def inspect(self, frame, minute_path=None):
        return dq.inspect_quality(frame, pd.to_datetime(frame.timestamp), 11, 5, self.targets,
                                  ['Pump1_NaHCO3 (mL)'], self.config, minute_path)

    def test_complete_bin_variance_and_aggregation_are_exact(self):
        original = self.frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'minute.csv'
            means = self.frame.set_index('timestamp')[self.targets].resample('min').mean()
            means['n_data'] = 12
            means.to_csv(path)
            result = self.inspect(self.frame, path)
        metrics = result['metrics'].set_index('target')
        self.assertEqual(len(metrics), 12)
        self.assertTrue(metrics.provided_minute_pairs.eq(20).all())
        self.assertTrue(metrics.provided_mean_within_tolerance_pct.eq(100).all())
        for target in self.targets:
            row = metrics.loc[target]
            self.assertAlmostEqual(row.within_minute_variance_mean + row.between_minute_variance,
                                   self.frame[target].var(ddof=0), places=10)
        pd.testing.assert_frame_equal(self.frame, original)

    def test_spike_and_action_distance_are_independent_of_datetime_storage_unit(self):
        for unit in ['us', 'ns']:
            frame = self.frame.copy()
            frame['timestamp'] = frame.timestamp.astype(f'datetime64[{unit}]')
            frame.loc[100, '1_ph'] = 12.
            frame.loc[99, 'Pump1_NaHCO3 (mL)'] = 2.
            result = self.inspect(frame)
            row = result['metrics'].set_index('target').loc['1_ph']
            self.assertGreaterEqual(row.isolated_spike_candidates, 1)
            self.assertGreaterEqual(row.spikes_near_logged_action_60s, 1)
            detail = result['detail']['columns']['1_ph']['raw']
            self.assertGreater(np.isfinite(detail).mean(), .95)
            frame.loc[99, 'Pump1_NaHCO3 (mL)'] = 0.
            frame.loc[1, 'Pump1_NaHCO3 (mL)'] = 2.
            row = self.inspect(frame)['metrics'].set_index('target').loc['1_ph']
            self.assertEqual(row.spikes_near_logged_action_60s, 0)

    def test_gap_is_not_a_change_or_complete_minute(self):
        frame = self.frame.iloc[:20].copy()
        frame.loc[frame.index >= 10, 'timestamp'] += pd.Timedelta(hours=2)
        frame.loc[frame.index >= 10, '1_do(mg/L)'] += 100
        row = self.inspect(frame)['metrics'].set_index('target').loc['1_do(mg/L)']
        self.assertEqual(row.adjacent_pairs, 18)
        self.assertEqual(row.complete_native_minutes, 0)

    def test_constant_channel_has_undefined_outlier_screen_and_no_noise_certificate(self):
        frame = self.frame.copy()
        frame['1_tds(ppm)'] = 50.
        metrics = self.inspect(frame)['metrics']
        row = metrics.set_index('target').loc['1_tds(ppm)']
        self.assertFalse(row.outlier_screen_assessed)
        self.assertAlmostEqual(row.longest_flat_minutes, 239 * 5 / 60)
        verdict = dq.channel_verdicts(metrics, self.config).set_index('target').loc['1_tds(ppm)']
        self.assertEqual(verdict.screen_verdict, 'REVIEW')

    def test_duplicates_and_nonfinite_values_are_excluded_from_temporal_metrics(self):
        frame = self.frame.copy()
        frame.loc[1, 'timestamp'] = frame.loc[0, 'timestamp']
        frame.loc[2, '1_ph'] = np.inf
        frame.loc[3, '1_cod(mg/L)'] = np.nan
        metrics = self.inspect(frame)['metrics'].set_index('target')
        self.assertEqual(metrics.loc['1_ph', 'infinite_values'], 1)
        self.assertEqual(metrics.loc['1_ph', 'usable_rows'], 237)
        self.assertGreater(metrics.loc['1_cod(mg/L)', 'missing_pct'], 0)

    def test_persistent_step_is_not_an_isolated_spike(self):
        frame = self.frame.copy()
        frame.loc[120:, '1_ph'] += 1.
        examples = self.inspect(frame)['examples']
        if not examples.empty:
            self.assertFalse((examples.target.eq('1_ph') & examples.timestamp.eq(frame.timestamp.iloc[120])).any())


if __name__ == '__main__':
    unittest.main()
