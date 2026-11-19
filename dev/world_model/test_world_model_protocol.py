"""Regression tests for cohort coverage, causal histories and invalid-score handling."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import world_models as wm
import world_model_protocol as wp


def example_frame(rows=800, interval=5, cultivation=11):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({c: 10 + rng.normal(size=rows).cumsum() * .005 for c in dict.fromkeys(wm.FEATURES + wm.CONTROL_TARGETS)})
    frame['1_ph'] = 7 + rng.normal(0,.01,rows)
    frame['kultivasi'] = cultivation
    frame['timestamp'] = pd.date_range('2026-01-01', periods=rows, freq=f'{interval}s')
    frame['n_data'] = 12
    frame.attrs['sampling_interval_seconds'] = interval
    return frame


def settings(interval=5):
    return {'targets': wm.CONTROL_TARGETS.copy(), 'features': wm.FEATURES.copy(), 'actions': wm.ACTIONS.copy(),
            'context': wm.CONTEXT.copy(), 'label_mode': 'proxy_prototype', 'target_transform': 'mixed',
            'target_provenance': {t:'test proxy' for t in wm.CONTROL_TARGETS},
            'history_minutes': 1, 'sampling_interval_seconds':interval, 'sampling_block_minutes':1,
            'window_sampling':'temporal_strata', 'seed':42, 'progress_bars':False,
            'evaluation_working_budget_gib':1, 'rollout_starts':4,
            'input_filter':{'mode':'none'}}


class WorldModelProtocolTests(unittest.TestCase):
    def test_nonfinite_target_invalidates_aggregate_and_zero_variance_is_distinct(self):
        truth=np.ones((5,12)); prediction=truth.copy(); prediction[:,0]=np.nan
        prep={'targets':wm.CONTROL_TARGETS,'training_level_std':[1.]*12}
        table,score=wm.point_metrics(truth,prediction,np.zeros_like(truth),prep)
        self.assertEqual(score,np.inf)
        self.assertEqual(table.prediction_status.ne('valid').sum(),1)
        prep['training_level_std'][0]=0
        table,score=wm.point_metrics(truth,truth,np.zeros_like(truth),prep)
        self.assertEqual(score,0.)
        self.assertEqual(table.normalization_status.iloc[0],'zero_training_variance')

    def test_temporal_sample_spans_episode_without_using_target_values(self):
        frame=example_frame(10000)
        eligible=np.arange(12,len(frame))
        selected=wp.spread_sample(eligible,100,frame,42)
        self.assertEqual(len(np.unique(selected)),100)
        self.assertLess(selected[0],120)
        self.assertGreater(selected[-1],9900)
        frame['1_ph']=np.arange(len(frame))*100
        np.testing.assert_array_equal(selected,wp.spread_sample(eligible,100,frame,42))
        report=wp.coverage(frame,selected,eligible,wm.ACTIONS)
        self.assertGreater(report['selected_hour_bins'],10)

    def test_filter_is_causal_and_never_filters_treatments_or_labels(self):
        raw=np.arange(6*len(wm.FEATURES),dtype=float).reshape(1,6,len(wm.FEATURES))
        spec={'mode':'trailing_median','window_seconds':15,'features':[wm.TARGETS[0]]}
        filtered=wp.input_filter(raw,spec,wm.FEATURES,5)
        altered=raw.copy();altered[:,4:,:]+=100000
        second=wp.input_filter(altered,spec,wm.FEATURES,5)
        np.testing.assert_array_equal(filtered[:,:4],second[:,:4])
        np.testing.assert_array_equal(filtered[:,:,-5:],raw[:,:,-5:])
        with self.assertRaises(ValueError):wp.input_filter(raw,{**spec,'features':[wm.ACTIONS[0]]},wm.FEATURES,5)

    def test_lazy_and_evaluation_transforms_agree_with_filter_and_target_levels(self):
        frame=example_frame(); config=settings()
        config['input_filter']={'mode':'trailing_median','window_seconds':15,'features':[wm.TARGETS[0]]}
        index=wm.WindowIndex({11:frame},config,24)
        prep=wm.fit_preprocessing(index,{11:frame})
        raw=index.raw_batch([0,1]); prepared=wm.transform(raw,prep)
        self.assertEqual(prep['input_filter'],config['input_filter'])
        np.testing.assert_array_equal(raw['truth'],frame[wm.CONTROL_TARGETS].iloc[index.entries[:2,1]].to_numpy(np.float32))
        if wm.torch is not None:
            lazy=wm.LazyWindows(index,prep)
            seq,extra,target=lazy[0]
            np.testing.assert_allclose(seq.numpy(),prepared['sequence'][0],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(extra.numpy(),prepared['extra'][0],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(target.numpy(),prepared['y'][0],rtol=1e-6,atol=1e-6)

    def test_gap_exclusion_and_explicit_cohort_reconstruction(self):
        frame=example_frame();frame.loc[frame.index>=400,'timestamp']+=pd.Timedelta(hours=1)
        config=settings();raw,_=wm.raw_windows({11:frame},config,30)
        starts=[a['index'] for a in raw['anchors']]
        self.assertFalse(any(400<=i<412 for i in starts))
        again,_=wm.raw_windows({11:frame},config,anchor_override={11:starts})
        self.assertEqual(raw['anchors'],again['anchors'])
        with self.assertRaises(ValueError):wm.raw_windows({11:frame},config,anchor_override={11:[401]})

    def test_recursive_starts_do_not_depend_on_small_checkpoint_pool(self):
        frame=example_frame(2000);config=settings()
        chosen,coverage=wm.rollout_cohort({11:frame},config,[1,2])
        self.assertEqual(len(chosen),4)
        self.assertGreater(chosen[-1]['index'],1300)
        self.assertGreater(coverage.selected_hour_bins.iloc[0],1)

    def test_quality_flags_are_diagnostic_only(self):
        frame=example_frame();original=frame[wm.CONTROL_TARGETS].copy()
        config={'enabled':True,'median_seconds':180,'mad_multiplier':6.,'return_fraction':.25,
                'flat_minutes':1.,'ph_screen':[0.,14.]}
        frame.loc[100,'1_ph']=17.
        expected=frame[wm.CONTROL_TARGETS].copy()
        report=wp.attach_quality(frame,wm.CONTROL_TARGETS,5,config)
        pd.testing.assert_frame_equal(frame[wm.CONTROL_TARGETS],expected)
        self.assertEqual(len(report),12)
        self.assertTrue(frame['__quality_range_1_ph'].iloc[100])
        self.assertEqual(wm.valid_anchors(frame,12,5).size,788)
        self.assertTrue(wp.priority_summary(pd.DataFrame({'target':wp.PRIORITIES,'NRMSE':[.1]*5,'prediction_status':['valid']*5}),wp.PRIORITIES)['priority_complete_valid'])

    def test_nonfinite_recursive_prediction_marks_all_later_horizons_invalid(self):
        frame=example_frame(200);config=settings()
        index=wm.WindowIndex({11:frame},config,12);prep=wm.fit_preprocessing(index,{11:frame})
        original=wm.predict
        wm.predict=lambda fitted,prepared:(np.full((len(prepared['y']),12),np.nan),None)
        try:
            table=wm.recursive_metrics({'algorithm':'ridge','preprocessing':prep}, {11:frame}, {}, config, [1/12,1/6])
        finally:
            wm.predict=original
        self.assertTrue(table.prediction_status.eq('invalid_rollout_nonfinite').all())
        self.assertTrue(table.failed_at_step.eq(1).all())
        self.assertEqual(table.horizon_minutes.nunique(),2)

    def test_minute_incomplete_bins_are_excluded_without_filling(self):
        frame=example_frame(30,interval=60);frame.loc[10,'n_data']=11
        config=settings(60)
        raw,_=wm.raw_windows({11:frame},config,None)
        indices=[a['index'] for a in raw['anchors']]
        self.assertNotIn(10,indices);self.assertNotIn(11,indices)
        self.assertEqual(raw['observation_operator'],'interval_mean')

    def test_reference_initialization_replaces_switches_without_changing_behavior(self):
        self.assertEqual(wm.initialization_strategy({'initialization':'reference'},'mixed'),'persistence')
        self.assertEqual(wm.initialization_strategy({'initialization':'reference'},'delta'),'persistence')
        self.assertEqual(wm.initialization_strategy({'initialization':'reference'},'level'),'training_mean')
        self.assertEqual(wm.initialization_strategy({'initialization':'random'},'mixed'),'random')
        self.assertEqual(wm.initialization_strategy({'persistence_initialization':True},'mixed'),'persistence')
        self.assertEqual(wm.initialization_strategy({'training_mean_initialization':True},'level'),'training_mean')
        with self.assertRaises(ValueError):wm.initialization_strategy({'initialization':'reference','persistence_initialization':True},'mixed')
        with self.assertRaises(ValueError):wm.initialization_strategy({'initialization':'typo'},'mixed')

    def test_candidate_only_classical_settings_match_previous_redundant_settings(self):
        if wm.missing_dependencies('ridge') or wm.missing_dependencies('gpr'):
            self.skipTest('Classical fitting dependencies unavailable')
        frame=example_frame(80);config=settings()
        config.update({'cpu_threads':1,'gpr_max_train_windows':100,'gpr_working_budget_gib':1})
        index=wm.WindowIndex({11:frame},config,12);prep=wm.fit_preprocessing(index,{11:frame})
        raw=index.raw_batch(np.arange(len(index)));prepared=wm.transform(raw,prep)
        train={'flat':prepared['flat'],'y':prepared['y']}
        pairs={
            'ridge':({'alpha':1000.,'alpha_candidates':[10.,100.],'solver':'cholesky','fit_intercept':True},['alpha']),
            'gpr':({'length_scale_factor':1.,'noise_level':.05,'persistence_prior':True,
                    'kernel_candidates':[{'length_scale_factor':.5,'noise_level':.05},{'length_scale_factor':1.,'noise_level':.2}]},['length_scale_factor','noise_level'])}
        for algorithm,(old,removed) in pairs.items():
            config['classical']={algorithm:old}
            previous=wm.train_classical(algorithm,train,prepared,prep,config)
            config['classical']={algorithm:{k:v for k,v in old.items() if k not in removed}}
            current=wm.train_classical(algorithm,train,prepared,prep,config)
            np.testing.assert_allclose(wm.predict(previous,prepared)[0],wm.predict(current,prepared)[0],rtol=1e-12,atol=1e-12)
            self.assertEqual(previous['selected_hyperparameters'],current['selected_hyperparameters'])

    def test_treatment_sampling_preserves_rare_events_and_shared_budget(self):
        frame=example_frame(400);frame[wm.ACTIONS]=0.
        for action,at in zip(wm.ACTIONS[:3],[100,140,180]):frame.loc[at,action]=1.
        frame.loc[240:,wm.ACTIONS[3]]=1.
        config=settings();config['training_sampling']={'event_fraction':.25,'start_spacing_seconds':30,'spacing_candidates_seconds':[5,30,60]}
        index=wm.WindowIndex({11:frame},config,40)
        selected=index.entries[:,1]
        self.assertEqual(len(selected),40)
        self.assertEqual(len(np.unique(selected)),40)
        self.assertTrue(np.isin([100,140,180,240],selected).all())
        again=wm.WindowIndex({11:frame},config,40)
        np.testing.assert_array_equal(index.entries,again.entries)
        self.assertEqual(index.raw_batch([0])['sequence'].shape[1],12)
        config['training_sampling']['event_fraction']=1.1
        with self.assertRaises(ValueError):wm.WindowIndex({11:frame},config,40)

    def test_spacing_changes_starts_only_and_preserves_events_in_uncapped_track(self):
        frame=example_frame(400);frame[wm.ACTIONS]=0.;frame.loc[101,wm.ACTIONS[0]]=1.
        config=settings();config['training_sampling']={'event_fraction':.25,'start_spacing_seconds':30,'spacing_candidates_seconds':[5,30,60]}
        original=frame.copy(deep=True)
        index=wm.WindowIndex({11:frame},config,None)
        self.assertIn(101,index.entries[:,1])
        self.assertLess(len(index),100)
        pd.testing.assert_frame_equal(frame,original)
        preview=wp.training_spacing_preview({11:frame},config,wm.ACTIONS,lambda f:wm.valid_anchors(f,12,5))
        self.assertEqual(preview.iloc[0].uncapped_starts_with_events_preserved,388)
        self.assertGreater(preview.iloc[0].uncapped_starts_with_events_preserved,preview.iloc[-1].uncapped_starts_with_events_preserved)

    def test_treatment_audit_is_separate_and_does_not_change_checkpoint_score(self):
        frame=example_frame(100);frame[wm.ACTIONS]=0.;frame.loc[40,wm.ACTIONS[0]]=1.
        config=settings();config['treatment_diagnostics']={'windows_per_cultivation':16,'post_event_minutes':1}
        chosen,coverage=wp.treatment_audit_starts({11:frame},config,wm.ACTIONS,lambda f:wm.valid_anchors(f,12,5))
        self.assertIn(40,chosen[11]);self.assertLessEqual(chosen[11].max(),52)
        index=wm.WindowIndex({11:frame},config,16);prep=wm.fit_preprocessing(index,{11:frame})
        raw,_=wm.raw_windows({11:frame},config,16);validation=wm.transform(raw,prep)
        event,_=wm.raw_windows({11:frame},config,anchor_override=chosen)
        validation['treatment_diagnostic']=wm.transform(event,prep)
        class ZeroEstimator:
            def predict(self,x):return np.zeros((len(x),12))
        fitted={'kind':'classical','algorithm':'ridge','estimator':ZeroEstimator(),'preprocessing':prep}
        table,summary=wm.treatment_validation_diagnostic(fitted,validation)
        self.assertEqual(len(table),12)
        self.assertEqual(summary['validation_treatment_starts'],len(chosen[11]))
        self.assertTrue(np.isfinite(summary['validation_treatment_macro_NRMSE']))
        np.testing.assert_array_equal(validation['raw']['truth'],raw['truth'])

    def test_dependency_and_light_reports_are_descriptive(self):
        frame=example_frame(150);frame['1_tss(mg/L)']=1.5*frame['1_turbidity(NTU)']
        frame['Lampu (PWM%)']=0.;frame.loc[60:,'Lampu (PWM%)']=100.
        original=frame.copy(deep=True)
        dependency=wp.output_dependency_report({11:frame},[('1_turbidity(NTU)','1_tss(mg/L)')])
        self.assertAlmostEqual(dependency.affine_slope.iloc[0],1.5,places=10)
        self.assertLess(dependency.residual_std_fraction.iloc[0],1e-10)
        report,hourly=wp.light_regime_report({11:frame},5)
        self.assertEqual(report.loc[report.group.eq('lamp_changed'),'readings'].iloc[0],1)
        self.assertEqual(report.loc[report.group.eq('all'),'logged_lamp_changes'].iloc[0],1)
        self.assertIn(11,hourly)
        pd.testing.assert_frame_equal(frame,original)

    def test_neural_treatment_audit_does_not_change_updates_or_checkpoint_choice(self):
        if wm.torch is None:self.skipTest('PyTorch unavailable')
        frame=example_frame(80);config=settings()
        config.update({'cpu_threads':1,'loader_workers':0,'pin_memory':False,'validation_batch_size':4,
                       'neural':{'gru':{'hidden_size':4,'epochs':1,'batch_size':4,'learning_rate':.001,
                       'optimizer':'adamw','weight_decay':.0001,'head_dropout':.1,'gradient_clip_norm':1.,
                       'loss':'normalized_mse_plus_nll','nll_weight':.01,'initialization':'reference',
                       'include_initial_checkpoint':True,'lr_scheduler':False,'patience':2,'min_delta':1e-6}}})
        index=wm.WindowIndex({11:frame},config,12);prep=wm.fit_preprocessing(index,{11:frame})
        validation=wm.transform(index.raw_batch(np.arange(8)),prep)
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):
            first=wm.train_neural('gru',index,validation,prep,config,'cpu')
            with_audit={**validation,'treatment_diagnostic':wm.transform(index.raw_batch([8,9,10,11]),prep)}
            second=wm.train_neural('gru',index,with_audit,prep,config,'cpu')
        for key,weights in first['models'][0].state_dict().items():
            np.testing.assert_array_equal(weights.numpy(),second['models'][0].state_dict()[key].numpy())
        self.assertEqual(first['selected_checkpoints'],second['selected_checkpoints'])
        self.assertIn('validation_treatment_macro_NRMSE',second['training_curve'][-1])

    def test_neural_batch_logging_does_not_force_per_batch_refresh(self):
        if wm.torch is None or not wm.TQDM_AVAILABLE:
            self.skipTest('PyTorch and tqdm are required')
        import contextlib
        import io
        from unittest.mock import patch
        from tqdm import tqdm as text_tqdm

        batch_updates = []
        batch_intervals = []

        class ObservedProgress(text_tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.unit == 'batch':
                    batch_intervals.append(self.mininterval)

            def set_postfix(self, *args, **kwargs):
                if self.unit == 'batch':
                    batch_updates.append(kwargs.get('refresh', True))
                return super().set_postfix(*args, **kwargs)

        frame = example_frame(80)
        config = settings()
        config.update(cpu_threads=1, loader_workers=0, pin_memory=False, validation_batch_size=4)
        config['neural'] = {'gru': {'hidden_size': 4, 'epochs': 1, 'batch_size': 4,
            'learning_rate': .001, 'optimizer': 'adamw', 'weight_decay': .0001,
            'loss': 'normalized_mse_plus_nll', 'nll_weight': .01, 'initialization': 'reference',
            'include_initial_checkpoint': True, 'lr_scheduler': False, 'patience': 2, 'min_delta': 1e-6}}
        index = wm.WindowIndex({11: frame}, config, 12)
        prep = wm.fit_preprocessing(index, {11: frame})
        validation = wm.transform(index.raw_batch(np.arange(8)), prep)
        config['progress_bars'] = True
        with patch.object(wm, 'tqdm', ObservedProgress), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            fitted = wm.train_neural('gru', index, validation, prep, config, 'cpu')
        self.assertEqual(len(fitted['models']), 1)
        self.assertEqual(batch_updates, [False, False, False])
        self.assertEqual(batch_intervals, [2.])


if __name__=='__main__':unittest.main()
