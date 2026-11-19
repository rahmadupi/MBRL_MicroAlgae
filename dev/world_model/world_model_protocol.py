"""Shared sampling, descriptive quality flags and reporting for world-model experiments."""

import numpy as np
import pandas as pd
import dataset_quality as dq

PRIORITIES = ['1_ph', '1_cod(mg/L)', '1_bod(mg/L)', '1_do(mg/L)', '1_nh3(mg/L)']


def spread_sample(eligible, count, frame, seed=42):
    """Equal temporal strata, without inspecting target values or treatment outcomes."""
    eligible = np.asarray(eligible, dtype=int)
    if count is None or count >= len(eligible):
        return eligible.copy()
    if count < 1:
        raise ValueError('Window cap must be positive or None.')
    ticks = dq.nanosecond_ticks(frame.timestamp.iloc[eligible])
    # Timestamp strata span the full episode; empty strata need no invented observations.
    edges = np.linspace(float(ticks[0]), float(ticks[-1]) + 1, count + 1)
    groups = np.clip(np.searchsorted(edges, ticks, side='right') - 1, 0, count - 1)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(eligible)), groups))
    _, first = np.unique(groups[order], return_index=True)
    chosen = order[first]
    if len(chosen) < count:
        remainder = np.setdiff1d(np.arange(len(eligible)), chosen)
        chosen = np.r_[chosen, rng.choice(remainder, count - len(chosen), replace=False)]
    return np.sort(eligible[chosen])


def select_starts(eligible, count, frame, config, legacy_sampler, rng):
    if config.get('window_sampling', 'legacy_blocks') == 'legacy_blocks':
        return legacy_sampler(eligible, count, config['sampling_block_steps'], rng)
    if config['window_sampling'] != 'temporal_strata':
        raise ValueError('Unknown window-sampling strategy.')
    return spread_sample(eligible, count, frame, int(rng.integers(0, 2**31)))


def coverage(frame, selected, eligible, actions):
    selected, eligible = np.asarray(selected, int), np.asarray(eligible, int)
    def facts(indices, prefix):
        t = frame.timestamp.iloc[indices]
        hour = t.dt.hour
        pump = frame.iloc[indices][actions[:3]].gt(0).any(axis=1)
        return {prefix + 'starts': len(indices), prefix + 'dates': t.dt.date.nunique(),
                prefix + 'hour_bins': t.dt.floor('h').nunique(),
                prefix + 'pump_active_starts': int(pump.sum()),
                prefix + 'day_starts': int(hour.between(6, 17).sum()),
                prefix + 'night_starts': int((~hour.between(6, 17)).sum())}
    return {**facts(eligible, 'eligible_'), **facts(selected, 'selected_'),
            'coverage_note': 'Clock day/night proxy; logged dose rows are not verified commands. Cohort selection rule is reported separately.'}


def attach_quality(frame, targets, interval, settings):
    """Flags are retrospective diagnostics only; they never select model inputs or starts."""
    if not settings.get('enabled', False):
        return pd.DataFrame()
    times = pd.DatetimeIndex(frame.timestamp)
    good = np.asarray(times.notna() & ~times.duplicated(keep=False))
    if 'status' in frame:
        good &= frame.status.eq('OK').to_numpy()
    if interval == 60:
        good &= frame.n_data.eq(12).to_numpy()
    adjacent = np.r_[False, np.diff(dq.nanosecond_ticks(times)) == interval * 10**9]
    window = max(3, round(settings['median_seconds'] / interval))
    window += window % 2 == 0
    rows = []
    for name in targets:
        values = frame[name].to_numpy(float)
        valid, connected, segments = dq.segments_for(values, good, adjacent)
        median = dq.local_median(values, valid, segments, window)
        residual = values - median
        scale = dq.robust_sigma(residual)
        assessed = np.isfinite(scale) and scale > 0
        candidate = valid & (np.abs(residual) > settings['mad_multiplier'] * scale) if assessed else np.zeros(len(frame), bool)
        dx = values - np.r_[np.nan, values[:-1]]
        next_dx = np.r_[dx[1:], np.nan]
        spike = (candidate & connected & np.r_[connected[1:], False] & (dx * next_dx < 0) &
                 (np.abs(np.r_[values[1:], np.nan] - np.r_[np.nan, values[:-1]]) <=
                  settings['return_fraction'] * np.maximum(np.abs(dx), np.abs(next_dx))))
        same = connected & (values == np.r_[np.nan, values[:-1]])
        run = np.cumsum(~same)
        counts = pd.Series(run[valid]).value_counts()
        duration = (counts - 1) * interval / 60
        flat = valid & np.isin(run, duration.index[duration >= settings['flat_minutes']])
        bounds = settings['ph_screen']
        outside = np.isfinite(values) & (((values < bounds[0]) | (values > bounds[1])) if name == '1_ph'
                                       else (values < 0) if name != '1_temp_air(C)' else False)
        for kind, flags in [('spike', spike), ('flat', flat), ('range', outside)]:
            frame[f'__quality_{kind}_{name}'] = flags
        rows.append({'target': name, 'spike_candidates': int(spike.sum()),
                     'long_flat_run_readings': int(flat.sum()), 'range_screen_readings': int(outside.sum()),
                     'outlier_screen_assessed': assessed, 'residual_robust_sigma': scale,
                     'provenance': dq.provenance(name, int(frame.kultivasi.iloc[0]))})
    return pd.DataFrame(rows)


def input_filter(sequence, settings, features, interval):
    mode = settings.get('mode', 'none')
    if mode == 'none':
        return sequence
    if mode != 'trailing_median':
        raise ValueError('Only none or trailing_median input filters are supported.')
    width = float(settings['window_seconds']) / interval
    if width < 1 or not np.isclose(width, round(width)):
        raise ValueError('Filter duration must be a positive whole number of sample steps.')
    columns = settings['features']
    if not set(columns).issubset(features) or any(c.startswith('Pump') or c in ['Aerator_RU (L/s)', 'Lampu (PWM%)'] for c in columns):
        raise ValueError('Filtering is limited to declared sensor-history columns; treatments remain unchanged.')
    result = np.array(sequence, copy=True)
    indices = [features.index(c) for c in columns]
    width = int(round(width))
    for t in range(sequence.shape[1]):
        result[:, t, indices] = np.median(sequence[:, max(0, t-width+1):t+1, :][:, :, indices], axis=1)
    return result


def quality_groups(frame, indices, target, actions):
    indices = np.asarray(indices, dtype=int)
    out = {'all_readings': np.ones(len(indices), bool)}
    for kind in ['spike', 'flat', 'range']:
        key = f'__quality_{kind}_{target}'
        if key in frame:
            out[kind + '_flagged'] = frame[key].iloc[indices].to_numpy(bool)
    flagged = np.logical_or.reduce([v for k,v in out.items() if k != 'all_readings']) if len(out)>1 else np.zeros(len(indices), bool)
    out['no_quality_flag'] = ~flagged
    out['pump_active'] = frame.iloc[indices][actions[:3]].gt(0).any(axis=1).to_numpy()
    previous = np.maximum(indices - 1, 0)
    ticks = dq.nanosecond_ticks(frame.timestamp.iloc[indices])
    previous_ticks = dq.nanosecond_ticks(frame.timestamp.iloc[previous])
    continuous = (indices > 0) & ((ticks - previous_ticks) == frame.attrs.get('sampling_interval_seconds', 60) * 10**9)
    values = frame[actions[3:]].iloc[indices].to_numpy()
    prior = frame[actions[3:]].iloc[previous].to_numpy()
    out['actuator_change'] = np.any(values != prior, axis=1) & continuous
    hours = frame.timestamp.iloc[indices].dt.hour
    out['clock_day'] = hours.between(6,17).to_numpy()
    out['clock_night'] = ~out['clock_day']
    return out


def cohort_groups(frames, anchors, target, actions):
    table = pd.DataFrame(anchors)
    combined = {}
    for cultivation in table.cultivation.unique():
        positions = np.flatnonzero(table.cultivation.eq(cultivation))
        indices = table.iloc[positions]['index'].to_numpy(int)
        for name, mask in quality_groups(frames[int(cultivation)], indices, target, actions).items():
            if name not in combined:
                combined[name] = np.zeros(len(anchors), bool)
            combined[name][positions] = mask
    return combined


def stratified_metrics(raw, predicted, targets, preprocessing, frames, actions, metric):
    rows = []
    anchors = pd.DataFrame(raw['anchors'])
    for k in anchors.cultivation.unique():
        positions = np.flatnonzero(anchors.cultivation.eq(k))
        indices = anchors.iloc[positions]['index'].to_numpy(int)
        for j, target in enumerate(targets):
            for group, mask in quality_groups(frames[int(k)], indices, target, actions).items():
                selected = positions[mask]
                if not len(selected):
                    continue
                table, _ = metric(raw['truth'][selected], predicted[selected, j:j+1], raw['latest'][selected], preprocessing, [target])
                rows.append({'cultivation': int(k), 'group': group, 'observations': len(selected), **table.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def priority_summary(table, priority_targets):
    selected = table.loc[table.target.isin(priority_targets)]
    valid = len(selected) == len(priority_targets) and selected.get('prediction_status', pd.Series('valid',index=selected.index)).eq('valid').all()
    values = selected.NRMSE.to_numpy(float)
    return {'priority_macro_NRMSE': float(np.mean(values[np.isfinite(values)])) if valid and np.isfinite(values).any() else np.nan,
            'priority_targets_present': len(selected), 'priority_complete_valid': bool(valid)}


def prediction_checks(predicted, targets):
    result = {'nonfinite_predictions': int((~np.isfinite(predicted)).sum()),
              'negative_nonnegative_target_predictions': 0, 'ph_outside_0_14_predictions': 0}
    for j,name in enumerate(targets):
        values = predicted[:,j]
        finite = np.isfinite(values)
        if name == '1_ph':
            result['ph_outside_0_14_predictions'] += int((finite & ((values<0)|(values>14))).sum())
        elif name != '1_temp_air(C)':
            result['negative_nonnegative_target_predictions'] += int((finite & (values<0)).sum())
    return result


def treatment_groups(frame, eligible, actions, interval):
    groups = {name: eligible[frame[name].iloc[eligible].gt(0).to_numpy()] for name in actions[:3]}
    previous = np.maximum(eligible - 1, 0)
    continuous = ((eligible > 0) & ((dq.nanosecond_ticks(frame.timestamp.iloc[eligible]) -
                  dq.nanosecond_ticks(frame.timestamp.iloc[previous])) == interval * 10**9))
    for name in actions[3:]:
        changed = frame[name].iloc[eligible].to_numpy() != frame[name].iloc[previous].to_numpy()
        groups[name] = eligible[continuous & changed]
    return groups


def spaced_starts(eligible, seconds, interval):
    steps = float(seconds) / interval
    if steps < 1 or not np.isclose(steps, round(steps)):
        raise ValueError('Training-start spacing must be a positive whole number of source steps.')
    cuts = np.flatnonzero(np.diff(eligible) != 1) + 1
    runs = [run[::int(round(steps))] for run in np.split(np.asarray(eligible, int), cuts) if len(run)]
    return np.concatenate(runs) if runs else np.array([], int)


def event_reserve(groups, count, frame, rng):
    available = [v for v in groups.values() if len(v)]
    if not available or count <= 0:
        return np.array([], int)
    per_group = max(1, count // len(available))
    selected = np.unique(np.concatenate([spread_sample(v, min(per_group, len(v)), frame,
                            int(rng.integers(0, 2**31))) for v in available]))
    if len(selected) > count:
        selected = spread_sample(selected, count, frame, int(rng.integers(0, 2**31)))
    union = np.unique(np.concatenate(available))
    remaining = np.setdiff1d(union, selected)
    if len(selected) < count and len(remaining):
        selected = np.r_[selected, spread_sample(remaining, count - len(selected), frame, int(rng.integers(0, 2**31)))]
    return np.sort(selected)


def training_starts(eligible, count, frame, config, actions, legacy_sampler, rng):
    settings = config.get('training_sampling')
    if settings is None:
        return select_starts(eligible, count, frame, config, legacy_sampler, rng)
    fraction = settings['event_fraction']
    if not 0 <= fraction <= 1:
        raise ValueError('Treatment-event training fraction must lie between zero and one.')
    interval = config['sampling_interval_seconds']
    groups = treatment_groups(frame, eligible, actions, interval)
    events = np.unique(np.concatenate(list(groups.values())))
    background = spaced_starts(eligible, settings['start_spacing_seconds'], interval)
    pool = np.union1d(background, events)
    if count is None or count >= len(pool):
        return pool
    reserved = event_reserve(groups, min(count, int(np.ceil(count * fraction))), frame, rng)
    remaining = np.setdiff1d(pool, reserved)
    selected = spread_sample(remaining, count - len(reserved), frame, int(rng.integers(0, 2**31))) if len(reserved) < count else []
    return np.sort(np.r_[reserved, selected].astype(int))


def training_spacing_preview(frames, config, actions, eligible_function):
    rows = []
    for k, frame in sorted(frames.items()):
        eligible = eligible_function(frame)
        groups = treatment_groups(frame, eligible, actions, config['sampling_interval_seconds'])
        events = np.unique(np.concatenate(list(groups.values())))
        for seconds in config['training_sampling']['spacing_candidates_seconds']:
            selected = np.union1d(spaced_starts(eligible, seconds, config['sampling_interval_seconds']), events)
            rows.append({'cultivation': k, 'start_spacing_seconds': seconds, 'all_eligible_starts': len(eligible),
                         'uncapped_starts_with_events_preserved': len(selected), 'logged_treatment_starts_preserved': len(events)})
    return pd.DataFrame(rows)


def treatment_audit_starts(frames, config, actions, eligible_function):
    settings = config['treatment_diagnostics']
    cap = settings['windows_per_cultivation']
    if not isinstance(cap, (int, np.integer)) or cap < 1:
        raise ValueError('Treatment audit window cap must be a positive integer.')
    if not np.isfinite(settings['post_event_minutes']) or settings['post_event_minutes'] < 0:
        raise ValueError('Treatment follow-up duration must be finite and nonnegative.')
    chosen, reports = {}, []
    for k, frame in sorted(frames.items()):
        eligible = eligible_function(frame)
        groups = treatment_groups(frame, eligible, actions, config['sampling_interval_seconds'])
        events = np.unique(np.concatenate(list(groups.values())))
        ticks = dq.nanosecond_ticks(frame.timestamp.iloc[eligible])
        event_ticks = dq.nanosecond_ticks(frame.timestamp.iloc[events])
        if len(events):
            right = np.searchsorted(event_ticks, ticks, side='right') - 1
            since = ticks - event_ticks[np.maximum(right, 0)]
            # Segment identity prevents an audit window from bridging a sensor gap.
            cuts = np.r_[True, np.diff(dq.nanosecond_ticks(frame.timestamp)) != config['sampling_interval_seconds'] * 10**9]
            segment = np.cumsum(cuts)
            same_run = segment[eligible] == segment[events[np.maximum(right, 0)]]
            candidates = eligible[(right >= 0) & same_run & (since <= settings['post_event_minutes'] * 60 * 10**9)]
        else:
            candidates = np.array([], int)
        cap = settings['windows_per_cultivation']
        rng = np.random.default_rng(config['seed'] + k)
        reserved = event_reserve(groups, cap // 2, frame, rng)
        rest = np.setdiff1d(candidates, reserved)
        extra = spread_sample(rest, cap-len(reserved), frame, config['seed']) if len(reserved)<cap else []
        chosen[k] = np.sort(np.r_[reserved,extra].astype(int))
        reports.append({'cultivation': k, 'cohort': 'separate_treatment_audit', **coverage(frame, chosen[k], candidates, actions)})
    return chosen, pd.DataFrame(reports)


def output_dependency_report(frames, pairs):
    rows = []
    for k, frame in sorted(frames.items()):
        for source, target in pairs:
            x, y = frame[source].to_numpy(float), frame[target].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            slope, intercept, corr, ratio = np.nan, np.nan, np.nan, np.nan
            if len(x)>1 and np.std(x)>0 and np.std(y)>0:
                slope = np.mean((x-x.mean())*(y-y.mean())) / np.var(x)
                intercept = y.mean()-slope*x.mean()
                corr = np.corrcoef(x,y)[0,1]
                ratio = np.std(y-(slope*x+intercept))/np.std(y)
            rows.append({'cultivation':k,'source':source,'target':target,'finite_pairs':len(x),
                         'correlation':corr,'affine_slope':slope,'affine_intercept':intercept,
                         'residual_std_fraction':ratio,'interpretation':'Descriptive dependence; does not prove independent calibration or causality'})
    return pd.DataFrame(rows)


def light_regime_report(frames, interval):
    rows, hourly = [], {}
    for k, frame in sorted(frames.items()):
        light = frame['2_light(Lux)'].to_numpy(float)
        pwm = frame['Lampu (PWM%)'].to_numpy(float)
        times = pd.DatetimeIndex(frame.timestamp)
        connected = np.r_[False,np.diff(dq.nanosecond_ticks(times))==interval*10**9]
        change = connected & (pwm != np.r_[np.nan,pwm[:-1]])
        clock_day = (times.hour>=6)&(times.hour<18)
        dx = light-np.r_[np.nan,light[:-1]]
        for group, mask in {'all':np.ones(len(frame),bool),'clock_day':clock_day,'clock_night':~clock_day,
                            'lamp_changed':change,'lamp_steady':~change}.items():
            valid = mask & np.isfinite(light)
            pairs = valid & connected & np.isfinite(np.r_[np.nan,light[:-1]])
            correlation = (np.corrcoef(light[pairs],np.r_[np.nan,light[:-1]][pairs])[0,1]
                           if pairs.sum()>1 and np.std(light[pairs])>0 and np.std(np.r_[np.nan,light[:-1]][pairs])>0 else np.nan)
            rows.append({'cultivation':k,'group':group,'readings':int(valid.sum()),
                         'light_mean_lux':float(np.mean(light[valid])) if valid.any() else np.nan,
                         'light_std_lux':float(np.std(light[valid])) if valid.any() else np.nan,
                         'adjacent_change_std_lux':float(np.std(dx[pairs])) if pairs.any() else np.nan,
                         'lag1_correlation':correlation,'logged_lamp_changes':int((valid&change).sum())})
        data = pd.DataFrame({'light':light,'pwm':pwm},index=times)
        hourly[k] = data.resample('h').agg(light_median=('light','median'),light_p10=('light',lambda x:x.quantile(.1)),
                                         light_p90=('light',lambda x:x.quantile(.9)),pwm_mean=('pwm','mean'))
    return pd.DataFrame(rows), hourly


def treatment_coverage(frames, selections, config, actions, eligible_function):
    rows = []
    for k,frame in sorted(frames.items()):
        eligible = eligible_function(frame)
        groups = treatment_groups(frame,eligible,actions,config['sampling_interval_seconds'])
        selected = np.asarray(selections[k],int)
        for action,events in groups.items():
            rows.append({'cultivation':k,'treatment':action,'eligible_event_starts':len(events),
                         'selected_event_starts':int(np.isin(selected,events).sum()),
                         'event_definition':'positive logged dose' if action in actions[:3] else 'setting changed',
                         'availability':'available' if len(events) else 'no eligible logged event; not invented'})
    return pd.DataFrame(rows)
