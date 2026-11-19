"""Selected-checkpoint diagnostics; never used in the optimizer or selection score."""

import numpy as np
import pandas as pd


def validity_rows(predicted, targets, cohort, step, interval):
    rows = []
    for j, name in enumerate(targets):
        values = predicted[:, j]
        finite = np.isfinite(values)
        nonnegative = name not in ['1_ph', '1_temp_air(C)', '2_temp(C)']
        negative = finite & (values < 0) if nonnegative else np.zeros(len(values), bool)
        ph_bad = finite & ((values < 0) | (values > 14)) if name == '1_ph' else np.zeros(len(values), bool)
        rows.append({'cohort': cohort, 'target': name, 'rollout_step': step,
                     'horizon_seconds': step * interval, 'predictions': len(values),
                     'nonfinite': int((~finite).sum()), 'negative': int(negative.sum()),
                     'ph_outside_0_14': int(ph_bad.sum()),
                     'minimum_prediction': float(values[finite].min()) if finite.any() else np.nan,
                     'maximum_negative_magnitude': float(-values[negative].min()) if negative.any() else 0.,
                     'mean_negative_magnitude': float(-values[negative].mean()) if negative.any() else 0.})
    return rows


def event_response_diagnostics(wm, fitted, frames, config):
    options = config.get('response_diagnostics', {})
    if not options.get('enabled', False):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    interval = wm.time_settings(config)['sampling_interval_seconds']
    history_steps = wm.time_settings(config)['history_steps']
    horizons = options.get('horizons_minutes', [1, 5, 15])
    steps = sorted(set(wm.duration_steps(h, interval) for h in horizons))
    if not steps or min(steps) < 1:
        raise ValueError('Response horizons must be positive.')
    per_action = options.get('events_per_action', 2)
    background = options.get('no_event_starts', 4)
    fraction = options.get('response_fraction', .1)
    tolerance = options.get('direction_tolerance', 1e-8)
    if per_action < 1 or background < 1 or not 0 < fraction <= 1 or tolerance < 0:
        raise ValueError('Invalid event-response diagnostic limits/thresholds.')
    targets = wm.target_order(fitted['preprocessing'])
    features = wm.feature_order(fitted['preprocessing'])
    maximum = max(steps)
    rng = np.random.default_rng(config.get('sampling_seed', config['seed']))
    origins, coverage, stores = [], [], {}
    for k, frame in sorted(frames.items()):
        valid = wm.observed_rows(frame, interval, features) & np.isfinite(frame[targets].to_numpy()).all(axis=1)
        lengths = wm.continuous_run_lengths(valid, frame.timestamp, interval)
        eligible = np.flatnonzero(lengths >= history_steps + maximum) - maximum + 1
        groups = wm.wp.treatment_groups(frame, eligible, wm.ACTIONS, interval)
        changes = np.zeros(len(frame), bool)
        for action in wm.ACTIONS[:3]:
            changes |= frame[action].to_numpy() > 0
        for action in wm.ACTIONS[3:]:
            values = frame[action].to_numpy()
            changes[1:] |= values[1:] != values[:-1]
        cumulative = np.r_[0, changes.cumsum()]
        no_event = eligible[(cumulative[eligible + maximum] - cumulative[eligible]) == 0]
        groups = {**groups, 'no_event': no_event}
        for group, candidates in groups.items():
            count = background if group == 'no_event' else per_action
            chosen = np.sort(rng.choice(candidates, min(len(candidates), count), replace=False))
            origins.extend((k, int(i), group) for i in chosen)
            coverage.append({'cultivation': k, 'group': group, 'eligible_origins': len(candidates),
                             'selected_origins': len(chosen),
                             'status': 'available' if len(candidates) else 'unavailable; no complete eligible segment',
                             'definition': ('no positive dose or aeration/lamp change during forecast'
                                            if group == 'no_event' else 'logged dose/change at first predicted reading')})
        stores[k] = {'features': frame[features].to_numpy(np.float32),
                     'targets': frame[targets].to_numpy(np.float32),
                     'clock': wm.clock_features(frame.timestamp, frame.timestamp.iloc[0]),
                     'timestamp': frame.timestamp}
    if not origins:
        return pd.DataFrame(), pd.DataFrame(coverage), pd.DataFrame()
    histories = np.stack([stores[k]['features'][i-history_steps:i] for k, i, _ in origins])
    initial = np.stack([stores[k]['targets'][i-1] for k, i, _ in origins])
    observed = np.stack([stores[k]['targets'][i:i+maximum] for k, i, _ in origins])
    forecasts, validity = [], []
    for step in wm.progress(range(maximum), enabled=config.get('progress_bars', False),
                            desc='Selected-checkpoint event response', unit='step'):
        extras = np.stack([np.r_[stores[k]['features'][i+step, -len(wm.ACTIONS):],
                                  stores[k]['clock'][i+step]] for k, i, _ in origins])
        latest = np.full((len(origins), len(targets)), np.nan, np.float32)
        for j, name in enumerate(targets):
            if name in features:
                latest[:, j] = histories[:, -1, features.index(name)]
        raw = {'sequence': histories, 'extra': extras, 'latest': latest,
               'truth': observed[:, step],
               'sampling_interval_seconds': interval,
               'observation_operator': wm.time_settings(config)['observation_operator']}
        predicted, _ = wm.predict(fitted, wm.transform(raw, fitted['preprocessing']))
        forecasts.append(predicted)
        for group in sorted(set(group for _, _, group in origins)):
            mask = np.array([g == group for _, _, g in origins])
            validity.extend(validity_rows(predicted[mask], targets, group, step+1, interval))
        next_row = histories[:, -1].copy()
        for j, name in enumerate(targets):
            if name in features:
                next_row[:, features.index(name)] = predicted[:, j]
        next_row[:, -len(wm.ACTIONS):] = extras[:, :len(wm.ACTIONS)]
        histories = np.concatenate([histories[:, 1:], next_row[:, None]], axis=1)
    forecasts = np.stack(forecasts, axis=1)
    rows = []
    for n, (k, origin, group) in enumerate(origins):
        for h in steps:
            for j, name in enumerate(targets):
                truth, prediction = observed[n, :h, j], forecasts[n, :h, j]
                actual_delta, forecast_delta = truth - initial[n, j], prediction - initial[n, j]
                amplitude = float(np.max(np.abs(actual_delta)))
                threshold = fraction * amplitude
                actual_cross = np.flatnonzero(np.abs(actual_delta) >= threshold) if amplitude > tolerance else []
                predicted_cross = (np.flatnonzero(np.abs(forecast_delta) >= threshold)
                                   if amplitude > tolerance else [])
                finite = np.isfinite(prediction).all()
                rows.append({'cultivation': k, 'origin_index': origin,
                             'timestamp': stores[k]['timestamp'].iloc[origin].isoformat(),
                             'group': group, 'target': name, 'horizon_minutes': h*interval/60,
                             'initial_observed_value': float(initial[n, j]),
                             'actual_change': float(actual_delta[-1]),
                             'predicted_change': float(forecast_delta[-1]),
                             'direction_correct': bool(np.sign(actual_delta[-1]) == np.sign(forecast_delta[-1]))
                                                  if finite and abs(actual_delta[-1]) > tolerance else None,
                             'actual_response_amplitude': amplitude,
                             'predicted_response_amplitude': float(np.max(np.abs(forecast_delta))) if finite else np.nan,
                             'response_threshold': threshold,
                             'actual_response_delay_seconds': float((actual_cross[0]+1)*interval) if len(actual_cross) else np.nan,
                             'predicted_response_delay_seconds': float((predicted_cross[0]+1)*interval) if len(predicted_cross) else np.nan,
                             'endpoint_absolute_error': float(abs(prediction[-1]-truth[-1])),
                             'trajectory_RMSE': float(np.sqrt(np.mean((prediction-truth)**2))),
                             'persistence_trajectory_RMSE': float(np.sqrt(np.mean(actual_delta**2))),
                             'prediction_status': 'valid' if finite else 'invalid_nonfinite',
                             'interpretation': 'conditional logged-action association; not a causal treatment effect'})
    return pd.DataFrame(rows), pd.DataFrame(coverage), pd.DataFrame(validity)


def action_response_gate(response, coverage, config):
    """Decide whether a fitted model is usable as an RL dynamics model.

    Level accuracy is not the gate for offline RL: a policy explores actions, so the
    model must show a response to them that is distinguishable from doing nothing.
    Direction agreement near 0.5 is a coin flip, meaning the model has not learned the
    action effect and a policy would exploit model error rather than learn control.

    The comparison is against the `no_event` cohort in the same diagnostic, so the
    threshold is measured rather than arbitrary.
    """
    options = config.get('action_response_gate', {})
    if not options.get('enabled', True):
        return {'rl_eligible': None, 'gate_status': 'disabled', 'per_action': []}
    if response is None or not len(response):
        return {'rl_eligible': False, 'gate_status': 'no_response_diagnostics',
                'reason': 'response diagnostics produced no rows',
                'min_direction_agreement': options.get('min_direction_agreement', .60),
                'min_events_per_action': options.get('min_events_per_action', 10),
                'per_action': []}
    minimum_agreement = float(options.get('min_direction_agreement', .60))
    minimum_events = int(options.get('min_events_per_action', 10))
    horizon = options.get('horizon_minutes')
    if horizon is not None:
        response = response[response.horizon_minutes == float(horizon)]
    available = set(coverage.group) if coverage is not None and len(coverage) else set()
    per_action, reasons = [], []
    for group, rows in response.groupby('group'):
        decided = rows[rows.direction_correct.notna()]
        agreement = float(decided.direction_correct.mean()) if len(decided) else float('nan')
        if group == 'no_event':
            continue
        if group not in available:
            per_action.append({'action': group, 'status': 'unavailable',
                               'events': 0, 'direction_agreement': None})
            reasons.append(f'{group}: no eligible event in the validation cohort')
            continue
        status = 'available'
        if len(decided) < minimum_events:
            status = 'insufficient_events'
            reasons.append(f'{group}: {len(decided)} decided events < {minimum_events}')
        elif not np.isfinite(agreement) or agreement < minimum_agreement:
            status = 'weak_response'
            reasons.append(f'{group}: direction agreement {agreement:.2f} < {minimum_agreement:.2f}')
        per_action.append({'action': group, 'status': status,
                           'events': int(len(decided)),
                           'direction_agreement': None if not np.isfinite(agreement) else round(agreement, 4)})
    control = response[(response.group == 'no_event') & response.direction_correct.notna()]
    control_agreement = float(control.direction_correct.mean()) if len(control) else float('nan')
    usable = [r for r in per_action if r['status'] == 'available']
    eligible = bool(usable)
    return {'rl_eligible': eligible,
            'gate_status': 'pass' if eligible else 'fail',
            'reason': '; '.join(reasons) if reasons else 'at least one action shows a graded response',
            'min_direction_agreement': minimum_agreement,
            'min_events_per_action': minimum_events,
            'control_direction_agreement': (None if not np.isfinite(control_agreement)
                                            else round(control_agreement, 4)),
            'usable_actions': [r['action'] for r in usable],
            'per_action': per_action}


def action_coverage_limitations(coverage):
    """Record per-action gradeable-event counts so dataset limits travel with the artifact."""
    if coverage is None or not len(coverage):
        return []
    rows = []
    for _, row in coverage.iterrows():
        eligible = row.get('eligible_origins')
        rows.append({'action': row.get('group'),
                     'eligible_origins': None if eligible is None else int(eligible),
                     'selected_origins': None if row.get('selected_origins') is None else int(row['selected_origins']),
                     'status': row.get('status'),
                     'gradeable': bool(eligible) if eligible is not None else False})
    return rows
