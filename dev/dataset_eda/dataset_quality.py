"""Gap-aware descriptive sensor diagnostics; never cleans or rewrites source data."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {
    '1_tds(ppm)': 'TDS', '1_do(mg/L)': 'DO', '1_ph': 'pH', '1_cod(mg/L)': 'COD',
    '1_toc(mg/L)': 'TOC', '1_turbidity(NTU)': 'Turbidity', '1_nh3(mg/L)': 'Water NH3',
    '1_temp_air(C)': 'Water temperature', '2_light(Lux)': 'Measured light',
    'biomassa_estimasi(g/L)': 'Biomass estimate', '1_bod(mg/L)': 'BOD proxy', '1_tss(mg/L)': 'TSS',
}


def provenance(column, cultivation):
    if column == 'biomassa_estimasi(g/L)':
        return 'Interpolated OD-derived estimate; synthetic fluctuations; not independent sensor noise'
    if column == '1_bod(mg/L)':
        return 'Firmware-derived/provisional BOD; not independent laboratory BOD'
    if column == '1_tss(mg/L)' and cultivation <= 9:
        return 'Older estimated TSS mapping; not independent measured TSS'
    return 'Sensor CSV proxy; calibration, resolution and measurement independence unverified'


def robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(np.abs(values - np.median(values))) / .6744897501960817) if len(values) else np.nan


def box_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    p05, q1, median, q3, p95 = np.quantile(values, [.05, .25, .5, .75, .95])
    return {'whislo': float(p05), 'q1': float(q1), 'med': float(median),
            'q3': float(q3), 'whishi': float(p95), 'fliers': []}


def nanosecond_ticks(times):
    return pd.DatetimeIndex(times).to_numpy(dtype='datetime64[ns]').view(np.int64)


def segments_for(values, clock_good, continuous):
    valid = np.isfinite(values) & clock_good
    connected = continuous & valid & np.r_[False, valid[:-1]]
    return valid, connected, np.cumsum(~connected)


def local_median(values, valid, segments, window):
    series = pd.Series(np.where(valid, values, np.nan))
    return series.groupby(segments, sort=False).transform(
        lambda part: part.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    ).to_numpy()


def longest_flat_minutes(values, valid, connected, interval):
    if not valid.any():
        return np.nan
    same = connected & (values == np.r_[np.nan, values[:-1]])
    groups = np.cumsum(~same)
    counts = pd.Series(groups[valid]).value_counts()
    return float(max(int(counts.max()) - 1, 0) * interval / 60)


def spectrum(values, valid, segments, interval, config):
    try:
        from scipy.signal import welch
    except ImportError:
        return {'status': 'unavailable: scipy missing'}
    runs = pd.DataFrame({'segment': segments[valid]}).value_counts(sort=True)
    if runs.empty or runs.iloc[0] < config['spectrum_min_samples']:
        return {'status': 'unavailable: no sufficiently long contiguous finite segment'}
    longest = runs.index[0][0]
    indices = np.flatnonzero(valid & (segments == longest))[:config['spectrum_max_samples']]
    if np.ptp(values[indices]) == 0:
        return {'status': 'unavailable: selected contiguous segment is constant', 'samples': len(indices)}
    frequencies, power = welch(values[indices], fs=1 / interval, detrend='linear',
                               nperseg=min(config['spectrum_nperseg'], len(indices)))
    total = float(power.sum())
    cutoff = 1 / config['fast_period_seconds']
    fraction = (float(power[frequencies >= cutoff].sum() / total * 100)
                if total > 0 and frequencies.max() >= cutoff else np.nan)
    return {'status': 'assessed on one longest contiguous segment, capped in length',
            'frequency_hz': frequencies, 'power': power, 'samples': len(indices),
            'segment_start_row': int(indices[0]), 'segment_end_row': int(indices[-1]),
            'fast_power_pct': fraction}


def inspect_quality(frame, parsed, cultivation, interval, targets, actions, config, minute_path=None):
    """Use all rows for exact descriptive statistics, retaining only compact plot data."""
    times = pd.DatetimeIndex(parsed)
    n = len(frame)
    unique_clock = ~times.duplicated(keep=False)
    clock_good = np.asarray(times.notna() & unique_clock)
    quality_good = np.ones(n, dtype=bool)
    if 'status' in frame:
        quality_good &= frame.status.eq('OK').to_numpy()
    if interval == 60 and 'n_data' in frame:
        quality_good &= frame.n_data.eq(12).to_numpy()
    usable = clock_good & quality_good
    differences_s = pd.Series(times).diff().dt.total_seconds().to_numpy()
    continuous = np.r_[False, differences_s[1:] == interval]
    data = pd.DataFrame(index=frame.index)
    for name in targets:
        data[name] = pd.to_numeric(frame[name], errors='coerce') if name in frame else np.nan
    data = data.astype(float).where(np.isfinite(data))
    data.loc[~usable] = np.nan
    minute_stats, complete_minutes = None, None
    if interval == 5:
        selected = np.flatnonzero(clock_good)
        minute_data = data.iloc[selected].copy()
        minute_data.index = times[selected]
        minute_stats = minute_data.groupby(minute_data.index.floor('min')).agg(['count', 'mean', 'var'])
        ticks = pd.Series(times[selected], index=times[selected]).groupby(times[selected].floor('min')).agg(['count', 'min', 'max'])
        complete_minutes = ticks['count'].eq(12) & (ticks['max'] - ticks['min']).dt.total_seconds().eq(55)
        same_bin = pd.Series(times).dt.floor('min').eq(pd.Series(times).shift().dt.floor('min')).to_numpy()
        bad_internal = clock_good & np.r_[False, clock_good[:-1]] & same_bin & ~continuous
        bad_bins = pd.Series(bad_internal[selected], index=times[selected].floor('min')).groupby(level=0).sum()
        complete_minutes &= bad_bins.reindex(complete_minutes.index, fill_value=0).eq(0)
    provided = None
    if interval == 5 and minute_path is not None and Path(minute_path).is_file():
        header = pd.read_csv(minute_path, nrows=0).columns
        provided = pd.read_csv(minute_path, usecols=[c for c in ['timestamp', 'n_data', *targets] if c in header])
        provided['timestamp'] = pd.to_datetime(provided.timestamp, errors='coerce')
        keep = provided.timestamp.notna() & ~provided.timestamp.duplicated(keep=False)
        if 'n_data' in provided:
            keep &= provided.n_data.eq(12)
        provided = provided.loc[keep].set_index('timestamp')
    action_event = np.zeros(n, dtype=bool)
    for column in actions:
        if column not in frame:
            continue
        a = pd.to_numeric(frame[column], errors='coerce').to_numpy(float)
        if column.startswith('Pump'):
            action_event |= np.isfinite(a) & (a > 0)
        else:
            action_event |= continuous & np.isfinite(a) & np.isfinite(np.r_[np.nan, a[:-1]]) & (a != np.r_[np.nan, a[:-1]])
    near_action = np.zeros(n, dtype=bool)
    ticks_ns = nanosecond_ticks(times)
    event_times = ticks_ns[action_event & clock_good]
    if len(event_times):
        event_times = np.sort(event_times)
        right = np.searchsorted(event_times, ticks_ns)
        left_distance = np.abs(ticks_ns - event_times[np.maximum(right - 1, 0)])
        right_distance = np.abs(ticks_ns - event_times[np.minimum(right, len(event_times) - 1)])
        near_action = clock_good & (np.minimum(left_distance, right_distance) <= 60 * 10**9)
    else:
        near_action[:] = False
    window = max(3, int(round(config['median_window_seconds'] / interval)))
    window += window % 2 == 0
    detail_start = times[clock_good][0] if clock_good.any() else pd.NaT
    detail_reason = 'first available interval; no isolated priority-channel spike detected'
    first_priority_spike = None
    rows, distributions, detail_arrays, spectra, examples = [], {}, {}, {}, []
    for name in targets:
        raw = pd.to_numeric(frame[name], errors='coerce').to_numpy(float) if name in frame else np.full(n, np.nan)
        values = data[name].to_numpy()
        valid, paired, segments = segments_for(values, usable, continuous)
        finite = raw[np.isfinite(raw)]
        dx = values - np.r_[np.nan, values[:-1]]
        paired &= np.isfinite(dx)
        changes = dx[paired]
        step_sigma = robust_sigma(changes)
        median = local_median(values, valid, segments, window)
        residual = values - median
        residue_sigma = robust_sigma(residual)
        threshold_valid = np.isfinite(residue_sigma) and residue_sigma > 0
        candidates = valid & np.isfinite(residual) & (np.abs(residual) > config['mad_multiplier'] * residue_sigma) if threshold_valid else np.zeros(n, bool)
        right_pair = np.r_[paired[1:], False]
        next_dx = np.r_[dx[1:], np.nan]
        returns = np.abs(values - np.r_[np.nan, values[:-1]])
        neighbour_return = np.abs(np.r_[values[1:], np.nan] - np.r_[np.nan, values[:-1]])
        spike = (candidates & paired & right_pair & (dx * next_dx < 0) &
                 (neighbour_return <= config['spike_return_fraction'] * np.maximum(returns, np.abs(next_dx))))
        spike_indices = np.flatnonzero(spike)
        if name in config['priority_targets'] and len(spike_indices):
            t = times[spike_indices[0]]
            if first_priority_spike is None or t < first_priority_spike:
                first_priority_spike = t
        for index in spike_indices[:config['max_event_examples_per_target']]:
            examples.append({'cultivation': cultivation, 'target': name, 'timestamp': times[index],
                             'previous': values[index - 1], 'value': values[index], 'next': values[index + 1],
                             'local_median': median[index], 'near_logged_action_60s': bool(near_action[index])})
        level_box = box_summary(finite)
        level_std = float(np.std(values[valid])) if valid.any() else np.nan
        low, high = np.nanquantile(finite, [.25, .75]) if len(finite) else [np.nan, np.nan]
        edge = max(1, int(valid.sum() * .1))
        ordered = values[valid]
        shift = ((np.median(ordered[-edge:]) - np.median(ordered[:edge])) / (high - low)
                 if len(ordered) and high > low else np.nan)
        if len(changes) > 2 and np.std(values[paired]) > 0 and np.std(np.r_[np.nan, values[:-1]][paired]) > 0:
            lag1 = float(np.corrcoef(values[paired], np.r_[np.nan, values[:-1]][paired])[0, 1])
        else:
            lag1 = np.nan
        counts = minute_stats[(name, 'count')] if minute_stats is not None else None
        complete = (complete_minutes & counts.eq(12)) if counts is not None else None
        means = minute_stats.loc[complete, (name, 'mean')] if complete is not None else pd.Series(dtype=float)
        minute_variances = minute_stats.loc[complete, (name, 'var')].to_numpy() if complete is not None else np.array([])
        within = float(np.mean(minute_variances * 11 / 12)) if len(minute_variances) else np.nan
        between = float(means.var(ddof=0)) if len(means) else np.nan
        within_share = 100 * within / (within + between) if within + between > 0 else np.nan
        hist_sigma = step_sigma if step_sigma > 0 else None
        normalized = (changes - np.median(changes)) / hist_sigma if hist_sigma else changes
        edges = np.linspace(-12, 12, 81) if hist_sigma else (np.histogram_bin_edges(changes, bins=40) if len(changes) else np.linspace(-1, 1, 41))
        hist, edges = np.histogram(normalized, bins=edges)
        distributions[name] = {'levels': level_box, 'minute_variance': box_summary(minute_variances),
                               'change_histogram': hist, 'change_edges': edges,
                               'change_normalized': hist_sigma is not None, 'change_count': len(changes),
                               'change_outside_plot_count': len(changes) - int(hist.sum())}
        spec = spectrum(values, valid, segments, interval, config)
        spectra[name] = spec
        screen = np.zeros(n, bool)
        if name in config['nonnegative_targets']:
            screen = np.isfinite(raw) & (raw < 0)
        elif name == '1_ph':
            screen = np.isfinite(raw) & ((raw < config['ph_screen'][0]) | (raw > config['ph_screen'][1]))
        row = {'cultivation': cultivation, 'target': name, 'label': LABELS.get(name, name),
               'priority': name in config['priority_targets'], 'provenance': provenance(name, cultivation),
               'column_present': name in frame, 'rows': n, 'finite_raw': len(finite),
               'missing_pct': float(pd.isna(raw).mean() * 100),
               'infinite_values': int(np.isinf(raw).sum()), 'usable_rows': int(valid.sum()),
               'quality_excluded_rows': int((clock_good & ~quality_good).sum()),
               'mean': float(np.mean(finite)) if len(finite) else np.nan,
               'level_std': level_std, 'level_variance': level_std**2,
               'unique_values': int(len(np.unique(finite))),
               'zero_pct': float(np.mean(finite == 0) * 100) if len(finite) else np.nan,
               'min_pileup_pct': float(np.mean(finite == finite.min()) * 100) if len(finite) else np.nan,
               'max_pileup_pct': float(np.mean(finite == finite.max()) * 100) if len(finite) else np.nan,
               'range_screen_count': int(screen.sum()), 'range_screen_pct': float(screen.mean() * 100),
               'adjacent_pairs': int(paired.sum()), 'unchanged_adjacent_pct': float(np.mean(changes == 0) * 100) if len(changes) else np.nan,
               'longest_flat_minutes': longest_flat_minutes(values, valid, paired, interval),
               'lag1_correlation': lag1, 'step_robust_sigma': step_sigma,
               'noise_sigma_proxy': step_sigma / np.sqrt(2),
               'noise_proxy_to_level_std': step_sigma / np.sqrt(2) / level_std if level_std > 0 else np.nan,
               'residual_robust_sigma': residue_sigma, 'outlier_screen_assessed': threshold_valid,
               'local_outlier_candidates': int(candidates.sum()),
               'local_outlier_pct': float(candidates.sum() / valid.sum() * 100) if valid.any() else np.nan,
               'isolated_spike_candidates': int(spike.sum()),
               'isolated_spike_pct': float(spike.sum() / valid.sum() * 100) if valid.any() else np.nan,
               'spikes_near_logged_action_60s': int((spike & near_action).sum()),
               'first_last_median_shift_IQR': float(shift), 'complete_native_minutes': len(means),
               'within_minute_variance_mean': within, 'between_minute_variance': between,
               'within_minute_variance_share_pct': within_share,
               'spectral_fast_power_pct': spec.get('fast_power_pct', np.nan),
               'spectrum_samples': spec.get('samples', 0), 'spectrum_status': spec['status'],
               'provided_minute_pairs': 0, 'provided_mean_MAE': np.nan,
               'provided_mean_p99_abs_difference': np.nan, 'provided_mean_within_tolerance_pct': np.nan}
        if provided is not None and name in provided and len(means):
            joined = pd.concat([means.rename('native_mean'), pd.to_numeric(provided[name], errors='coerce').rename('provided_mean')], axis=1, join='inner').dropna()
            error = np.abs(joined.native_mean - joined.provided_mean)
            tolerance = config['rounding_tolerance'].get(name, config['default_rounding_tolerance'])
            row.update(provided_minute_pairs=len(joined), provided_mean_MAE=float(error.mean()),
                       provided_mean_p99_abs_difference=float(error.quantile(.99)),
                       provided_mean_within_tolerance_pct=float(error.le(tolerance).mean() * 100),
                       provided_mean_tolerance=tolerance)
        rows.append(row)
        detail_arrays[name] = (values, median, spike)
    if first_priority_spike is not None:
        detail_start = max(times[clock_good][0], first_priority_spike - pd.Timedelta(minutes=config['detail_minutes'] / 3))
        detail_reason = 'window around earliest isolated-spike candidate among priority channels; selected for diagnosis, not model score'
    detail_end = detail_start + pd.Timedelta(minutes=config['detail_minutes'])
    chosen = np.flatnonzero(clock_good & (times >= detail_start) & (times < detail_end))
    detail = {'timestamp': times[chosen], 'reason': detail_reason, 'columns': {}}
    if minute_stats is not None:
        detail_minutes = (minute_stats.index >= detail_start) & (minute_stats.index < detail_end)
        detail['minute_timestamp'] = minute_stats.index[detail_minutes]
    for name, (values, median, spike) in detail_arrays.items():
        y, m = values[chosen].copy(), median[chosen].copy()
        if len(chosen):
            gaps = np.r_[False, np.diff(ticks_ns[chosen]) != interval * 10**9]
            y[gaps], m[gaps] = np.nan, np.nan
        detail['columns'][name] = {'raw': y, 'median': m, 'spike': spike[chosen]}
        if minute_stats is not None:
            full = complete_minutes & minute_stats[(name, 'count')].eq(12)
            means = minute_stats[(name, 'mean')].where(full).loc[detail_minutes].to_numpy().copy()
            if len(means):
                means[np.r_[False, np.diff(nanosecond_ticks(detail['minute_timestamp'])) != 60 * 10**9]] = np.nan
            detail['columns'][name]['minute_mean'] = means
    return {'metrics': pd.DataFrame(rows), 'distributions': distributions, 'detail': detail,
            'spectra': spectra, 'examples': pd.DataFrame(examples), 'settings': config.copy()}


def channel_verdicts(metrics, config):
    result = metrics.copy()
    verdicts, reasons = [], []
    for row in result.to_dict('records'):
        flags = []
        if not row['column_present'] or not row['finite_raw']:
            verdicts.append('UNAVAILABLE'); reasons.append('Missing column or no finite observations'); continue
        if row['missing_pct'] > 0 or row['infinite_values']:
            flags.append('missing/nonfinite readings')
        if row['range_screen_count']:
            flags.append('values outside declared screening range')
        if row['unique_values'] <= 1:
            flags.append('constant channel; variation/response cannot be identified')
        elif row['longest_flat_minutes'] >= config['flatline_review_minutes']:
            flags.append('long identical-value run: stable process, quantization or sensor freeze require review')
        if row['isolated_spike_pct'] >= config['spike_review_pct']:
            flags.append('isolated-spike rate exceeds exploratory review threshold')
        if row['within_minute_variance_share_pct'] >= config['within_variance_review_pct']:
            flags.append('substantial within-minute variability: compare cadences on validation')
        if row['provided_minute_pairs'] and row['provided_mean_within_tolerance_pct'] < 99:
            flags.append('provided minute means differ from complete native-bin reconstruction')
        if not row['outlier_screen_assessed']:
            flags.append('MAD-based outlier screen unavailable: zero/undefined residual scale')
        verdicts.append('REVIEW' if flags else 'NO SCREEN TRIGGER')
        reasons.append('; '.join(flags) if flags else 'No configured statistical trigger; calibration and noise identity remain unverified')
    result['screen_verdict'], result['reason'] = verdicts, reasons
    return result


def _emit(fig, name, config):
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    fig.tight_layout(rect=(0, .06, 1, .95))
    if config.get('figure_dir'):
        folder = Path(config['figure_dir']); folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / (name + '.png'), dpi=config['figure_dpi'], bbox_inches='tight')
    if config.get('show_plots', True):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=config['figure_dpi'], bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    plt.close(fig)


def plot_distributions(records, targets, config, variance=False):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(17, 13))
    key = 'minute_variance' if variance else 'levels'
    for ax, name in zip(axes.flat, targets):
        boxes = []
        for record in records:
            box = record['quality']['distributions'][name][key]
            if box is not None:
                boxes.append({**box, 'label': f'K{record["cultivation"]:02d}'})
        if boxes:
            ax.bxp(boxes, showfliers=False, patch_artist=True,
                   boxprops={'facecolor': '#B7DCE9'}, medianprops={'color': '#222222'})
            if variance:
                positive = [b['whislo'] for b in boxes if b['whislo'] > 0]
                ax.set_yscale('symlog', linthresh=max(min(positive) * .1, 1e-12) if positive else 1e-6)
        else:
            ax.text(.5, .5, 'No complete native minutes' if variance else 'No finite values', ha='center', transform=ax.transAxes)
        unit = name.rsplit('(', 1)[-1][:-1] if '(' in name else 'pH units'
        ax.set_title(LABELS.get(name, name) + (' [priority]' if name in config['priority_targets'] else ''), fontsize=10)
        ax.set_ylabel(f'Within-minute sample variance ({unit})²' if variance else unit)
        ax.tick_params(axis='x', rotation=70, labelsize=7); ax.grid(axis='y', alpha=.2)
    for ax in list(axes.flat)[len(targets):]:
        ax.set_visible(False)
    fig.suptitle('Distribution of within-minute variance: 12 native readings per complete minute' if variance else 'Value distributions across cultivations: all 12 training targets', fontsize=14)
    fig.text(.5, .008, 'Boxes: 25th–75th percentiles; line: median; whiskers: 5th–95th. Full-data counts remain in tables. Variance is variability, not identified sensor noise.', ha='center', fontsize=9)
    _emit(fig, 'within_minute_variance' if variance else 'target_value_distributions', config)


def plot_quality_heatmaps(verdicts, targets, config):
    import matplotlib.pyplot as plt
    fields = [('within_minute_variance_share_pct', 'Within-minute share of variance (%)', 100),
              ('isolated_spike_pct', 'Isolated-spike candidates (%)', None),
              ('longest_flat_minutes', 'Longest identical run (minutes)', None),
              ('missing_pct', 'Missing target readings (%)', 100)]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, (field, title, maximum) in zip(axes.flat, fields):
        table = verdicts.pivot(index='cultivation', columns='target', values=field).reindex(columns=targets)
        values = table.to_numpy(float)
        im = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='YlOrRd', vmin=0, vmax=maximum)
        ax.set_xticks(range(len(targets)), [LABELS.get(t, t) for t in targets], rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(len(table)), [f'K{k:02d}' for k in table.index]); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Descriptive quality screens across all 12 targets; blank cells mean unavailable', fontsize=14)
    _emit(fig, 'quality_heatmaps', config)


def plot_native_diagnostics(record, targets, config):
    import matplotlib.pyplot as plt
    q = record['quality']; k = record['cultivation']
    fig, axes = plt.subplots(4, 3, figsize=(17, 12))
    for ax, name in zip(axes.flat, targets):
        d = q['detail']['columns'][name]; times = q['detail']['timestamp']
        ax.plot(times, d['raw'], color='#186A8C', linewidth=.8, label='Recorded values')
        ax.plot(times, d['median'], color='#C67414', linewidth=1.1, label='Centered local median (EDA only)')
        if 'minute_mean' in d:
            ax.plot(q['detail']['minute_timestamp'], d['minute_mean'], color='#3F823F', linewidth=1., label='Complete-minute mean')
        ax.scatter(times[d['spike']], d['raw'][d['spike']], color='#B2182B', s=18, zorder=3, label='Isolated-spike candidate')
        ax.set_title(LABELS.get(name, name) + (' [priority]' if name in config['priority_targets'] else ''), fontsize=10)
        ax.tick_params(axis='x', rotation=25, labelsize=7); ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=9)
    fig.suptitle(f'K{k:02d}: {record["interval"]}s readings and local baseline for all 12 targets', fontsize=14)
    fig.text(.5, .955, q['detail']['reason'], ha='center', fontsize=9)
    _emit(fig, f'K{k:02d}_native_detail', config)
    fig, axes = plt.subplots(4, 3, figsize=(17, 12))
    for ax, name in zip(axes.flat, targets):
        d = q['distributions'][name]; edges = d['change_edges']
        fraction = d['change_histogram'] / max(d['change_count'], 1)
        ax.stairs(fraction, edges, fill=True, color='#387E9B')
        ax.set_yscale('symlog', linthresh=.001)
        ax.set_title(LABELS.get(name, name), fontsize=10)
        ax.set_xlabel('Change / robust change scale' if d['change_normalized'] else 'Native-unit change (MAD=0 or unavailable)', fontsize=8)
        ax.set_ylabel('Fraction per bin', fontsize=8)
        outside = 100 * d['change_outside_plot_count'] / max(d['change_count'], 1)
        ax.text(.02, .96, f'Outside displayed range: {outside:.3g}%', transform=ax.transAxes, va='top', fontsize=8)
    fig.suptitle(f'K{k:02d}: distribution of adjacent {record["interval"]}s changes; gaps are excluded', fontsize=14)
    _emit(fig, f'K{k:02d}_change_distributions', config)
    fig, axes = plt.subplots(4, 3, figsize=(17, 12))
    for ax, name in zip(axes.flat, targets):
        d = q['spectra'][name]
        if 'frequency_hz' in d and np.any(d['power'] > 0):
            keep = (d['frequency_hz'] > 0) & (d['power'] > 0)
            ax.loglog(d['frequency_hz'][keep], d['power'][keep], linewidth=.9)
            ax.axvline(1 / config['fast_period_seconds'], linestyle='--', color='#9C432A', linewidth=.8)
            ax.text(.02, .04, f'{d["samples"]:,} contiguous samples', transform=ax.transAxes, fontsize=8)
        else:
            ax.text(.5, .5, 'No usable spectrum / zero variation', transform=ax.transAxes, ha='center', fontsize=8)
        ax.set_title(LABELS.get(name, name), fontsize=10); ax.set_xlabel('Frequency (Hz)', fontsize=8)
        ax.set_ylabel('PSD (native unit²/Hz)', fontsize=8); ax.grid(alpha=.15)
    fig.suptitle(f'K{k:02d}: Welch spectra on one contiguous segment; dashed line = 60s period', fontsize=14)
    fig.text(.5, .008, 'Fast-frequency power can include true process changes and cleaning/actuator events; it is not proof of sensor noise.', ha='center', fontsize=9)
    _emit(fig, f'K{k:02d}_spectra', config)
