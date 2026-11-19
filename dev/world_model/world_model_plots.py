"""Figures for saved world-model evaluation results; no model fitting."""

from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from IPython.display import Image, display

ORDER = ['reference_baseline', 'gru', 'lstm', 'tcn', 'pg_ac_tcn', 'pets_mlp', 'dlinear', 'nhits',
         'tft', 'dnlssm', 'ridge', 'random_forest', 'xgboost', 'hrp_xgb', 'lightgbm', 'gpr', 'grey_box']
NAMES = {'reference_baseline': 'Reference baseline', 'gru': 'GRU', 'lstm': 'LSTM', 'tcn': 'Causal TCN',
         'pg_ac_tcn': 'Physics-guided action-conditioned TCN', 'pets_mlp': 'PETS-style MLP ensemble',
         'dlinear': 'DLinear', 'nhits': 'N-HiTS', 'tft': 'Temporal Fusion Transformer',
         'dnlssm': 'Deep nonlinear state-space model', 'ridge': 'Ridge ARX',
         'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'hrp_xgb': 'HRP-XGBoost ensemble',
         'lightgbm': 'LightGBM', 'gpr': 'Gaussian Process', 'grey_box': 'Grey-box (2 targets)'}
COLORS = dict(zip(ORDER, ['#666666', '#0072B2', '#D55E00', '#009E73', '#56B4E9', '#CC79A7',
                          '#9467BD', '#8C564B', '#E69F00', '#117733', '#AA4499', '#332288',
                          '#44AA99', '#999933', '#DDCC77', '#88CCEE', '#222222']))
TARGET_LABELS = {'1_tds(ppm)': 'TDS', '1_do(mg/L)': 'DO', '1_ph': 'pH', '1_cod(mg/L)': 'COD',
                 '1_toc(mg/L)': 'TOC', '1_turbidity(NTU)': 'Turbidity', '1_nh3(mg/L)': 'Water NH3',
                 '1_temp_air(C)': 'Water temperature', '2_light(Lux)': 'Measured light'}
TARGET_LABELS.update({'biomassa_estimasi(g/L)': 'Biomass estimate', '1_bod(mg/L)': 'BOD proxy',
                      '1_tss(mg/L)': 'TSS'})
UNIT_LABELS = {'1_ph': 'pH', '1_do(mg/L)': 'DO (mg/L)', '1_cod(mg/L)': 'COD (mg/L)',
               '1_temp_air(C)': 'Water temperature (°C)'}
UNIT_LABELS.update({'biomassa_estimasi(g/L)': 'Biomass estimate (g/L)',
                    '1_bod(mg/L)': 'BOD proxy (mg/L)', '1_tss(mg/L)': 'TSS (mg/L)'})


def run_label(metadata, split):
    ids = metadata['config']['split']['train']
    cohort = 'Recent-only' if ids == [11, 12, 13] else 'Pooled'
    track = metadata['config'].get('training_track', 'legacy_capped_pilot')
    train = ', '.join(f'K{i:02d}' for i in ids)
    evaluation = ', '.join(f'K{i:02d}' for i in metadata['config']['split'][split])
    mode = metadata.get('label_mode', 'sensor proxies')
    return f'{cohort} / {track} | {mode} | {metadata.get("sampling_interval_seconds", 60)}s step | {metadata.get("observation_operator", "interval_mean")} | train: {train}\n{split}: {evaluation} | run {metadata["run_id"]}'


def emit(fig, filename, output_dir=None, dpi=150, show=True, bottom_margin=0):
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    content = buffer.getvalue()
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (filename + '.png')).write_bytes(content)
    if show:
        display(Image(data=content))
    plt.close(fig)


def ordered(names):
    return [name for name in ORDER if name in names]


def overall_error(summary, label, output_dir, options):
    data = summary.loc[summary.category.ne('physical') & np.isfinite(summary.macro_NRMSE)].sort_values('macro_NRMSE')
    fig, axis = plt.subplots(figsize=(12, 7))
    axis.barh([NAMES.get(name, name) for name in data.algorithm], data.macro_NRMSE,
              color=[COLORS[name] for name in data.algorithm])
    axis.invert_yaxis()
    baseline = data.loc[data.algorithm.eq('reference_baseline'), 'macro_NRMSE'].iloc[0]
    axis.axvline(baseline, color=COLORS['reference_baseline'], linestyle='--', label=NAMES['reference_baseline'])
    for y, value in enumerate(data.macro_NRMSE):
        axis.text(value, y, f'  {value:.5f}', va='center', fontsize=10)
    axis.set_xlim(0, max(float(data.macro_NRMSE.max()) * 1.2, 1e-8))
    axis.set_xlabel('One-step macro NRMSE over saved common targets — lower is better')
    count = int(data.category.ne('reference').sum())
    axis.set_title(f'Prediction error: {count} evaluated learned families and reference baseline\n' + label, fontsize=11)
    axis.legend(loc='lower right')
    axis.grid(axis='x', alpha=.2)
    emit(fig, '01_overall_error', output_dir, **options)


def target_error_heatmap(points, targets, label, output_dir, options):
    names = ordered(points.loc[points.algorithm.ne('grey_box'), 'algorithm'].unique())
    table = points.loc[points.algorithm.ne('grey_box')].pivot(index='algorithm', columns='target', values='NRMSE').reindex(index=names, columns=targets)
    values = table.to_numpy(float)
    finite = values[np.isfinite(values)]
    fig, axis = plt.subplots(figsize=(14, 7))
    maximum = max(float(finite.max()), 1e-8)
    image = axis.imshow(np.ma.masked_invalid(values), cmap='YlOrRd', aspect='auto',
                        norm=PowerNorm(gamma=.5, vmin=0, vmax=maximum))
    axis.set_xticks(range(len(targets)), [TARGET_LABELS.get(t, t) for t in targets], rotation=30, ha='right')
    axis.set_yticks(range(len(names)), [NAMES[n] for n in names])
    for y in range(len(names)):
        for x in range(len(targets)):
            v = values[y, x]
            axis.text(x, y, f'{v:.3f}' if np.isfinite(v) else 'N/A', ha='center', va='center',
                      color='white' if np.isfinite(v) and v > maximum * .3 else 'black', fontsize=9)
    fig.colorbar(image, ax=axis, label='NRMSE — lower is better; square-root color scale')
    axis.set_title('Which targets are difficult for each model?\n' + label, fontsize=11)
    emit(fig, '02_target_error_heatmap', output_dir, **options)


def recursive_error_curves(recursive, label, output_dir, options):
    curve = recursive.groupby(['algorithm', 'horizon_minutes']).NRMSE.mean().unstack(0)
    baseline_column = 'baseline_NRMSE' if 'baseline_NRMSE' in recursive else 'persistence_NRMSE'
    baselines = recursive.groupby(['algorithm', 'horizon_minutes'])[baseline_column].mean().unstack(0)
    if not np.allclose(baselines.to_numpy(), baselines.iloc[:, :1].to_numpy(), rtol=1e-6, atol=1e-10, equal_nan=True):
        raise ValueError('Recursive baselines differ: the models may not share evaluation starts.')
    curve['reference_baseline'] = baselines.iloc[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    handles = []
    for name in ordered(curve.columns):
        for axis in axes:
            line, = axis.plot(curve.index, curve[name], marker='o', linewidth=2,
                              linestyle='--' if name == 'reference_baseline' else '-',
                              color=COLORS[name], label=NAMES[name])
        handles.append(line)
    axes[0].set_title('Linear error scale')
    if (curve.to_numpy() > 0).all():
        axes[1].set_yscale('log')
        axes[1].set_title('Log error scale — easier to compare smaller errors')
    else:
        axes[1].set_title('Linear scale; a zero error prevents logarithmic scaling')
    for axis in axes:
        axis.set_xticks(curve.index)
        axis.set_xlabel('Minutes predicted recursively without new process observations')
        axis.set_ylabel('Macro NRMSE over saved targets — lower is better')
        axis.grid(alpha=.25, which='both')
    fig.suptitle('How prediction error grows with rollout duration\n' + label, fontsize=11)
    fig.legend(handles, [h.get_label() for h in handles], loc='lower center', bbox_to_anchor=(.5, .01), ncol=3)
    emit(fig, '03_recursive_error_curves', output_dir, bottom_margin=.16, **options)


def matched_physical_comparison(points, label, output_dir, options):
    subset = points.loc[points.target.isin(['1_do(mg/L)', '1_temp_air(C)'])]
    scores = subset.groupby('algorithm').NRMSE.mean().sort_values()
    fig, axis = plt.subplots(figsize=(12, 7))
    axis.barh([NAMES[n] for n in scores.index], scores.values, color=[COLORS[n] for n in scores.index])
    axis.invert_yaxis()
    axis.axvline(scores['reference_baseline'], color=COLORS['reference_baseline'], linestyle='--')
    for y, value in enumerate(scores.values):
        axis.text(value, y, f'  {value:.5f}', va='center')
    axis.set_xlim(0, max(scores) * 1.2)
    axis.set_xlabel('One-step macro NRMSE over DO and water temperature only')
    axis.set_title('Fair comparison with the grey-box baseline: two matched targets\n' + label, fontsize=11)
    axis.grid(axis='x', alpha=.2)
    emit(fig, '04_matched_physical_comparison', output_dir, **options)


def neural_learning_curves(metadata_records, label, output_dir, options):
    has_treatment = any(any('validation_treatment_macro_NRMSE' in row for row in m.get('training_curve', [])) for m in metadata_records)
    fig, axes = plt.subplots(1, 3 if has_treatment else 2, figsize=(21 if has_treatment else 16, 7))
    handles = []
    for metadata in metadata_records:
        if not metadata.get('training_curve'):
            continue
        data = pd.DataFrame(metadata['training_curve'])
        for member, curve in data.groupby('member'):
            name = metadata['algorithm']
            name_label = NAMES[name] + (f' — member {member + 1}' if name == 'pets_mlp' else '')
            style = ['-', '--', ':'][int(member) % 3]
            line, = axes[0].plot(curve.epoch, curve.training_gaussian_NLL_scaled_delta,
                                 color=COLORS[name], linestyle=style, marker='.', label=name_label)
            axes[1].plot(curve.epoch, curve.validation_macro_NRMSE,
                         color=COLORS[name], linestyle=style, marker='.')
            best = curve.loc[curve.validation_macro_NRMSE.idxmin()]
            axes[1].scatter(best.epoch, best.validation_macro_NRMSE, marker='*', s=100, color=COLORS[name])
            if has_treatment and 'validation_treatment_macro_NRMSE' in curve:
                axes[2].plot(curve.epoch, curve.validation_treatment_macro_NRMSE, color=COLORS[name], linestyle=style, marker='.')
            handles.append(line)
    if not handles:
        plt.close(fig)
        return
    axes[0].set_ylabel('Training Gaussian NLL of standardized targets — lower is better')
    axes[1].set_ylabel('Validation macro NRMSE — lower is better')
    axes[0].set_title('Gaussian NLL component during neural training')
    axes[1].set_title('Actual validation curves; stars mark selected member checkpoints')
    if has_treatment:
        axes[2].set_ylabel('Treatment-audit macro NRMSE')
        axes[2].set_title('Separate treatment audit; not used for checkpoint selection')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
    fig.suptitle('Neural learning curves (MLP members shown individually)\n' + label, fontsize=11)
    fig.legend(handles, [h.get_label() for h in handles], loc='lower center', bbox_to_anchor=(.5, .01), ncol=3)
    emit(fig, '05_neural_learning_curves', output_dir, bottom_margin=.16, **options)


def resource_comparison(summary, label, output_dir, options):
    names = ordered(summary.loc[summary.category.ne('reference'), 'algorithm'].unique())
    table = summary.set_index('algorithm').reindex(names)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, column, xlabel in zip(axes, ['fit_seconds', 'median_inference_ms'],
                                    ['Fit / checkpoint-selection time (seconds)', 'Warmed batch-one inference (milliseconds)']):
        axis.barh([NAMES[n] for n in names], table[column], color=[COLORS[n] for n in names])
        axis.invert_yaxis()
        if table[column].gt(0).all():
            axis.set_xscale('log')
        axis.set_xlabel(xlabel + ' — log scale when all values are positive')
        axis.grid(axis='x', alpha=.2, which='both')
    fig.suptitle('Compute comparison (grey-box predicts only two targets)\n' + label, fontsize=11)
    fig.text(.5, .015, 'Neural fit times include validation/checkpoint selection; inference excludes preprocessing.', ha='center', fontsize=10)
    emit(fig, '06_resource_comparison', output_dir, bottom_margin=.07, **options)


def uncertainty_coverage(uncertainty, targets, label, output_dir, options):
    if uncertainty is None or uncertainty.empty:
        return
    names = ordered(uncertainty.algorithm.unique())
    table = uncertainty.pivot(index='algorithm', columns='target', values='coverage_90pct').reindex(index=names, columns=targets)
    fig, axis = plt.subplots(figsize=(14, 5))
    image = axis.imshow(table, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    axis.set_xticks(range(len(targets)), [TARGET_LABELS[t] for t in targets], rotation=30, ha='right')
    axis.set_yticks(range(len(names)), [NAMES[n] for n in names])
    for y, row in enumerate(table.to_numpy()):
        for x, value in enumerate(row):
            axis.text(x, y, f'{100 * value:.0f}%', ha='center', va='center',
                      color='black' if value > .6 else 'white', fontsize=10)
    fig.colorbar(image, ax=axis, label='Observed fraction inside the nominal 90% interval')
    axis.set_title('Prediction-interval coverage: compare each cell with 90%\n' + label, fontsize=11)
    fig.text(.5, .015, 'Point-only models have no native intervals. Coverage alone is insufficient; inspect interval widths in the report.', ha='center', fontsize=10)
    emit(fig, '07_uncertainty_coverage', output_dir, bottom_margin=.09, **options)


def actual_prediction_traces(raw, predictions, targets, max_minutes, label, output_dir, options):
    anchors = pd.DataFrame(raw['anchors'])
    anchors['time'] = pd.to_datetime(anchors.timestamp)
    interval = raw.get('sampling_interval_seconds', 60)
    operator = raw.get('observation_operator', 'interval_mean')
    groups = (anchors.cultivation.ne(anchors.cultivation.shift()) | anchors['time'].diff().ne(pd.Timedelta(seconds=interval))).cumsum()
    longest = groups.value_counts().sort_index().idxmax()
    chosen = np.flatnonzero(groups.eq(longest).to_numpy())[:max(1, int(max_minutes * 60 / interval))]
    times = anchors['time'].iloc[chosen] + pd.Timedelta(seconds=interval if operator == 'interval_mean' else 0)
    fig, axes = plt.subplots(math_ceil(len(targets) / 2), 2, figsize=(16, 10), squeeze=False)
    handles = {}
    for axis, target in zip(axes.flat, targets):
        target_index = raw['target_order'].index(target)
        for name in ordered(predictions):
            entry = predictions[name]
            if target not in entry['targets']:
                continue
            line, = axis.plot(times, entry['values'][chosen, entry['targets'].index(target)],
                              color=COLORS[name], linestyle='--' if name in ['reference_baseline', 'grey_box'] else '-',
                              linewidth=1.2, alpha=.85, label=NAMES[name])
            handles[name] = line
        line, = axis.plot(times, raw['truth'][chosen, target_index], color='black', marker='o',
                          markersize=3, linewidth=2, label='Actual recorded / proxy value')
        handles['actual'] = line
        axis.set_title(TARGET_LABELS.get(target, target))
        axis.set_ylabel(UNIT_LABELS.get(target, target))
        axis.grid(alpha=.2)
        axis.tick_params(axis='x', rotation=25)
    for axis in list(axes.flat)[len(targets):]:
        axis.set_visible(False)
    ids = anchors.cultivation.iloc[chosen].unique()
    date = times.iloc[0].strftime('%Y-%m-%d')
    fig.suptitle(f'Actual vs predicted {interval}-second values — K{ids[0]:02d}, {date}, {len(chosen)} consecutive evaluated observations\n' + label, fontsize=11)
    fig.legend(list(handles.values()), [h.get_label() for h in handles.values()], loc='lower center', bbox_to_anchor=(.5, .04), ncol=4)
    fig.text(.5, .015, 'X-axis: observation availability (sample timestamp or aggregate-bin end). Clip chosen by continuity/length, not model performance; no lines cross data gaps.', ha='center', fontsize=10)
    emit(fig, '08_actual_vs_predictions', output_dir, bottom_margin=.16, **options)


def math_ceil(value):
    return int(np.ceil(value))


def priority_comparison(points, recursive, targets, label, output_dir, options):
    selected = points.loc[points.target.isin(targets) & points.algorithm.ne('grey_box')]
    scores = selected.groupby('algorithm').NRMSE.mean().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh([NAMES.get(n,n) for n in scores.index], scores.values,
                 color=[COLORS[n] for n in scores.index])
    axes[0].invert_yaxis(); axes[0].set_xlabel('Priority-group macro NRMSE')
    axes[0].set_title('One-step error: pH, COD, BOD, DO and water ammonia')
    if not recursive.empty:
        priority = recursive.loc[recursive.target.isin(targets)]
        curves = priority.groupby(['algorithm', 'horizon_minutes']).NRMSE.mean().unstack(0)
        baseline = priority.groupby('horizon_minutes').baseline_NRMSE.mean()
        curves['reference_baseline'] = baseline
        for name in ordered(curves.columns):
            axes[1].plot(curves.index, curves[name], marker='o', label=NAMES[name], color=COLORS[name])
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel('Forecast duration (minutes)'); axes[1].set_ylabel('Priority-group macro NRMSE')
    axes[1].set_title('Recursive priority-channel prediction')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Priority channels are reported separately; no extra loss weights are assumed\n' + label, fontsize=10)
    emit(fig, '09_priority_channels', output_dir, **options)
