"""World-model data preparation, architectures, training, artifacts, and evaluation."""

from pathlib import Path
import ast
from contextlib import nullcontext
import gc
import hashlib
import importlib
import json
import math
import time
import sys

import world_model_recovery as recovery

import numpy as np
import pandas as pd
import world_model_protocol as wp

OPTIONAL = {}
IMPORT_ERRORS = {}
for _name in ('torch', 'sklearn', 'xgboost', 'lightgbm', 'scipy', 'joblib'):
    try:
        OPTIONAL[_name] = importlib.import_module(_name)
    except (ImportError, OSError) as _error:
        OPTIONAL[_name] = None
        IMPORT_ERRORS[_name] = str(_error)

torch = OPTIONAL['torch']
if torch is not None:
    nn = torch.nn
    from torch.utils.checkpoint import checkpoint as activation_checkpoint

TARGETS = ['1_tds(ppm)', '1_do(mg/L)', '1_ph', '1_cod(mg/L)', '1_toc(mg/L)',
           '1_turbidity(NTU)', '1_nh3(mg/L)', '1_temp_air(C)', '2_light(Lux)']
CONTEXT = ['2_o2(ppm)', '2_temp(C)', '2_pressure(hPa)', '2_humidity(%)',
           '2_nh3_udara(ppm)', '2_co2(ppm)']
ACTIONS = ['Pump1_NaHCO3 (mL)', 'Pump2_Molase (mL)', 'Pump3_Microalgae (mL)',
           'Aerator_RU (L/s)', 'Lampu (PWM%)']
FEATURES = TARGETS + CONTEXT + ['1_tss(mg/L)'] + ACTIONS
CONTROL_TARGETS = TARGETS + ['biomassa_estimasi(g/L)', '1_bod(mg/L)', '1_tss(mg/L)']
REQUIRES = {**dict.fromkeys(['gru', 'lstm', 'tcn', 'pg_ac_tcn', 'pets_mlp', 'dlinear', 'nhits', 'tft', 'dnlssm'], ['torch']),
            **dict.fromkeys(['mopo_ensemble', 'moup_mlp'], ['torch']),
            **dict.fromkeys(['ridge', 'random_forest', 'gpr'], ['sklearn', 'joblib']),
            'xgboost': ['sklearn', 'xgboost', 'joblib'],
            'hrp_xgb': ['sklearn', 'xgboost', 'joblib'],
            'lightgbm': ['sklearn', 'lightgbm', 'joblib'], 'grey_box': ['scipy', 'joblib']}

# Offline-RL dynamics models. Each is a probabilistic network ensemble; they differ
# only in how ensemble disagreement is aggregated into a scalar model uncertainty.
# The reward penalty, rollout policy and policy constraint from the source papers act
# on a reward function and an actor-critic, which this world-model stage does not have.
ENSEMBLE_ALGORITHMS = {
    # MOPO (Yu et al., NeurIPS 2020) Sec. 4.3: u = max_i ||sigma_i||, most pessimistic member.
    'mopo_ensemble': {'uncertainty_mode': 'max_std', 'mc_dropout': 0.},
    # MOUP (Zhu et al., IEEE TAI 2024) Alg. 1 / Eq. 12-13: MC-dropout variance plus mean
    # predicted variance, which fuses epistemic and aleatoric uncertainty.
    'moup_mlp': {'uncertainty_mode': 'mc_dropout', 'mc_dropout': .1},
}


def resolved_uncertainty_mode(algorithm, members):
    """Which uncertainty aggregation an artifact exposes, if any.

    A declared mode wins. Otherwise any algorithm trained with more than one member has
    genuine epistemic disagreement, so it exposes ensemble dispersion. A single-member
    model cannot express disagreement and reports no model uncertainty.
    """
    declared = ENSEMBLE_ALGORITHMS.get(algorithm, {}).get('uncertainty_mode')
    if declared:
        return declared
    count = members or 1
    return 'ensemble_std' if count > 1 else None


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def availability():
    return pd.DataFrame([{'package': name, 'available': module is not None,
                          'version': getattr(module, '__version__', None),
                          'import_error': IMPORT_ERRORS.get(name)}
                         for name, module in OPTIONAL.items()])


def missing_dependencies(algorithm):
    return [name for name in REQUIRES[algorithm] if OPTIONAL[name] is None]


def target_order(settings):
    return list(settings.get('targets', TARGETS))


def feature_order(settings=None):
    return list((settings or {}).get('features', FEATURES))


def residual_target_order(settings):
    mode = settings.get('target_transform', 'delta')
    if mode == 'level':
        return []
    if mode == 'delta':
        return target_order(settings)
    # TSS uses observed history but remains a level target in the mixed formulation.
    return list(settings.get('residual_targets', [name for name in target_order(settings) if name in TARGETS]))


def validate_schema(config):
    targets = target_order(config)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('Targets must be a nonempty ordered list of unique columns.')
    features = feature_order(config)
    if features != FEATURES or config.get('actions', ACTIONS) != ACTIONS:
        raise ValueError('History requires the 21-channel sensor/context/action schema including measured TSS.')
    if '1_tss(mg/L)' in features and '1_tss(mg/L)' not in targets:
        raise ValueError('TSS history requires a TSS output for recursive feedback.')
    mode = config.get('target_transform', 'delta')
    if mode not in ['delta', 'level', 'mixed']:
        raise ValueError('target_transform must be delta, level, or mixed.')
    if mode == 'delta' and not set(targets).issubset(features):
        raise ValueError('Delta targets require observed target history; use level prediction for output-only targets.')
    if set(targets) & {'biomassa_estimasi(g/L)', '1_bod(mg/L)'}:
        if config.get('label_mode') != 'proxy_prototype':
            raise ValueError('Estimated biomass/provisional BOD require label_mode=proxy_prototype; measured training needs verified label columns.')
        if mode not in ['level', 'mixed']:
            raise ValueError('Output-only proxy targets require level or mixed prediction without interpolated target history.')
    residuals = residual_target_order(config)
    if len(set(residuals)) != len(residuals) or not set(residuals).issubset(set(features) & set(targets)):
        raise ValueError('Residual targets must be unique observed-history targets.')
    derived_output_specs(config)
    headline = config.get('headline_targets')
    if headline is not None:
        if not headline or len(set(headline)) != len(headline):
            raise ValueError('headline_targets must be a nonempty list of unique columns.')
        missing = set(headline) - set(targets)
        if missing:
            raise ValueError(f'headline_targets not in targets: {sorted(missing)}')
        overlaps = set(headline) & set((config.get('derived_outputs') or {}))
        if overlaps:
            raise ValueError(f'Derived outputs cannot be headline targets: {sorted(overlaps)}')
    provenance = config.get('target_provenance', {})
    if config.get('label_mode') and set(provenance) != set(targets):
        raise ValueError('Declare provenance for every selected target.')
    return targets


def baseline_values(raw, preprocessing):
    latest = np.asarray(raw['latest'], dtype=np.float32)
    if preprocessing.get('target_transform', 'delta') in ['level', 'mixed']:
        fill = np.asarray(preprocessing.get('training_target_mean', preprocessing['delta_mean']))
        return np.where(np.isfinite(latest), latest, fill).astype(np.float32)
    return latest


def selection_policy(settings, validation=None):
    """How a neural run chooses its final weights.

    'validation'    -- early-stop on the held-out cultivation (the default).
    'fixed_epochs'  -- train for an externally justified budget and keep the last
                       epoch. Used for the final all-data model, which has no
                       held-out cultivation left to select on. The epoch budget
                       must come from cross-validation evidence, not from the
                       model's own training error.
    """
    mode = settings.get('selection', 'validation')
    if mode not in ['validation', 'fixed_epochs']:
        raise ValueError("selection must be 'validation' or 'fixed_epochs'.")
    if mode == 'validation':
        if validation is None or not len(validation.get('y', [])):
            raise ValueError('Validation selection requires a nonempty validation cohort; '
                             "use selection='fixed_epochs' for an all-data model.")
        return {'mode': 'validation', 'fixed_epochs': None}
    budget = settings.get('fixed_epochs')
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("selection='fixed_epochs' requires a positive integer fixed_epochs.")
    if budget > settings['epochs']:
        raise ValueError(f'fixed_epochs {budget} exceeds the epoch ceiling {settings["epochs"]}.')
    if settings.get('include_initial_checkpoint', False):
        raise ValueError('include_initial_checkpoint is meaningless with fixed_epochs; '
                         'that mode always keeps trained weights.')
    return {'mode': 'fixed_epochs', 'fixed_epochs': int(budget)}


def validation_available(validation):
    return validation is not None and bool(len(validation.get('y', [])))


def baseline_window_steps(preprocessing):
    """How many trailing observations the delta anchor averages over.

    1 reproduces the original persistence anchor (the last observed value). A larger
    window anchors the prediction to a local level estimate, which is markedly less
    noisy: measured on K16 at a 15-minute horizon, a 15-step trailing mean predicts the
    level with about +0.19 skill over the last value on 8 of 9 measured channels.
    """
    settings = preprocessing or {}
    steps = settings.get('baseline_window_steps', 1)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('baseline_window_steps must be a positive integer.')
    return int(steps)


def history_anchor(sequence, steps):
    """Per-channel anchor from the trailing mean of the most recent `steps` rows.

    Accepts NumPy arrays or torch tensors; the final axis before channels is always time.
    """
    steps = min(int(steps), sequence.shape[-2])
    if steps <= 1:
        return sequence[..., -1, :]
    if torch is not None and isinstance(sequence, torch.Tensor):
        return sequence[..., -steps:, :].mean(dim=-2)
    sequence = np.asarray(sequence)
    return sequence[..., -steps:, :].mean(axis=-2)


def target_baseline(latest, preprocessing):
    mode = preprocessing.get('target_transform', 'delta')
    if mode == 'level':
        return 0.
    if mode == 'mixed':
        return np.where([name in residual_target_order(preprocessing) for name in target_order(preprocessing)], latest, 0.)
    return latest


def derived_output_specs(settings):
    """Declared algebraic output relations. The value is derived, not learned."""
    specs = dict((settings or {}).get('derived_outputs') or {})
    targets, features = target_order(settings or {}), feature_order(settings)
    for name, spec in specs.items():
        if name not in targets:
            raise ValueError(f'Derived output {name!r} is not a declared target.')
        source = spec.get('source')
        if source not in targets:
            raise ValueError(f'Derived output {name!r} source {source!r} must be a predicted target.')
        if source not in features:
            raise ValueError(f'Derived output {name!r} source {source!r} must be an input channel.')
        if name == source:
            raise ValueError(f'Derived output {name!r} cannot derive from itself.')
        for key in ['scale', 'offset']:
            value = spec.get(key, 1. if key == 'scale' else 0.)
            if not np.isfinite(float(value)):
                raise ValueError(f'Derived output {name!r} {key} must be finite.')
    return specs


def derived_target_indices(preprocessing):
    """(output column, source column, scale, offset) for each derived target."""
    targets = target_order(preprocessing)
    return [(targets.index(name), targets.index(spec['source']),
             float(spec.get('scale', 1.)), float(spec.get('offset', 0.)))
            for name, spec in (preprocessing.get('derived_outputs') or {}).items()]


def apply_derived_outputs(levels, preprocessing):
    """Overwrite derived target columns from their declared source column."""
    columns = derived_target_indices(preprocessing)
    if not columns:
        return levels
    levels = np.array(levels, dtype=levels.dtype, copy=True)
    for output, source, scale, offset in columns:
        levels[:, output] = scale * levels[:, source] + offset
    return levels


def derived_loss_mask(preprocessing, dtype=np.float32):
    """Zero weight for derived outputs: their network column is discarded, not fitted."""
    mask = np.ones(len(target_order(preprocessing)), dtype=dtype)
    for output, _, _, _ in derived_target_indices(preprocessing):
        mask[output] = 0.
    if mask.sum() == 0:
        raise ValueError('Every target is derived; at least one must remain learned.')
    return mask


def headline_target_names(preprocessing, names=None):
    """Targets that drive selection; defaults to every scored, non-derived channel."""
    names = list(names) if names is not None else target_order(preprocessing)
    derived = set((preprocessing.get('derived_outputs') or {}).keys())
    declared = preprocessing.get('headline_targets')
    selected = [name for name in names if name not in derived and (declared is None or name in declared)]
    return selected or [name for name in names if name not in derived] or names


def load_frames(dataset_dir, ids, config=None):
    targets = validate_schema(config) if config is not None else TARGETS
    features = feature_order(config)
    interval = time_settings(config)['sampling_interval_seconds'] if config is not None else 60
    suffix = '5detik' if interval == 5 else 'menit'
    frames, reports = {}, []
    for cultivation in progress(ids, enabled=(config or {}).get('progress_bars', True), desc='Load and review sources', unit='cultivation'):
        path = Path(dataset_dir) / f'K{cultivation:02d}_sensor_{suffix}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {interval}-second source: {path}. Copy the matching original CSV; no upsampling is performed.')
        columns = pd.read_csv(path, nrows=0).columns
        numeric_columns = list(dict.fromkeys(features + targets))
        required = numeric_columns + ['timestamp', 'kultivasi'] + (['n_data'] if interval == 60 else [])
        missing = set(required) - set(columns)
        if missing:
            raise ValueError(f'{path.name}: missing {sorted(missing)}')
        usecols = required + (['status'] if 'status' in columns else [])
        frame = pd.read_csv(path, usecols=usecols, dtype=dict.fromkeys(numeric_columns, np.float32))
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='raise')
        if frame.timestamp.isna().any() or frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
            raise ValueError(f'{path.name}: timestamps must be valid, unique, and ordered.')
        if not frame.kultivasi.eq(cultivation).all():
            raise ValueError(f'{path.name}: unexpected cultivation IDs.')
        frame.attrs['sampling_interval_seconds'] = interval
        frame.attrs['observation_operator'] = 'instantaneous_sample' if interval == 5 else 'interval_mean'
        quality = wp.attach_quality(frame, targets, interval, (config or {}).get('quality_diagnostics', {}))
        frame.attrs['quality_summary'] = quality.to_dict('records')
        frames[cultivation] = frame
        reports.append({'cultivation': cultivation, 'file': path.name, 'rows': len(frame),
                        'start': frame.timestamp.min(), 'end': frame.timestamp.max(),
                        'sampling_interval_seconds': interval, 'observation_operator': frame.attrs['observation_operator'],
                        'full_bins': int(frame.n_data.eq(12).sum()) if interval == 60 else None,
                        'eligible_observation_rows': int(observed_rows(frame, interval, features).sum()),
                        'quality_flags_available': 'status' in frame, 'sha256': file_hash(path)})
    return frames, pd.DataFrame(reports)


def valid_anchors(frame, history, interval_seconds=None, features=None):
    interval = interval_seconds or frame.attrs.get('sampling_interval_seconds', 60)
    lengths = continuous_run_lengths(observed_rows(frame, interval, features), frame.timestamp, interval)
    return np.flatnonzero(lengths >= history + 1)


def sample_blocks(eligible, count, block_size, rng):
    if count is None or len(eligible) <= count:
        return eligible.copy()
    cuts = np.flatnonzero(np.diff(eligible) != 1) + 1
    blocks = [run[start:start + block_size] for run in np.split(eligible, cuts)
              for start in range(0, len(run), block_size)]
    selected, total = [], 0
    for block_index in rng.permutation(len(blocks)):
        block = blocks[block_index]
        selected.append(block[:count - total])
        total += len(selected[-1])
        if total >= count:
            break
    return np.sort(np.concatenate(selected))


def raw_windows(frames, config, per_cultivation_cap=None, anchor_override=None):
    target_names = validate_schema(config)
    features = feature_order(config)
    timing = time_settings(config)
    history, interval = timing['history_steps'], timing['sampling_interval_seconds']
    rng = np.random.default_rng(config['seed'])
    selections, counts = [], []
    for cultivation, frame in sorted(frames.items()):
        eligible = valid_anchors(frame, history, interval, features)
        eligible = eligible[np.isfinite(frame[target_names].to_numpy(dtype=np.float32)[eligible]).all(axis=1)]
        selected = (np.asarray(anchor_override.get(cultivation, []), dtype=int) if anchor_override is not None else
                    wp.select_starts(eligible, per_cultivation_cap, frame, {**config, **timing}, sample_blocks, rng))
        if not np.isin(selected, eligible).all():
            raise ValueError('Explicit evaluation anchors must be eligible complete windows.')
        selections.append((cultivation, frame, selected))
        counts.append({'cultivation': cultivation, 'eligible_windows': len(eligible), 'selected_windows': len(selected),
                       **wp.coverage(frame, selected, eligible, ACTIONS)})
    count = sum(len(selected) for _, _, selected in selections)
    if not count:
        raise ValueError('No complete consecutive windows available.')
    estimated_gib = count * (history * len(features) + len(ACTIONS) + 3 + 2 * len(target_names)) * 4 * 4 / 1024 ** 3
    if estimated_gib > config.get('evaluation_working_budget_gib', 1):
        raise MemoryError(f'Evaluation windows need about {estimated_gib:.2f} GiB including working copies. Set an explicit evaluation-window cap.')
    sequences = np.empty((count, history, len(features)), np.float32)
    extras = np.empty((count, len(ACTIONS) + 3), np.float32)
    latest = np.full((count, len(target_names)), np.nan, np.float32)
    targets = np.empty_like(latest)
    anchors, row = [], 0
    for cultivation, frame, selected in selections:
        values = frame[features].to_numpy(dtype=np.float32)
        target_values = frame[target_names].to_numpy(dtype=np.float32)
        clocks = clock_features(frame.timestamp, frame.timestamp.iloc[0])
        for index in selected:
            sequences[row] = values[index - history:index]
            extras[row] = np.r_[values[index, -len(ACTIONS):], clocks[index]]
            targets[row] = target_values[index]
            window = baseline_window_steps(config)
            anchor_values = history_anchor(sequences[row], window)
            for j, name in enumerate(target_names):
                if name in features:
                    latest[row, j] = anchor_values[features.index(name)]
            anchors.append({'cultivation': cultivation, 'index': int(index), 'timestamp': frame.timestamp.iloc[index].isoformat()})
            row += 1
    return {'sequence': sequences, 'features': features, 'extra': extras, 'latest': latest, 'truth': targets, 'anchors': anchors,
            'sampling_interval_seconds': interval, 'observation_operator': timing['observation_operator']}, pd.DataFrame(counts)




def transform(raw, preprocessing):
    features = feature_order(preprocessing)
    if raw['sequence'].shape[-1] != len(features):
        raise ValueError('History width differs from saved preprocessing feature order.')
    mean, scale = np.array(preprocessing['input_mean']), np.array(preprocessing['input_scale'])
    filtered = wp.input_filter(raw['sequence'], preprocessing.get('input_filter', {'mode': 'none'}), features, preprocessing.get('sampling_interval_seconds', 60))
    sequence = ((filtered - mean) / scale).astype(np.float32)
    extra = raw['extra'].copy()
    extra[:, :len(ACTIONS)] = (extra[:, :len(ACTIONS)] - mean[-len(ACTIONS):]) / scale[-len(ACTIONS):]
    raw = {**raw, 'features': features, 'latest': baseline_values(raw, preprocessing)}
    baseline = target_baseline(raw['latest'], preprocessing)
    y = ((raw['truth'] - baseline - np.array(preprocessing['delta_mean'])) / np.array(preprocessing['delta_scale'])).astype(np.float32)
    return {'sequence': sequence, 'extra': extra, 'flat': np.concatenate([sequence.reshape(len(y), -1), extra], axis=1),
            'y': y, 'raw': raw}


def point_metrics(truth, predicted, latest, preprocessing, columns=None):
    targets = target_order(preprocessing)
    features = feature_order(preprocessing)
    indices = np.arange(len(targets)) if columns is None else np.array([targets.index(c) for c in columns])
    truth, latest = truth[:, indices], latest[:, indices]
    if predicted.shape[1] == len(targets):
        predicted = predicted[:, indices]
    elif predicted.shape[1] != len(indices):
        raise ValueError('Prediction width does not match the scored target subset.')
    std = np.array(preprocessing['training_level_std'])[indices]
    error = predicted - truth
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    persistence = np.sqrt(np.mean((latest - truth) ** 2, axis=0))
    normalized = np.divide(rmse, std, out=np.full_like(rmse, np.nan), where=std > 0)
    skill = 1 - np.divide(rmse, persistence, out=np.full_like(rmse, np.nan), where=persistence > 0)
    names = [targets[i] for i in indices]
    methods = ['persistence' if name in features else 'training_mean' for name in names]
    table = pd.DataFrame({'target': names, 'MAE': np.mean(np.abs(error), axis=0),
                          'RMSE': rmse, 'NRMSE': normalized, 'baseline_RMSE': persistence,
                          'baseline_skill': skill, 'baseline_method': methods,
                          'persistence_RMSE': np.where(np.array(methods) == 'persistence', persistence, np.nan),
                          'persistence_skill': np.where(np.array(methods) == 'persistence', skill, np.nan)})
    finite_columns = np.isfinite(predicted).all(axis=0)
    table['prediction_status'] = np.where(finite_columns, 'valid', 'invalid_nonfinite')
    table['normalization_status'] = np.where(std > 0, 'available', 'zero_training_variance')
    derived = set((preprocessing.get('derived_outputs') or {}).keys())
    table['is_derived'] = [name in derived for name in names]
    active = std > 0
    score = float(np.mean(normalized[active])) if finite_columns.all() and active.any() else np.inf
    headline = np.array([name in set(headline_target_names(preprocessing)) and name not in derived for name in names])
    headline_mask = active & headline
    headline_score = (float(np.mean(normalized[headline_mask]))
                      if finite_columns.all() and headline_mask.any() else np.inf)
    table.attrs['macro_NRMSE_all'] = score
    table.attrs['macro_NRMSE_headline'] = headline_score
    table.attrs['headline_targets'] = [names[i] for i in np.flatnonzero(headline_mask)]
    return table, headline_score


if torch is not None:
    class CausalBlock(nn.Module):
        def __init__(self, incoming, channels, dilation):
            super().__init__()
            self.pad = 2 * dilation
            self.first = nn.Conv1d(incoming, channels, 3, padding=self.pad, dilation=dilation)
            self.second = nn.Conv1d(channels, channels, 3, padding=self.pad, dilation=dilation)
            self.shortcut = nn.Conv1d(incoming, channels, 1) if incoming != channels else nn.Identity()

        def forward(self, x):
            h = torch.relu(self.first(x)[..., :-self.pad])
            h = self.second(h)[..., :-self.pad]
            return torch.relu(h + self.shortcut(x))


    class DynamicsNetwork(nn.Module):
        def __init__(self, architecture):
            super().__init__()
            self.architecture = architecture
            name, width = architecture['algorithm'], architecture['hidden_size']
            channels = architecture.get('input_channels', len(FEATURES))
            outputs = architecture.get('outputs', len(TARGETS))
            if name in ['gru', 'lstm']:
                recurrent = nn.GRU if name == 'gru' else nn.LSTM
                self.encoder = recurrent(channels, width, batch_first=True)
                encoded_size = width
            elif name == 'tft':
                self.variable_score = nn.Linear(channels, channels)
                self.encoder = nn.LSTM(channels, width, batch_first=True)
                self.attention = nn.MultiheadAttention(width, architecture['attention_heads'], batch_first=True)
                self.tft_norm = nn.LayerNorm(width)
                encoded_size = width
            elif name == 'dnlssm':
                latent = architecture['latent_size']
                self.encoder = nn.LSTM(channels, width, batch_first=True)
                self.latent_posterior = nn.Linear(width, 2 * latent)
                self.latent_transition = nn.Sequential(nn.Linear(latent + len(ACTIONS) + 3, width), nn.Tanh(),
                                                       nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2 * latent))
                self.observation_emission = nn.Sequential(nn.Linear(latent, width), nn.Tanh(),
                                                          nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2 * outputs))
                encoded_size = 0
            elif name == 'nhits':
                length = architecture['history_steps']
                self.pool_scales = architecture['pool_scales']
                self.nhits_blocks = nn.ModuleList([
                    nn.Sequential(nn.Linear(math.ceil(length / scale) * channels + len(ACTIONS) + 3, width),
                                  nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                  nn.Linear(width, math.ceil(length / scale) * channels + 2 * outputs))
                    for scale in self.pool_scales])
                encoded_size = 0
            elif name == 'dlinear':
                length = architecture['history_steps']
                kernel = architecture['moving_average_kernel']
                self.trend_pool = nn.AvgPool1d(kernel, stride=1, padding=0)
                self.trend_projection = nn.Linear(length, outputs)
                self.residual_projection = nn.Linear(length, outputs)
                self.action_projection = nn.Linear(len(ACTIONS) + 3, outputs)
                self.variance_projection = nn.Linear(length * channels + len(ACTIONS) + 3, outputs)
                encoded_size = 0
            elif name == 'pets_mlp':
                encoded_size = architecture['history_steps'] * channels
            elif name in ENSEMBLE_ALGORITHMS:
                # Shared trunk for the three offline-RL dynamics models. MOUP keeps
                # dropout active at inference so MC-dropout variance is available.
                length = architecture['history_steps'] * channels
                p_drop = float(architecture.get('mc_dropout', 0.))
                trunk = [nn.Linear(length + len(ACTIONS) + 3, width), nn.LayerNorm(width), nn.ReLU()]
                if p_drop:
                    trunk.append(nn.Dropout(p_drop))
                trunk.extend([nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU()])
                if p_drop:
                    trunk.append(nn.Dropout(p_drop))
                self.ensemble_trunk = nn.Sequential(*trunk)
                self.ensemble_head = nn.Linear(width, 2 * outputs)
                self.mc_dropout = p_drop
                encoded_size = 0
            elif name == 'tcn':
                blocks, incoming = [], channels
                for dilation in architecture['dilations']:
                    blocks.append(CausalBlock(incoming, width, dilation))
                    incoming = width
                self.encoder = nn.Sequential(*blocks)
                encoded_size = width
            elif name == 'pg_ac_tcn':
                sensor_channels, action_channels = architecture['sensor_channels'], architecture['action_channels']
                blocks, incoming = [], sensor_channels
                for dilation in architecture['dilations']:
                    blocks.append(CausalBlock(incoming, width, dilation))
                    incoming = width
                self.sensor_encoder = nn.Sequential(*blocks)
                self.action_encoder = nn.GRU(action_channels, width, batch_first=True)
                self.action_context = nn.Sequential(nn.Linear(width + len(ACTIONS) + 3, width), nn.ReLU(),
                                                   nn.Linear(width, width), nn.ReLU())
                self.film = nn.Linear(width, 2 * width)
                encoded_size = 2 * width
            else:
                encoded_size = architecture.get('history_steps', architecture['history_minutes']) * channels
            latest_sensor_names = architecture.get('latest_sensor_names', []) if name == 'tcn' else []
            inputs = (encoded_size if name == 'pg_ac_tcn' else
                      encoded_size + len(ACTIONS) + 3 + len(latest_sensor_names))
            dropout = architecture.get('head_dropout', 0.)
            def make_head(output_count):
                layers = [nn.Linear(inputs, width), nn.ReLU()]
                if dropout:
                    layers.append(nn.Dropout(dropout))
                layers.extend([nn.Linear(width, width), nn.ReLU()])
                if dropout:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(width, 2 * output_count))
                return nn.Sequential(*layers)
            self.head_layout = architecture.get('head_layout', 'shared') if name in ['tcn', 'pg_ac_tcn', 'dlinear', 'nhits', 'tft'] else 'shared'
            if name == 'nhits':
                self.head_layout = 'shared'
            if name == 'dlinear':
                self.head_layout = 'shared'
            if name == 'tft':
                self.head_layout = 'shared'
            if name in ['dlinear', 'nhits', 'dnlssm']:
                pass
            elif self.head_layout == 'dedicated':
                groups = architecture['head_groups']
                self.heads = nn.ModuleDict({key: make_head(len(indices)) for key, indices in groups.items()})
            else:
                self.head = make_head(outputs)

        def forward(self, sequence, extra):
            name = self.architecture['algorithm']
            if name in ['gru', 'lstm']:
                encoded = self.encoder(sequence)[0][:, -1]
            elif name == 'tft':
                weights = torch.softmax(self.variable_score(sequence), dim=-1)
                selected = sequence * weights
                temporal, _ = self.encoder(selected)
                attended, _ = self.attention(temporal, temporal, temporal, need_weights=False)
                encoded = self.tft_norm(temporal[:, -1] + attended[:, -1])
            elif name == 'dnlssm':
                encoded, _ = self.encoder(sequence)
                latent_mean, raw_latent_variance = self.latent_posterior(encoded[:, -1]).chunk(2, dim=1)
                latent_variance = (-8 + 10 * torch.sigmoid(raw_latent_variance)).exp().clamp_min(1e-8)
                batch, dimension = latent_mean.shape
                eye = torch.eye(dimension, device=sequence.device, dtype=sequence.dtype)

                def sigma_points(mean, variance):
                    spread = torch.sqrt(variance.clamp_min(1e-8) * dimension)
                    offsets = spread.unsqueeze(1) * eye.unsqueeze(0)
                    return torch.cat([mean.unsqueeze(1), mean.unsqueeze(1) + offsets,
                                      mean.unsqueeze(1) - offsets], dim=1)

                mean_weights = torch.cat([sequence.new_zeros(1), sequence.new_full((2 * dimension,), 1 / (2 * dimension))])
                covariance_weights = mean_weights.clone()
                covariance_weights[0] = 2.
                transition_points = sigma_points(latent_mean, latent_variance)
                repeated_extra = extra.unsqueeze(1).expand(-1, 2 * dimension + 1, -1)
                transition_mean, transition_raw_variance = self.latent_transition(
                    torch.cat([transition_points, repeated_extra], dim=-1)).chunk(2, dim=-1)
                predicted_latent_mean = (transition_mean * mean_weights[None, :, None]).sum(dim=1)
                transition_variance = (-8 + 10 * torch.sigmoid(transition_raw_variance)).exp()
                predicted_latent_variance = (covariance_weights[None, :, None] *
                    ((transition_mean - predicted_latent_mean[:, None]).square() + transition_variance)).sum(dim=1).clamp_min(1e-8)

                emission_points = sigma_points(predicted_latent_mean, predicted_latent_variance)
                emission_mean, emission_raw_variance = self.observation_emission(emission_points).chunk(2, dim=-1)
                predicted_mean = (emission_mean * mean_weights[None, :, None]).sum(dim=1)
                emission_variance = (-8 + 10 * torch.sigmoid(emission_raw_variance)).exp()
                predicted_variance = (covariance_weights[None, :, None] *
                    ((emission_mean - predicted_mean[:, None]).square() + emission_variance)).sum(dim=1).clamp_min(1e-8)
                return predicted_mean, predicted_variance.log()
            elif name == 'dlinear':
                # Channel-independent: keep (batch, channel, time) so the projection
                # acts on the temporal axis, then average the per-channel forecasts.
                x = sequence.transpose(1, 2)
                pad = self.architecture['moving_average_kernel'] // 2
                padded = torch.nn.functional.pad(x, (pad, pad), mode='replicate')
                trend = self.trend_pool(padded)
                seasonal = x - trend
                mean = (self.trend_projection(trend).mean(dim=1)
                        + self.residual_projection(seasonal).mean(dim=1)
                        + self.action_projection(extra))
                raw_variance = self.variance_projection(torch.cat([sequence.flatten(1), extra], dim=1))
                return mean, -8 + 10 * torch.sigmoid(raw_variance)
            elif name in ENSEMBLE_ALGORITHMS:
                flat = torch.cat([sequence.flatten(1), extra], dim=1)
                mean, raw_variance = self.ensemble_head(self.ensemble_trunk(flat)).chunk(2, dim=1)
                return mean, -8 + 10 * torch.sigmoid(raw_variance)
            elif name == 'nhits':
                forecasts, variances = [], []
                residual = sequence
                for scale, block in zip(self.pool_scales, self.nhits_blocks):
                    pooled = torch.nn.functional.adaptive_avg_pool1d(residual.transpose(1, 2), math.ceil(sequence.shape[1] / scale))
                    features = torch.cat([pooled.flatten(1), extra], dim=1)
                    backcast_and_forecast = block(features)
                    pooled_steps = pooled.shape[-1]
                    backcast = backcast_and_forecast[:, :pooled_steps * sequence.shape[-1]]
                    backcast = backcast.reshape(len(sequence), pooled_steps, sequence.shape[-1]).transpose(1, 2)
                    residual = residual - torch.nn.functional.interpolate(backcast, size=sequence.shape[1], mode='linear', align_corners=False).transpose(1, 2)
                    forecast = backcast_and_forecast[:, pooled_steps * sequence.shape[-1]:]
                    mean, raw_variance = forecast.chunk(2, dim=1)
                    forecasts.append(mean)
                    variances.append(torch.exp(-8 + 10 * torch.sigmoid(raw_variance)))
                stacked_means, stacked_variances = torch.stack(forecasts), torch.stack(variances)
                mean = stacked_means.mean(0)
                variance = (stacked_variances + stacked_means.square()).mean(0) - mean.square()
                return mean, variance.clamp_min(1e-8).log()
            elif name == 'tcn':
                encoded = self.encoder(sequence.transpose(1, 2))[:, :, -1]
            elif name == 'pg_ac_tcn':
                sensor_width, action_width = self.architecture['sensor_channels'], self.architecture['action_channels']
                sensor_encoding = self.sensor_encoder(sequence[:, :, :sensor_width].transpose(1, 2))[:, :, -1]
                action_sequence = sequence[:, :, sensor_width:sensor_width + action_width]
                action_encoding = self.action_encoder(action_sequence)[0][:, -1]
                action_context = self.action_context(torch.cat([action_encoding, extra], dim=1))
                gamma, beta = self.film(action_context).chunk(2, dim=1)
                conditioned = sensor_encoding * (1 + torch.tanh(gamma)) + beta
                head_input = torch.cat([conditioned, action_context], dim=1)
            else:
                encoded = sequence.flatten(1)
            if name != 'pg_ac_tcn':
                latest_indices = self.architecture.get('latest_sensor_indices', []) if name == 'tcn' else []
                parts = [encoded, extra]
                if latest_indices:
                    parts.append(sequence[:, -1, latest_indices])
                head_input = torch.cat(parts, dim=1)
            if self.head_layout == 'dedicated':
                mean_columns, variance_columns = [None] * self.architecture['outputs'], [None] * self.architecture['outputs']
                for key, indices in self.architecture['head_groups'].items():
                    group_mean, group_log_variance = self.heads[key](head_input).chunk(2, dim=1)
                    for local, target_index in enumerate(indices):
                        mean_columns[target_index] = group_mean[:, local]
                        variance_columns[target_index] = group_log_variance[:, local]
                mean = torch.stack(mean_columns, dim=1)
                raw_variance = torch.stack(variance_columns, dim=1)
            else:
                mean, raw_variance = self.head(head_input).chunk(2, dim=1)
            log_variance = -8 + 10 * torch.sigmoid(raw_variance)
            return mean, log_variance


def make_architecture(algorithm, config):
    settings = config['neural'][algorithm]
    timing = time_settings(config)
    if settings.get('latest_sensor_to_head') and algorithm != 'tcn':
        raise ValueError(f'{algorithm}: latest_sensor_to_head is implemented for tcn only. '
                         'Remove the setting or select tcn; silently ignoring it would let the '
                         'artifact record a flag the network does not use.')
    if algorithm not in ['tcn', 'pg_ac_tcn'] and settings.get('head_layout', 'shared') != 'shared':
        raise ValueError(f'{algorithm}: dedicated heads are implemented for the TCN family only.')
    head_layout = settings.get('head_layout', 'shared') if algorithm in ['tcn', 'pg_ac_tcn', 'dlinear', 'nhits', 'tft', 'dnlssm'] else 'shared'
    latest_sensor_to_head = bool(settings.get('latest_sensor_to_head', False)) if algorithm == 'tcn' else False
    targets, features = target_order(config), feature_order(config)
    dedicated_names = ['1_bod(mg/L)', 'biomassa_estimasi(g/L)', '1_tss(mg/L)']
    if head_layout not in ['shared', 'dedicated']:
        raise ValueError("Neural head_layout must be 'shared' or 'dedicated'.")
    if algorithm not in ['tcn', 'pg_ac_tcn'] and (head_layout != 'shared' or latest_sensor_to_head):
        raise ValueError('Dedicated heads and direct latest-sensor inputs are currently TCN-family options.')
    if algorithm == 'tft' and settings['hidden_size'] % settings.get('attention_heads', 4):
        raise ValueError('TFT hidden_size must be divisible by attention_heads.')
    if algorithm == 'dlinear':
        kernel = settings.get('moving_average_kernel', 25)
        if isinstance(kernel, bool) or not isinstance(kernel, int) or kernel < 1 or kernel % 2 == 0 or kernel > timing['history_steps']:
            raise ValueError('DLinear moving_average_kernel must be a positive odd integer no longer than the input history.')
    if algorithm == 'nhits':
        scales = settings.get('pool_scales', [1, 4, 16])
        if not scales or any(isinstance(scale, bool) or not isinstance(scale, int) or scale < 1 for scale in scales):
            raise ValueError('N-HiTS pool_scales must be a nonempty list of positive integers.')
    if head_layout == 'dedicated' and not set(dedicated_names).issubset(targets):
        raise ValueError('Dedicated TCN-family heads require BOD, biomass estimate, and TSS targets.')
    latest_sensor_names = ([name for name in targets if name in features]
                           if latest_sensor_to_head else [])
    head_groups = {'shared': [i for i, name in enumerate(targets) if name not in dedicated_names]}
    dedicated_keys = {'1_bod(mg/L)': 'bod', 'biomassa_estimasi(g/L)': 'biomass', '1_tss(mg/L)': 'tss'}
    head_groups.update({key: [targets.index(name)] for name, key in dedicated_keys.items() if name in targets})
    if head_layout == 'dedicated' and not head_groups['shared']:
        raise ValueError('Dedicated TCN-family head configuration requires at least one shared output.')
    if algorithm in ENSEMBLE_ALGORITHMS:
        ensemble = ENSEMBLE_ALGORITHMS[algorithm]
        if settings.get('members', 1) < 2:
            raise ValueError(f'{algorithm}: ensemble uncertainty needs members >= 2; '
                             'a single member has no disagreement to measure.')
        mc_dropout = float(settings.get('mc_dropout', ensemble['mc_dropout']))
        if not 0. <= mc_dropout < 1.:
            raise ValueError(f'{algorithm}: mc_dropout must be in [0, 1).')
        if ensemble['uncertainty_mode'] == 'mc_dropout' and mc_dropout <= 0.:
            raise ValueError(f'{algorithm}: MC-dropout uncertainty requires mc_dropout > 0.')
        if ensemble['uncertainty_mode'] != 'mc_dropout' and mc_dropout > 0.:
            raise ValueError(f'{algorithm}: dropout is only used by MC-dropout uncertainty.')
        if settings.get('head_layout', 'shared') != 'shared':
            raise ValueError(f'{algorithm}: only a shared head is implemented.')
        if settings.get('latest_sensor_to_head'):
            raise ValueError(f'{algorithm}: latest_sensor_to_head is not implemented.')
    else:
        mc_dropout = 0.
    return {'algorithm': algorithm, 'history_minutes': config['history_minutes'],
            'history_steps': timing['history_steps'], 'sampling_interval_seconds': timing['sampling_interval_seconds'],
            'hidden_size': settings['hidden_size'], 'dilations': settings.get('dilations', []),
            'head_dropout': settings.get('head_dropout', 0.),
            'mc_dropout': mc_dropout,
            'uncertainty_mode': resolved_uncertainty_mode(algorithm, settings.get('members', 1)),
            'head_layout': head_layout, 'head_groups': head_groups,
            'attention_heads': settings.get('attention_heads', 4),
            'latent_size': settings.get('latent_size', 16),
            'sensor_channels': len(features) - len(ACTIONS),
            'action_channels': len(ACTIONS),
            'moving_average_kernel': settings.get('moving_average_kernel', 25),
            'pool_scales': settings.get('pool_scales', [1, 4, 16]),
            'latest_sensor_to_head': latest_sensor_to_head,
            'latest_sensor_names': latest_sensor_names,
            'latest_sensor_indices': [features.index(name) for name in latest_sensor_names],
            'input_channels': len(feature_order(config)), 'outputs': len(target_order(config)),
            'target_order': target_order(config), 'log_variance_bounds': [-8, 2]}


def bootstrap_indices(anchors, block_size, rng):
    blocks, block = [], []
    for index, anchor in enumerate(anchors):
        previous = anchors[index - 1] if index else None
        if block and (len(block) >= block_size or anchor['cultivation'] != previous['cultivation']
                      or anchor['index'] != previous['index'] + 1):
            blocks.append(block)
            block = []
        block.append(index)
    if block:
        blocks.append(block)
    selected = []
    while len(selected) < len(anchors):
        selected.extend(blocks[rng.integers(len(blocks))])
    return np.array(selected[:len(anchors)])


def mc_dropout_enabled(model):
    """MOUP keeps dropout active at inference; other architectures do not."""
    return float(getattr(model, 'mc_dropout', 0.) or 0.) > 0.


def neural_components(models, prepared, device='cpu', batch_size=256, mc_passes=None):
    """Per-member mean/variance prediction.

    With `mc_passes` each member contributes that many stochastic forward passes, so the
    returned component axis is (members * mc_passes) and the downstream Gaussian mixture
    represents Var[dropout] + mean predicted variance (MOUP Eq. 13). Passing None keeps
    dropout disabled, which is what per-epoch monitoring uses so the score stays
    deterministic and cheap.
    """
    means, variances = [], []
    for model in models:
        stochastic = mc_dropout_enabled(model) and mc_passes
        passes = int(mc_passes) if stochastic else 1
        for _ in range(passes):
            model.eval()
            if stochastic:
                # Dropout active; everything else stays in eval mode.
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.train()
            output_mean, output_variance = [], []
            with torch.no_grad():
                for start in range(0, len(prepared['sequence']), batch_size):
                    s = torch.as_tensor(prepared['sequence'][start:start + batch_size], device=device)
                    e = torch.as_tensor(prepared['extra'][start:start + batch_size], device=device)
                    mean, log_variance = model(s, e)
                    output_mean.append(mean.cpu().numpy())
                    output_variance.append(log_variance.exp().cpu().numpy())
            means.append(np.concatenate(output_mean))
            variances.append(np.concatenate(output_variance))
        if mc_dropout_enabled(model):
            model.eval()  # leave the stochastic mode so callers do not inherit it
    return np.stack(means), np.stack(variances)


def normalize_action_vector(actions):
    """Scale each action by its training spread so distance is not dominated by units."""
    actions = np.asarray(actions, dtype=np.float64)
    spread = actions.std(axis=0)
    spread = np.where(spread > 1e-12, spread, 1.)
    return (actions - actions.mean(axis=0)) / spread


def fit_action_reference(index, preprocessing):
    """Training action set used for novelty scoring, standardised per channel."""
    actions = []
    for cultivation, store in index.stores.items():
        positions = index.entries[index.entries[:, 0] == cultivation, 1]
        if not len(positions):
            continue
        actions.append(store['values'][positions, -len(ACTIONS):])
    if not actions:
        raise ValueError('No training actions available for the novelty reference.')
    pooled = np.concatenate(actions).astype(np.float64)
    # Keep the raw values; every distance is computed after standardising both sides with
    # the same training mean and scale, so units cannot dominate the nearest-neighbour search.
    return {'mean': pooled.mean(axis=0).tolist(),
            'scale': np.where(pooled.std(axis=0) > 1e-12, pooled.std(axis=0), 1.).tolist(),
            'reference': pooled.astype(np.float32)}


def scale_actions(actions, mean, scale, clip=8.0):
    """Standardise action vectors with an explicit training-derived mean and scale.

    Dosing is sparse and heavy tailed: a normal 25 mL bolus is already ~19 standard
    deviations from a mean near zero, so unscaled distances are dominated by routine
    dose events rather than genuinely untried regimes. The clip bounds each channel's
    influence so "far" means far in a comparable sense across actions.
    """
    actions = np.asarray(actions, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    scaled = (actions - mean) / np.maximum(scale, 1e-12)
    if clip is not None:
        scaled = np.clip(scaled, -float(clip), float(clip))
    return scaled


def action_novelty(actions, reference, k=5, mean=None, scale=None):
    """Distance from each action vector to its k-th nearest logged training action.

    This answers "has this dosing/aeration regime been tried?" rather than "have I seen
    this state?" -- which is what a controller needs when it proposes an action. It needs
    only one model and cannot collapse to a constant the way ensemble disagreement does
    on a slow process.

    `reference` must already be standardised with `mean`/`scale`. When those are given,
    the query is standardised the same way; otherwise the reference's own statistics are
    used, which is only correct if the reference is raw.
    """
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or not len(reference):
        raise ValueError('Action novelty requires a non-empty training reference.')
    if mean is None or scale is None:
        mean, scale = reference.mean(axis=0), reference.std(axis=0)
    # Both sides use the same training-derived scaling; otherwise a post-pump channel in
    # mL would dominate an aerator channel in L/s.
    reference = scale_actions(reference, mean, scale)
    query = scale_actions(actions, mean, scale)
    k = int(max(1, min(k, len(reference))))
    distances = np.empty(len(query))
    for start in range(0, len(query), 512):
        block = query[start:start + 512]
        d2 = ((block ** 2).sum(1)[:, None] + (reference ** 2).sum(1)[None, :]
              - 2 * block @ reference.T)
        distances[start:start + 512] = np.sqrt(np.maximum(d2, 0)).min(axis=1)
    return distances


def model_uncertainty(means, variances, mode):
    """Aggregate ensemble members into one model-uncertainty estimate per target.

    modes:
      'max_std'        MOPO  -- max_i sigma_i, the most pessimistic member.
      'ensemble_std'   Std_i mu_i, pure epistemic disagreement across members.
      'mc_dropout'     MOUP  -- Var_i mu_i + mean_i sigma_i^2 (epistemic + aleatoric).
    Inputs are (components, batch, targets); returns (batch, targets).
    """
    means = np.asarray(means, dtype=np.float64)
    variances = np.maximum(np.asarray(variances, dtype=np.float64), 1e-12)
    if means.shape[0] < 2 and mode in ['max_std', 'ensemble_std']:
        raise ValueError(f'{mode} needs at least two components to measure disagreement.')
    if mode == 'max_std':
        return np.sqrt(variances).max(axis=0)
    if mode == 'ensemble_std':
        return means.std(axis=0)
    if mode == 'mc_dropout':
        return np.sqrt((means - means.mean(axis=0)).var(axis=0) + variances.mean(axis=0))
    raise ValueError(f'Unknown uncertainty mode: {mode}')


def uncertainty_discrimination(fitted, in_distribution, out_of_distribution, preprocessing):
    """Is the uncertainty estimate informative, or effectively constant?

    A pessimism penalty u(s,a) only works if it is low where data is dense and high where
    it is sparse. Two axes are measured because they answer different questions:

      state axis  -- ensemble disagreement (epistemic doubt about the regime)
      action axis -- distance to the logged action set (has this control input been tried)

    For a controller the action axis is the operative one: a policy proposes actions, and
    on a slow process ensemble members can agree everywhere while the action region is
    entirely novel.
    """
    _, in_dist = predict(fitted, in_distribution)
    _, out_dist = predict(fitted, out_of_distribution)
    result = {'mode': fitted.get('uncertainty_mode')}
    if 'model_uncertainty' in in_dist:
        in_u = np.asarray(in_dist['model_uncertainty'])
        out_u = np.asarray(out_dist['model_uncertainty'])
        in_mean, out_mean = float(in_u.mean()), float(out_u.mean())
        ratio = out_mean / in_mean if in_mean > 0 else np.inf
        if not np.isfinite(ratio) or ratio < 1.05:
            note = ('ensemble uncertainty is nearly identical on held-out data, so a '
                    'pessimism penalty would be close to constant')
            status = 'not_discriminative'
        elif ratio > 1.25:
            status, note = 'discriminative', 'held-out data carries clearly higher ensemble uncertainty'
        else:
            status, note = 'weakly_discriminative', 'held-out data is only slightly less certain'
        result.update({'state_axis_status': status,
                       'in_distribution_mean_uncertainty': in_mean,
                       'out_of_distribution_mean_uncertainty': out_mean,
                       'ratio': None if not np.isfinite(ratio) else ratio,
                       'state_axis_interpretation': note})
    if 'action_novelty' in in_dist:
        in_n = np.asarray(in_dist['action_novelty'])
        out_n = np.asarray(out_dist['action_novelty'])
        in_mean, out_mean = float(in_n.mean()), float(out_n.mean())
        ratio = out_mean / in_mean if in_mean > 0 else np.inf
        if not np.isfinite(ratio) or ratio < 1.10:
            status = 'not_discriminative'
            note = 'held-out actions sit inside the logged training action range'
        elif ratio > 1.50:
            status = 'discriminative'
            note = 'held-out actions are clearly outside the logged training action range'
        else:
            status = 'weakly_discriminative'
            note = 'held-out actions are only slightly more novel than training actions'
        result.update({'action_axis_status': status,
                       'in_distribution_action_novelty': in_mean,
                       'out_of_distribution_action_novelty': out_mean,
                       'action_ratio': None if not np.isfinite(ratio) else ratio,
                       'action_axis_interpretation': note})
    if not result.get('state_axis_status') and not result.get('action_axis_status'):
        result.update({'status': 'no_model_uncertainty',
                       'interpretation': 'this artifact exposes neither ensemble spread nor action novelty'})
    else:
        result['status'] = result.get('action_axis_status') or result.get('state_axis_status')
        result['interpretation'] = (result.get('action_axis_interpretation')
                                    or result.get('state_axis_interpretation'))
    return result


def uncertainty_label(mode):
    return {'max_std': 'max ensemble std (MOPO)',
            'ensemble_std': 'ensemble mean dispersion (epistemic disagreement)',
            'mc_dropout': 'MC-dropout variance + mean predictive variance (MOUP)',
            None: 'Gaussian mixture'}.get(mode, str(mode))


def restore_levels(delta, latest, preprocessing):
    baseline = target_baseline(latest, preprocessing)
    levels = baseline + delta * np.array(preprocessing['delta_scale']) + np.array(preprocessing['delta_mean'])
    return apply_derived_outputs(levels, preprocessing)


def rollout_validation_cohort(frames, preprocessing, config, steps, cap=None):
    """Build a fixed multi-step validation cohort.

    Checkpoint selection then follows the training rollout length automatically: the
    cohort carries `steps` future logged actions and labels per anchor, all inside one
    continuous observation run, so every epoch is scored on the same rollout starts.
    """
    if steps < 1:
        raise ValueError('Rollout validation needs at least one step.')
    timing = time_settings(config)
    features, targets = feature_order(preprocessing), target_order(preprocessing)
    history_steps, interval = timing['history_steps'], timing['sampling_interval_seconds']
    action_count = len(ACTIONS)
    cap = cap if cap is not None else config.get('validation_windows_per_cultivation')
    sections = []
    for cultivation, frame in sorted(frames.items()):
        values = frame[features].to_numpy(np.float32)
        truth = frame[targets].to_numpy(np.float32)
        observed = observed_rows(frame, interval, features) & np.isfinite(truth).all(axis=1)
        run = continuous_run_lengths(observed, frame.timestamp, interval)
        # A usable anchor needs a complete history behind it and `steps` observed
        # values ahead, all inside one continuous run.
        anchors = np.flatnonzero(run >= history_steps)
        anchors = anchors[anchors >= history_steps]
        anchors = anchors[anchors + steps - 1 < len(frame)]
        if len(anchors):
            anchors = anchors[run[anchors + steps - 1] >= steps]
        if not len(anchors):
            continue
        if cap is not None and len(anchors) > cap:
            anchors = anchors[np.linspace(0, len(anchors) - 1, int(cap)).astype(int)]
        clock = clock_features(frame.timestamp, frame.timestamp.iloc[0])
        sections.append({
            'cultivation': cultivation,
            'history': np.stack([values[a - history_steps:a] for a in anchors]),
            'extra': np.stack([np.concatenate([values[a:a + steps, -action_count:],
                                               clock[a:a + steps]], axis=1)
                               for a in anchors]).astype(np.float32),
            'truth': np.stack([truth[a:a + steps] for a in anchors]),
            'anchors': [{'cultivation': cultivation, 'index': int(a),
                         'timestamp': frame.timestamp.iloc[a].isoformat()} for a in anchors]})
    if not sections:
        raise ValueError(f'No validation starts support a {steps}-step rollout.')
    return {'history': np.concatenate([s['history'] for s in sections]),
            'extra': np.concatenate([s['extra'] for s in sections]),
            'truth': np.concatenate([s['truth'] for s in sections]),
            'anchors': [a for s in sections for a in s['anchors']],
            'steps_available': steps,
            'sampling_interval_seconds': interval,
            'observation_operator': timing['observation_operator']}


def score_rollout(model, cohort, preprocessing, steps, device='cpu', batch_size=256):
    """Recursive score over a fixed cohort, using the same feedback rules as evaluation.

    Predicted target channels are fed back, unpredicted context persists, and logged
    future actions are replayed. Returns the point-metric table and headline score.
    """
    if not 1 <= steps <= cohort['steps_available']:
        raise ValueError(f'Requested {steps} rollout steps but the cohort has '
                         f"{cohort['steps_available']}.")
    targets, features = target_order(preprocessing), feature_order(preprocessing)
    fitted = {'kind': 'neural', 'models': [model], 'device': device,
              'preprocessing': preprocessing, 'inference_batch_size': batch_size}
    history = cohort['history'].copy()
    # Anchor exactly as the targets were built, so training and inference agree.
    window = baseline_window_steps(preprocessing)
    anchor_values = history_anchor(history, window)
    latest = np.full((len(history), len(targets)), np.nan, np.float32)
    for j, name in enumerate(targets):
        if name in features:
            latest[:, j] = anchor_values[:, features.index(name)]
    reference = baseline_values({'latest': latest}, preprocessing)
    truth_rows, prediction_rows = [], []
    for step in range(steps):
        raw = {'sequence': history, 'extra': cohort['extra'][:, step],
               'latest': latest, 'truth': cohort['truth'][:, step],
               'features': features,
               'sampling_interval_seconds': cohort['sampling_interval_seconds'],
               'observation_operator': cohort['observation_operator']}
        predicted, _ = predict(fitted, transform(raw, preprocessing))
        truth_rows.append(cohort['truth'][:, step])
        prediction_rows.append(predicted)
        if step + 1 < steps:
            next_row = history[:, -1].copy()
            for j, name in enumerate(targets):
                if name in features:
                    next_row[:, features.index(name)] = predicted[:, j]
            next_row[:, -len(ACTIONS):] = cohort['extra'][:, step, :len(ACTIONS)]
            history = np.concatenate([history[:, 1:], next_row[:, None]], axis=1)
    table, _ = point_metrics(np.concatenate(truth_rows), np.concatenate(prediction_rows),
                             np.tile(reference, (steps, 1)), preprocessing)
    return table, float(table.attrs['macro_NRMSE_headline'])






def mean_linear_ode(initial, rate, forcing, hours):
    x = np.asarray(rate) * hours
    factor = np.full(np.broadcast_shapes(np.shape(initial), np.shape(x)), .5, dtype=float)
    np.divide(x + np.expm1(-x), x ** 2, out=factor, where=np.abs(x) > 1e-6)
    return initial + (forcing - rate * initial) * hours * factor


def physical_prediction(parameters, raw, light_gain, sampling_interval_seconds=None, observation_operator=None):
    interval = sampling_interval_seconds or raw.get('sampling_interval_seconds', 60)
    operator = observation_operator or raw.get('observation_operator', 'interval_mean')
    integrate = mean_linear_ode if operator == 'interval_mean' else linear_ode_endpoint
    if operator not in ['interval_mean', 'instantaneous_sample']:
        raise ValueError('Unsupported physical observation operator.')
    features = raw.get('features', FEATURES)
    last = raw['sequence'][:, -1]
    next_actions = raw['extra'][:, :len(ACTIONS)]
    oxygen = last[:, features.index('1_do(mg/L)')]
    temperature = last[:, features.index('1_temp_air(C)')]
    ambient = last[:, features.index('2_temp(C)')]
    light = np.maximum(last[:, features.index('2_light(Lux)')] + light_gain *
                       (next_actions[:, 4] - last[:, features.index('Lampu (PWM%)')]), 0) / 1000
    k0, kq, equilibrium, production, respiration, heat_exchange, lamp_heat, bias = parameters
    rate = k0 + kq * next_actions[:, 3]
    hours = interval / 3600
    oxygen_next = integrate(oxygen, rate, rate * equilibrium + production * light - respiration, hours)
    temperature_next = integrate(temperature, heat_exchange,
                                 heat_exchange * ambient + lamp_heat * next_actions[:, 4] / 100 + bias, hours)
    return np.column_stack([oxygen_next, temperature_next])




def predict(fitted, prepared):
    preprocessing = fitted['preprocessing']
    latest = prepared['raw']['latest']
    if fitted['kind'] == 'physical':
        prediction = physical_prediction(fitted['parameters'], prepared['raw'], fitted['light_gain'],
                                         fitted.get('sampling_interval_seconds', 60),
                                         fitted.get('observation_operator', 'interval_mean'))
        return prediction, None
    if fitted['kind'] == 'neural':
        means, variances = neural_components(fitted['models'], prepared, fitted['device'],
                                            fitted.get('inference_batch_size', 256),
                                            mc_passes=fitted.get('mc_passes'))
        delta = means.mean(axis=0)
        distribution = {'component_means': means, 'component_variances': variances}
        mode = fitted.get('uncertainty_mode')
        if mode:
            distribution['model_uncertainty'] = model_uncertainty(means, variances, mode)
            distribution['uncertainty_mode'] = mode
            distribution['uncertainty_label'] = uncertainty_label(mode)
        reference = fitted.get('action_reference')
        if reference is not None:
            # Action novelty is per-window, not per-target, so it is reported separately.
            distribution['action_novelty'] = action_novelty(
                prepared['extra'][:, :len(ACTIONS)], reference['reference'],
                reference.get('k', 5), mean=reference['mean'], scale=reference['scale'])
            distribution['action_novelty_k'] = reference.get('k', 5)
    elif fitted['algorithm'] == 'gpr':
        delta, std = fitted['estimator'].predict(prepared['flat'], return_std=True)
        delta = delta + np.asarray(fitted.get('delta_offset', 0.))
        if std.ndim == 1:
            std = np.broadcast_to(std[:, None], delta.shape)
        distribution = {'component_means': delta[None], 'component_variances': (std ** 2)[None]}
    elif fitted['algorithm'] == 'hrp_xgb':
        means = np.stack([estimator.predict(prepared['flat']) for estimator in fitted['estimators']])
        delta = means.mean(axis=0)
        variances = np.broadcast_to(np.asarray(fitted['residual_variance'])[None, None, :], means.shape)
        distribution = {'component_means': means, 'component_variances': variances}
    else:
        delta = fitted['estimator'].predict(prepared['flat'])
        distribution = None
    levels = restore_levels(delta, latest, preprocessing)
    return levels, distribution


def uncertainty_metrics(distribution, prepared, preprocessing):
    if distribution is None:
        return None
    means, variances = distribution['component_means'], np.maximum(distribution['component_variances'], 1e-12)
    residual = prepared['y'][None] - means
    log_density = -.5 * (np.log(2 * np.pi * variances) + residual ** 2 / variances)
    largest = np.max(log_density, axis=0)
    log_mixture = largest + np.log(np.exp(log_density - largest).mean(axis=0))
    nll = (-log_mixture + np.log(np.array(preprocessing['delta_scale']))).mean(axis=0)
    # Exact equal-weight Gaussian-mixture marginal quantiles by deterministic bisection.
    from scipy.special import ndtr
    sd = np.sqrt(variances)
    def quantile(probability):
        low, high = (means - 12 * sd).min(axis=0), (means + 12 * sd).max(axis=0)
        for _ in range(45):
            mid = (low + high) / 2
            cdf = ndtr((mid[None] - means) / sd).mean(axis=0)
            low, high = np.where(cdf < probability, mid, low), np.where(cdf >= probability, mid, high)
        return (low + high) / 2
    lower, upper = quantile(.05), quantile(.95)
    covered = ((prepared['y'] >= lower) & (prepared['y'] <= upper)).mean(axis=0)
    width = ((upper - lower) * np.array(preprocessing['delta_scale'])).mean(axis=0)
    return pd.DataFrame({'target': target_order(preprocessing), 'marginal_NLL_native_units': nll,
                         'coverage_90pct': covered, 'mean_interval_width_native_units': width})


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def save_model(fitted, output_dir, config, preprocessing, sources, index, validation, validation_metrics):
    algorithm = fitted['algorithm']
    stem = f"{config['prefix']}_{algorithm}_{config['run_id']}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = '.pt' if fitted['kind'] == 'neural' else '.joblib'
    artifact, metadata_path = output_dir / (stem + extension), output_dir / (stem + '.json')
    if artifact.exists() or metadata_path.exists():
        raise FileExistsError(f'Run already exists: {stem}')
    if fitted['kind'] == 'neural':
        reference = fitted.get('action_reference')
        payload = {'architecture': fitted['architecture'], 'preprocessing': preprocessing,
                   'state_dicts': [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in fitted['models']],
                   'action_reference': (None if reference is None else
                                        {'reference': torch.as_tensor(np.asarray(reference['reference'], np.float32)),
                                         'mean': list(reference['mean']), 'scale': list(reference['scale']),
                                         'k': int(reference['k'])})}
        torch.save(payload, artifact)
    else:
        payload = {k: v for k, v in fitted.items() if k != 'device'}
        OPTIONAL['joblib'].dump(payload, artifact)
    metadata = {
        'schema_version': 'world_model_v5', 'algorithm': algorithm, 'category': fitted['kind'],
        'status': 'completed', 'run_id': config['run_id'], 'artifact': artifact.name,
        'artifact_sha256': file_hash(artifact), 'helper_sha256': file_hash(__file__),
        'inference_contract_sha256': inference_contract_hash(),
        'training_backend': 'indexed_windows_v1', 'training_backend_file': Path(__file__).name,
        'training_backend_sha256': file_hash(__file__), 'protocol_sha256': file_hash(wp.__file__),
        'training_track': config['training_track'], 'quality_diagnostics': config.get('quality_diagnostics', {}),
        'eda_reference': config.get('eda_reference'), 'priority_targets': config.get('priority_targets', []),
        'window_loading': 'on-demand neural histories; shared float32 disk-backed classical matrix; physical boundary rows only',
        'training_pool_windows': len(index),
        'config': config, 'preprocessing': preprocessing, 'source_manifest': sources,
        'training_anchors': index.anchors,
        'validation_anchors': (validation.get('raw', {}).get('anchors', [])
                               if validation_available(validation) else []),
        'validation_available': validation_available(validation),
        'validation_treatment_anchors': (validation or {}).get('treatment_diagnostic', {}).get('raw', {}).get('anchors', []),
        'validation_treatment_metrics': fitted.get('validation_treatment_metrics', []),
        'sampling_interval_seconds': time_settings(config)['sampling_interval_seconds'],
        'history_minutes': config['history_minutes'], 'history_steps': time_settings(config)['history_steps'],
        'observation_operator': time_settings(config)['observation_operator'],
        'horizon_seconds': config['sampling_interval_seconds'], 'horizon_minutes': config['sampling_interval_seconds'] / 60,
        'timestamp_convention': ('sample timestamp; value observed at timestamp' if config['sampling_interval_seconds'] == 5
                                 else 'bin start; means available after bin end'),
        'candidate_action_order': ACTIONS, 'action_semantics': config.get('action_semantics', 'historical next-bin logged exposure; conditional replay, not unknown-action online forecasting'),
        'input_order': feature_order(preprocessing), 'target_order': fitted.get('targets', target_order(preprocessing)),
        'target_transform': preprocessing.get('target_transform', 'delta'),
        'derived_outputs': preprocessing.get('derived_outputs') or {},
        'uncertainty_mode': fitted.get('uncertainty_mode'),
        'action_novelty_reference': (None if fitted.get('action_reference') is None else
                                     {'mean': fitted['action_reference']['mean'],
                                      'scale': fitted['action_reference']['scale'],
                                      'k': fitted['action_reference']['k'],
                                      'reference_rows': int(len(fitted['action_reference']['reference']))}),
        'uncertainty_label': uncertainty_label(fitted.get('uncertainty_mode')),
        'mc_passes': fitted.get('mc_passes'),
        'headline_targets': headline_target_names(preprocessing),
        'macro_NRMSE_headline': validation_metrics.attrs.get('macro_NRMSE_headline'),
        'macro_NRMSE_all': validation_metrics.attrs.get('macro_NRMSE_all'),
        'point_loss_weights': dict(zip(target_order(preprocessing), point_loss_weights(preprocessing).tolist())),
        'point_loss_weight_bounds': preprocessing.get('point_loss_weight_bounds'),
        'label_mode': config.get('label_mode', 'sensor_proxy'),
        'baseline_description': preprocessing.get('baseline_description', 'Persistence'),
        'target_provenance': config.get('target_provenance', 'CSV sensor proxies; calibration/independence unresolved'),
        'outcome_claims': ('Proxy-column forecasting only; no independent biomass/BOD/TSS or treatment-compliance validation'
                           if config.get('label_mode') == 'proxy_prototype' else 'See target provenance'),
        'physical_units': 'units encoded in column headers; pH dimensionless; 1_temp_air(C) is water temperature',
        'selection_rule': fitted['selection_rule'], 'validation_point_metrics': validation_metrics.to_dict('records'),
        'training_recipe': config.get('training_recipe', 'legacy'),
        'selected_hyperparameters': fitted.get('selected_hyperparameters', {}),
        'hyperparameter_trials': fitted.get('hyperparameter_trials', []),
        'selected_checkpoints': fitted.get('selected_checkpoints', []),
        'training_rollout': fitted.get('training_rollout'),
        'checkpoint_metric': (fitted.get('training_rollout') or {}).get('checkpoint_metric'),
        'peak_cuda_allocated_gib': fitted.get('peak_cuda_allocated_gib'),
        'resume_source': fitted.get('resume_source'),
        'training_mode': ('full_state_recovery' if fitted.get('recovery_source') else
                          'warm_start_finetuning' if fitted.get('resume_source') else 'from_scratch'),
        'optimizer_state_restored': fitted.get('optimizer_state_restored', False),
        'recovery_source': fitted.get('recovery_source'),
        'recovery_checkpoint': fitted.get('recovery_checkpoint'),
        'validation_event_response': fitted.get('validation_event_response', []),
        'action_response_gate': fitted.get('action_response_gate', {}),
        'rl_eligible': (fitted.get('action_response_gate') or {}).get('rl_eligible'),
        'uncertainty_discrimination': fitted.get('uncertainty_discrimination', {}),
        'action_coverage_limitations': fitted.get('action_coverage_limitations', []),
        'validation_event_coverage': fitted.get('validation_event_coverage', []),
        'validation_prediction_validity': fitted.get('validation_prediction_validity', []),
        'fit_seconds': fitted['fit_seconds'], 'actual_device': fitted['device'],
        'versions': {name: getattr(module, '__version__', None) for name, module in OPTIONAL.items() if module is not None},
        'training_curve': fitted.get('training_curve', []), 'calibration_runs': fitted.get('calibration_runs', []),
        'limitations': fitted.get('limitations', 'single-seed capped-data prototype; no biological or controller validation'),
    }
    metadata_path.write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False), encoding='utf-8')
    return metadata_path


def load_model(metadata_path, device='cpu'):
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    version = metadata['schema_version']
    current_contract = inference_contract_hash()
    if version != 'world_model_v5':
        raise ValueError('This experimental backend requires world_model_v5 metadata. Start a new run; legacy artifacts are not migrated.')
    contract_changed = metadata['inference_contract_sha256'] != current_contract
    validate_schema(metadata['config'])
    if metadata['preprocessing']['targets'] != metadata['config']['targets']:
        raise ValueError('Saved preprocessing and configuration target order differ.')
    if metadata.get('target_transform') != metadata['preprocessing'].get('target_transform'):
        raise ValueError('Saved target transforms differ.')
    artifact_time_settings(metadata)
    if metadata['input_order'] != feature_order(metadata['preprocessing']) or metadata['candidate_action_order'] != ACTIONS:
        raise ValueError('Feature or action schema mismatch.')
    if (metadata['input_order'] != feature_order(metadata['config']) or
            residual_target_order(metadata['preprocessing']) != residual_target_order(metadata['config'])):
        raise ValueError('Saved input/target transform schemas disagree.')
    expected_targets = metadata['preprocessing'].get('targets', TARGETS)
    if metadata['target_order'] != expected_targets and metadata['category'] != 'physical':
        raise ValueError('Saved target order differs from preprocessing.')
    artifact = metadata_path.parent / metadata['artifact']
    if artifact.stem != metadata_path.stem or file_hash(artifact) != metadata['artifact_sha256']:
        raise ValueError('Artifact name/hash mismatch.')
    missing = missing_dependencies(metadata['algorithm'])
    if missing:
        raise ImportError(f'Missing packages for this artifact: {missing}')
    if contract_changed:
        # Optional TCN-family head variants keep old shared-head and classical
        # inference behavior intact, so existing v5 artifacts remain readable.
        compatible_legacy = metadata['category'] == 'classical'
        if metadata['category'] == 'neural':
            legacy_payload = torch.load(artifact, map_location='cpu', weights_only=True)
            legacy_architecture = legacy_payload.get('architecture', {})
            legacy_algorithm = legacy_architecture.get('algorithm')
            compatible_legacy = (
                legacy_algorithm not in ['tcn', 'pg_ac_tcn'] or
                (legacy_algorithm == 'tcn' and
                 'head_layout' not in legacy_architecture and
                 'latest_sensor_to_head' not in legacy_architecture and
                 'latest_sensor_indices' not in legacy_architecture) or
                (legacy_algorithm == 'pg_ac_tcn' and
                 legacy_architecture.get('head_layout', 'shared') == 'shared')
            )
        if not compatible_legacy:
            raise ValueError('Inference implementation differs from the saved artifact contract.')
    if metadata['category'] == 'neural':
        payload = torch.load(artifact, map_location='cpu', weights_only=True)
        if payload['architecture'].get('outputs', len(TARGETS)) != len(expected_targets):
            raise ValueError('Checkpoint output width differs from saved target schema.')
        if payload['architecture'].get('input_channels', len(FEATURES)) != len(metadata['input_order']):
            raise ValueError('Checkpoint input width differs from saved feature schema.')
        if payload['preprocessing'].get('derived_outputs') != (metadata.get('derived_outputs') or {}):
            raise ValueError('Saved artifact and metadata disagree on derived outputs.')
        reference = payload.get('action_reference')
        if reference is not None:
            fitted_reference = {'reference': np.asarray(reference['reference'], np.float32),
                                'mean': list(reference['mean']), 'scale': list(reference['scale']),
                                'k': int(reference['k'])}
        else:
            fitted_reference = None
        models = []
        for state in payload['state_dicts']:
            model = DynamicsNetwork(payload['architecture']).to(device)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        fitted = {'kind': 'neural', 'algorithm': metadata['algorithm'], 'models': models,
                  'device': device, 'preprocessing': payload['preprocessing'],
                  'action_reference': fitted_reference}
    else:
        fitted = OPTIONAL['joblib'].load(artifact)
        fitted['device'] = 'cpu'
    if fitted['preprocessing'] != metadata['preprocessing']:
        raise ValueError('Preprocessing differs between artifact and metadata.')
    fitted['sampling_interval_seconds'] = metadata.get('sampling_interval_seconds', 60)
    fitted['observation_operator'] = metadata.get('observation_operator', 'interval_mean')
    fitted['inference_batch_size'] = metadata['config'].get('validation_batch_size', 128)
    if fitted.get('kind') == 'neural':
        architecture = (metadata['config'].get('neural', {})
                        .get(metadata['algorithm'], {}))
        fitted['uncertainty_mode'] = architecture.get('uncertainty_mode')
        if fitted['uncertainty_mode'] is None:
            fitted['uncertainty_mode'] = resolved_uncertainty_mode(
                metadata['algorithm'], architecture.get('members', 1))
        fitted['mc_passes'] = architecture.get('mc_passes')
    return fitted, metadata


def rollout_cohort(frames, config, horizons):
    timing = time_settings(config)
    features = feature_order(config)
    maximum = max(duration_steps(m, timing['sampling_interval_seconds']) for m in horizons)
    per_cultivation = max(1, config['rollout_starts'] // len(frames))
    chosen, reports = [], []
    for cultivation, frame in sorted(frames.items()):
        valid = observed_rows(frame, timing['sampling_interval_seconds'], features) & np.isfinite(frame[target_order(config)].to_numpy()).all(axis=1)
        lengths = continuous_run_lengths(valid, frame.timestamp, timing['sampling_interval_seconds'])
        eligible = np.flatnonzero(lengths >= timing['history_steps'] + maximum) - maximum + 1
        selected = wp.spread_sample(eligible, per_cultivation, frame, config['seed']) if len(eligible) else np.array([], int)
        chosen.extend({'cultivation': cultivation, 'index': int(i), 'timestamp': frame.timestamp.iloc[i].isoformat()} for i in selected)
        reports.append({'cultivation': cultivation, **wp.coverage(frame, selected, eligible, ACTIONS)})
    if not chosen:
        raise ValueError('No complete segments for the longest requested rollout duration.')
    return chosen, pd.DataFrame(reports)


def recursive_metrics(fitted, frames, raw_validation, config, horizons):
    targets = target_order(fitted['preprocessing'])
    features = feature_order(fitted['preprocessing'])
    timing = time_settings(config)
    interval, history_steps = timing['sampling_interval_seconds'], timing['history_steps']
    requested = {duration_steps(minutes, interval): float(minutes) for minutes in horizons}
    maximum = max(requested)
    chosen, coverage = rollout_cohort(frames, config, horizons)
    stores = {k: {'features': f[features].to_numpy(np.float32), 'targets': f[targets].to_numpy(np.float32),
                  'actions': f[ACTIONS].to_numpy(np.float32), 'clock': clock_features(f.timestamp, f.timestamp.iloc[0])}
              for k, f in frames.items()}
    history = np.stack([stores[a['cultivation']]['features'][a['index'] - history_steps:a['index']] for a in chosen])
    window = baseline_window_steps(fitted['preprocessing'])
    def history_reference():
        anchor_values = history_anchor(history, window)
        latest = np.full((len(chosen), len(targets)), np.nan, dtype=np.float32)
        for j, name in enumerate(targets):
            if name in features:
                latest[:, j] = anchor_values[:, features.index(name)]
        return latest
    initial = baseline_values({'latest': history_reference()}, fitted['preprocessing'])
    rows, failed_step = [], None
    cumulative = {'nonfinite_predictions': 0, 'negative_nonnegative_target_predictions': 0, 'ph_outside_0_14_predictions': 0}
    iterator = progress(range(maximum), enabled=config.get('progress_bars', False), desc='Recursive forecast steps', unit='step')
    for step in iterator:
        extras, truth = [], []
        for anchor in chosen:
            store = stores[anchor['cultivation']]
            index = anchor['index'] + step
            extras.append(np.r_[store['actions'][index], store['clock'][index]])
            truth.append(store['targets'][index])
        raw = {'sequence': history, 'extra': np.asarray(extras, dtype=np.float32),
               'latest': history_reference(), 'truth': np.asarray(truth),
               'anchors': [{**a, 'index': a['index'] + step} for a in chosen],
               'sampling_interval_seconds': interval, 'observation_operator': timing['observation_operator']}
        columns = fitted.get('targets', targets)
        predicted = (np.full((len(chosen), len(columns)), np.nan) if failed_step is not None else
                     predict(fitted, transform(raw, fitted['preprocessing']))[0])
        checks = wp.prediction_checks(predicted, columns)
        if failed_step is None:
            for key in cumulative:
                cumulative[key] += checks[key]
            if checks['nonfinite_predictions']:
                failed_step = step + 1
        if step + 1 in requested:
            group_masks = {name: wp.cohort_groups(frames, raw['anchors'], name, ACTIONS) for name in columns}
            for group in ['all_readings', 'pump_active', 'actuator_change', 'clock_day', 'clock_night', 'spike_flagged', 'flat_flagged', 'range_flagged', 'no_quality_flag']:
                for j, name in enumerate(columns):
                    mask = group_masks[name].get(group, np.zeros(len(chosen), bool))
                    if not mask.any():
                        continue
                    table, _ = point_metrics(raw['truth'][mask], predicted[mask, j:j+1], initial[mask], fitted['preprocessing'], [name])
                    row = table.iloc[0].to_dict()
                    if failed_step is not None:
                        row['prediction_status'] = 'invalid_rollout_nonfinite'
                    rows.append({'algorithm': fitted['algorithm'], 'horizon_minutes': requested[step + 1],
                                 'horizon_seconds': (step + 1) * interval, 'recursive_steps': step + 1,
                                 'sampling_interval_seconds': interval, 'evaluation_starts': int(mask.sum()),
                                 'group': group, 'failed_at_step': failed_step, **cumulative, **row})
        if failed_step is None:
            next_row = history[:, -1].copy()
            for j, name in enumerate(columns):
                if name in features:
                    next_row[:, features.index(name)] = predicted[:, j]
            next_row[:, -len(ACTIONS):] = raw['extra'][:, :len(ACTIONS)]
            history = np.concatenate([history[:, 1:], next_row[:, None]], axis=1)
    result = pd.DataFrame(rows)
    result.attrs['rollout_anchors'] = chosen
    result.attrs['coverage'] = coverage.to_dict('records')
    return result


# Indexed training and progress

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


class QuietProgress:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, count=1):
        pass

    def set_postfix(self, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def progress(iterable=None, enabled=True, **kwargs):
    if enabled and TQDM_AVAILABLE:
        kwargs.setdefault('mininterval', 2.)
        return tqdm(iterable, **kwargs)
    return QuietProgress(iterable, **kwargs)


class WindowIndex:
    def __init__(self, frames, config, cap=None):
        self.targets = validate_schema(config)
        self.features = feature_order(config)
        self.timing = time_settings(config)
        self.history = self.timing['history_steps']
        self.config = config
        self.stores = {}
        entries, counts = [], []
        rng = np.random.default_rng(config['seed'])
        iterator = progress(sorted(frames.items()), enabled=config.get('progress_bars', True),
                            desc='Index cultivations', unit='cultivation')
        for cultivation, frame in iterator:
            eligible = valid_anchors(frame, self.history, self.timing['sampling_interval_seconds'], self.features)
            target_values = frame[self.targets].to_numpy(dtype=np.float32)
            eligible = eligible[np.isfinite(target_values[eligible]).all(axis=1)]
            selected = wp.training_starts(eligible, cap, frame, {**config, **self.timing}, ACTIONS, sample_blocks, rng)
            values = frame[self.features].to_numpy(dtype=np.float32)
            timestamps = frame.timestamp.to_numpy()
            clock = clock_features(frame.timestamp, frame.timestamp.iloc[0])
            self.stores[cultivation] = {'values': values, 'targets': target_values,
                                        'timestamps': timestamps, 'clock': clock,
                                        'observed': observed_rows(frame, self.timing['sampling_interval_seconds'], self.features)}
            entries.extend((cultivation, int(index)) for index in selected)
            counts.append({'cultivation': cultivation, 'eligible_windows': len(eligible), 'selected_windows': len(selected),
                           **wp.coverage(frame, selected, eligible, ACTIONS)})
        if not entries:
            raise ValueError('No complete windows are available.')
        self.entries = np.asarray(entries, dtype=np.int64)
        self.counts = pd.DataFrame(counts)
        self._anchors = None

    def __len__(self):
        return len(self.entries)

    @property
    def anchors(self):
        if self._anchors is None:
            self._anchors = [{'cultivation': int(k), 'index': int(index),
                              'timestamp': pd.Timestamp(self.stores[int(k)]['timestamps'][index]).isoformat()}
                             for k, index in self.entries]
        return self._anchors

    def raw_batch(self, positions):
        positions = np.asarray(positions, dtype=np.int64)
        count = len(positions)
        sequence = np.empty((count, self.history, len(self.features)), dtype=np.float32)
        extra = np.empty((count, len(ACTIONS) + 3), dtype=np.float32)
        latest = np.full((count, len(self.targets)), np.nan, dtype=np.float32)
        truth = np.empty_like(latest)
        for row, position in enumerate(positions):
            cultivation, index = self.entries[position]
            store = self.stores[int(cultivation)]
            sequence[row] = store['values'][index - self.history:index]
            extra[row] = np.r_[store['values'][index, -len(ACTIONS):], store['clock'][index]]
            truth[row] = store['targets'][index]
            anchor_values = history_anchor(sequence[row], baseline_window_steps(self.config))
            for j, name in enumerate(self.targets):
                if name in self.features:
                    latest[row, j] = anchor_values[self.features.index(name)]
        return {'sequence': sequence, 'features': self.features, 'extra': extra, 'latest': latest, 'truth': truth,
                'sampling_interval_seconds': self.timing['sampling_interval_seconds'],
                'observation_operator': self.timing['observation_operator']}

    def boundary_rows(self):
        count = len(self)
        window = baseline_window_steps(self.config)
        width = max(1, min(window, self.history))
        sequence = np.empty((count, width, len(self.features)), dtype=np.float32)
        extra = np.empty((count, len(ACTIONS) + 3), dtype=np.float32)
        truth = np.empty((count, len(self.targets)), dtype=np.float32)
        latest = np.full_like(truth, np.nan)
        for row, (cultivation, index) in enumerate(self.entries):
            store = self.stores[int(cultivation)]
            sequence[row] = store['values'][index - width:index]
            extra[row] = np.r_[store['values'][index, -len(ACTIONS):], store['clock'][index]]
            truth[row] = store['targets'][index]
            anchor_values = history_anchor(sequence[row], window)
            for j, name in enumerate(self.targets):
                if name in self.features:
                    latest[row, j] = anchor_values[self.features.index(name)]
        return {'sequence': sequence, 'features': self.features, 'extra': extra, 'truth': truth,
                'latest': latest,
                'sampling_interval_seconds': self.timing['sampling_interval_seconds'],
                'observation_operator': self.timing['observation_operator']}


def fit_preprocessing(index, frames):
    weighted_rows = []
    total = 0
    features = index.features
    sums = np.zeros(len(features), dtype=np.float64)
    for cultivation, store in index.stores.items():
        ends = index.entries[index.entries[:, 0] == cultivation, 1]
        changes = np.zeros(len(store['values']) + 1, dtype=np.int64)
        np.add.at(changes, ends - index.history, 1)
        np.add.at(changes, ends, -1)
        weights = np.cumsum(changes[:-1])
        selected = weights > 0
        values, weights = store['values'][selected].astype(np.float64), weights[selected]
        sums += (values * weights[:, None]).sum(axis=0)
        total += int(weights.sum())
        weighted_rows.append((values, weights))
    mean = sums / total
    variance = sum(((values - mean) ** 2 * weights[:, None]).sum(axis=0)
                   for values, weights in weighted_rows) / total
    filter_settings = index.config.get('input_filter', {'mode': 'none'})
    if filter_settings.get('mode', 'none') != 'none':
        total, sums, squares = 0, np.zeros(len(features)), np.zeros(len(features))
        for start in progress(range(0, len(index), 128), enabled=index.config.get('progress_bars', True), desc='Fit causal-history scaler', unit='batch'):
            raw = index.raw_batch(np.arange(start, min(start+128, len(index))))
            values = wp.input_filter(raw['sequence'], filter_settings, features, index.timing['sampling_interval_seconds']).astype(np.float64)
            sums += values.sum(axis=(0,1)); squares += (values**2).sum(axis=(0,1))
            total += values.shape[0] * values.shape[1]
        mean = sums / total
        variance = np.maximum(squares / total - mean**2, 0)
    scale = np.sqrt(variance)
    boundaries = index.boundary_rows()
    mode = index.config.get('target_transform', 'delta')
    baseline = target_baseline(boundaries['latest'], {**index.config, 'targets': index.targets})
    delta = (boundaries['truth'] - baseline).astype(np.float64)
    levels = np.concatenate([frame.loc[observed_rows(frame, index.timing['sampling_interval_seconds'], features), index.targets].to_numpy(float)
                             for frame in frames.values()])
    levels = levels[np.isfinite(levels).all(axis=1)]
    delta_scale = delta.std(axis=0)
    return {'input_filter': filter_settings, 'features': features, 'targets': index.targets, 'actions': ACTIONS, 'context': CONTEXT,
            'target_transform': mode, 'residual_targets': residual_target_order(index.config),
            'derived_outputs': derived_output_specs(index.config),
            'headline_targets': list(index.config.get('headline_targets') or []),
            'point_loss_weight_bounds': (list(index.config['point_loss_weight_bounds'])
                                         if index.config.get('point_loss_weight_bounds') is not None else None),
            'training_target_mean': boundaries['truth'].astype(np.float64).mean(axis=0).tolist(),
            'baseline_window_steps': baseline_window_steps(index.config),
            'baseline_description': (
                ('Trailing-mean anchor for targets in input history; training mean for output-only targets'
                 if baseline_window_steps(index.config) > 1 else
                 'Persistence for targets in input history; training mean for output-only targets')
                if mode in ['level', 'mixed'] else
                ('Trailing-mean anchor' if baseline_window_steps(index.config) > 1 else 'Persistence')),
            **index.timing,
            'input_mean': mean.tolist(), 'input_scale': np.where(scale > 1e-12, scale, 1).tolist(),
            'delta_mean': delta.mean(axis=0).tolist(), 'delta_scale': np.where(delta_scale > 1e-12, delta_scale, 1).tolist(),
            'training_level_std': levels.std(axis=0).tolist(),
            'fit_scope': 'selected training histories only, using exact history-row multiplicities without expanding windows; deltas from selected anchors; level std from complete training bins',
            'extra_order': ACTIONS + ['clock_sin', 'clock_cos', 'elapsed_since_first_record_days_divided_by_30']}


if torch is not None:
    class LazyWindows(torch.utils.data.Dataset):
        def __init__(self, index, preprocessing, positions=None):
            self.index = index
            self.positions = np.arange(len(index)) if positions is None else np.asarray(positions)
            self.mean = np.array(preprocessing['input_mean'])
            self.scale = np.array(preprocessing['input_scale'])
            self.delta_mean = np.array(preprocessing['delta_mean'])
            self.delta_scale = np.array(preprocessing['delta_scale'])
            self.filter_settings = preprocessing.get('input_filter', {'mode': 'none'})

        def __len__(self):
            return len(self.positions)

        def __getitem__(self, position):
            cultivation, end = self.index.entries[self.positions[position]]
            store = self.index.stores[int(cultivation)]
            values = store['values']
            raw_sequence = values[None, end - self.index.history:end]
            filtered = wp.input_filter(raw_sequence, self.filter_settings, self.index.features, self.index.timing['sampling_interval_seconds'])[0]
            sequence = ((filtered - self.mean) / self.scale).astype(np.float32)
            extra = np.r_[values[end, -len(ACTIONS):], store['clock'][end]].astype(np.float32)
            extra[:len(ACTIONS)] = (extra[:len(ACTIONS)] - self.mean[-len(ACTIONS):]) / self.scale[-len(ACTIONS):]
            delta = store['targets'][end].copy()
            mode = self.index.config.get('target_transform', 'delta')
            if mode in ['delta', 'mixed']:
                window = baseline_window_steps(self.index.config)
                anchor_values = history_anchor(values[end - window:end], window)
                for j, name in enumerate(self.index.targets):
                    if name in residual_target_order(self.index.config):
                        delta[j] -= anchor_values[self.index.features.index(name)]
            target = ((delta - self.delta_mean) / self.delta_scale).astype(np.float32)
            return torch.from_numpy(sequence), torch.from_numpy(extra), torch.from_numpy(target)


def load_resume_checkpoint(metadata_path, algorithm, config, preprocessing):
    """Load compatible neural weights for warm-start fine-tuning, not optimizer resume."""
    if torch is None:
        raise ImportError('PyTorch is required to continue neural training.')
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Resume metadata does not exist: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 'world_model_v5':
        raise ValueError('Warm start requires a new-schema world_model_v5 artifact.')
    if metadata.get('status') != 'completed' or metadata.get('category') != 'neural':
        raise ValueError('Resume source must be a completed neural-model artifact.')
    if metadata.get('algorithm') != algorithm:
        raise ValueError(f'Resume algorithm {metadata.get("algorithm")} does not match {algorithm}.')

    source_config = metadata.get('config', {})
    source_preprocessing = metadata.get('preprocessing', {})
    source_targets = metadata.get('target_order', [])
    expected_targets = target_order(config)
    saved_target_transform = (metadata.get('target_transform') or source_preprocessing.get('target_transform')
                             or source_config.get('target_transform', 'delta'))
    contract_fields = {
        'targets': (source_targets, expected_targets),
        'input features': (metadata.get('input_order'), feature_order(config)),
        'candidate actions': (metadata.get('candidate_action_order'), ACTIONS),
        'context channels': (source_config.get('context'), CONTEXT),
        'sampling interval': (metadata.get('sampling_interval_seconds'), time_settings(config)['sampling_interval_seconds']),
        'history duration': (metadata.get('history_minutes'), config['history_minutes']),
        'target transform': (saved_target_transform, config.get('target_transform', 'delta')),
    }
    mismatches = [name for name, (saved, current) in contract_fields.items() if saved != current]
    if mismatches:
        saved_detail = ', '.join(f'{name}={contract_fields[name][0]!r}' for name in mismatches)
        current_detail = ', '.join(f'{name}={contract_fields[name][1]!r}' for name in mismatches)
        raise ValueError('Resume checkpoint is incompatible (' + '; '.join(mismatches) + '). '
                         f'Saved: {saved_detail}. Current: {current_detail}.')

    artifact = metadata_path.parent / metadata.get('artifact', '')
    if not artifact.is_file() or artifact.stem != metadata_path.stem:
        raise FileNotFoundError(f'Resume checkpoint artifact is missing or mismatched: {artifact}')
    if file_hash(artifact) != metadata.get('artifact_sha256'):
        raise ValueError('Resume checkpoint SHA-256 does not match its metadata.')
    payload = torch.load(artifact, map_location='cpu', weights_only=True)
    expected_architecture = make_architecture(algorithm, config)
    saved_architecture = payload.get('architecture', {})
    architecture_keys = ('algorithm', 'history_minutes', 'history_steps', 'sampling_interval_seconds',
                         'hidden_size', 'dilations', 'head_dropout', 'input_channels', 'outputs',
                         'log_variance_bounds', 'head_layout', 'latest_sensor_to_head',
                         'latest_sensor_names', 'latest_sensor_indices', 'head_groups',
                         'attention_heads', 'latent_size', 'moving_average_kernel', 'pool_scales',
                         'sensor_channels', 'action_channels', 'mc_dropout', 'uncertainty_mode')
    legacy_architecture_defaults = {'head_layout': 'shared', 'latest_sensor_to_head': False,
                                    'latest_sensor_names': [], 'latest_sensor_indices': [],
                                    'attention_heads': 4, 'latent_size': 16, 'moving_average_kernel': 25,
                                    'mc_dropout': 0., 'uncertainty_mode': None,
                                    'pool_scales': [1, 4, 16],
                                    'sensor_channels': expected_architecture['sensor_channels'],
                                    'action_channels': expected_architecture['action_channels']}
    architecture_mismatch = [key for key in architecture_keys
                              if saved_architecture.get(key, legacy_architecture_defaults.get(key)) != expected_architecture.get(key)
                              and not (key == 'head_groups' and key not in saved_architecture)]
    if architecture_mismatch:
        raise ValueError('Resume checkpoint architecture differs from the current configuration.')

    for key in ('features', 'targets', 'actions', 'context',
                'sampling_interval_seconds', 'history_steps', 'observation_operator',
                'derived_outputs', 'point_loss_weight_bounds', 'baseline_window_steps'):
        saved = source_preprocessing.get(key)
        current = preprocessing.get(key)
        if saved != current:
            raise ValueError(f'Resume preprocessing mismatch for {key}: saved={saved!r}, current={current!r}.')
    if source_preprocessing.get('input_filter', {'mode': 'none'}) != preprocessing.get('input_filter', {'mode': 'none'}):
        raise ValueError('Resume input-filter recipe differs.')
    current_target_transform = preprocessing.get('target_transform', config.get('target_transform', 'delta'))
    if current_target_transform != saved_target_transform:
        raise ValueError('Resume preprocessing mismatch for target_transform.')
    if residual_target_order(source_preprocessing) != residual_target_order(preprocessing):
        raise ValueError('Resume residual/level target assignments differ.')
    scaler_keys = ['input_mean', 'input_scale', 'delta_mean', 'delta_scale']
    if saved_target_transform in ['level', 'mixed']:
        scaler_keys.append('training_target_mean')
    for key in scaler_keys:
        if key not in source_preprocessing and key not in preprocessing:
            continue
        if key not in source_preprocessing or key not in preprocessing:
            raise ValueError(f'Resume preprocessing is missing fitted scaler {key} on one side.')
        saved = np.asarray(source_preprocessing.get(key), dtype=np.float64)
        current = np.asarray(preprocessing.get(key), dtype=np.float64)
        if saved.shape != current.shape or not np.allclose(saved, current, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Resume preprocessing mismatch for fitted scaler {key}.')

    settings = config['neural'][algorithm]
    states = payload.get('state_dicts', [])
    expected_members = settings.get('members', 1)
    if len(states) != expected_members:
        raise ValueError(f'Resume checkpoint has {len(states)} member(s); current configuration expects {expected_members}.')
    for state in states:
        probe = DynamicsNetwork(expected_architecture)
        probe.load_state_dict(state, strict=True)
    source = {'metadata': str(metadata_path), 'run_id': metadata.get('run_id'),
              'artifact': artifact.name, 'artifact_sha256': metadata['artifact_sha256'],
              'optimizer_state_restored': False}
    return states, source


def initialization_strategy(settings, target_transform):
    choice = settings.get('initialization')
    if choice is not None:
        if 'persistence_initialization' in settings or 'training_mean_initialization' in settings:
            raise ValueError('Use one initialization choice instead of mixing it with legacy switches.')
        if choice not in ['reference', 'random']:
            raise ValueError('Initialization must be reference or random.')
        strategy = ('training_mean' if target_transform == 'level' else 'persistence') if choice == 'reference' else 'random'
    else:
        # Old saved configurations retain their original switch precedence.
        strategy = ('persistence' if settings.get('persistence_initialization', False) else
                    'training_mean' if settings.get('training_mean_initialization', False) else 'random')
    if strategy == 'persistence' and target_transform not in ['delta', 'mixed']:
        raise ValueError('Persistence initialization requires delta or mixed targets.')
    if strategy == 'training_mean' and target_transform != 'level':
        raise ValueError('Training-mean initialization requires level targets.')
    return strategy


def treatment_validation_diagnostic(fitted, validation):
    prepared = (validation or {}).get('treatment_diagnostic')
    if prepared is None:
        return None, {}
    predicted, _ = predict(fitted, prepared)
    columns = fitted.get('targets', target_order(fitted['preprocessing']))
    table, score = point_metrics(prepared['raw']['truth'], predicted, prepared['raw']['latest'], fitted['preprocessing'], columns)
    priorities = [name for name in wp.PRIORITIES if name in columns]
    priority = wp.priority_summary(table, priorities) if priorities else {}
    return table, {'validation_treatment_macro_NRMSE': score,
                   'validation_treatment_priority_macro_NRMSE': priority.get('priority_macro_NRMSE', np.nan),
                   'validation_treatment_starts': len(prepared['y'])}


def rollout_settings(settings):
    steps = settings.get('rollout_steps', 1)
    curriculum = settings.get('rollout_curriculum', [])
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('rollout_steps must be a positive integer.')
    previous_epoch = 0
    for stage in curriculum:
        epoch, count = stage['start_epoch'], stage['steps']
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (epoch, count)):
            raise ValueError('Rollout curriculum epochs and steps must be integers.')
        if epoch <= previous_epoch or count < 1 or count > steps:
            raise ValueError('Rollout curriculum epochs must increase and steps must be within rollout_steps.')
        previous_epoch = epoch
    if curriculum and (curriculum[0]['start_epoch'] != 1 or curriculum[-1]['steps'] != steps
                       or previous_epoch > settings['epochs']):
        raise ValueError('Curriculum must start at epoch 1 and reach rollout_steps within the epoch limit.')
    discount = float(settings.get('rollout_discount', 1.))
    if not np.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError('rollout_discount must be in (0, 1].')
    checkpointing = settings.get('rollout_checkpointing', True)
    if not isinstance(checkpointing, bool):
        raise ValueError('rollout_checkpointing must be a boolean.')
    return {'steps': int(steps), 'curriculum': curriculum, 'discount': discount,
            'activation_checkpointing': checkpointing}


def rollout_steps_at_epoch(recipe, epoch):
    steps = recipe['steps'] if not recipe['curriculum'] else 1
    for stage in recipe['curriculum']:
        if epoch >= stage['start_epoch']:
            steps = stage['steps']
    return steps


def rollout_training_positions(index, steps):
    """Keep selected starts with a complete future segment in the same continuous episode."""
    valid_positions = np.zeros(len(index), dtype=bool)
    for cultivation, store in index.stores.items():
        positions = np.flatnonzero(index.entries[:, 0] == cultivation)
        starts = index.entries[positions, 1]
        valid = store['observed'] & np.isfinite(store['targets']).all(axis=1)
        lengths = continuous_run_lengths(valid, store['timestamps'], index.timing['sampling_interval_seconds'])
        ends = starts + steps - 1
        inside = ends < len(lengths)
        valid_positions[positions[inside]] = lengths[ends[inside]] >= steps
    return np.flatnonzero(valid_positions)


if torch is not None:
    class LazyRolloutWindows(torch.utils.data.Dataset):
        def __init__(self, index, positions, steps):
            self.index, self.positions, self.steps = index, np.asarray(positions), steps

        def __len__(self):
            return len(self.positions)

        def __getitem__(self, position):
            cultivation, start = self.index.entries[self.positions[position]]
            store = self.index.stores[int(cultivation)]
            stop = start + self.steps
            history = store['values'][start - self.index.history:start].copy()
            extra = np.concatenate([store['values'][start:stop, -len(ACTIONS):],
                                    store['clock'][start:stop]], axis=1).astype(np.float32)
            truth = store['targets'][start:stop].copy()
            return tuple(torch.from_numpy(value) for value in (history, extra, truth))


def torch_history_filter(history, preprocessing):
    features = feature_order(preprocessing)
    spec = preprocessing.get('input_filter', {'mode': 'none'})
    if spec.get('mode', 'none') == 'none':
        return history
    # Use the same prefix medians as evaluation, retaining gradients through feedback.
    wp.input_filter(np.zeros((1, 1, len(features))), spec, features,
                    preprocessing['sampling_interval_seconds'])
    width = min(history.shape[1], duration_steps(spec['window_seconds'] / 60,
                                               preprocessing['sampling_interval_seconds']))
    columns = [features.index(name) for name in spec['features']]
    if not columns:
        return history
    values = history[:, :, columns]
    prefix = [values[:, :end].quantile(.5, dim=1) for end in range(1, width)]
    full = values.unfold(1, width, 1).quantile(.5, dim=-1)
    filtered = torch.cat([torch.stack(prefix, dim=1), full], dim=1) if prefix else full
    return history.index_copy(2, torch.as_tensor(columns, device=history.device), filtered)


def neural_rollout_loss(model, history, extras, truth, preprocessing, settings):
    """Differentiate through mean-feedback rollouts; future observations are labels only."""
    recipe = rollout_settings(settings)
    if not 1 <= truth.shape[1] <= recipe['steps'] or extras.shape[1] != truth.shape[1]:
        raise ValueError('Rollout actions/clock and labels must share a positive length within rollout_steps.')
    tensor = lambda value: torch.as_tensor(value, dtype=history.dtype, device=history.device)
    input_mean, input_scale = tensor(preprocessing['input_mean']), tensor(preprocessing['input_scale'])
    delta_mean, delta_scale = tensor(preprocessing['delta_mean']), tensor(preprocessing['delta_scale'])
    weights = tensor(point_loss_weights(preprocessing))
    targets = target_order(preprocessing)
    features = feature_order(preprocessing)
    mode = preprocessing.get('target_transform', 'delta')
    mapped = [(j, features.index(name)) for j, name in enumerate(targets) if name in features]
    feature_columns = torch.as_tensor([col for _, col in mapped], device=history.device, dtype=torch.long)
    target_columns = torch.as_tensor([j for j, _ in mapped], device=history.device, dtype=torch.long)
    residuals = [(j, features.index(name)) for j, name in enumerate(targets) if name in residual_target_order(preprocessing)]
    residual_outputs = torch.as_tensor([j for j, _ in residuals], device=history.device, dtype=torch.long)
    residual_features = torch.as_tensor([col for _, col in residuals], device=history.device, dtype=torch.long)
    derived = derived_target_indices(preprocessing)
    derived_weight = tensor(derived_loss_mask(preprocessing))
    derived_count = float(derived_loss_mask(preprocessing).sum())
    derived_output = torch.as_tensor([o for o, _, _, _ in derived], device=history.device, dtype=torch.long)
    derived_source = torch.as_tensor([s for _, s, _, _ in derived], device=history.device, dtype=torch.long)
    derived_scale = tensor([s for _, _, s, _ in derived])
    derived_offset = tensor([o for _, _, _, o in derived])
    nll_sum = point_sum = physics_sum = 0.
    normalizer = 0.
    for step in range(truth.shape[1]):
        sequence = (torch_history_filter(history, preprocessing) - input_mean) / input_scale
        extra = torch.cat([(extras[:, step, :len(ACTIONS)] - input_mean[-len(ACTIONS):]) /
                           input_scale[-len(ACTIONS):], extras[:, step, len(ACTIONS):]], dim=1)
        if recipe['activation_checkpointing'] and model.training and torch.is_grad_enabled():
            mean, log_variance = activation_checkpoint(model, sequence, extra, use_reentrant=False)
        else:
            mean, log_variance = model(sequence, extra)
        baseline = torch.zeros_like(mean)
        if mode in ['delta', 'mixed']:
            # Anchor on the same trailing window the targets were built with.
            window = baseline_window_steps(preprocessing)
            anchor_values = history_anchor(history, window).index_select(1, residual_features)
            baseline = baseline.index_copy(1, residual_outputs, anchor_values)
        predicted = baseline + mean * delta_scale + delta_mean
        if derived:
            # Derived values follow their source column; the network column is not fitted.
            predicted = predicted.index_copy(1, derived_output,
                                             derived_scale * predicted[:, derived_source] + derived_offset)
        target_levels = truth[:, step]
        if derived:
            target_levels = target_levels.index_copy(1, derived_output,
                                                     derived_scale * target_levels[:, derived_source] + derived_offset)
        scaled_error = mean - (target_levels - baseline - delta_mean) / delta_scale
        nll = .5 * (math.log(2 * math.pi) + log_variance + scaled_error.square() * torch.exp(-log_variance))
        weight = recipe['discount'] ** step
        nll_sum = nll_sum + weight * (nll * derived_weight).sum(dim=1).mean() / derived_count
        point_sum = point_sum + weight * (scaled_error * weights * derived_weight).square().sum(dim=1).mean() / derived_count
        physics_sum = physics_sum + weight * physics_bound_penalty(mean, sequence, preprocessing, settings)
        normalizer += weight
        if step + 1 < truth.shape[1]:
            next_row = history[:, -1].index_copy(1, feature_columns, predicted[:, target_columns])
            next_row = torch.cat([next_row[:, :-len(ACTIONS)], extras[:, step, :len(ACTIONS)]], dim=1)
            history = torch.cat([history[:, 1:], next_row[:, None]], dim=1)
    nll, point_loss = nll_sum / normalizer, point_sum / normalizer
    loss = nll if settings.get('loss', 'gaussian_nll') == 'gaussian_nll' else point_loss + settings['nll_weight'] * nll
    loss = loss + settings.get('physics_loss_weight', 0.) * physics_sum / normalizer
    return loss, nll, point_loss


def train_neural(algorithm, index, validation, preprocessing, config, device,
                 resume_state_dicts=None, resume_source=None, validation_frames=None):
    torch.set_num_threads(config['cpu_threads'])
    settings = config['neural'][algorithm]
    architecture = make_architecture(algorithm, config)
    models, curves, checkpoints = [], [], []
    start = time.perf_counter()
    recipe = rollout_settings(settings)
    eligible_positions = (rollout_training_positions(index, recipe['steps']) if recipe['steps'] > 1
                          else np.arange(len(index)))
    if not len(eligible_positions):
        raise ValueError('No selected training starts support the complete training rollout.')
    excluded = np.setdiff1d(np.arange(len(index)), eligible_positions)
    rollout_report = {**recipe, 'horizon_seconds': recipe['steps'] * index.timing['sampling_interval_seconds'],
                      'selected_pool_windows': len(index), 'eligible_windows': len(eligible_positions),
                      'excluded_windows': len(excluded), 'excluded_pool_positions': excluded.tolist(),
                      'feedback': 'predicted means; full gradient through all steps; no teacher forcing',
                      'context': 'persist unpredicted channels; replay logged actions and deterministic clock',
                      'loss_aggregation': 'normalized discounted mean across all active rollout steps',
                      'checkpoint_metric': ('one-step validation macro NRMSE' if recipe['steps'] == 1 else
                                            f"{recipe['steps']}-step recursive validation macro NRMSE")}
    # Selection follows the training rollout automatically: at rollout k the checkpoint is
    # scored on a k-step recursive validation, so early stopping stops preferring
    # one-step accuracy that multi-step training deliberately trades away.
    selection_cohort = None
    if recipe['steps'] > 1 and validation_frames and validation_available(validation):
        selection_cohort = rollout_validation_cohort(validation_frames, preprocessing, config,
                                                     recipe['steps'])
        rollout_report['checkpoint_metric_starts'] = len(selection_cohort['anchors'])
        print(f"{algorithm}: checkpoint selection uses {recipe['steps']}-step recursive validation "
              f"over {len(selection_cohort['anchors']):,} starts.", flush=True)
    print(f"{algorithm}: rollout maximum {recipe['steps']} step(s) / {rollout_report['horizon_seconds']} seconds; "
          f'{len(eligible_positions):,}/{len(index):,} selected starts eligible; {len(excluded):,} excluded.')
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    loss_name = settings.get('loss', 'gaussian_nll')
    if loss_name not in ['gaussian_nll', 'normalized_mse_plus_nll']:
        raise ValueError(f'Unknown neural objective: {loss_name}')
    initialization = initialization_strategy(settings, preprocessing.get('target_transform', 'delta'))
    physics_weight = float(settings.get('physics_loss_weight', 0.))
    physics_bounds = settings.get('physics_bounds', {})
    if not np.isfinite(physics_weight) or physics_weight < 0:
        raise ValueError('physics_loss_weight must be finite and nonnegative.')
    if physics_weight > 0 and not physics_bounds:
        raise ValueError('A nonzero physics_loss_weight requires explicit validated physics_bounds.')
    for name, bounds in physics_bounds.items():
        if name not in target_order(preprocessing) or not any(bounds.get(key) is not None for key in ('minimum', 'maximum')):
            raise ValueError(f'Invalid physics bounds for target {name!r}.')
        for key in ('minimum', 'maximum'):
            value = bounds.get(key)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f'Physics bound {key} for {name!r} must be finite.')
        if (bounds.get('minimum') is not None and bounds.get('maximum') is not None and
                float(bounds['minimum']) > float(bounds['maximum'])):
            raise ValueError(f'Physics minimum exceeds maximum for {name!r}.')
    weights = torch.as_tensor(point_loss_weights(preprocessing), device=device)
    policy = selection_policy(settings, validation)
    validate_only = policy['mode'] == 'fixed_epochs'
    if validate_only:
        print(f'{algorithm}: selection=fixed_epochs ({policy["fixed_epochs"]} epochs), '
              'no held-out checkpoint selection; validation metrics are monitoring only.', flush=True)
    recovery_options = config.get('recovery', {})
    recovery_enabled = recovery_options.get('enabled', False)
    recovery_input = recovery_options.get('resume_by_algorithm', {}).get(algorithm)
    if recovery_input and resume_state_dicts is not None:
        raise ValueError('Choose full-state recovery or weight-only warm start, not both.')
    if recovery_input and not recovery_enabled:
        raise ValueError('Enable recovery before configuring a recovery source.')
    if recovery_enabled and config['loader_workers'] != 0:
        raise ValueError('Epoch recovery currently requires loader_workers=0 for reproducible loading.')
    every = recovery_options.get('every_epochs', 1)
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError('Recovery every_epochs must be a positive integer.')
    recovery_path = None
    loaded_recovery = None
    completed_states = []
    prior_seconds = 0.
    prior_peak_cuda = 0.
    if recovery_enabled:
        recovery_path = Path(recovery_options['directory']) / f"{config['prefix']}_{algorithm}_{config['run_id']}.recovery.pt"
        if recovery_path.exists() and (not recovery_input or recovery_path.resolve() != Path(recovery_input).resolve()):
            raise FileExistsError(f'Recovery destination already exists: {recovery_path}')
        identity = recovery.recovery_identity(sys.modules[__name__], algorithm, index, validation,
                                              preprocessing, config, device)
        if recovery_input:
            loaded_recovery = recovery.load(torch, recovery_input, identity)
            completed_states = loaded_recovery['completed_states']
            curves = loaded_recovery['training_curve']
            checkpoints = loaded_recovery['selected_checkpoints']
            prior_seconds = loaded_recovery['fit_seconds']
            prior_peak_cuda = loaded_recovery.get('peak_cuda_allocated_gib') or 0.
            resume_source = loaded_recovery.get('warm_start_source')
            resume_state_dicts = loaded_recovery.get('warm_start_states')
            print(f'{algorithm}: recovering after the last saved epoch; unfinished epoch work is replayed.', flush=True)
        print(f'{algorithm}: full-state recovery file: {recovery_path}', flush=True)

    resuming = resume_state_dicts is not None or bool(resume_source)

    def save_recovery(active=None):
        if recovery_enabled:
            recovery.atomic_save(torch, {'schema': 'world_model_recovery_v1', 'identity': identity,
                'origin_run_id': config['run_id'], 'warm_start_source': resume_source,
                'warm_start_states': resume_state_dicts,
                'completed_states': completed_states, 'training_curve': curves,
                'selected_checkpoints': checkpoints, 'active': active,
                'fit_seconds': prior_seconds + time.perf_counter() - start,
                'peak_cuda_allocated_gib': (max(prior_peak_cuda, torch.cuda.max_memory_allocated(device) / 1024 ** 3)
                                            if str(device).startswith('cuda') else None),
                'status': ('training' if active is not None else
                           'training_complete' if len(completed_states) == settings.get('members', 1) else 'between_members')}, recovery_path)

    for member in range(settings.get('members', 1)):
        if member < len(completed_states):
            model = DynamicsNetwork(architecture).to(device)
            model.load_state_dict(completed_states[member])
            models.append(model.eval())
            continue
        seed = config['seed'] + member
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        if recipe['steps'] > 1:
            positions = eligible_positions
            if settings.get('members', 1) > 1:
                anchors = [index.anchors[int(position)] for position in eligible_positions]
                positions = positions[bootstrap_indices(anchors, time_settings(config)['sampling_block_steps'], rng)]
            dataset = LazyRolloutWindows(index, positions, recipe['steps'])
        else:
            positions = bootstrap_indices(index.anchors, time_settings(config)['sampling_block_steps'], rng) if settings.get('members', 1) > 1 else None
            dataset = LazyWindows(index, preprocessing, positions)
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True,
                  num_workers=config['loader_workers'], pin_memory=device.startswith('cuda') and config['pin_memory'],
                  generator=generator)
        model = DynamicsNetwork(architecture).to(device)
        resuming = resume_state_dicts is not None
        if resuming:
            if len(resume_state_dicts) != settings.get('members', 1):
                raise ValueError('Resume checkpoint member count differs from the configured model.')
            model.load_state_dict(resume_state_dicts[member], strict=True)
        elif initialization == 'persistence':
            initialize_persistence_head(model, preprocessing)
        elif initialization == 'training_mean':
            initialize_output_heads(model, preprocessing, persistence=False)
        optimizer_name = settings.get('optimizer', 'adam')
        if optimizer_name not in ['adam', 'adamw']:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')
        optimizer_class = torch.optim.AdamW if optimizer_name == 'adamw' else torch.optim.Adam
        optimizer = optimizer_class(model.parameters(), lr=settings['learning_rate'], weight_decay=settings.get('weight_decay', 0.))
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                       factor=settings['scheduler_factor'], patience=settings['scheduler_patience'],
                       threshold=settings.get('min_delta', 0.), threshold_mode='abs', min_lr=settings['min_learning_rate'])
                     if settings.get('lr_scheduler', False) else None)
        best, state, stale, best_epoch = np.inf, None, 0, None
        if (not validate_only and not (loaded_recovery and loaded_recovery.get('active'))
                and (resuming or settings.get('include_initial_checkpoint', False))):
            # Monitoring stays deterministic so early stopping is not MC-noisy.
            if selection_cohort is not None:
                _, best = score_rollout(model, selection_cohort, preprocessing, recipe['steps'],
                                        device, config['validation_batch_size'])
                event_diagnostic = {}
            else:
                means, _ = neural_components([model], validation, device, config['validation_batch_size'])
                prediction = restore_levels(means[0], validation['raw']['latest'], preprocessing)
                _, best = point_metrics(validation['raw']['truth'], prediction,
                                        validation['raw']['latest'], preprocessing)
                _, event_diagnostic = treatment_validation_diagnostic(
                    {'kind': 'neural', 'models': [model], 'device': device,
                     'preprocessing': preprocessing,
                     'inference_batch_size': config['validation_batch_size']}, validation)
            if not np.isfinite(best):
                raise FloatingPointError('Non-finite initial validation score.')
            best_epoch = 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            curves.append({'member': member, 'epoch': 0, 'phase': 'warm_start' if resuming else 'initialization',
                           'training_gaussian_NLL_scaled_delta': None, 'training_objective': None,
                           'training_normalized_point_loss': None, 'validation_macro_NRMSE': best,
                           'learning_rate': settings['learning_rate'], 'rollout_steps': 0, **event_diagnostic})
        epoch, seen = 0, 0
        active_recovery = loaded_recovery.get('active') if loaded_recovery else None
        if active_recovery is not None:
            if active_recovery['member'] != member:
                raise ValueError('Recovery member order is inconsistent.')
            if not np.array_equal(active_recovery['positions'], positions):
                raise ValueError('Recovery bootstrap/training positions changed.')
            model.load_state_dict(active_recovery['model'])
            optimizer.load_state_dict(active_recovery['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(active_recovery['scheduler'])
            best, best_epoch = active_recovery['best'], active_recovery['best_epoch']
            stale, state = active_recovery['stale'], active_recovery['best_state']
            epoch, seen = active_recovery['epoch'], active_recovery['seen']
            curves = loaded_recovery['training_curve'][:active_recovery['curve_count']]
            recovery.restore_rng(torch, generator, active_recovery['rng'])
            loaded_recovery = None

        def snapshot():
            return {'member': member, 'epoch': epoch, 'seen': seen, 'positions': positions,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'best': best, 'best_epoch': best_epoch, 'stale': stale, 'best_state': state,
                    'rng': recovery.capture_rng(torch, generator), 'curve_count': len(curves),
                    'amp_scaler': None}

        if active_recovery is None:
            save_recovery(snapshot())
        first_epoch = epoch + 1 if stale < settings['patience'] else settings['epochs'] + 1
        epoch_progress = progress(range(first_epoch, settings['epochs'] + 1), enabled=config['progress_bars'],
                                  desc=f'{algorithm} member {member + 1}: epochs', unit='epoch')
        epoch_durations = {}
        for epoch in epoch_progress:
            epoch_started = time.perf_counter()
            model.train()
            loss_sum, nll_sum, point_sum, seen = 0., 0., 0., 0
            epoch_lr = optimizer.param_groups[0]['lr']
            active_steps = rollout_steps_at_epoch(recipe, epoch)
            if not config['progress_bars']:
                previous_duration = epoch_durations.get(active_steps)
                estimate = (f'~{previous_duration / 60:.1f} min (previous epoch; includes validation/checkpoint)'
                            if previous_duration is not None else
                            'pending (first epoch at this rollout length)')
                print(f"{algorithm} member {member + 1} | Starting epoch {epoch}/{settings['epochs']}"
                      f' | rollout: {active_steps} step(s) | learning rate: {epoch_lr:.2g}'
                      f' | estimated epoch time: {estimate}', flush=True)
            batches = progress(loader, enabled=config['progress_bars'], total=len(loader),
                               desc=f'Epoch {epoch}: batches', unit='batch', leave=False)
            for sequence, extra, target in batches:
                sequence = sequence.to(device, non_blocking=True)
                extra, target = extra.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if recipe['steps'] > 1:
                    loss, nll, point_loss = neural_rollout_loss(
                        model, sequence, extra[:, :active_steps], target[:, :active_steps], preprocessing, settings)
                else:
                    mean, log_variance = model(sequence, extra)
                    single_weight = torch.as_tensor(derived_loss_mask(preprocessing), device=device)
                    single_count = float(single_weight.sum())
                    nll = .5 * (math.log(2 * math.pi) + log_variance + (target - mean).square() * torch.exp(-log_variance))
                    nll = (nll * single_weight).sum(dim=1).mean() / single_count
                    point_loss = ((mean - target) * weights * single_weight).square().sum(dim=1).mean() / single_count
                    loss = nll if loss_name == 'gaussian_nll' else point_loss + settings['nll_weight'] * nll
                    loss = loss + settings.get('physics_loss_weight', 0.) * physics_bound_penalty(
                        mean, sequence, preprocessing, settings)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite neural loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get('gradient_clip_norm', 5.))
                optimizer.step()
                count = len(target)
                loss_sum += loss.item() * count
                nll_sum += nll.item() * count
                point_sum += point_loss.item() * count
                seen += count
                batches.set_postfix(loss=f'{loss_sum / seen:.4f}', NLL=f'{nll_sum / seen:.4f}',
                                    rollout_steps=active_steps, refresh=False)
            batches.close()
            score, event_diagnostic = np.nan, {}
            if validation_available(validation):
                if selection_cohort is not None:
                    # Score the same recursive objective the epoch was trained on.
                    # With a curriculum the cohort carries the maximum length, and each
                    # epoch is scored at its own stage length.
                    _, score = score_rollout(model, selection_cohort, preprocessing,
                                             active_steps, device,
                                             config['validation_batch_size'])
                    event_diagnostic = {}
                else:
                    means, _ = neural_components([model], validation, device, config['validation_batch_size'])
                    prediction = restore_levels(means[0], validation['raw']['latest'], preprocessing)
                    _, score = point_metrics(validation['raw']['truth'], prediction,
                                             validation['raw']['latest'], preprocessing)
                    _, event_diagnostic = treatment_validation_diagnostic(
                        {'kind': 'neural', 'models': [model], 'device': device,
                         'preprocessing': preprocessing,
                         'inference_batch_size': config['validation_batch_size']}, validation)
            curves.append({'member': member, 'epoch': epoch, 'phase': 'training',
                           'training_gaussian_NLL_scaled_delta': nll_sum / seen, 'training_objective': loss_sum / seen,
                           'training_normalized_point_loss': point_sum / seen,
                           'validation_macro_NRMSE': None if validate_only else score,
                           'learning_rate': epoch_lr, **event_diagnostic})
            curves[-1]['rollout_steps'] = active_steps
            if validate_only:
                # Keep the final epoch: the budget replaced early stopping, so nothing
                # to rank here and the curve stays monitoring-only.
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_epoch, best = epoch, score
                epoch_progress.set_postfix(loss=f'{loss_sum / seen:.4f}', fixed_epochs=policy['fixed_epochs'])
                if epoch % every == 0 or epoch == policy['fixed_epochs']:
                    save_recovery(snapshot())
                epoch_durations[active_steps] = time.perf_counter() - epoch_started
                if epoch >= policy['fixed_epochs']:
                    break
                continue
            if np.isfinite(score) and score < best - settings.get('min_delta', 0.):
                best, best_epoch, stale = score, epoch, 0
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                message = (f'{algorithm} member {member + 1} | New best epoch: {best_epoch}'
                           f' | validation macro NRMSE: {best:.8f}')
                if config['progress_bars'] and TQDM_AVAILABLE:
                    tqdm.write(message)
                else:
                    print(message, flush=True)
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step(score)
            epoch_progress.set_postfix(val_NRMSE=f'{score:.5f}', best=f'{best:.5f}',
                                       lr=f'{optimizer.param_groups[0]["lr"]:.1e}', stale=f'{stale}/{settings["patience"]}')
            if epoch % every == 0 or stale >= settings['patience'] or epoch == settings['epochs']:
                save_recovery(snapshot())
            epoch_durations[active_steps] = time.perf_counter() - epoch_started
            if stale >= settings['patience']:
                break
        epoch_progress.close()
        if state is None:
            raise FloatingPointError('No trained checkpoint was retained.')
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        checkpoints.append({'member': member, 'selected_epoch': best_epoch, 'epochs_run': epoch,
                            'selection': policy['mode'],
                            'fixed_epochs': policy['fixed_epochs'],
                            'validation_macro_NRMSE': None if validate_only else best,
                            'selected_rollout_steps': rollout_steps_at_epoch(recipe, best_epoch) if best_epoch else 0,
                            'initialization_selected': (not validate_only) and best_epoch == 0 and not resuming,
                            'warm_start_selected': (not validate_only) and best_epoch == 0 and resuming})
        print(f'{algorithm} member {member + 1}: ran {epoch} epochs; selected epoch {best_epoch}; {seen:,} examples in final epoch.')
        if validate_only:
            print(f'Fixed-epoch budget used ({policy["fixed_epochs"]} epochs): the last epoch is kept and no '
                  'held-out selection was performed. Report this model as fitted on all available data '
                  'with no held-out performance estimate.')
        elif best_epoch == 0 and resuming:
            print('Selected the resumed checkpoint: additional training did not improve validation.')
        elif best_epoch == 0:
            print('Selected the initialization checkpoint: training did not improve validation; no learned improvement is claimed.')
        completed_states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        save_recovery()
        del loader, dataset, optimizer, scheduler
        gc.collect()
    return {'algorithm': algorithm, 'kind': 'neural', 'models': models, 'architecture': architecture,
            'preprocessing': preprocessing, 'device': device, 'fit_seconds': prior_seconds + time.perf_counter() - start,
            'inference_batch_size': config['validation_batch_size'], 'training_curve': curves,
            'uncertainty_mode': architecture.get('uncertainty_mode'),
            'member_count': len(models),
            'action_reference': ({**fit_action_reference(index, preprocessing),
                                  'k': int(settings.get('action_novelty_k', 5))}
                                 if settings.get('action_novelty', True) else None),
            'mc_passes': settings.get('mc_passes'),
            'selected_checkpoints': checkpoints, 'selected_hyperparameters': settings,
            'training_rollout': rollout_report,
            'peak_cuda_allocated_gib': (max(prior_peak_cuda, torch.cuda.max_memory_allocated(device) / 1024 ** 3)
                                        if str(device).startswith('cuda') else None),
            'resume_source': resume_source,
            'recovery_source': str(recovery_input) if recovery_input else None,
            'recovery_checkpoint': str(recovery_path) if recovery_path else None,
            'optimizer_state_restored': bool(recovery_input),
            'selection_policy': policy,
            'selection_rule': (
                f'fixed_epochs={policy["fixed_epochs"]}: last epoch retained, no held-out selection; '
                'fitted on all supplied cultivations; no held-out performance estimate'
                if validate_only else
                'minimum validation macro NRMSE with declared minimum improvement; resumed weights included as epoch zero; optimizer/scheduler reinitialized; on-demand windows'
                if resuming else
                'minimum validation macro NRMSE with declared minimum improvement; initialization included only when configured; on-demand windows')}


def prepare_tabular(index, preprocessing, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    width = index.history * len(index.features) + len(ACTIONS) + 3
    estimate = len(index) * width * 4 * config['classical_working_copy_factor'] / 1024 ** 3
    if estimate > config['classical_working_budget_gib']:
        raise MemoryError(f'Estimated classical working arrays {estimate:.2f} GiB exceed configured budget; use capped_all_models.')
    path = directory / 'standardized_training_features.float32.memmap'
    if path.exists():
        raise FileExistsError(f'Cache already exists: {path}')
    flat = np.memmap(path, dtype=np.float32, mode='w+', shape=(len(index), width))
    y = np.empty((len(index), len(index.targets)), dtype=np.float32)
    chunk = config['classical_cache_batch_size']
    for start in progress(range(0, len(index), chunk), enabled=config['progress_bars'],
                          desc='Prepare disk-backed classical features', unit='chunk'):
        stop = min(start + chunk, len(index))
        raw = index.raw_batch(np.arange(start, stop))
        batch = transform(raw, preprocessing)
        flat[start:stop], y[start:stop] = batch['flat'], batch['y']
    flat.flush()
    del raw, batch
    return {'flat': flat, 'y': y, 'raw': {'anchors': index.anchors},
            'cache_path': str(path), 'estimated_working_arrays_gib': estimate}


def gpr_skip_reason(window_count, config):
    working_gib = window_count ** 2 * 8 * 3 / 1024 ** 3
    if window_count > config['gpr_max_train_windows'] or working_gib > config['gpr_working_budget_gib']:
        return (f'Dense GPR skipped: {window_count:,} windows, about {working_gib:.2f} GiB for three dense working matrices. '
                'Use training_track=capped_all_models for an equal-budget eight-family comparison.')
    return None


def numerical_threads(config):
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        print('threadpoolctl unavailable; estimator worker limits remain active, BLAS thread limit cannot be enforced.')
        return nullcontext()
    return threadpool_limits(limits=config['cpu_threads'])


def train_classical(algorithm, train, validation, preprocessing, config):
    with numerical_threads(config):
        return _train_classical(algorithm, train, validation, preprocessing, config)


def _train_classical(algorithm, train, validation, preprocessing, config):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    settings = config['classical'][algorithm]
    enabled = config['progress_bars']
    start = time.perf_counter()
    trials, selected, delta_offset = [], settings.copy(), 0.

    def validation_score(estimator, offset=0.):
        prediction = restore_levels(estimator.predict(validation['flat']) + np.asarray(offset), validation['raw']['latest'], preprocessing)
        score = point_metrics(validation['raw']['truth'], prediction, validation['raw']['latest'], preprocessing)[1]
        _, diagnostic = treatment_validation_diagnostic(
            {'kind': 'classical', 'algorithm': algorithm, 'estimator': estimator, 'delta_offset': offset,
             'preprocessing': preprocessing}, validation)
        return score, diagnostic

    if algorithm == 'ridge':
        parameters = {key: value for key, value in settings.items() if key != 'alpha_candidates'}
        candidates = settings.get('alpha_candidates')
        if candidates is None:
            candidates = [parameters['alpha']]
        if not candidates:
            raise ValueError('Ridge requires at least one alpha candidate.')
        best = np.inf
        for alpha in progress(candidates, enabled=enabled, desc='Ridge: validation candidates', unit='fit'):
            candidate = Ridge(**{**parameters, 'alpha': alpha})
            candidate.fit(train['flat'], train['y'])
            score, diagnostic = validation_score(candidate)
            trials.append({'alpha': alpha, 'validation_macro_NRMSE': score, **diagnostic})
            if np.isfinite(score) and score < best:
                best, estimator, selected = score, candidate, {**parameters, 'alpha': alpha}
        if not np.isfinite(best):
            raise FloatingPointError('No finite Ridge validation candidate.')
    elif algorithm == 'random_forest':
        target_trees = settings['n_estimators']
        parameters = {key: value for key, value in settings.items() if key != 'n_estimators'}
        estimator = RandomForestRegressor(n_estimators=1, warm_start=True, random_state=config['seed'],
                                         n_jobs=config['cpu_threads'], **parameters)
        built = 0
        with progress(total=target_trees, enabled=enabled, desc='Random Forest: trees', unit='tree') as bar:
            while built < target_trees:
                next_count = min(built + config['rf_tree_chunk_size'], target_trees)
                estimator.set_params(n_estimators=next_count)
                estimator.fit(train['flat'], train['y'])
                bar.update(next_count - built)
                built = next_count
    elif algorithm == 'xgboost':
        base = OPTIONAL['xgboost'].XGBRegressor(random_state=config['seed'], n_jobs=config['cpu_threads'],
                                                tree_method='hist', **settings)
        estimator = MultiOutputRegressor(base, n_jobs=1)
        estimator.estimators_ = []
        targets = target_order(preprocessing)
        for target in progress(range(len(targets)), enabled=enabled, desc='XGBoost: target regressions', unit='target'):
            from sklearn.base import clone
            regressor = clone(base)
            arguments = ({'eval_set': [(validation['flat'], validation['y'][:, target])], 'verbose': False}
                         if settings.get('early_stopping_rounds') else {})
            regressor.fit(train['flat'], train['y'][:, target], **arguments)
            estimator.estimators_.append(regressor)
            trials.append({'target': targets[target], 'selected_iteration': getattr(regressor, 'best_iteration', settings['n_estimators'] - 1)})
        estimator.n_features_in_ = estimator.estimators_[0].n_features_in_
    elif algorithm == 'hrp_xgb':
        from sklearn.multioutput import MultiOutputRegressor
        parameters = dict(settings['xgb_params'])
        member_count = int(settings.get('members', 5))
        if member_count < 2:
            raise ValueError('HRP-XGB requires at least two bootstrap members.')
        anchors = train['raw']['anchors']
        cultivation_ids = np.asarray([anchor['cultivation'] for anchor in anchors])
        groups = np.unique(cultivation_ids)
        if len(groups) < 2:
            raise ValueError('HRP-XGB cultivation bootstrap requires at least two independent training cultivations.')
        rng = np.random.default_rng(config['seed'])
        estimators, squared_errors, residual_counts = [], np.zeros(train['y'].shape[1]), np.zeros(train['y'].shape[1], dtype=np.int64)
        chunk_size = config['classical_cache_batch_size']
        for member in progress(range(member_count), enabled=enabled, desc='HRP-XGB cultivation bootstrap', unit='member'):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            positions = np.concatenate([np.flatnonzero(cultivation_ids == group) for group in sampled_groups])
            base = OPTIONAL['xgboost'].XGBRegressor(random_state=config['seed'] + member,
                n_jobs=config['cpu_threads'], tree_method='hist', **parameters)
            member_model = MultiOutputRegressor(base, n_jobs=1)
            member_model.fit(train['flat'][positions], train['y'][positions])
            estimators.append(member_model)
            oob_groups = np.setdiff1d(groups, sampled_groups)
            oob_positions = np.flatnonzero(np.isin(cultivation_ids, oob_groups))
            for start in range(0, len(oob_positions), chunk_size):
                batch_positions = oob_positions[start:start + chunk_size]
                errors = np.asarray(train['y'][batch_positions]) - member_model.predict(train['flat'][batch_positions])
                squared_errors += np.square(errors).sum(axis=0)
                residual_counts += len(batch_positions)
            trials.append({'member': member, 'sampled_cultivations': sampled_groups.tolist(),
                           'oob_cultivations': oob_groups.tolist(), 'oob_rows': int(len(oob_positions))})
        residual_variance = np.divide(squared_errors, residual_counts,
            out=np.var(train['y'], axis=0).astype(np.float64), where=residual_counts > 0)
        residual_variance = np.maximum(residual_variance, 1e-8)
        estimator = None
        selected = {'members': member_count, 'bootstrap_unit': 'whole cultivation run',
                    'baseline': preprocessing.get('baseline_description'),
                    'xgb_params': parameters, 'residual_noise': 'training out-of-bootstrap mean squared error',
                    'residual_variance_fallback_targets': np.flatnonzero(residual_counts == 0).tolist()}
    elif algorithm == 'lightgbm':
        from sklearn.multioutput import MultiOutputRegressor
        params = dict(settings)
        params.setdefault('verbosity', -1)
        estimator = MultiOutputRegressor(OPTIONAL['lightgbm'].LGBMRegressor(
            random_state=config['seed'], n_jobs=max(1, config['cpu_threads']), **params), n_jobs=1)
        estimator.fit(train['flat'], train['y'])
        trials.append({'target_count': len(target_order(preprocessing)),
                       'training_rows': int(len(train['y'])),
                       'validation_rows': int(len(validation['y']))})
    elif algorithm == 'gpr':
        reason = gpr_skip_reason(len(train['y']), config)
        if reason:
            raise MemoryError(reason)
        if settings.get('persistence_prior', False):
            if preprocessing.get('target_transform', 'delta') not in ['delta', 'mixed']:
                raise ValueError('GPR persistence prior requires delta targets; disable it for level prediction.')
            delta_offset = -np.array(preprocessing['delta_mean']) / np.array(preprocessing['delta_scale'])
            if preprocessing.get('target_transform') == 'mixed':
                delta_offset = np.where([name in residual_target_order(preprocessing) for name in target_order(preprocessing)], delta_offset, 0.)
        candidates = settings.get('kernel_candidates')
        if candidates is None:
            candidates = [{'length_scale_factor': settings['length_scale_factor'], 'noise_level': settings['noise_level']}]
        if not candidates:
            raise ValueError('GPR requires at least one kernel candidate.')
        best = np.inf
        for candidate_settings in progress(candidates, enabled=enabled, desc='GPR: validation candidates', unit='fit'):
            length = candidate_settings['length_scale_factor'] * np.sqrt(train['flat'].shape[1])
            kernel = ConstantKernel(1, constant_value_bounds='fixed') * Matern(length, length_scale_bounds='fixed', nu=1.5) + WhiteKernel(candidate_settings['noise_level'], noise_level_bounds='fixed')
            candidate = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False,
                                                 random_state=config['seed'], alpha=1e-6)
            candidate.fit(train['flat'], train['y'] - delta_offset)
            score, diagnostic = validation_score(candidate, delta_offset)
            trials.append({**candidate_settings, 'validation_macro_NRMSE': score, **diagnostic})
            if np.isfinite(score) and score < best:
                best, estimator, selected = score, candidate, {**candidate_settings, 'persistence_prior': settings.get('persistence_prior', False)}
        if not np.isfinite(best):
            raise FloatingPointError('No finite GPR validation candidate.')
    else:
        raise ValueError(algorithm)
    result = {'algorithm': algorithm, 'kind': 'classical', 'estimator': estimator, 'preprocessing': preprocessing,
            'delta_offset': delta_offset, 'device': 'cpu', 'fit_seconds': time.perf_counter() - start,
            'settings': settings, 'selected_hyperparameters': selected, 'hyperparameter_trials': trials,
            'selection_rule': ('minimum K14 validation macro NRMSE over configured candidates' if algorithm in ['ridge', 'gpr']
                               else 'per-target K14 early stopping' if algorithm == 'xgboost' and settings.get('early_stopping_rounds')
                               else 'predeclared regularized recipe; no test tuning')}
    if algorithm == 'hrp_xgb':
        result.update({'estimators': estimators, 'residual_variance': residual_variance,
                       'oob_residual_observations_per_target': residual_counts.tolist()})
        result['selection_rule'] = 'fixed XGBoost recipe; cultivation-level bootstrap members; residual variance from training OOB errors'
    return result


def train_physical(index, validation, preprocessing, config):
    with numerical_threads(config):
        return _train_physical(index, validation, preprocessing, config)


def _train_physical(index, validation, preprocessing, config):
    if not {'1_do(mg/L)', '1_temp_air(C)'}.issubset(target_order(preprocessing)):
        raise ValueError('Reduced physical baseline requires both DO and temperature targets; it cannot predict the selected outcome set.')
    from scipy.optimize import least_squares
    raw = index.boundary_rows()
    last = raw['sequence'][:, -1]
    features = feature_order(preprocessing)
    pwm, light = last[:, features.index('Lampu (PWM%)')], last[:, features.index('2_light(Lux)')]
    gain = max(float(np.polyfit(pwm, light, 1)[0]), 0) if np.std(pwm) > 1e-8 else 0.
    targets = ['1_do(mg/L)', '1_temp_air(C)']
    indices = [target_order(preprocessing).index(column) for column in targets]
    scale = np.maximum(np.array(preprocessing['training_level_std'])[indices], 1e-8)
    truth = raw['truth'][:, indices]
    rng = np.random.default_rng(config['seed'])
    initial = np.array([.1, 1., 8., .1, .1, .1, .1, 0.])
    start, solutions = time.perf_counter(), []
    for attempt in progress(range(config['physical']['multistarts']), enabled=config['progress_bars'],
                            desc='Grey-box: calibration starts', unit='start'):
        guess = initial.copy() if attempt == 0 else initial * rng.uniform(.5, 1.5, len(initial))
        fit = least_squares(lambda parameters: ((physical_prediction(parameters, raw, gain) - truth) / scale).ravel(),
                            guess, bounds=([0, 0, 0, 0, 0, 0, 0, -np.inf], np.full(8, np.inf)),
                            max_nfev=config['physical']['max_nfev'])
        if np.isfinite(fit.cost) and np.isfinite(fit.x).all():
            solutions.append({'parameters': fit.x.tolist(), 'training_cost': float(fit.cost),
                              'optimizer_converged': bool(fit.success), 'nfev': fit.nfev})
    if not solutions:
        raise FloatingPointError('No finite physical calibration.')
    chosen = min(solutions, key=lambda row: row['training_cost'])
    return {'algorithm': 'grey_box', 'kind': 'physical', 'parameters': chosen['parameters'], 'light_gain': gain,
            'sampling_interval_seconds': index.timing['sampling_interval_seconds'],
            'observation_operator': index.timing['observation_operator'],
            'targets': targets, 'preprocessing': preprocessing, 'device': 'cpu',
            'fit_seconds': time.perf_counter() - start, 'calibration_runs': solutions,
            'selection_rule': 'minimum training least-squares objective',
            'limitations': 'reduced equations with actual time step; minute means approximate boundary states, five-second targets use endpoint states; no pump-response model or verified identifiability'}




def cleanup_tabular(holder):
    prepared = holder.pop('prepared', None)
    if prepared is None:
        return
    path = Path(prepared['cache_path'])
    prepared['flat'].flush()
    del prepared
    gc.collect()
    try:
        path.unlink()
        if not any(path.parent.iterdir()):
            path.parent.rmdir()
    except OSError as error:
        print('Feature cache retained; remove it after closing the kernel:', path, error)


INFERENCE_CONTRACT_NAMES = ['feature_order', 'residual_target_order', 'TARGETS', 'CONTEXT', 'ACTIONS', 'FEATURES', 'CONTROL_TARGETS', 'target_order', 'validate_schema', 'baseline_values', 'target_baseline', 'derived_output_specs', 'derived_target_indices', 'apply_derived_outputs', 'derived_loss_mask', 'headline_target_names', 'load_frames', 'valid_anchors', 'raw_windows', 'transform', 'point_metrics', 'CausalBlock', 'DynamicsNetwork', 'make_architecture', 'neural_components', 'restore_levels', 'mean_linear_ode', 'physical_prediction', 'predict', 'uncertainty_metrics', 'rollout_cohort', 'recursive_metrics', 'duration_steps', 'time_settings', 'observed_rows', 'continuous_run_lengths', 'clock_features', 'linear_ode_endpoint']


def inference_contract_hash():
    """Fingerprint prediction semantics independently of training-only code."""
    source = Path(__file__).read_text(encoding='utf-8')
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in INFERENCE_CONTRACT_NAMES:
            found[name] = ast.get_source_segment(source, node)
    if set(found) != set(INFERENCE_CONTRACT_NAMES):
        raise ValueError('Incomplete inference compatibility contract.')
    payload = (json.dumps([found[name] for name in INFERENCE_CONTRACT_NAMES]) + Path(wp.__file__).read_text()).encode()
    return hashlib.sha256(payload).hexdigest()


def duration_steps(minutes, interval_seconds):
    steps = float(minutes) * 60 / interval_seconds
    if not np.isfinite(steps) or steps < 1 or not np.isclose(steps, round(steps)):
        raise ValueError('Duration must be a positive whole number of prediction steps.')
    return int(round(steps))


def time_settings(config):
    interval = config.get('sampling_interval_seconds', 60)
    if interval not in (5, 60):
        raise ValueError('sampling_interval_seconds must be 5 or 60.')
    interval = int(interval)
    return {'sampling_interval_seconds': interval,
            'history_steps': duration_steps(config['history_minutes'], interval),
            'sampling_block_steps': duration_steps(config.get('sampling_block_minutes', 10), interval),
            'observation_operator': 'instantaneous_sample' if interval == 5 else 'interval_mean',
            'expected_source_rows': 1 if interval == 5 else 12}


def observed_rows(frame, interval_seconds=None, features=None):
    interval = interval_seconds or frame.attrs.get('sampling_interval_seconds', 60)
    valid = np.isfinite(frame[FEATURES if features is None else features].to_numpy(dtype=np.float32)).all(axis=1)
    if interval == 60:
        if 'n_data' not in frame:
            raise ValueError('Minute aggregates require n_data coverage.')
        valid &= frame.n_data.eq(12).to_numpy()
    if 'status' in frame:
        valid &= frame.status.eq('OK').to_numpy()
    return valid


def continuous_run_lengths(valid, timestamps, interval_seconds):
    valid = np.asarray(valid, dtype=bool)
    if not len(valid):
        return np.array([], dtype=np.int64)
    times = pd.DatetimeIndex(timestamps)
    nanoseconds = times.to_numpy(dtype='datetime64[ns]').view('i8')
    adjacent = np.r_[False, np.diff(nanoseconds) == interval_seconds * 1_000_000_000]
    positions = np.arange(len(valid))
    starts = ~valid | ~adjacent | ~np.r_[False, valid[:-1]]
    last_start = np.maximum.accumulate(np.where(starts, positions, 0))
    return np.where(valid, positions - last_start + 1, 0)


def clock_features(timestamps, first_timestamp):
    times = pd.DatetimeIndex(timestamps)
    hours = times.hour.to_numpy() + times.minute.to_numpy() / 60 + times.second.to_numpy() / 3600
    elapsed = (times - pd.Timestamp(first_timestamp)).total_seconds().to_numpy() / (86400 * 30)
    return np.column_stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24), elapsed]).astype(np.float32)


def linear_ode_endpoint(initial, rate, forcing, hours):
    rate = np.asarray(rate)
    factor = np.full(np.broadcast_shapes(np.shape(initial), np.shape(rate)), hours, dtype=float)
    np.divide(-np.expm1(-rate * hours), rate, out=factor, where=np.abs(rate) > 1e-12)
    return initial + (forcing - rate * initial) * factor


def tabular_skip_reason(window_count, config):
    timing = time_settings(config)
    width = timing['history_steps'] * len(feature_order(config)) + len(ACTIONS) + 3
    estimate = window_count * width * 4 * config['classical_working_copy_factor'] / 1024 ** 3
    if estimate > config['classical_working_budget_gib']:
        return (f'Classical working-array estimate {estimate:.2f} GiB exceeds '
                f"{config['classical_working_budget_gib']:.2f} GiB. Choose capped_all_models or reduce history/data explicitly.")
    return None



def artifact_time_settings(metadata):
    timing = time_settings(metadata['config'])
    if metadata.get('sampling_interval_seconds', 60) != timing['sampling_interval_seconds']:
        raise ValueError('Artifact cadence differs between metadata and configuration.')
    if metadata['history_minutes'] != metadata['config']['history_minutes']:
        raise ValueError('Artifact history differs between metadata and configuration.')
    if not np.isclose(metadata['horizon_minutes'] * 60, timing['sampling_interval_seconds']):
        raise ValueError('Artifact horizon differs from its one-step time contract.')
    if metadata['schema_version'] in ['world_model_v3', 'world_model_v4', 'world_model_v5']:
        for key in ['history_steps', 'observation_operator']:
            if metadata.get(key) != timing[key]:
                raise ValueError(f'Artifact {key} differs from its time contract.')
        if metadata.get('horizon_seconds') != timing['sampling_interval_seconds']:
            raise ValueError('Artifact horizon seconds differ from its time contract.')
        for key, value in timing.items():
            if metadata['preprocessing'].get(key) != value:
                raise ValueError(f'Saved preprocessing {key} differs from its time contract.')
    return timing


def initialize_output_heads(model, preprocessing, persistence):
    offset = -np.array(preprocessing['delta_mean']) / np.array(preprocessing['delta_scale'])
    if preprocessing.get('target_transform') == 'mixed':
        offset = np.where([name in residual_target_order(preprocessing) for name in target_order(preprocessing)], offset, 0.)
    groups = (model.architecture.get('head_groups', {'shared': list(range(len(target_order(preprocessing))))})
              if model.head_layout == 'dedicated' else {'shared': list(range(len(target_order(preprocessing))))})
    with torch.no_grad():
        if model.architecture['algorithm'] == 'dlinear':
            for layer in (model.trend_projection, model.residual_projection, model.action_projection):
                layer.weight.zero_()
                layer.bias.zero_()
            if persistence:
                model.action_projection.bias.copy_(torch.as_tensor(offset, dtype=model.action_projection.bias.dtype,
                                                                     device=model.action_projection.bias.device))
            model.variance_projection.weight.zero_()
            model.variance_projection.bias.fill_(1.386)
            return
        if model.architecture['algorithm'] == 'nhits':
            for block in model.nhits_blocks:
                final = block[-1]
                final.weight.zero_()
                final.bias.zero_()
                outputs = len(target_order(preprocessing))
                if persistence:
                    final.bias[-2 * outputs:-outputs].copy_(torch.as_tensor(offset, dtype=final.bias.dtype,
                                                                             device=final.bias.device))
                final.bias[-outputs:].fill_(1.386)
            return
        for key, indices in groups.items():
            head = (model.heads[key] if model.head_layout == 'dedicated' else model.head)[-1]
            outputs = len(indices)
            head.weight.zero_()
            if persistence:
                head.bias[:outputs].copy_(torch.as_tensor(offset[indices], dtype=head.bias.dtype,
                                                           device=head.bias.device))
            else:
                head.bias[:outputs].zero_()
            head.bias[outputs:].fill_(math.log(4.))


def initialize_persistence_head(model, preprocessing):
    initialize_output_heads(model, preprocessing, persistence=True)


def point_loss_weights(preprocessing):
    levels = np.array(preprocessing['training_level_std'], dtype=float)
    deltas = np.array(preprocessing['delta_scale'], dtype=float)
    weights = np.divide(deltas, levels, out=np.zeros_like(deltas), where=levels > 0)
    active = weights > 0
    if not active.any():
        raise ValueError('No target has nonzero training-level variance.')
    weights /= np.sqrt(np.mean(weights[active] ** 2))
    bounds = preprocessing.get('point_loss_weight_bounds')
    if bounds is not None:
        low, high = float(bounds[0]), float(bounds[1])
        if not (np.isfinite(low) and np.isfinite(high)) or low < 0 or low > high:
            raise ValueError('point_loss_weight_bounds must be two finite, increasing, nonnegative values.')
        weights = np.clip(weights, low, high)
    return weights.astype(np.float32)


def physics_bound_penalty(mean, normalized_history, preprocessing, settings):
    """Optional bound penalty; only applies bounds explicitly supplied by the experiment."""
    bounds = settings.get('physics_bounds', {})
    if not bounds or settings.get('physics_loss_weight', 0.) <= 0:
        return mean.new_zeros(())
    targets, features = target_order(preprocessing), feature_order(preprocessing)
    input_mean = mean.new_tensor(preprocessing['input_mean'])
    input_scale = mean.new_tensor(preprocessing['input_scale'])
    latest = mean.new_tensor(preprocessing.get('training_target_mean', preprocessing['delta_mean']))
    latest = latest.unsqueeze(0).expand(len(mean), -1).clone()
    for j, name in enumerate(targets):
        if name in features:
            index = features.index(name)
            latest[:, j] = normalized_history[:, -1, index] * input_scale[index] + input_mean[index]
    residuals = set(residual_target_order(preprocessing))
    baseline = torch.stack([latest[:, j] if name in residuals else torch.zeros_like(latest[:, j])
                            for j, name in enumerate(targets)], dim=1)
    level = baseline + mean * mean.new_tensor(preprocessing['delta_scale']) + mean.new_tensor(preprocessing['delta_mean'])
    scale = mean.new_tensor(preprocessing['training_level_std']).clamp_min(1e-6)
    penalties = []
    for name, limits in bounds.items():
        index = targets.index(name)
        violation = torch.zeros_like(level[:, index])
        if limits.get('minimum') is not None:
            violation = violation + torch.relu(mean.new_tensor(limits['minimum']) - level[:, index]) / scale[index]
        if limits.get('maximum') is not None:
            violation = violation + torch.relu(level[:, index] - mean.new_tensor(limits['maximum'])) / scale[index]
        penalties.append(violation.square())
    return torch.stack(penalties, dim=1).mean()




