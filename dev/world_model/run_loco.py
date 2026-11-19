"""Leave-one-cultivation-out driver for the world-model notebooks.

Each fold is one independent training run: the held-out cultivation is the
selection split and everything else is fitting data. Cross-validation is a
diagnostic, not a deployable model -- report the per-fold values and their
spread, never the mean alone, and fit a separate all-data model for production.

Usage
    python dev/world_model/run_loco.py --algorithms tcn --folds 11 12 13 14 16
    python dev/world_model/run_loco.py --algorithms tcn xgboost --dry-run

Requires papermill (already used by this project) and an executable "python3"
kernel. Set --allow-missing to skip folds whose run produces no artifact
instead of aborting.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = ROOT / 'dev' / 'world_model' / 'model'


def parse_args():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--algorithms', nargs='+', default=['tcn'],
                        help='Algorithm identifiers copied into the training run.')
    parser.add_argument('--folds', nargs='+', type=int, default=[11, 12, 13, 14, 16],
                        help='Cultivations to rotate through the validation slot.')
    parser.add_argument('--cultivations', nargs='+', type=int, default=None,
                        help='Full usable set. Defaults to --folds.')
    parser.add_argument('--prefix', default='wm_12targets_v6')
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--interval', type=int, default=60, choices=[5, 60])
    parser.add_argument('--model-root', default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument('--report', default=None,
                        help='Summary CSV path. Defaults to <model-root>/loco_summary_<prefix>.csv')
    parser.add_argument('--timeout-seconds', type=int, default=None,
                        help='Per-notebook execution timeout passed to papermill.')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without executing.')
    parser.add_argument('--allow-missing', action='store_true',
                        help='Record a fold as missing instead of aborting the whole sweep.')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Train only; the training notebook already records validation metrics.')
    parser.add_argument('--final', action='store_true',
                        help='After the folds, also fit the all-data model with a fixed epoch budget '
                             'derived from the folds. It has no held-out score.')
    parser.add_argument('--fixed-epochs', type=int, default=None,
                        help='Explicit fixed epoch budget for --final. Defaults to the median '
                             'selected epoch across folds.')
    return parser.parse_args()


def fold_plan(cultivations, folds):
    """One entry per fold: (validation cultivation, train cultivations)."""
    cultivations = list(cultivations)
    missing = [k for k in folds if k not in cultivations]
    if missing:
        raise ValueError(f'Fold cultivations not in the usable set: {missing}')
    if len(cultivations) < 2:
        raise ValueError('At least two cultivations are required for a fold.')
    plan = []
    for holdout in folds:
        train = [k for k in cultivations if k != holdout]
        if not train:
            raise ValueError(f'Fold {holdout} would have no training cultivations.')
        plan.append((holdout, train))
    return plan


def run_notebook(notebook, parameters, output_path, timeout):
    import papermill
    papermill.execute_notebook(
        str(notebook), str(output_path),
        parameters=parameters, kernel_name='python3',
        progress_bar=False, execution_timeout=timeout)


def read_fold_metrics(model_root, prefix, run_id, algorithms):
    """Collect per-algorithm metrics from a finished fold's saved metadata."""
    run_dir = Path(model_root) / run_id
    rows = []
    for algorithm in algorithms:
        metadata_path = run_dir / f'{prefix}_{algorithm}_{run_id}.json'
        if not metadata_path.is_file():
            rows.append({'algorithm': algorithm, 'status': 'missing_artifact',
                         'macro_NRMSE_headline': None, 'macro_NRMSE_all': None})
            continue
        metadata = json.loads(metadata_path.read_text())
        rows.append({
            'algorithm': algorithm,
            'status': metadata.get('status', 'unknown'),
            'macro_NRMSE_headline': metadata.get('macro_NRMSE_headline'),
            'macro_NRMSE_all': metadata.get('macro_NRMSE_all'),
            'headline_targets': len(metadata.get('headline_targets') or []),
            'derived_outputs': ','.join(sorted(metadata.get('derived_outputs') or {})),
            'sampling_interval_seconds': metadata.get('sampling_interval_seconds'),
            'rollout_steps': (metadata.get('training_rollout') or {}).get('steps'),
            'selected_epochs': ','.join(str(c.get('selected_epoch')) for c in metadata.get('selected_checkpoints') or []),
            'selection': ','.join(str(c.get('selection', 'validation')) for c in metadata.get('selected_checkpoints') or []),
            'validation_available': metadata.get('validation_available', True),
            'fit_seconds': metadata.get('fit_seconds'),
        })
    return rows


def median_epoch_budget(records, column='selected_epochs'):
    """Median selected epoch across scored folds -- the fixed_epochs budget for the final model."""
    values = []
    for row in records:
        if row.get('status') != 'completed' or not row.get(column):
            continue
        for part in str(row[column]).split(','):
            part = part.strip()
            if part.isdigit() and int(part) > 0:
                values.append(int(part))
    if not values:
        return None
    import numpy as np
    return int(np.median(values))


def main():
    args = parse_args()
    cultivations = args.cultivations or args.folds
    plan = fold_plan(cultivations, args.folds)
    model_root = Path(args.model_root)
    report_path = Path(args.report) if args.report else model_root / f'loco_summary_{args.prefix}.csv'

    print(f'Usable cultivations: {cultivations}')
    print(f'Folds: {len(plan)} | algorithms: {args.algorithms} | interval: {args.interval}s')
    print(f'Summary: {report_path}')
    for index, (holdout, train) in enumerate(plan, start=1):
        print(f'  fold {index}: validation=K{holdout:02d} train={["K%02d" % k for k in train]}')

    if args.dry_run:
        print('\nDry run: no notebook executed.')
        return 0

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_dir = model_root / f'loco_{stamp}'
    log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (holdout, train) in enumerate(plan, start=1):
        label = f'fold{index}_K{holdout:02d}'
        print(f"\n=== fold {index}/{len(plan)}: validation K{holdout:02d} ===", flush=True)
        run_id = f'{args.interval}S_LOCO_{label}_{stamp}'
        parameters = {
            'SAMPLING_INTERVAL_SECONDS': args.interval,
            'SEED': args.seed,
            'PREFIX': args.prefix,
            'ALGORITHMS': list(args.algorithms),
            'CULTIVATIONS': list(cultivations),
            'FOLD_VALIDATION': holdout,
            'RUN_ID': run_id,
        }
        executed = log_dir / f'{label}_training.ipynb'
        try:
            run_notebook(NOTEBOOKS / 'training.ipynb', parameters, executed, args.timeout_seconds)
        except Exception as error:
            print(f'fold {index} training failed: {type(error).__name__}: {error}', flush=True)
            records.append({'fold': index, 'validation_cultivation': holdout,
                            'train_cultivations': ' '.join(map(str, train)), 'run_id': run_id,
                            'algorithm': ','.join(args.algorithms), 'status': 'failed',
                            'reason': f'{type(error).__name__}: {error}'})
            if not args.allow_missing:
                raise
            continue

        # RUN_ID is passed explicitly, so the fold directory is known without globbing.
        actual_run_id = run_id
        if not args.skip_evaluation:
            try:
                run_notebook(NOTEBOOKS / 'evaluation.ipynb',
                             {'EVAL_RUN_ID': actual_run_id, 'EVAL_SPLIT': 'validation'},
                             log_dir / f'{label}_evaluation.ipynb', args.timeout_seconds)
            except Exception as error:
                print(f'fold {index} evaluation failed: {type(error).__name__}: {error}', flush=True)

        for row in read_fold_metrics(model_root, args.prefix, actual_run_id, args.algorithms):
            records.append({'fold': index, 'validation_cultivation': holdout,
                            'train_cultivations': ' '.join(map(str, train)),
                            'run_id': actual_run_id, **row})

    import pandas as pd
    frame = pd.DataFrame(records)
    frame.to_csv(report_path, index=False)
    print(f'\nWrote {len(frame)} rows to {report_path}')

    scored = frame[frame.get('macro_NRMSE_headline').notna()] if 'macro_NRMSE_headline' in frame else frame.iloc[:0]
    if len(scored):
        print('\nPer-fold headline macro NRMSE (never report the mean alone):')
        for algorithm, group in scored.groupby('algorithm'):
            values = group.sort_values('fold')
            joined = ', '.join(f"K{int(k):02d}={v:.5f}" for k, v in
                               zip(values.validation_cultivation, values.macro_NRMSE_headline))
            print(f'  {algorithm}: {joined}')
            print(f'    mean={values.macro_NRMSE_headline.mean():.5f} '
                  f'std={values.macro_NRMSE_headline.std():.5f} n={len(values)}')

    if args.final:
        budget = args.fixed_epochs or median_epoch_budget(records)
        if not budget:
            print('\n--final requested but no fold produced a usable selected epoch; '
                  'pass --fixed-epochs explicitly. Skipping the all-data run.')
            return 1
        print(f'\n=== final all-data model: fixed_epochs={budget} ===')
        print('No cultivation is held out; this run produces no held-out score.')
        final_id = f'{args.interval}S_FINAL_all_{stamp}'
        final_parameters = {
            'SAMPLING_INTERVAL_SECONDS': args.interval,
            'SEED': args.seed,
            'PREFIX': args.prefix,
            'ALGORITHMS': list(args.algorithms),
            'CULTIVATIONS': list(cultivations),
            'FOLD_VALIDATION': 'all',
            'RUN_ID': final_id,
            'SELECTION': 'fixed_epochs',
            'FIXED_EPOCHS': budget,
        }
        try:
            run_notebook(NOTEBOOKS / 'training.ipynb', final_parameters,
                         log_dir / 'final_all_training.ipynb', args.timeout_seconds)
            print(f'Final artifact run: {final_id}')
            print('Report performance from the folds above, not from this run.')
        except Exception as error:
            print(f'final all-data training failed: {type(error).__name__}: {error}')
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
