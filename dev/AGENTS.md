# Notebook workspace instructions

These instructions govern the development workspace under `dev/`. Follow the user's current request when it changes these instructions. Use project research plans and evaluation protocols in `context/` where available, while preserving the notebook structure defined here.

use bold italic text on the header for example: # **_Header_** or ### **_Header_** for all header type

## **_Development Folder Architecture_**

`dev/` is the active development workspace. Keep each subsystem's source, notebooks, tests, and artifacts inside its owning folder. Do not create or retain a generic `dev/_shared/`; place shared MBRL contracts and interfaces in the subsystem that owns them or in an explicitly defined project-level package when a later architecture decision requires one.

| Path | Responsibility |
| --- | --- |
| `dev/world_model/` | World-model source code, protocol/recovery/plotting helpers, tests, training/evaluation notebooks, and world-model model/evaluation artifacts. |
| `dev/world_model/model/<run_id>/` | A run's checkpoints, matching metadata, evaluation exports, figures, manifests, and recovery files. Use run ID as the immediate folder name. |
| `dev/dataset/` | Prepared experiment data used by development tools. Keep original source data intact and document copies/manifests. |
| `dev/dataset_eda/` | Dataset EDA source/notebook, quality helpers, tests, and EDA reports. |
| `dev/reward_schema/` | Reward function source and versioned/selectable reward schemas only; no evaluation results. |
| `dev/reward_model/evaluation/<run_id>/` | Reward-schema case/edge-case evaluation reports, metrics, and figures. |
| `dev/rl_controller/` | Modular RL algorithm implementations (for example PPO, DQN), shared controller interfaces, algorithm-specific tests, model registry rules, and the RL model contract. Keep trained artifacts out of this source folder. |
| `dev/train_rl.py` | Terminal entry point for the RL training/evaluation interface. |
| `dev/train_rl/` | Curses-based UI, configuration/validation, checkpoint and continuation workflow, RL model training/evaluation orchestration, and `model/<run_id>/` outputs. |
| `dev/cultivation_run/` | Closed-loop model-based cultivation simulation tools and `result/<run_id>/` outputs, including CSV traces, metrics, and figures. |
| `dev/` environment files | A virtual environment or dependency lockfile may be added later; do not make environment setup a prerequisite for this structural move. |

### **_World-model ownership_**

All world-model-specific code belongs in `dev/world_model/`, including `world_models.py`, protocol, recovery, training diagnostics, plots, and their tests. Training and evaluation notebooks import those local modules. World-model artifacts and evaluation outputs live together under `dev/world_model/model/<run_id>/`. The notebook configuration must resolve datasets from `dev/dataset/` and model results from `dev/world_model/model/`; do not leave active paths pointing into `notebook/` or `dev/model/`.

For now, the MBRL feature contract remains the existing 21-input history schema and five control/action features. Keep that interface explicit and consistent between the world model, RL controller, and cultivation simulator. Do not silently change or independently redefine its feature order. A future contract change must update metadata, loaders, controller/simulator integration, and compatibility checks together.

### **_Reward schemas and validation_**

Support multiple selectable reward definitions under `dev/reward_schema/`, preferably using a stable schema format with explicit units, target channels, weights/terms, bounds, and version. A modular callable is acceptable when a schema cannot express the required calculation; keep its interface and version explicit. Provide a `simulate_case.py`-style validator that evaluates each schema against normal, boundary, missing/invalid-input, and edge cases. Save validation evidence under `dev/reward_model/evaluation/<run_id>/`; do not mix it into the schema source directory.

### **_RL controller and model registry_**

Keep PPO/DQN and future algorithms as modular source under `dev/rl_controller/`, with a common observation/action interface and explicit algorithm configs. The model registry is the source of truth for available algorithm identifiers, module entry points, configuration schema, supported action/observation contracts, and checkpoint format. Define an RL model rule covering run ID, algorithm, source world-model ID/hash, reward-schema ID/version, observation/action schema, training configuration, seed, checkpoint/resume lineage, and evaluation status. Reject incompatible world-model or reward contracts explicitly.

### **_RL training and evaluation tool_**

`dev/train_rl.py` launches a terminal curses interface backed by `dev/train_rl/`. Its planned modes are **Train RL model** and **Evaluate trained RL model**. Training settings must include algorithm selection from the registry, an explicit world-model run, reward schema, environment/episode settings, hyperparameters, seed, checkpoint cadence, and continuation from a selected checkpoint. Write each run and its evaluations to `dev/train_rl/model/<run_id>/`; retain resumable checkpoints and metadata there.

Keep training-time policy diagnostics distinct from full closed-loop cultivation runs. The training/evaluation mode should use repeatable fixed initial-state/target benchmark cases and report reward components, constraint violations, action use, target error, success rate, and steps/time-to-target across seeds. Where a dataset cultivation is replayed, label it as a benchmark/reference trajectory; the policy's simulated closed-loop path is not an observed counterfactual. Full scenario setup, user-defined initial states/targets, and dataset-derived initial-state/target comparisons belong to `dev/cultivation_run/`, avoiding duplicate cultivation-run tooling in the RL trainer.

### **_Cultivation run tool_**

`dev/cultivation_runs.py` is the planned command-line entry point for `dev/cultivation_run/`. It selects an RL policy from `dev/train_rl/model/` and a world model from `dev/world_model/model/`, verifies their recorded contracts, and supports two modes: manually supplied initial state and target, or initial state/target derived from an explicitly selected dataset cultivation. Save complete per-step simulation traces, actions, target/constraint metrics, and figures under `dev/cultivation_run/result/<run_id>/`. Each report must include simulated elapsed time to target or timeout; cumulative use of each controllable resource/action in its native units; and start, end, absolute change, and percentage change for each configured monitored concentration. Include CO2 only when the CO2 channel is configured and label it carbon dioxide; never conflate it with carbon monoxide (CO), which is a different gas and is not currently in the sensor schema. Mark predicted/simulated values separately from measured values, and calculate change from the declared initial state. Do not describe simulated outcomes as real cultivation outcomes.

## Required notebook structures

Use Markdown cells with the following level-two headings, in this order. Keep these headings even when a section is short or not applicable. Add supporting subsections beneath them rather than rearranging top-level sections. The optional `Physical Baselines` training section described below is an allowed extension. A brief title or purpose cell may precede the sections.

Preserve this structure for every algorithm addition, removal, configuration change, and progress/reporting edit. Keep each algorithm's training, checkpoint report, and artifact export together under its existing parent section. Use bold italic notebook headings, for example `## ***Deep Learning***`; the section names below identify those same headings.

### Data evaluation notebook

```markdown
## Import

## Per Data Evaluation Report
```

- **Import:** imports and any shared path settings or helpers needed to produce the reports. Keep them visible and reusable.
- **Per Data Evaluation Report:** load and inspect each dataset or cultivation, identify it clearly, and display its report. Include available time range, row counts, sampling intervals, missing values, duplicate timestamps, units, observed/derived fields, and action availability where relevant. Summaries must come from inspected data. Distinguish unavailable information from a measured zero. A cross-dataset summary may be a subsection here.

Default filename: `dataset_eval/data_evaluation.ipynb`.

### Model evaluation notebook

```markdown
## Import

## Config

## Data Load

## Function Section

## Evaluation Metric Report
```

- **Import:** common imports and explicit optional-package availability checks. Keep dependency handling in this section; selected algorithms are checked after `Config`.
- **Config:** dataset path, model directory or explicit model path, naming prefix, selected model/run, evaluation split, device, seed if needed, and report/export options. Derive paths and names from this section.
- **Data Load:** read evaluation data and the selected model's JSON metadata, resolve referenced artifact paths, and check required files and data fields. Read checkpoint contents here only when this requires no later-defined functions or architecture classes. Leave model reconstruction and preprocessing-bundle loading to the report section after their helpers are defined.
- **Function Section:** define or bind shared model reconstruction/loading helpers, inference, recursive prediction, metrics, and plotting functions. Group algorithm-specific loaders or predictors into subsections as needed. Do not invoke an evaluation run while defining these helpers.
- **Evaluation Metric Report:** first invoke the defined helpers to reconstruct/load the model and its saved preprocessing, verify compatibility, and prepare evaluation inputs. Then run inference and display labeled metrics, tables, plots, and limitations. Identify the algorithm, artifact/run, dataset split, target channels, and prediction horizon.

Every function or architecture must be available before its first use. Preserve this section order; do not rely on manually running a later cell first to make `Data Load` succeed.

Evaluation loads the artifacts produced by training. It must not retrain the model, refit preprocessing, or select hyperparameters using test data. Use saved feature/target order, units, transformations, and history settings. Treat mismatched metadata, missing artifacts, or incompatible data as explicit errors rather than silently substituting another model or fitting replacements.

Put aggregate model comparisons and their CSV exports in a `Model Comparison and CSV Export` subsection under `Evaluation Metric Report`. Include model status, run/split, target units, forecast horizon, and evaluation mode. Keep physical-baseline subset comparisons separate; unavailable metrics remain missing with a reason. Training may retain per-model validation for checkpoint selection and metadata plus a run manifest, but must not produce cross-model rankings or comparison CSV files.

### Model training notebook

```markdown
## Import

## Config

## Data Load

## Preprocessing

## Deep Learning

## Machine Learning
```

- **Import:** common imports and explicit optional-package availability checks. Keep dependency handling in this section; selected algorithms are checked after `Config`.
- **Config:** dataset path, model output directory, naming prefix, run identifier, algorithms to run, split definition, features/targets, time resolution, history length, prediction horizon, random seeds, device, and algorithm settings. Keep adjustable settings here rather than hiding them inside functions.
- **Data Load:** load the used data, preserve source identifiers, and check the fields required by the configured experiment. Display a concise loading summary.
- **Preprocessing:** construct splits and transitions/windows, apply documented cleaning, fit training-only transformations, and prepare the inputs shared by the algorithms. Keep validation/test data separate. Include any preprocessing helpers under this section.
- **Deep Learning:** the parent section for neural algorithms. Give each algorithm its own subsection containing its architecture, model-specific helpers, training function, training invocation, validation summary, and artifact export.
- **Machine Learning:** the parent section for classical ML algorithms. Give each algorithm its own subsection containing estimator construction, model-specific preparation, training function, training invocation, validation summary, and artifact export.

When a notebook includes a conventional physical or grey-box model, append this optional parent after `Machine Learning`:

```markdown
## Physical Baselines
```

Each physical-baseline subsection contains its equations, parameter-calibration helpers, calibration invocation, validation summary, and model/metadata export. Parameter calibration belongs here, not in `Preprocessing`. Omit this optional parent when no physical baseline is included. Preserve the six required training headings in either case.

For example, GRU and LSTM belong under `Deep Learning`; Ridge and Random Forest belong under `Machine Learning`; grey-box reactor dynamics belong under `Physical Baselines`. Keep a required parent heading and mark it not applicable if the notebook has no algorithms in that category. Label controllers accurately: neural policies may be grouped under `Deep Learning`, while MPC is a controller using a predictor and must not be presented as a world-model architecture. Other structural extensions require user direction; do not silently reorganize the existing sections.

Algorithm-specific training and saving belong inside the corresponding algorithm subsection. Do not add a separate top-level training, architecture, or export section.

## Progress bars and checkpoint reporting

- Use `tqdm.auto` for long notebook operations, directly or through the existing shared progress wrapper. Check availability in `Import`, expose `progress_bars` in `Config`, and print a clear fallback message if tqdm is unavailable. Do not install packages as an import side effect.
- Show progress for actual work: epochs and batches for neural training, ensemble members, trees, target regressions, search candidates, calibration attempts, dataset files, and rollout steps as applicable. Set meaningful descriptions, totals when known, and units such as `epoch`, `batch`, `tree`, or `step`. A batch is a group of windows; never label batch counts as sample/window counts.
- Keep nested batch bars temporary (`leave=False`) and avoid printing every batch. Display relevant loss/validation values in the progress postfix. Do not invent smooth percentage progress for opaque estimator calls or imply a fit is complete before it finishes.
- After iterative neural training, explicitly print `Best epoch`, its checkpoint-selection metric and value, and actual epochs run. Report each ensemble member separately. Use the selected checkpoint metadata, respecting the configured selection rule and minimum improvement, rather than inferring the best checkpoint from the last epoch or training loss.
- If epoch 0 is selected, explicitly label it as the initialization checkpoint and state that training did not improve validation. Preserve checkpoint details in the matching model metadata. For estimators without epochs, use applicable settings or best iteration when available; do not fabricate a best epoch.
- Preserve existing completed results as artifacts. Clear stale output from edited notebook execution cells without rerunning training merely to refresh display text.

### Interruption recovery and diagnostic rules

- Long neural training must expose full-state recovery in Config and save atomically to a separate recovery artifact at least after every completed epoch and ensemble member. Keep inference/best-model artifacts separate. Print the recovery path before training.
- Save current and best weights, optimizer/scheduler, epoch/member, early-stopping counters, training curve, completed ensemble members, data-loader generator, Python/NumPy/PyTorch/CUDA RNG states, and AMP scaler state when AMP is used. Save or validate bootstrap positions, preprocessing, data identity, training settings, validation windows and code/runtime identity.
- Distinguish full-state recovery from weight-only warm-start/fine-tuning. Recovery restores the original schedule and selection history; warm start may reset them and must say so. Never silently reinterpret an inference artifact as a resumable training checkpoint.
- State recovery granularity: epoch-boundary recovery replays an unfinished epoch. Reject incompatible settings/data/code rather than mixing experiments, use a new output run ID, and preserve completed and source artifacts. Only load trusted recovery files. Verify recovery against an uninterrupted short run before declaring it supported; no full production training is needed for this check.
- Report selected-checkpoint treatment-event and no-event validation diagnostics separately from checkpoint selection. Include response direction, amplitude and explicitly defined threshold-based delay, plus per-channel invalid prediction counts, magnitudes and recursive steps. Report missing eligible events rather than inventing them. Logged-action responses are conditional associations, not causal effects.
- Follow the dataset README for TSS provenance: newer K11-K14 hardware has a dedicated water-quality TSS sensor; older K1-K9 `u1_tss_est` is a different source. Do not group measured TSS with interpolated biomass solely because both are currently output-only. TSS history is a separately configured/identified experiment.
- BOD remains an independent learned target until the user authorizes a derivation change; do not enforce COD/2 merely because the README documents it. Record post-treatment sensor timestamps: the logged transition treatment precedes its corresponding reading, with uncertain within-bin exposure duration. No future sensor readings may enter rollout feedback.
- All new world-model runs require the fixed 21-channel input schema, including measured TSS before the five actions. There is no TSS-history switch or supported schema without TSS. Do not mutate the schema at runtime. Keep the original nine residual targets explicit so TSS remains a level target; recursive training, evaluation and diagnostics must feed predicted TSS back rather than using measured future values. Use TSS persistence when its history is available and record input/target assignments with the artifact.
- Per the user's 24 October decision, the current experimental backend supports `world_model_v5` only; legacy loading/migration is not required. Preserve older files as historical records. Train matched controls under the required TSS schema when comparing subsequent changes; never overwrite or silently reinterpret old artifacts.

## Shared helpers and dependencies

Keep training/evaluation helpers in the owning subsystem folder (for example, `dev/world_model/`) when they implement that subsystem. Keep one authoritative implementation rather than copying code between notebooks. Import helpers through `Import`; expose their invocation and algorithm-specific wrappers in the appropriate notebook sections. Keep configurable experiment settings in `Config`, not in hidden helper globals. Record the helper/code version used by a saved artifact where available.

Notebooks remain the visible experiment workflow: configuration, data summaries, algorithm execution, and reports must stay readable there. Shared helpers must not perform training or evaluation as an import side effect. Training and evaluation must be able to import them independently from a fresh kernel.

Handle missing optional packages explicitly in `Import`, recording availability and the actual import error. Once `Config` selects algorithms, check their required packages. A missing PyTorch or XGBoost package must not block an unrelated available Ridge/Random Forest experiment. Skip unavailable selected algorithms with a clear status and reason while running the others. If no selected algorithm is runnable, stop with an actionable dependency report. Do not silently substitute algorithms, install packages as an import side effect, or catch unrelated runtime errors as if they were missing dependencies. Verify accelerator availability separately from package importability and report the actual device used.

## Model and metadata naming contract

Every successful training run must save its model and metadata using the same base filename with different extensions:

```text
{prefix}_{algorithm}_{run_id}.pt       # example neural checkpoint
{prefix}_{algorithm}_{run_id}.json     # matching metadata

{prefix}_{algorithm}_{run_id}.joblib   # example classical model bundle
{prefix}_{algorithm}_{run_id}.json     # matching metadata
```

The extensions may differ for other frameworks, but the shared base filename is mandatory. Configure the prefix and run identifier in `Config`, and derive the filename consistently. Use descriptive algorithm identifiers and unique run identifiers so one algorithm or run does not overwrite another unintentionally.

The metadata must contain enough information for evaluation to load and interpret the model without rerunning training:

- Algorithm name, model category, run identifier, and artifact filenames.
- Dataset sources/version or manifest and train/validation/test split definitions.
- Ordered input features and output targets, with units and measurement/derived status where relevant.
- Sampling interval, timestamp/bin convention, history length, and prediction horizon.
- Preprocessing definitions, fitted parameters or artifact references, and inverse transformations.
- Architecture/estimator settings and training hyperparameters.
- Seed, actual training device, and relevant dependency versions.
- Checkpoint selection rule, validation results, and completion status.

Save fitted preprocessing with the model bundle where practical. If it requires separate files, reference those exact filenames in the matching metadata. Evaluation must reuse them. Metadata stores experiment facts, not fabricated metrics or assumed hardware availability. Record incomplete/failed runs clearly; do not present their artifacts as completed training results.

Evaluation discovers the chosen model through its explicit configured path or matching naming prefix and metadata. If several runs match, list the candidates and use an explicit selection; never silently load an arbitrary file or the newest unrelated checkpoint.

## Rules for adding, removing, or changing algorithms

1. Preserve required section names, order, and parent/child hierarchy. Add or remove the optional `Physical Baselines` parent only as described above.
2. Add each algorithm as a subsection under its correct parent. Remove an algorithm's subsection when requested, while retaining the required parent heading.
3. Keep its notebook subsection cohesive: architecture/estimator construction or shared-helper reference, training helpers, invocation, and exports. Update the shared implementation, evaluation loader, and metadata contract together when an implementation changes.
4. Put changed settings in `Config`. Keep shared loading/preprocessing behavior consistent across algorithms unless a difference is scientifically necessary and documented.
5. Maintain matching model/metadata filenames and keep completed artifacts identifiable by run.
6. Update comparisons and reports to reflect the algorithms actually executed. Mark skipped, failed, and timed-out algorithms explicitly.

Do not break or simplify away the required structure to meet a time budget, add a new algorithm, or remove an existing one. Reduce optional experiments or tuning within the structure instead. The user can explicitly revise the structure.

## Reproducibility and evaluation

- Notebooks must run in order from a fresh kernel, without relying on variables or outputs from another notebook or an earlier interactive session.
- Use portable paths resolved from a declared project/notebook root. Avoid assumptions that depend on the kernel's current working directory.
- Training writes model artifacts and metadata; evaluation consumes them. Keep this boundary explicit.
- For time-series cultivation data, split by cultivation before forming windows and never cross episode boundaries. Fit scaling/imputation on training data only.
- Keep sampling interval, prediction step, history duration, and controller update interval distinct. Convert history/rollout durations to sample counts explicitly. Training and EDA support native 5-second sensor files or provided 60-second aggregates; native MPPT remains 10 seconds. Never synthesize faster readings from minute means. Evaluation must reuse the saved resolution, observation operator, preprocessing, and time step; label point targets versus interval means and retain memory guards at either cadence.
- Exclude future observations from model inputs. Record whether future action exposure is provided for conditional prediction. Score biological outcomes against actual lab observations, not future-informed interpolated estimates.
- Use the metrics appropriate to the task, as specified in `../context/evaluation_protocol.md`. Compare models on the same targets, evaluation starts, and declared data budget. Keep physical-baseline subset scores separate from larger-target aggregates.
- Display reports in the notebook. Label units, data splits, model/run names, and horizons. Never fabricate results or retain stale outputs that appear to describe changed code.
- After an authorized code change, check the notebook structure and artifact paths. Run the relevant cells when dependencies/data permit; distinguish structural checks from an executed training/evaluation run.
- Creating or polishing these instructions does not authorize starting model training. Execute training only as part of a user-authorized implementation/run task.
