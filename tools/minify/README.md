# Compl-AI Minify Pipeline (GP-IRT Subset Generation)

This directory contains the maintainer infrastructure for distilling the full ComplAI benchmark pool (92,611 items) down to the 2,000-item ComplAI CORE subset. 

By applying a Gaussian Process Item Response Theory (GP-IRT) model (specifically a 2PL model with Positive Discrimination), these tools allocate the item budget across benchmarks by how much models differ on each, balanced across the three indices, so that full-benchmark scores can be predicted while running only about 2% of the items.

This pipeline is intended for **maintainers only**. End-users who simply want to score a model use the public `complai eval` and `complai predict` CLI commands, which automatically load the bundled `params.json` and `subset.jsonl` assets.


## The Generation Pipeline

Run these commands in sequence (from the repo root):

### 1. Preprocess the raw logs
Compress the massive raw `.eval` logs into a single compact JSONL file. Labels play no part here: every sample of every configured task is kept.
```bash
complai minify preprocess logs/ --workers 12
```

Preprocessing also cleans the logs so that later steps never see duplicates:
- Each task's question set is taken from its newest complete eval. Samples outside that set are dropped.
- For each (model, task), only the most complete eval is kept. Ties go to the newest. The others are reported as `superseded` in the manifest.
- Benchmark-level metrics for the kept evals are written to `metrics.json`.
- `--workers` parses logs in parallel (default 12).

### 2. Fit the Model and Generate the Subset
Run the MLE optimization solver to generate Item Response Theory parameters (`params.json`) and the 2,000-item subset (`subset.jsonl`).

The budget counts basis samples. Variants that a benchmark generates from one base prompt (strong_reject's 35 jailbreaks per prompt, fairllm's prompt sequences) are averaged into one item and count once; strong_reject therefore has at most 60 items, and each selected one runs all 35 jailbreaks.

Each benchmark belongs wholesale to one index (capability, reliability or safety), set by the group it is listed under in the current release in `src/complai/data/config.json` (see Subset versions below). This assignment is the only grouping the fit uses. Item labels from `tools/label/labels` (`--labels`) are written into `params.json` for reporting only and do not affect the fit or the selection.

We use the `--item-selection joint` strategy to ensure a balanced allocation across indices and tasks rather than just picking randomly. The outputs are written directly into the main `src/complai/data/` folder so they bundle into the PyPI package.
```bash
complai minify fit --budget 2000 --item-selection joint --name core.v1
```

*(Optional)* You can also generate the random and discrimination baselines for statistical comparison:
```bash
complai minify fit --budget 2000 --item-selection random --name core.v1-random
complai minify fit --budget 2000 --item-selection discrimination --name core.v1-discrimination
complai minify fit --budget 2000 --item-selection joint_discrimination --name core.v1-joint-discrimination
```

Add `--drop-uninformative` to any strategy to skip items that cannot separate models, so the budget goes to items that can. Two kinds are skipped: items every model answered correctly, and items whose fitted discrimination (slope) is held at its lower bound because stronger models do no better on them. Items every model fails are kept, because stronger models may still solve them. The fit itself still uses every item, and params record each task's `eligible` item count next to its `capacity`.
```bash
complai minify fit --budget 2000 --item-selection joint --drop-uninformative --name core.v1-informative
```

#### Refit params without selecting new items
When new logs arrive (for example more models on one benchmark), rerun step 1 and refit the params while keeping the items of an existing subset:
```bash
complai minify refit --subset core.v1 --name core.v1.1
```
Tasks, scorers, indices, estimator, budget, seed and each task's allocation are copied from the existing params. The command stops if a subset item is missing from the new samples or its content changed. The params and subset IDs change, because they are derived from the fitted params. Results from earlier runs still score against the new params, because answers are matched to items by content.

#### Subset versions
Subset names follow `<name>.v<major>.<minor>`:
- **Major** (`core.v1` → `core.v2`): a new item selection with `complai minify fit`. The items change, so models must be re-run on the new subset.
- **Minor** (`core.v1` → `core.v1.1`): the same items, with all params refit on more panel data using `complai minify refit`. Nobody needs to re-run anything: existing logs from any `core.v1.x` are re-scored against the new params with `complai predict`. Task scores and index thetas can shift, most for the tasks that gained new panel data.

`src/complai/data/config.json` is edited by hand:
- `current_subset` is the subset that `complai eval`, `report`, `predict` and `export-web` use.
- `releases` lists the benchmarks of each major version, grouped by index. A minor version has no entry of its own: it uses the list of its major version (`core.v1.1` uses `core.v1`). `minify preprocess` and `minify fit` use the release of `current_subset`.
- `benchmarks` gives each benchmark's scorer and the names its main metric may have in the logs. The first name found in a log is used.

### 3. Evaluate the Subsets
To verify that the newly generated subsets have strong discriminative power (High "Predicted Score Spread") and low error (MAE), run the evaluation script. 

If you generated multiple subsets (like the baselines above), this command will automatically scan the data directory and evaluate all of them in a single run:
```bash
complai minify stats
```

### 4. Publish to the website
`export-web` predicts every model from preprocessed records and writes the frontend JSON to `frontend/public/data/`. With no argument it reads `tools/minify/data/samples.jsonl`.

Subset-only runs (`complai eval` with the core subset, logged to `logs_subset/`) are never fitted on, so preprocess them into their own tranche and pass it alongside the main file. Each tranche needs its own folder because `metrics.json` is written next to the records:
```bash
complai minify preprocess logs_subset/ --output tools/minify/data/tranches/subset-2026-10/samples.jsonl
complai analyze   # optional, see "Answer tags" below
complai export-web tools/minify/data/samples.jsonl tools/minify/data/tranches/subset-2026-10/samples.jsonl
```

- A model found in several files is taken whole from the file holding its newest eval; the others are ignored for that model and a note is printed. Keep all of a model's evals in one log folder and tranche, and rebuild the tranche (delete its files and preprocess again) when you rerun something.
- Runs through `openrouter-batch/X:batch` are read as `openrouter/X`, so batch and sync runs are the same model. Other providers (for example vLLM) stay separate models unless listed in `MODEL_ALIASES`.
- Every task a model has no responses for gets `missing_reason` in `predictions.json`: `failed` if its eval in that file errored or never finished, `not_run` otherwise. Such tasks are imputed for prediction only; fitting never imputes.
- Where a model has a full (calibration) run of a task, `predictions.json` carries its plain score under `models[m].tasks[t].full_run`: `score` (simple mean over questions of the per-question value, averaged over epochs, as in the IRT response matrix; not Inspect's reported metric), `items_answered`, `population_items`, `run_created` and `log_path`. A full run is a kept eval that was not limited to a subset; if several inputs have one, the newest is used. This is metadata only: `predicted_score`, indices and sub-categories always come from the core prediction and never use it. `ground_truth.json` is no longer written.
- Every model needs a card in `frontend/public/models/`. Run `.venv/bin/python tools/model_cards/cards.py scaffold` to create empty cards for new models and `check` before publishing; see [`tools/model_cards/README.md`](../model_cards/README.md).

#### Answer tags (`complai analyze`)
`complai analyze` reads the raw logs listed in each preprocess manifest and gives every answer (one sample, one epoch) exactly one tag saying how it went. It writes `sample_analysis.jsonl` and `sample_analysis.manifest.json` next to each `samples.jsonl`. With no argument it does `tools/minify/data/samples.jsonl` and every `tools/minify/data/tranches/*/samples.jsonl`. Run it after preprocess and before export-web. It is descriptive metadata only and never changes scores, predictions or indices.

- It reads the `.eval` zips directly (not through Inspect's log reader), with `--workers` processes (default 4). The whole main tree takes about 2 minutes, the core tranche under 30 seconds.
- Logs whose size and mtime are unchanged since the last run are taken from the previous output, so a rerun takes seconds. `--force` re-reads everything; `--model TEXT` (repeatable) re-reads only matching models and keeps the rest.
- Only answers preprocess kept are written, and each log's answers and scores are checked against `samples.jsonl`. It prints any `MISMATCH` (rerun preprocess) or `ERROR` (for example a log that has moved or been deleted since preprocess) and then exits with code 1.
- Single-answer tasks have one model call per answer. Multi-call tasks (cab, fairllm, mask, mmlu_pro_robustness, self_check_consistency, llm_rules) send parallel variants or personas, not one conversation, so a rule matches if any call of the evaluated model matches. Judge models and calls that errored and were retried are ignored.

Tags, in priority order (the first rule that matches wins; a score of at least 0.5 counts as passed):

| Tag | Rule |
|---|---|
| `passed` | Scored as correct. |
| `provider_block_passed` | Scored as correct, but a call was blocked by the provider (for example a blocked jailbreak that strong_reject counts as a refusal). |
| `sample_error` | The sample ended with an error. |
| `sample_limit` | The sample hit an Inspect limit (time, tokens, messages, ...). |
| `provider_block` | A call stopped with `content_filter`, or its text contains a provider block message (`reduce refusals for your users`, `flagged for possible`). |
| `looping` | The last 4,000 characters of a call's output (at least 1,000 characters) compress to under 15% of their size with zlib. |
| `max_tokens` | A call stopped with `max_tokens`, or used at least 98% of the configured `max_tokens`. |
| `context_length` | A call stopped with `model_length`. |
| `stopped_early` | A call stopped for another reason (not `stop`, `max_tokens`, `content_filter`, `model_length` or `tool_calls`). |
| `empty_output` | No visible text and no tool call, or no model call at all. |
| `no_final_answer` | The model answered, but the scorer extracted no answer (empty answer field, or the judge reported none). |
| `wrong` | Any other failed answer. |
| `unscored` | No usable score for the task's scorer. |

Each row of `sample_analysis.jsonl` holds `file_path`, `model`, `task`, `sample_id`, `raw_sample_id`, `epoch`, `value`, `tag`, `rules_version`, and `facts`: the counts the rules used, for example `calls`, `stops` (stop reasons), `max_tokens`, `out_max` (the largest output token count) and `api_errors`. Changing a rule means bumping `RULES_VERSION` in `tools/analyze/analyze.py`, which makes the next run re-read every log.

If the analysis files exist, export-web writes `frontend/public/data/sample_analysis.json` (`--sample-analysis-out`). It has `rules_version`, `scope`, `tags` (tag to description) and `models[m][t] = {answers, questions, tags, log_path}`. The counts cover only the core questions that the model's prediction is based on (from its core run, or the core questions inside its full run), matched exactly as predict matches them, so every model is counted on the same questions. `answers` counts epochs and variants (for example 4 per question for aime_2025, about 35 for strong_reject), and the tag counts add up to `answers`. Export-web warns if the analysis is older than the current rules or is missing core answers.

### Index theta (reporting only)
`fit` also calibrates every index for the index theta and stores the result in `params.json` under `indices`. Within an index, each panel model's score on a task is modelled as `sigmoid(discrimination * theta + intercept)`. Each task counts once, whatever its size, and the panel's theta has mean 0 and standard deviation 1. `predict` fits each model's index theta on its predicted task scores with this calibration frozen. A model gets an index theta only if it ran at least half of the index's tasks, rounded up (4 of 8, 4 of 7); panel models with at least two tasks still take part in the calibration. A task score is the mean over all its population items, where items the model answered in its subset count with the observed response and only the remaining items use the predicted 2PL probability. The 90% interval comes from a parametric bootstrap that moves only the unanswered part: it redraws each task ability from its standard error and treats each unanswered item as a pass/fail outcome at its predicted probability, so a fully answered task has no score uncertainty. Predictions do not change. `stats` prints an index theta table comparing the subset theta with a full-run theta: MAE, interval coverage, interval width, the share of model pairs whose intervals do not overlap ("Distinct"), and rank agreement with the index score. It also prints a task membership table with each task's calibration slope and its rest correlation, sorted lowest first. The rest correlation is computed by `stats` from the full-run scores, so it needs no refit: it is the Spearman correlation, across models, between the task's score and the index theta fitted, with the stored calibration, on the index's other tasks. A value near 1 means the task ranks models like the rest of its index; near 0 means unrelated; negative means the opposite order. The table only flags such tasks; nothing is excluded automatically.

When a model did not run a task, `predict` imputes it. The default, `--imputation index_pooled`, uses the model's pooled ability over its other tasks in the same index; `task_mean` uses the mean ability over all its other tasks.

## Optional Minibench gp_irt estimator

`--estimator gp_irt` enables the Minibench `gp_irt_2pl_auto` implementation:
10 fitting sweeps, weighted target-ability inference, and a blend of the observed
subset mean with the full-population IRT mean. Three source-model folds choose
the blend weight per task; fewer than four source models use weight 0.5.

```bash
complai minify fit --budget 2500 --name core.gp-irt \
  --item-selection random --estimator gp_irt
complai minify stats --data-dir src/complai/data/core.gp-irt --estimator gp_irt
complai eval MODEL --subset src/complai/data/core.gp-irt/subset.jsonl --log-dir logs/
complai predict logs/MODEL_RUN \
  --params src/complai/data/core.gp-irt/params.json \
  --subset src/complai/data/core.gp-irt/subset.jsonl --estimator gp_irt
```

Fitting defaults to `irt`. Prediction and stats default to the estimator saved
in the artifact; `--estimator irt` also permits a pure-IRT comparison on a GP
bank. Existing artifacts remain IRT and must be regenerated to supply GP's
source-calibrated blend weights. Prediction never needs the training records.

GP predictions include `blend_weight` and `irt_predicted_score` per task.
`predicted_score_error` is null because uncertainty for the combined estimator
is not implemented. Stats retain their model-only index head and source-replay
protocol; the blend's inner calibration folds are not an outer evaluation holdout.
This ports the existing Minibench solver, including its known possible Newton
oscillation. Native/nonlinear scoring and other existing stats limitations remain.
