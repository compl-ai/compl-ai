# Compl-AI Minify Pipeline (GP-IRT Subset Generation)

This directory contains the maintainer infrastructure for distilling the 100K+ item COMPL-AI benchmark pool down to the highly discriminative CORE subset. 

By applying a Gaussian Process Item Response Theory (GP-IRT) model (specifically a 2PL model with Positive Discrimination), these tools identify the most informative items that maximize score differentiation across models, allowing you to reconstruct full-benchmark capabilities while running only 5% of the items or less.

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
Run the MLE optimization solver to generate Item Response Theory parameters (`params.json`) and the 5,000 item subset (`subset.jsonl`).

The budget counts basis samples. Variants that a benchmark generates from one base prompt (strong_reject's 35 jailbreaks per prompt, fairllm's prompt sequences) are averaged into one item and count once; strong_reject therefore has at most 60 items, and each selected one runs all 35 jailbreaks.

Each benchmark belongs wholesale to one index (capability, reliability or safety), set by its `index` field in `subset_config.json`. This assignment is the only grouping the fit uses. Item labels from `tools/label/labels` (`--labels`) are written into `params.json` for reporting only and do not affect the fit or the selection.

We use the `--item-selection joint` strategy to ensure a balanced allocation across indices and tasks rather than just picking randomly. The outputs are written directly into the main `src/complai/data/` folder so they bundle into the PyPI package.
```bash
complai minify fit --budget 5000 --item-selection joint --name core.v1
```

*(Optional)* You can also generate the random and discrimination baselines for statistical comparison:
```bash
complai minify fit --budget 5000 --item-selection random --name core.v1-random
complai minify fit --budget 5000 --item-selection discrimination --name core.v1-discrimination
complai minify fit --budget 5000 --item-selection joint_discrimination --name core.v1-joint-discrimination
```

Add `--drop-uninformative` to any strategy to skip items that cannot separate models, so the budget goes to items that can. Two kinds are skipped: items every model answered correctly, and items whose fitted discrimination (slope) is held at its lower bound because stronger models do no better on them. Items every model fails are kept, because stronger models may still solve them. The fit itself still uses every item, and params record each task's `eligible` item count next to its `capacity`.
```bash
complai minify fit --budget 5000 --item-selection joint --drop-uninformative --name core.v1-informative
```

### 3. Evaluate the Subsets
To verify that the newly generated subsets have strong discriminative power (High "Predicted Score Spread") and low error (MAE), run the evaluation script. 

If you generated multiple subsets (like the baselines above), this command will automatically scan the data directory and evaluate all of them in a single run:
```bash
complai minify stats
```

### Index theta (reporting only)
`fit` also calibrates every index for the index theta and stores the result in `params.json` under `indices`. Within an index, each panel model's score on a task is modelled as `sigmoid(discrimination * theta + intercept)`. Each task counts once, whatever its size, and the panel's theta has mean 0 and standard deviation 1. `predict` fits each model's index theta on its predicted task scores with this calibration frozen. A model needs at least two tasks in the index to get an index theta. A task score is the mean over all its population items, where items the model answered in its subset count with the observed response and only the remaining items use the predicted 2PL probability. The 90% interval comes from a parametric bootstrap that moves only the unanswered part: it redraws each task ability from its standard error and treats each unanswered item as a pass/fail outcome at its predicted probability, so a fully answered task has no score uncertainty. Predictions do not change. `stats` prints an index theta table comparing the subset theta with a full-run theta: MAE, interval coverage, interval width, the share of model pairs whose intervals do not overlap ("Distinct"), and rank agreement with the index score. It also prints a task membership table with each task's calibration slope and its rest correlation, sorted lowest first. The rest correlation is computed by `stats` from the full-run scores, so it needs no refit: it is the Spearman correlation, across models, between the task's score and the index theta fitted, with the stored calibration, on the index's other tasks. A value near 1 means the task ranks models like the rest of its index; near 0 means unrelated; negative means the opposite order. The table only flags such tasks; nothing is excluded automatically.

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
