# Compl-AI Minify Pipeline (GP-IRT Subset Generation)

This directory contains the maintainer infrastructure for distilling the 100K+ item COMPL-AI benchmark pool down to the highly discriminative CORE subset. 

By applying a Gaussian Process Item Response Theory (GP-IRT) model (specifically a 2PL model with Positive Discrimination), these tools identify the most informative items that maximize score differentiation across models, allowing you to reconstruct full-benchmark capabilities while running only 5% of the items or less.

This pipeline is intended for **maintainers only**. End-users who simply want to score a model use the public `complai eval` and `complai predict` CLI commands, which automatically load the bundled `params.json` and `subset.jsonl` assets.


## The Generation Pipeline

Run these commands in sequence (from the repo root):

### 1. Preprocess the raw logs
Compress the massive raw `.eval` logs into a single compact JSONL file. 

By passing the `--domain-labels` path and the `--valid-labels-only` flag, you guarantee that only samples that have passed human labeling review in `tools/label/labels` are included.
```bash
complai minify preprocess logs/ --domain-labels tools/label/labels --valid-labels-only
```

### 2. Fit the Model and Generate the Subset
Run the MLE optimization solver to generate Item Response Theory parameters (`params.json`) and the 5,000 item subset (`subset.jsonl`).

We use the `--item-selection joint` strategy to ensure a balanced allocation across domains and tasks rather than just picking randomly. The outputs are written directly into the main `src/complai/data/` folder so they bundle into the PyPI package.
```bash
complai minify fit --budget 5000 --item-selection joint --domain-labels tools/label/labels --duplicates mean --name core.v1
```

*(Optional)* You can also generate the random and discrimination baselines for statistical comparison:
```bash
complai minify fit --budget 5000 --item-selection random --domain-labels tools/label/labels --duplicates mean --name core.v1-random
complai minify fit --budget 5000 --item-selection discrimination --domain-labels tools/label/labels --duplicates mean --name core.v1-discrimination
```

### 3. Evaluate the Subsets
To verify that the newly generated subsets have strong discriminative power (High "Predicted Score Spread") and low error (MAE), run the evaluation script. 

If you generated multiple subsets (like the baselines above), this command will automatically scan the data directory and evaluate all of them in a single run:
```bash
complai minify stats
```

## Optional Minibench gp_irt estimator

`--estimator gp_irt` enables the Minibench `gp_irt_2pl_auto` implementation:
10 fitting sweeps, weighted target-ability inference, and a blend of the observed
subset mean with the full-population IRT mean. Three source-model folds choose
the blend weight per task; fewer than four source models use weight 0.5.

```bash
complai minify fit --budget 2500 --name core.gp-irt \
  --item-selection random --duplicates latest --estimator gp_irt
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
is not implemented. Stats retain their model-only domain head and source-replay
protocol; the blend's inner calibration folds are not an outer evaluation holdout.
This ports the existing Minibench solver, including its known possible Newton
oscillation. Native/nonlinear scoring and other existing stats limitations remain.
