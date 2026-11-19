# method

Scripts that build the data behind the "How it works" page (`frontend/public/data/agreement.json`) and the homepage
experiments (`frontend/public/data/experiments.json`). Every cost and token number on the site comes from
`frontend/public/data/costs.json`, built by `complai export-web`, not from these scripts. Paths are relative to each script, so they run from any
directory with the repo venv:

```sh
.venv/bin/python -W ignore tools/method/<script>.py
```

## Scripts

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `lomo.py` | Leave-one-model-out validation of core.v1: refits the IRT model without each panel model, predicts that model from its core subset, summarises the folds against in-sample predictions, and counts the calibration panel. `--summary` skips the slow refit and reuses the saved folds. | `tools/minify/data/samples.jsonl` (local, gitignored), `src/complai/data/core.v1/` | `out/lomo/*.{pkl,csv}`, `out/lomo/meta.json`, `out/panel.json` |
| `agreement.py` | Everything quoted on the "How it works" page: LOMO figure rows, agreement of our Capability score with Epoch ECI and the AA Intelligence Index (correlations, pairwise rank agreement, seeded bootstrap intervals), panel counts. | `out/lomo/`, `out/panel.json`, `data/`, `frontend/public/data/predictions.json`, `frontend/public/models/*.yaml` | `frontend/public/data/agreement.json` |
| `experiments.py` | The homepage findings (capability vs safety, safety areas, scores over time, parameters, labs). `--verify` checks headline numbers against the values signed off on 2026-11-08. | `frontend/public/data/{predictions,sample_analysis,labels}.json`, `frontend/public/models/*.yaml` | `frontend/public/data/experiments.json` |

## Data snapshots (`data/`)

| File | Source | Pulled |
|---|---|---|
| `aa_models.json` | Artificial Analysis models API (Intelligence Index v4) | 2026-10-26 |
| `eci_scores.csv` | Epoch AI Capabilities Index | 2026-10-26 |

## Run order

1. `complai export-web` (refreshes `frontend/public/data/*.json`, including `costs.json`)
2. `lomo.py` (only when the panel or the core.v1 parameters change; `lomo.py --summary` to re-summarise)
3. `agreement.py`

`experiments.py` is independent of the "How it works" chain: rerun it whenever the experiments change or after step 1.
`out/` holds the intermediate outputs, so step 3 can be rerun on its own.
