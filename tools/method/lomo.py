"""Leave-one-model-out (LOMO) validation of core.v1, using the `minify stats` code paths, plus the calibration-panel
counts quoted on the methodology page.

For each panel model M:
  1. take the full-run response matrices (same as `minify refit` builds) with M's row removed,
  2. `irt.refit` with core.v1 items/allocation fixed (task 2PLs + index calibration refitted without M),
  3. `predict_detailed` on all records (M's subset answers are used, its other answers are not),
  4. keep M's rows only and compare with full-run truth exactly as `minify stats` does.
Also runs the in-sample prediction (production params, no refit) and a no-exclusion refit as a sanity check.

Writes
  out/lomo/{in_sample,refit_all,lomo}.pkl, meta.json   the folds (slow; skipped with --summary)
  out/lomo/{summary,index_summary,task_summary,per_model_index,per_model_task}.csv
                                                         in-sample vs LOMO, `minify stats` metric definitions.
                                                         Index intervals are the raw bootstrap ones (published
                                                         widening undone); index_summary also gives the widening
                                                         needed for 90% coverage and the coverage at the current one.
  out/panel.json                                         calibration panel: models, scored responses, models per benchmark
All three read tools/minify/data/samples.jsonl (gitignored), so this is the only step that needs the raw samples.

Run:  .venv/bin/python tools/method/lomo.py [--workers N] [--limit K]    refit all folds, then summarise
      .venv/bin/python tools/method/lomo.py --summary                    summarise saved folds + panel counts only
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import multiprocessing as mp
import pickle
import re
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import complai.irt as irt  # noqa: E402
from complai.irt import index_thetas  # noqa: E402
from complai.predict import DEFAULT_IMPUTATION, INDEX_THETA_INTERVAL_SCALE, predict_detailed  # noqa: E402
from complai.utils.io import write_outputs  # noqa: E402
from complai.utils.log_parser import load_records  # noqa: E402
from tools.minify.evaluate_subsets import task_predictions  # noqa: E402
from tools.minify.ground_truth import (  # noqa: E402
    load_ground_truth,
    masked_label_scores,
    masked_pairs,
    masked_task_scores,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out/lomo"
DATA = REPO / "src/complai/data/core.v1"
SAMPLES = REPO / "tools/minify/data/samples.jsonl"
LABELS = REPO / "tools/label/labels"
IMPUTATION = "index_pooled"

STATE: dict = {}


def full_tasks(records, scorers):
    return irt.prepare_tasks(records, scorers)


def without_model(tasks, model):
    out = {}
    for name, task in tasks.items():
        task = dict(task)
        if model in task["models"]:
            row = task["models"].index(model)
            task["models"] = [m for m in task["models"] if m != model]
            task["matrix"] = np.delete(task["matrix"], row, axis=0)
        else:
            task["matrix"] = task["matrix"].copy()
        task["items"] = list(task["items"])
        out[name] = task
    return out


def model_rows(prediction, truth, model, fold):
    """The held-out model's task / index / overall comparisons, using the stats definitions."""
    result = prediction.result["models"][model]
    tasks_rows, index_rows = [], []
    preds = task_predictions(prediction, model)
    masked = masked_task_scores(prediction, truth).get(model, {})
    for task, predicted in sorted(preds.items()):
        tasks_rows.append({
            "fold": fold, "model": model, "task": task,
            "index": prediction.params["tasks"][task]["index"] if "tasks" in prediction.params else None,
            "predicted_score": predicted,
            "benchmark_score": truth.benchmark_scores.get(task, {}).get(model),
            "masked_predicted": masked.get(task, {}).get("predicted_score"),
            "masked_observed": masked.get(task, {}).get("observed_score"),
            "subset_answered": result["tasks"][task].get("observations"),
            "subset_items": result["tasks"][task].get("subset_items"),
        })
    label = masked_label_scores(prediction, truth, "index").get(model, {})
    for index, calibration in sorted(prediction.params.get("indices", {}).items()):
        names = sorted(calibration["tasks"])
        a = np.asarray([calibration["tasks"][n]["discrimination"] for n in names])
        c = np.asarray([calibration["tasks"][n]["intercept"] for n in names])
        entry = result.get("indices", {}).get(index) or {}
        estimate = entry.get("index_theta")
        row = {
            "fold": fold, "model": model, "index": index,
            "masked_predicted": label.get(index, {}).get("predicted_score"),
            "masked_observed": label.get(index, {}).get("observed_score"),
            "theta_predicted": None, "theta_lower": None, "theta_upper": None,
            "theta_full_run": None, "tasks_used": None, "tasks_total": len(names),
        }
        if estimate is not None:
            observed = np.full(len(names), np.nan)
            for position, name in enumerate(names):
                task_result = result["tasks"].get(name, {})
                if task_result.get("imputed") or task_result.get("status") not in ("ok", "partial"):
                    continue
                if name in masked:
                    observed[position] = masked[name]["observed_score"]
            # Store the raw bootstrap interval: undo the published widening around theta.
            theta, scale = estimate["theta"], estimate.get("interval_scale", 1.0)
            row.update({
                "theta_predicted": theta,
                "theta_lower": theta - (theta - estimate["interval"][0]) / scale,
                "theta_upper": theta + (estimate["interval"][1] - theta) / scale,
                "theta_full_run": float(index_thetas(observed[None, :], a, c)[0]),
                "tasks_used": estimate["tasks"],
            })
        index_rows.append(row)
    predicted, observed = [], []
    for name in prediction.item_probabilities.get(model, {}):
        p, o = masked_pairs(prediction, truth, model, name)
        predicted += p
        observed += o
    overall = {
        "fold": fold, "model": model,
        "masked_predicted": float(np.mean(predicted)) if observed else None,
        "masked_observed": float(np.mean(observed)) if observed else None,
        "imputed_tasks": result.get("imputed_tasks"),
    }
    return tasks_rows, index_rows, overall


def run_fold(model):
    """Refit without `model` (None = no exclusion) and return that model's rows (all models if None)."""
    start = time.time()
    tasks = STATE["tasks"] if model is None else without_model(STATE["tasks"], model)
    original = irt.prepare_tasks
    irt.prepare_tasks = lambda records, scorers, **kw: copy.copy(tasks)
    try:
        result = irt.refit(STATE["records"], copy.deepcopy(STATE["params"]), copy.deepcopy(STATE["subset"]),
                           labels_dir=LABELS)
    finally:
        irt.prepare_tasks = original
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "fold"
        write_outputs(result, out)
        prediction = predict_detailed(SAMPLES, out / "params.json", out / "subset.jsonl",
                                      imputation=IMPUTATION)
    truth = STATE["truth"]
    fold = "refit_all" if model is None else "lomo"
    targets = STATE["models"] if model is None else [model]
    rows = [model_rows(prediction, truth, m, fold) for m in targets]
    calibration = {i: v["tasks"] for i, v in prediction.params.get("indices", {}).items()}
    return {"model": model, "rows": rows, "seconds": time.time() - start,
            "estimator": prediction.result.get("estimator", "irt"), "calibration": calibration}


def refit(args):
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    print("Loading ground truth (also warms the prediction matrix memo)...", flush=True)
    truth = load_ground_truth(REPO / "logs", SAMPLES)
    records = load_records(SAMPLES)
    params = json.loads((DATA / "params.json").read_text())
    subset = [json.loads(line) for line in (DATA / "subset.jsonl").open() if line.strip()]
    print(f"  {time.time() - t0:.0f}s; building refit matrices...", flush=True)
    tasks = full_tasks(records, params["task_scorers"])
    STATE.update(truth=truth, records=records, params=params, subset=subset, tasks=tasks)

    print(f"  {time.time() - t0:.0f}s; in-sample prediction (production params)...", flush=True)
    prediction = predict_detailed(SAMPLES, DATA / "params.json", DATA / "subset.jsonl", imputation=IMPUTATION)
    models = sorted(prediction.result["models"])
    STATE["models"] = models
    in_sample = [model_rows(prediction, truth, m, "in_sample") for m in models]
    production_calibration = {i: v["tasks"] for i, v in prediction.params.get("indices", {}).items()}
    print(f"  {len(models)} models; estimator={prediction.result.get('estimator', 'irt')}; "
          f"{time.time() - t0:.0f}s", flush=True)
    with open(OUT / "in_sample.pkl", "wb") as handle:
        pickle.dump({"rows": in_sample, "calibration": production_calibration, "models": models}, handle)

    print("Sanity fold: refit on all models...", flush=True)
    sanity = run_fold(None)
    print(f"  sanity fold {sanity['seconds']:.0f}s", flush=True)
    with open(OUT / "refit_all.pkl", "wb") as handle:
        pickle.dump(sanity, handle)

    targets = models[: args.limit] if args.limit else models
    folds = []
    context = mp.get_context("fork")
    with context.Pool(args.workers) as pool:
        for done, fold in enumerate(pool.imap_unordered(run_fold, targets), start=1):
            folds.append(fold)
            print(f"  [{done}/{len(targets)}] {fold['model']} {fold['seconds']:.0f}s "
                  f"(elapsed {time.time() - t0:.0f}s)", flush=True)
            with open(OUT / "lomo.pkl", "wb") as handle:
                pickle.dump(folds, handle)
    meta = {"workers": args.workers, "seconds": time.time() - t0, "models": models,
            "imputation": IMPUTATION, "default_imputation": DEFAULT_IMPUTATION,
            "subset_dir": str(DATA), "samples": str(SAMPLES)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {time.time() - t0:.0f}s", flush=True)



# ---------------------------------------------------------------- summary
SCALE = 15.0  # published index score = 100 + 15 * theta


def load_folds():
    in_sample = pickle.loads((OUT / "in_sample.pkl").read_bytes())
    refit_all = pickle.loads((OUT / "refit_all.pkl").read_bytes())
    lomo = pickle.loads((OUT / "lomo.pkl").read_bytes())
    schemes = {
        "in_sample": in_sample["rows"],
        "refit_all": refit_all["rows"],
        "lomo": [row for fold in lomo for row in fold["rows"]],
    }
    return in_sample, refit_all, lomo, schemes


def flatten(rows):
    tasks = [t for task_rows, _, _ in rows for t in task_rows]
    indices = [i for _, index_rows, _ in rows for i in index_rows]
    overall = [o for _, _, o in rows]
    return tasks, indices, overall


def pairwise_agreement(predicted, observed):
    """Share of pairs (untied in truth) ordered the same way by prediction."""
    agree = total = 0
    for i, j in itertools.combinations(range(len(predicted)), 2):
        truth = np.sign(observed[i] - observed[j])
        if truth == 0:
            continue
        total += 1
        agree += np.sign(predicted[i] - predicted[j]) == truth
    return agree / total if total else None, total


def distinct_share(intervals):
    pairs = [a[1] < b[0] or b[1] < a[0] for a, b in itertools.combinations(intervals, 2)]
    return float(np.mean(pairs)) if pairs else None


def task_metrics(tasks):
    rows = [t for t in tasks if t["benchmark_score"] is not None]
    errors = [abs(t["predicted_score"] - t["benchmark_score"]) for t in rows]
    by_task = {}
    for t in rows:
        by_task.setdefault(t["task"], []).append(t)
    per_task, taus = {}, []
    for task, group in sorted(by_task.items()):
        pred = [g["predicted_score"] for g in group]
        true = [g["benchmark_score"] for g in group]
        tau = scipy.stats.kendalltau(true, pred).statistic if len(group) > 1 else float("nan")
        if np.isfinite(tau):
            taus.append(tau)
        agree, pairs = pairwise_agreement(pred, true)
        per_task[task] = {
            "index": group[0]["index"], "models": len(group),
            "mae": float(np.mean([abs(p - o) for p, o in zip(pred, true)])),
            "median_ae": float(np.median([abs(p - o) for p, o in zip(pred, true)])),
            "kendall_tau": float(tau), "pairwise_agreement": agree, "pairs": pairs,
        }
    return {
        "task_pairs": len(errors),
        "task_mae_mean": float(np.mean(errors)),
        "task_mae_median": float(np.median(errors)),
        "avg_task_mae": float(np.mean([v["mae"] for v in per_task.values()])),
        "task_pairwise_rank_accuracy": (float(np.mean(taus)) + 1) / 2,
        "task_pairwise_agreement_pooled": (
            sum(v["pairwise_agreement"] * v["pairs"] for v in per_task.values() if v["pairs"])
            / sum(v["pairs"] for v in per_task.values())
        ),
    }, per_task


def needed_scale(pred, low, high, true, target=0.9):
    """Smallest factor >= 1 that, widening each interval around pred, makes `target` of them contain true."""
    half = np.where(true < pred, pred - low, high - pred)
    need = np.sort(np.abs(true - pred) / np.maximum(half, 1e-12))
    return max(1.0, float(need[int(np.ceil(target * len(need))) - 1]))


def index_metrics(indices):
    out = {}
    for index in sorted({i["index"] for i in indices}):
        rows = [i for i in indices if i["index"] == index]
        masked = [abs(i["masked_predicted"] - i["masked_observed"]) for i in rows if i["masked_observed"] is not None]
        th = [i for i in rows if i["theta_predicted"] is not None]
        pred = np.array([i["theta_predicted"] for i in th])
        true = np.array([i["theta_full_run"] for i in th])
        low = np.array([i["theta_lower"] for i in th])
        high = np.array([i["theta_upper"] for i in th])
        width = high - low
        scale = INDEX_THETA_INTERVAL_SCALE.get(index, 1.0)
        wide_low, wide_high = pred - scale * (pred - low), pred + scale * (high - pred)
        agree, pairs = pairwise_agreement(pred, true)
        out[index] = {
            "models": len(th), "null_theta": len(rows) - len(th),
            "masked_mae": float(np.mean(masked)),
            "theta_mae": float(np.mean(np.abs(pred - true))),
            "theta_median_ae": float(np.median(np.abs(pred - true))),
            "score_mae": SCALE * float(np.mean(np.abs(pred - true))),
            "score_max_ae": SCALE * float(np.max(np.abs(pred - true))),
            "coverage_90": float(np.mean((low <= true) & (true <= high))),
            "needed_scale": needed_scale(pred, low, high, true),
            "published_scale": scale,
            "coverage_90_published": float(np.mean((wide_low <= true) & (true <= wide_high))),
            "bias_theta": float(np.mean(pred - true)),
            "width_median_theta": float(np.median(width)),
            "width_mean_theta": float(np.mean(width)),
            "width_mean_score": SCALE * float(np.mean(width)),
            "distinct_share": distinct_share(list(zip(low, high))),
            "spearman": float(scipy.stats.spearmanr(pred, true).statistic),
            "kendall": float(scipy.stats.kendalltau(pred, true).statistic),
            "pairwise_agreement": agree, "pairs": pairs,
        }
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise():
    in_sample, refit_all, lomo, schemes = load_folds()
    models = in_sample["models"]
    assert sorted(f["model"] for f in lomo) == sorted(models), (len(lomo), len(models))
    summary, index_rows, task_rows = [], [], []
    flat = {name: flatten(rows) for name, rows in schemes.items()}
    for name, (tasks, indices, overall) in flat.items():
        tm, per_task = task_metrics(tasks)
        im = index_metrics(indices)
        overall_errors = [abs(o["masked_predicted"] - o["masked_observed"]) for o in overall if o["masked_observed"] is not None]
        summary.append({"scheme": name, "models": len(overall), **tm,
                        "overall_masked_mae": float(np.mean(overall_errors)),
                        "overall_masked_median_ae": float(np.median(overall_errors)),
                        "index_masked_mae_mean": float(np.mean([v["masked_mae"] for v in im.values()])),
                        "index_score_mae_mean": float(np.mean([v["score_mae"] for v in im.values()]))})
        index_rows += [{"scheme": name, "index": k, **v} for k, v in im.items()]
        task_rows += [{"scheme": name, "task": k, **v} for k, v in per_task.items()]
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "index_summary.csv", index_rows)
    write_csv(OUT / "task_summary.csv", task_rows)

    # Per-model index table (observed-vs-predicted scatter), in-sample and LOMO.
    base = {(i["model"], i["index"]): i for i in flat["in_sample"][1]}
    per_model = []
    for name in ("in_sample", "lomo"):
        for i in flat[name][1]:
            if i["theta_predicted"] is None:
                continue
            ref = base[(i["model"], i["index"])]
            per_model.append({
                "scheme": name, "model": i["model"], "index": i["index"],
                "theta_observed": i["theta_full_run"], "theta_predicted": i["theta_predicted"],
                "theta_lower": i["theta_lower"], "theta_upper": i["theta_upper"],
                "score_observed": 100 + SCALE * i["theta_full_run"], "score_predicted": 100 + SCALE * i["theta_predicted"],
                "score_lower": 100 + SCALE * i["theta_lower"], "score_upper": 100 + SCALE * i["theta_upper"],
                "inside_90": i["theta_lower"] <= i["theta_full_run"] <= i["theta_upper"],
                "theta_observed_production_calibration": ref["theta_full_run"],
                "masked_observed": i["masked_observed"], "masked_predicted": i["masked_predicted"],
                "tasks_used": i["tasks_used"], "tasks_total": i["tasks_total"],
            })
    write_csv(OUT / "per_model_index.csv", per_model)
    per_task_model = [
        {"scheme": name, **{k: t[k] for k in ("model", "task", "index", "benchmark_score", "predicted_score",
                                                "subset_answered", "subset_items")}}
        for name in ("in_sample", "lomo") for t in flat[name][0] if t["benchmark_score"] is not None
    ]
    write_csv(OUT / "per_model_task.csv", per_task_model)

    # Calibration drift: the same full-run answers scored under each fold's calibration vs production.
    drift = [abs(r["theta_observed"] - r["theta_observed_production_calibration"]) for r in per_model if r["scheme"] == "lomo"]
    meta = json.loads((OUT / "meta.json").read_text())
    print(f"folds={len(lomo)} models={len(models)} runtime={meta['seconds']:.0f}s workers={meta['workers']} "
          f"fold median {statistics.median(f['seconds'] for f in lomo):.0f}s")
    print(f"calibration drift |theta_full(fold)-theta_full(prod)|: mean {np.mean(drift):.4f} max {np.max(drift):.4f}")
    for s in summary:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    for r in index_rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nWorst LOMO tasks vs in-sample (MAE pp):")
    ins = {r["task"]: r for r in task_rows if r["scheme"] == "in_sample"}
    for r in sorted((r for r in task_rows if r["scheme"] == "lomo"), key=lambda r: -r["mae"]):
        print(f"  {r['task']:28s} {r['index']:12s} in {ins[r['task']]['mae']*100:5.2f}  lomo {r['mae']*100:5.2f}  "
              f"tau {ins[r['task']]['kendall_tau']:.3f}->{r['kendall_tau']:.3f} n={r['models']}")
    print("\nLargest LOMO index errors (score points):")
    lo = sorted((r for r in per_model if r["scheme"] == "lomo"), key=lambda r: -abs(r["score_predicted"] - r["score_observed"]))
    for r in lo[:10]:
        print(f"  {r['model'][-40:]:40s} {r['index']:12s} obs {r['score_observed']:6.1f} pred {r['score_predicted']:6.1f} "
              f"[{r['score_lower']:.1f},{r['score_upper']:.1f}] inside={r['inside_90']}")



# ---------------------------------------------------------------- calibration panel
def panel():
    """Models, scored responses and models per benchmark in the refit input -> out/panel.json."""
    pat = re.compile(r'"model":"([^"]*)".*?"task":"([^"]*)"')
    responses, per_task = Counter(), defaultdict(set)
    with open(SAMPLES) as f:
        for line in f:
            m, t = pat.search(line).groups()
            responses[m] += 1
            per_task[t].add(m)
    models = sorted(responses)
    out = {"models": models, "responses": sum(responses.values()),
           "per_task_models": {t: len(s) for t, s in sorted(per_task.items())},
           "tasks_per_model": {m: sum(m in s for s in per_task.values()) for m in models}}
    (HERE / "out/panel.json").write_text(json.dumps(out, indent=1))
    print("panel:", len(models), "models,", out["responses"], "responses,", out["per_task_models"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary", action="store_true", help="skip the refit; summarise the saved folds")
    args = parser.parse_args()
    if not args.summary:
        refit(args)
    summarise()
    panel()


if __name__ == "__main__":
    main()
