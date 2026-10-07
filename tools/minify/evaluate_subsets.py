"""``minify stats``: score a subset by comparing complai.predict output to full-run ground truth.

All prediction (ability re-estimation, imputation, item probabilities, index
scores) comes from ``complai.predict``; this module only compares those
predictions against ground truth and prints a report.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import collections
from typing import Any

import numpy as np
import scipy.stats

from complai.gp_irt import Estimator
from complai.irt import index_thetas
from complai.predict import (
    DEFAULT_IMPUTATION,
    ImputationMethod,
    Prediction,
    gap_reason,
    label_pools,
    predict_detailed,
)
from tools.minify.ground_truth import (
    GroundTruth,
    load_ground_truth,
    masked_task_scores,
    masked_label_scores,
    masked_pairs,
)


def find_subset_dirs(data_dir: Path) -> list[Path]:
    if (data_dir / "subset.jsonl").exists():
        return [data_dir]
    return [sub for sub in sorted(data_dir.iterdir()) if sub.is_dir() and (sub / "subset.jsonl").exists()]


def task_predictions(prediction: Prediction, model: str) -> dict[str, float]:
    """Task -> predicted task score, leaving out tasks that are missing or imputed for the model.

    Every task score here therefore comes from the model's responses to that task.
    """
    return {
        task: task_result["predicted_score"]
        for task, task_result in prediction.result["models"][model]["tasks"].items()
        if task_result["status"] not in ("missing", "imputed")
    }


def task_errors(prediction: Prediction, truth: GroundTruth) -> dict[str, Any]:
    """Per-task |predicted - true| for models with ground truth, plus rank stats."""
    per_task: dict[str, list[float]] = collections.defaultdict(list)
    taus: list[float] = []
    true_gaps: list[float] = []
    pred_gaps: list[float] = []
    spreads: list[float] = []
    predicted = {model: task_predictions(prediction, model) for model in prediction.result["models"]}
    tasks = sorted({task for per_model in predicted.values() for task in per_model})
    for task in tasks:
        true_scores = truth.benchmark_scores.get(task, {})
        predicted_all = [per_model[task] for per_model in predicted.values() if task in per_model]
        pairs = [
            (true_scores[model], per_model[task])
            for model, per_model in predicted.items()
            if task in per_model and model in true_scores
        ]
        if predicted_all:
            spreads.append(float(np.std(predicted_all)))
        for true_score, predicted_score in pairs:
            per_task[task].append(abs(predicted_score - true_score))
        if len(pairs) > 1:
            tau, _ = scipy.stats.kendalltau([p[0] for p in pairs], [p[1] for p in pairs])
            if np.isfinite(tau):
                taus.append(float(tau))
            true_gaps.append(float(np.mean(np.diff(sorted(p[0] for p in pairs)))))
            pred_gaps.append(float(np.mean(np.diff(sorted(p[1] for p in pairs)))))
    return {
        "per_task": per_task,
        "taus": taus,
        "true_gaps": true_gaps,
        "pred_gaps": pred_gaps,
        "spreads": spreads,
    }


def overall_errors(prediction: Prediction, truth: GroundTruth) -> list[float]:
    """Per model, masked |mean predicted - mean observed| pooled over every answered item."""
    errors: list[float] = []
    for model in prediction.result["models"]:
        predicted: list[float] = []
        observed: list[float] = []
        for name in prediction.item_probabilities.get(model, {}):
            p, o = masked_pairs(prediction, truth, model, name)
            predicted += p
            observed += o
        if observed:
            errors.append(abs(float(np.mean(predicted)) - float(np.mean(observed))))
    return errors


def masked_task_errors(prediction: Prediction, truth: GroundTruth) -> dict[str, dict[str, Any]]:
    """Masked MAE per task: predicted vs observed item means over the items each model answered.

    Imputed tasks are skipped: their error measures imputation, not the fit.
    """
    models = prediction.result["models"]
    first_model = next(iter(models.values()))
    masked = masked_task_scores(prediction, truth)
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(prediction.populations):
        scores = [
            masked[model][name]
            for model, model_result in models.items()
            if name in masked.get(model, {}) and not model_result["tasks"][name].get("imputed")
        ]
        out[name] = {
            "population_items": len(prediction.populations[name]),
            "subset_items": first_model["tasks"][name]["subset_items"],
            "masked_mae": float(np.mean([abs(s["predicted_score"] - s["observed_score"]) for s in scores])) if scores else None,
            "avg_gt_items": int(np.mean([s["items"] for s in scores])) if scores else 0,
            "models": len(scores),
        }
    return out


def label_errors(prediction: Prediction, truth: GroundTruth, column: str) -> dict[str, dict[str, Any]]:
    """Masked MAE per label value: predicted vs observed item means over full-pool items with ground truth."""
    pools = label_pools(prediction, column)
    masked = masked_label_scores(prediction, truth, column)
    subset_ids = {str(row["item_id"]) for row in prediction.subset}
    out: dict[str, dict[str, Any]] = {}
    for label, members in sorted(pools.items()):
        scores = [per_label[label] for per_label in masked.values() if label in per_label]
        errors = [abs(s["predicted_score"] - s["observed_score"]) for s in scores]
        subset_items = sum(
            1 for task, index in members
            if str(prediction.populations[task][index]["item_id"]) in subset_ids
        )
        out[label] = {
            "population_items": len(members),
            "subset_items": subset_items,
            "masked_mae": float(np.mean(errors)) if errors else None,
            "gap": gap_reason(subset_items),
            "avg_gt_items": int(np.mean([s["items"] for s in scores])) if scores else 0,
            "models": len(errors),
        }
    return out


def index_theta_errors(prediction: Prediction, truth: GroundTruth) -> dict[str, dict[str, Any]]:
    """Per index, the predicted index theta against the index theta of the full run.

    The full-run theta is fitted, with the same frozen calibration, on the
    observed scores of exactly the tasks the predicted theta used. Also
    reports how often it falls inside the predicted interval, the interval
    width, and the rank agreement between the index theta and the predicted
    index score.
    """
    masked = masked_task_scores(prediction, truth)
    out: dict[str, dict[str, Any]] = {}
    for index, calibration in sorted(prediction.params.get("indices", {}).items()):
        names = sorted(calibration["tasks"])
        a = np.asarray([calibration["tasks"][name]["discrimination"] for name in names])
        c = np.asarray([calibration["tasks"][name]["intercept"] for name in names])
        errors, inside, widths, thetas, scores, intervals = [], [], [], [], [], []
        missing = 0
        for model, model_result in prediction.result["models"].items():
            entry = model_result.get("indices", {}).get(index)
            estimate = entry.get("index_theta") if entry else None
            if estimate is None:
                missing += 1
                continue
            observed = np.full(len(names), np.nan)
            for position, name in enumerate(names):
                result = model_result["tasks"].get(name, {})
                if result.get("imputed") or result.get("status") not in ("ok", "partial"):
                    continue
                if name in masked.get(model, {}):
                    observed[position] = masked[model][name]["observed_score"]
            full_run = float(index_thetas(observed[None, :], a, c)[0])
            low, high = estimate["interval"]
            errors.append(abs(estimate["theta"] - full_run))
            inside.append(low <= full_run <= high)
            widths.append(high - low)
            intervals.append((low, high))
            thetas.append(estimate["theta"])
            scores.append(entry["predicted_score"])
        out[index] = {
            "tasks": len(names),
            "models": len(errors),
            "missing": missing,
            "theta_mae": float(np.mean(errors)) if errors else None,
            "interval_hit_rate": float(np.mean(inside)) if inside else None,
            "median_width": float(np.median(widths)) if widths else None,
            "distinguishable_pairs": distinguishable_share(intervals),
            "score_rank_correlation": (
                float(scipy.stats.spearmanr(thetas, scores).statistic) if len(thetas) > 2 else None
            ),
        }
    return out


def distinguishable_share(intervals: list[tuple[float, float]]) -> float | None:
    """Share of model pairs whose index theta intervals do not overlap."""
    pairs = [
        first[1] < second[0] or second[1] < first[0]
        for index, first in enumerate(intervals)
        for second in intervals[index + 1:]
    ]
    return float(np.mean(pairs)) if pairs else None


def print_index_theta_table(stats: dict[str, dict[str, Any]]) -> None:
    if not stats:
        return
    width = max(len("Index"), *(len(index) for index in stats)) + 1
    print("\n  --- Index Theta ---")
    print(
        f"    {'Index':<{width}} | {'Tasks':>5} | {'Models':>6} | {'Null':>4} | {'Theta MAE':>9}"
        f" | {'In 90% CI':>9} | {'Width':>5} | {'Distinct':>8} | {'Rank vs score':>13}"
    )
    print("    " + "-" * (width + 83))
    for index, row in stats.items():
        def number(value: float | None, pattern: str, size: int) -> str:
            return format(value, pattern) if value is not None else f"{'n/a':>{size}}"
        print(
            f"    {index:<{width}} | {row['tasks']:>5} | {row['models']:>6} | {row['missing']:>4}"
            f" | {number(row['theta_mae'], '>9.3f', 9)}"
            f" | {number(None if row['interval_hit_rate'] is None else row['interval_hit_rate'] * 100, '>8.1f', 8)}%"
            f" | {number(row['median_width'], '>5.2f', 5)}"
            f" | {number(None if row['distinguishable_pairs'] is None else row['distinguishable_pairs'] * 100, '>7.1f', 7)}%"
            f" | {number(row['score_rank_correlation'], '>13.3f', 13)}"
        )
    print("    Distinct: share of model pairs whose 90% intervals do not overlap.")


def rest_correlations(prediction: Prediction, truth: GroundTruth) -> dict[str, dict[str, Any]]:
    """Per index and task, the slope and the rest correlation on the full-run scores.

    A model's full-run score on a task is its mean over all the task items it
    answered. The rest correlation is the Spearman correlation, across models,
    between that score and the index theta fitted, with the index calibration
    frozen, on the model's scores for the index's other tasks. Null when fewer
    than three models have both values.
    """
    masked = masked_task_scores(prediction, truth)
    out: dict[str, dict[str, Any]] = {}
    for index, calibration in sorted(prediction.params.get("indices", {}).items()):
        names = sorted(calibration["tasks"])
        a = np.asarray([calibration["tasks"][name]["discrimination"] for name in names])
        c = np.asarray([calibration["tasks"][name]["intercept"] for name in names])
        scores = np.asarray([
            [masked[model][name]["observed_score"] if name in masked[model] else np.nan for name in names]
            for model in masked
        ]).reshape(-1, len(names))
        out[index] = {}
        for column, name in enumerate(names):
            rest = np.delete(np.arange(len(names)), column)
            usable = np.isfinite(scores[:, column]) & np.any(np.isfinite(scores[:, rest]), axis=1)
            correlation = None
            if np.sum(usable) >= 3:
                theta = index_thetas(scores[usable][:, rest], a[rest], c[rest])
                value = scipy.stats.spearmanr(scores[usable, column], theta).statistic
                correlation = float(value) if np.isfinite(value) else None
            out[index][name] = {
                "models": int(np.sum(usable)),
                "discrimination": float(a[column]),
                "rest_correlation": correlation,
            }
    return out


def print_membership_table(stats: dict[str, dict[str, Any]]) -> None:
    rows = [
        (index, name, entry)
        for index, entries in stats.items()
        for name, entry in sorted(entries.items(), key=lambda kv: (kv[1]["rest_correlation"] is None, kv[1]["rest_correlation"] or 0.0))
    ]
    if not rows:
        return
    index_width = max(len("Index"), *(len(index) for index, _, _ in rows)) + 1
    task_width = max(len("Task"), *(len(name) for _, name, _ in rows)) + 1
    print("\n  --- Task Membership ---")
    print(
        f"    {'Index':<{index_width}} | {'Task':<{task_width}} | {'Models':>6}"
        f" | {'Discrimination':>14} | {'Rest corr':>9}"
    )
    print("    " + "-" * (index_width + task_width + 42))
    for index, name, entry in rows:
        rest = entry["rest_correlation"]
        rest_text = format(rest, ">9.2f") if rest is not None else f"{'n/a':>9}"
        print(
            f"    {index:<{index_width}} | {name:<{task_width}} | {entry['models']:>6}"
            f" | {entry['discrimination']:>14.2f}"
            f" | {rest_text}"
        )
    print(
        "    Rest corr: Spearman correlation, over models, between the task's full-run"
        " score and the index theta fitted on the index's other tasks"
        " (1 = same ranking, 0 = unrelated, negative = opposite). Sorted lowest first."
    )


def print_label_table(title: str, stats: dict[str, dict[str, Any]], heading: str | None = None) -> None:
    width = max([len(title)] + [len(label) for label in stats]) + 1
    print(f"\n  --- {heading or f'Per-{title} Breakdown'} ---")
    print(f"    {title:<{width}} | {'Subset':>6} | {'Pool':>6} | {'Masked MAE':>10} | {'Avg GT':>8} | {'Models':>6} | Gap")
    print("    " + "-" * (width + 60))
    for label, row in stats.items():
        mae = f"{row['masked_mae'] * 100:>9.2f}%" if row["masked_mae"] is not None else f"{'n/a':>10}"
        gap = f"GAP ({row['gap']})" if row.get("gap") else ""
        print(f"    {label:<{width}} | {row['subset_items']:>6} | {row['population_items']:>6} | {mae} | {row['avg_gt_items']:>8} | {row['models']:>6} | {gap}")
    maes = [r["masked_mae"] for r in stats.values() if r["masked_mae"] is not None and not r.get("gap")]
    if maes:
        gaps = sum(1 for r in stats.values() if r.get("gap"))
        excluded = f", {gaps} GAP rows excluded" if gaps else ""
        print(f"\n  Overall {title} MAE (Masked): {np.mean(maes) * 100:.2f}% ({len(maes)} {title.lower()} rows{excluded})")


def evaluate_stats(
    data_dir: Path = Path("src/complai/data"),
    logs_dir: Path = Path("logs/"),
    estimator: Estimator | None = None,
    imputation: ImputationMethod = DEFAULT_IMPUTATION,
) -> None:
    print("Loading ground truth...")
    truth = load_ground_truth(logs_dir)

    subset_dirs = find_subset_dirs(data_dir)
    if not subset_dirs:
        print(f"No subset.jsonl found in {data_dir} or its subdirectories.")
        return

    print("\n=========================================")
    print("        SUBSET EVALUATION RESULTS        ")
    print("=========================================\n")

    for target_dir in subset_dirs:
        print(f"\n--- Evaluating {target_dir.name} ---")
        prediction = predict_detailed(
            truth.samples_path, target_dir / "params.json", target_dir / "subset.jsonl",
            estimator=estimator, imputation=imputation,
        )
        result = prediction.result
        print(f"  Estimator: {result.get('estimator', 'irt')}")
        print(f"  Imputation: {result.get('imputation', DEFAULT_IMPUTATION)}")
        subset_items = sum(
            next(iter(result["models"].values()))["tasks"][t]["subset_items"]
            for t in next(iter(result["models"].values()))["tasks"]
        )
        pool_size = truth.pool_size
        percent = f"{subset_items / pool_size * 100:.1f}%" if pool_size else "0%"
        print(f"  Coverage: {subset_items:,} items ({percent} of {pool_size:,} available items)")
        overall = overall_errors(prediction, truth)
        if overall:
            print(f"  Overall Score MAE (Masked): {np.mean(overall) * 100:.2f}% ({len(overall)} models)")
        print_label_table("Index", label_errors(prediction, truth, "index"))
        print_label_table("Sub-category", label_errors(prediction, truth, "subcategory"))
        print_label_table("Task", masked_task_errors(prediction, truth), "Per-Task Item Breakdown")
        print_index_theta_table(index_theta_errors(prediction, truth))
        print_membership_table(rest_correlations(prediction, truth))

        bench = task_errors(prediction, truth)
        print("\n  --- Per-Task Benchmark Score Breakdown ---")
        first_model = next(iter(result["models"].values()))
        task_items = {task: task_result["subset_items"] for task, task_result in first_model["tasks"].items()}
        for task, errors in sorted(bench["per_task"].items(), key=lambda kv: np.mean(kv[1]), reverse=True):
            print(f"    {task:<25} | {task_items[task]:>4} items | {np.mean(errors) * 100:>5.2f}% MAE ({len(errors)} models)")
        all_errors = [e for errors in bench["per_task"].values() for e in errors]
        print("  -------------------------------")
        if all_errors:
            print(f"  Avg Task MAE: {np.mean(all_errors) * 100:.2f}%\n")
        if bench["spreads"]:
            print(f"  Avg Pred. Score Spread: {np.mean(bench['spreads']):.4f} (Standard Deviation)")
        if bench["taus"]:
            print(f"  Pairwise Rank Accuracy: {(np.mean(bench['taus']) + 1.0) / 2.0 * 100:.1f}%")
        if bench["true_gaps"] and bench["pred_gaps"]:
            print(f"  Avg True Model Gap:     {np.mean(bench['true_gaps']) * 100:.2f}%")
            print(f"  Avg Predicted Gap:      {np.mean(bench['pred_gaps']) * 100:.2f}%")
        ran = [
            task_result
            for model_result in result["models"].values()
            for name, task_result in model_result["tasks"].items()
            if task_result["status"] in ("ok", "partial")
        ]
        if ran:
            at_bounds = sum(1 for t in ran if t.get("observed_subset_score") in (0.0, 1.0))
            print(f"  Subset score at 0/1:    {at_bounds / len(ran) * 100:.1f}% of model-task pairs")
        imputed = sum(m.get("imputed_tasks", 0) for m in result["models"].values())
        print(f"  Imputed model-task pairs: {imputed}")
        print("\n" + "-" * 40 + "\n")
