"""Load full-run ground truth for minification evaluation.

Shared by ``minify stats`` and ``tools/export_web``. Contains no prediction
method logic; that lives in ``complai.predict``.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from complai.predict import Prediction, label_pools, prepared_tasks
from complai.utils.log_parser import load_records, preprocess_logs
from tools.minify.config import get_primary_metrics, load_scorers

DEFAULT_SAMPLES = Path("tools/minify/data/samples.jsonl")
DEFAULT_METRICS = Path("tools/minify/data/metrics.json")
# A benchmark score is ground truth only if its eval answered this share of the task's questions.
MIN_GROUND_TRUTH_COVERAGE = 0.95


@dataclass
class GroundTruth:
    """Full-run observations for every model on every task."""

    samples_path: Path
    # task -> prepare_tasks output: {"models": [...], "items": [...], "matrix": np.ndarray}
    tasks: dict[str, Any]
    # task -> model -> primary-metric benchmark score
    benchmark_scores: dict[str, dict[str, float]]

    @property
    def pool_size(self) -> int:
        return sum(len(task["items"]) for task in self.tasks.values())

    def item_responses(self, name: str, model: str) -> dict[str, float] | None:
        """Return ``{item_id: response}`` for items the model answered, or None if it never ran the task."""
        task_data = self.tasks.get(name)
        if task_data is None or model not in task_data["models"]:
            return None
        row = task_data["matrix"][task_data["models"].index(model)]
        return {
            str(item["item_id"]): float(value)
            for item, value in zip(task_data["items"], row)
            if np.isfinite(value)
        }


def masked_pairs(
    prediction: Prediction, truth: GroundTruth, model: str, name: str
) -> tuple[list[float], list[float]]:
    """Predicted probabilities and observed responses over the task's population items the model answered."""
    probabilities = prediction.item_probabilities.get(model, {}).get(name)
    responses = truth.item_responses(name, model)
    predicted: list[float] = []
    observed: list[float] = []
    if probabilities is None or responses is None:
        return predicted, observed
    for index, row in enumerate(prediction.populations[name]):
        value = responses.get(str(row["item_id"]))
        if value is not None and np.isfinite(probabilities[index]):
            predicted.append(float(probabilities[index]))
            observed.append(value)
    return predicted, observed


def masked_task_scores(
    prediction: Prediction, truth: GroundTruth
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per model and task, the predicted and observed means over the items the model answered."""
    out: dict[str, dict[str, dict[str, float | int]]] = collections.defaultdict(dict)
    for model in prediction.result["models"]:
        for name in prediction.item_probabilities.get(model, {}):
            predicted, observed = masked_pairs(prediction, truth, model, name)
            if observed:
                out[model][name] = {
                    "predicted_score": float(np.mean(predicted)),
                    "observed_score": float(np.mean(observed)),
                    "items": len(observed),
                }
    return dict(out)


def masked_label_scores(
    prediction: Prediction,
    truth: GroundTruth,
    column: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per model and label value, the predicted and observed means over the *same* items.

    Both sides are restricted ("masked") to full-population items the model actually
    answered, so the difference is prediction error rather than coverage. Labels come
    from the decoded params ``column`` (``index`` or ``subcategory``).
    """
    out: dict[str, dict[str, dict[str, float | int]]] = collections.defaultdict(dict)
    for label, members in label_pools(prediction, column).items():
        by_task: dict[str, list[int]] = collections.defaultdict(list)
        for task, index in members:
            by_task[task].append(index)
        for model in prediction.result["models"]:
            predicted: list[float] = []
            observed: list[float] = []
            for task, indices in by_task.items():
                probabilities = prediction.item_probabilities.get(model, {}).get(task)
                responses = truth.item_responses(task, model)
                if probabilities is None or responses is None:
                    continue
                population = prediction.populations[task]
                for index in indices:
                    value = responses.get(str(population[index]["item_id"]))
                    if value is not None and np.isfinite(probabilities[index]):
                        predicted.append(float(probabilities[index]))
                        observed.append(value)
            if observed:
                out[model][label] = {
                    "predicted_score": float(np.mean(predicted)),
                    "observed_score": float(np.mean(observed)),
                    "items": len(observed),
                }
    return dict(out)


def primary_metric_value(task: str, value: Any, primary_metrics: dict[str, Any]) -> float | None:
    """Pick the primary metric out of a metrics.json entry."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return float(value)
    expected = primary_metrics.get(task, ["accuracy"])
    if isinstance(expected, str):
        expected = [expected]
    for key in expected:
        if value.get(key) is not None:
            return float(value[key])
    fallback = [v for k, v in value.items() if k not in ("cerr", "stderr") and v is not None]
    return float(fallback[0]) if fallback else None


def load_ground_truth(
    logs_dir: Path = Path("logs/"),
    samples_path: Path = DEFAULT_SAMPLES,
    metrics_path: Path = DEFAULT_METRICS,
) -> GroundTruth:
    """Load the preprocessed full-run sample matrix and benchmark-level scores.

    Benchmark scores from evals that answered fewer than
    ``MIN_GROUND_TRUTH_COVERAGE`` of the task's questions are dropped: their
    logged score describes a different question set.
    """
    scorers = load_scorers(None)
    primary_metrics = get_primary_metrics(None)
    if not samples_path.exists():
        preprocess_logs([logs_dir], scorers, samples_path)
    records = load_records(samples_path)
    tasks = prepared_tasks(records, scorers)
    coverage = {
        (row["task"], row["model"]): row.get("coverage", 1.0)
        for row in records.files
        if row["parse_status"] == "ok"
    }

    benchmark_scores: dict[str, dict[str, float]] = collections.defaultdict(dict)
    if metrics_path.exists():
        with open(metrics_path) as handle:
            for task, per_model in json.load(handle).items():
                for model, value in per_model.items():
                    if coverage.get((task, model), 1.0) < MIN_GROUND_TRUTH_COVERAGE:
                        continue
                    score = primary_metric_value(task, value, primary_metrics)
                    if score is not None:
                        benchmark_scores[task][model] = score
    else:
        print(f"WARNING: {metrics_path} not found. Using item means as benchmark scores.")
        for task, task_data in tasks.items():
            with np.errstate(invalid="ignore"):
                means = np.nanmean(task_data["matrix"], axis=1)
            for model, mean in zip(task_data["models"], means):
                if np.isfinite(mean):
                    benchmark_scores[task][model] = float(mean)
    return GroundTruth(
        samples_path=samples_path, tasks=tasks,
        benchmark_scores=dict(benchmark_scores),
    )
