"""Fit GP-IRT models and select reduced evaluation sets."""

import hashlib
import json
import math
import os
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np

from complai.core.records import PreprocessedRecords


METHOD_VERSION = "dispersion23-gp-irt-2pl-v1"
PARAMS_SCHEMA = "complai-core-params-v1"
SCORE_LABEL_MAPS = {
    "accuracy_and_honesty/accuracy": {
        "correct": 1.0,
        "incorrect": 0.0,
        "no-belief": 0.0,
        "no-belief-elicitation-done": 0.0,
    },
    "accuracy_and_honesty/honesty": {
        "honest": 1.0,
        "lie": 0.0,
        "evade": 1.0,
        "no-belief": 1.0,
        "error": 1.0,
    },
    "accuracy_and_honesty/honesty@n": {
        "honest": 1.0,
        "lie": 0.0,
        "evade": 1.0,
        "no-belief": 1.0,
        "error": 1.0,
    },
}
SCORE_ALIASES = {"simpleqa_scorer/correct": "schema_tool_graded_scorer"}


@dataclass(frozen=True)
class TwoPLFit:
    """Fitted parameters and diagnostics for a 2PL model."""

    abilities: np.ndarray
    difficulties: np.ndarray
    discriminations: np.ndarray
    intercepts: np.ndarray
    slope_identified: np.ndarray
    iterations: int
    converged: bool
    log_loss: float


@dataclass(frozen=True)
class FitResult:
    """A fitted params and selected subset."""

    params: dict[str, Any]
    subset: tuple[dict[str, Any], ...]


DuplicatePolicy = Literal["error", "latest", "mean"]


def fit(
    records: PreprocessedRecords,
    scorers: dict[str, str],
    budget: int,
    *,
    floor: int = 10,
    seed: int = 0,
    duplicate_policy: DuplicatePolicy = "error",
    _ignore_unseen_tasks: bool = False,
) -> FitResult:
    """Fit and select from a normalized sample source."""
    if not scorers or any(not task or not scorer for task, scorer in scorers.items()):
        raise ValueError("scorers must be a non-empty task-to-scorer mapping")
    if budget <= 0:
        raise ValueError("budget must be positive")

    if _ignore_unseen_tasks:
        scorers = filter_seen_scorers(records, scorers)

    tasks = prepare_tasks(records, scorers, duplicate_policy)
    total_items = sum(len(task["items"]) for task in tasks.values())
    if budget > total_items:
        raise ValueError(f"budget {budget} exceeds the available items ({total_items})")

    # Fit 2PL models
    fits, capacities, dispersions = fit_tasks(tasks)

    # Select items using dispersion^2/3 allocation
    allocation, selected_keys = select_items(
        capacities, dispersions, budget, floor, seed
    )

    return build_result(
        records=records,
        scorers=scorers,
        budget=budget,
        seed=seed,
        duplicate_policy=duplicate_policy,
        tasks=tasks,
        fits=fits,
        capacities=capacities,
        dispersions=dispersions,
        allocation=allocation,
        selected_keys=selected_keys,
    )


def filter_seen_scorers(
    records: PreprocessedRecords, scorers: dict[str, str]
) -> dict[str, str]:
    """Keep only tasks supplied in the scorer mapping."""
    seen_tasks = {
        str(row["task"]) for row in records.files if row["parse_status"] == "ok"
    }
    filtered = {task: scorer for task, scorer in scorers.items() if task in seen_tasks}
    if not filtered:
        known = ", ".join(sorted(seen_tasks)) or "none"
        raise ValueError(
            "No preprocessed tasks match the scorer mapping; "
            f"found tasks: {known}. Supply --config for custom tasks."
        )

    return filtered


def prepare_tasks(
    records: PreprocessedRecords,
    scorers: dict[str, str],
    duplicate_policy: DuplicatePolicy,
    *,
    _min_models: int = 3,
) -> dict[str, dict[str, Any]]:
    """Build model-by-item score matrices from preprocessed records."""
    selected_files, canonical_datasets = _latest_content_versions(records, scorers)

    epochs: dict[tuple[str, str, str, str], list[float]] = {}
    metadata: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    content_by_question: dict[tuple[str, str], str] = {}
    for sample_row in records.iter_samples():
        task = str(sample_row["task"])
        if task not in scorers:
            continue
        file_path = str(sample_row["file_path"])
        if file_path not in selected_files:
            continue
        scorer = scorers[task]
        value: float | None
        if isinstance(sample_row, dict) and "score" in sample_row:
            value = float(sample_row["score"])
        else:
            scores = (
                sample_row["scores"]
                if isinstance(sample_row, dict) and "scores" in sample_row
                else json.loads(sample_row["scores_json"])
            )
            raw_score = select_score(scores, scorer)
            if raw_score is None:
                raise ValueError(
                    f"Configured scorer {scorer!r} is missing for task {task!r}, "
                    f"sample {sample_row['sample_id']!r} in {sample_row['file_path']}"
                )
            value = normalize_score(raw_score, scorer)
        if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Configured score {task}:{scorer} for sample {sample_row['sample_id']!r} "
                f"in {sample_row['file_path']} is not a scalar value in [0, 1]"
            )
        content_hash = str(sample_row["content_hash"])
        question_hash = str(sample_row.get("question_hash", content_hash))
        previous = content_by_question.setdefault((task, question_hash), content_hash)
        if previous != content_hash:
            raise ValueError(
                f"Conflicting scoring content for logical question {task}::{question_hash}"
            )
        sample_id = str(sample_row["sample_id"])
        key = (file_path, str(sample_row["model"]), task, question_hash)
        epochs.setdefault(key, []).append(value)
        metadata[key] = {
            "file_path": file_path,
            "created": str(sample_row["created"]),
            "model": str(sample_row["model"]),
            "task": task,
            "dataset": str(sample_row["dataset"]),
            "sample_id": sample_id,
            "question_hash": question_hash,
            "content_hash": content_hash,
        }

    runs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for key, values in epochs.items():
        record = {**metadata[key], "value": float(np.mean(values))}
        runs.setdefault(
            (record["model"], record["task"], record["question_hash"]), []
        ).append(record)
    resolved: list[dict[str, Any]] = []
    for group_key, run_records in runs.items():
        if len(run_records) == 1:
            resolved.append(run_records[0])
            continue
        match duplicate_policy:
            case "error":
                raise ValueError(
                    f"Duplicate successful evaluations for model={group_key[0]!r}, "
                    f"question={group_key[2]!r}; "
                    "use --duplicates mean or latest"
                )
            case "mean":
                resolved.append(
                    {
                        **run_records[0],
                        "value": float(np.mean([row["value"] for row in run_records])),
                    }
                )
            case "latest":
                latest = max(row["created"] for row in run_records)
                winners = [row for row in run_records if row["created"] == latest]
                resolved.append(winners[0])

    # Create (model x sample scores) matrix for each task
    output: dict[str, dict[str, Any]] = {}
    for task_name in sorted(scorers):
        rows = [row for row in resolved if row["task"] == task_name]
        if not rows:
            raise ValueError(
                f"No eligible samples found for configured task {task_name!r}"
            )

        models = sorted({row["model"] for row in rows})
        if len(models) < _min_models:
            raise ValueError(
                f"Task {task_name!r} has {len(models)} contributing models; "
                f"Supply at least {_min_models}, or exclude the task."
            )

        rows_by_question: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            rows_by_question.setdefault(row["question_hash"], []).append(row)
        items = []
        for question_hash, item_rows in rows_by_question.items():
            representative = max(
                item_rows,
                key=lambda row: (row["created"], row["sample_id"], row["file_path"]),
            )
            sample_id = representative["sample_id"]
            dataset = canonical_datasets[task_name]
            items.append(
                {
                    "item_id": f"{task_name}::{dataset}::{sample_id}",
                    "task": task_name,
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "question_hash": question_hash,
                    "content_hash": representative["content_hash"],
                }
            )
        items.sort(key=lambda item: item["item_id"])
        if len({item["item_id"] for item in items}) != len(items):
            raise ValueError(f"Latest sample IDs are not unique for task {task_name!r}")
        model_index = {model: index for index, model in enumerate(models)}
        item_index = {item["question_hash"]: index for index, item in enumerate(items)}
        matrix = np.full((len(models), len(items)), np.nan)
        for row in rows:
            matrix[model_index[row["model"]], item_index[row["question_hash"]]] = row[
                "value"
            ]

        output[task_name] = {"models": models, "items": items, "matrix": matrix}

    return output


def _latest_content_versions(
    records: PreprocessedRecords, scorers: dict[str, str]
) -> tuple[set[str], dict[str, str]]:
    """Select the newest question set while allowing dataset renames."""
    file_rows = {
        str(row["path"]): row
        for row in records.files
        if row["parse_status"] == "ok" and str(row["task"]) in scorers
    }
    questions_by_file: dict[str, set[str]] = {path: set() for path in file_rows}
    for row in records.iter_samples():
        path = str(row["file_path"])
        if path in questions_by_file:
            questions_by_file[path].add(
                str(row.get("question_hash", row["content_hash"]))
            )

    selected_files: set[str] = set()
    canonical_datasets: dict[str, str] = {}
    for task in scorers:
        versions: dict[str, list[dict[str, Any]]] = {}
        for path, row in file_rows.items():
            if str(row["task"]) != task or not questions_by_file[path]:
                continue
            signature = digest_json(sorted(questions_by_file[path]))
            versions.setdefault(signature, []).append(row)
        if not versions:
            raise ValueError(f"No eligible samples found for configured task {task!r}")
        latest = max(
            max(str(row["created"]) for row in rows) for rows in versions.values()
        )
        winners = [
            signature
            for signature, rows in versions.items()
            if max(str(row["created"]) for row in rows) == latest
        ]
        if len(winners) != 1:
            raise ValueError(
                f"Cannot determine the latest content version for task {task!r}"
            )
        winner_rows = versions[winners[0]]
        selected_files.update(str(row["path"]) for row in winner_rows)
        canonical_datasets[task] = str(
            max(winner_rows, key=lambda row: (str(row["created"]), str(row["path"])))[
                "dataset"
            ]
        )
    return selected_files, canonical_datasets


def fit_tasks(
    tasks: dict[str, dict[str, Any]],
) -> tuple[dict[str, TwoPLFit], dict[str, int], dict[str, float]]:
    """Fit each task and calculate its capacity and score dispersion."""
    fits, capacities, dispersions = {}, {}, {}
    for task_name in tasks:
        matrix = tasks[task_name]["matrix"]

        # Fit model
        fits[task_name] = fit_2pl(matrix, ridge=0.01, slope_ridge=0.01, iterations=30)

        # Number of samples in task
        capacities[task_name] = matrix.shape[1]

        # Compute dispersion
        per_model = np.nanstd(matrix, axis=1)
        finite = per_model[np.isfinite(per_model)]
        dispersions[task_name] = float(np.mean(finite)) if len(finite) else 0.0

    return fits, capacities, dispersions


def fit_2pl(
    scores: np.ndarray,
    *,
    ridge: float = 0.01,
    slope_ridge: float = 0.01,
    iterations: int = 30,
) -> TwoPLFit:
    """Fit a positive-discrimination two-parameter logistic model."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("2PL scores must be a two-dimensional model-by-item matrix")
    mask = np.isfinite(values)
    if np.any(mask & ((values < 0.0) | (values > 1.0))):
        raise ValueError("2PL response values must lie in [0, 1]")
    values = np.where(mask, values, 0.0)
    n_models, n_items = values.shape
    regularization = max(float(ridge), 1e-8)
    slope_regularization = max(float(slope_ridge), 1e-8)
    observed_rows = np.any(mask, axis=1)
    item_counts = np.sum(mask, axis=0).astype(float)
    observed_items = item_counts > 0
    abilities = smoothed_logits(values, mask, axis=1)
    abilities = np.where(observed_rows, abilities, 0.0)
    discriminations = np.ones(n_items, dtype=float)
    intercepts = smoothed_logits(values, mask, axis=0)
    intercepts = np.where(observed_items, intercepts, 0.0)
    identify(abilities, discriminations, intercepts, observed_rows)
    item_sums = np.sum(np.where(mask, values, 0.0), axis=0)
    item_means = np.divide(
        item_sums, item_counts, out=np.zeros(n_items), where=item_counts > 0
    )
    variation = np.sum(np.where(mask, (values - item_means[None, :]) ** 2, 0.0), axis=0)
    slope_identified = (item_counts >= 3) & (variation > 1e-10)
    converged = False
    used_iterations = 0

    for iteration in range(max(int(iterations), 1)):
        used_iterations = iteration + 1
        predicted = sigmoid(
            abilities[:, None] * discriminations[None, :] + intercepts[None, :]
        )
        residual = np.where(mask, values - predicted, 0.0)
        variance = np.where(mask, predicted * (1.0 - predicted), 0.0)
        gradient = (
            np.sum(discriminations[None, :] * residual, axis=1)
            - regularization * abilities
        )
        information = (
            np.sum((discriminations[None, :] ** 2) * variance, axis=1) + regularization
        )
        ability_step = np.divide(
            gradient, information, out=np.zeros(n_models), where=observed_rows
        )
        ability_step = np.clip(ability_step, -1.5, 1.5)
        abilities += ability_step

        predicted = sigmoid(
            abilities[:, None] * discriminations[None, :] + intercepts[None, :]
        )
        residual = np.where(mask, values - predicted, 0.0)
        variance = np.where(mask, predicted * (1.0 - predicted), 0.0)
        theta = abilities[:, None]
        g_a = np.sum(theta * residual, axis=0) - slope_regularization * (
            discriminations - 1.0
        )
        g_c = np.sum(residual, axis=0) - regularization * intercepts
        h_aa = np.sum((theta**2) * variance, axis=0) + slope_regularization
        h_ac = np.sum(theta * variance, axis=0)
        h_cc = np.sum(variance, axis=0) + regularization
        determinant = h_aa * h_cc - h_ac * h_ac
        valid = observed_items & (determinant > 1e-12)
        delta_a = np.divide(
            g_a * h_cc - g_c * h_ac,
            determinant,
            out=np.zeros(n_items),
            where=valid & slope_identified,
        )
        delta_c = np.divide(
            g_c * h_aa - g_a * h_ac,
            determinant,
            out=np.zeros(n_items),
            where=valid & slope_identified,
        )
        intercept_only = observed_items & ~slope_identified
        delta_c = np.where(
            intercept_only,
            np.divide(g_c, h_cc, out=np.zeros(n_items), where=h_cc > 0),
            delta_c,
        )
        delta_a = np.clip(delta_a, -0.75, 0.75)
        delta_c = np.clip(delta_c, -2.0, 2.0)
        discriminations = np.where(
            slope_identified, np.clip(discriminations + delta_a, 0.05, 5.0), 1.0
        )
        intercepts += delta_c
        identify(abilities, discriminations, intercepts, observed_rows)
        if (
            max(
                float(np.max(np.abs(ability_step))),
                float(np.max(np.abs(delta_a))),
                float(np.max(np.abs(delta_c))),
            )
            < 1e-6
        ):
            converged = True
            break

    predicted = np.clip(
        sigmoid(abilities[:, None] * discriminations[None, :] + intercepts[None, :]),
        1e-12,
        1.0 - 1e-12,
    )
    loss_values = -(
        values[mask] * np.log(predicted[mask])
        + (1.0 - values[mask]) * np.log1p(-predicted[mask])
    )
    discriminations = np.where(slope_identified, discriminations, 1.0)
    difficulties = np.divide(
        -intercepts, discriminations, out=np.zeros(n_items), where=discriminations > 0
    )
    return TwoPLFit(
        abilities=np.where(np.isfinite(abilities), abilities, 0.0),
        difficulties=np.where(np.isfinite(difficulties), difficulties, 0.0),
        discriminations=np.where(np.isfinite(discriminations), discriminations, 1.0),
        intercepts=np.where(np.isfinite(intercepts), intercepts, 0.0),
        slope_identified=slope_identified,
        iterations=used_iterations,
        converged=converged,
        log_loss=float(np.mean(loss_values)) if len(loss_values) else float("nan"),
    )


def select_items(
    capacities: dict[str, int],
    dispersions: dict[str, float],
    budget: int,
    floor: int,
    seed: int,
) -> tuple[dict[str, int], list[tuple[str, int]]]:
    """Allocate the budget and select items uniformly at random."""
    allocation = dispersion23_allocation(capacities, dispersions, budget, floor)

    selected_keys: list[tuple[str, int]] = []
    for task_name in sorted(capacities):
        rng = np.random.default_rng(zlib.crc32(f"stratified{task_name}{seed}".encode()))
        permuted = rng.permutation(capacities[task_name])
        n = allocation[task_name]
        selected_keys.extend((task_name, int(index)) for index in permuted[:n])

    return allocation, selected_keys


def dispersion23_allocation(
    capacities: dict[str, int], dispersions: dict[str, float], budget: int, floor: int
) -> dict[str, int]:
    r"""Allocate subset budget using $\\sigma^{2/3}$.

    Returns:
        Dict mapping task name to its allocated capacity.
    """
    assert floor >= 0, "floor must be non-negative"
    if sum(capacities.values()) <= budget:
        return capacities.copy()

    floors = {name: min(floor, capacity) for name, capacity in capacities.items()}
    weights = {name: dispersions[name] ** (2 / 3) for name in capacities}
    weights = weights if any(weights.values()) else floors.copy()
    allocation = (
        floors if sum(floors.values()) <= budget else {name: 0 for name in capacities}
    )

    names = list(capacities)
    base = dict(allocation)
    if not any(value > 0 for value in weights.values()):
        weights = {name: float(capacities[name] - allocation[name]) for name in names}
    while sum(allocation.values()) < budget:
        eligible = [name for name in names if allocation[name] < capacities[name]]
        chosen = max(
            eligible,
            key=lambda name: (
                weights[name] / (allocation[name] - base[name] + 1.0),
                capacities[name] - allocation[name],
                name,
            ),
        )
        allocation[chosen] += 1

    return allocation


def build_result(
    *,
    records: PreprocessedRecords,
    scorers: dict[str, str],
    budget: int,
    seed: int,
    duplicate_policy: Literal["error", "latest", "mean"],
    tasks: dict[str, dict[str, Any]],
    fits: dict[str, TwoPLFit],
    capacities: dict[str, int],
    dispersions: dict[str, float],
    allocation: dict[str, int],
    selected_keys: list[tuple[str, int]],
) -> FitResult:
    """Build the fitted params and ordered subset records."""
    configuration_digest = digest_json(
        {
            "task_scorers": dict(sorted(scorers.items())),
            "duplicate_policy": duplicate_policy,
            "hyperparameters": {"ridge": 0.01, "slope_ridge": 0.01, "iterations": 10},
        }
    )
    params_basis = {
        "schema": PARAMS_SCHEMA,
        "method": METHOD_VERSION,
        "inventory_digest": records.digest,
        "task_scorers": dict(sorted(scorers.items())),
        "duplicate_policy": duplicate_policy,
        "hyperparameters": {"ridge": 0.01, "slope_ridge": 0.01, "iterations": 10},
    }
    params_id = digest_json(params_basis)[:24]
    selected_ids = [
        tasks[task]["items"][index]["item_id"] for task, index in selected_keys
    ]
    subset_id = digest_json(
        {"params_id": params_id, "budget": budget, "seed": seed, "items": selected_ids}
    )[:24]
    selected_set = set(selected_ids)

    item_records: list[dict[str, Any]] = []
    task_records: dict[str, Any] = {}
    abilities: dict[str, dict[str, float]] = {}
    for task_name in sorted(tasks):
        task = tasks[task_name]
        fit = fits[task_name]
        abilities[task_name] = {
            model: float(value) for model, value in zip(task["models"], fit.abilities)
        }
        task_records[task_name] = {
            "scorer": scorers[task_name],
            "dataset": task["items"][0]["dataset"],
            "models": len(task["models"]),
            "capacity": capacities[task_name],
            "dispersion": dispersions[task_name],
            "allocation": allocation[task_name],
            "observations": int(np.sum(np.isfinite(task["matrix"]))),
            "response_coverage": float(np.mean(np.isfinite(task["matrix"]))),
            "fit": {
                "iterations": fit.iterations,
                "converged": fit.converged,
                "log_loss": fit.log_loss,
            },
        }
        for index, item in enumerate(task["items"]):
            item_id = item["item_id"]
            item_records.append(
                {
                    **item,
                    "observation_count": int(
                        np.sum(np.isfinite(task["matrix"][:, index]))
                    ),
                    "source_mean": float(np.nanmean(task["matrix"][:, index])),
                    "slope_identified": bool(fit.slope_identified[index]),
                    "difficulty": float(fit.difficulties[index]),
                    "discrimination": float(fit.discriminations[index]),
                    "intercept": float(fit.intercepts[index]),
                    "selected": item_id in selected_set,
                }
            )

    by_id = {item["item_id"]: item for item in item_records}
    selected_records = []
    for rank, (task_name, _) in enumerate(selected_keys, start=1):
        item = by_id[selected_ids[rank - 1]]
        probability = allocation[task_name] / capacities[task_name]
        selected_records.append(
            {
                "params_id": params_id,
                "subset_id": subset_id,
                "rank": rank,
                "item_id": item["item_id"],
                "task": task_name,
                "dataset": item["dataset"],
                "sample_id": item["sample_id"],
                "question_hash": item["question_hash"],
                "content_hash": item["content_hash"],
                "task_allocation": allocation[task_name],
                "inclusion_probability": probability,
                "design_weight": 1.0 / probability,
                "difficulty": item["difficulty"],
                "discrimination": item["discrimination"],
                "intercept": item["intercept"],
            }
        )

    stable_inventory = records.inventory
    params = {
        "schema_version": PARAMS_SCHEMA,
        "method": METHOD_VERSION,
        "params_id": params_id,
        "subset_id": subset_id,
        "budget": budget,
        "seed": seed,
        "duplicate_policy": duplicate_policy,
        "task_scorers": dict(sorted(scorers.items())),
        "hyperparameters": params_basis["hyperparameters"],
        "configuration_digest": configuration_digest,
        "input_digest": records.digest,
        "inventory": stable_inventory,
        "ability_scale": "Each task is independently normalized to mean 0 and standard deviation 1.",
        "tasks": task_records,
        "model_abilities": abilities,
        "items": item_records,
    }

    return FitResult(
        params=json_safe(params),
        subset=tuple(json_safe(row) for row in selected_records),
    )


def select_score(scores: dict[str, Any], scorer: str) -> Any:
    """Select a configured score, including nested and aliased scores."""
    if scorer in scores:
        return scores[scorer]
    alias = SCORE_ALIASES.get(scorer)
    if alias is not None and alias in scores:
        return scores[alias]
    top_level, separator, subkey = scorer.partition("/")
    if not separator:
        return None
    value = scores.get(top_level)
    if isinstance(value, dict):
        return value.get(subkey)
    # HLE changed from a scalar score to {"score": ..., "confidence": ...}.

    return value if subkey == "score" else None


def normalize_score(value: Any, scorer: str) -> float | None:
    """Convert a configured score to a numeric response."""
    label_map = SCORE_LABEL_MAPS.get(scorer)
    if label_map is not None and isinstance(value, str):
        return label_map.get(value.strip().lower())

    return score_value_to_float(value)


def score_value_to_float(value: Any) -> float | None:
    """Convert a standard Inspect score value to a float."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "c": 1.0,
            "correct": 1.0,
            "true": 1.0,
            "yes": 1.0,
            "pass": 1.0,
            "passed": 1.0,
            "i": 0.0,
            "incorrect": 0.0,
            "false": 0.0,
            "no": 0.0,
            "fail": 0.0,
            "failed": 0.0,
            "p": 0.5,
            "partial": 0.5,
            "partially_correct": 0.5,
            "n": 0.0,
            "noanswer": 0.0,
            "no_answer": 0.0,
            "refusal": 0.0,
        }
        if normalized in mapping:
            return mapping[normalized]
        try:
            return float(normalized)
        except ValueError:
            return None

    return None


def smoothed_logits(values: np.ndarray, mask: np.ndarray, *, axis: int) -> np.ndarray:
    """Compute smoothed empirical logits along an array axis."""
    counts = np.sum(mask, axis=axis).astype(float)
    successes = np.sum(np.where(mask, values, 0.0), axis=axis)
    probabilities = np.divide(
        successes + 0.5,
        counts + 1.0,
        out=np.full_like(successes, 0.5, dtype=float),
        where=counts > 0,
    )

    return np.log(
        np.clip(probabilities, 1e-4, 1 - 1e-4)
        / np.clip(1 - probabilities, 1e-4, 1 - 1e-4)
    )


def identify(
    abilities: np.ndarray,
    discriminations: np.ndarray,
    intercepts: np.ndarray,
    observed_rows: np.ndarray,
) -> None:
    """Normalize the ability scale and adjust item parameters."""
    if not np.any(observed_rows):
        return
    center = float(np.mean(abilities[observed_rows]))
    scale = float(np.std(abilities[observed_rows]))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    old = discriminations.copy()
    abilities[:] = (abilities - center) / scale
    discriminations[:] = np.clip(old * scale, 0.05, 5.0)
    intercepts[:] = intercepts + old * center


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid without numerical overflow."""
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)

    return output


def json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values to standard JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def digest_json(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON value."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()

    return hashlib.sha256(encoded).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    """Write text through a temporary file and atomic replacement."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_outputs(
    result: FitResult, output_dir: Path, overwrite: bool = False
) -> tuple[Path, Path]:
    """Write the fitted params and selected subset atomically."""
    params_path, subset_path = check_output_available(output_dir, overwrite)
    output_dir = params_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    params_text = (
        json.dumps(result.params, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    subset_text = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in result.subset
    )
    atomic_write(params_path, params_text)
    try:
        atomic_write(subset_path, subset_text)
    except Exception:
        params_path.unlink(missing_ok=True)
        raise

    return params_path, subset_path


def check_output_available(
    output_dir: Path, overwrite: bool = False
) -> tuple[Path, Path]:
    """Return output paths if writing them is allowed."""
    output_dir = output_dir.expanduser().resolve()
    params_path = output_dir / "params.json"
    subset_path = output_dir / "subset.jsonl"

    existing = [path for path in (params_path, subset_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}")

    return params_path, subset_path
