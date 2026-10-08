"""Fit GP-IRT models and select reduced evaluation sets."""

import hashlib
import json
import math
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np

from complai.constants import (
    INDEX_THETA_RIDGE,
    ITEM_COLUMNS,
    MAX_DISCRIMINATION,
    METHOD_VERSION,
    MIN_DISCRIMINATION,
    MIN_INDEX_THETA_TASKS,
    PARAMS_SCHEMA,
)
from complai.labels import (
    UNKNOWN_LABEL,
    ItemLabel,
    ItemLabels,
    Subcategories,
    load_labels,
    load_subcategories,
    lookup_label,
)
from complai.utils.log_parser import PreprocessedRecords
from complai.utils.scores import normalize_score, select_score


@dataclass(frozen=True)
class TwoPLFit:
    """Fitted parameters and diagnostics for a 2PL model."""

    abilities: np.ndarray
    difficulties: np.ndarray
    discriminations: np.ndarray
    intercepts: np.ndarray
    slope_identified: np.ndarray
    slope_at_floor: np.ndarray
    iterations: int
    converged: bool
    log_loss: float


@dataclass(frozen=True)
class FitResult:
    """A fitted params and selected subset."""

    params: dict[str, Any]
    subset: tuple[dict[str, Any], ...]



def fit(
    records: PreprocessedRecords,
    scorers: dict[str, str],
    budget: int,
    *,
    floor: int | dict[str, int] = 5,
    floor_percent: float | None = None,
    seed: int = 0,
    item_selection: str = "random",
    drop_uninformative: bool = False,
    indices: dict[str, str],
    labels_dir: Path | None = None,
    estimator: Literal["irt", "gp_irt"] = "irt",
    _ignore_unseen_tasks: bool = False,
) -> FitResult:
    """Fit and select from a normalized sample source."""
    if not scorers or any(not task or not scorer for task, scorer in scorers.items()):
        raise ValueError("scorers must be a non-empty task-to-scorer mapping")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if estimator not in {"irt", "gp_irt"}:
        raise ValueError("estimator must be irt or gp_irt")

    if _ignore_unseen_tasks:
        scorers = filter_seen_scorers(records, scorers)

    missing = sorted(set(scorers) - set(indices))
    if missing:
        raise ValueError(f"tasks without an index: {', '.join(missing)}")

    # Every task belongs to one index, which alone drives the fit. Item labels
    # never affect fitting or selection; when available their secondary label is
    # written into params as reporting metadata, together with the taxonomy's
    # sub-categories so params are self-contained for reporting.
    labels = load_labels(labels_dir) if labels_dir is not None else {}
    subcategories = (
        load_subcategories() if labels_dir is not None else Subcategories(by_label={}, definitions={})
    )
    tasks = prepare_tasks(records, scorers)
    for name, task in tasks.items():
        task["index"] = indices[name]
    total_items = sum(len(task["items"]) for task in tasks.values())
    if budget > total_items:
        raise ValueError(f"budget {budget} exceeds the available items ({total_items})")

    # Fit 2PL models
    fits, capacities, dispersions = fit_tasks(tasks, estimator=estimator)
    index_calibration = fit_indices(tasks)

    # Selection draws only from eligible items.
    excluded: set[tuple[str, int]] = set()
    if drop_uninformative:
        excluded = uninformative_items(tasks, fits)
    excluded_counts = Counter(name for name, _ in excluded)
    eligible = {name: capacity - excluded_counts[name] for name, capacity in capacities.items()}
    if budget > sum(eligible.values()):
        raise ValueError(f"budget {budget} exceeds the eligible items ({sum(eligible.values())})")

    # Determine task floors.
    if floor_percent is not None:
        base_floor = max(1, int(round(budget * (floor_percent / 100.0))))
    elif isinstance(floor, int):
        base_floor = floor
    else:
        base_floor = 5

    if floor_percent is not None or isinstance(floor, dict):
        floor = {
            name: floor[name] if isinstance(floor, dict) and name in floor else base_floor
            for name in capacities
        }

    # Select items using the chosen strategy
    if item_selection in {"joint", "joint_discrimination"}:
        allocation, selected_keys = select_items_joint(
            tasks,
            eligible,
            dispersions,
            fits,
            budget,
            floor,
            excluded,
            seed=seed,
            item_selection="discrimination" if item_selection == "joint_discrimination" else "random",
        )
    elif item_selection == "smoke":
        n_items = max(1, budget // len(capacities))
        allocation = {t: min(n_items, c) for t, c in eligible.items()}
        selected_keys = []
        for t, n in allocation.items():
            kept = [i for i in range(capacities[t]) if (t, i) not in excluded]
            selected_keys.extend((t, i) for i in kept[:n])
    else:
        allocation, selected_keys = select_items(
            eligible, dispersions, fits, budget, floor, seed, excluded, item_selection
        )

    calibration = None
    if estimator == "gp_irt":
        from complai.gp_irt import calibrate_tasks
        calibration = calibrate_tasks(tasks, selected_keys)

    return build_result(
        records=records,
        scorers=scorers,
        budget=budget,
        seed=seed,
        tasks=tasks,
        fits=fits,
        capacities=capacities,
        eligible=eligible,
        dispersions=dispersions,
        allocation=allocation,
        selected_keys=selected_keys,
        labels=labels,
        subcategories=subcategories,
        indices=index_calibration,
        gp_irt=calibration,
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
    *,
    _min_models: int = 3,
    _allow_missing: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build model-by-item score matrices from preprocessed records.

    Preprocessing already keeps one evaluation per model and task, restricted to
    the task's current question set, so records map directly onto matrix cells.
    ``_allow_missing`` (prediction) leaves out configured tasks without records
    instead of raising.
    """
    canonical_datasets = _canonical_datasets(records, scorers, allow_missing=_allow_missing)

    epochs: dict[tuple[str, str, str, str], list[float]] = {}
    metadata: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    content_by_question: dict[tuple[str, str], str] = {}
    for sample_row in records.iter_samples():
        task = str(sample_row["task"])
        if task not in scorers:
            continue
        file_path = str(sample_row["file_path"])
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
                continue
            value = normalize_score(raw_score, scorer)
        if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            continue
        content_hash = str(sample_row["content_hash"])
        question_hash = str(sample_row.get("question_hash", content_hash))
        previous = content_by_question.setdefault((task, question_hash), content_hash)
        if previous != content_hash:
            # Dataset bug: two different questions have the identical prompt (so same question_hash) 
            # but different targets (so different content_hash). We split them so they don't crash.
            question_hash = content_hash
            content_by_question[(task, question_hash)] = content_hash
        
        # Ensure row uses the modified hash for downstream indexing
        sample_row["question_hash"] = question_hash
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
    for (model, task, _), run_records in runs.items():
        if len(run_records) > 1:
            paths = sorted(row["file_path"] for row in run_records)
            raise ValueError(
                f"Duplicate evaluations for model={model!r}, task={task!r} "
                f"({', '.join(paths)}); re-run preprocess to select one"
            )
        resolved.append(run_records[0])

    # Create (model x sample scores) matrix for each task
    output: dict[str, dict[str, Any]] = {}
    for task_name in sorted(scorers):
        rows = [row for row in resolved if row["task"] == task_name]
        if not rows and _allow_missing:
            continue
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


def _canonical_datasets(
    records: PreprocessedRecords, scorers: dict[str, str], *, allow_missing: bool = False
) -> dict[str, str]:
    """Name each task's items after the dataset of its newest evaluation."""
    newest: dict[str, dict[str, Any]] = {}
    for row in records.files:
        task = str(row["task"])
        if row["parse_status"] != "ok" or task not in scorers:
            continue
        if task not in newest or (str(row["created"]), str(row["path"])) > (
            str(newest[task]["created"]), str(newest[task]["path"])
        ):
            newest[task] = row
    missing = sorted(set(scorers) - set(newest))
    if missing and not allow_missing:
        raise ValueError(f"No eligible samples found for configured task {missing[0]!r}")
    return {task: str(row["dataset"]) for task, row in newest.items()}


def fit_tasks(
    tasks: dict[str, dict[str, Any]],
    *, estimator: Literal["irt", "gp_irt"] = "irt",
) -> tuple[dict[str, TwoPLFit], dict[str, int], dict[str, float]]:
    """Fit each task and calculate its capacity and score dispersion."""
    fits, capacities, dispersions = {}, {}, {}
    for task_name in tasks:
        matrix = tasks[task_name]["matrix"]

        # Fit model
        if estimator == "gp_irt":
            from complai.gp_irt import fit_model
            fits[task_name] = fit_model(matrix)
        else:
            fits[task_name] = fit_2pl(matrix, ridge=0.01, slope_ridge=0.01, iterations=30)

        # Number of samples in task
        capacities[task_name] = matrix.shape[1]

        # Compute dispersion
        per_model = np.nanstd(matrix, axis=1)
        finite = per_model[np.isfinite(per_model)]
        dispersions[task_name] = float(np.mean(finite)) if len(finite) else 0.0

    return fits, capacities, dispersions


def uninformative_items(
    tasks: dict[str, dict[str, Any]], fits: dict[str, TwoPLFit]
) -> set[tuple[str, int]]:
    """Items that cannot separate models.

    Two kinds: items every model answered fully correctly, and items whose
    fitted slope is held at ``MIN_DISCRIMINATION`` because stronger models do
    no better on them. Items every model fails are kept: stronger models may
    still solve them.
    """
    uninformative: set[tuple[str, int]] = set()
    for name, task in tasks.items():
        saturated = np.nanmin(task["matrix"], axis=0) >= 1.0
        flat = saturated | fits[name].slope_at_floor
        uninformative.update((name, int(index)) for index in np.flatnonzero(flat))
    return uninformative


def fit_indices(tasks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Calibrate every index's tasks for the index theta (Epoch-style).

    Within an index, a panel model's observed score on a task (its mean
    response over the task items it answered) is modelled as
    ``sigmoid(discrimination * theta + intercept)``: one fractional observation
    per task, so every task counts once whatever its size. Only models with
    scores on at least ``MIN_INDEX_THETA_TASKS`` of the index's tasks take part,
    and an index with fewer tasks is not calibrated. The index theta is
    normalized to mean 0 and standard deviation 1 over the participating panel
    models.
    """
    by_index: dict[str, list[str]] = {}
    for name, task in tasks.items():
        by_index.setdefault(task["index"], []).append(name)
    output: dict[str, dict[str, Any]] = {}
    for index, names in sorted(by_index.items()):
        names = sorted(names)
        if len(names) < MIN_INDEX_THETA_TASKS:
            continue
        models = sorted({model for name in names for model in tasks[name]["models"]})
        row_of = {model: row for row, model in enumerate(models)}
        scores = np.full((len(models), len(names)), np.nan)
        for column, name in enumerate(names):
            matrix = tasks[name]["matrix"]
            answered = np.isfinite(matrix)
            counts = np.sum(answered, axis=1)
            sums = np.sum(np.where(answered, matrix, 0.0), axis=1)
            for model, count, total in zip(tasks[name]["models"], counts, sums):
                if count:
                    scores[row_of[model], column] = total / count
        scores = scores[np.sum(np.isfinite(scores), axis=1) >= MIN_INDEX_THETA_TASKS]
        if not len(scores):
            continue
        fit = fit_2pl(scores, ridge=INDEX_THETA_RIDGE, slope_ridge=0.01, iterations=200)
        output[index] = {
            "models": len(scores),
            "tasks": {
                name: {
                    "discrimination": float(fit.discriminations[column]),
                    "intercept": float(fit.intercepts[column]),
                    "models": int(np.sum(np.isfinite(scores[:, column]))),
                }
                for column, name in enumerate(names)
            },
            "fit": {
                "iterations": fit.iterations,
                "converged": fit.converged,
                "log_loss": fit.log_loss,
            },
        }
    return output


def index_thetas(
    scores: np.ndarray,
    discrimination: np.ndarray,
    intercept: np.ndarray,
    *,
    iterations: int = 100,
) -> np.ndarray:
    """Fit one index theta per row of task scores with frozen calibration.

    Each score in [0, 1] is a fractional observation of
    ``sigmoid(discrimination * theta + intercept)``; NaN scores are skipped.
    """
    values = np.asarray(scores, dtype=float)
    mask = np.isfinite(values)
    values = np.clip(np.where(mask, values, 0.0), 0.0, 1.0)
    theta = np.zeros(len(values))
    for _ in range(iterations):
        predicted = sigmoid(theta[:, None] * discrimination[None, :] + intercept[None, :])
        residual = np.where(mask, values - predicted, 0.0)
        variance = np.where(mask, predicted * (1.0 - predicted), 0.0)
        gradient = residual @ discrimination - INDEX_THETA_RIDGE * theta
        information = variance @ discrimination**2 + INDEX_THETA_RIDGE
        step = np.clip(gradient / information, -1.5, 1.5)
        theta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return theta


def fit_2pl(
    scores: np.ndarray,
    *,
    ridge: float = 0.01,
    slope_ridge: float = 0.01,
    iterations: int = 30,
    _reset_unidentified_slopes: bool = True,
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
    slope_at_floor = np.zeros(n_items, dtype=bool)
    converged = False
    used_iterations = 0

    for iteration in range(max(int(iterations), 1)):
        used_iterations = iteration + 1
        previous = (abilities.copy(), discriminations.copy(), intercepts.copy())
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
        proposed = discriminations + delta_a
        slope_at_floor = slope_identified & (proposed <= MIN_DISCRIMINATION)
        discriminations = np.where(
            slope_identified,
            np.clip(proposed, MIN_DISCRIMINATION, MAX_DISCRIMINATION),
            1.0,
        )
        intercepts += delta_c
        identify(abilities, discriminations, intercepts, observed_rows)
        # Measure the change after clipping and identification: the raw Newton
        # steps stay non-zero at a fixed point where a bound or the ability
        # normalization cancels them.
        change = max(
            float(np.max(np.abs(new - old), initial=0.0))
            for new, old in zip((abilities, discriminations, intercepts), previous)
        )
        if change < 1e-6:
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
    # Minibench retains the last identification rescaling on constant/thin items.
    if _reset_unidentified_slopes:
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
        slope_at_floor=slope_at_floor,
        iterations=used_iterations,
        converged=converged,
        log_loss=float(np.mean(loss_values)) if len(loss_values) else float("nan"),
    )


def select_items(
    capacities: dict[str, int],
    dispersions: dict[str, float],
    fits: dict[str, TwoPLFit],
    budget: int,
    floor: int | dict[str, int],
    seed: int,
    excluded: set[tuple[str, int]],
    item_selection: str = "random",
) -> tuple[dict[str, int], list[tuple[str, int]]]:
    """Allocate the budget and select items based on the chosen strategy.

    ``capacities`` counts the eligible items of each task; ``excluded`` items
    are never selected.
    """
    allocation = dispersion23_allocation(capacities, dispersions, budget, floor)

    selected_keys: list[tuple[str, int]] = []
    for task_name in sorted(capacities):
        n = allocation[task_name]
        size = len(fits[task_name].discriminations)
        if item_selection == "discrimination":
            safe_disc = np.nan_to_num(fits[task_name].discriminations, nan=-np.inf)
            order = np.argsort(-safe_disc)
        else:
            rng = np.random.default_rng(zlib.crc32(f"stratified{task_name}{seed}".encode()))
            order = rng.permutation(size)
        kept = [int(index) for index in order if (task_name, int(index)) not in excluded]
        selected_keys.extend((task_name, index) for index in kept[:n])

    return allocation, selected_keys


_UNLABELED = ItemLabel(UNKNOWN_LABEL, None, ())

def item_label(labels: ItemLabels, item: dict[str, Any]) -> ItemLabel:
    """Return the label of a prepared item, falling back to ``unknown``."""
    found = lookup_label(labels, item["task"], item.get("dataset", ""), str(item["sample_id"]))
    return found or _UNLABELED


def select_items_joint(
    tasks: dict[str, dict[str, Any]],
    capacities: dict[str, int],
    dispersions: dict[str, float],
    fits: dict[str, TwoPLFit],
    budget: int,
    floor: int | dict[str, int],
    excluded: set[tuple[str, int]],
    seed: int = 0,
    item_selection: str = "random",
) -> tuple[dict[str, int], list[tuple[str, int]]]:
    """Allocate budget dynamically using a joint marginal utility strategy.

    ``capacities`` counts the eligible items of each task; ``excluded`` items
    are never selected.
    """
    import collections

    # 1. Target benchmark proportions (the soft quota)
    target_allocation = dispersion23_allocation(capacities, dispersions, budget, floor)

    # 2. Group items into (task, index) bins
    bins = collections.defaultdict(list)

    for task_name, task_data in tasks.items():
        safe_disc = None
        if task_name in fits and fits[task_name].discriminations is not None:
            discriminations = fits[task_name].discriminations
            safe_disc = np.nan_to_num(discriminations, nan=-np.inf)

        for index, item in enumerate(task_data.get("items", [])):
            if (task_name, index) in excluded:
                continue
            disc_val = safe_disc[index] if safe_disc is not None else 0.0
            bins[(task_name, task_data["index"])].append((index, disc_val))
            
    if item_selection == "discrimination":
        # Sort each bin descending by discrimination
        for k in bins:
            bins[k].sort(key=lambda x: x[1], reverse=True)
    else:
        # Random sampling: shuffle items within each bin deterministically
        for k in sorted(bins.keys()):
            bin_rng = np.random.default_rng(zlib.crc32(f"joint_{k[0]}_{k[1]}_{seed}".encode()))
            bin_rng.shuffle(bins[k])
        
    allocated_task = collections.defaultdict(int)
    allocated_index = collections.defaultdict(int)
    selected_keys = []
    
    # 3a. HARD FLOOR: Pre-allocate the floor for every task
    for t_name in sorted(capacities):
        f_val = floor.get(t_name, 5) if isinstance(floor, dict) else floor
        f_cap = min(f_val, capacities[t_name])
        
        pulled = 0
        while pulled < f_cap:
            available_bins = [k for k, v in bins.items() if k[0] == t_name and v]
            if not available_bins:
                break
            if item_selection == "random":
                k = min(available_bins, key=lambda b: (allocated_index[b[1]], b[1]))
            else:
                k = available_bins[0]
            picked_idx, _ = bins[k].pop(0)
            allocated_task[t_name] += 1
            allocated_index[k[1]] += 1
            selected_keys.append((t_name, picked_idx))
            pulled += 1
            
    # 3b. Greedy selection
    for _ in range(budget - len(selected_keys)):
        best_score = -float('inf')
        best_bin = None
        
        for (t_name, i_name), items_list in bins.items():
            if not items_list:
                continue
            if allocated_task[t_name] >= capacities[t_name]:
                continue
                
            target_q = target_allocation.get(t_name, 0)
            if target_q == 0:
                continue
                
            if item_selection == "discrimination":
                _, top_disc = items_list[0]
                # Marginal utility: Item discrimination weighted by relative starvation
                score = max(top_disc, 0.001) * (target_q / (allocated_task[t_name] + 1.0)) * (1.0 / (allocated_index[i_name] + 1.0))
            else:
                # Joint marginal utility: Relative benchmark starvation * index starvation
                score = (target_q / (allocated_task[t_name] + 1.0)) * (1.0 / (allocated_index[i_name] + 1.0))
            
            if score > best_score:
                best_score = score
                best_bin = (t_name, i_name)
                
        if best_bin is None:
            break
            
        t_name, i_name = best_bin
        picked_idx, _ = bins[best_bin].pop(0)
        allocated_task[t_name] += 1
        allocated_index[i_name] += 1
        selected_keys.append((t_name, picked_idx))

    # Keep explicit zero allocations for tasks with no selected items.
    return {name: allocated_task[name] for name in capacities}, selected_keys


def dispersion23_allocation(
    capacities: dict[str, int], dispersions: dict[str, float], budget: int, floor: int | dict[str, int]
) -> dict[str, int]:
    r"""Allocate subset budget using $\\sigma^{2/3}$.

    Returns:
        Dict mapping task name to its allocated capacity.
    """
    if isinstance(floor, int):
        assert floor >= 0, "floor must be non-negative"
    else:
        assert all(v >= 0 for v in floor.values()), "floor must be non-negative"
    if sum(capacities.values()) <= budget:
        return capacities.copy()

    floors = {name: min(floor if isinstance(floor, int) else floor.get(name, 5), capacity) for name, capacity in capacities.items()}
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
    tasks: dict[str, dict[str, Any]],
    fits: dict[str, TwoPLFit],
    capacities: dict[str, int],
    eligible: dict[str, int],
    dispersions: dict[str, float],
    allocation: dict[str, int],
    selected_keys: list[tuple[str, int]],
    labels: ItemLabels | None = None,
    subcategories: Subcategories | None = None,
    indices: dict[str, dict[str, Any]] | None = None,
    gp_irt: dict[str, Any] | None = None,
) -> FitResult:
    """Build the fitted params and ordered subset records.

    Params carry one row per population item: ``item_id`` (which encodes
    ``task::dataset::sample_id``), the 2PL parameters and the secondary label's
    position in the ``labels`` vocabulary; ``labels.subcategories`` maps each
    (index, secondary label) to its reporting sub-category. The subset file carries only membership/design data;
    everything else about a subset item is looked up in params.
    """
    labels = labels or {}
    subcategories = subcategories or Subcategories(by_label={}, definitions={})
    configuration_digest = digest_json(
        {
            "task_scorers": dict(sorted(scorers.items())),
            "hyperparameters": {"ridge": 0.01, "slope_ridge": 0.01, "iterations": 10},
        }
    )
    params_basis = {
        "schema": PARAMS_SCHEMA,
        "method": METHOD_VERSION,
        "inventory_digest": records.digest,
        "task_scorers": dict(sorted(scorers.items())),
        "hyperparameters": {"ridge": 0.01, "slope_ridge": 0.01, "iterations": 10},
    }
    if gp_irt is not None:
        from complai.gp_irt import METHOD_VERSION as GP_IRT_METHOD_VERSION
        params_basis.update(method=GP_IRT_METHOD_VERSION, gp_irt=gp_irt)
        configuration_digest = digest_json({"base": configuration_digest, "gp_irt": gp_irt})

    secondary_vocab: list[str] = []

    def vocab_index(vocab: list[str], value: str | None) -> int:
        value = value or UNKNOWN_LABEL
        if value not in vocab:
            vocab.append(value)
        return vocab.index(value)

    item_rows: list[list[Any]] = []
    task_records: dict[str, Any] = {}
    abilities: dict[str, dict[str, float]] = {}
    for task_name in sorted(tasks):
        task = tasks[task_name]
        fit = fits[task_name]
        abilities[task_name] = {
            (model.split("/")[-1] if "/" in model else model): float(value) 
            for model, value in zip(task["models"], fit.abilities)
        }
        task_records[task_name] = {
            "index": task["index"],
            "scorer": scorers[task_name],
            "dataset": task["items"][0]["dataset"],
            "models": len(task["models"]),
            "capacity": capacities[task_name],
            "eligible": eligible[task_name],
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
            label = item_label(labels, item)
            subcategories.of(task["index"], label.secondary or UNKNOWN_LABEL)
            item_rows.append(
                [
                    item["item_id"],
                    float(fit.discriminations[index]),
                    float(fit.intercepts[index]),
                    vocab_index(secondary_vocab, label.secondary),
                ]
            )

    # Index assignment and labels change index/sub-category scores, so they are
    # part of the params identity.
    params_basis.update(
        task_indices={name: tasks[name]["index"] for name in sorted(tasks)},
        labels_digest=digest_json(
            {
                "secondary": secondary_vocab,
                "items": [row[ITEM_COLUMNS.index("secondary")] for row in item_rows],
                "subcategories": subcategories.to_params(),
            }
        ),
    )
    params_id = digest_json(params_basis)[:24]
    selected_ids = [
        tasks[task]["items"][index]["item_id"] for task, index in selected_keys
    ]
    subset_id = digest_json(
        {"params_id": params_id, "budget": budget, "seed": seed, "items": selected_ids}
    )[:24]

    selected_records = []
    for rank, (task_name, index) in enumerate(selected_keys, start=1):
        item = tasks[task_name]["items"][index]
        probability = allocation[task_name] / eligible[task_name]
        selected_records.append(
            {
                "params_id": params_id,
                "subset_id": subset_id,
                "rank": rank,
                "item_id": item["item_id"],
                "content_hash": item["content_hash"],
                "inclusion_probability": probability,
                "design_weight": 1.0 / probability,
            }
        )

    params = {
        "schema_version": PARAMS_SCHEMA,
        "method": params_basis["method"],
        "params_id": params_id,
        "subset_id": subset_id,
        "budget": budget,
        "seed": seed,
        "task_scorers": dict(sorted(scorers.items())),
        "hyperparameters": params_basis["hyperparameters"],
        "configuration_digest": configuration_digest,
        "input_digest": records.digest,
        "ability_scale": "Each task is independently normalized to mean 0 and standard deviation 1.",
        "tasks": task_records,
        "model_abilities": abilities,
        "labels": {"secondary": secondary_vocab, "subcategories": subcategories.to_params()},
        "indices": indices or {},
        "items": {"columns": list(ITEM_COLUMNS), "data": item_rows},
    }
    if gp_irt is not None:
        params.update(estimator="gp_irt", gp_irt=gp_irt)

    return FitResult(
        params=json_safe(params),
        subset=tuple(json_safe(row) for row in selected_records),
    )


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
    discriminations[:] = np.clip(old * scale, MIN_DISCRIMINATION, MAX_DISCRIMINATION)
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


