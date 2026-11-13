
import collections
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from complai.utils.file_ops import atomic_write
from complai.utils.file_ops import digest_json
from complai.utils.file_ops import json_safe
from complai.constants import (
    ITEM_COLUMNS,
    METHOD_VERSION,
    MIN_INDEX_THETA_TASK_SHARE,
    MIN_MEASURED_POOL_SHARE,
    MIN_PREDICTION_CORRECTION_MODELS,
    MIN_SUBSET_ITEMS,
    PARAMS_SCHEMA,
    PREDICTION_CORRECTION_TASKS,
)
from complai.irt import index_thetas, observed_task_scores, prepare_tasks, sigmoid
from complai.gp_irt import METHOD_VERSION as GP_IRT_METHOD_VERSION
from complai.gp_irt import Estimator, predict_task, resolve_estimator, task_blend
from complai.labels import UNKNOWN_LABEL, Subcategories, split_item_id
from complai.utils.log_parser import load_records


PREDICTION_SCHEMA = "complai-core-prediction-v4"
# Label columns carried on every decoded params item; each gets a score breakdown per model.
LABEL_COLUMNS = {"index": "indices", "subcategory": "subcategories"}
ImputationMethod = Literal["index_pooled", "task_mean"]
IMPUTATION_METHODS: tuple[str, ...] = ("task_mean", "index_pooled")
DEFAULT_IMPUTATION: ImputationMethod = "index_pooled"
# Bootstrap for the index theta and label score intervals: draws, interval
# coverage, and a fixed seed so a prediction is reproducible.
INDEX_THETA_DRAWS = 200
INDEX_THETA_INTERVAL = (5.0, 95.0)
INDEX_THETA_SEED = 0
# The bootstrap leaves out calibration and IRT model error, so its index theta
# intervals are too narrow for some indices. Each interval is widened around the
# point estimate by this factor: the smallest that makes 90% of the intervals
# contain the full-run index theta in leave-one-model-out validation (core.v1,
# 2026-11-12, with prediction corrections: raw coverage 91% Capability, 89%
# Reliability, 79% Safety). Never below 1, so an interval is never narrowed.
INDEX_THETA_INTERVAL_SCALE = {"reliability": 1.05, "safety": 1.55}
# Task score curves are tabulated on this ability grid for the bootstrap.
SCORE_CURVE_GRID = np.linspace(-10.0, 10.0, 801)


@dataclass
class Observation:
    """One model's matched subset responses for one task, in subset order."""

    item_ids: list[str]
    responses: np.ndarray
    discrimination: np.ndarray
    intercept: np.ndarray


@dataclass
class Prediction:
    """A prediction result plus the intermediate arrays consumers need without recomputing."""

    result: dict[str, Any]
    params: dict[str, Any]
    tasks: dict[str, Any]
    # task -> params item rows in params order (the full population being predicted)
    populations: dict[str, list[dict[str, Any]]]
    # subset rows in rank order (what the model actually needs to run)
    subset: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[tuple[str, str], Observation] = field(default_factory=dict)
    # model -> task -> probability per population item (population order)
    item_probabilities: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # (model, task) -> per-population-item theta for imputed tasks
    imputed_thetas: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    # memoized lookups (see item_labels / pooled_label_abilities)
    cache: dict[str, Any] = field(default_factory=dict)


def predict_scores(
    records_path: Path,
    params_path: Path,
    subset_path: Path,
    *,
    estimator: Estimator | None = None,
    imputation: ImputationMethod = DEFAULT_IMPUTATION,
) -> dict[str, Any]:
    """Predict full-population scores from preprocessed subset responses.

    Runs the core per-task prediction, imputes abilities for tasks the model did
    not run, then adds per-index / per-sub-category scores and coverage. This is
    the complete shareable result; exporters only package it.
    """
    return predict_detailed(
        records_path, params_path, subset_path,
        estimator=estimator, imputation=imputation,
    ).result


def predict_detailed(
    records_path: Path | Sequence[Path],
    params_path: Path,
    subset_path: Path,
    *,
    estimator: Estimator | None = None,
    imputation: ImputationMethod = DEFAULT_IMPUTATION,
) -> Prediction:
    """Like predict_scores, but also return item probabilities and observations."""
    params, subset = read_inputs(params_path, subset_path)
    records = load_records(records_path)
    prediction = _predict_task_scores(
        records, params, subset, estimator=estimator,
    )
    hyperparameters = params.get("hyperparameters", {})
    ridge = float(hyperparameters.get("ridge", 0.01))
    iterations = int(hyperparameters.get("iterations", 10))
    impute_missing_abilities(prediction, imputation, ridge=ridge, iterations=iterations)
    prediction.result["imputation"] = imputation
    if imputation != DEFAULT_IMPUTATION:
        prediction.result["prediction_id"] = digest_json(
            {"prediction_id": prediction.result["prediction_id"], "imputation": imputation}
        )[:24]
    prediction.item_probabilities = item_probabilities(prediction)
    add_label_scores(prediction, ridge=ridge, iterations=iterations)
    add_index_thetas(prediction)
    add_coverage(prediction)
    prediction.result = json_safe(prediction.result)
    return prediction


_PREPARED_TASKS: dict[tuple[str, str, str], dict[str, Any]] = {}


def prepared_tasks(
    records: Any, task_scorers: dict[str, str]
) -> dict[str, Any]:
    """prepare_tasks with an in-process memo, so the response matrix is built once per records file."""
    missing = {
        task: scorer for task, scorer in task_scorers.items()
        if (records.digest, task, scorer) not in _PREPARED_TASKS
    }
    if missing:
        built = prepare_tasks(records, missing, _min_models=1, _allow_missing=True)
        for task, scorer in missing.items():
            _PREPARED_TASKS[(records.digest, task, scorer)] = built.get(task)
    return {
        task: _PREPARED_TASKS[(records.digest, task, scorer)]
        for task, scorer in task_scorers.items()
        if _PREPARED_TASKS[(records.digest, task, scorer)] is not None
    }


def _predict_task_scores(
    records: Any,
    params: dict[str, Any],
    subset: list[dict[str, Any]],
    *,
    estimator: Estimator | None,
    tasks: dict[str, dict[str, Any]] | None = None,
    apply_corrections: bool = True,
) -> Prediction:
    """Core per-task ability re-estimation and score prediction.

    ``tasks`` replaces the response matrices prepared from ``records`` (used to
    predict exactly the models a fit saw). Tasks with a stored
    ``prediction_corrections`` entry get their score corrected unless
    ``apply_corrections`` is false.
    """
    estimator = resolve_estimator(params, estimator)
    selected_tasks = {str(row["task"]) for row in subset}
    task_scorers = {
        task: scorer
        for task, scorer in params["task_scorers"].items()
        if task in selected_tasks
    }
    if set(task_scorers) != selected_tasks:
        raise ValueError("params is missing a scorer for a selected task")
    blends = (
        {name: task_blend(params, name) for name in selected_tasks}
        if estimator == "gp_irt" else {}
    )

    hyperparameters = params.get("hyperparameters", {})
    ridge = float(hyperparameters.get("ridge", 0.01))
    iterations = int(hyperparameters.get("iterations", 10))
    params_items = {str(row["item_id"]): row for row in params["items"]}
    if tasks is None:
        tasks = prepared_tasks(records, task_scorers)
    else:
        tasks = {name: task for name, task in tasks.items() if name in task_scorers}
    corrections = params.get("prediction_corrections", {}) if apply_corrections else {}
    if not tasks:
        raise ValueError("No responses found for any subset task")
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in subset:
        selected_by_task.setdefault(str(row["task"]), []).append(row)
    populations: dict[str, list[dict[str, Any]]] = {}
    for row in params["items"]:
        populations.setdefault(str(row["task"]), []).append(row)
    observations: dict[tuple[str, str], Observation] = {}
    changed_items: set[str] = set()

    models = sorted({model for task in tasks.values() for model in task["models"]})
    model_results: dict[str, Any] = {}
    for model in models:
        task_results: dict[str, Any] = {}
        weighted_scores: list[tuple[float, float | None, int]] = []
        for task_name, selected in sorted(selected_by_task.items()):
            task = tasks.get(task_name)
            if task is None or model not in task["models"]:
                task_results[task_name] = missing_task_result(len(selected))
                continue
            model_index = task["models"].index(model)
            item_index = {}
            for item_position, row in enumerate(task["items"]):
                item_index[row["item_id"]] = item_position
                item_index[row.get("question_hash", row["item_id"])] = item_position
            responses: list[float] = []
            selected_parameters: list[dict[str, Any]] = []
            selected_weights: list[float] = []
            for selected_item in selected:
                item_id = str(selected_item["item_id"])
                identity = selected_item.get("question_hash", item_id)
                selected_index = item_index.get(identity)
                if selected_index is None or not np.isfinite(
                    task["matrix"][model_index, selected_index]
                ):
                    continue
                observed_item = task["items"][selected_index]
                if (
                    observed_item["content_hash"] != selected_item["content_hash"]
                    and item_id not in changed_items
                ):
                    changed_items.add(item_id)
                    print(
                        f"Warning: core item {item_id} changed since the core set was "
                        "built; its answer is still used",
                        file=sys.stderr,
                    )
                responses.append(float(task["matrix"][model_index, selected_index]))
                selected_parameters.append(params_items[item_id])
                if estimator == "gp_irt":
                    selected_weights.append(float(selected_item.get("design_weight", 1.0)))
            if not responses:
                task_results[task_name] = missing_task_result(len(selected))
                continue

            population = populations.get(task_name, [])
            discrimination = np.asarray(
                [float(row["discrimination"]) for row in selected_parameters]
            )
            intercept = np.asarray(
                [float(row["intercept"]) for row in selected_parameters]
            )
            observations[(model, task_name)] = Observation(
                item_ids=[str(row["item_id"]) for row in selected_parameters],
                responses=np.asarray(responses, dtype=float),
                discrimination=discrimination,
                intercept=intercept,
            )

            population_a = np.asarray([float(row["discrimination"]) for row in population])
            population_c = np.asarray([float(row["intercept"]) for row in population])
            if estimator == "gp_irt":
                prediction = predict_task(
                    np.asarray(responses), discrimination, intercept, population_a, population_c,
                    blend=blends[task_name], weights=np.asarray(selected_weights), ridge=ridge,
                )
            else:
                ability, standard_error, ability_iterations = estimate_ability(
                    np.asarray(responses), discrimination, intercept, ridge=ridge, iterations=iterations,
                )
                answered_responses = observed_responses(population, observations[(model, task_name)])
                answered = np.isfinite(answered_responses)
                answered_total = float(np.sum(answered_responses[answered]))
                score, upper, lower = (
                    (answered_total + float(np.sum(sigmoid(population_a[~answered] * theta + population_c[~answered]))))
                    / len(population)
                    for theta in (ability, ability + standard_error, ability - standard_error)
                )
                prediction = {
                    "predicted_score": score,
                    "predicted_score_error": (upper - lower) / 2.0,
                    "observed_subset_score": float(np.mean(responses)),
                    "ability": ability,
                    "ability_standard_error": standard_error,
                    "ability_iterations": ability_iterations,
                }
            if task_name in corrections:
                correct_prediction(prediction, corrections[task_name])
            task_results[task_name] = {
                **prediction,
                "status": "ok" if len(responses) == len(selected) else "partial",
                "observations": len(responses),
                "subset_items": len(selected),
                "coverage": len(responses) / len(selected),
                "population_items": len(population),
            }
            weighted_scores.append((prediction["predicted_score"], prediction["predicted_score_error"], len(population)))

        if not weighted_scores:
            raise ValueError(
                f"Model {model!r} has no responses matching the selected subset"
            )
        total_population = sum(count for _, _, count in weighted_scores)
        model_results[model] = {
            "predicted_score": sum(score * count for score, _, count in weighted_scores) / total_population,
            "predicted_score_error": (
                sum(error * count for _, error, count in weighted_scores if error is not None) / total_population
                if all(error is not None for _, error, _ in weighted_scores) else None
            ),
            "task_macro_score": float(np.mean([score for score, _, _ in weighted_scores])),
            "predicted_tasks": len(weighted_scores),
            "params_tasks": len(selected_by_task),
            "tasks": task_results,
        }

    result = {
        "schema_version": PREDICTION_SCHEMA,
        "prediction_id": digest_json(
            {
                "params_id": params["params_id"],
                "subset_id": params["subset_id"],
                "input_digest": records.digest,
                **({"estimator": estimator} if params["method"] == GP_IRT_METHOD_VERSION else {}),
            }
        )[:24],
        "params_id": params["params_id"],
        "subset_id": params["subset_id"],
        "method": params["method"],
        "ability_scale": params.get("ability_scale"),
        "score_interpretation": (
            "Per-task scores are the mean over all params items of the observed "
            "response on subset items the model answered and the fixed-item 2PL "
            "probability on the rest; predicted_score_error covers only the rest. "
            "Tasks in params prediction_corrections are then corrected linearly "
            "toward the panel's full-run scores (uncorrected_score keeps the raw value). "
            "Model predicted_score is population-item-weighted."
        ),
        "input_digest": records.digest,
        "inventory": records.inventory["summary"],
        "models": model_results,
    }
    if params["method"] == GP_IRT_METHOD_VERSION:
        result["estimator"] = estimator
    if estimator == "gp_irt":
        result["score_interpretation"] = (
            "Per-task scores blend the weighted observed subset mean with the "
            "full-population 2PL mean using frozen source-calibrated weights. "
            "Model predicted_score is population-item-weighted. "
            "predicted_score_error is null because blend uncertainty is not estimated."
        )
    return Prediction(
        result=result, params=params, tasks=tasks, populations=populations,
        subset=subset, observations=observations,
    )


def correct_prediction(prediction: dict[str, Any], correction: dict[str, Any]) -> None:
    """Correct a task prediction to intercept + slope * score, clipped to [0, 1]."""
    slope = float(correction["slope"])
    score = prediction["predicted_score"]
    prediction["uncorrected_score"] = score
    prediction["predicted_score"] = float(np.clip(float(correction["intercept"]) + slope * score, 0.0, 1.0))
    if prediction.get("predicted_score_error") is not None:
        prediction["predicted_score_error"] *= abs(slope)


def correction_slope(params: dict[str, Any], task_name: str) -> float:
    """The slope a task's predicted scores were corrected with (1 without a correction)."""
    return float(params.get("prediction_corrections", {}).get(task_name, {}).get("slope", 1.0))


def fit_prediction_corrections(
    records: Any,
    params: dict[str, Any],
    subset: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fit each ``PREDICTION_CORRECTION_TASKS`` task's line full_run ~ intercept + slope * predicted.

    ``params`` is a freshly fitted, still columnar params and ``subset`` its
    subset; every model in ``tasks`` is predicted from its subset responses
    without corrections, and ``full_run`` is the model's mean response over all
    task questions it answered, the score the index calibration uses. A task
    needs ``MIN_PREDICTION_CORRECTION_MODELS`` models.
    """
    names = [name for name in PREDICTION_CORRECTION_TASKS if name in tasks]
    rows = [
        {**row, "task": task_name}
        for row in subset
        if (task_name := split_item_id(str(row["item_id"]))[0]) in names
    ]
    if not rows:
        return {}
    prediction = _predict_task_scores(
        records,
        decode_params(dict(params)),
        rows,
        estimator=None,
        tasks={name: tasks[name] for name in names},
        apply_corrections=False,
    )
    corrections: dict[str, dict[str, Any]] = {}
    for task_name in names:
        full_run = observed_task_scores(tasks[task_name])
        pairs = [
            (result["predicted_score"], full_run[model])
            for model in tasks[task_name]["models"]
            if model in full_run
            and (result := prediction.result["models"].get(model, {}).get("tasks", {}).get(task_name))
            and result["status"] in {"ok", "partial"}
        ]
        if len(pairs) < MIN_PREDICTION_CORRECTION_MODELS:
            continue
        predicted, observed = np.asarray(pairs).T
        slope, intercept = np.polyfit(predicted, observed, 1)
        corrections[task_name] = {
            "intercept": float(intercept),
            "slope": float(slope),
            "models": len(pairs),
        }
    return corrections


def ran_abilities(prediction: Prediction, model: str, *, exclude_task: str | None = None) -> dict[str, float]:
    """Task -> re-estimated ability for tasks the model actually ran (never imputed ones), except ``exclude_task``."""
    return {
        name: float(t["ability"])
        for name, t in prediction.result["models"][model]["tasks"].items()
        if name != exclude_task and t.get("ability") is not None and not t.get("imputed")
        and t["status"] in {"ok", "partial"}
    }


def item_labels(prediction: Prediction, column: str) -> dict[str, str]:
    """Item id -> label (``column``: index or subcategory) over all populations."""
    key = ("item_labels", column)
    if key not in prediction.cache:
        prediction.cache[key] = {
            str(row["item_id"]): str(row.get(column, UNKNOWN_LABEL))
            for rows in prediction.populations.values() for row in rows
        }
    return prediction.cache[key]


def pooled_label_abilities(
    prediction: Prediction,
    model: str,
    column: str,
    *,
    ridge: float,
    iterations: int,
    exclude_task: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Label -> ability re-fitted on the model's subset responses pooled across its tasks.

    One ridge-MAP ability per label (e.g. per index) from every answered subset
    item carrying that label, using the items' fixed task parameters. This is
    the reported label ability and the ``index_pooled`` imputation source;
    ``exclude_task`` drops that task's responses (leave-one-task-out).
    """
    key = ("pooled", model, column, exclude_task)
    if key in prediction.cache:
        return prediction.cache[key]
    label_of = item_labels(prediction, column)
    pooled: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for (observed_model, observed_task), observation in prediction.observations.items():
        if observed_model != model or observed_task == exclude_task:
            continue
        for item_id, response, a, c in zip(
            observation.item_ids, observation.responses,
            observation.discrimination, observation.intercept,
        ):
            bucket = pooled.setdefault(label_of.get(item_id, UNKNOWN_LABEL), ([], [], []))
            bucket[0].append(float(response))
            bucket[1].append(float(a))
            bucket[2].append(float(c))
    abilities: dict[str, dict[str, Any]] = {}
    for label, (responses, a, c) in pooled.items():
        ability, standard_error, _ = estimate_ability(
            np.asarray(responses), np.asarray(a), np.asarray(c), ridge=ridge, iterations=iterations,
        )
        abilities[label] = {
            "ability": ability,
            "ability_standard_error": standard_error,
            "observations": len(responses),
            "observed_subset_score": float(np.mean(responses)),
        }
    prediction.cache[key] = abilities
    return abilities


def imputation_thetas(
    prediction: Prediction,
    model: str,
    task: str,
    method: ImputationMethod,
    *,
    ridge: float,
    iterations: int,
) -> np.ndarray | None:
    """Ability for every population item of ``task``, as if ``model`` had not run it.

    Uses only the model's *other* tasks, so the same function serves production
    imputation and leave-one-task-out evaluation.

    - ``task_mean``: unweighted mean of the model's ran task abilities (one theta).
    - ``index_pooled``: the model's pooled ability for the task's index
      (``pooled_label_abilities``); if the model has no responses in that
      index, it falls back to ``task_mean``.

    Returns None when the model has no ran-task abilities.
    """
    if method not in IMPUTATION_METHODS:
        raise ValueError(f"Unknown imputation method {method!r}; expected one of {IMPUTATION_METHODS}")
    population = prediction.populations.get(task, [])
    abilities = ran_abilities(prediction, model, exclude_task=task)
    if not population or not abilities:
        return None
    thetas = np.full(len(population), float(np.mean(list(abilities.values()))))
    if method == "task_mean":
        return thetas
    per_index = pooled_label_abilities(
        prediction, model, "index", ridge=ridge, iterations=iterations, exclude_task=task,
    )
    index = prediction.params["tasks"][task]["index"]
    if index in per_index:
        thetas[:] = per_index[index]["ability"]
    return thetas


def impute_missing_abilities(
    prediction: Prediction,
    method: ImputationMethod = DEFAULT_IMPUTATION,
    *,
    ridge: float = 0.01,
    iterations: int = 10,
) -> None:
    """Fill subset tasks a model never ran using ``imputation_thetas``.

    Imputed tasks get ``status: "imputed"`` and ``imputed: true``; their ``ability``
    is the mean item theta (per-item thetas are kept in
    ``prediction.imputed_thetas`` for item probabilities). They feed
    ``predicted_score`` and label pools, but consumers computing errors against
    ground truth should skip them.
    """
    if method not in IMPUTATION_METHODS:
        raise ValueError(f"Unknown imputation method {method!r}; expected one of {IMPUTATION_METHODS}")
    populations = prediction.populations
    for model, model_result in prediction.result["models"].items():
        model_result["imputed_tasks"] = 0
        for task_result in model_result["tasks"].values():
            task_result.setdefault("imputed", False)
        for task_name, task_result in model_result["tasks"].items():
            if task_result["status"] != "missing":
                continue
            thetas = imputation_thetas(
                prediction, model, task_name, method, ridge=ridge, iterations=iterations,
            )
            if thetas is None:
                continue
            population = populations[task_name]
            a = np.asarray([float(row["discrimination"]) for row in population])
            c = np.asarray([float(row["intercept"]) for row in population])
            prediction.imputed_thetas[(model, task_name)] = thetas
            task_result.update(
                {
                    "status": "imputed",
                    "imputed": True,
                    "ability": float(np.mean(thetas)),
                    "predicted_score": float(np.mean(sigmoid(a * thetas + c))),
                    "population_items": len(population),
                }
            )
            correction = prediction.params.get("prediction_corrections", {}).get(task_name)
            if correction is not None:
                correct_prediction(task_result, correction)
            model_result["imputed_tasks"] += 1
        # predicted_score is population-weighted over ran + imputed tasks.
        scored = [
            (t["predicted_score"], t["population_items"])
            for t in model_result["tasks"].values()
            if t["status"] != "missing"
        ]
        total = sum(n for _, n in scored)
        model_result["predicted_score"] = sum(s * n for s, n in scored) / total
        model_result["task_macro_score"] = float(np.mean([s for s, _ in scored]))


def item_probabilities(prediction: Prediction) -> dict[str, dict[str, np.ndarray]]:
    """Per-model, per-task probability of a correct answer for every population item.

    Items the model answered in its subset take the observed response; the rest
    use sigmoid(a*theta + c) with the (re-estimated or imputed) task ability.
    Under gp_irt every item instead takes the task's blend of the observed
    subset mean and its 2PL probability, so the items average to the task score.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for model, model_result in prediction.result["models"].items():
        per_task: dict[str, np.ndarray] = {}
        for task_name, task_result in model_result["tasks"].items():
            population = prediction.populations.get(task_name, [])
            if not population or task_result["status"] == "missing":
                continue
            ability = prediction.imputed_thetas.get((model, task_name), task_result["ability"])
            observation = prediction.observations.get((model, task_name))
            observed = (
                observed_responses(population, observation)
                if observation is not None else np.full(len(population), np.nan)
            )
            a = np.asarray([float(row["discrimination"]) for row in population])
            c = np.asarray([float(row["intercept"]) for row in population])
            probabilities = sigmoid(a * np.asarray(ability, dtype=float) + c)
            blend = task_result.get("blend_weight")
            if blend is not None and not task_result.get("imputed"):
                probabilities = blend * task_result["observed_subset_score"] + (1 - blend) * probabilities
            else:
                answered = np.isfinite(observed)
                probabilities[answered] = observed[answered]
            per_task[task_name] = probabilities
        out[model] = per_task
    return out


def label_pools(prediction: Prediction, column: str) -> dict[str, list[tuple[str, int]]]:
    """Map each value of a params label column to its (task, population index) members."""
    pools: dict[str, list[tuple[str, int]]] = {}
    for task_name, population in prediction.populations.items():
        for index, row in enumerate(population):
            pools.setdefault(str(row.get(column, UNKNOWN_LABEL)), []).append((task_name, index))
    return pools


def add_label_scores(prediction: Prediction, *, ridge: float, iterations: int) -> None:
    """Attach per-index (``indices``) and per-sub-category (``subcategories``) results to every model."""
    items_by_id = {str(row["item_id"]): row for rows in prediction.populations.values() for row in rows}
    for column, key in LABEL_COLUMNS.items():
        pools = label_pools(prediction, column)
        subset_items = collections.Counter(
            str(items_by_id[str(row["item_id"])].get(column, UNKNOWN_LABEL)) for row in prediction.subset
        )
        weights = label_task_weights(prediction, pools) if column == "subcategory" else None
        for model, model_result in prediction.result["models"].items():
            model_result[key] = pool_scores(
                prediction, model, column, pools, subset_items,
                weights=weights, ridge=ridge, iterations=iterations,
            )


def label_task_weights(
    prediction: Prediction, pools: dict[str, list[tuple[str, int]]]
) -> dict[str, dict[str, float]]:
    """Weight of each member task in each label pool: its index discrimination times its population share in the pool.

    A task's index weight (its fitted benchmark discrimination in
    ``params["indices"]``) is split over the labels by the share of its own
    population items carrying each label, so a pool counts benchmarks as the
    index does rather than by how many items each contributes.
    """
    discrimination = {
        task_name: float(calibration["discrimination"])
        for index in prediction.params.get("indices", {}).values()
        for task_name, calibration in index["tasks"].items()
    }
    weights: dict[str, dict[str, float]] = {}
    for label, members in pools.items():
        counts = collections.Counter(task_name for task_name, _ in members)
        weights[label] = {
            task_name: discrimination.get(task_name, 1.0) * count / len(prediction.populations[task_name])
            for task_name, count in counts.items()
        }
    return weights


def add_index_thetas(prediction: Prediction) -> None:
    """Attach each model's index theta (``indices[i]["index_theta"]``).

    The index theta is fitted on the model's predicted task scores with the
    index calibration frozen (``params["indices"]``), so panel and new models
    are scored identically. Only tasks the model ran count, and the model needs
    at least ``MIN_INDEX_THETA_TASK_SHARE`` of the index's tasks (rounded up);
    otherwise the value is null.
    The interval comes from a parametric bootstrap: each task ability is
    redrawn from its estimate and standard error, the predicted score moves by
    the change in the score curve of the items the model did not answer
    (answered items are fixed at their observed response), plus the pass/fail
    spread of those unanswered items (times the task's prediction correction
    slope, if it has one), and the index theta is refitted. The
    bootstrap range is then widened around ``theta`` by the index's
    ``INDEX_THETA_INTERVAL_SCALE`` (stored as ``interval_scale``).
    """
    for index, calibration in prediction.params.get("indices", {}).items():
        names = sorted(calibration["tasks"])
        a = np.asarray([calibration["tasks"][name]["discrimination"] for name in names])
        c = np.asarray([calibration["tasks"][name]["intercept"] for name in names])
        for model, model_result in prediction.result["models"].items():
            entry = model_result.get("indices", {}).get(index)
            if entry is None:
                continue
            ran = [
                position for position, name in enumerate(names)
                if (model, name) not in prediction.imputed_thetas
                and model_result["tasks"].get(name, {}).get("status") in {"ok", "partial"}
            ]
            if len(ran) < math.ceil(MIN_INDEX_THETA_TASK_SHARE * len(names)):
                entry["index_theta"] = None
                continue
            results = [model_result["tasks"][names[position]] for position in ran]
            scores = np.asarray([[result["predicted_score"] for result in results]])
            theta = float(index_thetas(scores, a[ran], c[ran])[0])
            rng = np.random.default_rng(INDEX_THETA_SEED)
            draws = np.empty((INDEX_THETA_DRAWS, len(ran)))
            for column, (position, result) in enumerate(zip(ran, results)):
                abilities = rng.normal(
                    result["ability"], result["ability_standard_error"], INDEX_THETA_DRAWS
                )
                curve = score_curve(prediction, model, names[position])
                shift = np.interp(abilities, SCORE_CURVE_GRID, curve) - np.interp(
                    result["ability"], SCORE_CURVE_GRID, curve
                )
                shift += rng.normal(
                    0.0, unanswered_spread(prediction, model, names[position], result["ability"]),
                    INDEX_THETA_DRAWS,
                )
                shift *= correction_slope(prediction.params, names[position])
                draws[:, column] = np.clip(result["predicted_score"] + shift, 0.0, 1.0)
            low, high = np.percentile(index_thetas(draws, a[ran], c[ran]), INDEX_THETA_INTERVAL)
            scale = INDEX_THETA_INTERVAL_SCALE.get(index, 1.0)
            low, high = theta - scale * (theta - low), theta + scale * (high - theta)
            entry["index_theta"] = {
                "theta": theta,
                "interval": [float(low), float(high)],
                "interval_scale": scale,
                "tasks": len(ran),
                "tasks_total": len(names),
            }


def score_curve(prediction: Prediction, model: str, task_name: str) -> np.ndarray:
    """The part of a task's score that depends on ability, at every ``SCORE_CURVE_GRID`` ability.

    It is the summed 2PL probability of the population items the model did not
    answer, divided by the population size. Memoized per set of answered items.
    """
    population = prediction.populations[task_name]
    answered = np.isfinite(observed_responses(population, prediction.observations[(model, task_name)]))
    key = (task_name, answered.tobytes())
    curves = prediction.cache.setdefault("score_curves", {})
    if key not in curves:
        a = np.asarray([float(row["discrimination"]) for row in population])[~answered]
        c = np.asarray([float(row["intercept"]) for row in population])[~answered]
        curves[key] = np.asarray(
            [np.sum(sigmoid(a * theta + c)) / len(population) for theta in SCORE_CURVE_GRID]
        )
    return curves[key]


def unanswered_spread(prediction: Prediction, model: str, task_name: str, ability: float) -> float:
    """Standard deviation of a task's score from the population items the model did not answer.

    Each unanswered item is a pass/fail outcome with its 2PL probability at
    ``ability``; the score is their sum divided by the population size.
    """
    population = prediction.populations[task_name]
    answered = np.isfinite(observed_responses(population, prediction.observations[(model, task_name)]))
    a = np.asarray([float(row["discrimination"]) for row in population])[~answered]
    c = np.asarray([float(row["intercept"]) for row in population])[~answered]
    probabilities = sigmoid(a * ability + c)
    return float(np.sqrt(np.sum(probabilities * (1.0 - probabilities))) / len(population))


def observed_responses(population: list[dict[str, Any]], observation: Observation) -> np.ndarray:
    """The observed response for each population item the model answered in its subset, NaN elsewhere."""
    observed = dict(zip(observation.item_ids, observation.responses))
    return np.asarray(
        [observed.get(str(row["item_id"]), np.nan) for row in population], dtype=float
    )


def add_coverage(prediction: Prediction) -> None:
    """Attach how much of the subset, and of the full calibrated population, each model answered."""
    for model, model_result in prediction.result["models"].items():
        ran = [t for t in model_result["tasks"].values() if t["status"] in {"ok", "partial"}]
        model_result["coverage"] = {
            "tasks_completed": len(ran),
            "tasks_total": len(model_result["tasks"]),
            "samples_completed": sum(int(t.get("observations", 0)) for t in ran),
            "samples_total": len(prediction.subset),
            "population_samples_observed": sum(
                int(np.isfinite(task["matrix"][task["models"].index(model)]).sum())
                for task in prediction.tasks.values()
                if model in task["models"]
            ),
        }


NOT_RUN = "not run"


def gap_reason(subset_items: int) -> str | None:
    """Why a label pool is not scored, or None when its subset items reach ``MIN_SUBSET_ITEMS``."""
    if subset_items >= MIN_SUBSET_ITEMS:
        return None
    if subset_items == 0:
        return "no subset items"
    return f"{subset_items} subset items, needs {MIN_SUBSET_ITEMS}"


def pool_scores(
    prediction: Prediction,
    model: str,
    column: str,
    pools: dict[str, list[tuple[str, int]]],
    subset_items: collections.Counter[str],
    *,
    weights: dict[str, dict[str, float]] | None = None,
    ridge: float,
    iterations: int,
) -> dict[str, Any]:
    """Score one model on each label pool of population items.

    A pool with fewer than ``MIN_SUBSET_ITEMS`` subset items is a gap: the
    subset cannot measure it, so ``gap`` holds the reason (``gap_reason``) and
    the pool keeps its coverage counts but no scores. A pool the model answered
    no subset items of is a gap too (``NOT_RUN``): its score would be entirely
    imputed. So is a pool whose tasks the model ran carry less than
    ``MIN_MEASURED_POOL_SHARE`` of its weight. ``gap`` is None otherwise.

    Only tasks the model ran are scored: imputed tasks are left out and the
    weights renormalized over the rest, so no published pool score is imputed.
    ``measured_weight`` is the share of the pool's full weight those tasks carry.
    ``predicted_score`` averages ``item_probabilities`` over their population
    items: observed responses on answered subset items, otherwise the
    probability under the task's ability. Without ``weights`` it is the plain
    mean over items; with ``weights`` (``label_task_weights``) it is the
    weighted mean of each member task's mean, and ``task_weights`` holds the
    normalized weights. ``ability`` (with
    ``ability_standard_error``) is the label ability from
    ``pooled_label_abilities``: a summary of the model's standing on that label,
    and the ``index_pooled`` imputation source. It is null when the model
    answered no subset items with the label.

    ``score_interval`` is the ``INDEX_THETA_INTERVAL`` range of a parametric
    bootstrap of ``predicted_score`` (``pool_score_draws``).
    """
    probabilities = prediction.item_probabilities.get(model, {})
    pooled = pooled_label_abilities(prediction, model, column, ridge=ridge, iterations=iterations)
    scores: dict[str, Any] = {}
    for label, members in sorted(pools.items()):
        member_indices: dict[str, list[int]] = {}
        for task_name, index in members:
            member_indices.setdefault(task_name, []).append(index)
        if all(probabilities.get(task_name) is None for task_name in member_indices):
            continue
        pool_weights = weights.get(label, {}) if weights is not None else None
        full = {
            task_name: (pool_weights[task_name] if pool_weights is not None else float(len(indices)))
            for task_name, indices in member_indices.items()
        }
        task_means: dict[str, float] = {}
        task_counts: dict[str, int] = {}
        for task_name, indices in member_indices.items():
            task_probabilities = probabilities.get(task_name)
            if task_probabilities is None or (model, task_name) in prediction.imputed_thetas:
                continue
            values = task_probabilities[sorted(indices)]
            finite = values[np.isfinite(values)]
            if len(finite):
                task_means[task_name] = float(np.mean(finite))
                task_counts[task_name] = len(finite)
        full_total = sum(full.values())
        measured_weight = sum(full[task_name] for task_name in task_means) / full_total if full_total else 0.0
        raw = {
            task_name: (pool_weights[task_name] if pool_weights is not None else float(count))
            for task_name, count in task_counts.items()
        }
        total = sum(raw.values())
        task_weights = {task_name: value / total for task_name, value in raw.items()} if total else {}
        point = sum(task_weights[task_name] * task_means[task_name] for task_name in task_means)
        observed = pooled.get(label, {})
        measured = observed.get("observations", 0) and measured_weight >= MIN_MEASURED_POOL_SHARE
        gap = gap_reason(subset_items[label]) or (None if measured else NOT_RUN)
        interval = None
        if not gap:
            draws = pool_score_draws(
                prediction, model, {task_name: member_indices[task_name] for task_name in task_means},
                weights=pool_weights,
            )
            low, high = np.percentile(draws, INDEX_THETA_INTERVAL)
            interval = [float(low), float(high)]
        scores[label] = {
            "gap": gap,
            "predicted_score": None if gap else float(point),
            "score_interval": interval,
            "population_items": len(members),
            "population_tasks": len(member_indices),
            "subset_items": subset_items[label],
            "measured_weight": float(measured_weight),
            **({"task_weights": task_weights} if weights is not None else {}),
            "observations": observed.get("observations", 0),
            "ability": None if gap else observed.get("ability"),
            "ability_standard_error": None if gap else observed.get("ability_standard_error"),
            "observed_subset_score": None if gap else observed.get("observed_subset_score"),
        }
    return scores


def pool_score_draws(
    prediction: Prediction,
    model: str,
    member_indices: dict[str, list[int]],
    *,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Bootstrap draws of a label pool's ``predicted_score``.

    Without ``weights`` the score is the mean item probability; with
    ``weights`` it is the weighted mean of each member task's mean item
    probability (weights renormalized over the tasks that have items).

    Mirrors the index theta bootstrap: each member task's ability is redrawn
    from its estimate and standard error, answered items stay at their observed
    response, and the pass/fail spread of the unanswered items is added. Only
    tasks the model ran are passed in (``pool_scores``), never imputed ones.
    """
    model_result = prediction.result["models"][model]
    probabilities = prediction.item_probabilities.get(model, {})
    rng = np.random.default_rng(INDEX_THETA_SEED)
    totals = np.zeros(INDEX_THETA_DRAWS)
    variance = 0.0
    weight_total = 0.0
    for task_name, indices in member_indices.items():
        point = probabilities.get(task_name)
        if point is None:
            continue
        indices = np.asarray(sorted(indices))
        point = point[indices]
        keep = np.isfinite(point)
        indices, point = indices[keep], point[keep]
        if not len(indices):
            continue
        task_result = model_result["tasks"][task_name]
        population = prediction.populations[task_name]
        a, c = population_item_params(prediction, task_name)
        a, c = a[indices], c[indices]
        theta = float(task_result["ability"])
        standard_error = task_result.get("ability_standard_error")
        observation = prediction.observations.get((model, task_name))
        answered = (
            np.isfinite(observed_responses(population, observation)[indices])
            if observation is not None
            else np.zeros(len(indices), dtype=bool)
        )
        shifts = rng.normal(0.0, float(standard_error or 0.0), INDEX_THETA_DRAWS)
        drawn = sigmoid(a * (theta + shifts[:, None]) + c)
        blend = task_result.get("blend_weight")
        if blend is not None:
            drawn = blend * task_result["observed_subset_score"] + (1 - blend) * drawn
            answered = np.zeros(len(indices), dtype=bool)
        drawn[:, answered] = point[answered]
        weight = float(len(indices)) if weights is None else weights[task_name]
        totals += weight * drawn.mean(axis=1)
        variance += weight**2 * float(np.sum(point[~answered] * (1.0 - point[~answered]))) / len(indices) ** 2
        weight_total += weight
    totals += rng.normal(0.0, math.sqrt(variance), INDEX_THETA_DRAWS)
    return np.clip(totals / weight_total, 0.0, 1.0)


def population_item_params(
    prediction: Prediction, task_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Discrimination and intercept of every population item of a task (memoized)."""
    cache = prediction.cache.setdefault("population_item_params", {})
    if task_name not in cache:
        population = prediction.populations[task_name]
        cache[task_name] = (
            np.asarray([float(row["discrimination"]) for row in population]),
            np.asarray([float(row["intercept"]) for row in population]),
        )
    return cache[task_name]


def write_prediction(result: dict[str, Any], output_path: Path) -> Path:
    """Write predicted scores atomically as JSON."""
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(output_path, content)
    return output_path


def read_inputs(
    params_path: Path, subset_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and validate a fitted params and matching subset."""
    try:
        params = json.loads(params_path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read params JSON: {params_path}") from exc
    params = decode_params(params)

    subset: list[dict[str, Any]] = []
    try:
        with subset_path.expanduser().open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(
                        f"{subset_path}:{line_number}: expected a JSON object"
                    )
                subset.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read subset JSONL: {subset_path}") from exc
    if not subset:
        raise ValueError("Subset JSONL is empty")

    params_items = {str(row["item_id"]): row for row in params["items"]}
    item_ids: set[str] = set()
    for row in subset:
        item_id = str(row.get("item_id", ""))
        if not isinstance(row.get("content_hash"), str):
            raise TypeError(f"Selected item {item_id!r} is missing content_hash")
        if (
            row.get("params_id") != params["params_id"]
            or row.get("subset_id") != params["subset_id"]
        ):
            raise ValueError(
                f"Subset identity does not match params for item {item_id!r}"
            )
        if item_id in item_ids:
            raise ValueError(f"Duplicate selected item {item_id!r}")
        if item_id not in params_items:
            raise ValueError(f"Unknown item {item_id!r}")
        item_ids.add(item_id)
        row["task"] = params_items[item_id]["task"]
    subset.sort(key=lambda row: int(row.get("rank", 0)))
    recomputed = digest_json({
        "params_id": params["params_id"],
        "budget": params.get("budget"),
        "seed": params.get("seed"),
        "items": [str(row["item_id"]) for row in subset],
    })[:24]
    if recomputed != params["subset_id"]:
        raise ValueError("Subset JSONL is not the exact subset recorded by the params")
    return params, subset


def decode_params(params: Any) -> dict[str, Any]:
    """Validate a fitted params dict and expand its columnar items in place."""
    if not isinstance(params, dict) or not isinstance(params.get("task_scorers"), dict):
        raise TypeError("Params is missing its task_scorers mapping")
    if (
        params.get("schema_version") != PARAMS_SCHEMA
        or params.get("method") not in {METHOD_VERSION, GP_IRT_METHOD_VERSION}
    ):
        raise ValueError("Params is not a supported GP-IRT 2PL params")
    if not params["task_scorers"] or any(
        not isinstance(task, str) or not isinstance(scorer, str)
        for task, scorer in params["task_scorers"].items()
    ):
        raise ValueError("Params has an invalid task_scorers mapping")
    items_data = params.get("items")
    if not isinstance(items_data, dict) or "columns" not in items_data or "data" not in items_data:
        raise ValueError("Params items must be in columnar format (dict with 'columns' and 'data')")
    columns = list(items_data["columns"])
    if columns != list(ITEM_COLUMNS):
        raise ValueError(f"Params item columns must be {list(ITEM_COLUMNS)}, got {columns}")
    vocab = params.get("labels")
    if not isinstance(vocab, dict) or not isinstance(vocab.get("secondary"), list):
        raise ValueError("Params is missing its secondary label vocabulary")
    if not isinstance(vocab.get("subcategories"), dict):
        raise ValueError("Params is missing its sub-categories; refit with the current irt")
    task_indices = {name: str(task["index"]) for name, task in params.get("tasks", {}).items()}
    subcategories = Subcategories.from_params(vocab["subcategories"])
    params["items"] = [
        decode_item(dict(zip(columns, row)), vocab, task_indices, subcategories)
        for row in items_data["data"]
    ]
    required = {"params_id", "subset_id", "method"}
    if not required.issubset(params):
        raise ValueError("Params is missing identity or method fields")
    return params


def decode_item(
    row: dict[str, Any],
    vocab: dict[str, list[str]],
    task_indices: dict[str, str],
    subcategories: Subcategories,
) -> dict[str, Any]:
    """Expand a columnar params item: resolve its secondary label, derive task/dataset/sample_id,
    its task's index, and the sub-category its label reports under within that index."""
    task, dataset, sample_id = split_item_id(str(row["item_id"]))
    row["task"], row["dataset"], row["sample_id"] = task, dataset, sample_id
    position = row.get("secondary")
    row["secondary"] = vocab["secondary"][int(position)] if position is not None else UNKNOWN_LABEL
    row["index"] = task_indices[task]
    row["subcategory"] = subcategories.of(row["index"], row["secondary"])
    return row


def estimate_ability(
    responses: np.ndarray,
    discrimination: np.ndarray,
    intercept: np.ndarray,
    *,
    ridge: float = 0.01,
    iterations: int = 10,
) -> tuple[float, float, int]:
    """Estimate one model ability from fixed item parameters."""
    ability = 0.0
    used_iterations = 0
    for iteration in range(iterations):
        used_iterations = iteration + 1
        predicted = sigmoid(discrimination * ability + intercept)
        information = float(
            np.sum(discrimination**2 * predicted * (1.0 - predicted)) + ridge
        )
        gradient = float(
            np.sum(discrimination * (responses - predicted)) - ridge * ability
        )
        step = float(np.clip(gradient / information, -1.5, 1.5))
        ability += step
        if abs(step) < 1e-8:
            break
    predicted = sigmoid(discrimination * ability + intercept)
    information = float(
        np.sum(discrimination**2 * predicted * (1.0 - predicted)) + ridge
    )
    return ability, 1.0 / math.sqrt(information), used_iterations


def missing_task_result(subset_items: int) -> dict[str, Any]:
    """Return the result recorded when a task has no responses."""
    return {
        "status": "missing",
        "predicted_score": None, "predicted_score_error": None,
        "observations": 0,
        "subset_items": subset_items,
        "coverage": 0.0,
    }
