import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from complai.core.fit import PARAMS_SCHEMA
from complai.core.fit import atomic_write
from complai.core.fit import DEFAULT_CACHE_DIR
from complai.core.fit import digest_json
from complai.core.fit import json_safe
from complai.core.fit import METHOD_VERSION
from complai.core.fit import prepare_tasks
from complai.core.fit import sigmoid
from complai.core.index import index_logs


PREDICTION_SCHEMA = "complai-minify-prediction-v1"


def predict_scores(
    log_paths: list[Path],
    params_path: Path,
    subset_path: Path,
    *,
    duplicate_policy: str = "error",
    cache_dir: Path | None = None,
    reindex: bool = False,
) -> dict[str, Any]:
    """Predict full-task scores for new models evaluated on a minified subset."""
    if duplicate_policy not in {"error", "mean", "latest"}:
        raise ValueError("duplicate_policy must be error, mean, or latest")
    params, subset = read_inputs(params_path, subset_path)
    selected_tasks = {str(row["task"]) for row in subset}
    task_scorers = {
        task: scorer
        for task, scorer in params["task_scorers"].items()
        if task in selected_tasks
    }
    if set(task_scorers) != selected_tasks:
        raise ValueError("Params is missing a scorer for a selected task")
    hyperparameters = params.get("hyperparameters", {})
    ridge = float(hyperparameters.get("ridge", 0.01))
    iterations = int(hyperparameters.get("iterations", 10))
    cache_root = (cache_dir or DEFAULT_CACHE_DIR).expanduser()
    indexed = index_logs(
        log_paths, cache_root / "index-v1.sqlite3", reindex=reindex, tasks=task_scorers
    )
    tasks = prepare_tasks(indexed, task_scorers, duplicate_policy, _min_models=1)
    params_items = {str(row["item_id"]): row for row in params["items"]}
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in subset:
        selected_by_task.setdefault(str(row["task"]), []).append(row)

    models = sorted({model for task in tasks.values() for model in task["models"]})
    model_results: dict[str, Any] = {}
    for model in models:
        task_results: dict[str, Any] = {}
        weighted_scores: list[tuple[float, int]] = []
        for task_name, selected in sorted(selected_by_task.items()):
            task = tasks.get(task_name)
            if task is None or model not in task["models"]:
                task_results[task_name] = missing_task_result(len(selected))
                continue
            model_index = task["models"].index(model)
            item_index = {
                row["item_id"]: index for index, row in enumerate(task["items"])
            }
            responses: list[float] = []
            selected_parameters: list[dict[str, Any]] = []
            for selected_item in selected:
                item_id = str(selected_item["item_id"])
                index = item_index.get(item_id)
                if index is None or not np.isfinite(task["matrix"][model_index, index]):
                    continue
                observed_item = task["items"][index]
                if observed_item["content_hash"] != selected_item["content_hash"]:
                    raise ValueError(f"Content mismatch for selected item {item_id}")
                responses.append(float(task["matrix"][model_index, index]))
                selected_parameters.append(params_items[item_id])
            if not responses:
                task_results[task_name] = missing_task_result(len(selected))
                continue

            discrimination = np.asarray(
                [float(row["discrimination"]) for row in selected_parameters]
            )
            intercept = np.asarray(
                [float(row["intercept"]) for row in selected_parameters]
            )
            ability, standard_error, ability_iterations = estimate_ability(
                np.asarray(responses),
                discrimination,
                intercept,
                ridge=ridge,
                iterations=iterations,
            )
            population = [
                row for row in params["items"] if str(row["task"]) == task_name
            ]
            population_score = float(
                np.mean(
                    sigmoid(
                        np.asarray([float(row["discrimination"]) for row in population])
                        * ability
                        + np.asarray([float(row["intercept"]) for row in population])
                    )
                )
            )
            task_results[task_name] = {
                "status": "ok" if len(responses) == len(selected) else "partial",
                "predicted_score": population_score,
                "observed_subset_score": float(np.mean(responses)),
                "ability": ability,
                "ability_standard_error": standard_error,
                "ability_iterations": ability_iterations,
                "observations": len(responses),
                "subset_items": len(selected),
                "coverage": len(responses) / len(selected),
                "population_items": len(population),
            }
            weighted_scores.append((population_score, len(population)))

        if not weighted_scores:
            raise ValueError(
                f"Model {model!r} has no responses matching the selected subset"
            )
        model_results[model] = {
            "predicted_score": sum(score * count for score, count in weighted_scores)
            / sum(count for _, count in weighted_scores),
            "task_macro_score": float(np.mean([score for score, _ in weighted_scores])),
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
                "input_digest": indexed.digest,
                "duplicate_policy": duplicate_policy,
            }
        )[:24],
        "params_id": params["params_id"],
        "subset_id": params["subset_id"],
        "method": params["method"],
        "ability_scale": params.get("ability_scale"),
        "score_interpretation": (
            "Per-task scores are the mean fixed-item 2PL probability over all "
            "params items. Model predicted_score is population-item-weighted."
        ),
        "input_digest": indexed.digest,
        "duplicate_policy": duplicate_policy,
        "inventory": asdict(indexed.summary),
        "models": model_results,
    }
    return json_safe(result)


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
    if not isinstance(params, dict) or not isinstance(
        params.get("task_scorers"), dict
    ):
        raise TypeError("Params is missing its task_scorers mapping")
    if (
        params.get("schema_version") != PARAMS_SCHEMA
        or params.get("method") != METHOD_VERSION
    ):
        raise ValueError("Params is not a supported GP-IRT 2PL params")
    if not params["task_scorers"] or any(
        not isinstance(task, str) or not isinstance(scorer, str)
        for task, scorer in params["task_scorers"].items()
    ):
        raise ValueError("Params has an invalid task_scorers mapping")
    if not isinstance(params.get("items"), list) or not params["items"]:
        raise ValueError("Params has no fitted items")
    required = {"params_id", "subset_id", "method"}
    if not required.issubset(params):
        raise ValueError("Params is missing identity or method fields")

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

    params_items = {str(row.get("item_id")): row for row in params["items"]}
    item_ids: set[str] = set()
    for row in subset:
        item_id = str(row.get("item_id", ""))
        if not isinstance(row.get("task"), str) or not isinstance(
            row.get("content_hash"), str
        ):
            raise TypeError(
                f"Selected item {item_id!r} is missing task or content_hash"
            )
        if (
            row.get("params_id") != params["params_id"]
            or row.get("subset_id") != params["subset_id"]
        ):
            raise ValueError(
                f"Subset identity does not match params for item {item_id!r}"
            )
        if item_id in item_ids:
            raise ValueError(f"Duplicate selected item {item_id!r}")
        if item_id not in params_items or not params_items[item_id].get("selected"):
            raise ValueError(f"Unknown or unselected item {item_id!r}")
        if row["task"] != params_items[item_id].get("task"):
            raise ValueError(
                f"Subset task does not match params for item {item_id!r}"
            )
        if row.get("content_hash") != params_items[item_id].get("content_hash"):
            raise ValueError(
                f"Subset content does not match params for item {item_id!r}"
            )
        item_ids.add(item_id)
    expected = {
        str(row["item_id"]) for row in params["items"] if bool(row.get("selected"))
    }
    if item_ids != expected:
        raise ValueError(
            "Subset JSONL is not the exact subset recorded by the params"
        )
    return params, subset


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
        "predicted_score": None,
        "observations": 0,
        "subset_items": subset_items,
        "coverage": 0.0,
    }
