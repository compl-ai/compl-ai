"""Minibench gp_irt_2pl_auto: source-calibrated direct/IRT score blending.

Ported from minibench_lab/artifacts/irt.py. The shared 2PL fitter is reused;
the local 10-sweep fit, 40-step ability solver and three-fold blend calibration
are preserved. Calibration is frozen at fit time for the selected subset.
"""

from typing import Any, Literal

import numpy as np

from complai.irt import TwoPLFit, fit_2pl, sigmoid

Estimator = Literal["irt", "gp_irt"]
METHOD_VERSION = "dispersion23-gp-irt-2pl-auto-v1"
RIDGE = 0.01
FIT_ITERATIONS = 10
CV_FOLDS = 3
BLENDS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def fit_model(scores: np.ndarray) -> TwoPLFit:
    return fit_2pl(
        scores, ridge=RIDGE, slope_ridge=RIDGE, iterations=FIT_ITERATIONS,
        _reset_unidentified_slopes=False,
    )


def resolve_estimator(params: dict[str, Any], requested: Estimator | None) -> Estimator:
    estimator = requested or params.get(
        "estimator", "gp_irt" if params.get("method") == METHOD_VERSION else "irt",
    )
    if estimator not in {"irt", "gp_irt"}:
        raise ValueError("estimator must be irt or gp_irt")
    if estimator == "gp_irt" and (
        params.get("method") != METHOD_VERSION
        or not isinstance(params.get("gp_irt"), dict)
        or not isinstance(params["gp_irt"].get("task_blends"), dict)
    ):
        raise ValueError(
            "gp_irt requires source-calibrated parameters; regenerate with "
            "complai minify fit --estimator gp_irt"
        )
    return estimator


def task_blend(params: dict[str, Any], task: str) -> float:
    value = params["gp_irt"]["task_blends"].get(task)
    if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Missing or invalid gp_irt blend for task {task!r}")
    return float(value)


def estimate_ability(
    values: np.ndarray, a: np.ndarray, c: np.ndarray, weights: np.ndarray,
    *, ridge: float = RIDGE,
) -> tuple[float, float, int]:
    """The existing Minibench weighted MAP Newton solver, without algorithm changes."""
    weights = weights * (len(weights) / weights.sum())
    theta = 0.0
    regularization = max(float(ridge), 1e-8)
    used_iterations = 0
    for iteration in range(40):
        used_iterations = iteration + 1
        predicted = sigmoid(a * theta + c)
        gradient = float(np.sum(weights * a * (values - predicted)) - regularization * theta)
        information = float(np.sum(weights * a * a * predicted * (1.0 - predicted)) + regularization)
        step = float(np.clip(gradient / information, -2.0, 2.0))
        theta += step
        if abs(step) < 1e-8:
            break
    predicted = sigmoid(a * theta + c)
    information = float(np.sum(weights * a * a * predicted * (1.0 - predicted)) + regularization)
    return theta, float(1 / np.sqrt(information)), used_iterations


def predict_task(
    responses: np.ndarray, selected_a: np.ndarray, selected_c: np.ndarray,
    population_a: np.ndarray, population_c: np.ndarray, *, blend: float,
    weights: np.ndarray | None = None, ridge: float = RIDGE,
) -> dict[str, Any]:
    """Use only selected responses; report no unsupported blend error interval."""
    weights = np.ones(len(responses)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != responses.shape or not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("gp_irt design weights must be finite and positive")
    if not np.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError("gp_irt blend must lie in [0, 1]")
    observed = np.isfinite(responses)
    if np.any(observed):
        local_weights = weights[observed] / weights[observed].sum()
        ability, standard_error, iterations = estimate_ability(
            responses[observed], selected_a[observed], selected_c[observed], local_weights,
            ridge=ridge,
        )
        direct = float(np.sum(responses[observed] * local_weights))
    else:
        ability, standard_error, iterations, direct = 0.0, None, 0, None
    irt_score = float(np.mean(sigmoid(population_a * ability + population_c)))
    score = irt_score if direct is None else blend * direct + (1 - blend) * irt_score
    return {
        "predicted_score": float(np.clip(score, 0, 1)),
        "predicted_score_error": None,
        "observed_subset_score": direct,
        "irt_predicted_score": irt_score,
        "blend_weight": blend,
        "ability": ability,
        "ability_standard_error": standard_error,
        "ability_iterations": iterations,
    }


def calibrate_blend(scores: np.ndarray, selected: np.ndarray) -> float:
    """Choose the direct weight by inner source-model validation, as in Minibench."""
    if scores.shape[0] < 4:
        return 0.5
    errors = np.zeros(len(BLENDS))
    count = 0
    folds = min(CV_FOLDS, scores.shape[0])
    indices = np.arange(scores.shape[0])
    for fold in range(folds):
        training = indices[indices % folds != fold]
        validation = indices[indices % folds == fold]
        if len(training) < 2:
            continue
        fitted = fit_model(scores[training])
        for index in validation:
            row = scores[index]
            if not np.any(np.isfinite(row[selected])):
                continue
            prediction = predict_task(
                row[selected], fitted.discriminations[selected], fitted.intercepts[selected],
                fitted.discriminations, fitted.intercepts, blend=0,
            )
            estimates = BLENDS * prediction["observed_subset_score"] + (1 - BLENDS) * prediction["irt_predicted_score"]
            errors += np.abs(estimates - np.nanmean(row))
            count += 1
    return float(BLENDS[np.argmin(errors / count)]) if count else 0.5


def calibrate_tasks(tasks: dict[str, Any], selected_keys: list[tuple[str, int]]) -> dict[str, Any]:
    return {
        "ridge": RIDGE,
        "fit_iterations": FIT_ITERATIONS,
        "ability_iterations": 40,
        "cv_folds": CV_FOLDS,
        "blend_candidates": BLENDS.tolist(),
        "task_blends": {
            task: calibrate_blend(
                data["matrix"], np.array([i for t, i in selected_keys if t == task], dtype=int),
            )
            for task, data in tasks.items()
        },
    }
