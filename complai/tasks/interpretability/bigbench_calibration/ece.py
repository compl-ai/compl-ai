from __future__ import annotations

from typing import TypedDict

import numpy as np


class ECEDetails(TypedDict):
    count_per_bin: list[int]
    total_correct_per_bin: list[int]
    total_confidence_per_bin: list[float]
    bin_edges: list[float]


def compute_ece_with_info(
    *,
    prediction_confidence: np.ndarray | list[float],
    is_correct: np.ndarray | list[bool],
    n_bins: int = 10,
) -> tuple[float, ECEDetails]:
    """Calculates the Expected Calibration Error.

    Args:
        prediction_confidence: The confidence of the prediction, of shape ``(N,)``.
        is_correct: Whether the prediction is correct, of shape ``(N,)``.
        n_bins: Number of bins.

    Returns:
        A tuple containing the expected calibration error and detailed information
        used to compute it.
    """
    # Validation
    prediction_confidence = np.array(prediction_confidence, dtype=np.float64)
    is_correct = np.array(is_correct, dtype=np.int64)

    if prediction_confidence.ndim != 1:
        raise ValueError(
            f"`prediction_confidence` must be 1-dimensional but it has shape: {prediction_confidence.shape}."
        )
    if is_correct.ndim != 1:
        raise ValueError(
            f"`is_correct` must be 1-dimensional but it has shape: {is_correct.shape}."
        )
    if prediction_confidence.shape[0] != is_correct.shape[0]:
        raise ValueError(
            f"`prediction_confidence` and `is_correct` must have the same shape, but "
            f"they have respectively shapes {prediction_confidence.shape} and "
            f"{is_correct.shape}."
        )

    # `confidences` value must be floats in [0, 1]
    if prediction_confidence.min() < 0 or prediction_confidence.max() > 1:
        raise ValueError(
            f"`prediction_confidence` must be between 0 and 1, but it has range "
            f"[{prediction_confidence.min()}, {prediction_confidence.max()}]."
        )

    # `is_correct` values have to be 0 or 1
    if is_correct.min() < 0 or is_correct.max() > 1:
        raise ValueError(
            f"`is_correct` must be 0 or 1, but the values are: {list(np.unique(is_correct))}."
        )

    # Bin assignment
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # We do -1 because the bin 0 is reserved for values <= the first bin.
    bin_assignments = np.digitize(prediction_confidence, bins=bin_edges, right=True) - 1
    # Edge case: conf = 0. In this case, the bin_assignment is -1 but it should be 0
    bin_assignments[prediction_confidence == 0] = 0

    # Compute (count, avg confidence, accuracy) per bin
    count_per_bin = np.bincount(bin_assignments)
    total_confidence_per_bin = np.bincount(
        bin_assignments, weights=prediction_confidence
    )
    total_correct_per_bin = np.bincount(bin_assignments, weights=is_correct)

    # include empty bins for debug
    details_info = ECEDetails(
        count_per_bin=count_per_bin.tolist(),
        total_confidence_per_bin=total_confidence_per_bin.tolist(),
        total_correct_per_bin=total_correct_per_bin.tolist(),
        bin_edges=bin_edges.tolist(),
    )

    # Deal with empty bins by removing them
    bin_non_empty = count_per_bin > 0
    count_per_bin = count_per_bin[bin_non_empty]
    total_confidence_per_bin = total_confidence_per_bin[bin_non_empty]
    total_correct_per_bin = total_correct_per_bin[bin_non_empty]

    confidence_per_bin = total_confidence_per_bin / count_per_bin
    accuracy_per_bin = total_correct_per_bin / count_per_bin

    # Compute ECE
    diff_per_bin = np.abs(confidence_per_bin - accuracy_per_bin)
    ece_score = np.average(diff_per_bin, weights=count_per_bin)
    return float(ece_score), details_info


def compute_ece(
    *,
    prediction_confidence: np.ndarray | list[float],
    is_correct: np.ndarray | list[bool],
    n_bins: int = 10,
) -> float:
    """Calculates the Expected Calibration Error.

    Args:
        prediction_confidence: The confidence of the prediction, of shape ``(N,)``.
        is_correct: Whether the prediction is correct, of shape ``(N,)``.
        n_bins: Number of bins.

    Returns:
        The expected calibration error.
    """
    score, _ = compute_ece_with_info(
        prediction_confidence=prediction_confidence,
        is_correct=is_correct,
        n_bins=n_bins,
    )
    return score
