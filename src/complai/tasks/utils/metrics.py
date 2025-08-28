import itertools
from typing import Any


def iou(x: set[Any], y: set[Any]) -> float:
    """Calculates the intersection over union (IoU) between two sets."""
    intersection = len(x & y)
    union = len(x | y)
    if union == 0:
        return 1
    return intersection / union


def serp_ms(results: tuple[Any, ...], reference: set[Any]) -> float:
    """Calculates the SERP-MS (Search Engine Result Page - Mean Squared) metric.

    The metric is contained in the interval [-1,1]. Higher is better.

    Args:
        results: The search engine results page.
        reference: The set of acceptable search results.

    Notes:
        Formula described at: https://eslam-hussein.me/pdfs/papers/hussein_CSCW2020.pdf.
    """
    n = len(results)
    if n == 0:
        if len(reference) == 0:
            return 1
        else:
            return 0

    # We use ranks starting at zero. This leads to a simplified (equivalent) formula.
    s = sum(n - rank for rank, result in enumerate(results) if result in reference)
    return s / (n * (n + 1) / 2)


def prag_score(results: tuple[Any, ...], reference: tuple[Any, ...]) -> float:
    """Calculate the Pairwise Ranking Accuracy Gap (PRAG) score.

    The metric is contained in the interval [0,1]. Higher is better.

    Note that this implementation does not filter out or check for duplicates.

    Notes:
        Formula described at: https://arxiv.org/pdf/2305.07609. Note that it does not
        exactly match the original formula derived at https://arxiv.org/pdf/1903.00780.
    """
    n = len(results)
    if n == 0:
        if len(reference) == 0:
            return 1
        else:
            return 0

    if n == 1:
        # We cannot form any pairs. Instead, we just check whether the single result
        # is relevant.
        if results[0] in reference:
            return 1
        else:
            return 0

    # Compute the fraction of pairs of values in `results` that have the same order as
    # in the reference.
    s = 0
    for (result_rank_x, x), (result_rank_y, y) in itertools.combinations(
        enumerate(results), 2
    ):
        if result_rank_x >= result_rank_y:
            continue

        reference_rank_x = None
        reference_rank_y = None
        for reference_rank, z in enumerate(reference):
            if z == x:
                reference_rank_x = reference_rank
            if z == y:
                reference_rank_y = reference_rank

        # In the result we have rank_x < rank_y. The following checks whether also
        # in the reference we have rank_x < rank_y. If a value is not in the
        # reference, the rank is None and treated as +inf.
        if reference_rank_x is not None and (
            reference_rank_y is None or reference_rank_x < reference_rank_y
        ):
            s += 1

    return s / (n * (n - 1) / 2)
