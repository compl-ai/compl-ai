import numpy as np


def simplified_gini_coefficient(scores: list[float]) -> float:
    # Link to formula: https://www.statsdirect.com/help/nonparametric_methods/gini.htm
    n = len(scores)
    if n == 0 or all(score == 0 for score in scores):
        return np.nan

    mean_score = sum(scores) / n
    sum_of_differences = sum(abs(x - y) for x in scores for y in scores)

    # The simplified Gini coefficient formula
    gini = sum_of_differences / (2 * n**2 * mean_score)

    return gini
