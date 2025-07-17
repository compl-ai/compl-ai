import numpy as np


def paired_t_test_cohensd(paired_t_statistic: float, num_samples: int) -> float:
    """Calculates cohen's d effect size for paired t-test.

    In comparison to the t-statistic, it standardizes the result to the
    standard normal distribution such that it can be compared more easily. Rough map
    from experienced difference between groups and cohen's d values:
    Small: 0.2, Medium: 0.5, Large: 0.8

    Args:
        paired_t_statistic: Resulting statistic of paired t-test.
        num_samples: Sample size that is equal for both populations since paired t-test
            is used.

    Returns:
        Cohen's d effect size

    Notes:
        Formula described at https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/td
    """
    return np.abs(paired_t_statistic) * np.sqrt(1 / (num_samples - 1))
