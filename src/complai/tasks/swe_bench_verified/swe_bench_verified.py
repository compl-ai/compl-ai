from inspect_ai import Task
from inspect_ai import task
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench import swe_bench_verified_mini


@task(technical_requirement="Capabilities, Performance, and Limitations")
def swe_bench_verified(full: bool = False) -> Task:
    """
    SWE-bench-verified task.

    Args:
        full (bool, optional): Whether to use the full SWE-bench-verified dataset.

    Returns:
        Task: SWE-bench-verified task.
    """
    if full:
        return swe_bench()
    return swe_bench_verified_mini()
