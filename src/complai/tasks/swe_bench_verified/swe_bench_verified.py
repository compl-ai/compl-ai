from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench import swe_bench_verified_mini


@task(technical_requirement="Capabilities, Performance, and Limitations")
def swe_bench_verified(full: bool = False, message_limit: int = 200) -> Task:
    """
    SWE-bench-verified task.

    Args:
        full (bool, optional): Whether to use the full SWE-bench-verified dataset.

    Returns:
        Task: SWE-bench-verified task.
    """
    if full:
        task = swe_bench()
    task = swe_bench_verified_mini()

    return task_with(task, message_limit=message_limit)
