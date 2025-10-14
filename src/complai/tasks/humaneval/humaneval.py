from inspect_ai import Task
from inspect_ai import task
from inspect_evals.humaneval import humaneval as inspect_humaneval


@task(technical_requirement="Capabilities, Performance, and Limitations")
def humaneval() -> Task:
    return inspect_humaneval()
