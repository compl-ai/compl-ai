from inspect_ai import Task
from inspect_ai import task
from inspect_evals.humaneval import humaneval


@task(
    name="humaneval", technical_requirement="Capabilities, Performance, and Limitations"
)
def wrapped_humaneval() -> Task:
    return humaneval()
