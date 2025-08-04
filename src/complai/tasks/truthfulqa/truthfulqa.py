from inspect_ai import Task
from inspect_ai import task
from inspect_evals.truthfulqa import truthfulqa


@task(
    name="truthfulqa",
    technical_requirement="Capabilities, Performance, and Limitations",
)
def wrapped_truthfulqa() -> Task:
    return truthfulqa()
