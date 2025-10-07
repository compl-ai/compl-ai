from inspect_ai import Task
from inspect_ai import task
from inspect_evals.arc import arc_challenge


@task(
    name="arc_challenge",
    technical_requirement="Capabilities, Performance, and Limitations",
)
def wrapped_arc_challenge() -> Task:
    return arc_challenge()
