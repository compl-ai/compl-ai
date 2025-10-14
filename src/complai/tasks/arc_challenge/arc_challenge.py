from inspect_ai import Task
from inspect_ai import task
from inspect_evals.arc import arc_challenge as inspect_arc_challenge


@task(technical_requirement="Capabilities, Performance, and Limitations")
def arc_challenge() -> Task:
    return inspect_arc_challenge()
