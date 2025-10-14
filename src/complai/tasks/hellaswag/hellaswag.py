from inspect_ai import Task
from inspect_ai import task
from inspect_evals.hellaswag import hellaswag as inspect_hellaswag


@task(technical_requirement="Capabilities, Performance, and Limitations")
def hellaswag() -> Task:
    return inspect_hellaswag()
