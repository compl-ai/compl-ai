from inspect_ai import Task
from inspect_ai import task
from inspect_evals.hle import hle


@task(name="hle", technical_requirement="Capabilities, Performance, and Limitations")
def wrapped_hle() -> Task:
    return hle()
