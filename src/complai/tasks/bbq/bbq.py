from inspect_ai import Task
from inspect_ai import task
from inspect_evals.bbq import bbq


@task(name="bbq", technical_requirement="Representation — Absence of Bias")
def wrapped_bbq() -> Task:
    return bbq()
