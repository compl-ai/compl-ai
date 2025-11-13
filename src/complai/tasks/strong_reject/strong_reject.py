from inspect_ai import Task
from inspect_ai import task
from inspect_evals.strong_reject import strong_reject as inspect_strong_reject


@task(technical_requirement="Cyberattack Resilience")
def strong_reject() -> Task:
    return inspect_strong_reject()
