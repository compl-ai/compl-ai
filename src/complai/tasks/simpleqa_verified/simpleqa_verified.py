from inspect_ai import Task
from inspect_ai import task
from inspect_evals.simpleqa import simpleqa_verified as inspect_simpleqa_verified


@task(technical_requirement="Societal Alignment")
def simpleqa_verified() -> Task:
    return inspect_simpleqa_verified()
