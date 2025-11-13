from inspect_ai import Task
from inspect_ai import task
from inspect_evals.mask import mask as inspect_mask


@task(technical_requirement="Societal Alignment")
def mask() -> Task:
    return inspect_mask()
