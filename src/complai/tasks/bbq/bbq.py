from inspect_ai import Task
from inspect_ai import task
from inspect_evals.bbq import bbq
from inspect_evals.bbq import BBQSubset


@task(name="bbq", technical_requirement="Representation — Absence of Bias")
def wrapped_bbq(
    subsets: BBQSubset | list[BBQSubset] | None = None, shuffle: bool = False
) -> Task:
    return bbq(subsets=subsets, shuffle=shuffle)
