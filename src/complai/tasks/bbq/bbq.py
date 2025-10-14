from inspect_ai import Task
from inspect_ai import task
from inspect_evals.bbq import bbq as inspect_bbq
from inspect_evals.bbq.bbq import BBQSubset


@task(technical_requirement="Representation — Absence of Bias")
def bbq(
    subsets: BBQSubset | list[BBQSubset] | None = None, shuffle: bool = False
) -> Task:
    return inspect_bbq(subsets, shuffle)
