from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_evals.bbq import bbq as inspect_bbq
from inspect_evals.bbq.bbq import BBQSubset

from complai.scorers.choice import choice


@task
def bbq(
    subsets: BBQSubset | list[BBQSubset] | None = None, shuffle: bool = False
) -> Task:
    return task_with(inspect_bbq(subsets, shuffle), scorer=choice())
