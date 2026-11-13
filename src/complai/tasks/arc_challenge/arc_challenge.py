from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_evals.arc import arc_challenge as inspect_arc_challenge

from complai.scorers.choice import choice


@task
def arc_challenge() -> Task:
    return task_with(inspect_arc_challenge(), scorer=choice())
