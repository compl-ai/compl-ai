from typing import Literal

from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_evals.truthfulqa import truthfulqa as inspect_truthfulqa

from complai.scorers.choice import choice


@task
def truthfulqa(target: Literal["mc1", "mc2"] = "mc2") -> Task:
    return task_with(
        inspect_truthfulqa(target), scorer=choice(multiple_correct=target == "mc2")
    )
