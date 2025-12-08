from typing import Literal

from inspect_ai import Task
from inspect_ai import task
from inspect_evals.truthfulqa import truthfulqa as inspect_truthfulqa


@task(technical_requirement="Societal Alignment")
def truthfulqa(target: Literal["mc1", "mc2"] = "mc2") -> Task:
    return inspect_truthfulqa(target)
