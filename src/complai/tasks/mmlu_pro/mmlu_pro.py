from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_evals.mmlu_pro import mmlu_pro as inspect_mmlu_pro

from complai.scorers.choice import choice


@task
def mmlu_pro(num_fewshot: int = 0) -> Task:
    return task_with(inspect_mmlu_pro(fewshot=num_fewshot), scorer=choice())
