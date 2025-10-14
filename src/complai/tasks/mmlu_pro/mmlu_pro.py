from inspect_ai import Task
from inspect_ai import task
from inspect_evals.mmlu_pro import mmlu_pro as inspect_mmlu_pro


@task(technical_requirement="Capabilities, Performance, and Limitations")
def mmlu_pro(num_fewshot: int = 0) -> Task:
    return inspect_mmlu_pro(fewshot=num_fewshot)
