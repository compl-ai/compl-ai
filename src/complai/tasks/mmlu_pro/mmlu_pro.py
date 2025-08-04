from inspect_ai import Task
from inspect_ai import task
from inspect_evals.mmlu_pro import mmlu_pro


@task(
    name="mmlu_pro", technical_requirement="Capabilities, Performance, and Limitations"
)
def wrapped_mmlu_pro() -> Task:
    return mmlu_pro()
