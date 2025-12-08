from inspect_ai import Epochs
from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_evals.gpqa import gpqa_diamond as inspect_gpqa_diamond

# Default is 4 epochs as per OpenAI's simple-evals implementation
# https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/gpqa_eval.py#L20


@task(technical_requirement="Capabilities, Performance, and Limitations")
def gpqa_diamond(num_epochs: int = 4) -> Task:
    return task_with(inspect_gpqa_diamond(), epochs=Epochs(num_epochs, "mean"))
