from inspect_ai import Epochs
from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_evals.aime2025.aime2025 import aime2025 as inspect_aime2025


@task(
    name="aime_2025", technical_requirement="Capabilities, Performance, and Limitations"
)
def aime_2025(num_epochs: int = 4) -> Task:
    return task_with(inspect_aime2025(), epochs=Epochs(num_epochs, "mean"))
