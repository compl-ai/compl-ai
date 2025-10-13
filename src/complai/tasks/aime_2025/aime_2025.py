from inspect_ai import Epochs
from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_ai.scorer import Score
from inspect_ai.scorer import score_reducer
from inspect_ai.scorer import ScoreReducer
from inspect_ai.scorer import value_to_float
from inspect_evals.aime2025.aime2025 import aime2025 as inspect_aime2025


@score_reducer(name="aime_2025")
def aime_2025_reducer() -> ScoreReducer:
    to_float = value_to_float()

    def reduce(scores: list[Score]) -> Score:
        """Compute a mean value of all scores."""
        values = [to_float(score.value) for score in scores]
        mean_value = sum(values) / len(values)
        if mean_value == 0:
            return Score(value=0)
        elif mean_value == 1:
            return Score(value=1)
        else:
            return Score(value=0.5)

    return reduce


@task(
    name="aime_2025", technical_requirement="Capabilities, Performance, and Limitations"
)
def aime_2025(num_epochs: int = 4) -> Task:
    return task_with(inspect_aime2025(), epochs=Epochs(num_epochs, aime_2025_reducer()))
