from typing import Literal

from inspect_ai import Task
from inspect_ai import task
from inspect_ai import task_with
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model
from inspect_evals.simpleqa import simpleqa_verified as inspect_simpleqa_verified


@task(technical_requirement="Societal Alignment")
def simpleqa_verified(
    scorer: Literal["tool", "original"] = "tool",
    temperature: float = 1.0,
    max_tokens: int = 2048,
    grader_model: str = "openai/gpt-4.1-2025-04-14",
    grader_temperature: float = 1.0,
) -> Task:
    return task_with(
        inspect_simpleqa_verified(scorer=scorer),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        model_roles={
            "grader": get_model(
                grader_model, config=GenerateConfig(temperature=grader_temperature)
            )
        },
    )
