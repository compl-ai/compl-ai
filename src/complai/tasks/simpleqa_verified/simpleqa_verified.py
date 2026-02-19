from inspect_ai import Task
from inspect_ai import task
from inspect_evals.simpleqa import simpleqa_verified as inspect_simpleqa_verified


@task(technical_requirement="Societal Alignment")
def simpleqa_verified(
    temperature: float = 1.0,
    max_tokens: int = 2048,
    grader_model: str = "openai/gpt-4.1-2025-04-14",
    grader_temperature: float = 1.0,
) -> Task:
    return inspect_simpleqa_verified(
        temperature=temperature,
        max_tokens=max_tokens,
        grader_model=grader_model,
        grader_temperature=grader_temperature,
    )
