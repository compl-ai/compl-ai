from inspect_ai import Task
from inspect_ai import task
from inspect_evals.simpleqa import simpleqa_verified as inspect_simpleqa_verified


@task(technical_requirement="Societal Alignment")
def simpleqa_verified(grader_model: str = "openai/gpt-4.1-2025-04-14") -> Task:
    return inspect_simpleqa_verified(grader_model=grader_model)
