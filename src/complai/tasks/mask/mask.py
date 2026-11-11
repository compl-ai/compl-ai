from inspect_ai import Task
from inspect_ai import task
from inspect_evals.mask import mask as inspect_mask


@task
def mask(
    binary_judge_model: str = "openai/gpt-6-luna",
    numeric_judge_model: str = "openai/gpt-6-luna",
) -> Task:
    return inspect_mask(
        binary_judge_model=binary_judge_model, numeric_judge_model=numeric_judge_model
    )
