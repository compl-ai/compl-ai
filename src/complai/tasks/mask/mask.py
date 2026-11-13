from inspect_ai import Task
from inspect_ai import task
from inspect_evals.mask import mask as inspect_mask


# Reasoning judges need more than inspect_evals' default 500-token judge budget, or
# their JSON verdict is cut off.
@task
def mask(
    binary_judge_model: str = "openai/gpt-6-luna",
    numeric_judge_model: str = "openai/gpt-6-luna",
    judge_max_tokens: int = 8000,
) -> Task:
    return inspect_mask(
        binary_judge_model=binary_judge_model,
        numeric_judge_model=numeric_judge_model,
        judge_max_tokens=judge_max_tokens,
    )
