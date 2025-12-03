from inspect_ai import Task
from inspect_ai import task
from inspect_evals.mask import mask as inspect_mask


@task(technical_requirement="Societal Alignment")
def mask(
    binary_judge_model: str = "openai/gpt-4o",
    numeric_judge_model: str = "openai/o3-mini",
) -> Task:
    return inspect_mask(
        binary_judge_model=binary_judge_model, numeric_judge_model=numeric_judge_model
    )
