from typing import Any

from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate
from inspect_ai.solver import prompt_template


DATASET_PATH = "MathArena/aime_2025"

PROMPT_TEMPLATE = """
Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


def record_to_sample(record: dict[str, Any]) -> Sample:
    sample = Sample(
        id=record["problem_idx"],
        input=record["problem"],
        target=str(record["answer"]),
        metadata={"problem_type": record["problem_type"]},
    )
    return sample


def aime_2025_dataset() -> Dataset:
    return hf_dataset(path=DATASET_PATH, split="train", sample_fields=record_to_sample)


@task(technical_requirement="Capabilities, Performance, and Limitations")
def aime_2025() -> Task:
    return Task(
        dataset=aime_2025_dataset(),
        solver=[prompt_template(PROMPT_TEMPLATE), generate()],
        scorer=match(numeric=True),
    )
