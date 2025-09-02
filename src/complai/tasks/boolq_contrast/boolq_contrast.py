from collections import defaultdict
from functools import partial
from itertools import count
from typing import Any
from typing import cast

import numpy as np
from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import match
from inspect_ai.scorer import Metric
from inspect_ai.scorer import metric
from inspect_ai.scorer import SampleScore
from inspect_ai.solver import generate
from inspect_ai.solver import system_message


DATASET_PATH = "compl-ai/boolq_contrastset"

BOOLQ_SYSTEM_PROMPT = """
You are a helpful assistant. For each snippet of text, answer the question by selecting from the option 'TRUE' or 'FALSE'. The answer should be exactly 'TRUE' or 'FALSE'.
""".strip()


def record_to_sample(
    record: dict[str, Any],
    num_contrasts: int,
    rng: np.random.Generator,
    id_provider: count,
) -> list[Sample]:
    # Extract relevant fields from the record
    paragraph = record["paragraph"]
    original_question = record["question"]
    original_answer = record["answer"]
    perturbed_questions = record["perturbed_questions"]

    # Filter perturbed questions that have a valid answer that is different from
    # the original answer
    contrasts = [
        perturbed_question
        for perturbed_question in perturbed_questions
        if perturbed_question["answer"] != original_answer
        and perturbed_question["answer"] in ["TRUE", "FALSE"]
    ]

    # Randomly choose num_contrasts pairs
    chosen_contrasts = rng.choice(
        contrasts, size=min(num_contrasts, len(contrasts)), replace=False
    )

    # Format paragraph
    paragraph = paragraph.strip().capitalize()

    # Format questions
    def format_question(question: str) -> str:
        question = question.strip().capitalize()
        if not question.endswith("?"):
            return question + "?"
        return question

    original_question = format_question(original_question)
    for contrast in chosen_contrasts:
        contrast["question"] = format_question(contrast["perturbed_q"])

    # Init messages
    messages: list[ChatMessage] = [
        ChatMessageUser(
            content=f"{paragraph}\nQuestion: {original_question}\nAnswer: {original_answer}"
        ),
        ChatMessageAssistant(content=original_answer),
    ]

    # Get a unique id for this set of contrasts
    id = next(id_provider)

    # Create one sample per contrast question
    samples = [
        Sample(
            input=messages
            + [ChatMessageUser(content=f"Question: {contrast['question']}\nAnswer: ")],
            target=contrast["answer"],
            metadata={
                "paragraph": paragraph,
                "original_question": original_question,
                "original_answer": original_answer,
                "id": id,
            },
        )
        for contrast in chosen_contrasts
    ]

    return samples


def boolq_contrastset_dataset(num_contrasts: int, contrast_seed: int) -> Dataset:
    rng = np.random.default_rng(seed=contrast_seed)
    id_provider = count(0)

    return hf_dataset(
        path=DATASET_PATH,
        split="test",
        sample_fields=partial(
            record_to_sample,
            num_contrasts=num_contrasts,
            rng=rng,
            id_provider=id_provider,
        ),
    )


@metric
def boolq_contrastset_metric() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        # Aggregate samples with the same original question (group by id)
        correct: dict[int, int] = defaultdict(int)
        total: dict[int, int] = defaultdict(int)
        for score in scores:
            id = cast(dict, score.sample_metadata)["id"]
            if score.score.as_bool():
                correct[id] += 1
            total[id] += 1

        # Calculate accuracy for each original question
        accuracies = [correct[id] / total[id] for id in total]

        return np.mean(accuracies, dtype=float)

    return metric


@task(technical_requirement="Robustness and Predictability")
def boolq_contrast(num_contrasts: int = 3, contrast_seed: int = 0) -> Task:
    """BoolQ contrast sets task.

    BoolQ contrast sets measure if the model can answer a given yes-no question
    accurately, when presented with a distraction: seemingly similar questions
    that have the opposite answer.
    """
    assert num_contrasts > 0, "Number of contrasts must be at least 1."
    return Task(
        dataset=boolq_contrastset_dataset(num_contrasts, contrast_seed),
        solver=[system_message(BOOLQ_SYSTEM_PROMPT), generate()],
        scorer=match(location="begin", ignore_case=True),
        metrics=[boolq_contrastset_metric()],
    )
