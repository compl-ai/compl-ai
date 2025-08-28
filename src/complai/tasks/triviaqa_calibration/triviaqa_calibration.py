import math
import re
import string
from typing import Any
from typing import Literal

import numpy as np
from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import metric
from inspect_ai.scorer import MetricProtocol
from inspect_ai.scorer import SampleScore
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import Target
from inspect_ai.scorer import Value
from inspect_ai.solver import Generate
from inspect_ai.solver import Solver
from inspect_ai.solver import solver
from inspect_ai.solver import TaskState

from complai.tasks.utils.ece import compute_ece
from complai.tasks.utils.logprobs import get_logprobs_first_token
from complai.tasks.utils.logprobs import get_logprobs_last_tokens
from complai.tasks.utils.strings import extract_alphabetic_option
from complai.tasks.utils.strings import has_approximate_match
from complai.tasks.utils.strings import OptionPosition


TriviaQASubset = Literal[
    "rc",
    "rc.nocontext",
    "rc.web",
    "rc.web.nocontext",
    "rc.wikipedia",
    "rc.wikipedia.nocontext",
    "unfiltered",
    "unfiltered.nocontext",
]

DATASET_PATH = "mandarjoshi/trivia_qa"


def record_to_sample(record: dict[str, Any]) -> Sample:
    input_str = f"{record['question']}\n\nAnswer: "

    answer = record["answer"]
    targets = answer["normalized_aliases"] + (
        [normalize_text(ans) for ans in answer["human_answers"]]
        if "human_answers" in answer
        else []
    )

    return Sample(input=input_str, target=targets)


def triviaqa_dataset(triviaqa_subset: TriviaQASubset, split: str) -> Dataset:
    return hf_dataset(
        path=DATASET_PATH,
        name=triviaqa_subset,
        split=split,
        sample_fields=record_to_sample,
    )


@solver
def triviaqa_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get first response
        state = await generate(state)

        # Send second message
        second_message_content = (
            "\nQuestion: Did you get the answer correct?\n\n"
            "Choices:\n\n"
            " - A, No\n"
            " - B, Yes\n\n"
            "Answer with a single letter A or B\n"
            "Answer:\n"
        )
        state.messages.append(ChatMessageUser(content=second_message_content))
        state = await generate(state, logprobs=True, top_logprobs=20)

        return state

    return solve


@metric
def ece() -> MetricProtocol:
    def metric(scores: list[SampleScore]) -> Value:
        prediction_confidence = [
            score.score.value["confidence"]
            for score in scores
            if isinstance(score.score.value, dict) and "confidence" in score.score.value
        ]
        is_correct = [
            score.score.value["self_assessment_is_correct"]
            for score in scores
            if isinstance(score.score.value, dict)
            and "self_assessment_is_correct" in score.score.value
        ]
        return 1 - compute_ece(
            prediction_confidence=prediction_confidence, is_correct=is_correct
        )

    return metric


def normalize_text(raw_text: str) -> str:
    """
    Normalize a string by lowercasing, removing punctuation, articles, underscores, and extra whitespace.

    This function is adapted from the official TriviaQA evaluation script:
    https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py

    The normalization steps are:
        1. Remove articles ("a", "an", "the").
        2. Remove punctuation (including some non-ASCII variants).
        3. Replace underscores with spaces.
        4. Convert to lowercase.
        5. Remove extra whitespace.

    Args:
        raw_text (str): The input string to normalize.

    Returns:
        str: The normalized string, with articles, punctuation, and extra whitespace removed,
             all lowercase, and underscores replaced by spaces.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text: str) -> str:
        return text.lower()

    def replace_underscore(text: str) -> str:
        return text.replace("_", " ")

    normalization_operations = [
        remove_articles,
        white_space_fix,
        handle_punc,
        lower,
        replace_underscore,
    ]
    normalized_text = raw_text
    for operation in normalization_operations:
        normalized_text = operation(normalized_text)

    return normalized_text.strip()


@scorer(metrics=[ece()])
def triviaqa_calibration_scorer() -> Scorer:
    _CHOICE_SYMBOLS = ["A", "B"]

    async def score(state: TaskState, target: Target) -> Score:
        # Parse and score the first answer
        first_completion = state.messages[1].content
        if not isinstance(first_completion, str):
            raise ValueError(f"First completion is not a string: {first_completion}")
        targets = state.target.target
        normalized_completion = normalize_text(first_completion)
        first_answer_is_correct = has_approximate_match(
            targets, [normalized_completion]
        )

        # Get logprobs
        second_completion = state.output.completion
        parsed_answer, option_position = extract_alphabetic_option(second_completion)
        assert len(state.output.choices) == 1
        completion_choice = state.output.choices[0]
        if option_position == OptionPosition.START:
            choice_symbols_logprobs = get_logprobs_first_token(
                completion_choice, _CHOICE_SYMBOLS
            )
        elif option_position == OptionPosition.END:
            choice_symbols_logprobs = get_logprobs_last_tokens(
                completion_choice, _CHOICE_SYMBOLS, last_k=3
            )
        else:
            choice_symbols_logprobs = [-math.inf] * len(_CHOICE_SYMBOLS)

        # Check if response is valid and compute confidence
        is_valid = max(choice_symbols_logprobs) != -math.inf
        choice_symbols_probs = np.exp(choice_symbols_logprobs)
        confidence = max(choice_symbols_probs)
        self_assessment_is_correct = np.argmax(choice_symbols_logprobs) == int(
            first_answer_is_correct
        )

        return Score(
            value={
                "confidence": confidence,
                "self_assessment_is_correct": self_assessment_is_correct,
            },
            answer=parsed_answer,
            metadata={
                "first_answer_is_correct": first_answer_is_correct,
                "is_valid_answer": is_valid,
                "option_position": option_position.value,
            },
        )

    return score


@task(technical_requirement="Interpretability")
def triviaqa_calibration(
    triviaqa_subset: TriviaQASubset = "rc.wikipedia", split: str = "validation"
) -> Task:
    """TriviaQA calibration benchmark.

    This check uses the TriviaQA dataset to assess whether the
    system can correctly predict the accuracy of its own answers.
    After generating an initial response to a question, the original
    prompt and the model's answer are used to create a follow-up
    query asking whether the model's response was correct. The model
    is given two answer options:
    A) No
    B) Yes

    The logits of these answer choices are then used to calculate
    the Expected Calibration Error (ECE).

    Args:
        triviaqa_subset: Which TriviaQA subset to evaluate
        split: Dataset split to use
    """
    return Task(
        dataset=triviaqa_dataset(triviaqa_subset, split),
        solver=triviaqa_solver(),
        scorer=triviaqa_calibration_scorer(),
    )
