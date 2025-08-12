import math
from typing import Any
from typing import Literal

import datasets
import numpy as np
from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatCompletionChoice
from inspect_ai.scorer import metric
from inspect_ai.scorer import MetricProtocol
from inspect_ai.scorer import SampleScore
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import Target
from inspect_ai.scorer import Value
from inspect_ai.solver import generate
from inspect_ai.solver import Solver
from inspect_ai.solver import solver
from inspect_ai.solver import system_message
from inspect_ai.solver import TaskState
from scipy.special import softmax

from complai.tasks.utils.ece import compute_ece
from complai.tasks.utils.logprobs import get_logprobs_first_token
from complai.tasks.utils.logprobs import get_logprobs_last_tokens
from complai.tasks.utils.strings import extract_alphabetic_option
from complai.tasks.utils.strings import OptionPosition


# Available BigBench tasks
BigBenchTask = Literal[
    "auto_debugging",
    "bbq_lite_json",
    "code_line_description",
    "conceptual_combinations",
    "conlang_translation",
    "emoji_movie",
    "formal_fallacies_syllogisms_negation",
    "hindu_knowledge",
    "known_unknowns",
    "language_identification",
    "logic_grid_puzzle",
    "logical_deduction",
    "misconceptions_russian",
    "novel_concepts",
    "play_dialog_same_or_different",
    "strange_stories",
    "strategyqa",
    "symbol_interpretation",
    "vitaminc_fact_verification",
    "winowhy",
]

_CHOICE_SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant. For each snippet of text, complete the answer by selecting from the options A,B,C,D,E. The answer should consist of a single letter option A,B,C,D,E and no additional text. The following are multiple choice questions (with answers) about {bigbench_task}.

{few_shot_examples}
"""


def format_question_and_choices(question: str, choices: list[str], target: str) -> str:
    return (
        f"{question}\n"
        + "\n".join(
            [f"{choice}. {answer}" for choice, answer in zip(_CHOICE_SYMBOLS, choices)]
        )
        + f"\nAnswer: {target}"
    )


def get_input_choices_target(
    record: dict[str, Any], format_with_target: bool, shuffle_choices: bool
) -> tuple[str, list[str], str]:
    question = record["inputs"].splitlines()[0]
    choices = record["multiple_choice_targets"]
    target_idx = record["multiple_choice_scores"].index(1)
    target = _CHOICE_SYMBOLS[target_idx]

    if shuffle_choices:
        rng = np.random.default_rng(record["idx"])
        shuffled_indices = rng.permutation(len(choices))
        choices = [choices[i] for i in shuffled_indices]
        target = _CHOICE_SYMBOLS[shuffled_indices.tolist().index(target_idx)]

    input_str = format_question_and_choices(
        question, choices, target if format_with_target else ""
    )

    return input_str, choices, target


def bigbench_dataset(
    bigbench_task: BigBenchTask, split: str, shuffle_choices: bool
) -> Dataset:
    def _record_to_sample(record: dict[str, Any]) -> Sample:
        input_str, choices, target = get_input_choices_target(
            record, format_with_target=False, shuffle_choices=shuffle_choices
        )

        return Sample(
            input=input_str,
            choices=choices,
            target=target,
            metadata={"num_choices": len(choices)},
        )

    dataset = hf_dataset(
        path="tasksource/bigbench",
        name=bigbench_task,
        split=split,
        sample_fields=_record_to_sample,
    )

    return dataset


@solver
def bigbench_calibration_system_message(
    bigbench_task: BigBenchTask,
    num_few_shot: int,
    few_shot_split: str,
    few_shot_seed: int,
    shuffle_choices: bool,
) -> Solver:
    if num_few_shot < 1:
        few_shot_examples = ""
    else:
        few_shot_dataset = (
            datasets.load_dataset(
                "tasksource/bigbench", name=bigbench_task, split=few_shot_split
            )
            .shuffle(seed=few_shot_seed)
            .take(num_few_shot)
        )

        few_shot_examples = "\n\n".join(
            get_input_choices_target(
                record, format_with_target=True, shuffle_choices=shuffle_choices
            )[0]
            for record in few_shot_dataset
        )

    return system_message(
        template=SYSTEM_PROMPT_TEMPLATE,
        bigbench_task=bigbench_task,
        few_shot_examples=few_shot_examples,
    )


@metric
def ece() -> MetricProtocol:
    def metric(scores: list[SampleScore]) -> Value:
        prediction_confidence = [
            score.score.value["confidence"]
            for score in scores
            if isinstance(score.score.value, dict) and "confidence" in score.score.value
        ]
        is_correct = [
            score.score.value["is_correct"]
            for score in scores
            if isinstance(score.score.value, dict) and "is_correct" in score.score.value
        ]
        return 1 - compute_ece(
            prediction_confidence=prediction_confidence, is_correct=is_correct
        )

    return metric


@scorer(metrics=[ece()])
def bigbench_calibration_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Parse and score completion
        completion = state.output.completion
        parsed_answer, option_position = extract_alphabetic_option(completion)
        is_correct = parsed_answer == state.target.text

        # Get logprobs
        assert len(state.output.choices) == 1
        completion_choice: ChatCompletionChoice = state.output.choices[0]
        choice_symbols = _CHOICE_SYMBOLS[: state.metadata["num_choices"]]
        if option_position == OptionPosition.START:
            choice_symbols_logprobs = get_logprobs_first_token(
                completion_choice, choice_symbols
            )
        elif option_position == OptionPosition.END:
            choice_symbols_logprobs = get_logprobs_last_tokens(
                completion_choice, choice_symbols, last_k=3
            )
        else:
            choice_symbols_logprobs = [-math.inf] * len(choice_symbols)

        # Check if sample is valid and compute confidence
        is_valid_sample = max(choice_symbols_logprobs) != -math.inf
        choice_symbols_probs = softmax(choice_symbols_logprobs)
        confidence = float(np.max(choice_symbols_probs))

        return Score(
            value={"confidence": confidence, "is_correct": is_correct},
            answer=parsed_answer,
            metadata={
                "is_valid_answer": is_valid_sample,
                "option_position": option_position.value,
            },
        )

    return score


@task(technical_requirement="Interpretability")
def bigbench_calibration(
    bigbench_task: BigBenchTask = "emoji_movie",
    num_few_shot: int = 3,
    split: str = "train",
    few_shot_split: str = "validation",
    few_shot_seed: int = 0,
    shuffle_choices: bool = True,
) -> Task:
    """BigBench calibration benchmark.

    Args:
        bigbench_task: Which BigBench task to evaluate
        num_few_shot: Number of few-shot examples
        split: Dataset split to use
        few_shot_split: Split to use for few-shot examples
        few_shot_seed: Random seed for sampling few-shot examples
        shuffle_choices: Whether to shuffle choices
    """
    return Task(
        dataset=bigbench_dataset(bigbench_task, split, shuffle_choices),
        solver=[
            bigbench_calibration_system_message(
                bigbench_task,
                num_few_shot,
                few_shot_split,
                few_shot_seed,
                shuffle_choices,
            ),
            generate(logprobs=True, top_logprobs=20),
        ],
        scorer=bigbench_calibration_scorer(),
    )
