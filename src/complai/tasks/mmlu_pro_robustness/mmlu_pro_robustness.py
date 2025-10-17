import re
from typing import cast
from typing import get_args

import numpy as np
from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import FieldSpec
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage
from inspect_ai.model import ChatMessageSystem
from inspect_ai.model import ChatMessageUser
from inspect_ai.model import get_model
from inspect_ai.scorer import choice
from inspect_ai.scorer import CORRECT
from inspect_ai.scorer import metric
from inspect_ai.scorer import MetricProtocol
from inspect_ai.scorer import SampleScore
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import Target
from inspect_ai.solver import Choices
from inspect_ai.solver import Generate
from inspect_ai.solver import multiple_choice
from inspect_ai.solver import Solver
from inspect_ai.solver import solver
from inspect_ai.solver import TaskState
from inspect_ai.solver._multiple_choice import answer_character
from inspect_ai.solver._multiple_choice import prompt
from inspect_ai.solver._multiple_choice import SINGLE_ANSWER_TEMPLATE_COT
from inspect_ai.util import collect

from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import Perturbation
from complai.tasks.mmlu_pro_robustness.perturbations.registry import PerturbationName
from complai.tasks.mmlu_pro_robustness.perturbations.registry import PERTURBATIONS


DATASET_PATH = "TIGER-Lab/MMLU-Pro"

SYSTEM_PROMPT_TEMPLATE = """
The following are multiple choice questions (with answers) about {category}. Think step by step and then finish your answer with 'ANSWER: $LETTER' (without quotes) where LETTER is the correct letter choice.

{examples}
""".strip()

FIELD_SPEC = FieldSpec(
    input="question",
    target="answer",
    choices="options",
    id="question_id",
    metadata=["cot_content", "category"],
)


def filter_dataset(
    dataset: Dataset, categories: list[str] | None, num_samples: int | None = None
) -> Dataset:
    """
    Filters the MMLU-Pro dataset by categories. If num_samples
    is provided, returns num_samples samples.

    Args:
        dataset (Dataset): Dataset object to be filtered.
        categories (List): List of categories to filter on.
        num_samples (int): Number of samples to take.
    """
    # Filter dataset by categories, if required
    if categories:
        dataset = dataset.filter(
            name=f"{dataset.name}_category-{'-'.join(categories)}",
            predicate=lambda sample: sample.metadata["category"] in categories
            if sample.metadata is not None
            else False,
        )

    if num_samples:
        dataset = dataset[:num_samples]

    return dataset


def mmlu_pro_robustness_dataset(
    categories: list[str] | None, num_samples: int, seed: int
) -> Dataset:
    dataset = hf_dataset(
        path=DATASET_PATH,
        split="test",
        sample_fields=FIELD_SPEC,
        shuffle_choices=True,
        shuffle=True,
        seed=seed,
    )
    dataset = filter_dataset(dataset, categories, num_samples)

    return dataset


def sample_to_fewshot(sample: Sample, perturbation: Perturbation) -> str:
    q_str = f"""Question:\n{str(perturbation.perturb(cast(str, sample.input)))}"""
    options = sample.choices if sample.choices is not None else []
    perturbed_options = [perturbation.perturb(opt) for opt in options]
    opt_str_list = []
    for i, opt in enumerate(perturbed_options):
        opt_str_list.append(f"""{chr(65 + i)}) {opt}""")
    opt_str = "\n".join(opt_str_list)
    opt_str = f"""Options:\n{opt_str}"""
    ans_str = sample.metadata["cot_content"] if sample.metadata is not None else ""
    ans_str = ans_str.replace("The answer is", "ANSWER:")
    ans_opt = ans_str.split("ANSWER:")[-1].split(".")[0].strip().strip("(").strip(")")
    ans_str = ans_str.replace(f"ANSWER: ({ans_opt})", f"ANSWER: {ans_opt}")
    final_str = "\n".join([q_str, opt_str, ans_str])

    return final_str


def parse_answers(completion: str, num_choices: int) -> set[str]:
    """
    Source: https://github.com/UKGovernmentBEIS/inspect_ai/blob/37b74c6a146e6e70d12119c8457d749c43d8f4f1/src/inspect_ai/solver/_multiple_choice.py#L83

    Convenience function for extracting answers from the state output.

    The generated response must be in the format 'ANSWER: <answers>',
    otherwise we can't extract what the model thinks is "true". We can be a
    bit flexible whether these are "AB" vs "A,B" vs "A B".

    However, if the answer isn't in the expected format the model has
    failed in the task so we'll ultimately just mark it as incorrect
    """
    # First check whether the string strictly ends with the expected answer
    # In this case, we're looking for a single line which contains the expected
    # ANSWER: <answer> string with only whitespace or a period/full stop at the end.
    match = re.search(
        r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
        completion,
        flags=re.MULTILINE,
    )

    # If we couldn't match the strict version, we can try the less strict
    # version for backward compatibility
    if match is None:
        match = re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)", completion
        )

    if match is None:
        return set()

    matched = match.group(1)

    # Strip trailing period / full stop
    matched = matched.strip()
    matched = matched.rstrip(".")

    allowed_options = set(answer_character(i) for i in range(num_choices))

    # Match must contain a single letter in the allowed choices
    if matched in allowed_options:
        answers = {matched}
        return answers

    return set()


@solver
def mmlu_pro_robustness_solver(
    num_fewshot: int, perturbations: list[Perturbation]
) -> Solver:
    mc_solver = multiple_choice(template=SINGLE_ANSWER_TEMPLATE_COT)

    fewshot_samples = hf_dataset(
        path=DATASET_PATH, split="validation", sample_fields=FIELD_SPEC
    )
    category_to_fewshot = {}

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        original_question = state.input_text
        original_choices = state.choices
        num_choices = len(original_choices)

        state = await mc_solver(state, generate)
        state.metadata["unperturbed_answer"] = parse_answers(
            state.output.completion, num_choices
        )

        category = state.metadata["category"]
        if category not in category_to_fewshot:
            category_to_fewshot[category] = filter_dataset(
                fewshot_samples, [category], num_fewshot
            )

        perturbed: list[list[ChatMessage]] = [
            [
                ChatMessageSystem(
                    content=SYSTEM_PROMPT_TEMPLATE.format(
                        category=category,
                        examples="\n\n".join(
                            [
                                sample_to_fewshot(sample, perturbation)
                                for sample in category_to_fewshot[category]
                            ]
                        ),
                    )
                ),
                ChatMessageUser(
                    content=prompt(
                        question=perturbation.perturb(original_question),
                        choices=Choices(
                            [
                                perturbation.perturb(choice.value)
                                for choice in original_choices
                            ]
                        ),
                        template=SINGLE_ANSWER_TEMPLATE_COT,
                    )
                ),
            ]
            for perturbation in perturbations
        ]

        outputs = await collect(*[get_model().generate(input) for input in perturbed])
        answers = [parse_answers(output.completion, num_choices) for output in outputs]
        state.metadata["perturbed_answers"] = ["".join(answer) for answer in answers]

        return state

    return solve


@metric
def robustness() -> MetricProtocol:
    def metric(scores: list[SampleScore]) -> float:
        mean_perturbed_score: float = np.mean(
            [score.score.value["perturbed_score"] for score in scores]  # type: ignore
        )
        mean_unperturbed_score: float = np.mean(
            [score.score.value["unperturbed_score"] for score in scores]  # type: ignore
        )

        if mean_unperturbed_score == 0:
            return 1
        else:
            return min(1, mean_perturbed_score / mean_unperturbed_score)

    return metric


@scorer(metrics=[robustness()])
def mmlu_pro_robustness_scorer() -> Scorer:
    mc_scorer = choice()

    async def score(state: TaskState, target: Target) -> Score:
        unperturbed_score = cast(Score, await mc_scorer(state, target))

        perturbed_answers = state.metadata["perturbed_answers"]
        perturbed_scores = [target.text == answer for answer in perturbed_answers]

        return Score(
            value={
                "perturbed_score": np.mean(perturbed_scores),
                "unperturbed_score": int(unperturbed_score.value == CORRECT),
            }
        )

    return score


@task(technical_requirement="Robustness and Predictability")
def mmlu_pro_robustness(
    categories: str | list[str] | None = None,
    num_samples: int = 200,
    num_fewshot: int = 5,
    perturbations: PerturbationName | list[PerturbationName] | None = None,
    seed: int = 0,
) -> Task:
    """
    MMLU-Pro robustness benchmark.

    Args:
        categories (str | list[str] | None): List of categories to evaluate on.
        num_samples (int): Number of samples to take.
        num_fewshot (int): Number of few shot examples to use.
        perturbations (PerturbationName | list[PerturbationName] | None): List of perturbations to apply.
        seed (int): Seed for choosing the samples to be evaluated on.
    """
    categories = (
        categories
        if isinstance(categories, list)
        else [categories]
        if categories
        else None
    )

    # Parse and initialize perturbations
    if perturbations is None:
        perturbations = list(get_args(PerturbationName))
    elif not isinstance(perturbations, list):
        perturbations = [perturbations]
    initialized_perturbations = [
        perturbation()
        for perturbation in PERTURBATIONS
        if perturbation.name in perturbations
    ]

    return Task(
        dataset=mmlu_pro_robustness_dataset(categories, num_samples, seed),
        solver=mmlu_pro_robustness_solver(num_fewshot, initialized_perturbations),
        scorer=mmlu_pro_robustness_scorer(),
    )
