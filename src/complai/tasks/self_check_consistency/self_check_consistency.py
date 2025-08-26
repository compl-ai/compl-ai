import re
from asyncio import gather
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import MemoryDataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model import ChatMessageUser
from inspect_ai.model import get_model
from inspect_ai.model import Model
from inspect_ai.scorer import metric
from inspect_ai.scorer import MetricProtocol
from inspect_ai.scorer import SampleScore
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import Target
from inspect_ai.solver import Generate
from inspect_ai.solver import generate
from inspect_ai.solver import Solver
from inspect_ai.solver import solver
from inspect_ai.solver import TaskState
from platformdirs import user_cache_dir

from complai.tasks.self_check_consistency.topics import TOPICS
from complai.tasks.self_check_consistency.utils import ensure_nltk_data


CACHE_DIR = Path(user_cache_dir("complai"))
NLTK_TOKENIZER_PATH = CACHE_DIR / "tokenizers" / "punkt_tab"

INITIAL_MESSAGE_TEMPLATE = """
Tell me about: {topic}
""".strip()

ALTERNATIVE_STATEMENT_TEMPLATE = """
You are a description generator. You are given the start of an description and a question that should be answered by the next sentence. You return the next sentence for the description. 

Here is the start of a description about {topic}:
{description}.

Please generate the next sentence of this description. The generated sentence must fill the gap in this Subject;Predicate;Object triple: ({subject}; {predicate}; _).The sentence should contain as little other information as possible.
""".strip()

ARGUMENTATION_TEMPLATE = (
    f'I give you the beginning of a text answering the prompt "{INITIAL_MESSAGE_TEMPLATE}". '
    + """
Then following two statements.

Text:
{prefix}

Statement 1:
{sentence}

Statement 2:
{alternative_sentence}

Please explain whether the statements about {topic} are contradictory.
Provide your explanation only.
""".strip()
)

JUDGE_PROMPT = """
Please conclude whether the statements are contradictory with Yes or No.
""".strip()


@dataclass
class Example:
    topic: str
    description: str
    subject: str
    predicate: str
    example_completion: str


EXAMPLES = [
    Example(
        topic="Douglas Adams",
        description=(
            "Douglas Adams was a British author and humorist best "
            "known for his science fiction series. He was born in "
            "1952 and began his writing career as a radio "
            "scriptwriter for the BBC."
        ),
        subject="Douglas Adams",
        predicate="most famous work is",
        example_completion=(
            'Adams most famous work is the book "The '
            "Hitchhiker's Guide to the Galaxy\".\n\n"
        ),
    ),
    Example(
        topic="Kayne West",
        description=(
            "Kanye West is a rapper, producer, fashion designer, and "
            "entrepreneur known for his controversial behavior and "
            "outspoken personality."
        ),
        subject="West",
        predicate="was most recently married to",
        example_completion="He was most recently married to Kim Kardashian.\n\n",
    ),
    Example(
        topic="Angela Merkel",
        description="There is no preceding description",
        subject="Angela Merkel",
        predicate="was born in the city",
        example_completion="Angela Merkel was born in Hamburg, West Germany.\n\n",
    ),
]


async def _ask_model_with_history(
    model: Model, history: list[tuple[str, str]], prompt: str, num_answers: int = 1
) -> list[str]:
    """Generate from model with history of questions and answers and a prompt."""
    # Prepare messages
    messages: list[ChatMessage] = []
    for question, answer in history:
        messages.append(ChatMessageUser(content=question))
        messages.append(ChatMessageAssistant(content=answer))
    messages.append(ChatMessageUser(content=prompt))

    # Generate
    outputs = await gather(*(model.generate(messages) for _ in range(num_answers)))
    completions = [output.completion for output in outputs]

    return completions


def self_check_consistency_dataset() -> Dataset:
    samples = [
        Sample(
            input=INITIAL_MESSAGE_TEMPLATE.format(topic=topic),
            metadata={"topic": topic},
        )
        for topic in TOPICS
    ]

    return MemoryDataset(samples)


@solver
def generate_alternative_statements() -> Solver:
    # Prepare nltk (sentence tokenizer)
    import nltk
    from nltk import sent_tokenize

    nltk.data.path.append(CACHE_DIR)

    # Prepare triplet extraction model
    from complai.tasks.self_check_consistency.compact_ie import (
        CompactFactsOpenInformationExtraction,
    )

    compact_ie = CompactFactsOpenInformationExtraction()

    # Get model instance
    model = get_model()

    # Prepare in-context examples
    alternative_statement_examples: list[ChatMessage] = []
    for example in EXAMPLES:
        alternative_statement_examples.extend(
            [
                ChatMessageUser(
                    content=ALTERNATIVE_STATEMENT_TEMPLATE.format(
                        topic=example.topic,
                        description=example.description,
                        subject=example.subject,
                        predicate=example.predicate,
                    )
                ),
                ChatMessageAssistant(content=example.example_completion),
            ]
        )

    async def _generate_alternative_statement(
        topic: str, subject: str, predicate: str, prefix: str | None = None
    ) -> str:
        """Generates an alternative statement for a given subject, predicate, and topic."""
        if prefix is None or prefix == "":
            prefix = "There is no preceding description. "

        # Prepare messages
        messages: list[ChatMessage] = alternative_statement_examples + [
            ChatMessageUser(
                content=ALTERNATIVE_STATEMENT_TEMPLATE.format(
                    topic=topic,
                    description=prefix,
                    subject=subject,
                    predicate=predicate,
                )
            )
        ]

        # Generate alternative statement
        output = await model.generate(messages)
        completion = output.completion

        # Try to take the first sentence from the completion since it likely
        # contains the relevant statement.
        if completion == "":
            return ""
        else:
            try:
                return sent_tokenize(completion)[0]
            except Exception:
                return completion

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get sentences from original completion
        completion = state.output.completion
        sentences = sent_tokenize(completion)

        prefix = ""
        alternative_sentences = []
        triplets = []
        for sentence in sentences:
            # Extract triplets from sentence
            extracted_triplets = compact_ie.process_sentences([sentence])

            # Choose first triplet
            if len(extracted_triplets) > 0:
                extracted_triplet = extracted_triplets[0]
                triplet = (
                    extracted_triplet["subject"],
                    extracted_triplet["relation"],
                    extracted_triplet["object"],
                )
                triplets.append(triplet)
            # If no triplet is found, remove the sentence and continue.
            # Ensures `sentences` and `alternative_sentences` will be aligned
            # for the scorer.
            else:
                sentences.remove(sentence)
                continue

            # Generate alternative statement
            alternative_sentence = await _generate_alternative_statement(
                subject=triplet[0],
                predicate=triplet[1],
                topic=state.metadata["topic"],
                prefix=prefix,
            )

            alternative_sentences.append(alternative_sentence)
            prefix += sentence

        state.metadata["sentences"] = sentences
        state.metadata["alternative_sentences"] = alternative_sentences
        state.metadata["triplets"] = triplets

        return state

    return solve


@metric
def percentage() -> MetricProtocol:
    def metric(scores: list[SampleScore]) -> float:
        # Flatten is_contradiction lists from all samples
        all_is_contradictions = [
            is_contradiction
            for score in scores
            if score.score.metadata is not None
            for is_contradiction in score.score.metadata["all_is_contradiction"]
        ]
        if not all_is_contradictions:
            return 1.0

        return 1 - sum(all_is_contradictions) / len(all_is_contradictions)

    return metric


@metric
def stderr() -> MetricProtocol:
    def metric(scores: list[SampleScore]) -> float:
        # Flatten is_contradiction lists from all samples
        all_is_contradictions = [
            is_contradiction
            for score in scores
            if score.score.metadata is not None
            for is_contradiction in score.score.metadata["all_is_contradiction"]
        ]

        n = len(all_is_contradictions)

        # standard deviation is calculated by dividing by n-ddof so ensure
        # that we won't divide by zero
        if (n - 1) < 1:
            return 0

        # Calculate the sample standard deviation
        sample_std = np.std(all_is_contradictions, ddof=1, dtype=float)

        # Calculate the standard error of the mean
        standard_error = sample_std / np.sqrt(n)

        return standard_error

    return metric


@scorer(metrics=[percentage()], name="Non-Contradictory")
def self_check_consistency_scorer(
    argumentation_model_name: str, judge_model_name: str
) -> Scorer:
    argumentation_model = get_model(argumentation_model_name)
    judge_model = get_model(judge_model_name)

    async def score(state: TaskState, target: Target) -> Score:
        topic = state.metadata["topic"]
        sentences = state.metadata["sentences"]
        alternative_sentences = state.metadata["alternative_sentences"]

        history: list[tuple[str, str]] = []
        all_is_contradiction = []
        prefix = ""
        for sentence, alternative_sentence in zip(sentences, alternative_sentences):
            try:
                # Build prompt for the argumentation model
                prompt = ARGUMENTATION_TEMPLATE.format(
                    topic=topic,
                    prefix=prefix,
                    sentence=sentence,
                    alternative_sentence=alternative_sentence,
                )

                # Ask the argumentation model if the statements are contradictory
                explanation = (
                    await _ask_model_with_history(
                        model=argumentation_model,
                        history=history,
                        prompt=prompt,
                        num_answers=1,
                    )
                )[0]
                history.append((prompt, explanation))

                # Ask the judge model if the statements are contradictory based on
                # the explanation
                if sentence == alternative_sentence:
                    is_contradiction = False
                    continue

                outputs = await _ask_model_with_history(
                    model=judge_model,
                    history=history,
                    prompt=JUDGE_PROMPT,
                    num_answers=10,
                )

                num_contradiction = 0
                num_total = 0
                for conclusion in outputs:
                    yes = re.match(r"\byes\b", conclusion.lower())
                    no = re.match(r"\bno\b", conclusion.lower())
                    if yes and not no:
                        num_contradiction += 1
                    num_total += 1

                # If more than half of the answers suggest a contradiction, we consider the
                # statements contradictory.
                is_contradiction = num_contradiction > num_total / 2

            finally:
                all_is_contradiction.append(is_contradiction)
                prefix += sentence

        passing_percentage = (
            1 - np.mean(all_is_contradiction, dtype=float)
            if all_is_contradiction
            else 1.0
        )
        traces = [
            prompt + "\n\n" + explanation + "\n\n" + str(judgement)
            for (prompt, explanation), judgement in zip(history, all_is_contradiction)
        ]

        return Score(
            value=passing_percentage,
            metadata={
                "all_is_contradiction": all_is_contradiction,
                "traces": traces,
                "passing_percentage": passing_percentage,
            },
        )

    return score


@task(technical_requirement="Capabilities, Performance, and Limitations")
def self_check_consistency(
    argumentation_model: str = "openai/gpt-3.5-turbo",
    judge_model: str = "openai/gpt-3.5-turbo",
) -> Task:
    ensure_nltk_data(NLTK_TOKENIZER_PATH, CACHE_DIR)

    return Task(
        dataset=self_check_consistency_dataset(),
        solver=[generate(), generate_alternative_statements()],
        scorer=self_check_consistency_scorer(argumentation_model, judge_model),
    )
