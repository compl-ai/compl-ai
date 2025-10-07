from typing import Any

from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import accuracy
from inspect_ai.scorer import CORRECT
from inspect_ai.scorer import INCORRECT
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import stderr
from inspect_ai.scorer import Target
from inspect_ai.solver import generate
from inspect_ai.solver import system_message
from inspect_ai.solver import TaskState


DATASET_PATH = "compl-ai/imdb_contrastset"

IMDB_SYSTEM_PROMPT = """
You are a helpful assistant. For each snippet of text, predict the Sentiment by selecting from the option 'Positive' or 'Negative'. The answer should be exactly 'Positive' or 'Negative'.
""".strip()


def record_to_sample(record: dict[str, Any]) -> Sample:
    original_text = record["Text_Original"]
    original_sentiment = record["Sentiment_Original"]
    contrast_text = record["Text_Contrast"]
    contrast_sentiment = record["Sentiment_Contrast"]

    messages: list[ChatMessage] = [
        ChatMessageUser(content=f"{original_text}\nSentiment: "),
        ChatMessageAssistant(content=original_sentiment),
        ChatMessageUser(content=f"{contrast_text}\nSentiment: "),
    ]

    return Sample(
        input=messages,
        target=contrast_sentiment,
        metadata={
            "original_text": original_text,
            "original_sentiment": original_sentiment,
        },
    )


def imdb_contrastset_dataset() -> Dataset:
    return hf_dataset(path=DATASET_PATH, split="train", sample_fields=record_to_sample)


def parse_sentiment(completion: str) -> str:
    completion = completion.strip().lower()

    if "positive" in completion and "negative" not in completion:
        return "positive"
    elif "negative" in completion and "positive" not in completion:
        return "negative"
    elif "neutral" in completion and "positive" in completion:
        # Models tend to conclude with a summary statement
        # Check the last 20 characters for the final sentiment
        completion_end = completion[-20:]
        if "positive" in completion_end and "negative" not in completion_end:
            return "positive"
        elif "negative" in completion_end and "positive" not in completion_end:
            return "negative"

    return "unknown"


@scorer(metrics=[accuracy(), stderr()])
def imdb_sentiment_scorer() -> Scorer:
    """Custom scorer for IMDB sentiment classification with robust parsing."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion

        predicted_sentiment = parse_sentiment(completion)
        expected_sentiment = target.text.lower()

        is_correct = predicted_sentiment == expected_sentiment
        is_valid = predicted_sentiment in ["positive", "negative"]

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            metadata={"predicted_sentiment": predicted_sentiment, "is_valid": is_valid},
        )

    return score


@task(technical_requirement="Robustness and Predictability")
def imdb_contrast() -> Task:
    """IMDB contrast sets task.

    IMDB contrast sets measure if the model can accurately deduce the
    sentiment of a given text, when presented with a distraction: a seemingly
    similar text that has an opposite sentiment.
    """
    return Task(
        dataset=imdb_contrastset_dataset(),
        solver=[system_message(IMDB_SYSTEM_PROMPT), generate()],
        scorer=imdb_sentiment_scorer(),
    )
