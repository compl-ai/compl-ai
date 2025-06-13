from __future__ import annotations

import re
import string
from typing import Any

import datasets
import numpy as np
from scipy.special import softmax

from complai.tasks.interpretability.bigbench_calibration.ece import compute_ece


def _process_row(doc: dict) -> dict:
    letters = iter(string.ascii_uppercase)
    letters_used = []
    correctly_ordered_choice_texts = []

    def replacement(match: re.Match) -> str:
        letter = next(letters)
        letters_used.append(letter)
        choice_text = match.string[match.end() :].split("\n")[0].strip()
        correctly_ordered_choice_texts.append(choice_text)
        return f"\n{letter}."

    input_str = re.sub(r"\n  choice:", replacement, doc["inputs"])
    label_idx = correctly_ordered_choice_texts.index(doc["targets"][0])
    target = letters_used[label_idx]

    return {
        "input": input_str,
        "choices": letters_used,
        "target": target,
        "label_idx": label_idx,
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_row, remove_columns=dataset.column_names)


def process_results(
    doc: dict[str, Any], results: list[tuple[float, bool]]
) -> dict[str, Any]:
    logprobs, _ = zip(*results)
    probs = softmax(logprobs)
    confidence = np.max(probs)
    is_correct = np.argmax(probs) == doc["label_idx"]

    return {"acc": is_correct, "ece": (confidence, is_correct)}


def ece(items: list[tuple[float, bool]]) -> float:
    prediction_confidence, is_correct = zip(*items)

    return compute_ece(
        prediction_confidence=list(prediction_confidence), is_correct=list(is_correct)
    )
