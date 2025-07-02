from __future__ import annotations

import re
import string
from typing import Any

import datasets
import numpy as np
from scipy.special import softmax

from complai.utils.ece import compute_ece


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
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

        return doc | {
            "input": input_str,
            "choices": letters_used,
            "target": target,
            "label_idx": label_idx,
        }

    return dataset.map(_process_doc)


def process_results(
    doc: dict[str, Any], results: list[tuple[float, bool]]
) -> dict[str, Any]:
    logprobs, _ = zip(*results)
    probs = softmax(logprobs)
    confidence = np.max(probs)
    is_correct = np.argmax(probs) == doc["label_idx"]

    return {"calibration": (confidence, is_correct), "acc": is_correct}


def calibration_score(items: list[tuple[float, bool]]) -> float:
    prediction_confidence, is_correct = zip(*items)

    ece = compute_ece(
        prediction_confidence=list(prediction_confidence), is_correct=list(is_correct)
    )

    return 1 - ece
