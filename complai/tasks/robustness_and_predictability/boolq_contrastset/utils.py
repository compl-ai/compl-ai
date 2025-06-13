import random
from typing import Optional

import datasets


def _process_example(item: dict) -> Optional[dict]:
    paragraph = item["paragraph"]
    original_question = item["question"]
    original_answer = item["answer"]

    if not all([paragraph, original_question, original_answer is not None]):
        return None

    original_question = original_question.strip().capitalize()
    if not original_question.endswith("?"):
        original_question += "?"

    perturbed_questions = item.get("perturbed_questions", [])
    assert all(isinstance(item, dict) for item in perturbed_questions)
    contrasting_pairs = [
        item
        for item in perturbed_questions
        if item.get("perturbed_q")
        and item.get("answer", None) is not None
        and item.get("answer") != original_answer
    ]
    if not contrasting_pairs:
        return None

    chosen_contrast = random.choice(contrasting_pairs)

    contrast_question = chosen_contrast["perturbed_q"].strip().capitalize()
    if not contrast_question.endswith("?"):
        contrast_question += "?"
    contrast_answer = chosen_contrast["answer"]

    input_str = (
        f"{paragraph}\\n\\n"
        f"Question: {original_question}\\n"
        f"Answer: {original_answer}\\n\\n"
        f"Question: {contrast_question}\\n"
        f"Answer:"
    )

    label_idx = 1 if contrast_answer == "TRUE" else 0

    return {"input": input_str, "label_idx": label_idx, "paragraph": paragraph}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    processed_dataset = dataset.map(
        _process_example, remove_columns=dataset.column_names
    )

    return processed_dataset
