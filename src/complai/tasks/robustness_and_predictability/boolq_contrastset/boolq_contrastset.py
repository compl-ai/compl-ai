import random
from typing import Any

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict[str, Any]) -> dict[str, Any] | None:
        paragraph: str = doc["paragraph"]
        original_question: str = doc["question"]
        original_answer: str = doc["answer"]

        original_question = original_question.strip().capitalize()
        if not original_question.endswith("?"):
            original_question += "?"

        perturbed_questions: list[dict[str, Any]] = doc.get("perturbed_questions", [])
        contrasting_pairs = [
            item
            for item in perturbed_questions
            if item.get("perturbed_q") is not None
            and item.get("answer") is not None
            and item["answer"] != original_answer
        ]
        if len(contrasting_pairs) == 0:
            return None

        chosen_contrast = random.choice(contrasting_pairs)

        contrast_question = chosen_contrast["perturbed_q"].strip().capitalize()
        if not contrast_question.endswith("?"):
            contrast_question += "?"
        contrast_answer = chosen_contrast["answer"]

        input_str = (
            f"{paragraph}\n"
            f"Question: {original_question}\n"
            f"Answer: {original_answer}\n"
            f"Question: {contrast_question}\n"
            f"Answer:"
        )

        return doc | {"input": input_str, "contrast_answer": contrast_answer}

    return dataset.map(_process_doc)


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    if len(results) == 0:
        return {"is_correct": False, "is_invalid": True}

    answer = results[0].strip().upper()
    expected_answer = doc["contrast_answer"].upper()

    is_correct = answer == expected_answer
    is_invalid = answer not in ["TRUE", "FALSE"]

    return {"is_correct": is_correct, "is_invalid": is_invalid}
