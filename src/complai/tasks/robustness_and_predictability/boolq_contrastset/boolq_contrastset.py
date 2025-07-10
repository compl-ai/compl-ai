import random
from typing import Optional

import datasets

from complai.utils.metrics import sum_aggregation  # noqa: F401


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> Optional[dict]:
        paragraph: str = doc["paragraph"]
        original_question: str = doc["question"]
        original_answer: str = doc["answer"]

        original_question = original_question.strip().capitalize()
        if not original_question.endswith("?"):
            original_question += "?"

        perturbed_questions: list[dict] = doc.get("perturbed_questions", [])
        contrasting_pairs = [
            item
            for item in perturbed_questions
            if item.get("perturbed_q")
            and item.get("answer")
            and item["answer"] != original_answer
        ]
        if not contrasting_pairs:
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


def process_results(doc: dict, results: list[str]) -> dict:
    if not results:
        return {"is_correct": False, "num_invalid": True}

    answer = results[0].strip().upper()
    expected_answer = doc["contrast_answer"].upper()

    is_correct = answer == expected_answer
    is_invalid = answer not in ["TRUE", "FALSE"]

    return {"is_correct": is_correct, "num_invalid": is_invalid}
