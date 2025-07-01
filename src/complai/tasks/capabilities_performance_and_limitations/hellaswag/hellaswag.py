import re

import datasets

from complai.utils.strings import remove_punctuation


def preprocess_text(text: str) -> str:
    # Source: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py
    # MIT License
    # Copyright (c) 2020 EleutherAI
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy of
    # this software and associated documentation files (the "Software"), to deal in the
    # Software without restriction, including without limitation the rights to use,
    # copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
    # Software, and to permit persons to whom the Software is furnished to do so,
    # subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    # FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
    # AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    # WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(docs: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        context = f"{doc['activity_label']}: {doc['ctx_a']} {doc['ctx_b'].capitalize()}"
        endings = [preprocess_text(ending) for ending in doc["endings"]]
        choices = ["A", "B", "C", "D"]
        input_str = (
            f"{context}\n"
            + "\n".join(
                [f"{choice}. {answer}" for choice, answer in zip(choices, endings)]
            )
            + "\nAnswer:"
        )
        label = int(doc["label"])
        correct_choice = choices[label]

        return doc | {
            "input": input_str,
            "label": label,
            "correct_choice": correct_choice,
        }

    return docs.map(_process_doc, remove_columns=docs.column_names)


def process_results(doc: dict, results: list[str]) -> dict:
    if results[0] is None:
        return {"is_correct": None, "num_invalid": True}

    answer = remove_punctuation(results[0].strip().upper())
    expected_answer = doc["correct_choice"]

    is_correct = answer == expected_answer
    is_invalid = answer not in ["A", "B", "C", "D"]

    return {"is_correct": is_correct, "num_invalid": is_invalid}
