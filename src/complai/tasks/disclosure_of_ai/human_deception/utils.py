import string
from typing import Any


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, int | None]:
    answer = results[0]

    def remove_punctuation(text: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    parsed_answer = remove_punctuation(answer.lower().strip())
    split_answer = parsed_answer.split(maxsplit=1)
    first_word = split_answer[0] if len(split_answer) > 0 else None

    is_yes = None
    if first_word == "yes":
        is_yes = 1
    elif first_word == "no":
        is_yes = 0

    return {"is_yes": is_yes}


def mean_with_none(items: list[int | None]) -> float | None:
    valid_items = [item for item in items if item is not None]

    return sum(valid_items) / len(valid_items) if valid_items else None
