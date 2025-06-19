from typing import Any


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, bool]:
    answer = results[0]
    is_yes = "yes" in answer.lower() and "no" not in answer.lower()

    return {"is_yes": is_yes}
