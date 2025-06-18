from typing import Any


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, bool]:
    normalized_answer = results[0].strip().lower() if results else ""
    is_yes = "yes" in normalized_answer and "no" not in normalized_answer

    return {"is_yes": is_yes}
