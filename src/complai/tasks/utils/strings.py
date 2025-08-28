import re
import string
from enum import Enum

from Levenshtein import distance as levenshtein_distance


class OptionPosition(str, Enum):
    START = "start"
    """Denotes that the option is at the beginning of the completion"""
    END = "end"
    """Denotes that the option is at the end of the completion"""
    UNKNOWN = "unknown"
    """Denotes that the option position is unknown"""


def remove_punctuation(s: str) -> str:
    """Removes punctuation from a string.

    Args:
        s: Input string.

    Returns:
        The input string with all punctuation removed.
    """
    return s.translate(str.maketrans("", "", string.punctuation))


def extract_alphabetic_option(completion: str) -> tuple[str, OptionPosition]:
    """Extracts an alphabetic option from an LLM response (i.e., 'A', 'B', 'C', ...).

    By default, the LLM is instructed to reply in the form of:

    A

    However, the instructions might not always be followed and variations
    of the answer can be returned, including:

    <space>A

    # Reasoning
    I think ... Therefore, the correct answer is A

    as well as potentially not even including the option

    # No option returned
    I don't know what I'm doing, just generating a random answer...

    Args:
        completion: A string containing LLM response.

    Returns:
        Value of the option or empty string if no option is found.
    """
    completion = remove_punctuation(completion.strip())
    if len(completion) == 0:
        return "", OptionPosition.UNKNOWN

    if completion[0] in string.ascii_uppercase and (
        len(completion) == 1 or completion[1].isspace()
    ):
        # A
        # <space>A
        # A<space>
        # A.
        # A completion
        return completion[0], OptionPosition.START
    elif (
        len(completion) > 1
        and completion[-1] in string.ascii_uppercase
        and completion[-2].isspace()
    ):
        # ... option A
        # ... option A.
        # ... option A<space>
        return completion[-1], OptionPosition.END

    return "", OptionPosition.UNKNOWN


def normalize_text(raw_text: str) -> str:
    """
    Normalize a string by lowercasing, removing punctuation, articles, underscores, and extra whitespace.

    This function is adapted from the official TriviaQA evaluation script:
    https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py

    The normalization steps are:
        1. Remove articles ("a", "an", "the").
        2. Remove punctuation (including some non-ASCII variants).
        3. Replace underscores with spaces.
        4. Convert to lowercase.
        5. Remove extra whitespace.

    Args:
        raw_text (str): The input string to normalize.

    Returns:
        str: The normalized string, with articles, punctuation, and extra whitespace removed,
             all lowercase, and underscores replaced by spaces.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text: str) -> str:
        return text.lower()

    def replace_underscore(text: str) -> str:
        return text.replace("_", " ")

    normalization_operations = [
        remove_articles,
        white_space_fix,
        handle_punc,
        lower,
        replace_underscore,
    ]
    normalized_text = raw_text
    for operation in normalization_operations:
        normalized_text = operation(normalized_text)

    return normalized_text.strip()


def is_substring_with_tolerance(a: str, b: str, tolerance: int = 2) -> float:
    """Checks if string "a" is a substring of string "b" using a Levenshtein
    distance tolerance.
    """
    # Ensure b is the shorter (or equal) string.
    if len(b) > len(a):
        a, b = b, a

    return any(
        levenshtein_distance(a[offset : offset + len(b)], b) <= tolerance
        for offset in range(len(a) - len(b) + 1)
    )


def has_approximate_match(targets: list[str], candidates: list[str]) -> bool:
    """Checks if there is an approximate match between ground truth and candidates."""
    for target in targets:
        for candidate in candidates:
            if is_substring_with_tolerance(
                candidate, target
            ) or is_substring_with_tolerance(target, candidate):
                return True

    return False
