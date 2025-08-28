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
