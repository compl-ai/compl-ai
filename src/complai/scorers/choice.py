"""Multiple-choice scorer that also accepts '$'-wrapped answer letters.

Inspect's multiple-choice prompts ask for 'ANSWER: $LETTER'. Some models copy the
dollar sign literally ('ANSWER: $B' or LaTeX-style 'ANSWER: $B$'). Inspect's answer
parser rejects 'ANSWER: $B' (since 0.3.287 it accepts the paired '$B$'; older versions
rejected both), so a correct letter is scored as wrong. When the solver marked no
choice, this scorer re-parses the completion with those dollar signs removed and then
defers to Inspect's ``choice()``. No choice is marked when re-scoring a log, because
the solver's markings are not stored, so live and re-scored results agree.

It is registered as ``choice`` so score keys (and ``complai predict``'s task-scorer
mapping) are unchanged. All complai multiple-choice tasks use this one scorer, so that
re-scoring a log by its recorded scorer name (``complai/choice`` plus its arguments)
rebuilds the scorer the task used.
"""

import re
from types import SimpleNamespace

from inspect_ai._util.answer import answer_index
from inspect_ai.scorer import accuracy
from inspect_ai.scorer import choice as inspect_choice
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import stderr
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_ai.solver._multiple_choice import parse_answers
from inspect_ai.solver._multiple_choice import set_choices_based_on_generated_response


# 'ANSWER: $B$' / 'ANSWER:$ B' / 'ANSWER: $A,C$' -> 'ANSWER: B' / 'ANSWER: A,C'.
# Only spaces and tabs are consumed around the dollars, never a line break.
DOLLAR_ANSWER = re.compile(
    r"(?i)(ANSWER[ \t]*:)[ \t]*\$+[ \t]*([A-Za-z\d ,]*[A-Za-z\d])[ \t]*\$*"
)
# A reply that is only the option letter ("C", "(C)", "**C**", "C.") or the
# letter followed by the option text ("C) 42 N"), on a single line.
BARE_LETTER = re.compile(r"^\(?\**([A-Z])\**\)?\.?$")
LEADING_LETTER = re.compile(r"^\(?\**([A-Z])\**\)\s")


def normalize_answer_line(text: str) -> str:
    """Remove template dollar signs around the answer letters of 'ANSWER:' lines."""
    return DOLLAR_ANSWER.sub(r"\1 \2", text)


def parse_bare_letter(completion: str, num_choices: int) -> str | None:
    """Read a single-line reply that is just the option letter.

    Multi-line replies are not read, so a reply that lists the options is not
    taken as an answer.
    """
    text = completion.strip()
    if "\n" in text:
        return None
    match = BARE_LETTER.match(text) or LEADING_LETTER.match(text)
    if match is None:
        return None
    letter = match.group(1)
    return letter if answer_index(letter) < num_choices else None


@scorer(metrics=[accuracy(), stderr()], name="choice")
def choice(multiple_correct: bool = False, bare_letter: bool = False) -> Scorer:
    """Inspect's ``choice()`` scorer with tolerant parsing of the 'ANSWER:' line.

    Args:
        multiple_correct: Whether several letters may be answered (as passed to the
            task's ``multiple_choice`` solver).
        bare_letter: Also accept a single-line reply that is just the option letter,
            for prompts that ask for the letter directly (mmmu_pro).
    """
    base = inspect_choice()

    async def score(state: TaskState, target: Target) -> Score | None:
        if not any(c.correct for c in state.choices):
            completion = state.output.completion or ""
            view = SimpleNamespace(
                output=SimpleNamespace(completion=normalize_answer_line(completion)),
                choices=state.choices,
            )
            answers = parse_answers(view, multiple_correct)  # type: ignore[arg-type]
            if not answers and bare_letter:
                letter = parse_bare_letter(completion, len(state.choices))
                answers = {letter} if letter else set()
            set_choices_based_on_generated_response(state, answers)
        return await base(state, target)

    return score
