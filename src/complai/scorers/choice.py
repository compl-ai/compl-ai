"""Multiple-choice scorer that also accepts '$'-wrapped answer letters.

Inspect's multiple-choice prompts ask for 'ANSWER: $LETTER'. Some models copy the
dollar sign literally ('ANSWER: $B' or LaTeX-style 'ANSWER: $B$'). Inspect's answer
parser rejects 'ANSWER: $B' (since 0.3.287 it accepts the paired '$B$'; older versions
rejected both), so a correct letter is scored as wrong. This scorer re-parses
the completion with those dollar signs removed and then defers to Inspect's
``choice()``; it is registered as ``choice`` so score keys (and ``complai predict``'s
task-scorer mapping) are unchanged. Because it parses the completion itself, it also
works when re-scoring existing logs, where the solver's choice markings are not stored.
"""

import re
from types import SimpleNamespace

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


# 'ANSWER: $B$' / 'ANSWER:$ B' / 'ANSWER: $A,C$' -> 'ANSWER: B' / 'ANSWER: A,C'
DOLLAR_ANSWER = re.compile(r"(?i)(ANSWER\s*:)\s*\$+\s*([A-Za-z\d ,]*[A-Za-z\d])\s*\$*")


def normalize_answer_line(text: str) -> str:
    """Remove template dollar signs around the answer letters of 'ANSWER:' lines."""
    return DOLLAR_ANSWER.sub(r"\1 \2", text)


@scorer(metrics=[accuracy(), stderr()], name="choice")
def choice(multiple_correct: bool = False) -> Scorer:
    """Inspect's ``choice()`` scorer with tolerant parsing of the 'ANSWER:' line.

    Args:
        multiple_correct: Whether several letters may be answered (as passed to the
            task's ``multiple_choice`` solver).
    """
    base = inspect_choice()

    async def score(state: TaskState, target: Target) -> Score | None:
        view = SimpleNamespace(
            output=SimpleNamespace(
                completion=normalize_answer_line(state.output.completion)
            ),
            choices=state.choices,
        )
        answers = parse_answers(view, multiple_correct)  # type: ignore[arg-type]
        set_choices_based_on_generated_response(state, answers)
        return await base(state, target)

    return score
