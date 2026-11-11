import asyncio

import pytest
from inspect_ai._util.registry import registry_unqualified_name
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from complai.scorers.choice import choice
from complai.scorers.choice import normalize_answer_line


def make_state(completion: str, choices: list[str]) -> TaskState:
    # as when re-scoring a log: no choice has been marked by the multiple_choice solver
    return TaskState(
        model="test/model",  # type: ignore[arg-type]
        sample_id=1,
        epoch=1,
        input="question",
        messages=[],
        choices=choices,
        output=ModelOutput.from_content("test/model", completion),
    )


def run(
    completion: str, target: str | list[str], multiple_correct: bool = False
) -> str:
    state = make_state(completion, ["w", "x", "y", "z"])
    score = asyncio.run(
        choice(multiple_correct=multiple_correct)(state, Target(target))
    )
    assert score is not None
    return str(score.value)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("ANSWER: $B$", "ANSWER: B"),
        ("ANSWER: $D", "ANSWER: D"),
        ("answer:$ C $.", "answer: C."),
        ("ANSWER: $A,C$", "ANSWER: A,C"),
        ("ANSWER: B", "ANSWER: B"),
        ("The price is $5, so ANSWER: C", "The price is $5, so ANSWER: C"),
    ],
)
def test_normalize_answer_line_strips_template_dollar_signs(
    line: str, expected: str
) -> None:
    assert normalize_answer_line(line) == expected


@pytest.mark.parametrize(
    "completion", ["reasoning...\n\nANSWER: $B$", "reasoning...\n\nANSWER: $B"]
)
def test_dollar_wrapped_answers_are_scored_correct(completion: str) -> None:
    # models copy the '$' from the prompt's 'ANSWER: $LETTER' template
    assert run(completion, "B") == "C"


def test_plain_answers_still_scored_as_before() -> None:
    assert run("ANSWER: B", "B") == "C"
    assert run("ANSWER: C", "B") == "I"


def test_unparsable_answer_is_incorrect() -> None:
    assert run("I think it is the second one.", "B") == "I"


def test_multiple_correct_answers() -> None:
    assert run("ANSWER: $A,C$", ["A", "C"], multiple_correct=True) == "C"


def test_scorer_keeps_the_choice_name_used_in_logs_and_predict() -> None:
    assert registry_unqualified_name(choice()) == "choice"


@pytest.mark.parametrize(
    "task_name",
    ["arc_challenge", "bbq", "gpqa_diamond", "include", "mmlu_pro", "truthfulqa"],
)
def test_multiple_choice_tasks_use_the_tolerant_scorer(task_name: str) -> None:
    from inspect_ai._util.registry import registry_info

    import complai._registry as tasks

    task = getattr(tasks, task_name)()
    assert [registry_info(s).name for s in task.scorer] == ["complai/choice"]


@pytest.mark.parametrize(
    "completion, expected",
    [("ANSWER: $B$", {"B"}), ("ANSWER: $B", {"B"}), ("ANSWER: B", {"B"}), ("B", set())],
)
def test_robustness_solver_parser_accepts_dollar_wrapped_answers(
    completion: str, expected: set[str]
) -> None:
    # mmlu_pro_robustness parses answers in its solver with its own copy of the parser
    from complai.tasks.mmlu_pro_robustness.mmlu_pro_robustness import parse_answers

    assert parse_answers(completion, 4) == expected
