import math
import warnings

from inspect_ai.scorer import SampleScore
from inspect_ai.scorer import Score

from complai.tasks.boolq_contrast.boolq_contrast import accuracy
from complai.tasks.boolq_contrast.boolq_contrast import grouped_accuracies
from complai.tasks.boolq_contrast.boolq_contrast import stderr


def sample_score(question_id: int, correct: bool) -> SampleScore:
    """Build a scored sample as the metrics actually receive it.

    The epoch reducer maps "C"/"I" to 1.0/0 before metrics run, even for a
    single epoch, so a test that feeds raw "C"/"I" strings would exercise a
    path production never takes (`Score(value="I").as_bool()` is True).
    """
    return SampleScore(
        score=Score(value=1.0 if correct else 0), sample_metadata={"id": question_id}
    )


class TestGroupedAccuracies:
    """Test the per-question aggregation both metrics are built on."""

    def test_groups_by_question_id(self) -> None:
        """Each original question contributes one accuracy, whatever its size."""
        scores = (
            [sample_score(0, True)] * 3
            + [sample_score(1, True), sample_score(1, False)]
            + [sample_score(2, False)] * 3
        )
        assert sorted(grouped_accuracies(scores)) == [0.0, 0.5, 1.0]


class TestAccuracy:
    """Test that accuracy() averages questions, not samples."""

    def test_is_mean_of_question_accuracies(self) -> None:
        """A large correct question does not outweigh a small wrong one."""
        scores = [sample_score(0, True)] * 5 + [sample_score(1, False)]
        # Pooled over samples this would be 5/6; over questions it is 1/2.
        assert accuracy()(scores) == 0.5


class TestStderr:
    """Test the standard error of the quantity accuracy() reports."""

    def test_matches_the_closed_form(self) -> None:
        """Accuracies [1.0, 0.5, 0.0] have sd 0.5, so the SE is 0.5/sqrt(3)."""
        scores = (
            [sample_score(0, True)] * 3
            + [sample_score(1, True), sample_score(1, False)]
            + [sample_score(2, False)] * 3
        )
        assert stderr()(scores) == 0.5 / math.sqrt(3)

    def test_is_zero_when_every_question_agrees(self) -> None:
        """No spread between questions is a real zero, not a missing value."""
        scores = [sample_score(i, True) for i in range(3)]
        assert stderr()(scores) == 0.0

    def test_is_nan_for_a_single_question(self) -> None:
        """One question supports no spread estimate, so report nan, not 0.0.

        Reachable with --limit 1. Returning a number here would state a
        precision the run cannot support.
        """
        scores = [sample_score(0, True), sample_score(0, False)]
        assert len(grouped_accuracies(scores)) == 1
        assert math.isnan(stderr()(scores))

    def test_is_nan_for_no_questions(self) -> None:
        """An empty run has nothing to measure."""
        assert math.isnan(stderr()([]))

    def test_degenerate_input_is_handled_here_not_by_scipy(self) -> None:
        """The nan is this function's contract, not scipy's current behaviour.

        `sem([x], ddof=1)` also returns nan, but only after dividing by a zero
        degrees-of-freedom behind a SmallSampleWarning — a path whose result has
        changed across scipy versions. Asserting the warning is absent is what
        distinguishes the explicit guard from relying on that behaviour.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert math.isnan(stderr()([sample_score(0, True)]))
            assert isinstance(stderr()([sample_score(0, True)]), float)
