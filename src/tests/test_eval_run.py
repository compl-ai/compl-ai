import importlib
import inspect
from pathlib import Path

import pytest
import typer
from inspect_ai.log import EvalSampleSummary
from inspect_ai.model import ModelUsage

from complai._cli.report import _error_text
from complai._cli.report import _may_be_truncated
from complai._cli.report import _missing
from complai._cli.status import apply_judge_fallback
from complai._cli.status import JUDGE_DEFAULTS
from complai._cli.status import judge_models
from complai._cli.status import OPENROUTER_JUDGES
from complai._cli.utils import default_subset_dir
from complai._cli.utils import get_complai_tasks
from complai._cli.utils import select_subset_items
from complai.utils.log_parser import SUBSET_METADATA_KEY
from complai.utils.log_parser import SUBSET_NAME_METADATA_KEY


def test_select_subset_items_defaults_to_every_subset_task() -> None:
    selected, metadata = select_subset_items([], [], None, {})

    assert len(selected) == 23
    assert sum(len(rows) for rows in selected.values()) == 2000
    assert metadata == {
        SUBSET_NAME_METADATA_KEY: default_subset_dir().name,
        SUBSET_METADATA_KEY: selected["bbq"][0]["subset_id"],
    }


def test_select_subset_items_filters_tasks_and_keeps_smoke_items() -> None:
    selected, _ = select_subset_items(["bbq", "hle"], ["hle"], 2, {})

    assert list(selected) == ["bbq"]
    assert len(selected["bbq"]) == 2


def test_select_subset_items_rejects_unknown_tasks_and_full_only_options() -> None:
    with pytest.raises(typer.BadParameter, match="notatask"):
        select_subset_items(["notatask"], [], None, {})
    with pytest.raises(typer.BadParameter, match="--limit"):
        select_subset_items([], [], None, {"--limit": "5", "--sample-id": None})


def test_judge_defaults_match_task_signatures() -> None:
    files = {info.name: Path(info.file) for info in get_complai_tasks()}
    for task, judges in JUDGE_DEFAULTS.items():
        module = importlib.import_module(
            f"complai.tasks.{files[task].parent.name}.{files[task].stem}"
        )
        parameters = inspect.signature(getattr(module, task)).parameters
        for arg, model in judges.items():
            assert parameters[arg].default == model, f"{task}:{arg}"
            assert model in OPENROUTER_JUDGES


def test_judge_fallback_uses_openrouter_only_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    task_args = {"mask": {"binary_judge_model": "anthropic/claude"}}

    switched = apply_judge_fallback(["mask", "bbq"], task_args)

    assert switched == ["mask:numeric_judge_model"]
    assert task_args["mask"] == {
        "binary_judge_model": "anthropic/claude",
        "numeric_judge_model": "openrouter/openai/o3-mini",
    }
    assert judge_models(["mask"], task_args) == {
        "anthropic/claude": ["mask:binary_judge_model"],
        "openrouter/openai/o3-mini": ["mask:numeric_judge_model"],
    }

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    assert apply_judge_fallback(["cab"], {}) == []


@pytest.mark.parametrize(
    ("value", "missing"),
    [
        (None, True),
        (float("nan"), True),
        ({"a": 1.0, "b": float("nan")}, True),
        ([1.0, None], True),
        ("C", False),
        (0.0, False),
        ({"a": 1.0}, False),
    ],
)
def test_report_counts_missing_scores(value: object, missing: bool) -> None:
    assert _missing(value) is missing


def test_report_groups_repeated_errors() -> None:
    first = (
        "Traceback\nCancelledError('Cancelled via cancel scope 12d4a2420 by <Task a>')"
    )
    second = "CancelledError('Cancelled via cancel scope 1298fa2a0 by <Task b>')"

    assert _error_text(first) == _error_text(second)


def test_report_reads_full_samples_only_near_max_tokens() -> None:
    sample = EvalSampleSummary(
        id=1,
        epoch=1,
        input="q",
        target="a",
        model_usage={"m": ModelUsage(input_tokens=10, output_tokens=100)},
    )

    assert _may_be_truncated(sample, "m", 100)
    assert not _may_be_truncated(sample, "m", 101)
    assert _may_be_truncated(sample, "m", None)
