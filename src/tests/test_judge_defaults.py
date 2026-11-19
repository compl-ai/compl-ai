import importlib
import inspect
from typing import Any

import pytest
from inspect_ai import Task

from complai.tasks.hle.hle import hle
from complai.tasks.hle.hle import hle_scorer


# the package re-exports the task function under the module's name
mask_module = importlib.import_module("complai.tasks.mask.mask")


# o3-mini (the original hle grader and mask numeric judge) was retired by OpenAI in
# October 2026 and now returns 404 "deprecated"; both judges are now gpt-6-luna.


def test_mask_defaults_to_luna_judges_with_room_for_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        mask_module, "inspect_mask", lambda **kwargs: captured.update(kwargs) or Task()
    )

    mask_module.mask()

    # Reasoning judges need more than inspect_evals' default 500-token judge budget.
    assert captured == {
        "binary_judge_model": "openai/gpt-6-luna",
        "numeric_judge_model": "openai/gpt-6-luna",
        "judge_max_tokens": 8000,
    }


def test_mask_judge_budget_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        mask_module, "inspect_mask", lambda **kwargs: captured.update(kwargs) or Task()
    )

    mask_module.mask(judge_max_tokens=500)

    assert captured["judge_max_tokens"] == 500


def test_hle_grader_defaults_to_luna() -> None:
    assert (
        inspect.signature(hle).parameters["grader_model"].default == "openai/gpt-6-luna"
    )
    assert (
        inspect.signature(hle_scorer).parameters["model"].default == "openai/gpt-6-luna"
    )
