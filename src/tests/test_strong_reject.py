import importlib
import json
from pathlib import Path

import polars as pl
import pytest

from complai._cli.status import check_gated_datasets
from complai._cli.status import GATED_DATASETS
from complai._cli.status import run_checks
from complai.tasks.strong_reject import jailbreaks
from complai.tasks.strong_reject.jailbreaks import REFERENCE_PROMPT_SOURCE
from complai.tasks.strong_reject.jailbreaks import REFERENCE_PROMPTS_DATASET
from complai.tasks.strong_reject.jailbreaks import REFERENCE_PROMPTS_REVISION
from complai.utils.log_parser import content_hash


# The package re-exports the task function under the module's name
strong_reject = importlib.import_module("complai.tasks.strong_reject.strong_reject")

RECORDS = pl.DataFrame(
    {"category": ["c1", "c2"], "source": ["s1", "s2"], "forbidden_prompt": ["p1", "p2"]}
)


@pytest.fixture
def fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Isolate the local cache, base CSV, reference download and generation."""
    hub = tmp_path / "hub"
    hub.mkdir()
    calls: dict[str, list] = {"downloads": [], "generated": []}

    def fake_download(repo_id: str, filename: str, **kwargs: str) -> str:
        calls["downloads"].append((repo_id, filename, kwargs))
        path = hub / filename
        if not path.exists():
            raise OSError("gated")
        return str(path)

    def fake_generate(
        method: str, prompt: str, cache: dict[str, str] | None
    ) -> tuple[str, bool]:
        if cache is not None and prompt in cache:
            return cache[prompt], False
        calls["generated"].append((method, prompt))
        return f"generated {method} {prompt}", True

    monkeypatch.setattr(jailbreaks, "STRONG_REJECT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(jailbreaks, "hf_hub_download", fake_download)
    monkeypatch.setattr(strong_reject, "get_jailbreak_prompt", fake_generate)
    monkeypatch.setattr(strong_reject.pl, "read_csv", lambda url: RECORDS)
    calls["hub"] = [hub]
    return calls


def _publish(hub: Path, method: str, prompts: dict[str, str]) -> None:
    (hub / f"{method}_small.json").write_text(json.dumps(prompts))


def _inputs_and_sources(dataset: object) -> list[tuple[str, str]]:
    return [
        (sample.input, sample.metadata["prompt_source"])  # type: ignore[index]
        for sample in dataset  # type: ignore[attr-defined]
    ]


def test_reference_prompts_are_used_and_pinned(fake_env: dict[str, list]) -> None:
    hub = fake_env["hub"][0]
    _publish(hub, "pair", {"p1": "ref pair p1", "p2": "ref pair p2"})
    _publish(hub, "pap_misrepresentation", {"p1": "ref pap p1", "p2": "ref pap p2"})

    dataset = strong_reject.strong_reject_dataset(["pair", "pap_misrepresentation"])

    assert _inputs_and_sources(dataset) == [
        ("ref pair p1", REFERENCE_PROMPT_SOURCE),
        ("ref pair p2", REFERENCE_PROMPT_SOURCE),
        ("ref pap p1", REFERENCE_PROMPT_SOURCE),
        ("ref pap p2", REFERENCE_PROMPT_SOURCE),
    ]
    assert fake_env["generated"] == []
    assert not jailbreaks.STRONG_REJECT_CACHE_DIR.exists()
    assert {
        (repo, kwargs["repo_type"], kwargs["revision"])
        for repo, _, kwargs in fake_env["downloads"]
    } == {(REFERENCE_PROMPTS_DATASET, "dataset", REFERENCE_PROMPTS_REVISION)}


def test_missing_access_falls_back_to_local_cache_then_generation(
    fake_env: dict[str, list], caplog: pytest.LogCaptureFixture
) -> None:
    hub = fake_env["hub"][0]
    _publish(hub, "pair", {"p1": "ref pair p1", "p2": "ref pair p2"})
    jailbreaks.save_method_cache("pair", {"p1": "cached pair p1"})

    # One unavailable method makes the whole reference set unusable
    dataset = strong_reject.strong_reject_dataset(["pair", "pap_misrepresentation"])

    assert _inputs_and_sources(dataset) == [
        ("cached pair p1", "local_cache"),
        ("generated pair p2", "generated"),
        ("generated pap_misrepresentation p1", "generated"),
        ("generated pap_misrepresentation p2", "generated"),
    ]
    assert "not comparable with reference runs" in caplog.text
    assert jailbreaks.load_method_cache("pair") == {
        "p1": "cached pair p1",
        "p2": "generated pair p2",
    }


def test_full_dataset_does_not_use_reference_prompts(fake_env: dict[str, list]) -> None:
    _publish(fake_env["hub"][0], "pair", {"p1": "ref pair p1", "p2": "ref pair p2"})

    dataset = strong_reject.strong_reject_dataset(["pair"], full=True)

    assert fake_env["downloads"] == []
    assert [source for _, source in _inputs_and_sources(dataset)] == ["generated"] * 2


def test_prompt_source_does_not_change_item_identity() -> None:
    metadata = {"category": "c", "source": "s", "jailbreak_method": "pair"}
    reference = content_hash(
        "ref prompt",
        "",
        {**metadata, "prompt_source": REFERENCE_PROMPT_SOURCE},
        task="strong_reject",
        sample_id=61,
    )

    assert reference == content_hash(
        "generated prompt", "", metadata, task="strong_reject", sample_id=61
    )


@pytest.fixture
def status_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Record which strong_reject checks run_checks performs."""
    from complai._cli import status

    calls: dict[str, list] = {"gated": [], "prompts": []}

    def gated(tasks: list[str]) -> list:
        calls["gated"].append(tasks)
        return []

    def prompts(args: dict) -> str:
        calls["prompts"].append(args)
        return "prompt check"

    monkeypatch.setattr(status, "check_target_model", lambda *args: [])
    monkeypatch.setattr(status, "check_judges", lambda *args: [])
    monkeypatch.setattr(status, "check_gated_datasets", gated)
    monkeypatch.setattr(status, "check_strong_reject_prompts", prompts)
    return calls


def test_strong_reject_status_checks_reference_access_for_small_dataset(
    status_calls: dict[str, list],
) -> None:
    run_checks("m", None, {}, ["hle", "strong_reject"], {})

    assert GATED_DATASETS["strong_reject"] == REFERENCE_PROMPTS_DATASET
    assert status_calls == {"gated": [["hle", "strong_reject"]], "prompts": []}


def test_strong_reject_status_checks_local_prompts_for_full_dataset(
    status_calls: dict[str, list],
) -> None:
    args = {"strong_reject": {"full": True}}
    results = run_checks("m", None, {}, ["hle", "strong_reject"], args)

    assert status_calls == {"gated": [["hle"]], "prompts": [{"full": True}]}
    assert results[-1] == "prompt check"


def test_gated_dataset_check_reports_missing_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    def no_access(repo_id: str, repo_type: str) -> None:
        raise OSError("gated")

    monkeypatch.setattr(huggingface_hub, "auth_check", no_access)

    [result] = check_gated_datasets(["strong_reject"])

    assert result.status == "fail"
    assert REFERENCE_PROMPTS_DATASET in result.detail
