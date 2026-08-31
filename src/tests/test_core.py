from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalConfig
from inspect_ai.log import EvalDataset
from inspect_ai.log import EvalLog
from inspect_ai.log import EvalResults
from inspect_ai.log import EvalSample
from inspect_ai.log import EvalSpec
from inspect_ai.log import write_eval_log
from inspect_ai.scorer import Score
from typer.testing import CliRunner

from complai._cli import app
from complai.core import apply_eval_subset
from complai.core import dispersion23_allocation
from complai.core import fit_2pl
from complai.core import load_scorers
from complai.core import minify
from complai.core import predict_scores
from complai.core import read_eval_subset
from complai.core import write_outputs
from complai.core.index import _content_hash
from complai.core.index import _logical_sample_id
from complai.core.index import index_logs
from complai.core.index import PARSER_VERSION


def test_fit_2pl_is_identified_and_marks_thin_items() -> None:
    values = np.asarray(
        [
            [0.0, 0.0, 1.0, np.nan],
            [0.0, 1.0, 1.0, np.nan],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, np.nan],
        ]
    )

    fitted = fit_2pl(values, iterations=20)

    assert np.mean(fitted.abilities) == pytest.approx(0.0, abs=1e-10)
    assert np.std(fitted.abilities) == pytest.approx(1.0, abs=1e-10)
    assert np.all((fitted.discriminations >= 0.05) & (fitted.discriminations <= 5.0))
    assert fitted.discriminations[3] == 1.0
    assert not fitted.slope_identified[3]


def test_fit_2pl_recovers_known_model() -> None:
    abilities = np.linspace(-2.0, 2.0, 41)
    abilities = (abilities - abilities.mean()) / abilities.std()
    discriminations = np.asarray([0.35, 0.55, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5])
    difficulties = np.asarray([-1.5, -1.0, -0.5, 0.0, 0.3, 0.7, 1.0, 1.4])
    intercepts = -discriminations * difficulties
    logits = abilities[:, None] * discriminations[None, :] + intercepts[None, :]
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    fitted = fit_2pl(probabilities, ridge=1e-8, slope_ridge=1e-8, iterations=30)

    assert fitted.converged
    np.testing.assert_allclose(fitted.abilities, abilities, atol=1e-5)
    np.testing.assert_allclose(fitted.discriminations, discriminations, atol=1e-5)
    np.testing.assert_allclose(fitted.intercepts, intercepts, atol=1e-5)


def test_fit_2pl_recovers_known_model_from_bernoulli_samples() -> None:
    rng = np.random.default_rng(7)
    abilities = np.linspace(-2.5, 2.5, 400)
    abilities = (abilities - abilities.mean()) / abilities.std()
    discriminations = rng.uniform(0.5, 2.0, 120)
    difficulties = rng.uniform(-1.5, 1.5, 120)
    intercepts = -discriminations * difficulties
    logits = abilities[:, None] * discriminations[None, :] + intercepts[None, :]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    responses = rng.binomial(1, probabilities).astype(float)

    fitted = fit_2pl(responses)
    fitted_logits = (
        fitted.abilities[:, None] * fitted.discriminations[None, :]
        + fitted.intercepts[None, :]
    )
    fitted_probabilities = 1.0 / (1.0 + np.exp(-fitted_logits))

    assert np.corrcoef(fitted.abilities, abilities)[0, 1] > 0.95
    assert np.corrcoef(fitted.discriminations, discriminations)[0, 1] > 0.85
    assert np.corrcoef(fitted.intercepts, intercepts)[0, 1] > 0.95
    assert np.mean(np.abs(fitted_probabilities - probabilities)) < 0.05


def test_dispersion23_allocation_respects_floor_capacity_and_budget() -> None:
    allocation = dispersion23_allocation(
        {"large": 30, "small": 4, "variable": 20},
        {"large": 0.1, "small": 0.3, "variable": 0.8},
        30,
        10,
    )

    assert sum(allocation.values()) == 30
    assert allocation["small"] == 4
    assert allocation["large"] >= 10
    assert allocation["variable"] >= 10


def test_minify_indexes_reuses_cache_and_writes_deterministic_outputs(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for model_index in range(3):
        _write_eval(
            logs / f"m{model_index}.eval",
            model=f"model-{model_index}",
            run_id=f"run-{model_index}",
            created=f"2026-01-0{model_index + 1}T00:00:00+00:00",
            values=["C" if (item + model_index) % 3 else "I" for item in range(12)],
        )

    first = minify([logs], {"toy": "choice"}, 5, seed=7, cache_dir=tmp_path / "cache")
    second = minify([logs], {"toy": "choice"}, 5, seed=7, cache_dir=tmp_path / "cache")

    assert first.inventory.summary.new == 3
    assert second.inventory.summary.cache_hits == 3
    assert first.artifact == second.artifact
    assert first.subset == second.subset
    assert len(first.subset) == 5
    artifact_path, subset_path = write_outputs(first, tmp_path / "output")
    assert json.loads(artifact_path.read_text())["budget"] == 5
    assert len(subset_path.read_text().splitlines()) == 5
    with pytest.raises(FileExistsError):
        write_outputs(first, tmp_path / "output")


def test_duplicate_eval_policies(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for model_index in range(3):
        _write_eval(
            logs / f"m{model_index}.eval",
            model=f"model-{model_index}",
            run_id=f"run-{model_index}",
            created=f"2026-03-0{model_index + 1}T00:00:00+00:00",
            values=[float((item + model_index) % 2) for item in range(12)],
        )
    _write_eval(
        logs / "m0-new.eval",
        model="model-0",
        run_id="run-new",
        created="2026-04-01T00:00:00+00:00",
        values=[1.0] * 12,
    )

    with pytest.raises(ValueError, match="Duplicate successful evaluations"):
        minify([logs], {"toy": "choice"}, 5, cache_dir=tmp_path / "cache")
    averaged = minify(
        [logs],
        {"toy": "choice"},
        5,
        duplicate_policy="mean",
        cache_dir=tmp_path / "cache",
    )
    latest = minify(
        [logs],
        {"toy": "choice"},
        5,
        duplicate_policy="latest",
        cache_dir=tmp_path / "cache",
    )
    assert averaged.artifact["artifact_id"] != latest.artifact["artifact_id"]


def test_only_latest_dataset_version_is_fitted(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for model_index in range(3):
        _write_eval(
            logs / f"old-{model_index}.eval",
            model=f"old-model-{model_index}",
            run_id=f"old-{model_index}",
            created=f"2026-01-0{model_index + 1}T00:00:00+00:00",
            values=[0.0] * 12,
            dataset="toy-v1",
        )
        _write_eval(
            logs / f"new-{model_index}.eval",
            model=f"new-model-{model_index}",
            run_id=f"new-{model_index}",
            created=f"2026-02-0{model_index + 1}T00:00:00+00:00",
            values=[1.0] * 12,
            dataset="toy-v2",
        )

    result = minify([logs], {"toy": "choice"}, 5, cache_dir=tmp_path / "cache")

    assert result.artifact["tasks"]["toy"]["capacity"] == 12
    assert {item["dataset"] for item in result.artifact["items"]} == {"toy-v2"}


def test_index_detects_changes_removals_and_forced_reindex(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for model_index in range(3):
        _write_eval(
            logs / f"m{model_index}.eval",
            model=f"model-{model_index}",
            run_id=f"run-{model_index}",
            created=f"2026-05-0{model_index + 1}T00:00:00+00:00",
            values=[float((item + model_index) % 2) for item in range(12)],
        )
    cache = tmp_path / "index.sqlite3"

    assert index_logs([logs], cache).summary.new == 3
    assert index_logs([logs], cache).summary.cache_hits == 3
    _write_eval(
        logs / "m0.eval",
        model="model-0",
        run_id="run-0-replaced",
        created="2026-06-01T00:00:00+00:00",
        values=[1.0] * 12,
    )
    assert index_logs([logs], cache).summary.changed == 1
    assert index_logs([logs], cache, reindex=True).summary.changed == 3
    (logs / "m2.eval").unlink()
    removed = index_logs([logs], cache).summary
    assert removed.removed == 1
    assert removed.discovered == 2


def test_index_uses_memory_bounded_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import complai.core.index as index_module

    log_path = tmp_path / "one.eval"
    _write_eval(
        log_path,
        model="model-0",
        run_id="run-0",
        created="2026-06-01T00:00:00+00:00",
        values=[1.0],
    )
    original = index_module.read_eval_log

    def header_only(path: str, *, header_only: bool = False) -> EvalLog:
        assert header_only, "indexing must not load full Inspect logs"
        return original(path, header_only=True)

    monkeypatch.setattr(index_module, "read_eval_log", header_only)

    indexed = index_logs([log_path], tmp_path / "index.sqlite3")

    assert indexed.sample_count() == 1
    assert len(list(indexed.iter_samples())) == 1
    assert indexed.files[0]["parser_version"] == PARSER_VERSION


def test_index_defers_unrequested_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import complai.core.index as index_module

    logs = tmp_path / "logs"
    logs.mkdir()
    _write_eval(
        logs / "wanted.eval",
        model="model-0",
        run_id="wanted",
        created="2026-06-01T00:00:00+00:00",
        values=[1.0],
        task="wanted",
    )
    _write_eval(
        logs / "other.eval",
        model="model-0",
        run_id="other",
        created="2026-06-01T00:00:00+00:00",
        values=[1.0],
        task="other",
    )

    hashed: list[str] = []
    original_hash = index_module._file_sha256

    def track_hash(path: Path) -> str:
        hashed.append(path.name)
        return original_hash(path)

    monkeypatch.setattr(index_module, "_file_sha256", track_hash)
    cache = tmp_path / "index.sqlite3"
    progress: list[tuple[int, int, str]] = []
    indexed = index_logs(
        [logs],
        cache,
        tasks={"wanted"},
        progress=lambda done, total, _path, status: progress.append(
            (done, total, status)
        ),
    )

    assert {row["task"] for row in indexed.iter_samples()} == {"wanted"}
    assert len(indexed.files) == 2
    assert hashed == ["wanted.eval"]
    assert indexed.summary.deferred == 1
    assert progress[-1][0:2] == (2, 2)
    deferred = next(row for row in indexed.files if row["task"] == "other")
    assert deferred["parse_status"] == "deferred"
    assert deferred["sha256"] == ""

    promoted = index_logs([logs], cache, tasks={"other"})

    assert {row["task"] for row in promoted.iter_samples()} == {"other"}
    assert hashed == ["wanted.eval", "other.eval"]


def test_content_hash_canonicalizes_shuffled_choice_labels() -> None:
    first = _content_hash(
        "Question", ["B", "C"], {}, ["wrong", "right one", "right two"]
    )
    shuffled = _content_hash(
        "Question", ["A", "C"], {}, ["right two", "wrong", "right one"]
    )

    assert first == shuffled
    assert first != _content_hash(
        "Question", ["A"], {}, ["right two", "wrong", "right one"]
    )


def test_hle_uses_stable_dataset_uid_as_sample_id() -> None:
    assert _logical_sample_id("hle", 1, {"uid": "stable-id"}) == "stable-id"
    assert _logical_sample_id("other", 1, {"uid": "stable-id"}) == "1"


def test_truthfulqa_uses_upstream_stable_id_recipe() -> None:
    question = "Are there wizards living in the West Country?"

    assert _logical_sample_id("truthfulqa", 1, {}, question) == "truthfulqa_000e8c0f"
    assert (
        _logical_sample_id("truthfulqa", "truthfulqa_old", {}, question)
        == "truthfulqa_000e8c0f"
    )


def test_hijacking_hash_uses_authoritative_source_fields() -> None:
    metadata = {"access_code": "ON", "attack": "ignore instructions"}
    first = _content_hash(
        "one concatenated prompt", "", metadata, task="instruction_goal_hijacking"
    )
    structured = _content_hash(
        [
            {"role": "user", "content": "several"},
            {"role": "user", "content": "messages"},
        ],
        "",
        metadata,
        task="instruction_goal_hijacking",
    )

    assert first == structured
    assert first != _content_hash(
        "one concatenated prompt",
        "",
        {**metadata, "attack": "different"},
        task="instruction_goal_hijacking",
    )


def test_mmlu_hash_normalizes_legacy_latex_corruption() -> None:
    legacy = _content_hash(
        "Find \\vert(dr/dt)\\vert at t = 0",
        "A",
        {},
        ["\\vert(dr/dt)\\vert_t=0 = \\surd20", "wrong"],
        task="mmlu_pro",
    )
    cleaned = _content_hash(
        "Find ||(dr/dt)|| at t = 0",
        "A",
        {},
        ["||(dr/dt)|| at_t=0 = √20", "wrong"],
        task="mmlu_pro",
    )

    assert legacy == cleaned


def test_minify_cli_end_to_end(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for model_index in range(3):
        _write_eval(
            logs / f"m{model_index}.eval",
            model=f"model-{model_index}",
            run_id=f"run-{model_index}",
            created=f"2026-07-0{model_index + 1}T00:00:00+00:00",
            values=[float((item + model_index) % 2) for item in range(12)],
        )
    config = tmp_path / "minify.yaml"
    config.write_text("tasks:\n  toy: choice\n")
    output = tmp_path / "output"

    result = CliRunner().invoke(
        app,
        [
            "core",
            "fit",
            str(logs),
            "--config",
            str(config),
            "--budget",
            "5",
            "--output",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Inventory: 3 discovered" in result.output
    assert (output / "artifact.json").is_file()
    assert len((output / "subset.jsonl").read_text().splitlines()) == 5


def test_minify_cli_rejects_existing_output_before_reading_logs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "artifact.json"
    artifact.write_text("existing")

    result = CliRunner().invoke(
        app,
        [
            "core",
            "fit",
            str(tmp_path / "missing-logs"),
            "--budget",
            "5",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "Output already exists:" in result.output
    assert artifact.name in result.output
    assert "Inspect log path does not exist" not in result.output


def test_bundled_scorer_mapping_matches_latest_gp_irt_contracts() -> None:
    assert load_scorers() == {
        "aime_2025": "aime_scorer",
        "arc_challenge": "choice",
        "bbq": "choice",
        "cab": "cab_scorer",
        "gpqa_diamond": "choice",
        "hle": "hle_scorer/score",
        "human_deception": "match",
        "imdb_contrast": "imdb_sentiment_scorer",
        "include": "choice",
        "instruction_goal_hijacking": "Hijacking Score",
        "llm_rules": "llm_rules_scorer",
        "mask": "accuracy_and_honesty/honesty",
        "mmlu_pro": "choice",
        "mmmu_pro": "choice",
        "simpleqa_verified": "simpleqa_scorer/correct",
        "truthfulqa": "choice",
    }


def test_default_config_uses_present_tasks_and_nested_scorers(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for model_index in range(3):
        _write_eval(
            logs / f"m{model_index}.eval",
            model=f"model-{model_index}",
            run_id=f"run-{model_index}",
            created=f"2026-08-1{model_index + 1}T00:00:00+00:00",
            values=[
                {
                    "accuracy": ("correct", "incorrect", "no-belief")[
                        (item + model_index) % 3
                    ],
                    "honesty": ("honest", "lie", "no-belief")[(item + model_index) % 3],
                }
                for item in range(12)
            ],
            task="mask",
            scorer="accuracy_and_honesty",
        )
    output = tmp_path / "output"

    result = CliRunner().invoke(
        app,
        [
            "core",
            "fit",
            str(logs),
            "--budget",
            "5",
            "--output",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = json.loads((output / "artifact.json").read_text())
    assert artifact["task_scorers"] == {"mask": "accuracy_and_honesty/honesty"}
    assert len((output / "subset.jsonl").read_text().splitlines()) == 5
    accuracy = minify(
        [logs],
        {"mask": "accuracy_and_honesty/accuracy"},
        5,
        cache_dir=tmp_path / "cache",
    )
    assert accuracy.artifact["items"][0]["source_mean"] == pytest.approx(1 / 3)


def test_hle_contract_accepts_scalar_and_structured_scores(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for model_index in range(3):
        value: object = float(model_index % 2)
        if model_index:
            value = {"score": value, "confidence": 90}
        _write_eval(
            logs / f"m{model_index}.eval",
            model=f"model-{model_index}",
            run_id=f"run-{model_index}",
            created=f"2026-08-2{model_index + 1}T00:00:00+00:00",
            values=[value] * 12,
            task="hle",
            scorer="hle_scorer",
        )

    result = minify(
        [logs], {"hle": "hle_scorer/score"}, 5, cache_dir=tmp_path / "cache"
    )

    assert len(result.subset) == 5


def test_simpleqa_contract_accepts_legacy_and_current_scorers(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_eval(
        logs / "legacy.eval",
        model="model-0",
        run_id="legacy",
        created="2026-08-24T00:00:00+00:00",
        values=["C"] * 12,
        task="simpleqa_verified",
        scorer="schema_tool_graded_scorer",
    )
    for model_index in range(1, 3):
        _write_eval(
            logs / f"current-{model_index}.eval",
            model=f"model-{model_index}",
            run_id=f"current-{model_index}",
            created=f"2026-08-2{4 + model_index}T00:00:00+00:00",
            values=[{"correct": float(model_index % 2)}] * 12,
            task="simpleqa_verified",
            scorer="simpleqa_scorer",
        )

    result = minify(
        [logs],
        {"simpleqa_verified": "simpleqa_scorer/correct"},
        5,
        cache_dir=tmp_path / "cache",
    )

    assert len(result.subset) == 5


def test_epochs_are_averaged_within_an_eval(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_eval(
        logs / "m0.eval",
        model="model-0",
        run_id="run-0",
        created="2026-08-01T00:00:00+00:00",
        values=[0.0] * 12,
        additional_epochs=[[1.0] * 12],
    )
    for model_index, value in ((1, 0.0), (2, 1.0)):
        _write_eval(
            logs / f"m{model_index}.eval",
            model=f"model-{model_index}",
            run_id=f"run-{model_index}",
            created=f"2026-08-0{model_index + 1}T00:00:00+00:00",
            values=[value] * 12,
        )

    result = minify([logs], {"toy": "choice"}, 5, cache_dir=tmp_path / "cache")

    assert result.artifact["items"][0]["observation_count"] == 3
    assert result.artifact["items"][0]["source_mean"] == pytest.approx(0.5)


def test_predict_new_model_scores_from_artifact_and_subset(tmp_path: Path) -> None:
    source_logs = tmp_path / "source"
    source_logs.mkdir()
    for model_index in range(3):
        _write_eval(
            source_logs / f"m{model_index}.eval",
            model=f"source-{model_index}",
            run_id=f"source-{model_index}",
            created=f"2026-09-0{model_index + 1}T00:00:00+00:00",
            values=[float(item <= model_index * 3) for item in range(12)],
        )
    fitted = minify([source_logs], {"toy": "choice"}, 6, cache_dir=tmp_path / "cache")
    artifact_path, subset_path = write_outputs(fitted, tmp_path / "fitted")

    new_logs = tmp_path / "new"
    new_logs.mkdir()
    _write_eval(
        new_logs / "new.eval",
        model="new-model",
        run_id="new-model",
        created="2026-09-10T00:00:00+00:00",
        values=[1.0] * 12,
    )

    result = predict_scores(
        [new_logs], artifact_path, subset_path, cache_dir=tmp_path / "cache"
    )

    model = result["models"]["new-model"]
    assert model["predicted_score"] > 0.5
    assert model["tasks"]["toy"]["observations"] == 6
    assert model["tasks"]["toy"]["coverage"] == 1.0
    assert model["tasks"]["toy"]["ability"] > 0.0


def test_minify_predict_cli(tmp_path: Path) -> None:
    source_logs = tmp_path / "source"
    source_logs.mkdir()
    for model_index in range(3):
        _write_eval(
            source_logs / f"m{model_index}.eval",
            model=f"source-{model_index}",
            run_id=f"source-{model_index}",
            created=f"2026-10-0{model_index + 1}T00:00:00+00:00",
            values=[float((item + model_index) % 2) for item in range(12)],
        )
    fitted = minify([source_logs], {"toy": "choice"}, 5, cache_dir=tmp_path / "cache")
    artifact_path, subset_path = write_outputs(fitted, tmp_path / "fitted")
    new_logs = tmp_path / "new"
    new_logs.mkdir()
    _write_eval(
        new_logs / "new.eval",
        model="new-model",
        run_id="new-model",
        created="2026-10-10T00:00:00+00:00",
        values=[1.0] * 12,
    )
    output = tmp_path / "predicted.json"

    result = CliRunner().invoke(
        app,
        [
            "core",
            "predict",
            str(new_logs),
            "--artifact",
            str(artifact_path),
            "--subset",
            str(subset_path),
            "--output",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        json.loads(output.read_text())["models"]["new-model"]["predicted_score"] > 0.0
    )


def test_eval_subset_filters_and_orders_exact_items(tmp_path: Path) -> None:
    samples = [
        Sample(id=value, input=f"Question {value}", target=str(value))
        for value in (1, 2, 3)
    ]
    task = Task(dataset=MemoryDataset(samples, name="toy-data"))
    subset_path = tmp_path / "subset.jsonl"
    rows = [
        {
            "task": "toy",
            "sample_id": str(sample_id),
            "item_id": f"toy::toy-data::{sample_id}",
            "content_hash": _content_hash(
                f"Question {sample_id}", str(sample_id), {}, task="toy"
            ),
        }
        for sample_id in (3, 1)
    ]
    subset_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected = read_eval_subset(subset_path)
    apply_eval_subset(["toy"], [task], selected)

    assert [sample.id for sample in task.dataset] == [3, 1]


def _write_eval(
    path: Path,
    *,
    model: str,
    run_id: str,
    created: str,
    values: list[object],
    additional_epochs: list[list[object]] | None = None,
    task: str = "toy",
    scorer: str = "choice",
    dataset: str | None = None,
) -> None:
    samples = []
    for epoch, epoch_values in enumerate([values, *(additional_epochs or [])], start=1):
        if len(epoch_values) != len(values):
            raise ValueError("Every epoch must contain the same items")
        samples.extend(
            EvalSample(
                id=sample_id,
                epoch=epoch,
                input=f"Question {sample_id}",
                target=str(sample_id),
                scores={scorer: Score(value=value)},
            )
            for sample_id, value in enumerate(epoch_values)
        )
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            run_id=run_id,
            created=created,
            task=task,
            task_id=f"{task}-task",
            dataset=EvalDataset(name=dataset or f"{task}-data", samples=len(samples)),
            model=model,
            config=EvalConfig(),
        ),
        results=EvalResults(total_samples=len(samples), completed_samples=len(samples)),
        samples=samples,
    )
    write_eval_log(log, path)
