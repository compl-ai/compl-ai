from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from complai._cli import app
from complai.gp_irt import METHOD_VERSION, calibrate_blend, fit_model, predict_task
from complai.irt import fit
from complai.predict import predict_scores
from complai.utils.io import write_outputs
from complai.utils.log_parser import RECORDS_SCHEMA, load_records


def _source() -> np.ndarray:
    values = np.random.default_rng(23).binomial(1, .5, (8, 20)).astype(float)
    values[:, 0] = 0
    values[:, 1] = 1
    values[:6, 2] = np.nan
    return values


def _records(folder: Path, name: str, matrix: np.ndarray):
    """Compact records with stable IDs, including missing scored responses."""
    folder.mkdir(parents=True, exist_ok=True)
    rows, files = [], []
    for model, values in enumerate(matrix):
        metadata = dict(model=f"{name}-{model}", task="toy", dataset="toy-data", created="2026-01-01")
        file_path = f"/fixtures/{name}-{model}.eval"
        files.append(dict(path=file_path, parse_status="ok", **metadata))
        for i, value in enumerate(values):
            rows.append(dict(file_path=file_path, **metadata, sample_id=str(i),
                epoch=1, content_hash=f"content-{i}", question_hash=f"question-{i}",
                scores={"choice": float(value) if np.isfinite(value) else None}))
    path = folder / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_suffix(".manifest.json").write_text(json.dumps(dict(
        schema_version=RECORDS_SCHEMA, files=files, records=len(rows),
        input_digest=name, scorers={"toy": "choice"})))
    return load_records(path)


def test_matches_minibench_reference() -> None:
    reference = json.loads((Path(__file__).parent / "fixtures/gp_irt_reference.json").read_text())
    source = _source()
    fitted = fit_model(source)
    for field in ("abilities", "discriminations", "intercepts"):
        np.testing.assert_allclose(getattr(fitted, field), reference[field], atol=1e-12, rtol=1e-12)
    selected = np.array(reference["selected"])
    blend = calibrate_blend(source, selected)
    assert blend == reference["blend"]
    responses = np.array([np.nan if v is None else v for v in reference["responses"]])
    for name, weights in (("uniform", np.ones(len(selected))), ("weighted", np.arange(1, len(selected)+1))):
        prediction = predict_task(
            responses, fitted.discriminations[selected], fitted.intercepts[selected],
            fitted.discriminations, fitted.intercepts, blend=blend, weights=weights,
        )
        assert prediction["ability"] == pytest.approx(reference[name]["ability"], abs=1e-12)
        assert prediction["predicted_score"] == pytest.approx(reference[name]["score"], abs=1e-12)


def test_small_source_fallback_and_full_sample_reconstruction() -> None:
    source = _source()
    assert calibrate_blend(source[:3], np.arange(6)) == .5
    assert calibrate_blend(source, np.arange(source.shape[1])) == 1.0


def test_artifact_roundtrip_and_prediction_without_training_data(tmp_path: Path, monkeypatch) -> None:
    records = _records(tmp_path, "source", _source())
    legacy = fit(records, records.scorers, 8)
    assert legacy == fit(records, records.scorers, 8, estimator="irt")
    fitted = fit(records, records.scorers, 8, estimator="gp_irt")
    assert fitted.params["method"] == METHOD_VERSION
    assert fitted.params["params_id"] != legacy.params["params_id"]
    assert fitted.params["subset_id"] != legacy.params["subset_id"]
    assert fitted.params["configuration_digest"] != legacy.params["configuration_digest"]
    params, subset = write_outputs(fitted, tmp_path / "gp")
    records.records_path.unlink()
    records.manifest_path.unlink()
    monkeypatch.setattr("complai.gp_irt.calibrate_blend", lambda *a: pytest.fail("Prediction must not calibrate on targets"))
    target = _records(tmp_path, "target", np.ones((1, 20)))
    automatic = predict_scores(target.records_path, params, subset)
    assert automatic == predict_scores(target.records_path, params, subset, estimator="gp_irt")
    model = automatic["models"]["target-0"]
    assert automatic["estimator"] == "gp_irt"
    assert model["predicted_score_error"] is None
    assert model["tasks"]["toy"]["blend_weight"] == fitted.params["gp_irt"]["task_blends"]["toy"]
    irt = predict_scores(target.records_path, params, subset, estimator="irt")
    assert irt["prediction_id"] != automatic["prediction_id"]
    assert irt["models"]["target-0"]["predicted_score_error"] is not None
    # Changing unobserved responses cannot change either gp score component.
    selected_ids = {int(row["sample_id"]) for row in fitted.subset}
    changed = np.array([[1. if i in selected_ids else 0. for i in range(20)]])
    target = _records(tmp_path, "target", changed)
    assert predict_scores(target.records_path, params, subset)["models"] == automatic["models"]


def test_legacy_artifact_rejects_uncalibrated_gp(tmp_path: Path) -> None:
    records = _records(tmp_path, "source", _source())
    fitted = fit(records, records.scorers, 8)
    params, subset = write_outputs(fitted, tmp_path / "irt")
    default = predict_scores(records.records_path, params, subset)
    assert default == predict_scores(records.records_path, params, subset, estimator="irt")
    with pytest.raises(ValueError, match="fit --estimator gp_irt"):
        predict_scores(records.records_path, params, subset, estimator="gp_irt")


@pytest.mark.parametrize("estimator", ["irt", "gp_irt"])
def test_fit_and_predict_cli(tmp_path: Path, estimator: str) -> None:
    records = _records(tmp_path, "source", _source())
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"tasks": {"toy": "choice"}}))
    directory = tmp_path / "fitted"
    runner = CliRunner()
    result = runner.invoke(app, ["minify", "fit", str(records.records_path), "--config", str(config),
        "--budget", "8", "--name", str(directory), "--estimator", estimator])
    assert result.exit_code == 0, result.output
    target = _records(tmp_path, "target", np.ones((1, 20)))
    output = tmp_path / "prediction.json"
    result = runner.invoke(app, ["predict", str(target.records_path),
        "--params", str(directory / "params.json"), "--subset", str(directory / "subset.jsonl"),
        "--estimator", estimator, "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["models"]["target-0"]["tasks"]["toy"]["observations"] == 8


def test_gp_stats_uses_saved_blend_and_production_estimator(tmp_path: Path, monkeypatch) -> None:
    from tools.minify import evaluate_subsets as stats

    records = _records(tmp_path, "source", _source())
    fitted = fit(records, records.scorers, 8, estimator="gp_irt")
    params, subset = write_outputs(fitted, tmp_path / "gp")
    target = np.random.default_rng(19).binomial(1, .6, (3, 20)).astype(float)
    target[:, 2] = np.nan
    data = tmp_path / "tools/minify/data"
    responses = _records(data, "samples", target)
    (data / "metrics.json").write_text(json.dumps({"toy": {f"samples-{i}": float(np.nanmean(row)) for i, row in enumerate(target)}}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stats, "load_scorers", lambda _: {"toy": "choice"})
    monkeypatch.setattr(stats, "get_task_allocations", lambda _: {"toy": "default"})
    monkeypatch.setattr(stats, "get_primary_metrics", lambda _: {"toy": ["accuracy"]})
    monkeypatch.setattr("complai.gp_irt.calibrate_blend", lambda *a: pytest.fail("Stats must not calibrate on targets"))
    captured = []
    def record_prediction(*args, **kwargs):
        result = predict_task(*args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(stats, "predict_task", record_prediction)
    result = CliRunner().invoke(app, ["minify", "stats", "--data-dir", str(params.parent), "--estimator", "gp_irt"])
    assert result.exit_code == 0, result.output
    assert "Estimator: gp_irt" in result.output
    assert "Avg Benchmark MAE:" in result.output
    predictions = predict_scores(responses.records_path, params, subset)
    assert len(captured) == 3
    for actual, model in zip(captured, predictions["models"].values()):
        assert actual["predicted_score"] == pytest.approx(model["tasks"]["toy"]["predicted_score"], abs=1e-12)
        assert actual["ability"] == pytest.approx(model["tasks"]["toy"]["ability"], abs=1e-12)


@pytest.mark.parametrize("blend", [None, -1, 2, float("nan")])
def test_invalid_calibration_rejected(tmp_path: Path, blend) -> None:
    records = _records(tmp_path, "source", _source())
    fitted = fit(records, records.scorers, 8, estimator="gp_irt")
    params, subset = write_outputs(fitted, tmp_path / "gp")
    payload = json.loads(params.read_text())
    payload["gp_irt"]["task_blends"]["toy"] = blend
    params.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Missing or invalid gp_irt blend"):
        predict_scores(records.records_path, params, subset)
