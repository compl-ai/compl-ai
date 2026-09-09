from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import typer
from rich import print

from complai._cli.utils import error_handler
from complai.predict import predict_scores, read_inputs, write_prediction
from complai.utils.log_parser import preprocess_logs

def predict_command(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Inspect log files directory, or preprocessed response JSONL."
        ),
    ],
    params: Annotated[
        Path | None, typer.Option("--params", help="Fitted COMPL-AI Core params.json. Defaults to bundled data.")
    ] = None,
    subset: Annotated[Path | None, typer.Option("--subset", help="Subset JSONL. Defaults to bundled data.")] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Output path for the predicted scores.")
    ] = None,
    duplicates: Annotated[
        Literal["error", "latest", "mean"],
        typer.Option(
            "--duplicates", help="How to handle samples with multiple results."
        ),
    ] = "error",
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Predict full-task scores from results on a selected GP-IRT subset."""
    with error_handler(debug):
        # Auto-resolve bundled data paths if not provided
        data_dir = Path(__file__).parent.parent / "data"
        if params is None or subset is None:
            import json
            config_path = data_dir / "config.json"
            default_version = "core.v1"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                        default_version = cfg.get("default_subset", default_version)
                except Exception:
                    pass
            
            if params is None:
                params = data_dir / default_version / "params.json"
            if subset is None:
                subset = data_dir / default_version / "subset.jsonl"
        if output is None:
            output = Path("predicted.json")

        if input_path.suffix == ".jsonl":
            result = predict_scores(
                input_path, params, subset, duplicate_policy=duplicates
            )
        else:
            fitted, _ = read_inputs(params, subset)
            with TemporaryDirectory(prefix="complai-core-") as temporary_dir:
                records = preprocess_logs(
                    [input_path],
                    fitted["task_scorers"],
                    Path(temporary_dir) / "samples.jsonl",
                )
                result = predict_scores(
                    records.records_path, params, subset, duplicate_policy=duplicates
                )
        output_path = write_prediction(result, output)
        print(f"Wrote {output_path} ({len(result['models'])} model(s))")
