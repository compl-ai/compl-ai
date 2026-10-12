from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import typer
from rich import print

from complai._cli.utils import default_subset_dir
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
    estimator: Annotated[
        Literal["irt", "gp_irt"] | None,
        typer.Option("--estimator", help="Score estimator; defaults to the fitted artifact (legacy artifacts use irt)."),
    ] = None,
    imputation: Annotated[
        Literal["index_pooled", "task_mean"],
        typer.Option("--imputation", help="How to estimate abilities for subset tasks a model did not run."),
    ] = "index_pooled",
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Predict full-task scores from results on a selected GP-IRT subset."""
    with error_handler(debug):
        if params is None:
            params = default_subset_dir() / "params.json"
        if subset is None:
            subset = default_subset_dir() / "subset.jsonl"
        if output is None:
            output = Path("predicted.json")

        if input_path.suffix == ".jsonl":
            result = predict_scores(
                input_path, params, subset, estimator=estimator,
                imputation=imputation,
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
                    records.records_path, params, subset, estimator=estimator,
                    imputation=imputation,
                )
        output_path = write_prediction(result, output)
        print(f"Wrote {output_path} ({len(result['models'])} model(s))")
