from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated
from typing import Literal

import typer
from rich import print

from complai._cli.utils import error_handler
from complai.core.config import load_scorers
from complai.core.fit import check_output_available
from complai.core.fit import fit
from complai.core.fit import write_outputs
from complai.core.predict import predict_scores
from complai.core.predict import read_inputs
from complai.core.predict import write_prediction
from complai.core.records import load_records
from complai.core.records import preprocess_logs


def fit_command(
    samples_path: Annotated[Path, typer.Argument(help="Preprocessed response JSONL.")],
    budget: Annotated[
        int, typer.Option("--budget", min=1, help="Size of the selected subset.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Directory to write output to.")
    ] = Path("complai-core-output"),
    scorers: Annotated[
        Path | None,
        typer.Option(
            "--scorers",
            "--config",
            help="YAML or JSON task-to-scorer mapping; uses bundled defaults if omitted.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Selection seed.")] = 0,
    duplicates: Annotated[
        Literal["error", "latest", "mean"],
        typer.Option(
            "--duplicates", help="How to handle samples with multiple results."
        ),
    ] = "error",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite existing params and subset files."),
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Fit GP-IRT 2PL and select an evaluation subset."""
    with error_handler(debug):
        check_output_available(output, overwrite)
        records = load_records(samples_path)
        scorer_mapping = load_scorers(scorers) if scorers else records.scorers
        result = fit(
            records,
            scorer_mapping,
            budget,
            seed=seed,
            duplicate_policy=duplicates,
            _ignore_unseen_tasks=scorers is None,
        )
        params_path, subset_path = write_outputs(result, output, overwrite)
        print(
            f"Wrote {params_path} and {subset_path} ({len(result.subset)} subset items)"
        )


def preprocess_command(
    log_paths: Annotated[
        list[Path], typer.Argument(help="Inspect .eval files or directories.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Compact response JSONL to write.")
    ] = Path("samples.jsonl"),
    scorers: Annotated[
        Path | None,
        typer.Option(
            "--scorers",
            "--config",
            help="YAML or JSON task-to-scorer mapping; uses bundled defaults if omitted.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite existing records and manifest."),
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Preprocess Inspect logs into compact response records."""
    with error_handler(debug):
        result = preprocess_logs(
            log_paths, load_scorers(scorers), output, overwrite=overwrite
        )
        print(
            f"Wrote {result.records_path} and {result.manifest_path} "
            f"({result.records} records)"
        )


def predict_command(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Inspect log files directory, or preprocessed response JSONL."
        ),
    ],
    params: Annotated[
        Path, typer.Option("--params", help="Fitted COMPL-AI Core params.json.")
    ],
    subset: Annotated[Path, typer.Option("--subset", help="Subset JSONL.")],
    output: Annotated[
        Path, typer.Option("--output", help="Output path for the predicted scores.")
    ],
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
