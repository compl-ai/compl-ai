import sys
from pathlib import Path

# When run directly from tools/minify/, we need the repo root in our path
# so that we can import `complai` from the `src/` directory.
_repo_root = Path(__file__).resolve().parent.parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json
import typer
from pathlib import Path
from typing import Annotated, Literal
from rich import print

from complai._cli.utils import error_handler
from tools.minify.config import load_indices, load_scorers
from complai.utils.log_parser import load_records, preprocess_logs
from complai.irt import fit, refit
from complai.utils.io import check_output_available, write_outputs

app = typer.Typer(help="Maintainer tools for generating COMPL-AI Core subsets.")

@app.command("stats")
def stats_command(
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Path to subset directory.")] = Path("src/complai/data"),
    logs_dir: Annotated[Path, typer.Option("--logs-dir", help="Path to logs.")] = Path("logs/"),
    estimator: Annotated[
        Literal["irt", "gp_irt"] | None,
        typer.Option("--estimator", help="Score estimator; defaults to each fitted artifact."),
    ] = None,
    imputation: Annotated[
        Literal["index_pooled", "task_mean"],
        typer.Option("--imputation", help="Imputation method for tasks a model did not run."),
    ] = "index_pooled",
) -> None:
    """Evaluate a subset's discriminative power and mean absolute error."""
    from tools.minify.evaluate_subsets import evaluate_stats
    evaluate_stats(
        data_dir=data_dir, logs_dir=logs_dir, estimator=estimator,
        imputation=imputation,
    )


@app.command("preprocess")
def preprocess_command(
    log_paths: Annotated[
        list[Path], typer.Argument(help="Inspect .eval files or directories.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Compact response JSONL to write.")
    ] = Path("tools/minify/data/samples.jsonl"),
    scorers: Annotated[
        Path | None,
        typer.Option(
            "--scorers",
            "--config",
            help="File in the format of src/complai/data/config.json; uses that file if omitted.",
        ),
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Parallel log-parsing processes.")
    ] = 12,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Preprocess Inspect logs into compact response records."""
    with error_handler(debug):
        result = preprocess_logs(
            log_paths,
            load_scorers(scorers),
            output,
            workers=workers,
        )
        print(
            f"Wrote {result.records_path} and {result.manifest_path} "
            f"({result.records} records)"
        )


@app.command("fit")
def fit_command(
    samples_path: Annotated[Path, typer.Argument(help="Preprocessed response JSONL.")] = Path("tools/minify/data/samples.jsonl"),
    budget: Annotated[
        int, typer.Option("--budget", min=1, help="Size of the selected subset.")
    ] = ...,
    name: Annotated[
        str, typer.Option("--name", help="Name of the subset version (e.g. core.v1).")
    ] = ...,
    scorers: Annotated[
        Path | None,
        typer.Option(
            "--scorers",
            "--config",
            help="File in the format of src/complai/data/config.json; uses that file if omitted.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Selection seed.")] = 0,
    item_selection: Annotated[
        Literal["random", "discrimination", "joint", "joint_discrimination", "smoke"],
        typer.Option(
            "--item-selection",
            help="Strategy for selecting items within a benchmark.",
        ),
    ] = "random",
    drop_uninformative: Annotated[
        bool,
        typer.Option(
            "--drop-uninformative",
            help=(
                "Never select items that cannot separate models: every model answered "
                "correctly, or stronger models do no better."
            ),
        ),
    ] = False,
    estimator: Annotated[
        Literal["irt", "gp_irt"],
        typer.Option("--estimator", help="irt or Minibench gp_irt with source-calibrated direct/IRT blending."),
    ] = "irt",
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            help="Item label directory; secondary labels are written into params for reporting only.",
        ),
    ] = Path("tools/label/labels"),
    floor_percent: Annotated[float, typer.Option("--floor-percent", help="Minimum percentage floor.")] = 1.0,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Fit GP-IRT 2PL and select an evaluation subset."""
    with error_handler(debug):
        output = Path(__file__).resolve().parent.parent.parent / "src" / "complai" / "data" / name
        output.mkdir(parents=True, exist_ok=True)
        check_output_available(output)
        records = load_records(samples_path)
        scorer_mapping = load_scorers(scorers) if scorers else records.scorers
        
        result = fit(
            records,
            scorer_mapping,
            budget,
            floor_percent=floor_percent,
            seed=seed,
            item_selection=item_selection,
            drop_uninformative=drop_uninformative,
            indices=load_indices(scorers),
            labels_dir=labels,
            estimator=estimator,
            _ignore_unseen_tasks=scorers is None,
        )
        params_path, subset_path = write_outputs(result, output)
        print(
            f"Wrote {params_path} and {subset_path} ({len(result.subset)} subset items)"
        )

@app.command("refit")
def refit_command(
    samples_path: Annotated[Path, typer.Argument(help="Preprocessed response JSONL.")] = Path("tools/minify/data/samples.jsonl"),
    subset: Annotated[
        str, typer.Option("--subset", help="Existing subset version whose items are kept (e.g. core.v1).")
    ] = ...,
    name: Annotated[
        str, typer.Option("--name", help="Name of the new subset version (e.g. core.v1.1).")
    ] = ...,
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            help="Item label directory; secondary labels are written into params for reporting only.",
        ),
    ] = Path("tools/label/labels"),
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Refit params on new samples, keeping the items of an existing subset."""
    with error_handler(debug):
        data_dir = Path(__file__).resolve().parent.parent.parent / "src" / "complai" / "data"
        output = data_dir / name
        check_output_available(output)
        params = json.loads((data_dir / subset / "params.json").read_text())
        with (data_dir / subset / "subset.jsonl").open() as file:
            subset_rows = [json.loads(line) for line in file if line.strip()]
        result = refit(load_records(samples_path), params, subset_rows, labels_dir=labels)
        params_path, subset_path = write_outputs(result, output)
        print(
            f"Wrote {params_path} and {subset_path} ({len(result.subset)} subset items)"
        )


if __name__ == "__main__":
    app()
