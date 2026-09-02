from pathlib import Path
from typing import Annotated
from typing import Literal

import typer
from rich import print
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from complai._cli.utils import error_handler
from complai.core.config import load_scorers
from complai.core.fit import check_output_available
from complai.core.fit import minify
from complai.core.fit import write_outputs
from complai.core.index import InventorySummary
from complai.core.prediction import predict_scores
from complai.core.prediction import write_prediction


def fit_command(
    log_paths: Annotated[
        list[Path], typer.Argument(help="Inspect .eval files or directories.")
    ],
    budget: Annotated[
        int, typer.Option("--budget", min=1, help="Size of the selected subset.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Directory to write output to.")
    ] = Path("minify_output"),
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
    reindex: Annotated[
        bool, typer.Option("--reindex", help="Rehash and reparse all supplied logs.")
    ] = False,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Directory for the minify log cache."),
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Fit GP-IRT 2PL and select an evaluation subset."""
    with error_handler(debug):
        check_output_available(output)
        scorer_mapping = load_scorers(scorers)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            index_task = progress.add_task("[cyan]Indexing logs...", total=None)
            fit_task = progress.add_task("[dim]Fitting GP-IRT (waiting...)", total=100, completed=0)
            
            def index_cb(current: int, total: int, path: Path, status: str) -> None:
                progress.update(index_task, completed=current, total=total)
                
            def fit_cb(current: int, total: int, task_name: str) -> None:
                progress.update(
                    fit_task, 
                    completed=current, 
                    total=total,
                    description=f"[green]Fitting GP-IRT... ({task_name})"
                )
                
            result = minify(
                log_paths,
                scorer_mapping,
                budget,
                seed=seed,
                duplicate_policy=duplicates,
                cache_dir=cache_dir,
                reindex=reindex,
                _ignore_unseen_tasks=scorers is None,
                index_progress=index_cb,
                fit_progress=fit_cb,
            )
        _print_inventory(result.inventory.summary)
        params_path, subset_path = write_outputs(result, output)
        print(
            f"Wrote {params_path} and {subset_path} "
            f"({len(result.subset)} subset items)"
        )


def predict_command(
    log_paths: Annotated[
        list[Path],
        typer.Argument(
            help="Inspect .eval files or directories on the subset for the new model(s)."
        ),
    ],
    params: Annotated[
        Path, typer.Option("--params", help="Fitted minify params.json.")
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
    reindex: Annotated[
        bool, typer.Option("--reindex", help="Rehash and reparse all supplied logs.")
    ] = False,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Directory for the minify log cache."),
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Predict full-task scores from results on a selected GP-IRT subset."""
    with error_handler(debug):
        result = predict_scores(
            log_paths,
            params,
            subset,
            duplicate_policy=duplicates,
            cache_dir=cache_dir,
            reindex=reindex,
        )
        output_path = write_prediction(result, output)
        print(f"Wrote {output_path} ({len(result['models'])} model(s))")


def _print_inventory(summary: InventorySummary) -> None:
    """Print a short log inventory summary."""
    print(
        "Inventory: "
        f"{summary.discovered} discovered, {summary.cache_hits} cache hits, "
        f"{summary.new} new, {summary.changed} changed, "
        f"{summary.removed} removed, {summary.excluded} excluded, "
        f"{summary.deferred} deferred, {summary.failed} failed"
    )
