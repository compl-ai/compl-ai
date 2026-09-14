import sys
import os
from pathlib import Path

# When run directly from tools/minify/, we need the repo root in our path
# so that we can import `complai` from the `src/` directory.
_repo_root = Path(__file__).resolve().parent.parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import typer
from pathlib import Path
from typing import Annotated, Literal
from rich import print

from complai._cli.utils import error_handler
from tools.minify.config import load_scorers, get_task_allocations
from complai.utils.log_parser import load_records, preprocess_logs
from complai.irt import fit, check_output_available, write_outputs

app = typer.Typer(help="Maintainer tools for generating COMPL-AI Core subsets.")

@app.command("stats")
def stats_command(
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Path to subset directory.")] = Path("src/complai/data"),
    logs_dir: Annotated[Path, typer.Option("--logs-dir", help="Path to logs.")] = Path("logs/"),
    labels_dir: Annotated[Path, typer.Option("--labels-dir", help="Path to domain labels.")] = Path("tools/label/labels"),
    estimator: Annotated[
        Literal["irt", "gp_irt"] | None,
        typer.Option("--estimator", help="Score estimator; defaults to each fitted artifact."),
    ] = None,
) -> None:
    """Evaluate a subset's discriminative power and mean absolute error."""
    from tools.minify.evaluate_subsets import evaluate_stats
    evaluate_stats(data_dir=data_dir, logs_dir=logs_dir, labels_dir=labels_dir, estimator=estimator)


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
            help="JSON subset configuration mapping; uses bundled defaults if omitted.",
        ),
    ] = None,
    domain_labels: Annotated[
        Path | None,
        typer.Option(
            "--domain-labels",
            help="Path to the tools/labeling/labels directory.",
        ),
    ] = None,
    valid_labels_only: Annotated[
        bool,
        typer.Option(
            "--valid-labels-only",
            help="If True, only items found in the domain labels are written.",
        ),
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Preprocess Inspect logs into compact response records."""
    with error_handler(debug):
        result = preprocess_logs(
            log_paths, load_scorers(scorers), output, domain_labels_dir=domain_labels, valid_labels_only=valid_labels_only
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
            help="JSON subset configuration mapping; uses bundled defaults if omitted.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Selection seed.")] = 0,
    duplicates: Annotated[
        Literal["error", "latest", "mean"],
        typer.Option(
            "--duplicates", help="How to handle samples with multiple results."
        ),
    ] = "error",
    item_selection: Annotated[
        Literal["random", "discrimination", "joint", "smoke"],
        typer.Option(
            "--item-selection",
            help="Strategy for selecting items within a benchmark.",
        ),
    ] = "random",
    estimator: Annotated[
        Literal["irt", "gp_irt"],
        typer.Option("--estimator", help="irt or Minibench gp_irt with source-calibrated direct/IRT blending."),
    ] = "irt",
    domain_labels: Annotated[
        Path | None,
        typer.Option(
            "--domain-labels",
            help="Path to the tools/labeling/labels directory (required for 'joint' selection).",
        ),
    ] = None,
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
        allocations = get_task_allocations(scorers)
        floor_dict = {t: 1000000 if a == "full" else 5 for t, a in allocations.items()}
        
        result = fit(
            records,
            scorer_mapping,
            budget,
            floor=floor_dict,
            seed=seed,
            duplicate_policy=duplicates,
            item_selection=item_selection,
            domain_labels_dir=domain_labels,
            estimator=estimator,
            _ignore_unseen_tasks=scorers is None,
        )
        params_path, subset_path = write_outputs(result, output)
        print(
            f"Wrote {params_path} and {subset_path} ({len(result.subset)} subset items)"
        )

if __name__ == "__main__":
    app()


@app.command("cost")
def cost_command(
    subset_dir: Annotated[Path, typer.Argument(help="Path to subset directory (e.g. src/complai/data/core.v1.2.5k)")] = Path("src/complai/data/core.v1.2.5k"),
    logs_dir: Annotated[Path, typer.Option("--logs-dir", help="Path to raw Inspect logs (default: logs/)")] = Path("logs/")
):
    """Estimate the API token cost to run a specific subset."""
    import json
    import collections
    
    print(f"Scanning raw logs in {logs_dir} to calculate average tokens per sample...")
    
    # 1. Calculate Average Tokens per Sample across all tasks
    task_tokens = collections.defaultdict(int)
    task_samples = collections.defaultdict(int)
    
    for logs_json in logs_dir.glob("logs/*/logs.json"):
        with open(logs_json, "r") as f:
            data = json.load(f)
            for eval_key, eval_obj in data.items():
                task_name = eval_obj.get("eval", {}).get("task", "unknown")
                
                # Sum all model tokens (including judges)
                usage = eval_obj.get("stats", {}).get("model_usage", {})
                total_tokens = sum(v.get("total_tokens", 0) for v in usage.values())
                
                # Count samples evaluated
                reductions = eval_obj.get("reductions", [])
                if reductions and isinstance(reductions, list):
                    samples = len(reductions[0].get("samples", []))
                else:
                    samples = 0
                    
                if samples > 0 and total_tokens > 0:
                    task_tokens[task_name] += total_tokens
                    task_samples[task_name] += samples

    avg_tokens = {}
    for task, tokens in task_tokens.items():
        avg_tokens[task] = tokens / task_samples[task]
        
    # 2. Read Subset Allocation
    print(f"Reading subset allocation from {subset_dir}...")
    subset_file = subset_dir / "subset.jsonl"
    if not subset_file.exists():
        print(f"Error: Could not find {subset_file}")
        return
        
    subset_counts = collections.defaultdict(int)
    with open(subset_file, "r") as f:
        for line in f:
            row = json.loads(line)
            task = row.get("task", "unknown")
            subset_counts[task] += 1
            
    # 3. Print Table
    print(f"\n=======================================================")
    print(f"               API COST ESTIMATION                     ")
    print(f"=======================================================\n")
    print(f"| Benchmark | Avg Tokens / Sample | Allocation | Projected Tokens |")
    print(f"| :--- | :--- | :--- | :--- |")
    
    total_projected_tokens = 0
    total_items = 0
    
    for task in sorted(subset_counts.keys()):
        count = subset_counts[task]
        avg = avg_tokens.get(task, 0.0)
        projected = count * avg
        
        total_items += count
        total_projected_tokens += projected
        
        print(f"| {task:<25} | {int(avg):>19,} | {count:>10,} | {int(projected):>16,} |")
        
    print(f"| {'-'*25} | {'-'*19} | {'-'*10} | {'-'*16} |")
    print(f"| **TOTAL**                 |                     | **{total_items:>8,}** | **{int(total_projected_tokens):>14,}** |")
    print("\n*Note: Estimates are averaged across all models in your logs (including LLM-as-a-judge overhead).*")
