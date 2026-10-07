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

import typer
from pathlib import Path
from typing import Annotated, Literal
from rich import print

from complai._cli.utils import error_handler
from tools.minify.config import load_indices, load_scorers
from complai.utils.log_parser import load_records, preprocess_logs
from complai.irt import fit
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
            help="JSON subset configuration mapping; uses bundled defaults if omitted.",
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
            help="JSON subset configuration mapping; uses bundled defaults if omitted.",
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

@app.command("cost")
def cost_command(
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Path to subset directory or parent directory.")] = Path("src/complai/data"),
    cost_file: Annotated[Path, typer.Option("--cost-file", help="Path to precomputed cost.json.")] = Path("tools/minify/data/cost.json"),
    price_in: Annotated[float, typer.Option("--price-in", help="Cost per 1M input tokens ($)")] = 3.0,
    price_out: Annotated[float, typer.Option("--price-out", help="Cost per 1M output tokens ($)")] = 15.0,
):
    """Estimate API token cost ranges for subsets based on precomputed token distributions."""
    import json
    import collections

    VARIANTS_PER_BASIS = {"strong_reject": 35}

    if not cost_file.exists():
        print(f"Error: {cost_file} not found. Run 'python tools/minify/extract_cost.py' to generate it.")
        return
        
    with open(cost_file, "r") as f:
        cost_distributions = json.load(f)
        
    subset_dirs = []
    if (data_dir / "subset.jsonl").exists():
        subset_dirs.append(data_dir)
    else:
        for t_dir in sorted(data_dir.iterdir()):
            if t_dir.is_dir() and (t_dir / "subset.jsonl").exists():
                subset_dirs.append(t_dir)
                
    if not subset_dirs:
        print(f"No subset.jsonl found in {data_dir} or its subdirectories.")
        return
        
    for target_dir in subset_dirs:
        subset_file = target_dir / "subset.jsonl"
        subset_counts = collections.defaultdict(int)
        with open(subset_file, "r") as f:
            for line in f:
                task = json.loads(line)["item_id"].split("::", 1)[0]
                # cost.json is per executed sample; a strong_reject basis item runs all its jailbreak variants.
                subset_counts[task] += VARIANTS_PER_BASIS.get(task, 1)
                
        print(f"\n--- Estimating Cost for {target_dir.name} ---")
        print(f"Assumed API Pricing: ${price_in:.2f} / 1M Input | ${price_out:.2f} / 1M Output")
        print(f"\n| Benchmark                  | Samples Run  | In / Out Tokens (Median) | Proj. Cost Range (Min -> Med -> Max) |")
        print(f"| :------------------------- | :----------- | :----------------------- | :----------------------------------- |")
        
        total_items = 0
        total_cost_min = 0.0
        total_cost_median = 0.0
        total_cost_max = 0.0
        
        for task in sorted(subset_counts.keys()):
            count = subset_counts[task]
            task_key = task
            if task_key not in cost_distributions:
                task_key = task.replace("-", "_")
                if task_key not in cost_distributions:
                    task_key = task.replace("_", "-")
            
            dist = cost_distributions.get(task_key, {
                "input_avg": 0.0, "output_min": 0.0, "output_median": 0.0, "output_max": 0.0, "output_stdev": 0.0
            })
            
            # If the user happens to have the old cost.json loaded
            if "input" in dist:
                dist = {"input_avg": dist["input"], "output_min": dist["output"], "output_median": dist["output"], "output_max": dist["output"]}
            
            in_avg = dist.get("input_avg", 0.0)
            out_min = dist.get("output_min", 0.0)
            out_med = dist.get("output_median", 0.0)
            out_max = dist.get("output_max", 0.0)
            
            base_in_cost = (count * in_avg * price_in) / 1_000_000
            
            c_min = base_in_cost + ((count * out_min * price_out) / 1_000_000)
            c_med = base_in_cost + ((count * out_med * price_out) / 1_000_000)
            c_max = base_in_cost + ((count * out_max * price_out) / 1_000_000)
            
            total_items += count
            total_cost_min += c_min
            total_cost_median += c_med
            total_cost_max += c_max
            
            cost_str = f"${c_min:.2f} → ${c_med:.2f} → ${c_max:.2f}"
            print(f"| {task:<26} | {count:>12,} | {int(in_avg):>8,} / {int(out_med):<14,} | {cost_str:>36} |")
            
        print(f"| {'-'*26} | {'-'*12} | {'-'*24} | {'-'*36} |")
        total_str = f"${total_cost_min:.2f} → ${total_cost_median:.2f} → ${total_cost_max:.2f}"
        print(f"| **TOTAL**                  | **{total_items:>10,}** |                          | **{total_str:>34}** |")



if __name__ == "__main__":
    app()
