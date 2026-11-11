import json
import statistics
import collections
import zipfile
from pathlib import Path

import zipfile_zstd  # noqa: F401  (Inspect dependency) lets zipfile read zstd-compressed .eval members

logs_dir = Path("logs/")
# Core runs: not used for cost.json (their items are a selected subset), only for measured per-eval usage.
subset_logs_dir = Path("logs_subset/")
task_model_accumulator = collections.defaultdict(lambda: collections.defaultdict(lambda: {"input": 0, "output": 0, "total": 0, "samples": 0}))
# Measured tokens per eval file, keyed like the minify manifest's "path" (e.g. "logs/<run>/<file>.eval").
eval_usage = {}


def split_usage(usage, model):
    """Tokens of the evaluated model (as Inspect logged it, e.g. a batch provider), plus any judge/grader models."""
    out = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "model_tokens": 0, "judge_tokens": 0}
    for name, tokens in usage.items():
        if name == model:
            out["input_tokens"] += tokens.get("input_tokens") or 0
            out["output_tokens"] += tokens.get("output_tokens") or 0
            out["reasoning_tokens"] += tokens.get("reasoning_tokens") or 0
            out["model_tokens"] += tokens.get("total_tokens") or 0
        else:
            out["judge_tokens"] += tokens.get("total_tokens") or 0
    return out


# 1. Read token usage from each .eval file's header.json (Inspect's stats.model_usage).
# Read straight from the zip: Inspect's header_only read also parses reductions.json (large for e.g. bbq).
# Per-run logs.json summaries are not used: they can be stale or missing.
for eval_path in sorted([*logs_dir.glob("*/*.eval"), *subset_logs_dir.glob("*/*.eval")]):
    in_logs_dir = eval_path.parent.parent == logs_dir
    try:
        with zipfile.ZipFile(eval_path) as zf:
            # Unfinished runs have no header.json, and no usage stats.
            if "header.json" not in zf.namelist():
                continue
            header = json.loads(zf.read("header.json"))
    except Exception as e:
        print(f"   skipping unreadable {eval_path}: {e}")
        continue

    task_name = header.get("eval", {}).get("task", "unknown")
    model = header.get("eval", {}).get("model", "unknown")
    split = split_usage(header.get("stats", {}).get("model_usage") or {}, model)
    tot_tok = split["model_tokens"] + split["judge_tokens"]

    # Unique samples: completed_samples counts every epoch.
    epochs = header.get("eval", {}).get("config", {}).get("epochs") or 1
    samples = ((header.get("results") or {}).get("completed_samples") or 0) // epochs

    if tot_tok > 0:
        eval_usage[f"{eval_path.parent}/{eval_path.name}"] = {
            "task": task_name,
            "model": model,
            "samples": samples,
            **split,
        }

    # Judge/grader tokens are left out of the per-task cost: only the evaluated model counts.
    if in_logs_dir and samples > 0 and split["model_tokens"] > 0:
        task_model_accumulator[task_name][model]["input"] += split["input_tokens"]
        task_model_accumulator[task_name][model]["output"] += split["output_tokens"]
        task_model_accumulator[task_name][model]["total"] += split["model_tokens"]
        task_model_accumulator[task_name][model]["samples"] += samples

# 2. Compute true statistical distributions per benchmark
cost_distributions = {}
for task_key, models in task_model_accumulator.items():
    global_input_toks = 0
    global_samples = 0
    model_out_avgs = []
    
    for m, totals in models.items():
        s = totals["samples"]
        if s > 0:
            global_input_toks += totals["input"]
            global_samples += s
            model_out_avgs.append(totals["output"] / s)
            
    if global_samples > 0 and model_out_avgs:
        avg_in = global_input_toks / global_samples
        model_out_avgs.sort()
        
        cost_distributions[task_key] = {
            "input_avg": avg_in,
            "output_min": min(model_out_avgs),
            "output_max": max(model_out_avgs),
            "output_median": statistics.median(model_out_avgs),
            "output_stdev": statistics.stdev(model_out_avgs) if len(model_out_avgs) > 1 else 0.0
        }

# 3. Write to exact JSON file
out_dir = Path("tools/minify/data")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "cost.json", "w") as f:
    json.dump(cost_distributions, f, indent=2)

with open(out_dir / "model_tokens.json", "w") as f:
    json.dump(dict(sorted(eval_usage.items())), f, indent=1)

print("Read .eval headers. Wrote tools/minify/data/cost.json and tools/minify/data/model_tokens.json")
