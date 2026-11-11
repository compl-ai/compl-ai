"""Token usage and run type per model, for showing what the core subset saves.

Nothing here reads the logs. Tokens come from the files ``tools/minify/extract_cost.py``
writes from them:

- ``model_tokens.json``: measured tokens per eval file (Inspect's ``stats.model_usage``),
  for both full runs (``logs/``) and core runs (``logs_subset/``).
- ``cost.json``: tokens per item of each task across models, used for the suite-wide
  core vs full estimate (the same figures as ``minify cost``).

There are no dollar amounts, so everything is in tokens. Totals count only the evaluated
model; judge/grader tokens are reported separately (``judge``) but left out of the core vs
full comparison.

For each task a model ran, its measured tokens per observed item give an estimate of
what the other scope would cost: a core run is extrapolated to the full population and
a full run is scaled down to the subset. The scope the model actually ran is measured; the other one is an estimate.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.minify.minify import VARIANTS_PER_BASIS

MINIFY_DATA = Path("tools/minify/data")
MODEL_TOKENS_PATH = MINIFY_DATA / "model_tokens.json"
COST_PATH = MINIFY_DATA / "cost.json"

# A task counts as a full run once the model answered this share of its population
# (some tasks lose items when scoring, e.g. mmlu_pro_robustness keeps 75-85%).
FULL_RUN_SHARE = 0.5

TOKEN_KEYS = ("input_tokens", "output_tokens", "reasoning_tokens", "model_tokens", "judge_tokens")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"   {path} not found, run 'python tools/minify/extract_cost.py' to generate it. Skipping tokens.")
        return {}
    with open(path) as f:
        return json.load(f)


def _task_run(observed: int, subset_items: int, population_items: int) -> str:
    if observed <= subset_items:
        return "core"
    if observed >= FULL_RUN_SHARE * population_items:
        return "full"
    return "partial"


def _scope_differs(entry: dict[str, Any]) -> bool:
    """Whether the core subset is meaningfully smaller than the task, so the run type is identifiable."""
    return entry["subset_items"] < FULL_RUN_SHARE * entry["population_items"]


def _observed_items(prediction: Any, model: str, task: str) -> int:
    task_data = prediction.tasks.get(task)
    if not task_data or model not in task_data["models"]:
        return 0
    row = task_data["matrix"][task_data["models"].index(model)]
    return int(np.isfinite(row).sum())


def model_usage(
    model: str,
    model_result: dict[str, Any],
    prediction: Any,
    manifest: dict[str, Any],
    eval_tokens: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run type and token usage for one model, from its kept (ok) evals in the manifest rows."""
    files_by_task: dict[str, list[str]] = defaultdict(list)
    for file_meta in manifest.get("files", []):
        if file_meta.get("model") == model and file_meta.get("parse_status") == "ok":
            files_by_task[file_meta["task"]].append(file_meta["path"])

    tasks: dict[str, dict[str, Any]] = {}
    for task, task_result in model_result["tasks"].items():
        observed = _observed_items(prediction, model, task)
        if observed == 0:
            continue
        subset_items = int(task_result["subset_items"])
        population_items = int(task_result["population_items"])
        entry: dict[str, Any] = {
            "run": _task_run(observed, subset_items, population_items),
            "observed_items": observed,
            "subset_items": subset_items,
            "population_items": population_items,
        }
        usages = [eval_tokens.get(path) for path in files_by_task.get(task, [])]
        if usages and all(usages):
            totals = {key: sum(u[key] for u in usages) for key in TOKEN_KEYS}
            if totals["model_tokens"] > 0:
                entry.update(totals)
        tasks[task] = entry

    runs = {entry["run"] for entry in tasks.values() if _scope_differs(entry)}
    run_type = runs.pop() if len(runs) == 1 else ("mixed" if runs else None)

    priced = {task: e for task, e in tasks.items() if "model_tokens" in e}
    summary: dict[str, Any] = {
        "run_type": run_type,
        "observed_items": sum(e["observed_items"] for e in tasks.values()),
        "tasks": tasks,
        "tokens": None,
    }
    if priced:
        core_tokens = full_tokens = 0.0
        for e in priced.values():
            total = e["model_tokens"]
            rate = total / e["observed_items"]
            core_tokens += total if e["run"] == "core" else rate * e["subset_items"]
            full_tokens += total if e["run"] == "full" else rate * e["population_items"]
        summary["tokens"] = {
            "tasks": len(priced),
            "measured": sum(e["model_tokens"] for e in priced.values()),
            "input": sum(e["input_tokens"] for e in priced.values()),
            "output": sum(e["output_tokens"] for e in priced.values()),
            "reasoning": sum(e["reasoning_tokens"] for e in priced.values()),
            "judge": sum(e["judge_tokens"] for e in priced.values()),
            "core": round(core_tokens),
            "full": round(full_tokens),
            "reduction": 1.0 - core_tokens / full_tokens if full_tokens else None,
        }
    return summary


def suite_efficiency(models: dict[str, dict[str, Any]], subset_size: int, cost: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Panel-wide core vs full cost, from cost.json's tokens per item of each task
    (average input plus median output across models)."""
    sizes: dict[str, tuple[int, int]] = {}
    for model_result in models.values():
        for task, e in (model_result.get("usage") or {}).get("tasks", {}).items():
            sizes[task] = (e["subset_items"], e["population_items"])

    tasks = {}
    for task in sorted(sizes):
        subset_items, population_items = sizes[task]
        dist = cost.get(task)
        per_item = (dist["input_avg"] + dist["output_median"]) * VARIANTS_PER_BASIS.get(task, 1) if dist else None
        tasks[task] = {
            "subset_items": subset_items,
            "population_items": population_items,
            "tokens_per_item_median": per_item,
        }
    priced = [t for t in tasks.values() if t["tokens_per_item_median"] is not None]
    core = sum(t["tokens_per_item_median"] * t["subset_items"] for t in priced)
    full = sum(t["tokens_per_item_median"] * t["population_items"] for t in priced)
    run_types: dict[str, int] = defaultdict(int)
    for model_result in models.values():
        run_types[(model_result.get("usage") or {}).get("run_type") or "none"] += 1
    return {
        "subset_items": subset_size,
        "population_items": sum(t["population_items"] for t in tasks.values()),
        "core_tokens_median": round(core),
        "full_tokens_median": round(full),
        "token_reduction": 1.0 - core / full if full else None,
        "unpriced_tasks": sorted(task for task, t in tasks.items() if t["tokens_per_item_median"] is None),
        "run_types": dict(sorted(run_types.items())),
        "tasks": tasks,
    }
