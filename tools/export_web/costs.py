"""Tokens and eval cost per model, for ``frontend/public/data/costs.json``.

How the numbers are made:

1. ``complai minify preprocess`` reads each .eval file's ``header.json`` and stores the
   tokens Inspect logged (``stats.model_usage``) in that file's manifest row as ``tokens``:
   ``input``, ``output``, ``reasoning`` and ``model`` (total) for the evaluated model, and
   ``judge`` for every other model (judges, graders).
2. ``model_usage`` sums those per model and task. The scope the model actually ran (core or
   full) is measured; the other one is estimated from the measured tokens per question.
3. ``build_costs`` multiplies the input and output tokens by the list rates in
   ``prices.json`` (USD per 1M tokens, never batch prices) to get USD.

Judge tokens are reported but left out of the core vs full comparison. Nothing here reads
the logs.
"""

import datetime as dt
import json
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats


# Token logs are per executed sample; a strong_reject basis item runs all its jailbreak variants.
VARIANTS_PER_BASIS = {"strong_reject": 35}

PRICES_PATH = Path(__file__).with_name("prices.json")
MODEL_CARDS_DIR = Path("frontend/public/models")
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# A task counts as a full run once the model answered this share of its population
# (some tasks lose items when scoring, e.g. mmlu_pro_robustness keeps 75-85%).
FULL_RUN_SHARE = 0.5
TREND_EPOCH = dt.date(2024, 1, 1)

TOKEN_KEYS = ("input_tokens", "output_tokens", "reasoning_tokens", "model_tokens", "judge_tokens")


# ---------------------------------------------------------------- prices.json


def load_prices(path: Path = PRICES_PATH) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def refresh_prices(path: Path = PRICES_PATH) -> None:
    """Update the rates whose ``source`` is ``"openrouter"`` from OpenRouter's model list.

    Entries with any other source (hand-entered rates) are never touched.
    """
    prices = load_prices(path)
    with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as response:
        listing = {m["id"]: m["pricing"] for m in json.load(response)["data"]}
    today = dt.date.today().isoformat()
    for model, rate in prices["rates"].items():
        if rate.get("source") != "openrouter":
            continue
        pricing = listing.get(rate["slug"])
        if pricing is None:
            print(f"   {model}: {rate['slug']} is no longer listed on OpenRouter; kept {rate['date']} rate")
            continue
        new_in = round(float(pricing["prompt"]) * 1e6, 6)
        new_out = round(float(pricing["completion"]) * 1e6, 6)
        if (new_in, new_out) != (rate["in"], rate["out"]):
            print(f"   {model}: {rate['in']}/{rate['out']} -> {new_in}/{new_out} USD per 1M tokens")
        rate.update({"in": new_in, "out": new_out, "date": today})
    with open(path, "w") as f:
        json.dump(prices, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------- tokens per model


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


def _has_tokens(row: dict[str, Any]) -> bool:
    tokens = row.get("tokens")
    return bool(tokens) and tokens["model"] + tokens["judge"] > 0


def model_usage(model: str, model_result: dict[str, Any], prediction: Any, files: list[dict[str, Any]]) -> dict[str, Any]:
    """Run type and token usage for one model, from its kept (ok) evals in the manifest rows."""
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in files:
        if row.get("model") == model and row.get("parse_status") == "ok":
            rows_by_task[row["task"]].append(row)

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
        rows = rows_by_task.get(task, [])
        if rows and all(_has_tokens(row) for row in rows):
            totals = {f"{key}_tokens": sum(row["tokens"][key] for row in rows) for key in ("input", "output", "reasoning", "model", "judge")}
            if totals["model_tokens"] > 0:
                entry.update(totals)
        tasks[task] = entry

    runs = {entry["run"] for entry in tasks.values() if _scope_differs(entry)}
    run_type = runs.pop() if len(runs) == 1 else ("mixed" if runs else None)

    priced = {task: e for task, e in tasks.items() if "model_tokens" in e}
    summary: dict[str, Any] = {"run_type": run_type, "tasks": tasks, "tokens": None}
    if priced:
        core_tokens = full_tokens = 0.0
        for e in priced.values():
            total = e["model_tokens"]
            rate = total / e["observed_items"]
            core_tokens += total if e["run"] == "core" else rate * e["subset_items"]
            full_tokens += total if e["run"] == "full" else rate * e["population_items"]
        summary["tokens"] = {
            "measured": sum(e["model_tokens"] for e in priced.values()),
            "input": sum(e["input_tokens"] for e in priced.values()),
            "output": sum(e["output_tokens"] for e in priced.values()),
            "reasoning": sum(e["reasoning_tokens"] for e in priced.values()),
            "judge": sum(e["judge_tokens"] for e in priced.values()),
            "core": round(core_tokens),
            "full": round(full_tokens),
        }
    return summary


# ---------------------------------------------------------------- suite-wide tokens


def task_tokens_per_item(files: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Tokens per question of each task, from the full runs in ``logs/``: average input
    over all models, and the median over models of each model's output per question."""
    totals: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"input": 0, "output": 0, "samples": 0}))
    for row in files:
        tokens = row.get("tokens")
        if not str(row.get("path", "")).startswith("logs/") or not tokens or tokens["model"] <= 0:
            continue
        samples = int(row.get("sample_count") or 0) // int(row.get("epochs") or 1)
        if samples <= 0:
            continue
        acc = totals[row["task"]][row["model"]]
        acc["input"] += tokens["input"]
        acc["output"] += tokens["output"]
        acc["samples"] += samples
    out = {}
    for task, by_model in totals.items():
        samples = sum(m["samples"] for m in by_model.values())
        out[task] = {
            "input_avg": sum(m["input"] for m in by_model.values()) / samples,
            "output_median": statistics.median(m["output"] / m["samples"] for m in by_model.values()),
        }
    return out


def suite_efficiency(usages: dict[str, dict[str, Any]], subset_size: int, files: list[dict[str, Any]]) -> dict[str, Any]:
    """Panel-wide core vs full tokens, from each task's tokens per question (average input
    plus median output across models)."""
    per_item_tokens = task_tokens_per_item(files)
    sizes: dict[str, tuple[int, int]] = {}
    for usage in usages.values():
        for task, e in usage["tasks"].items():
            sizes[task] = (e["subset_items"], e["population_items"])

    tasks = {}
    for task in sorted(sizes):
        subset_items, population_items = sizes[task]
        dist = per_item_tokens.get(task)
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
    for usage in usages.values():
        run_types[usage["run_type"] or "none"] += 1
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


# ---------------------------------------------------------------- costs.json


def load_model_cards(cards_dir: Path = MODEL_CARDS_DIR) -> dict[str, dict[str, Any]]:
    """Model card per eval model ID (a card lists its IDs under ``evals``)."""
    cards = {}
    for path in sorted(cards_dir.glob("*.yaml")):
        card = yaml.safe_load(path.read_text()) or {}
        for model in card.get("evals") or []:
            cards[model] = card
    return cards


def _scope_tokens(usage: dict[str, Any]) -> dict[str, float]:
    """Input and output tokens for a core and a full run, measured for the scope run and
    estimated per question for the other."""
    out = {"core_in": 0.0, "core_out": 0.0, "full_in": 0.0, "full_out": 0.0}
    for e in usage["tasks"].values():
        if "model_tokens" not in e:
            continue
        for scope, items in (("core", e["subset_items"]), ("full", e["population_items"])):
            share = 1.0 if e["run"] == scope else items / e["observed_items"]
            out[f"{scope}_in"] += e["input_tokens"] * share
            out[f"{scope}_out"] += e["output_tokens"] * share
    return out


def _years(day: str) -> float:
    return (dt.date.fromisoformat(str(day)) - TREND_EPOCH).days / 365.25


def _trend(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    fit = stats.linregress([_years(r["release"]) for r in rows], np.log10([r[key] for r in rows]))
    return {
        "log10_intercept": round(float(fit.intercept), 4),
        "log10_per_year": round(float(fit.slope), 4),
        "x_per_year": round(float(10**fit.slope), 2),
    }


def _size_control(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Yearly change in full-run USD with and without controlling for total parameters
    (OLS of log10 USD on years and log10 total parameters)."""
    rows = [r for r in rows if r.get("total_params")]
    if len(rows) < 4:
        return None
    years = np.array([_years(r["release"]) for r in rows])
    y = np.log10([r["full_usd"] for r in rows])
    X = np.column_stack([np.ones_like(years), years, np.log10([float(r["total_params"]) for r in rows])])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    n, k = X.shape
    resid = y - X @ b
    cov = (resid @ resid / (n - k)) * np.linalg.inv(X.T @ X)
    p_year = 2 * stats.t.sf(abs(b[1] / np.sqrt(cov[1, 1])), n - k)
    return {
        "n": n,
        "x_per_year_raw": round(float(10 ** stats.linregress(years, y).slope), 2),
        "x_per_year": round(float(10 ** b[1]), 2),
        "p_year": round(float(p_year), 4),
        "per_10x_total": round(float(10 ** b[2]), 2),
    }


def _spread(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median": round(statistics.median(values), 4) if values else None,
        "min": round(min(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def _task_entry(e: dict[str, Any]) -> dict[str, Any]:
    """One task of a model as written: run scope, questions answered and tokens (sizes are in ``suite.tasks``)."""
    return {key: e[key] for key in ("run", "observed_items", *TOKEN_KEYS) if key in e}


def build_costs(
    models: dict[str, dict[str, Any]],
    prediction: Any,
    files: list[dict[str, Any]],
    subset_size: int,
    prices: dict[str, Any],
    cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Per model run type, tokens per task, core and full tokens and USD, plus the panel summary and trends."""
    usages = {model: model_usage(model, model_result, prediction, files) for model, model_result in models.items()}
    efficiency = suite_efficiency(usages, subset_size, files)
    rates = prices["rates"]
    out_models: dict[str, dict[str, Any]] = {}
    for model, usage in sorted(usages.items()):
        tokens = usage["tokens"] or {}
        card = cards.get(model, {})
        row: dict[str, Any] = {
            "release": card.get("release_date"),
            "run_type": usage["run_type"],
            "measured_tokens": tokens.get("measured"),
            "input_tokens": tokens.get("input"),
            "output_tokens": tokens.get("output"),
            "reasoning_tokens": tokens.get("reasoning"),
            "judge_tokens": tokens.get("judge"),
            "core_tokens": tokens.get("core"),
            "full_tokens": tokens.get("full"),
            "ratio_tokens": round(tokens["full"] / tokens["core"], 4) if tokens.get("core") else None,
            "core_usd": None,
            "full_usd": None,
            "ratio_usd": None,
            "rate": None,
            "tasks": {task: _task_entry(e) for task, e in sorted(usage["tasks"].items())},
        }
        rate = rates.get(model)
        if rate and tokens:
            scope = _scope_tokens(usage)
            core_usd = (scope["core_in"] * rate["in"] + scope["core_out"] * rate["out"]) / 1e6
            full_usd = (scope["full_in"] * rate["in"] + scope["full_out"] * rate["out"]) / 1e6
            row.update({
                "core_usd": round(core_usd, 2),
                "full_usd": round(full_usd, 2),
                "ratio_usd": round(full_usd / core_usd, 4) if core_usd else None,
                "rate": {key: rate[key] for key in ("in", "out", "source", "date")},
            })
        out_models[model] = row

    priced = [
        {**{k: v for k, v in r.items() if k != "tasks"}, "id": m, "total_params": (cards.get(m, {}).get("specs") or {}).get("total_params")}
        for m, r in out_models.items()
        if r["core_usd"]
    ]
    dated = [r for r in priced if r["release"]]
    totals = {key: sum(r[key] for r in priced) for key in ("core_usd", "full_usd", "core_tokens", "full_tokens")}
    rate_dates = sorted({r["rate"]["date"] for r in priced})
    return {
        "_meta": {
            "description": "Tokens and list-price USD of a core run and a full run of every model. The scope a model actually ran is measured from its logs; the other scope is estimated from its measured tokens per question.",
            "generator": "complai export-web (tools/export_web/costs.py)",
            "prices": "tools/export_web/prices.json: list rates in USD per 1M tokens, never batch prices",
            "rate_dates": rate_dates,
            "tokens": "Evaluated model only (input + output, reasoning included in output); judge tokens are listed separately and not counted.",
            "run_type": "core: answered only the core questions; full: answered the full benchmarks; mixed: some of each.",
            "tasks": "Per model and benchmark: run (core, full or partial), observed_items (questions answered) and the tokens measured in its logs. Benchmark sizes are in suite.tasks.",
            "suite": "Panel-wide core vs full tokens from each benchmark's tokens per question (average input plus median output across models).",
            "ratio": "full / core of the same model.",
            "trend": f"Least-squares fit of log10(value) on years since {TREND_EPOCH.isoformat()} (model release date), over the priced models.",
            "size_control": "full_usd fit on years and log10(total parameters); x_per_year_raw is the fit on years alone over the same models.",
        },
        "summary": {
            "ratio_usd": _spread([r["ratio_usd"] for r in priced]),
            "ratio_tokens": _spread([r["ratio_tokens"] for r in out_models.values() if r["ratio_tokens"]]),
            "totals": {key: round(value, 2) if key.endswith("usd") else value for key, value in totals.items()},
            "pooled_ratio_usd": round(totals["full_usd"] / totals["core_usd"], 4) if priced else None,
            "pooled_ratio_tokens": round(totals["full_tokens"] / totals["core_tokens"], 4) if priced else None,
            "unpriced": sorted(m for m, r in out_models.items() if r["core_usd"] is None),
        },
        "suite": efficiency,
        "trend": {
            "epoch": TREND_EPOCH.isoformat(),
            "n": len(dated),
            **{key: _trend(dated, key) for key in ("core_usd", "full_usd", "core_tokens", "full_tokens")},
        } if len(dated) >= 2 else None,
        "size_control": _size_control(dated),
        "models": out_models,
    }
