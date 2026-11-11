"""Package COMPL-AI predictions and labels as static JSON for the frontend.

This tool does no method work: every score comes from ``complai.predict`` and
every label from ``complai.labels``. It only reshapes and writes files.
"""

import typer
import json
import math
import collections
from typing import Annotated, Any, Optional
from pathlib import Path

from complai._cli.utils import default_subset_dir
from complai.labels import UNKNOWN_LABEL, Subcategories, load_labels
from complai.predict import gap_reason, predict_detailed
from complai.utils.log_parser import load_records
from complai.utils.scores import normalize_score, select_score
from tools.export_web.usage import COST_PATH, MODEL_TOKENS_PATH, load_json, model_usage, suite_efficiency

SUBSET_DIR = default_subset_dir()

app = typer.Typer(help="Export COMPL-AI predictions and labels to static JSON files for the frontend.")

def compile_labels(labels_dir: Path, task_indices: dict[str, str], subcategories: Subcategories) -> dict:
    """Per fitted task: its index and each labeled sample's sub-category and tags, sorted for deterministic output."""
    labels = load_labels(labels_dir, with_aliases=False)
    out: dict[str, dict[str, Any]] = {}
    for task in sorted(set(labels) & set(task_indices)):
        index = task_indices[task]
        out[task] = {
            "index": index,
            "samples": {
                sid: {
                    "subcategory": subcategories.of(index, labels[task][sid].secondary or UNKNOWN_LABEL),
                    "tags": list(labels[task][sid].tags),
                }
                for sid in sorted(labels[task])
            },
        }
    return out


def newest_eval(files: list[dict], model: str) -> str:
    return max(
        (f.get("created", "") for f in files if f.get("model") == model and f.get("parse_status") == "ok"),
        default="",
    )


def mark_missing_reasons(models: dict[str, dict], files: list[dict]) -> None:
    """Add ``missing_reason`` to tasks a model has no responses for: ``failed`` if its eval errored, else ``not_run``."""
    failed = {
        (f.get("model"), f.get("task"))
        for f in files
        if f.get("parse_status") in {"excluded", "error"}
    }
    for model, model_data in models.items():
        for task, task_data in model_data["tasks"].items():
            if task_data["status"] in {"imputed", "missing"}:
                task_data["missing_reason"] = "failed" if (model, task) in failed else "not_run"


def full_run_results(input_paths: list[Path], task_scorers: dict[str, str]) -> dict[tuple[str, str], dict[str, Any]]:
    """Plain scores of each model's full (calibration) run, per task.

    A full run is an eval preprocess kept (``parse_status == "ok"``) that was not
    limited to a subset (``partial`` false). Where several inputs have one for
    the same model and task, the newest is used. The score is the simple mean
    over questions of the per-question value (mean over epochs), computed the
    same way as the IRT response matrix; Inspect's own reported metric is not used.

    Metadata only: never used for predicted_score, indices or sub-categories.
    """
    chosen: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    parts = [load_records(path) for path in input_paths]
    for index, part in enumerate(parts):
        for row in part.files:
            if row.get("parse_status") != "ok" or row.get("partial") is not False:
                continue
            key = (str(row["model"]), str(row["task"]))
            if str(row["task"]) not in task_scorers:
                continue
            current = chosen.get(key)
            if current is None or str(row.get("created", "")) >= str(current[1].get("created", "")):
                chosen[key] = (index, row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for index, part in enumerate(parts):
        wanted = {str(row["path"]): key for key, (i, row) in chosen.items() if i == index}
        if not wanted:
            continue
        values: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        questions: dict[str, set[str]] = collections.defaultdict(set)
        content_by_question: dict[tuple[str, str], str] = {}
        for sample in part.iter_samples():
            file_path = str(sample["file_path"])
            if file_path not in wanted:
                continue
            task = str(sample["task"])
            content_hash = str(sample["content_hash"])
            question = str(sample.get("question_hash", content_hash))
            previous = content_by_question.setdefault((file_path, question), content_hash)
            if previous != content_hash:
                question = content_hash
            questions[file_path].add(question)
            scorer = task_scorers[task]
            raw = select_score(sample["scores"], scorer)
            value = None if raw is None else normalize_score(raw, scorer)
            if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                continue
            values[file_path][question].append(float(value))
        for file_path, key in wanted.items():
            row = chosen[key][1]
            per_question = [sum(v) / len(v) for v in values[file_path].values()]
            out[key] = {
                "score": sum(per_question) / len(per_question) if per_question else None,
                "items_answered": len(per_question),
                "population_items": len(questions[file_path]),
                "run_created": row.get("created"),
                "log_path": file_path,
            }
    return out


def sample_analysis_summary(input_paths: list[Path], prediction: Any) -> dict[str, Any] | None:
    """Per model and task: how the model's answers to the core questions went, from ``complai analyze``.

    Counts cover exactly the answers the prediction used: the core questions
    (``prediction.observations``), matched to the model's own answers by
    question_hash the same way predict matches them, in the same eval. This
    holds whether that eval was a core run or a full (calibration) run.
    Every epoch counts as one answer; strong_reject jailbreak variants of one
    core question each count as an answer.

    Metadata only: never used for predicted_score or indices.
    """
    from tools.analyze.analyze import OUTCOMES, RULES_VERSION, TAGS, analysis_path

    missing = [path for path in input_paths if not analysis_path(path).is_file()]
    if missing:
        print(f"   Skipped: no sample analysis next to {', '.join(map(str, missing))}; run `complai analyze` first.")
        return None

    core_questions: dict[tuple[str, str], set[str]] = {}
    subset_rows = {str(row["item_id"]): row for row in prediction.subset}
    for (model, task), observation in prediction.observations.items():
        question_of: dict[str, str] = {}
        for item in prediction.tasks[task]["items"]:
            question_of[str(item["item_id"])] = str(item["question_hash"])
            question_of[str(item["question_hash"])] = str(item["question_hash"])
        core_questions[(model, task)] = {
            question_of[str(subset_rows[item_id].get("question_hash", item_id))] for item_id in observation.item_ids
        }

    # Same question identity as complai.irt.prepare_tasks: when two different questions
    # share a question_hash, the first scored one (in record order) keeps it and the
    # other uses its content_hash.
    scorers = prediction.params["task_scorers"]
    answers: list[tuple[str, str, str, str, str, str]] = []
    first_content: dict[tuple[str, str], str] = {}
    for sample in load_records(input_paths).iter_samples():
        task = str(sample["task"])
        if task not in scorers:
            continue
        content = str(sample["content_hash"])
        question = str(sample.get("question_hash", content))
        answers.append((str(sample["file_path"]), str(sample["sample_id"]), str(sample["model"]), task, question, content))
        if "score" in sample:
            value = float(sample["score"])
        else:
            scores = sample["scores"] if "scores" in sample else json.loads(sample["scores_json"])
            raw = select_score(scores, scorers[task])
            value = None if raw is None else normalize_score(raw, scorers[task])
        if value is not None and math.isfinite(value) and 0.0 <= value <= 1.0:
            first_content.setdefault((task, question), content)

    # (file_path, sample_id) -> question, for the answers the prediction used.
    question_by_answer: dict[tuple[str, str], str] = {}
    for file_path, sample_id, model, task, question, content in answers:
        wanted = core_questions.get((model, task))
        if not wanted:
            continue
        if first_content.get((task, question), content) != content:
            question = content
        if question in wanted:
            question_by_answer[(file_path, sample_id)] = question

    counts: dict[tuple[str, str], dict[str, Any]] = {}
    seen_rules: set[int] = set()
    for path in input_paths:
        with open(analysis_path(path)) as handle:
            for line in handle:
                row = json.loads(line)
                question = question_by_answer.get((row["file_path"], row["sample_id"]))
                if question is None:
                    continue
                seen_rules.add(int(row["rules_version"]))
                entry = counts.setdefault((row["model"], row["task"]), {
                    "answers": 0, "questions": set(), "tags": collections.Counter(), "log_paths": set(),
                })
                entry["answers"] += 1
                entry["questions"].add(question)
                entry["tags"][row["tag"]] += 1
                entry["log_paths"].add(row["file_path"])
    if seen_rules - {RULES_VERSION}:
        print(f"   Warning: sample analysis made with rules version(s) {sorted(seen_rules)}, current is {RULES_VERSION}; rerun `complai analyze`.")

    models: dict[str, dict[str, Any]] = {}
    incomplete = []
    for (model, task), wanted in sorted(core_questions.items()):
        entry = counts.get((model, task))
        found = len(entry["questions"]) if entry else 0
        if found != len(wanted):
            incomplete.append(f"{model} / {task}: {found} of {len(wanted)} core questions")
        if not entry:
            continue
        models.setdefault(model, {})[task] = {
            "answers": entry["answers"],
            "questions": found,
            "tags": {tag: entry["tags"][tag] for tag in TAGS if entry["tags"][tag]},
            "log_path": sorted(entry["log_paths"])[0] if len(entry["log_paths"]) == 1 else sorted(entry["log_paths"]),
        }
    for line in incomplete:
        print(f"   Warning: sample analysis is missing answers for {line}; rerun `complai analyze`.")
    model_outcomes, task_outcomes = outcome_shares(models)
    return {
        "rules_version": RULES_VERSION,
        "scope": "Per model and task, the model's answers to the core questions that its predicted score is based on "
                 "(from its core run, or the core questions within its full run). Each epoch is one answer. "
                 "Descriptive metadata only: not used for scores or indices.",
        "tags": TAGS,
        "outcomes": {
            "definition": OUTCOMES,
            "scope": "outcomes: share of the answers in each group (sums to 1 up to rounding). "
                     "model_outcomes: per model, the mean of its per-task shares, each task weighted equally. "
                     "task_outcomes: per task, the mean of the per-model shares, each model weighted equally.",
        },
        "models": models,
        "model_outcomes": model_outcomes,
        "task_outcomes": task_outcomes,
    }


def outcome_shares(models: dict[str, dict[str, dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add ``outcomes`` (share of answers per OUTCOMES group) to every model and task entry, in place.

    Returns the per-model rollup (mean over its tasks) and the per-task rollup
    (mean over its models), each entry weighted equally. Shares are computed from
    the unrounded per-entry values and rounded to 4 decimals at the end.
    """
    from tools.analyze.analyze import OUTCOMES

    exact: dict[tuple[str, str], dict[str, float]] = {}
    for model, tasks in models.items():
        for task, entry in tasks.items():
            exact[(model, task)] = {
                outcome: sum(entry["tags"].get(tag, 0) for tag in group["tags"]) / entry["answers"]
                for outcome, group in OUTCOMES.items()
            }
            entry["outcomes"] = {outcome: round(share, 4) for outcome, share in exact[(model, task)].items()}

    def mean(shares: list[dict[str, float]]) -> dict[str, float]:
        return {outcome: round(sum(share[outcome] for share in shares) / len(shares), 4) for outcome in OUTCOMES}

    by_model: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    by_task: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    for (model, task), shares in exact.items():
        by_model[model].append(shares)
        by_task[task].append(shares)
    return (
        {model: {"tasks": len(shares), "outcomes": mean(shares)} for model, shares in sorted(by_model.items())},
        {task: {"models": len(shares), "outcomes": mean(shares)} for task, shares in sorted(by_task.items())},
    )


SATURATED_MIDDLE_MEAN = 0.95
SATURATED_TOP_MEAN = 0.90
SCORE_EDGES = [round(0.1 * step, 1) for step in range(11)]
DISCRIMINATION_EDGES = [round(0.5 * step, 1) for step in range(11)]


def histogram(values: list[float], edges: list[float]) -> list[dict[str, Any]]:
    """Counts per bin. Bins are [low, high) except the last one, which includes its high edge."""
    bins: list[dict[str, Any]] = [
        {"low": low, "high": high, "count": 0} for low, high in zip(edges[:-1], edges[1:])
    ]
    for value in values:
        if value < edges[0] or value > edges[-1]:
            raise ValueError(f"Value {value} is outside the histogram range {edges[0]}..{edges[-1]}")
        position = next((i for i, high in enumerate(edges[1:]) if value < high), len(bins) - 1)
        bins[position]["count"] += 1
    return bins


def saturation(scores: list[float]) -> tuple[Optional[str], Optional[dict[str, float]]]:
    """Split the models into thirds by their score on the task and label how close each third is to 100%.

    ``saturated``: the middle third averages at least SATURATED_MIDDLE_MEAN, so only the
    weakest models still separate. ``saturated_at_top``: the top third averages at least
    SATURATED_TOP_MEAN (but the middle third does not reach SATURATED_MIDDLE_MEAN), so the
    best models no longer separate while the others still do. None with fewer than 3 models.
    """
    if len(scores) < 3:
        return None, None
    ranked = sorted(scores, reverse=True)
    size = len(ranked) // 3
    thirds = {"top": ranked[:size], "middle": ranked[size:len(ranked) - size], "bottom": ranked[len(ranked) - size:]}
    means = {name: sum(values) / len(values) for name, values in thirds.items()}
    if means["middle"] >= SATURATED_MIDDLE_MEAN:
        label = "saturated"
    elif means["top"] >= SATURATED_TOP_MEAN:
        label = "saturated_at_top"
    else:
        label = "not_saturated"
    return label, {name: round(mean, 4) for name, mean in means.items()}


def benchmark_health(prediction: Any) -> dict[str, Any]:
    """Per task: how spread out the models' scores are, how saturated the task is, and how discriminating its core questions are.

    Scores are each model's predicted_score for the task, over the models that
    answered at least one core question of it (status ``ok`` or ``partial``; models
    whose score is imputed from other tasks are left out). Item parameters come
    from params.json for the core (subset) questions only.

    Descriptive metadata only: not used for scores or indices.
    """
    params, result = prediction.params, prediction.result
    items_by_id = {str(item["item_id"]): item for item in params["items"]}
    core_items: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in prediction.subset:
        item = items_by_id[str(row["item_id"])]
        core_items[str(row["item_id"]).split("::", 1)[0]].append(item)

    tasks: dict[str, Any] = {}
    for index_name, index in sorted(params["indices"].items()):
        for task, fit in sorted(index["tasks"].items()):
            scores = sorted(
                (
                    float(model["tasks"][task]["predicted_score"])
                    for model in result["models"].values()
                    if task in model["tasks"]
                    and model["tasks"][task].get("status") in {"ok", "partial"}
                    and model["tasks"][task].get("predicted_score") is not None
                ),
                reverse=True,
            )
            label, third_means = saturation(scores)
            items = core_items.get(task, [])
            tasks[task] = {
                "index": index_name,
                "models": len(scores),
                "score_histogram": histogram(scores, SCORE_EDGES),
                "saturation": label,
                "third_means": third_means,
                "benchmark_discrimination": round(float(fit["discrimination"]), 4),
                "core_items": len(items),
                "item_discrimination_histogram": histogram(
                    [float(item["discrimination"]) for item in items], DISCRIMINATION_EDGES,
                ),
            }
    return {
        "params_id": params["params_id"],
        "subset_id": params["subset_id"],
        "scope": "Descriptive metadata only: not used for scores or indices.",
        "definitions": {
            "score_histogram": "Number of models per band of predicted score on this task (0 to 1), counting "
                               "the models that answered at least one of its core questions.",
            "saturation": "Models are split into thirds by their predicted score on this task. 'saturated': "
                          f"the middle third averages at least {SATURATED_MIDDLE_MEAN:.0%}, so only the weakest "
                          "models still separate. 'saturated_at_top': the top third averages at least "
                          f"{SATURATED_TOP_MEAN:.0%} (and the middle third is below {SATURATED_MIDDLE_MEAN:.0%}), "
                          "so the best models no longer separate but the others still do. 'not_saturated' "
                          "otherwise; null with fewer than 3 models.",
            "third_means": "Mean predicted score of the top, middle and bottom third of models on this task "
                           "(with a model count not divisible by 3, the middle third takes the extra models).",
            "benchmark_discrimination": "How sharply the task score rises with the index ability (2PL slope of "
                                        "the task's score within its index). Higher means the task separates "
                                        "models on that index more sharply; near 0 means it barely does.",
            "item_discrimination_histogram": "Core questions per band of discrimination (2PL slope). Higher means "
                                             "the question separates stronger from weaker models more sharply.",
            "bins": "Each bin covers low <= value < high; the last bin also includes its high edge.",
        },
        "tasks": tasks,
    }


@app.command()
def export(
    input_paths: Annotated[Optional[list[Path]], typer.Argument(help="Preprocessed samples.jsonl files (default: tools/minify/data/samples.jsonl); they are read together, and where several have an eval of the same model and task, the newest one is used")] = None,
    params_path: Annotated[Path, typer.Option(help="Path to core params.json (default: the subset named in src/complai/data/config.json)")] = SUBSET_DIR / "params.json",
    subset_path: Annotated[Path, typer.Option(help="Path to core subset.jsonl (default: the subset named in src/complai/data/config.json)")] = SUBSET_DIR / "subset.jsonl",
    labels_dir: Annotated[Path, typer.Option(help="Directory containing item labels")] = Path("tools/label/labels"),
    predictions_out: Annotated[Path, typer.Option(help="Output JSON file for frontend predictions")] = Path("frontend/public/data/predictions.json"),
    labels_out: Annotated[Path, typer.Option(help="Output JSON file for frontend labels")] = Path("frontend/public/data/labels.json"),
    sample_analysis_out: Annotated[Path, typer.Option(help="Output JSON file for per model and task failure-tag counts on the core questions (needs `complai analyze` output next to each input)")] = Path("frontend/public/data/sample_analysis.json"),
    benchmark_health_out: Annotated[Path, typer.Option(help="Output JSON file for per task score spread, saturation and core-question discrimination")] = Path("frontend/public/data/benchmark_health.json"),
):
    input_paths = input_paths or [Path("tools/minify/data/samples.jsonl")]
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    print("1. Predicting scores (complai.predict)...")
    prediction = predict_detailed(input_paths, params_path, subset_path)
    files = list(load_records(input_paths).files)
    params, subset, result = prediction.params, prediction.subset, prediction.result
    mark_missing_reasons(result["models"], files)
    result["subset"] = params_path.parent.name

    print("2. Compiling label dataset for the frontend...")
    subcategories = Subcategories.from_params(params["labels"]["subcategories"])
    task_indices = {name: task["index"] for name, task in params["tasks"].items()}
    full_labels = compile_labels(labels_dir, task_indices, subcategories)

    print("3. Summarising subset composition...")
    items_by_id = {str(item["item_id"]): item for item in params["items"]}
    subset_index_counts = collections.Counter(
        items_by_id[str(row["item_id"])]["index"] for row in subset
    )
    schema_obj: dict[str, Any] = {
        "indices": {
            index: {
                "subset_weight_pct": f"{count / len(subset) * 100:.1f}%",
                "subset_count": count,
            }
            for index, count in sorted(subset_index_counts.items())
        },
        "total_subset_items": len(subset),
    }
    subset_subcategory_counts = collections.Counter(
        items_by_id[str(row["item_id"])]["subcategory"] for row in subset
    )
    schema_obj["subcategories"] = {
        tag: {
            "name": definition["label_name"],
            "index": definition["core_index"],
            "description": definition["description"],
            "subset_count": subset_subcategory_counts[tag],
            "gap": gap_reason(subset_subcategory_counts[tag]),
        }
        for tag, definition in sorted(subcategories.definitions.items())
    }

    print("4. Attaching run configurations from manifest...")
    model_configs: dict[str, dict] = {}
    for file_meta in files:
        model_id = file_meta.get("model")
        if model_id in result["models"]:
            m_cfg = file_meta.get("model_config") or file_meta.get("eval_config")
            if m_cfg:
                model_configs.setdefault(model_id, {}).update(m_cfg)
    for model_id, model_data in result["models"].items():
        if model_id in model_configs:
            model_data["model_config"] = model_configs[model_id]
        model_data["last_evaluated"] = newest_eval(files, model_id) or None

    print("4b. Attaching full-run (calibration) scores as metadata...")
    # Metadata only: added after prediction; never used for predicted_score or indices.
    full_runs = full_run_results(input_paths, params["task_scorers"])
    for model_id, model_data in result["models"].items():
        for task, task_data in model_data["tasks"].items():
            if (model_id, task) in full_runs:
                task_data["full_run"] = full_runs[(model_id, task)]

    print("4c. Summarising sample analysis (failure tags) for the core questions...")
    analysis = sample_analysis_summary(input_paths, prediction)

    print("4d. Summarising benchmark health (score spread, saturation, item discrimination)...")
    health = benchmark_health(prediction)

    print("5. Attaching token usage and run type per model...")
    eval_tokens = load_json(MODEL_TOKENS_PATH)
    for model_id, model_data in result["models"].items():
        model_data["usage"] = model_usage(model_id, model_data, prediction, {"files": files}, eval_tokens)
    result["efficiency"] = suite_efficiency(result["models"], len(subset), load_json(COST_PATH))

    print(f"6. Writing predictions to {predictions_out}...")
    with open(predictions_out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"7. Writing columnar labels to {labels_out}...")
    output_str = "{\n"
    output_str += '  "_schema": ' + json.dumps(schema_obj, indent=4).replace('\n', '\n  ') + ',\n'
    for i, (task, entry) in enumerate(full_labels.items()):
        output_str += f'  "{task}": {{\n'
        output_str += f'    "index": "{entry["index"]}",\n'
        output_str += '    "columns": ["sample_id", "subcategory", "tags"],\n'
        rows = [json.dumps([sid, data["subcategory"], data["tags"]], separators=(",", ":")) for sid, data in entry["samples"].items()]
        output_str += '    "data": [\n      ' + ',\n      '.join(rows) + '\n    ]\n  }'
        output_str += ",\n" if i < len(full_labels) - 1 else "\n"
    output_str += "}\n"
    
    with open(labels_out, "w", encoding="utf-8") as f:
        f.write(output_str)

    if analysis is not None:
        print(f"8. Writing sample analysis to {sample_analysis_out}...")
        sample_analysis_out.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_analysis_out, "w") as f:
            json.dump(analysis, f, indent=2)

    print(f"9. Writing benchmark health to {benchmark_health_out}...")
    benchmark_health_out.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_health_out, "w") as f:
        json.dump(health, f, indent=2)

    print("Done! Artifacts successfully exported.")

if __name__ == "__main__":
    app()
