"""Package COMPL-AI predictions, labels and ground truth as static JSON for the frontend.

This tool does no method work: every score comes from ``complai.predict`` and
every label from ``complai.labels``. It only reshapes and writes files.
"""

import typer
import json
import collections
from typing import Annotated, Any, Optional
from pathlib import Path

from complai._cli.utils import default_subset_dir
from complai.labels import UNKNOWN_LABEL, Subcategories, load_labels
from complai.predict import gap_reason, predict_detailed
from complai.utils.log_parser import records_manifest_path

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


def load_manifest(records_path: Path) -> dict:
    path = records_manifest_path(records_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def newest_eval(manifest: dict, model: str) -> str:
    return max(
        (f.get("created", "") for f in manifest.get("files", []) if f.get("model") == model and f.get("parse_status") == "ok"),
        default="",
    )


def merge_models(input_paths: list[Path], results: list[dict], manifests: list[dict]) -> tuple[dict[str, dict], dict[str, int]]:
    """Take each model from the file with its newest kept eval (later file on ties); return models and source file index."""
    sources: dict[str, list[int]] = collections.defaultdict(list)
    for i, result in enumerate(results):
        for model in result["models"]:
            sources[model].append(i)
    winners = {
        model: max(files, key=lambda i: (newest_eval(manifests[i], model), i))
        for model, files in sources.items()
    }
    for model, files in sources.items():
        if len(files) > 1:
            ran = {i: results[i]["models"][model]["coverage"]["tasks_completed"] for i in files}
            ignored = ", ".join(f"{input_paths[i]} ({ran[i]} tasks)" for i in files if i != winners[model])
            print(f"Note: {model} is in {len(files)} files; using {input_paths[winners[model]]} "
                  f"({ran[winners[model]]} tasks), ignoring {ignored}")
    return {model: results[i]["models"][model] for model, i in winners.items()}, winners


def mark_missing_reasons(models: dict[str, dict], winners: dict[str, int], manifests: list[dict]) -> None:
    """Add ``missing_reason`` to tasks a model has no responses for: ``failed`` if its eval errored, else ``not_run``."""
    failed = {
        (i, f.get("model"), f.get("task"))
        for i, manifest in enumerate(manifests)
        for f in manifest.get("files", [])
        if f.get("parse_status") in {"excluded", "error"}
    }
    for model, model_data in models.items():
        for task, task_data in model_data["tasks"].items():
            if task_data["status"] in {"imputed", "missing"}:
                task_data["missing_reason"] = "failed" if (winners[model], model, task) in failed else "not_run"


@app.command()
def export(
    input_paths: Annotated[Optional[list[Path]], typer.Argument(help="Preprocessed samples.jsonl files (default: tools/minify/data/samples.jsonl); a model in several files is taken from the one with its newest eval")] = None,
    params_path: Annotated[Path, typer.Option(help="Path to core params.json (default: the subset named in src/complai/data/config.json)")] = SUBSET_DIR / "params.json",
    subset_path: Annotated[Path, typer.Option(help="Path to core subset.jsonl (default: the subset named in src/complai/data/config.json)")] = SUBSET_DIR / "subset.jsonl",
    labels_dir: Annotated[Path, typer.Option(help="Directory containing item labels")] = Path("tools/label/labels"),
    predictions_out: Annotated[Path, typer.Option(help="Output JSON file for frontend predictions")] = Path("frontend/public/data/predictions.json"),
    labels_out: Annotated[Path, typer.Option(help="Output JSON file for frontend labels")] = Path("frontend/public/data/labels.json"),
    ground_truth: Annotated[bool, typer.Option(help="Export observed scores to ground_truth.json")] = True,
):
    input_paths = input_paths or [Path("tools/minify/data/samples.jsonl")]
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    print("1. Predicting scores (complai.predict)...")
    predictions = [predict_detailed(path, params_path, subset_path) for path in input_paths]
    manifests = [load_manifest(path) for path in input_paths]
    prediction = predictions[0]
    params, subset, result = prediction.params, prediction.subset, prediction.result
    result["models"], winners = merge_models(input_paths, [p.result for p in predictions], manifests)
    mark_missing_reasons(result["models"], winners, manifests)
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
    for model_id, i in winners.items():
        for file_meta in manifests[i].get("files", []):
            m_cfg = file_meta.get("model_config") or file_meta.get("eval_config")
            if file_meta.get("model") == model_id and m_cfg:
                model_configs.setdefault(model_id, {}).update(m_cfg)
    for model_id, model_data in result["models"].items():
        if model_id in model_configs:
            model_data["model_config"] = model_configs[model_id]

    print(f"5. Writing predictions to {predictions_out}...")
    with open(predictions_out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"6. Writing columnar labels to {labels_out}...")
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

    if ground_truth:
        print("7. Writing observed (ground truth) scores...")
        from tools.minify.ground_truth import load_ground_truth, masked_label_scores, masked_task_scores

        # Ground truth is a temporary debug aid; it only covers the first input file.
        truth = load_ground_truth(samples_path=input_paths[0])
        true_scores: dict[str, dict[str, Any]] = collections.defaultdict(dict)
        for task, per_model in truth.benchmark_scores.items():
            for model, score in per_model.items():
                true_scores[model][task] = score
        masked_scores = [
            masked_label_scores(prediction, truth, "index"),
            masked_label_scores(prediction, truth, "subcategory"),
            masked_task_scores(prediction, truth),
        ]
        for per_model in masked_scores:
            for model, per_key in per_model.items():
                for key, scores in per_key.items():
                    true_scores[model][key] = scores["observed_score"]
                    true_scores[model].setdefault("_masked", {})[key] = scores
        gt_out = predictions_out.parent / "ground_truth.json"
        with open(gt_out, "w") as f:
            json.dump(true_scores, f, indent=2)
        print(f"Exported ground truth to {gt_out}")

    print("Done! Artifacts successfully exported.")

if __name__ == "__main__":
    app()
