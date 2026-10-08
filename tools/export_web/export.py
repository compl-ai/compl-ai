"""Package COMPL-AI predictions, labels and ground truth as static JSON for the frontend.

This tool does no method work: every score comes from ``complai.predict`` and
every label from ``complai.labels``. It only reshapes and writes files.
"""

import typer
import json
import collections
from typing import Annotated, Any
from pathlib import Path

from complai.labels import UNKNOWN_LABEL, Subcategories, load_labels
from complai.predict import gap_reason, predict_detailed

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


@app.command()
def export(
    input_path: Annotated[Path, typer.Argument(help="Preprocessed samples.jsonl")] = Path("tools/minify/data/samples.jsonl"),
    params_path: Annotated[Path, typer.Option(help="Path to core params.json")] = Path("src/complai/data/core.v1/params.json"),
    subset_path: Annotated[Path, typer.Option(help="Path to core subset.jsonl")] = Path("src/complai/data/core.v1/subset.jsonl"),
    labels_dir: Annotated[Path, typer.Option(help="Directory containing item labels")] = Path("tools/label/labels"),
    predictions_out: Annotated[Path, typer.Option(help="Output JSON file for frontend predictions")] = Path("frontend/public/data/predictions.core.v1.json"),
    labels_out: Annotated[Path, typer.Option(help="Output JSON file for frontend labels")] = Path("frontend/public/data/labels.json"),
    ground_truth: Annotated[bool, typer.Option(help="Export observed scores to ground_truth.json")] = True,
):
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    print("1. Predicting scores (complai.predict)...")
    prediction = predict_detailed(input_path, params_path, subset_path)
    params, subset, result = prediction.params, prediction.subset, prediction.result

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
    manifest_path = input_path.with_suffix(".manifest.json")
    model_configs: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for file_meta in json.load(f).get("files", []):
                    mid = file_meta.get("model")
                    m_cfg = file_meta.get("model_config") or file_meta.get("eval_config")
                    if mid and m_cfg:
                        model_configs.setdefault(mid, {}).update(m_cfg)
        except Exception as e:
            print(f"Warning: Could not read manifest eval configs: {e}")
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

        truth = load_ground_truth(samples_path=input_path)
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
