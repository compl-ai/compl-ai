import typer
import json
import collections
import numpy as np
import re
from typing import Annotated
from pathlib import Path

from complai.predict import predict_scores, read_inputs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

app = typer.Typer(help="Export COMPL-AI predictions and labels to static JSON files for the frontend.")

def parse_markdown_schema(md_path: Path) -> dict:
    schema_obj = {"domains": {}, "sub_labels": {}, "tags": {}}
    if not md_path.exists():
        return schema_obj
        
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    current_section = None
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        if line.startswith("## "):
            header = line[3:].strip()
            if header.startswith("Tags"): current_section = "tags"
            else:
                current_section = "domains"
                match = re.match(r"([a-z-]+)\s+\((.+)\)", header)
                if match:
                    current_domain = match.group(1)
                    desc = lines[i+1].strip() if i+1 < len(lines) and not lines[i+1].startswith("-") else ""
                    schema_obj["domains"][current_domain] = {"name": match.group(2), "description": desc}
        elif line.startswith("- **") and current_section:
            match = re.match(r"-\s+\*\*(.+?)\*\*\s+\((.+?)\)", line)
            if match:
                key, name = match.group(1), match.group(2)
                desc = line.split(")", 1)[1].strip() if ")" in line else ""
                if desc.startswith(": "): desc = desc[2:]
                schema_obj[current_section][key] = {"name": name, "description": desc}
                
    return schema_obj

def compile_labels(labels_dir: Path, schema_obj: dict) -> dict:
    if not labels_dir.exists():
        return {}
    
    full_labels = collections.defaultdict(dict)
    for lf in labels_dir.glob("*.jsonl"):
        if lf.stem.endswith("_patch"): continue
        task = lf.stem
        
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                sid = str(row.get("sample_id", ""))
                assigned = row.get("llm_assigned", {})
                
                primary = assigned.get("primary_label")
                secondary = assigned.get("secondary_label")
                tags = assigned.get("tags", [])
                
                if primary not in schema_obj["domains"]: primary = "unknown"
                if secondary not in schema_obj["sub_labels"]: secondary = None
                
                valid_tags = [t for t in tags if t in schema_obj["tags"]]
                full_labels[task][sid] = {
                    "primary": primary,
                    "secondary": secondary,
                    "tags": valid_tags
                }
                
    # Apply human patches
    for patch_f in labels_dir.glob("*_patch.jsonl"):
        task = patch_f.stem.replace("_patch", "")
        with open(patch_f, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                sid = str(row.get("sample_id", ""))
                
                if task in full_labels and sid in full_labels[task]:
                    if "human_primary_label" in row:
                        full_labels[task][sid]["primary"] = row["human_primary_label"]
                    if "human_secondary_label" in row:
                        full_labels[task][sid]["secondary"] = row["human_secondary_label"]
                    if "human_tags" in row:
                        full_labels[task][sid]["tags"] = row["human_tags"]
                        
    # Sort for deterministic output
    sorted_taxonomy = {}
    for task in sorted(full_labels.keys()):
        sorted_taxonomy[task] = {
            sid: full_labels[task][sid]
            for sid in sorted(full_labels[task].keys())
        }
    
    return sorted_taxonomy

@app.command()
def export(
    input_path: Annotated[Path, typer.Argument(help="Preprocessed samples.jsonl")] = Path("tools/minify/data/samples.jsonl"),
    params_path: Annotated[Path, typer.Option(help="Path to core params.json")] = Path("src/complai/data/core.v1/params.json"),
    subset_path: Annotated[Path, typer.Option(help="Path to core subset.jsonl")] = Path("src/complai/data/core.v1/subset.jsonl"),
    labels_dir: Annotated[Path, typer.Option(help="Directory containing domain labels")] = Path("tools/label/labels"),
    schema_path: Annotated[Path, typer.Option(help="Path to taxonomy schema md")] = Path("tools/label/harness/instructions/taxonomy_optimized.md"),
    predictions_out: Annotated[Path, typer.Option(help="Output JSON file for frontend predictions")] = Path("frontend/public/data/predictions.core.v1.json"),
    labels_out: Annotated[Path, typer.Option(help="Output JSON file for frontend labels")] = Path("frontend/public/data/labels.json"),
    ground_truth: Annotated[bool, typer.Option(help="Export true population scores to ground_truth.json")] = False
):
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading IRT parameters...")
    params, _ = read_inputs(params_path, subset_path)
    
    print("2. Calculating core IRT predictions (theta)...")
    result = predict_scores(input_path, params_path, subset_path, duplicate_policy="latest")
    
    print("3. Compiling full label dataset from raw data...")
    schema_obj = parse_markdown_schema(schema_path)
    full_labels = compile_labels(labels_dir, schema_obj)
    
    print("4. Calculating Subset Statistics...")
    subset_domain_counts = collections.defaultdict(int)
    total_subset_items = 0
    if subset_path.exists():
        with open(subset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                t_name = row["task"]
                sid = str(row.get("sample_id", ""))
                
                if t_name in full_labels and sid in full_labels[t_name]:
                    dom = full_labels[t_name][sid]["primary"]
                    subset_domain_counts[dom] += 1
                total_subset_items += 1
                
    for dom, count in subset_domain_counts.items():
        if dom in schema_obj["domains"]:
            # e.g., '12.4%'
            schema_obj["domains"][dom]["subset_weight_pct"] = f"{(count/total_subset_items)*100:.1f}%" if total_subset_items > 0 else "0%"
            schema_obj["domains"][dom]["subset_count"] = count
    schema_obj["total_subset_items"] = total_subset_items
    
    print("5. Extracting run configurations from manifest...")
    manifest_path = input_path.with_suffix(".manifest.json")
    model_configs = {}
    task_configs = collections.defaultdict(dict)
    
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                for file_meta in manifest_data.get("files", []):
                    mid = file_meta.get("model")
                    task = file_meta.get("task")
                    m_cfg = file_meta.get("model_config")
                    t_cfg = file_meta.get("task_config")
                    # Fallback to older format in case a manual patch remains
                    if not m_cfg and not t_cfg and "eval_config" in file_meta:
                        m_cfg = file_meta.get("eval_config")
                        
                    if mid:
                        if m_cfg:
                            if mid not in model_configs:
                                model_configs[mid] = {}
                            model_configs[mid].update(m_cfg)
                        if task and t_cfg:
                            task_configs[mid][task] = t_cfg
        except Exception as e:
            print(f"Warning: Could not read manifest eval configs: {e}")
    
    print("6. Calculating sample-level Domain scores for the frontend...")
    for model_id, model_data in result["models"].items():
        if model_id in model_configs:
            model_data["model_config"] = model_configs[model_id]
            
        domain_item_probs = collections.defaultdict(list)
        coverage_tasks = 0
        coverage_samples = 0
        total_samples = 0
        
        for task, data in model_data["tasks"].items():
            if data["status"] != "missing":
                coverage_tasks += 1
                coverage_samples += data.get("observations", 0)
            
            # Map subset item performance to domains
            if "ability" in data and data["ability"] is not None:
                theta = data["ability"]
                population = [item for item in params.get("items", []) if item.get("task") == task]
                if not population:
                    continue
                    
                discriminations = np.asarray([float(item["discrimination"]) for item in population])
                intercepts = np.asarray([float(item["intercept"]) for item in population])
                probs = sigmoid(discriminations * theta + intercepts)
                
                for idx, item in enumerate(population):
                    sid = str(item.get("sample_id", ""))
                    dom = "unknown"
                    if task in full_labels and sid in full_labels[task]:
                        dom = full_labels[task][sid]["primary"]
                    domain_item_probs[dom].append(float(probs[idx]))
            
            elif data.get("status") != "missing" and "predicted_score" in data:
                # FULL-ALLOCATION: Spread global predicted score across all items
                if task not in full_labels:
                    continue
                global_score = float(data["predicted_score"])
                for sid, label_info in full_labels[task].items():
                    dom = label_info["primary"]
                    domain_item_probs[dom].append(global_score)
                    
        model_data["domains"] = {dom: float(np.mean(probs)) for dom, probs in domain_item_probs.items()}
                
        # Fill minimal coverage stats
        if "inventory" in result and "summary" in result["inventory"]:
            # Recalculate accurately based on exact dataset mapping
            total_tasks_available = result["inventory"]["summary"].get("total_tasks", 0)
        else:
            total_tasks_available = len(model_data["tasks"])
            
        model_data["coverage"] = {
            "tasks_completed": coverage_tasks,
            "tasks_total": total_tasks_available,
            "samples_completed": coverage_samples,
            "samples_total": total_subset_items
        }

        
    print(f"7. Writing enriched predictions to {predictions_out}...")
    with open(predictions_out, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"8. Writing columnar labels to {labels_out}...")
    output_str = "{\n"
    output_str += '  "_schema": ' + json.dumps(schema_obj, indent=4).replace('\n', '\n  ') + ',\n'
    for i, (task, samples) in enumerate(full_labels.items()):
        output_str += f'  "{task}": {{\n'
        output_str += '    "columns": ["sample_id", "primary", "secondary", "tags"],\n'
        rows = [json.dumps([sid, data["primary"], data["secondary"], data["tags"]], separators=(",", ":")) for sid, data in samples.items()]
        output_str += '    "data": [\n      ' + ',\n      '.join(rows) + '\n    ]\n  }'
        output_str += ",\n" if i < len(full_labels) - 1 else "\n"
    output_str += "}\n"
    
    with open(labels_out, "w", encoding="utf-8") as f:
        f.write(output_str)

    if ground_truth:
        print("9. Calculating Ground Truth Population Scores...")
        try:
            from tools.minify.config import get_task_allocations, get_primary_metrics
            from complai.predict import load_records, prepare_tasks
            
            allocations = get_task_allocations()
            primary_metrics = get_primary_metrics()
            records = load_records(input_path)
            
            task_scorers = params.get("task_scorers", {})
            tasks = prepare_tasks(records, task_scorers, duplicate_policy="latest", _min_models=1)
            
            true_scores = collections.defaultdict(dict)
            
            for t_name, t_data in tasks.items():
                models = t_data["models"]
                matrix = t_data["matrix"]
                
                is_full = allocations.get(t_name, "default") == "full"
                if is_full:
                    scorer_name = task_scorers.get(t_name)
                    primary_metric = primary_metrics.get(t_name, ["accuracy"])
                    if isinstance(primary_metric, str): primary_metric = [primary_metric]
                    
                    for i, m_name in enumerate(models):
                        file_record = next((f for f in records.files if f["task"] == t_name and f["model"] == m_name), None)
                        if file_record:
                            scorer_metrics = file_record.get("metrics", {}).get(scorer_name, {})
                            val = None
                            for pm in primary_metric:
                                if pm in scorer_metrics and scorer_metrics[pm] is not None:
                                    val = float(scorer_metrics[pm])
                                    break
                            if val is None and scorer_metrics:
                                val = next((float(v) for v in scorer_metrics.values() if v is not None), None)
                                
                            if val is not None:
                                true_scores[m_name][t_name] = val
                                
                if not is_full:
                    with np.errstate(invalid='ignore'):
                        model_means = np.nanmean(matrix, axis=1)
                    for i, m_name in enumerate(models):
                        if not np.isnan(model_means[i]):
                            if t_name not in true_scores[m_name]:
                                true_scores[m_name][t_name] = float(model_means[i])
                                
            # --- Add True Domain Scores ---
            true_domain_scores_raw = collections.defaultdict(lambda: collections.defaultdict(list))
            for t_name, t_data in tasks.items():
                matrix = t_data["matrix"]
                models = t_data["models"]
                
                is_full = allocations.get(t_name, "default") == "full"
                for j, item in enumerate(t_data["items"]):
                    sid = str(item.get("sample_id", ""))
                    dom = "unknown"
                    if t_name in full_labels and sid in full_labels[t_name]:
                        dom = full_labels[t_name][sid]["primary"]
                    
                    for i, m_name in enumerate(models):
                        if is_full:
                            # Full-allocation: spread the global true score
                            val = true_scores.get(m_name, {}).get(t_name)
                        else:
                            # IRT: use item-level correctness
                            val = matrix[i, j]
                            
                        if val is not None and not np.isnan(val):
                            true_domain_scores_raw[dom][m_name].append(val)
                            
            for dom, m_dict in true_domain_scores_raw.items():
                for m_name, vals in m_dict.items():
                    if len(vals) > 0:
                        true_scores[m_name][dom] = float(np.mean(vals))

            gt_out = predictions_out.parent / "ground_truth.json"
            with open(gt_out, "w") as f:
                json.dump(true_scores, f, indent=2)
            print(f"Exported ground truth to {gt_out}")
        except Exception as e:
            print(f"Warning: Failed to export ground truth: {e}")

    print("Done! Artifacts successfully exported.")

if __name__ == "__main__":
    app()
