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
                key = match.group(1)
                obj = {"name": match.group(2), "apply_when": "", "do_not_apply_when": ""}
                for j in range(i+1, min(i+4, len(lines))):
                    subline = lines[j].strip()
                    if subline.startswith("- *Apply when:*"): obj["apply_when"] = subline.replace("- *Apply when:*", "").strip()
                    elif subline.startswith("- *Do not apply when:*"): obj["do_not_apply_when"] = subline.replace("- *Do not apply when:*", "").strip()
                if current_section == "domains": schema_obj["sub_labels"][key] = obj
                elif current_section == "tags": schema_obj["tags"][key] = obj
    return schema_obj

def compile_labels(labels_dir: Path, schema_obj: dict) -> dict:
    taxonomy = collections.defaultdict(dict)
    if not labels_dir.exists():
        return taxonomy
        
    for lf in labels_dir.glob("*.jsonl"):
        if lf.stem.endswith("_patch"):
            continue
        task_name = lf.stem
        
        # 1. AI labels
        try:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        row = json.loads(line)
                        sid = str(row.get("sample_id", ""))
                        if not sid: continue
                        llm = row.get("llm_assigned", {})
                        taxonomy[task_name][sid] = {
                            "primary": llm.get("primary_label", "unknown"),
                            "secondary": llm.get("secondary_labels", []),
                            "tags": llm.get("tags", [])
                        }
                    except json.JSONDecodeError: pass
        except IOError: pass
            
        # 2. Human patches
        patch_path = lf.with_name(f"{lf.stem}_patch.jsonl")
        if patch_path.exists():
            try:
                with open(patch_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            row = json.loads(line)
                            sid = str(row.get("sample_id", ""))
                            if not sid or sid not in taxonomy[task_name]: continue
                            if "human_primary_label" in row: taxonomy[task_name][sid]["primary"] = row["human_primary_label"]
                            if "human_secondary_labels" in row: taxonomy[task_name][sid]["secondary"] = row["human_secondary_labels"]
                            if "human_tags" in row: taxonomy[task_name][sid]["tags"] = row["human_tags"]
                        except json.JSONDecodeError: pass
            except IOError: pass
            
    sorted_taxonomy = {
        task: dict(sorted(samples.items(), key=lambda x: str(x[0])))
        for task, samples in sorted(taxonomy.items())
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
    labels_out: Annotated[Path, typer.Option(help="Output JSON file for frontend labels")] = Path("frontend/public/data/labels.json")
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
    
    print("4. Calculating sample-level Domain scores for the frontend...")
    for model_id, model_data in result["models"].items():
        domain_item_probs = collections.defaultdict(list)
        
        for task_name, task_result in model_data.get("tasks", {}).items():
            theta = task_result.get("ability")
            if theta is None: continue
                
            population = [item for item in params["items"] if item["task"] == task_name]
            if not population: continue
                
            discriminations = np.asarray([float(item["discrimination"]) for item in population])
            intercepts = np.asarray([float(item["intercept"]) for item in population])
            
            probs = sigmoid(discriminations * theta + intercepts)
            
            for idx, item in enumerate(population):
                sid = str(item.get("sample_id", ""))
                
                # Fetch domain from compiled labels
                dom = "unknown"
                if task_name in full_labels and sid in full_labels[task_name]:
                    dom = full_labels[task_name][sid]["primary"]
                
                domain_item_probs[dom].append(float(probs[idx]))
                
        model_data["domains"] = {dom: float(np.mean(probs)) for dom, probs in domain_item_probs.items()}
        
    print(f"5. Writing enriched predictions to {predictions_out}...")
    with open(predictions_out, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"6. Writing columnar labels to {labels_out}...")
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
        
    print("Done! Both artifacts successfully exported.")

if __name__ == "__main__":
    app()
