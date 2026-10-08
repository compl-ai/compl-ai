import json
import statistics
import collections
from pathlib import Path

logs_dir = Path("logs/logs/")
task_model_accumulator = collections.defaultdict(lambda: collections.defaultdict(lambda: {"input": 0, "output": 0, "total": 0, "samples": 0}))

# 1. Properly parse all JSON files using Python's json library
for logs_json in logs_dir.glob("*/*.json"):
    with open(logs_json, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            continue
            
        # These logs are dictionaries of multiple eval runs
        for eval_obj in data.values():
            if not isinstance(eval_obj, dict): 
                continue
                
            task_name = eval_obj.get("eval", {}).get("task", "unknown")
            model = eval_obj.get("eval", {}).get("model", "unknown")
            
            # The 'stats' object maps exactly to Inspect's EvalStats schema
            usage = eval_obj.get("stats", {}).get("model_usage", {})
            
            in_tok = 0
            out_tok = 0
            tot_tok = 0
            for model_name, tokens in usage.items():
                in_tok += tokens.get("input_tokens", 0)
                out_tok += tokens.get("output_tokens", 0)
                tot_tok += tokens.get("total_tokens", 0)
                
            # Safely extract samples from reductions (which is at the root in these logs)
            samples = 0
            reductions = eval_obj.get("reductions", [])
            if isinstance(reductions, list) and len(reductions) > 0:
                samples = len(reductions[0].get("samples", []))
                
            if samples > 0 and tot_tok > 0:
                task_model_accumulator[task_name][model]["input"] += in_tok
                task_model_accumulator[task_name][model]["output"] += out_tok
                task_model_accumulator[task_name][model]["total"] += tot_tok
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

print("Proper JSON parsing complete. Wrote tools/minify/data/cost.json")
