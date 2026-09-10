import sys
import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json
import collections
from pathlib import Path
import numpy as np
import scipy.optimize as opt
import scipy.stats
from complai.utils.log_parser import load_records, preprocess_logs
from complai.irt import prepare_tasks
from complai.constants import CACHE_DIR
from tools.minify.config import load_scorers, get_task_allocations, get_primary_metrics

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def nll(theta, a, c, y):
    z = a * theta + c
    return np.sum(np.log1p(np.exp(-z)) + (1 - y) * z)

def optimize_theta(a, c, y):
    mask = ~np.isnan(y)
    if not np.any(mask):
        return 0.0
    a_m = a[mask]
    c_m = c[mask]
    y_m = y[mask]
    if len(y_m) == 0:
        return 0.0
    res = opt.minimize_scalar(lambda t: nll(t, a_m, c_m, y_m), bounds=(-10, 10), method='bounded')
    return res.x

def evaluate_stats(
    data_dir: Path = Path("src/complai/data"),
    logs_dir: Path = Path("logs/"),
    labels_dir: Path = Path("tools/label/labels")
):
    print("Loading Domain Labels...")
    datasets_dir = Path("tools/label/datasets")
    domain_map = {}
    
    if labels_dir.exists():
        for lf in labels_dir.glob("*.jsonl"):
            if lf.stem.endswith("_patch"):
                continue
            task_name = lf.stem
            ds_file = datasets_dir / lf.name
            
            # Bridge mismatched IDs
            possible_to_lbl = collections.defaultdict(list)
            if ds_file.exists():
                with open(ds_file) as f:
                    for i, line in enumerate(f):
                        row = json.loads(line)
                        lbl_sid = str(row.get("sample_id", ""))
                        meta = row.get("metadata", {})
                        
                        candidates = [
                            lbl_sid,
                            str(meta.get("uid", "")),
                            str(meta.get("task_id", "")),
                            str(i),
                            str(i+1)
                        ]
                        for c in candidates:
                            if c:
                                possible_to_lbl[lbl_sid].append(c)

            with open(lf) as f:
                for line in f:
                    row = json.loads(line)
                    lbl_sid = str(row.get("sample_id", ""))
                    dom = row.get("llm_assigned", {}).get("primary_label", "unknown")
                    
                    domain_map[(task_name, lbl_sid)] = dom # Always map the direct one
                    for true_sid in possible_to_lbl.get(lbl_sid, []):
                        domain_map[(task_name, true_sid)] = dom

            patch_path = lf.with_name(f"{lf.stem}_patch.jsonl")
            if patch_path.exists():
                with open(patch_path) as f:
                    for line in f:
                        row = json.loads(line)
                        lbl_sid = str(row.get("sample_id", ""))
                        if "human_primary_label" in row:
                            dom = row["human_primary_label"]
                            domain_map[(task_name, lbl_sid)] = dom
                            for true_sid in possible_to_lbl.get(lbl_sid, []):
                                domain_map[(task_name, true_sid)] = dom



    print(f"Loading Ground Truth from tools/minify/data/samples.jsonl...")
    scorers = load_scorers(None)
    allocations = get_task_allocations(None)
    primary_metrics = get_primary_metrics(None)
    cache_samples = Path("tools/minify/data/samples.jsonl")
    if not cache_samples.exists():
        preprocess_logs([logs_dir], scorers, cache_samples, domain_labels_dir=None, valid_labels_only=False)
    indexed = load_records(cache_samples)
    
    try:
        tasks = prepare_tasks(indexed, scorers, "error")
    except ValueError as e:
        if "Duplicate" in str(e):
            print("Detected duplicate logs, falling back to --duplicates latest...")
            tasks = prepare_tasks(indexed, scorers, "latest", _min_models=1)
        else:
            print(f"Failed to prepare tasks: {e}")
            return


    print("Calculating True Scores...")
    true_bench_scores = collections.defaultdict(dict)
    
    metrics_file = Path("tools/minify/data/metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics_data = json.load(f)
            for t_name, m_dict in metrics_data.items():
                for m_name, val in m_dict.items():
                    if val is not None:
                        # Some values might be dicts if we couldn't find the exact scorer name.
                        # Try to extract a float if possible.
                        if isinstance(val, dict):
                            expected = primary_metrics.get(t_name, ["accuracy"])
                            if isinstance(expected, str):
                                expected = [expected]
                                
                            found = False
                            for k in expected:
                                if k in val:
                                    true_bench_scores[t_name][m_name] = float(val[k])
                                    found = True
                                    break
                            
                            if not found:
                                # Avoid pulling calibration errors or standard errors as the primary metric
                                valid_vals = [v for k, v in val.items() if k not in ("cerr", "stderr")]
                                if valid_vals:
                                    true_bench_scores[t_name][m_name] = float(valid_vals[0])
                                else:
                                    true_bench_scores[t_name][m_name] = 0.0
                        else:
                            true_bench_scores[t_name][m_name] = float(val)
    else:
        print(f"WARNING: {metrics_file} not found. Falling back to linear average for benchmarks.")
        for t_name, t_data in tasks.items():
            matrix = t_data["matrix"]
            models = t_data["models"]
            with np.errstate(invalid='ignore'):
                model_means = np.nanmean(matrix, axis=1)
            for i, m_name in enumerate(models):
                if not np.isnan(model_means[i]):
                    true_bench_scores[t_name][m_name] = model_means[i]

    true_domain_scores_raw = collections.defaultdict(lambda: collections.defaultdict(list))
    for t_name, t_data in tasks.items():
        matrix = t_data["matrix"]
        models = t_data["models"]
            
        for j, item in enumerate(t_data["items"]):
            sid = str(item.get("sample_id", ""))
            dom = domain_map.get((t_name, sid), "unknown")
            for i, m_name in enumerate(models):
                val = matrix[i, j]
                if not np.isnan(val):
                    true_domain_scores_raw[dom][m_name].append(val)

    true_domain_scores = {}
    for dom, m_dict in true_domain_scores_raw.items():
        true_domain_scores[dom] = {m: np.mean(vals) for m, vals in m_dict.items() if len(vals) > 0}

    pool_size = sum(len(t_data["items"]) for t_data in tasks.values())

    print("\n=========================================")
    print("        SUBSET EVALUATION RESULTS        ")
    print("=========================================\n")

    # Find subsets to evaluate
    subset_dirs = []
    if (data_dir / "subset.jsonl").exists():
        subset_dirs.append(data_dir)
    else:
        for sub in sorted(data_dir.iterdir()):
            if sub.is_dir() and (sub / "subset.jsonl").exists():
                subset_dirs.append(sub)
                
    if not subset_dirs:
        print(f"No subset.jsonl found in {data_dir} or its subdirectories.")
        return

    for target_dir in subset_dirs:
        print(f"--- Evaluating {target_dir.name} ---")
        items_path = target_dir / "subset.jsonl"
        
        full_params_map = {}
        params_file = target_dir / "params.json"
        if params_file.exists():
            with open(params_file) as f:
                params_data = json.load(f)
                
                items_data = params_data.get("items", [])
                if isinstance(items_data, dict) and "columns" in items_data:
                    columns = items_data["columns"]
                    items_list = [
                        {k: v for k, v in zip(columns, row) if v is not None}
                        for row in items_data.get("data", [])
                    ]
                else:
                    items_list = items_data
                    
                for item in items_list:
                    full_params_map[str(item.get("item_id"))] = item
            
        domain_counts = collections.defaultdict(int)
        subset_items_by_task = collections.defaultdict(list)
        total_items = 0
        
        with open(items_path) as f:
            for line in f:
                item = json.loads(line)
                t_name = item["task"]
                sid = str(item.get("sample_id", ""))
                subset_items_by_task[t_name].append(item)
                dom = domain_map.get((t_name, sid), "unknown")
                domain_counts[dom] += 1
                total_items += 1
                
        percent_str = f"{total_items/pool_size*100:.1f}%" if pool_size > 0 else "0%"
        print(f"1. Domain Coverage (Total: {total_items:,} items - {percent_str} of {pool_size:,} available items):")
        for dom, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            print(f"  {dom:<20} | {count:>4}")
            
        print("2. Re-estimating theta and calculating MAE...")
        bench_errs = []
        all_taus = []
        all_true_mads = []
        all_pred_mads = []
        per_bench_errs = collections.defaultdict(list)
        domain_pred_sum = collections.defaultdict(lambda: collections.defaultdict(list))
        
        pred_score_stds = []
        top_saturation = 0
        bottom_saturation = 0
        total_thetas = 0
        
        top_sat_models = collections.Counter()
        bottom_sat_models = collections.Counter()
        
        for t_name, t_data in tasks.items():
            if t_name not in subset_items_by_task:
                # PENALTY: The subset completely dropped this benchmark.
                # The prediction defaults to random guessing (0.5) against the true scores.
                if t_name in true_bench_scores:
                    is_full = allocations.get(t_name, "default") == "full"
                    for m_name, true_score in true_bench_scores[t_name].items():
                        penalty = abs(0.5 - true_score)
                        if not is_full:
                            bench_errs.append(penalty)
                        per_bench_errs[t_name].append(penalty)
                continue
                
            sub_items = subset_items_by_task[t_name]
            sub_a = np.array([float(i["discrimination"]) for i in sub_items])
            sub_c = np.array([float(i["intercept"]) for i in sub_items])
            
            full_item_ids = [str(i.get("sample_id", "")) for i in t_data["items"]]
            item_id_to_idx = {sid: j for j, sid in enumerate(full_item_ids)}
            
            sub_j_indices = []
            for i in sub_items:
                sid = str(i.get("sample_id", ""))
                # If for some reason a selected item isn't in the full task items, skip or zero
                sub_j_indices.append(item_id_to_idx.get(sid, -1))
                
            matrix = t_data["matrix"]
            models = t_data["models"]
            full_a = np.array([full_params_map.get(str(i.get("item_id")), {}).get("discrimination", 0.0) for i in t_data["items"]])
            full_c = np.array([full_params_map.get(str(i.get("item_id")), {}).get("intercept", 0.0) for i in t_data["items"]])
            
            task_pred_scores = []
            task_true_scores_for_rank = []
            task_pred_scores_for_rank = []
            for i, m_name in enumerate(models):
                y_m = np.array([matrix[i, j] if j != -1 else np.nan for j in sub_j_indices])
                
                theta_hat = optimize_theta(sub_a, sub_c, y_m)
                
                # Check for top saturation (ceiling) or bottom saturation (floor)
                if theta_hat >= 9.9:
                    top_saturation += 1
                    top_sat_models[m_name] += 1
                elif theta_hat <= -9.9:
                    bottom_saturation += 1
                    bottom_sat_models[m_name] += 1
                total_thetas += 1
                
                probs = sigmoid(full_a * theta_hat + full_c)
                
                # Use the model's specific NaN mask to calculate the mean.
                # For mmlu_pro_robustness, this correctly masks out the failed unperturbed items!
                full_y_m = matrix[i, :]
                valid_mask = ~np.isnan(full_y_m)
                if np.any(valid_mask):
                    pred_score = np.mean(probs[valid_mask])
                else:
                    pred_score = np.mean(probs)
                    
                task_pred_scores.append(pred_score)
                
                if m_name in true_bench_scores.get(t_name, {}):
                    true_score = true_bench_scores[t_name][m_name]
                    is_full = allocations.get(t_name, "default") == "full"
                    if not is_full:
                        task_true_scores_for_rank.append(true_score)
                        task_pred_scores_for_rank.append(pred_score)
                    if allocations.get(t_name, "default") == "full":
                        err = 0.0
                        # Do not append 0.0 to bench_errs so we don't artificially lower the overall IRT MAE
                    else:
                        err = abs(pred_score - true_score)
                        bench_errs.append(err)
                    per_bench_errs[t_name].append(err)
                
                for j, full_item in enumerate(t_data["items"]):
                    sid = str(full_item.get("sample_id", ""))
                    dom = domain_map.get((t_name, sid), "unknown")
                    domain_pred_sum[dom][m_name].append(probs[j])
                    
            if len(task_pred_scores) > 0:
                pred_score_stds.append(np.std(task_pred_scores))
                
            if len(task_true_scores_for_rank) > 1:
                tau, _ = scipy.stats.kendalltau(task_true_scores_for_rank, task_pred_scores_for_rank)
                if not np.isnan(tau):
                    all_taus.append(tau)
                sorted_true = sorted(task_true_scores_for_rank)
                true_diffs = [sorted_true[k] - sorted_true[k-1] for k in range(1, len(sorted_true))]
                if true_diffs:
                    all_true_mads.append(np.mean(true_diffs))
                    
                sorted_pred = sorted(task_pred_scores_for_rank)
                pred_diffs = [sorted_pred[k] - sorted_pred[k-1] for k in range(1, len(sorted_pred))]
                if pred_diffs:
                    all_pred_mads.append(np.mean(pred_diffs))
                    
        print("\n  --- Per-Benchmark Breakdown ---")
        for t, errs in sorted(per_bench_errs.items(), key=lambda x: np.mean(x[1]) if x[1] else 0, reverse=True):
            n_items = len(subset_items_by_task[t])
            mean_err = np.mean(errs) if errs else 0.0
            print(f"    {t:<25} | {n_items:>4} items | {mean_err*100:>5.2f}% MAE")
        print("  -------------------------------")
        print(f"  Avg Benchmark MAE: {np.mean(bench_errs):.4f}")
        
        domain_maes = []
        for dom, m_dict in domain_pred_sum.items():
            if dom not in true_domain_scores: continue
            for m_name, probs in m_dict.items():
                if m_name in true_domain_scores[dom]:
                    pred_score = np.mean(probs)
                    true_score = true_domain_scores[dom][m_name]
                    domain_maes.append(abs(pred_score - true_score))
                    
        if domain_maes:
            print(f"  Avg Domain MAE:    {np.mean(domain_maes):.4f}")
        else:
            print("  Avg Domain MAE:    N/A")
            
        print(f"  Avg Pred. Score Spread: {np.mean(pred_score_stds):.4f} (Standard Deviation)")
        if all_taus:
            avg_tau = np.mean(all_taus)
            accuracy = (avg_tau + 1.0) / 2.0 * 100.0
            print(f"  Pairwise Rank Accuracy: {accuracy:.1f}%")
        if all_true_mads and all_pred_mads:
            print(f"  Avg True Model Gap:     {np.mean(all_true_mads)*100:.2f}%")
            print(f"  Avg Predicted Gap:      {np.mean(all_pred_mads)*100:.2f}%")
        print(f"  Top Saturation:    {(top_saturation / total_thetas) * 100:.1f}% (Ceiling)")
        print(f"  Bottom Saturation: {(bottom_saturation / total_thetas) * 100:.1f}% (Floor)")
            
        print("\n" + "-"*40 + "\n")


