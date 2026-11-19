"""Build frontend/public/data/agreement.json: the figure rows and every number quoted on the methodology page
("How it works"): leave-one-model-out accuracy, agreement with Epoch ECI and Artificial Analysis v4, panel size.
The cost numbers on that page come from frontend/public/data/costs.json (built by complai export-web).

Our index thetas are joined to the references by exact name only; every mapped name is asserted to exist in its source.
Bootstrap intervals: seeded RNG, 2,000 draws; values depend on the order of the boot calls in agreement_numbers().

Inputs, in tools/method/ unless noted:
  out/lomo/, out/panel.json                 lomo.py
  data/eci_scores.csv, data/aa_models.json  Epoch ECI and AA v4 snapshots, both from 2026-10-26
  frontend/public/data/predictions.json     index thetas and intervals
  frontend/public/models/*.yaml             display names, open weights (matched by `evals`)

Run:  .venv/bin/python tools/method/agreement.py
"""
import glob
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
OUT = HERE / "out"
FE = ROOT / "frontend/public"
T0 = pd.Timestamp("2026-01-01")
SCALE = lambda t: 100 + 15 * t  # noqa: E731
INDICES = ("capability", "reliability", "safety")
NB = 2000

# Nov 8: the openrouter/nvidia/nemotron-3-ultra-550b-a55b duplicate is gone; Ultra is now only keyed by its vLLM id,
# so it is no longer dropped and the maps below use that id. GLM-5 has left the leaderboard.
DROP = ["GLM-5"]
ULTRA = "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
# Removed from the leaderboard after the Nov 2 run (Apertus to be re-added); their map entries are pruned in load().
REMOVED = ["Apertus-70B-Instruct-2509", "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16", "Olmo-3-32B-Think", "Qwen3.5-27B",
           "Qwen3.5-35B-A3B", "Qwen3.5-397B-A17B", "Seed-OSS-36B-Instruct"]
STAR = {
    "cap": ["nemotron-3.5-lightning"],
    "rel": ["Seed-OSS-36B-Instruct", "Olmo-3-32B-Think", "Olmo-3.1-32B-Instruct", "claude-opus-5.5"],
    "saf": ["Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B", "Qwen3.5-397B-A17B", "Qwen3.6-27B", "gpt-oss-20b",
            "gpt-oss-120b", "gemini-4-argon", "claude-opus-5.5"],
}

AA_MAP = {
    "claude-opus-5.5": "Claude Opus 5.5 (high with fallback)", "claude-sonnet-5.5": "Claude Sonnet 5.5 (high with fallback)",
    "claude-haiku-5.5": "Claude Haiku 5.5 (high)", "gemini-4-argon": "Gemini 4 Argon (high)",
    "gpt-6-luna": "GPT-6 Luna (high)", "gemini-3.8-flash": "Gemini 3.8 Flash (high)",
    "mistral-large-4-0": "Mistral Large 4 Preview", "muse-spark-open": "Muse Spark Open (xhigh)",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash (max)", "glm-5.3-flash": "GLM-5.3-Flash", "mimo-v2.6-flash": "MiMo-V2.6-Flash",
    # added 2026-11-10 for the methodology refresh (exact AA v4 names, cache of 10-26)
    "kimi-k3.1-pro": "Kimi K3.1 Pro", "mimo-v2.6-pro": "MiMo-V2.6-Pro", "minimax-m3.1-flash": "MiniMax-M3.1-Flash",
    "ling-3.1-flash": "Ling 3.1 Flash", "nemotron-3.5-lightning": "Nemotron 3.5 Lightning", "hy3": "Hy3",
    ULTRA: "Nemotron 3 Ultra",
    "Kimi-K2.6": "Kimi K2.6", "Qwen3.5-397B-A17B": "Qwen3.5 397B A17B", "Qwen3.5-9B": "Qwen3.5 9B", "Qwen3.5-27B": "Qwen3.5 27B",
    "Qwen3.5-35B-A3B": "Qwen3.5 35B A3B", "Qwen3.6-27B": "Qwen3.6 27B", "Qwen3.6-35B-A3B": "Qwen3.6 35B A3B", "Qwen3-8B": "Qwen3 8B",
    "MiniMax-M2.5": "MiniMax-M2.5", "MiniMax-M2.7": "MiniMax-M2.7", "MiMo-V2.5": "MiMo-V2.5",
    "Mistral-Medium-3.5-128B": "Mistral Medium 3.5", "Mistral-Small-4-119B-2603": "Mistral Small 4",
    "Ministral-3-14B-Instruct-2512": "Ministral 3 14B", "NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "Nemotron 3 Super",
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Nemotron 3 Nano", "NVIDIA-Nemotron-3-Nano-4B-BF16": "Nemotron 3 Nano 4B",
    "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": "Nemotron 3 Nano Omni 30B A3B",
    "gemma-4-E4B-it": "Gemma 4 E4B", "gemma-4-31B-it": "Gemma 4 31B", "gemma-4-26B-A4B-it": "Gemma 4 26B A4B",
    "gemma-4-12B-it": "Gemma 4 12B", "gemma-3-27b-it": "Gemma 3 27B", "DeepSeek-V4-Pro": "DeepSeek V4 Pro (max)",
    "DeepSeek-V4-Flash": "DeepSeek V4 Flash (max)", "DeepSeek-V3.2": "DeepSeek V3.2 (non-reasoning)",
    "gpt-oss-120b": "gpt-oss-120b (high)", "gpt-oss-20b": "gpt-oss-20b (high)",
    "Llama-3.1-8B-Instruct": "Llama 3.1 8B", "Llama-3.3-70B-Instruct": "Llama 3.3 70B", "phi-4": "Phi-4",
    "Olmo-3-32B-Think": "Olmo 3 32B Think", "Olmo-3.1-32B-Instruct": "Olmo 3.1 32B Instruct", "Seed-OSS-36B-Instruct": "Seed-OSS-36B-Instruct",
    "Step-3.5-Flash": "Step 3.5 Flash", "granite-4.0-h-small": "Granite 4.0 H Small", "Apertus-70B-Instruct-2509": "Apertus 70B Instruct",
    "diffusiongemma-26B-A4B-it": "DiffusionGemma 26B A4B", "Hy3-preview": "Hy3-preview",
}
ECI_MAP = {
    "claude-opus-5.5": "Claude Opus 5.5", "claude-sonnet-5.5": "Claude Sonnet 5.5", "claude-haiku-5.5": "Claude Haiku 5.5",
    "gemini-4-argon": "Gemini 4 Argon", "gpt-6-luna": "GPT-6 Luna", "gemini-3.8-flash": "Gemini 3.8 Flash",
    "mistral-large-4-0": "Mistral Large 4", "muse-spark-open": "Muse Spark Open",
    ULTRA: "Nemotron 3 Ultra", "glm-5.3-flash": "GLM-5.3-Flash", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "DeepSeek-V4-Pro": "DeepSeek-V4-Pro", "DeepSeek-V4-Flash": "DeepSeek-V4-Flash", "DeepSeek-V3.2": "DeepSeek-V3.2",
    "Kimi-K2.6": "Kimi K2.6", "Llama-3.1-8B-Instruct": "Llama 3.1-8B", "Llama-3.3-70B-Instruct": "Llama 3.3 70B",
    "MiniMax-M2.5": "MiniMax-M2.5", "MiniMax-M2.7": "MiniMax-M2.7", "Mistral-Medium-3.5-128B": "Mistral Medium 3.5",
    "Qwen3-8B": "Qwen3-8B", "Qwen3.5-35B-A3B": "Qwen3.5-35B-A3B", "Qwen3.5-397B-A17B": "Qwen3.5 397B-A17B", "Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen3.6-27B": "Qwen3.6 27B", "Qwen3.6-35B-A3B": "Qwen 3.6 35B-A3B", "gemma-3-27b-it": "Gemma 3 27B",
    "gemma-4-26B-A4B-it": "Gemma 4 26B A4B", "gemma-4-31B-it": "Gemma 4 31B IT", "gpt-oss-120b": "gpt-oss-120b",
    "gpt-oss-20b": "gpt-oss-20b", "phi-4": "Phi-4",
    "claude-haiku-4.5": "Claude Haiku 4.5", "claude-sonnet-4.5": "Claude Sonnet 4.5",  # added Nov 8 (new core runs)
}


# ---------------------------------------------------------------- metrics
def pra(a, b):
    """Pairwise rank accuracy: share of model pairs ordered the same way by a and b (pairs tied in either skipped)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = np.sign(a[:, None] - a[None, :]); db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(len(a), 1)
    da, db = da[iu], db[iu]
    ok = (da != 0) & (db != 0)
    return float((da[ok] == db[ok]).mean()) if ok.any() else np.nan


def pear(a, b):
    return float(pearsonr(a, b)[0])


def spear(a, b):
    return float(spearmanr(a, b)[0])


def boot(f, a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    est = f(a, b)
    bs = []
    for _ in range(NB):
        i = rng.integers(0, len(a), len(a))
        if len(np.unique(i)) < 3:
            continue
        bs.append(f(a[i], b[i]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi


# ---------------------------------------------------------------- data (ours joined to ECI and AA v4)
def build_meta():
    P = json.load(open(ROOT / "frontend/public/data/predictions.json"))["models"]
    cards = {}
    for f in glob.glob(str(ROOT / "frontend/public/models/*.yaml")):
        y = yaml.safe_load(open(f))
        for e in y.get("evals") or []:
            cards[e] = y
    rows = []
    for mid, m in P.items():
        y = cards.get(mid, {}); sp = y.get("specs") or {}; rg = y.get("region") or {}
        r = dict(id=mid, m=mid.split("/")[-1], name=y.get("name"), open=y.get("open_weights"), org=y.get("organization"),
                 date=pd.to_datetime(str(y.get("release_date")), errors="coerce"), tot=sp.get("total_params"),
                 act=sp.get("active_params"), region=(rg.get("region_developed") or [None])[0])
        for i in ["capability", "reliability", "safety"]:
            it = m["indices"][i].get("index_theta") or {}
            r[i[:3]] = it.get("theta"); r[i[:3] + "_se"] = it.get("standard_error")
        for k, v in m["subcategories"].items():
            r["sc:" + k] = v.get("ability") if v.get("observations") else None
        for t, v in m["tasks"].items():
            r["t:" + t] = v.get("observed_subset_score"); r["s:" + t] = v.get("status")
        rows.append(r)
    M = pd.DataFrame(rows)
    M["region"] = M.region.astype(str).str.strip("“”\"' ").replace({"GB-ENG": "GB", "None": None})
    M["yr"] = (M.date - T0).dt.days / 365.25
    M["LT"] = np.log10(M.tot.astype(float)); M["LA"] = np.log10(M.act.astype(float))
    return M


def load():
    for mp in (AA_MAP, ECI_MAP):
        for m in REMOVED:
            mp.pop(m, None)
    for k in STAR:
        STAR[k] = [m for m in STAR[k] if m not in REMOVED]
    M = build_meta()
    assert not set(REMOVED) & set(M.m), set(REMOVED) & set(M.m)
    M = M[~M.m.isin(DROP)].set_index("m")
    P = json.load(open(ROOT / "frontend/public/data/predictions.json"))["models"]
    for ix in ["capability", "reliability", "safety"]:
        k = ix[:3]
        lo, hi = [], []
        for mid in M.id:
            it = P[mid]["indices"][ix].get("index_theta") or {}
            iv = it.get("interval") if it.get("theta") is not None else None
            lo.append(iv[0] if iv else np.nan); hi.append(iv[1] if iv else np.nan)
        M[k + "_lo"], M[k + "_hi"] = lo, hi
        M[k + "_se"] = (M[k + "_hi"] - M[k + "_lo"]) / 3.29
        M[k + "_star"] = M.index.isin(STAR[k])
    nocard = M.index[M.open.isna()].tolist()
    assert not nocard, nocard  # Claude 4.5 cards filled Nov 8
    M["type"] = M.open.map({True: "open", False: "closed"}).fillna("closed°")
    for k in STAR:
        miss = set(STAR[k]) - set(M.index)
        assert not miss, (k, miss)

    aa = json.load(open(DATA / "aa_models.json"))
    miss = {m: n for m, n in AA_MAP.items() if n not in aa}
    assert not miss, f"AA v4 names not found: {miss}"
    E = pd.read_csv(DATA / "eci_scores.csv").set_index("Model")
    assert not E.index.duplicated().any()
    miss = {m: n for m, n in ECI_MAP.items() if n not in E.index}
    assert not miss, f"ECI names not found: {miss}"
    for mp in (AA_MAP, ECI_MAP):
        miss = set(mp) - set(M.index)
        assert not miss, f"our models not found: {miss}"

    M["aa"] = [aa[AA_MAP[m]]["ii"] if m in AA_MAP else np.nan for m in M.index]
    M["eci"] = [E.eci[ECI_MAP[m]] if m in ECI_MAP else np.nan for m in M.index]
    M["eci_lo"] = [E.eci_ci_low[ECI_MAP[m]] if m in ECI_MAP else np.nan for m in M.index]
    M["eci_hi"] = [E.eci_ci_high[ECI_MAP[m]] if m in ECI_MAP else np.nan for m in M.index]
    assert M.aa[list(AA_MAP)].notna().all(), M.aa[list(AA_MAP)][M.aa[list(AA_MAP)].isna()]
    return M, aa, E, P


# ---------------------------------------------------------------- agreement
def agreement_table(M):
    """Per-model ours vs ECI/AA v4 (unrounded), strongest Capability first."""
    C = M[M.cap.notna()].copy()
    C["aa_name"] = [AA_MAP.get(m) for m in C.index]
    C["eci_name"] = [ECI_MAP.get(m) for m in C.index]
    C["flagged"] = C.cap_star
    cols = ["id", "type", "cap", "cap_lo", "cap_hi", "rel", "saf", "eci_name", "eci", "eci_lo", "eci_hi", "aa_name", "aa", "flagged"]
    return C.sort_values("cap", ascending=False)[cols].rename(columns={
        "cap": "capability_theta", "cap_lo": "capability_theta_lower", "cap_hi": "capability_theta_upper", "rel": "reliability_theta",
        "saf": "safety_theta", "eci": "eci_score", "aa": "aa_v4_intelligence_index"}).rename_axis("model").reset_index()


def triple(t):
    return [round(float(x), 4) for x in t]


def separated_pairs(v):
    """Pairs of rows [ours, eci, eci_lo, eci_hi] untied in both: {separated ECI intervals: [pairs, same order], overlapping: ...}."""
    r = {True: [0, 0], False: [0, 0]}
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if v[i, 0] == v[j, 0] or v[i, 1] == v[j, 1]:
                continue
            sep = bool(v[i, 2] > v[j, 3] or v[j, 2] > v[i, 3])
            r[sep][0] += 1
            r[sep][1] += (v[i, 0] > v[j, 0]) == (v[i, 1] > v[j, 1])
    return r


def agreement_numbers(M, E):
    """Headline agreement numbers. Keep the order of the boot calls: each interval depends on the draws before it."""
    rng = np.random.default_rng(0)
    C = M[M.cap.notna()]
    clean = C[~C.cap_star]
    out = {"flagged_excluded": sorted(C.index[C.cap_star])}
    for ref in ("eci", "aa"):
        d = clean.dropna(subset=[ref])
        out[ref] = {"n": len(d), "pra": triple(boot(pra, d.cap, d[ref], rng)), "pearson": triple(boot(pear, d.cap, d[ref], rng))}

    S = clean.dropna(subset=["eci", "aa"])
    out["ceiling"] = {"n": len(S)}
    for a, b, k in [("aa", "eci", "aa_vs_eci"), ("cap", "eci", "ours_vs_eci"), ("cap", "aa", "ours_vs_aa")]:
        out["ceiling"][k] = {"pra": triple(boot(pra, S[a], S[b], rng)), "pearson": triple(boot(pear, S[a], S[b], rng))}

    r = separated_pairs(clean.dropna(subset=["eci"])[["cap", "eci", "eci_lo", "eci_hi"]].values)
    out["eci_pairs"] = {"separated": r[True][0], "separated_agree": round(r[True][1] / r[True][0], 4),
                        "overlap": r[False][0], "overlap_agree": round(r[False][1] / r[False][0], 4)}

    out["top3"] = {"ours": list(clean.cap.sort_values(ascending=False).index[:3]),
                   "eci": list(clean.dropna(subset=["eci"]).eci.sort_values(ascending=False).index[:3]),
                   "aa": list(clean.dropna(subset=["aa"]).aa.sort_values(ascending=False).index[:3])}
    ours_eci = set(ECI_MAP.values())
    out["eci_top15_missing"] = [n for n in E.sort_values("eci", ascending=False).head(15).index if n not in ours_eci]

    out["distinct"] = {}
    for k in ("rel", "saf"):
        for ref in ("eci", "aa"):
            dd = M[M[k].notna() & ~M[k + "_star"]].dropna(subset=[ref])
            out["distinct"][f"{k}_{ref}"] = {"n": len(dd), "spearman": round(spear(dd[k], dd[ref]), 4)}

    # Frontier check: the 15 strongest shared models by ECI.
    FRONTIER_K = 15
    F = clean.dropna(subset=["eci"]).sort_values("eci", ascending=False).head(FRONTIER_K)
    fs = separated_pairs(F[["cap", "eci", "eci_lo", "eci_hi"]].values)[True]
    FA = F.dropna(subset=["aa"])
    out["frontier"] = {"k": FRONTIER_K, "eci_min": round(float(F.eci.min()), 2), "eci_max": round(float(F.eci.max()), 2),
                       "pra": triple(boot(pra, F.cap, F.eci, rng)), "pairs": FRONTIER_K * (FRONTIER_K - 1) // 2,
                       "separated": fs[0], "separated_agree": round(fs[1] / fs[0], 4),
                       "aa_vs_eci": {"n": len(FA), "pra": triple(boot(pra, FA.aa, FA.eci, rng))},
                       "ours_vs_eci_same": triple(boot(pra, FA.cap, FA.eci, rng))}
    out["frontier"]["pearson"] = triple(boot(pear, F.cap, F.eci, rng))
    out["frontier"]["aa_vs_eci"]["pearson"] = triple(boot(pear, FA.aa, FA.eci, rng))
    out["frontier"]["ours_vs_eci_same_pearson"] = triple(boot(pear, FA.cap, FA.eci, rng))
    return out


# ---------------------------------------------------------------- page JSON
def display_names():
    names = {}
    for p in sorted((FE / "models").glob("*.yaml")):
        y = yaml.safe_load(p.read_text())
        for e in y.get("evals") or []:
            names[e] = y["name"]
    return names


def lomo_rows(pmi, names):
    d = pmi[(pmi.scheme == "lomo") & pmi.score_observed.notna() & pmi.score_predicted.notna()]
    return [{"model": names[r.model], "index": r.index, "observed": round(r.score_observed, 2), "predicted": round(r.score_predicted, 2),
             "lower": round(r.score_lower, 2), "upper": round(r.score_upper, 2), "inside": bool(r.inside_90)} for r in d.itertuples()]


def eci_rows(agree, names):
    d = agree[~agree.flagged & agree.eci_score.notna() & agree.capability_theta.notna()]
    return [{"model": names[r.id], "type": r.type, "ours": round(SCALE(r.capability_theta), 1), "ours_lower": round(SCALE(r.capability_theta_lower), 1),
             "ours_upper": round(SCALE(r.capability_theta_upper), 1), "eci": round(r.eci_score, 2), "eci_lower": round(r.eci_lo, 2),
             "eci_upper": round(r.eci_hi, 2)} for r in d.itertuples()]


def r4(x):
    return None if x is None or pd.isna(x) else round(float(x), 4)


def numbers(pmi, summ, idx, task, agree, an, panel, names):
    by_id = dict(zip(agree.model, agree.id))
    disp = lambda m: names[by_id[m]]  # noqa: E731
    ptm = panel["per_task_models"]
    params = json.loads((ROOT / "src/complai/data/core.v1/params.json").read_text())
    closed = [m for m in panel["models"] if m.startswith("openrouter/")]
    s = summ.set_index("scheme")
    lomo_idx = idx[idx.scheme == "lomo"].set_index("index")
    ins_idx = idx[idx.scheme == "in_sample"].set_index("index")
    lt = task[task.scheme == "lomo"]
    worst = lt.loc[lt.mae.idxmax()]
    lomo_pm = pmi[pmi.scheme == "lomo"]
    drift = (lomo_pm.theta_observed - lomo_pm.theta_observed_production_calibration).abs()
    def boot(t):
        return {"est": t[0], "lo": t[1], "hi": t[2]}

    return {
        "panel": {
            "models": len(panel["models"]), "closed_models": len(closed), "open_models": len(panel["models"]) - len(closed),
            "closed_names": sorted(names[m] for m in closed),
            "closed_tasks": {names[m]: panel["tasks_per_model"][m] for m in closed},
            "responses": panel["responses"], "benchmarks": len(ptm),
            "items": sum(t["capacity"] for t in params["tasks"].values()),
            "core_items": params["budget"],
            "models_per_benchmark": {"min": min(ptm.values()), "min_task": min(ptm, key=ptm.get), "max": max(ptm.values()),
                                     "median": statistics.median(ptm.values())},
        },
        "lomo": {
            "models": int(s.loc["lomo", "models"]),
            "overall_mae": r4(s.loc["lomo", "overall_masked_mae"]), "overall_mae_in_sample": r4(s.loc["in_sample", "overall_masked_mae"]),
            "task_mae_mean": r4(s.loc["lomo", "task_mae_mean"]), "task_mae_median": r4(s.loc["lomo", "task_mae_median"]),
            "worst_task": worst.task, "worst_task_mae": r4(worst.mae),
            "max_drift_theta": r4(drift.max()),
            "index": {i: {"n": int(lomo_idx.loc[i, "models"]), "null": int(lomo_idx.loc[i, "null_theta"]),
                          "masked_mae": r4(lomo_idx.loc[i, "masked_mae"]), "score_mae": r4(lomo_idx.loc[i, "score_mae"]),
                          "score_max_ae": r4(lomo_idx.loc[i, "score_max_ae"]), "pairwise": r4(lomo_idx.loc[i, "pairwise_agreement"]),
                          "spearman": r4(lomo_idx.loc[i, "spearman"]), "coverage_90": r4(lomo_idx.loc[i, "coverage_90"]),
                          "coverage_90_in_sample": r4(ins_idx.loc[i, "coverage_90"])} for i in INDICES},
        },
        "agreement": {
            "eci": {"n": an["eci"]["n"], "pra": boot(an["eci"]["pra"]), "pearson": boot(an["eci"]["pearson"])},
            "aa": {"n": an["aa"]["n"], "pra": boot(an["aa"]["pra"]), "pearson": boot(an["aa"]["pearson"])},
            "ceiling": {"n": an["ceiling"]["n"], "pra": boot(an["ceiling"]["aa_vs_eci"]["pra"]),
                        "pearson": boot(an["ceiling"]["aa_vs_eci"]["pearson"]), "ours_vs_eci_pearson": boot(an["ceiling"]["ours_vs_eci"]["pearson"])},
            "eci_pairs": an["eci_pairs"],
            "top3": {k: [disp(m) for m in v] for k, v in an["top3"].items()},
            "top3_theta": {disp(m): r4(agree.set_index("model").loc[m, "capability_theta"]) for m in an["top3"]["ours"]},
            "eci_top15_missing": len(an["eci_top15_missing"]),
            "frontier": {**{k: an["frontier"][k] for k in ("k", "eci_min", "eci_max", "pairs", "separated", "separated_agree")},
                         "pra": boot(an["frontier"]["pra"]), "pearson": boot(an["frontier"]["pearson"]),
                         "aa_vs_eci": {"n": an["frontier"]["aa_vs_eci"]["n"], "pra": boot(an["frontier"]["aa_vs_eci"]["pra"]),
                                       "pearson": boot(an["frontier"]["aa_vs_eci"]["pearson"])},
                         "ours_vs_eci_same": boot(an["frontier"]["ours_vs_eci_same"]),
                         "ours_vs_eci_same_pearson": boot(an["frontier"]["ours_vs_eci_same_pearson"])},
            "distinct": {"reliability": {"rho": an["distinct"]["rel_eci"]["spearman"], "n": an["distinct"]["rel_eci"]["n"]},
                         "safety": {"rho": an["distinct"]["saf_eci"]["spearman"], "n": an["distinct"]["saf_eci"]["n"]}},
        },
    }


def main():
    names = display_names()
    M, aa, E, P = load()
    agree = agreement_table(M)
    an = agreement_numbers(M, E)
    L = OUT / "lomo"
    pmi = pd.read_csv(L / "per_model_index.csv")
    rows = {"lomo": lomo_rows(pmi, names), "eci": eci_rows(agree, names)}
    out = {
        "_meta": {"source": "tools/method/agreement.py (scripts run 2026-11-12 on core.v1, slope prior tau = 0.5)",
                  "eci_pulled": "2026-10-26"},
        **rows,
        "numbers": numbers(pmi, pd.read_csv(L / "summary.csv"), pd.read_csv(L / "index_summary.csv"), pd.read_csv(L / "task_summary.csv"),
                           agree, an, json.loads((OUT / "panel.json").read_text()), names),
    }
    dest = FE / "data/agreement.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out["numbers"], indent=1, ensure_ascii=False))
    print({k: len(v) for k, v in rows.items()}, "->", dest)


if __name__ == "__main__":
    main()
