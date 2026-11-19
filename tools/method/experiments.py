"""Build frontend/public/data/experiments.json: plot rows + every number quoted in the homepage results explorer.

Inputs (all read-only):
  frontend/public/data/predictions.json      index / subcategory thetas
  frontend/public/data/sample_analysis.json  per model x task outcome tags (provider_block, provider_block_passed, ...)
  frontend/public/data/labels.json           subcategory display names
  frontend/public/models/*.yaml              release date, open weights, organisation, parameter counts

Findings (page order) -> json keys:
  1 open_lag + frontier Capability, reliability and safety over time (lag per index; frontier
                        gains over the past year are quoted in the same finding)
  2 capability_safety   Is the most capable model the safest?
  3 safety_areas        Which parts of safety are improving? (the scored safety areas in two groups, by release date)
  4 params              Total or active parameters: which buys capability?
  5 labs                How do labs compare on safety?

The evaluation-cost finding reads frontend/public/data/costs.json (built by complai export-web).

content_filter() is kept for reference but no longer exported: the content-filter finding was
dropped on 2026-11-09 (see BACKLOG.md, Frontend). manipulation() (area vs Capability) is likewise
kept but unexported: finding 2 became safety_areas() on 2026-11-12.

Scores are on the site's scale (100 + 15 * theta). "| release" = Spearman partial correlation with
release date (months) held fixed. Missing metadata is never filled in: models without it drop out
of the analyses that need it, and the counts say how many were used.

Run:  .venv/bin/python -W ignore tools/method/experiments.py
      .venv/bin/python -W ignore tools/method/experiments.py --verify   also compare headline numbers with the
                                                               values signed off on 2026-11-08
"""
import glob
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
FE = HERE.parents[1] / "frontend/public"
OUT = FE / "data/experiments.json"
P = json.load(open(FE / "data/predictions.json"))
S = json.load(open(FE / "data/sample_analysis.json"))
IX = ("capability", "reliability", "safety")
EPOCH = pd.Timestamp("2024-01-01")
MONTH = 30.44

# Finding 2: models labelled on the scatter (by display name, asserted to exist).
LABELS = ["Gemini 4 Argon", "Claude Opus 5.5", "GPT-6.1 Terra", "gpt-oss-20b", "Claude Sonnet 4.5", "Hy3",
          "Gemma 3 27B Instruct", "DeepSeek-V4-Flash", "Kimi-K2.6", "DeepSeek-V4.1-Flash"]
# manipulation() (unexported): safety subcategories split into resisting attacks vs. behaving well when asked.
ATTACK = ["safety:security-privacy", "safety:goal-hijacking", "safety:prompt-injection"]
BEHAVIOUR = ["safety:fairness-bias", "safety:deception-honesty", "safety:harmful-instruction-refusal"]
# Finding 3 groups: areas whose benchmarks move together across models (task-level partial correlations,
# areas_explore.py B). Jailbreak resistance is not scored (too few CORE items) and is left out. Fairness is
# left out (2026-11-12): many older models skipped one of its benchmarks, so its scores mix benchmark sets.
AREA_GROUPS = [("injection", "Injection attacks", ["safety:security-privacy", "safety:goal-hijacking", "safety:prompt-injection"]),
               ("refusal", "Refusal and honesty", ["safety:harmful-instruction-refusal", "safety:deception-honesty"])]
BOOT, BOOT_SEED = 2000, 0
# content_filter() (unexported): tasks offered in the selector (first is the default).
FILTER_TASKS = ["strong_reject", "ifbench", "mmlu_pro_robustness"]
FILTER_ORG = "Anthropic"
# Finding 5: labs with fewer models are pooled into "Other".
LAB_MIN = 2

# Values regenerated 2026-11-11 (finding 2 on weighted area predicted_score; Qwen3.8-Flash card now proprietary); signed off by user 2026-11-11 (previous 2026-11-10); --verify reports any drift.
EXPECTED = {
    "capability_safety.safety.rho": 0.83, "capability_safety.safety.rho_release": 0.77,
    "capability_safety.safety.n": 55, "capability_safety.safety.top10_overlap": 6,
    # Finding 3 (safety_areas) keys added 2026-11-12, NOT YET SIGNED OFF: group pp per year by release date
    # (areas scored only from benchmarks the model ran; fairness dropped).
    "safety_areas.injection": 17.1, "safety_areas.refusal": 1.4,
    "open_lag.capability.lag_months": 3.1, "open_lag.reliability.lag_months": 4.9, "open_lag.safety.lag_months": 12.6,
    "params.fit.per_10x_total": 7.2, "params.fit.per_10x_active": 0.2, "params.n_points": 43, "params.n_points_moe": 30, "params.fit.n": 43,
    "labs.safety.OpenAI": 13.0, "labs.safety.Anthropic": 7.9, "labs.safety.DeepSeek": -6.9,
    "frontier.capability.start": 110.2, "frontier.capability.end": 130.9,
    "frontier.reliability.start": 113.1, "frontier.reliability.end": 144.0,
    "frontier.safety.start": 125.5, "frontier.safety.end": 130.6,
    "frontier.capability.slope": 2.4, "frontier.reliability.slope": 3.48, "frontier.safety.slope": 0.53,
}


def r(x, k=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def iso(ts):
    return ts.date().isoformat()


def pcorr(d, x, y, z):
    """Spearman partial correlation of x and y with z held fixed (rank residualisation)."""
    d = d[[x, y] + z].dropna()
    R = d.apply(rankdata)
    Z = np.c_[np.ones(len(R)), R[z].values]
    rx = R[x] - Z @ np.linalg.lstsq(Z, R[x], rcond=None)[0]
    ry = R[y] - Z @ np.linalg.lstsq(Z, R[y], rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), len(d)


def ols(y, *xs):
    X = np.c_[np.ones(len(y)), *xs]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, y - X @ b


def best_so_far(d, col):
    """Best score among models released on or before each model's date (tie-safe, unlike cummax)."""
    return np.array([d[col][d.release <= x].max() for x in d.release])


def steps(d, col):
    """Running best by release date: one row per model that set a new best."""
    out, best = [], -np.inf
    for row in d.dropna(subset=[col, "release"]).sort_values(["release", col], ascending=[True, False]).itertuples():
        v = getattr(row, col)
        if v > best:
            best = v
            out.append({"id": row.id, "date": iso(row.release), "score": r(v, 1)})
    return out


# ------------------------------------------------------------------------------------------- data
# One row per model: predictions, sample_analysis outcome tags and model cards joined.


def cards():
    out = {}
    for f in glob.glob(f'{FE}/models/*.y*ml'):
        c = yaml.safe_load(open(f))
        for e in c.get('evals') or []:
            out[e] = c
    return out


def num(v):
    if isinstance(v, str):
        try: return float(v.replace('_', ''))
        except ValueError: return np.nan
    return np.nan if v is None else float(v)


def frame():
    C = cards(); rows = []
    for mid, m in P['models'].items():
        c = C.get(mid, {}); sp = c.get('specs') or {}
        r = {'id': mid, 'name': c.get('name', mid.split('/')[-1]), 'org': c.get('organization'),
             'release': pd.to_datetime(str(c.get('release_date')), errors='coerce'),
             'open': c.get('open_weights'),
             'region': ((c.get('region') or {}).get('region_developed') or [None])[0],
             'total_p': num(sp.get('total_params')), 'active_p': num(sp.get('active_params')),
             'arch': sp.get('architecture'),
             'max_tokens': (m.get('model_config') or {}).get('max_tokens'),
             'effort': (m.get('model_config') or {}).get('reasoning_effort')}
        for ix in ('capability', 'reliability', 'safety'):
            it = ((m.get('indices') or {}).get(ix) or {}).get('index_theta') or {}
            r[ix] = it.get('theta'); iv = it.get('interval') or [None, None]
            r[ix+'_lo'], r[ix+'_hi'] = iv
        for sc, v in (m.get('subcategories') or {}).items():
            r['sub:'+sc] = v.get('ability')
            r['area:'+sc] = v.get('predicted_score')
        for t, v in (m.get('tasks') or {}).items():
            r['task:'+t] = v.get('predicted_score')
            r['obs:'+t] = v.get('observed_subset_score')
        o = (S['model_outcomes'].get(mid) or {}).get('outcomes') or {}
        for k, v in o.items(): r['out:'+k] = v
        for t, v in (S['models'].get(mid) or {}).items():
            a = v.get('answers') or 0
            if not a: continue
            for tag, n in (v.get('tags') or {}).items(): r[f'tag:{t}:{tag}'] = n / a
        rows.append(r)
    df = pd.DataFrame(rows)
    for ix in ('capability', 'reliability', 'safety'): df[ix+'_s'] = 100 + 15*df[ix]
    return df


def load():
    df = frame()
    df["t"] = (df.release - EPOCH).dt.days / MONTH
    df["is_open"] = df.open.map({True: True, False: False})
    return df


# ------------------------------------------------------------------------------------------- findings


def capability_safety(df):
    names = dict(zip(df.name, df.id))
    miss = [n for n in LABELS if n not in names]
    assert not miss, f"label names not found: {miss}"
    out = {"labels": [names[n] for n in LABELS]}
    for y in ("safety", "reliability"):
        d = df.dropna(subset=["capability", y])
        rho = spearmanr(d.capability, d[y])[0]
        rho_t, n_t = pcorr(df, "capability", y, ["t"])
        top = lambda c: set(d.nlargest(10, c).id)  # noqa: E731
        out[y] = {"rho": r(rho), "rho_release": r(rho_t), "n": len(d), "n_release": n_t,
                  "top10_overlap": len(top("capability") & top(y))}
    return out


def manipulation(df):
    names = json.load(open(f"{FE}/data/labels.json"))["_schema"]["subcategories"]
    bars = []
    for group, subs in (("attack", ATTACK), ("behaviour", BEHAVIOUR)):
        for s in subs:
            rho_t, n = pcorr(df, "capability", "area:" + s, ["t"])
            raw = df[["capability", "area:" + s]].dropna()
            bars.append({"id": s, "name": names[s]["name"], "group": group, "rho_release": r(rho_t),
                         "rho": r(spearmanr(raw.capability, raw["area:" + s])[0]), "n": n})
    # Area score: the pool's weighted predicted_score (index discrimination x population share), in %.
    points = {s: [{"id": row.id, "x": r(row.capability_s, 1), "y": r(100 * row.v, 1)}
                  for row in df.assign(v=df["area:" + s]).dropna(subset=["capability_s", "v"]).itertuples()]
              for s in ATTACK + BEHAVIOUR}
    mean = lambda g: r(np.mean([b["rho_release"] for b in bars if b["group"] == g]))  # noqa: E731
    return {"bars": bars, "points": points, "mean": {"attack": mean("attack"), "behaviour": mean("behaviour")}}


def safety_areas(df):
    """The scored safety areas in two groups (equal-weight mean of each group's area predicted_score, %)
    by release date: OLS trend with a bootstrap 90% interval."""
    names = json.load(open(f"{FE}/data/labels.json"))["_schema"]["subcategories"]
    rng = np.random.default_rng(BOOT_SEED)
    groups = []
    for gid, gname, subs in AREA_GROUPS:
        cols = ["area:" + s for s in subs]
        d = df.dropna(subset=cols + ["release"]).assign(pct=lambda x: 100 * x[cols].mean(axis=1))
        b, _ = ols(d.pct.values, d.t.values)
        slope_boot = []
        for _ in range(BOOT):
            bd = d.iloc[rng.integers(0, len(d), len(d))]
            slope_boot.append(12 * ols(bd.pct.values, bd.t.values)[0][1])
        lo, hi = d.loc[d.t.idxmin()], d.loc[d.t.idxmax()]
        members = []
        for s, c in zip(subs, cols):
            members.append({"id": s, "name": names[s]["name"], "pp_per_year": r(12 * ols(100 * d[c].values, d.t.values)[0][1], 1)})
        groups.append({
            "id": gid, "name": gname, "n": len(d), "members": members,
            "pp_per_year": r(12 * b[1], 1),
            "pp_per_year_ci90": [r(np.nanpercentile(slope_boot, 5), 1), r(np.nanpercentile(slope_boot, 95), 1)],
            "trend": [{"date": iso(x.release), "y": r(b[0] + b[1] * x.t, 1)} for x in (lo, hi)],
            "points": [{"id": x.id, "date": iso(x.release), "y": r(x.pct, 1)} for x in d.itertuples()],
        })
    return {"groups": groups, "boot": BOOT}


def open_lag(df):
    out, last = {}, df.release.max()
    for ix in IX:
        c = ix + "_s"
        o = df[df.is_open == True].dropna(subset=[c, "release"]).sort_values("release")  # noqa: E712
        k = df[df.is_open == False].dropna(subset=[c, "release"])  # noqa: E712
        fr = best_so_far(o, c)
        rec = (o.t >= o.t.max() - 12).values
        slope = np.polyfit(o.t[rec], fr[rec], 1)[0]
        gap = k[c].max() - o[c].max()
        observed = []
        for row in k.sort_values("release").itertuples():
            m = o[(o[c] >= getattr(row, c)) & (o.release >= row.release)].sort_values("release")
            if len(m):
                observed.append({"id": row.id, "matched_by": m.id.iloc[0],
                                 "months": r((m.release.iloc[0] - row.release).days / MONTH, 1)})
            else:
                observed.append({"id": row.id, "matched_by": None, "months": r((last - row.release).days / MONTH, 1)})
        unmatched = [x for x in observed if x["matched_by"] is None]
        out[ix] = {
            "open_steps": steps(o, c), "closed_steps": steps(k, c),
            "best_open": {"id": o.loc[o[c].idxmax(), "id"], "score": r(o[c].max(), 1)},
            "best_closed": {"id": k.loc[k[c].idxmax(), "id"], "score": r(k[c].max(), 1)},
            "gap": r(gap, 1), "open_slope": r(slope), "lag_months": r(gap / slope, 1),
            # The oldest closed model no open model has matched yet: how long the gap has already lasted.
            "oldest_unmatched": min(unmatched, key=lambda x: -x["months"]) if unmatched else None,
            "observed": observed,
        }
    scored = df.dropna(subset=[ix + "_s" for ix in IX] + ["release"])
    out["n_open"], out["n_closed"] = int((scored.is_open == True).sum()), int((scored.is_open == False).sum())  # noqa: E712
    out["as_of"] = iso(last)
    return out


def params(df):
    d = df[df.is_open == True].dropna(subset=["total_p", "active_p", "capability_s"]).copy()  # noqa: E712
    d["moe"] = d.total_p > 1.05 * d.active_p
    points = [{"id": x.id, "total": float(x.total_p), "active": float(x.active_p), "moe": bool(x.moe),
               "capability": r(x.capability_s, 1)} for x in d.itertuples()]
    f = d.dropna(subset=["capability_s", "t"])
    lT, lA = np.log10(f.total_p), np.log10(f.active_p)
    b_both, _ = ols(f.capability_s, lT, lA, f.t)
    b_tot, _ = ols(f.capability_s, lT, f.t)
    b_act, _ = ols(f.capability_s, lA, f.t)
    moe_t = f[f.moe].release.median(), f[~f.moe].release.median()
    return {"points": points, "n_points": len(points), "n_points_moe": int(d.moe.sum()), "fit": {
        "n": len(f), "n_moe": int(f.moe.sum()), "per_10x_total": r(b_both[1], 1), "per_10x_active": r(b_both[2], 1),
        "per_10x_total_alone": r(b_tot[1], 1), "per_10x_active_alone": r(b_act[1], 1),
        "rho_total_release": r(pcorr(f.assign(lT=lT), "lT", "capability", ["t"])[0]),
        "rho_active_release": r(pcorr(f.assign(lA=lA), "lA", "capability", ["t"])[0]),
        "median_release_moe": iso(moe_t[0]), "median_release_dense": iso(moe_t[1])}}


def content_filter(df):
    org = dict(zip(df.id, df.org))
    task_index = json.loads((Path(FE) / "data/labels.json").read_text())
    out = {"org": FILTER_ORG, "tasks": FILTER_TASKS, "index": {t: task_index[t]["index"] for t in FILTER_TASKS}}
    for t in FILTER_TASKS:
        rows = []
        for mid, tasks in S["models"].items():
            v = tasks.get(t)
            if not v or not v.get("answers"):
                continue
            g = v.get("tags") or {}
            rows.append({"id": mid, "answers": v["answers"], "blocked_failed": g.get("provider_block", 0),
                         "blocked_passed": g.get("provider_block_passed", 0)})
        for x in rows:
            x["share"] = r((x["blocked_failed"] + x["blocked_passed"]) / x["answers"], 4)
        others = [x["share"] for x in rows if org.get(x["id"]) != FILTER_ORG]
        out[t] = {"rows": rows, "others_n": len(others), "others_max": r(max(others), 4),
                  "others_mean": r(np.mean(others), 4)}
    return out


def labs(df):
    out = {}
    for y in ("safety", "reliability"):
        d = df.dropna(subset=["capability_s", y + "_s", "t"]).copy()
        b, d["resid"] = ols(d[y + "_s"], d.capability_s, d.t)
        count = d.org.value_counts()
        d["lab"] = d.org.where(d.org.map(count) >= LAB_MIN, "Other")
        g = d.groupby("lab").resid.agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
        out[y] = {"n": len(d), "per_capability_point": r(b[1]), "per_month": r(b[2]),
                  "rows": [{"id": x.id, "lab": x.lab, "resid": r(x.resid, 1)} for x in d.itertuples()],
                  "labs": [{"lab": k, "mean": r(v["mean"], 1), "median": r(v["median"], 1), "n": int(v["count"])}
                           for k, v in g.iterrows()]}
    return out


def frontier(df):
    out, last = {}, df.release.max()
    start = last - pd.DateOffset(months=12)
    for ix in IX:
        c = ix + "_s"
        st = steps(df, c)
        before = [s for s in st if pd.Timestamp(s["date"]) <= start]
        d = df.dropna(subset=[c, "release"]).sort_values("release")
        fr = best_so_far(d, c)
        rec = (d.t >= d.t.max() - 12).values
        out[ix] = {"steps": st, "start": before[-1] if before else None, "end": st[-1],
                   "delta": r(st[-1]["score"] - before[-1]["score"], 1) if before else None,
                   "slope": r(np.polyfit(d.t[rec], fr[rec], 1)[0])}
    out["window"] = {"from": iso(start), "to": iso(last)}
    return out


# ------------------------------------------------------------------------------------------- main


def headline(J):
    """Flatten the numbers listed in EXPECTED for --verify."""
    name = {}
    for f in Path(f"{FE}/models").glob("*.y*ml"):
        c = yaml.safe_load(open(f))
        for e in c.get("evals") or []:
            name[e] = c.get("name")
    got = {}
    for y in ("safety",):
        for k in ("rho", "rho_release", "n", "top10_overlap"):
            got[f"capability_safety.{y}.{k}"] = J["capability_safety"][y][k]
    for g in J["safety_areas"]["groups"]:
        got[f"safety_areas.{g['id']}"] = g["pp_per_year"]
    for ix in IX:
        got[f"open_lag.{ix}.lag_months"] = J["open_lag"][ix]["lag_months"]
        fr = J["frontier"][ix]
        got[f"frontier.{ix}.start"], got[f"frontier.{ix}.end"] = fr["start"]["score"], fr["end"]["score"]
        got[f"frontier.{ix}.slope"] = fr["slope"]
    for k in ("per_10x_total", "per_10x_active", "n"):
        got[f"params.fit.{k}"] = J["params"]["fit"][k]
    got["params.n_points"], got["params.n_points_moe"] = J["params"]["n_points"], J["params"]["n_points_moe"]
    for x in J["labs"]["safety"]["labs"]:
        got[f"labs.safety.{x['lab']}"] = x["mean"]
    return got


def main():
    df = load()
    J = {
        "_meta": {
            "built": date.today().isoformat(),
            "generator": "tools/method/experiments.py",
            "prediction_id": P.get("prediction_id"), "core_release": P.get("subset", {}).get("id")
            if isinstance(P.get("subset"), dict) else P.get("subset"),
            "analysis_rules_version": S.get("rules_version"),
            "n_models": len(df), "n_scored": int(df.capability.notna().sum()),
            "scale": "100 + 15 * theta", "release_partial": "Spearman partial correlation with release month held fixed",
        },
        "capability_safety": capability_safety(df),
        "safety_areas": safety_areas(df),
        "open_lag": open_lag(df),
        "params": params(df),
        "labs": labs(df),
        "frontier": frontier(df),
    }
    OUT.write_text(json.dumps(J, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    if "--verify" in sys.argv:
        got, bad = headline(J), 0
        for k, want in EXPECTED.items():
            have = got.get(k)
            tol = 0.5 if "usd" in k else 0.0051 if isinstance(want, float) and abs(want) < 1 else 0.051
            ok = have is not None and abs(have - want) <= tol
            bad += not ok
            print(f"  {'ok ' if ok else 'DIFF'} {k:58s} expected {want!s:>8} got {have}")
        print(f"{len(EXPECTED) - bad}/{len(EXPECTED)} match")


if __name__ == "__main__":
    main()
