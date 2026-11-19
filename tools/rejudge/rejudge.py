"""Re-grade existing eval logs with a different judge and compare the verdicts.

The evaluated model is never called: its stored outputs are re-scored with the
task's own scorer, with only the judge argument(s) swapped. The original judge's
verdicts in the log are the baseline.

Usage (from the repo root):

    .venv/bin/python tools/rejudge/rejudge.py test_logs/ \
        --judge openai/gpt-6-luna --judge openai/o3 --limit 200

Outputs (in --out, default `<input>/rejudge`):
    <log>__<judge>.eval  re-scored log (only the selected samples)
    summary.csv          one row per (log, judge, score key): agreement, kappa, means
    disagreements.csv    every sample where the new judge disagrees, with both
                         judges' explanations

Use a second, independent judge (e.g. o3) as a noise floor: if luna agrees with
the original judge about as often as o3 does, it is a drop-in replacement.
"""

import argparse
import asyncio
import csv
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai import score_async
from inspect_ai._util.registry import has_registry_params
from inspect_ai._util.registry import registry_info
from inspect_ai._util.registry import registry_params
from inspect_ai._util.registry import set_registry_info
from inspect_ai._util.registry import set_registry_params
from inspect_ai.log import EvalLog
from inspect_ai.log import EvalSample
from inspect_ai.log import read_eval_log
from inspect_ai.log import write_eval_log
from inspect_ai.model import GenerateConfig
from inspect_ai.model import ModelOutput
from inspect_ai.model._model_config import model_roles_to_model_roles_config
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from complai._cli.status import JUDGE_DEFAULTS
from complai._cli.utils import get_complai_tasks
from complai._cli.utils import instantiate_task_from_info


SKIPPED_TASKS = {"strong_reject": "parked (sends harmful prompts to the judge)"}

ErrorList = list[tuple[Any, int, str]]


def fix_mask_sample(sample: EvalSample) -> None:
    """Restore the types inspect_evals.mask's scorer expects but JSON logs lose."""
    from inspect_evals.mask.types import MetadataKey
    from inspect_evals.mask.types import QuestionArchetype
    from inspect_evals.mask.types import StoreKey

    sample.metadata[MetadataKey.CONFIG] = QuestionArchetype(
        sample.metadata[MetadataKey.CONFIG]
    )
    for key in (
        StoreKey.PRESSURED_RESPONSES,
        StoreKey.BELIEF_ELICIT_1_RESPONSES,
        StoreKey.BELIEF_ELICIT_2_AND_3_RESPONSES,
    ):
        if isinstance(outputs := sample.store.get(key), list):
            sample.store[key] = [
                ModelOutput.model_validate(o) if isinstance(o, dict) else o
                for o in outputs
            ]


SAMPLE_FIXUPS = {"mask": fix_mask_sample}

# Task args that newer task versions added, with the value older logs implicitly
# used. simpleqa_verified's logs predating `scorer` were graded by the original
# (letter-grade) scorer; the new default "tool" would not be comparable.
LEGACY_TASK_ARGS = {"simpleqa_verified": {"scorer": "original"}}


def slug(model: str) -> str:
    return model.replace("/", "_")


def tolerant(scorer: Scorer, errors: ErrorList) -> Scorer:
    """Wrap a scorer so one failed judge call skips the sample instead of the run."""

    async def score(state: TaskState, target: Target) -> Score | None:
        try:
            return await scorer(state, target)
        except Exception as ex:  # noqa: BLE001
            errors.append((state.sample_id, state.epoch, repr(ex)[:500]))
            return None

    set_registry_info(score, registry_info(scorer))
    if has_registry_params(scorer):
        set_registry_params(score, registry_params(scorer))
    return score


def build_scorers(
    log: EvalLog, judge: str, judge_args: list[str] | None
) -> tuple[list[Scorer], dict[str, Any], Any]:
    """Instantiate the log's task with the judge argument(s) replaced."""
    task_name = log.eval.task
    args = JUDGE_DEFAULTS[task_name]
    names = [n for n in judge_args if n in args] if judge_args else list(args)
    if not names:
        raise ValueError(f"--judge-arg matches none of {task_name}'s {list(args)}")
    if task_name == "self_check_consistency" and "argumentation_model" in names:
        print(
            "  note: self_check_consistency's argumentation_model generates text, so "
            "this is not a pure re-judge. Pass --judge-arg judge_model to keep it."
        )

    task_args = {
        **LEGACY_TASK_ARGS.get(task_name, {}),
        **log.eval.task_args,
        **{name: judge for name in names},
    }
    infos = get_complai_tasks(filter=lambda t: t.name == task_name)
    if not infos:
        raise ValueError(f"Task {task_name} not found in complai")
    task = instantiate_task_from_info(infos[0], task_args)
    roles = model_roles_to_model_roles_config(task.model_roles)
    return list(task.scorer or []), task_args, roles


def select_samples(
    samples: list[EvalSample], limit: int | None, seed: int
) -> list[EvalSample]:
    if limit is None or limit >= len(samples):
        return samples
    idx = sorted(random.Random(seed).sample(range(len(samples)), limit))
    return [samples[i] for i in idx]


async def rejudge(
    log: EvalLog,
    samples: list[EvalSample],
    judge: str,
    judge_args: list[str] | None,
    display: str,
) -> tuple[EvalLog, ErrorList]:
    scorers, task_args, roles = build_scorers(log, judge, judge_args)
    errors: ErrorList = []

    # score_async instantiates the evaluated model from the header. Point it at
    # mockllm so no evaluated-model provider (vllm, batch APIs, ...) is touched.
    samples = [s.model_copy(deep=True) for s in samples]
    if fixup := SAMPLE_FIXUPS.get(log.eval.task):
        for sample in samples:
            fixup(sample)
    header = log.model_copy(
        update={"eval": log.eval.model_copy(deep=True), "samples": samples}
    )
    header.eval.model = "mockllm/model"
    header.eval.model_args = {}
    header.eval.model_generate_config = GenerateConfig()
    header.eval.model_roles = roles

    new_log = await score_async(
        header,
        [tolerant(s, errors) for s in scorers],
        action="overwrite",
        display=display,  # type: ignore[arg-type]
    )
    new_log.eval.model = log.eval.model
    new_log.eval.model_args = log.eval.model_args
    new_log.eval.model_generate_config = log.eval.model_generate_config
    new_log.eval.task_args = task_args
    return new_log, errors


def flatten(value: Any, like: Any = None) -> dict[str, Any]:
    """Score value as a dict. A scalar compared against a dict value takes the
    dict's first key (older hle logs stored a bare 0/1, newer ones {"score": ...})."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(like, dict) and like:
        return {next(iter(like)): value}
    return {"value": value}


def is_categorical(values: list[Any]) -> bool:
    return all(v is None or isinstance(v, (str, bool)) or v in (0, 1) for v in values)


def cohen_kappa(a: list[Any], b: list[Any]) -> float:
    n = len(a)
    if n == 0:
        return math.nan
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def compare(
    orig: list[EvalSample], new: list[EvalSample], scorer_names: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-key agreement rows and per-sample disagreement rows."""
    new_by_key = {(s.id, s.epoch): s for s in new}
    summary: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for name in scorer_names:
        pairs: list[tuple[EvalSample, Score, Score]] = []
        for s in orig:
            n = new_by_key.get((s.id, s.epoch))
            o_score = (s.scores or {}).get(name)
            n_score = (n.scores or {}).get(name) if n else None
            if o_score is not None and n_score is not None:
                pairs.append((s, o_score, n_score))
        if not pairs:
            summary.append({"scorer": name, "key": "-", "n": 0})
            continue
        keys = sorted({k for _, o, n in pairs for k in flatten(o.value, n.value)})
        for key in keys:
            # List values (e.g. self_check's per-sentence verdicts) are compared
            # element-wise; each element becomes its own item.
            items: list[tuple[EvalSample, Score, Score, Any, Any, int | None]] = []
            for s, o, n in pairs:
                x = flatten(o.value, n.value).get(key)
                y = flatten(n.value, o.value).get(key)
                if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
                    items += [(s, o, n, a, b, i) for i, (a, b) in enumerate(zip(x, y))]
                else:
                    items.append((s, o, n, x, y, None))
            ov = [it[3] for it in items]
            nv = [it[4] for it in items]
            row: dict[str, Any] = {"scorer": name, "key": key, "n": len(pairs)}
            if len(items) != len(pairs):
                row["n_items"] = len(items)
            if is_categorical(ov + nv):
                row["agreement"] = mean([float(x == y) for x, y in zip(ov, nv)])
                row["kappa"] = cohen_kappa(ov, nv)
                if all(isinstance(v, (int, float)) for v in ov + nv):
                    row["orig_mean"], row["new_mean"] = mean(ov), mean(nv)  # type: ignore[arg-type]
                else:
                    row["orig_dist"] = dict(Counter(ov).most_common())
                    row["new_dist"] = dict(Counter(nv).most_common())
                    row["confusion"] = {
                        f"{x}->{y}": c
                        for (x, y), c in Counter(zip(ov, nv)).items()
                        if x != y
                    }
                for s, o, n, x, y, i in items:
                    if x != y:
                        disagreements.append(
                            {
                                "sample_id": s.id,
                                "epoch": s.epoch,
                                "scorer": name,
                                "key": key if i is None else f"{key}[{i}]",
                                "orig": x,
                                "new": y,
                                "orig_explanation": o.explanation,
                                "new_explanation": n.explanation,
                                "answer": (o.answer or "")[:2000],
                            }
                        )
            else:
                both = [
                    (float(x), float(y))
                    for x, y in zip(ov, nv)
                    if x is not None and y is not None
                ]
                # Score.unscored() (e.g. cab's unparseable judge output) is NaN.
                unscored = sum(math.isnan(x) or math.isnan(y) for x, y in both)
                both = [(x, y) for x, y in both if not (math.isnan(x) or math.isnan(y))]
                if unscored:
                    row["unscored"] = unscored
                row["exact_agreement"] = mean([float(x == y) for x, y in both])
                row["mean_abs_diff"] = mean([abs(x - y) for x, y in both])
                row["orig_mean"] = mean([x for x, _ in both])
                row["new_mean"] = mean([y for _, y in both])
            summary.append(row)
    return summary, disagreements


def fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_rows(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        stats = ", ".join(
            f"{k}={fmt(v)}" for k, v in r.items() if k not in ("scorer", "key")
        )
        print(f"    {r['scorer']}.{r['key']}: {stats}")


def print_metrics(prefix: str, log: EvalLog) -> None:
    for r in log.results.scores if log.results else []:
        metrics = ", ".join(f"{k}={fmt(m.value)}" for k, m in r.metrics.items())
        print(f"  {prefix} {r.name}: {metrics}")


def find_logs(paths: list[Path]) -> list[Path]:
    logs: list[Path] = []
    for p in paths:
        logs.extend(sorted(p.rglob("*.eval")) if p.is_dir() else [p])
    return [p for p in logs if "rejudge" not in p.parts]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path, help=".eval files or dirs")
    parser.add_argument(
        "--judge", action="append", required=True, help="Judge model (repeatable)"
    )
    parser.add_argument(
        "--judge-arg",
        action="append",
        help="Only replace these task judge args (default: all of the task's), "
        "e.g. --judge-arg numeric_judge_model",
    )
    parser.add_argument("--limit", type=int, help="Random samples per log")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="Output dir")
    parser.add_argument("--display", default="rich", choices=["rich", "plain", "none"])
    args = parser.parse_args()

    logs = find_logs(args.paths)
    if not logs:
        sys.exit("No .eval logs found")
    first = args.paths[0]
    out = args.out or (first if first.is_dir() else first.parent) / "rejudge"
    out.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict[str, Any]] = []
    all_disagreements: list[dict[str, Any]] = []
    for path in logs:
        log = read_eval_log(str(path))
        task = log.eval.task
        print(f"\n{path.name}\n  task={task} model={log.eval.model}")
        if task in SKIPPED_TASKS:
            print(f"  skipped: {SKIPPED_TASKS[task]}")
            continue
        if task not in JUDGE_DEFAULTS:
            print("  skipped: task has no judge model")
            continue
        if log.status != "success" or not log.samples:
            print(f"  skipped: status={log.status}")
            continue
        orig_judges = {k: log.eval.task_args.get(k) for k in JUDGE_DEFAULTS[task]}
        samples = select_samples(log.samples, args.limit, args.seed)
        print(f"  original judge: {orig_judges}  samples: {len(samples)}")
        scorer_names = list(
            dict.fromkeys(r.name for r in (log.results.scores if log.results else []))
        )
        print_metrics(f"original metrics (all {len(log.samples)} samples)", log)

        for judge in args.judge:
            print(f"  -> judge {judge}")
            try:
                new_log, errors = asyncio.run(
                    rejudge(log, samples, judge, args.judge_arg, args.display)
                )
            except Exception as ex:  # noqa: BLE001
                print(f"    failed: {ex!r}"[:600])
                new_log, errors = None, [(None, None, repr(ex)[:500])] * len(samples)
            summary, disagreements = compare(
                samples, (new_log.samples if new_log else None) or [], scorer_names
            )
            if errors:
                print(
                    f"    {len(errors)} judge errors (samples skipped), "
                    f"e.g. {errors[0][2]}"
                )
            print_rows(summary)
            if new_log:
                write_eval_log(new_log, str(out / f"{path.stem}__{slug(judge)}.eval"))
                print_metrics(f"  new metrics ({len(samples)} samples)", new_log)
            meta = {
                "log": path.name,
                "task": task,
                "model": log.eval.model,
                "orig_judge": "|".join(str(v) for v in orig_judges.values()),
                "judge": judge,
                "errors": len(errors),
            }
            all_summary += [{**meta, **r} for r in summary]
            all_disagreements += [{**meta, **d} for d in disagreements]

    write_csv(out / "summary.csv", all_summary)
    write_csv(out / "disagreements.csv", all_disagreements)
    print(f"\nWrote summary.csv, disagreements.csv and re-scored logs to {out}")


if __name__ == "__main__":
    main()
