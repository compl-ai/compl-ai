"""Rule-based failure tags for every answer in preprocessed eval logs.

For each preprocessed ``samples.jsonl`` this reads the original ``.eval`` logs
listed as ``ok`` in its manifest and writes ``sample_analysis.jsonl`` next to it:
one row per answer (sample x epoch), with a few facts read from the log and one
tag saying how the answer went. Rows join 1:1 to ``samples.jsonl`` on
``(file_path, sample_id, epoch)`` (strong_reject jailbreak variants share a
``sample_id``, as they do in ``samples.jsonl``).

The tags are descriptive metadata only. Nothing here changes scores,
predictions or indices.

The logs are read as raw JSON from the zip, not through Inspect's reader, which
is far slower on large logs. If Inspect changes its log layout this reader
must be updated (see ``read_eval``).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Iterable, Optional

import typer

import inspect_ai._util.zipfile  # noqa: F401  (registers zstd support in zipfile)

from complai.utils.log_parser import logical_sample_id
from complai.utils.scores import SCORE_ALIASES, normalize_score, select_score


RULES_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "tools" / "minify" / "data"
OUTPUT_NAME = "sample_analysis.jsonl"
OUTPUT_MANIFEST_NAME = "sample_analysis.manifest.json"

# Text that OpenRouter / providers return in place of an answer when a request is blocked.
BLOCK_MARKERS = ("reduce refusals for your users", "flagged for possible")
# zlib compression ratio of the last 4,000 characters below which output counts as a loop.
LOOP_RATIO = 0.15
LOOP_WINDOW = 4000
LOOP_MIN_CHARS = 1000
# An answer counts as cut off if it used at least this share of max_tokens.
CAP_SHARE = 0.98
PASS_THRESHOLD = 0.5
NORMAL_STOPS = ("stop", "max_tokens", "content_filter", "model_length", "tool_calls")
# Scorers of these tasks never fill Score.answer, so an empty answer there means nothing.
NO_ANSWER_FIELD = {
    "cab", "fairllm", "imdb_contrast", "llm_rules", "mask",
    "mmlu_pro_robustness", "self_check_consistency", "strong_reject",
}

# One tag per answer. Passed answers (value >= 0.5) get one of the first two;
# failed answers get the first matching tag in the order below.
TAGS: dict[str, str] = {
    "passed": "Scored as correct.",
    "provider_block_passed": "Scored as correct, but at least one model call was blocked by the provider "
                             "(content filter or block message), e.g. a refusal that the benchmark counts as right.",
    "sample_error": "The sample ended with an error, so the model never finished.",
    "sample_limit": "The sample hit an Inspect limit (time, tokens, messages, ...).",
    "provider_block": "At least one model call was blocked by the provider (content filter or block message).",
    "looping": "The model repeated itself: the end of its output compresses to under "
               f"{LOOP_RATIO:.0%} of its size.",
    "max_tokens": "At least one model call was cut off at the max_tokens limit.",
    "context_length": "At least one model call ran out of context window.",
    "stopped_early": "At least one model call stopped for an unusual reason (not stop, max_tokens, "
                     "content_filter, model_length or tool_calls) without hitting the token limit.",
    "empty_output": "The model returned no visible text (or there were no model calls).",
    "no_final_answer": "The model answered, but the scorer found no final answer in it.",
    "wrong": "The model gave an answer and it was scored as wrong.",
    "unscored": "No usable score for the configured scorer.",
}

# Coarser groups of TAGS for display; every tag is in exactly one group.
OUTCOMES: dict[str, dict[str, Any]] = {
    "correct": {
        "label": "Correct",
        "tags": ["passed", "provider_block_passed"],
    },
    "wrong": {
        "label": "Wrong",
        "tags": ["wrong"],
    },
    "did_not_finish": {
        "label": "Didn't finish",
        "tags": ["sample_error", "sample_limit", "looping", "max_tokens", "context_length",
                 "stopped_early", "empty_output", "no_final_answer"],
    },
    "blocked": {
        "label": "Blocked by provider",
        "tags": ["provider_block"],
    },
    "unscored": {
        "label": "Unscored",
        "tags": ["unscored"],
    },
}


# --------------------------------------------------------------------------- rules


def tag_answer(value: float | None, facts: dict[str, Any], task: str) -> str:
    """Return the single tag for one answer from its score and log facts."""
    if value is None:
        return "unscored"
    flags = set(facts.get("flags", ()))
    if value >= PASS_THRESHOLD:
        return "provider_block_passed" if "block" in flags else "passed"
    if facts.get("error"):
        return "sample_error"
    if facts.get("limit"):
        return "sample_limit"
    for flag, tag in (
        ("block", "provider_block"),
        ("loop", "looping"),
        ("cap", "max_tokens"),
        ("ctx", "context_length"),
        ("early", "stopped_early"),
    ):
        if flag in flags:
            return tag
    if not facts.get("calls") or "empty" in flags:
        return "empty_output"
    if facts.get("judge_no_answer") or (facts.get("answer_empty") and task not in NO_ANSWER_FIELD):
        return "no_final_answer"
    return "wrong"


def call_flags(call: dict[str, Any], max_tokens: int | None) -> set[str]:
    """Flags raised by one model call; an answer gets a flag if any of its calls has it."""
    stop = call.get("stop")
    out = call.get("out")
    cap = stop == "max_tokens" or bool(max_tokens and out and out >= CAP_SHARE * max_tokens)
    flags = set()
    if call.get("block") or stop == "content_filter":
        flags.add("block")
    if call.get("z") is not None and call["z"] < LOOP_RATIO:
        flags.add("loop")
    if cap:
        flags.add("cap")
    if stop == "model_length":
        flags.add("ctx")
    if stop not in NORMAL_STOPS and not cap:
        flags.add("early")
    if call.get("tlen") == 0 and not call.get("tool") and stop == "stop":
        flags.add("empty")
    return flags


def score_value(scores: dict[str, Any], scorer: str) -> float | None:
    """The answer's value for the configured scorer, read the same way preprocess and the IRT fit read it."""
    raw = select_score(scores, scorer)
    return None if raw is None else normalize_score(raw, scorer)


def scorer_key(score_names: Iterable[str], scorer: str) -> str:
    """Name of the Score object the configured scorer reads from."""
    names = set(score_names)
    if scorer in names:
        return scorer
    alias = SCORE_ALIASES.get(scorer)
    if alias is not None and alias in names:
        return alias
    return scorer.split("/")[0]


# --------------------------------------------------------------------------- raw log reader


def _compression_ratio(text: str) -> float | None:
    tail = text[-LOOP_WINDOW:]
    if len(tail) < LOOP_MIN_CHARS:
        return None
    data = tail.encode()
    return round(len(zlib.compress(data)) / len(data), 3)


def _resolve(value: Any, attachments: dict[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("attachment://"):
        return attachments.get(value[len("attachment://"):], "")
    return value


def _message_text(content: Any, attachments: dict[str, str]) -> tuple[str, str]:
    """(reasoning, visible text) of an assistant message's content."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return "", _resolve(content, attachments) or ""
    reasoning, text = [], []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "reasoning":
            reasoning.append(_resolve(block.get("reasoning"), attachments) or "")
        elif block.get("type") == "text":
            text.append(_resolve(block.get("text"), attachments) or "")
    return "".join(reasoning), "".join(text)


def _call_from_output(output: dict[str, Any], attachments: dict[str, str]) -> dict[str, Any] | None:
    choices = output.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    message = choice.get("message") or {}
    reasoning, text = _message_text(message.get("content"), attachments)
    return {
        "stop": choice.get("stop_reason"),
        "out": (output.get("usage") or {}).get("output_tokens"),
        "tlen": len(text.strip()),
        "z": _compression_ratio(reasoning + text),
        "tool": bool(message.get("tool_calls")),
        "block": any(marker in text for marker in BLOCK_MARKERS),
    }


def sample_facts(sample: dict[str, Any], model: str, max_tokens: int | None, scorer: str) -> dict[str, Any]:
    """Facts about one raw sample: its calls to the evaluated model, errors, limits and scorer output.

    Only model events of the evaluated model count (judge and other helper
    models are ignored). Calls that errored and were retried are counted in
    ``api_errors`` and otherwise skipped. Logs without events fall back to the
    sample's final output.
    """
    attachments = sample.get("attachments") or {}
    calls, api_errors, from_output = [], 0, False
    for event in sample.get("events") or []:
        if event.get("event") != "model" or event.get("model") != model or event.get("pending"):
            continue
        if event.get("error"):
            api_errors += 1
            continue
        call = _call_from_output(event.get("output") or {}, attachments)
        if call is not None:
            calls.append(call)
    if not calls and not sample.get("events") and sample.get("output"):
        call = _call_from_output(sample["output"], attachments)
        if call is not None:
            calls.append(call)
            from_output = True

    flags: set[str] = set()
    for call in calls:
        flags |= call_flags(call, max_tokens)
    scores = sample.get("scores") or {}
    main = scores.get(scorer_key(scores, scorer)) or {}
    judge = _resolve((main.get("metadata") or {}).get("judge_response"), attachments)
    limit = sample.get("limit")
    outs = [c["out"] for c in calls if isinstance(c.get("out"), int)]
    zs = [c["z"] for c in calls if c.get("z") is not None]

    facts: dict[str, Any] = {
        "calls": len(calls),
        "stops": dict(collections.Counter(str(c.get("stop")) for c in calls)),
    }
    if max_tokens:
        facts["max_tokens"] = max_tokens
    if outs:
        facts["out_max"] = max(outs)
    if zs:
        facts["z_min"] = min(zs)
    if flags:
        facts["flags"] = sorted(flags)
    if api_errors:
        facts["api_errors"] = api_errors
    if from_output:
        facts["from_output"] = True
    if sample.get("error"):
        facts["error"] = True
    if isinstance(limit, dict) and limit.get("type"):
        facts["limit"] = str(limit["type"])
    if isinstance(judge, str) and "extracted_final_answer: None" in judge:
        facts["judge_no_answer"] = True
    if main and not main.get("answer"):
        facts["answer_empty"] = True
    return facts


def _header(archive: zipfile.ZipFile, names: set[str]) -> dict[str, Any]:
    for name in ("header.json", "_journal/start.json"):
        if name in names:
            return json.loads(archive.read(name))
    raise ValueError("no header.json or _journal/start.json")


def read_eval(path: Path, task: str, scorer: str) -> list[dict[str, Any]]:
    """Analyze every sample in one ``.eval`` log. Rows are in sample-file order."""
    rows = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        spec = _header(archive, names)["eval"]
        model = spec["model"]
        max_tokens = (spec.get("model_generate_config") or {}).get("max_tokens")
        for name in sorted(names):
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            sample = json.loads(archive.read(name))
            attachments = sample.get("attachments") or {}
            scores = {
                str(key): (score or {}).get("value")
                for key, score in (sample.get("scores") or {}).items()
            }
            value = score_value(scores, scorer)
            facts = sample_facts(sample, model, max_tokens, scorer)
            rows.append({
                "sample_id": logical_sample_id(
                    task, sample.get("id"), sample.get("metadata"),
                    _resolve(sample.get("input"), attachments),
                ),
                "raw_sample_id": str(sample.get("id")),
                "epoch": int(sample.get("epoch") or 1),
                "value": value,
                "tag": tag_answer(value, facts, task),
                "facts": facts,
            })
    return rows


def _analyze_file(job: tuple[str, str, str, str, str]) -> tuple[str, list[dict[str, Any]] | None, str | None, float]:
    root, rel_path, model, task, scorer = job
    started = time.time()
    try:
        rows = read_eval(Path(root) / rel_path, task, scorer)
    except Exception as error:  # noqa: BLE001 - reported per file, the run continues
        return rel_path, None, f"{type(error).__name__}: {error}"[:300], time.time() - started
    for row in rows:
        row.update(file_path=rel_path, model=model, task=task, rules_version=RULES_VERSION)
    return rel_path, rows, None, time.time() - started


# --------------------------------------------------------------------------- join check

_MASK = (1 << 64) - 1


def _answer_hash(sample_id: str, epoch: int, value: float | None) -> int:
    key = f"{sample_id}\x00{epoch}\x00{'' if value is None else round(value, 6)}"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def answer_digest(items: Iterable[tuple[str, int, float | None]]) -> tuple[int, int]:
    """Order-independent (count, digest) of a multiset of (sample_id, epoch, value)."""
    count, total = 0, 0
    for item in items:
        total = (total + _answer_hash(*item)) & _MASK
        count += 1
    return count, total


def samples_digests(
    samples_path: Path, scorers: dict[str, str], keys_for: set[str] = frozenset()
) -> tuple[dict[str, tuple[int, int]], dict[str, collections.Counter[tuple[str, int]]]]:
    """Per file in samples.jsonl: the digest of its answers, and for files in ``keys_for`` its (sample_id, epoch) keys.

    The digests check the analysis matches samples.jsonl 1:1; the keys say which
    answers of a log preprocess kept (it drops questions outside a task's
    reference question set).
    """
    digests: dict[str, list[int]] = {}
    keys: dict[str, collections.Counter[tuple[str, int]]] = {}
    with open(samples_path) as handle:
        for line in handle:
            row = json.loads(line)
            task, path = row["task"], row["file_path"]
            sample_id, epoch = str(row["sample_id"]), int(row["epoch"])
            value = score_value(row["scores"], scorers[task]) if task in scorers else None
            entry = digests.setdefault(path, [0, 0])
            entry[0] += 1
            entry[1] = (entry[1] + _answer_hash(sample_id, epoch, value)) & _MASK
            if path in keys_for:
                keys.setdefault(path, collections.Counter())[(sample_id, epoch)] += 1
    return {path: (count, total) for path, (count, total) in digests.items()}, keys


def keep_kept_answers(
    rows: list[dict[str, Any]], kept: collections.Counter[tuple[str, int]]
) -> tuple[list[dict[str, Any]], int]:
    """Rows whose (sample_id, epoch) preprocess kept in samples.jsonl, and how many were dropped."""
    left = collections.Counter(kept)
    out = []
    for row in rows:
        key = (row["sample_id"], row["epoch"])
        if left[key] > 0:
            left[key] -= 1
            out.append(row)
    return out, len(rows) - len(out)


# --------------------------------------------------------------------------- driver


def default_inputs() -> list[Path]:
    paths = [DEFAULT_DATA / "samples.jsonl", *sorted((DEFAULT_DATA / "tranches").glob("*/samples.jsonl"))]
    return [path for path in paths if path.is_file()]


def manifest_path(samples_path: Path) -> Path:
    return samples_path.with_name(samples_path.name.removesuffix(".jsonl") + ".manifest.json")


def analysis_path(samples_path: Path) -> Path:
    return samples_path.with_name(OUTPUT_NAME)


def _load_previous(output: Path, sidecar: Path) -> dict[str, dict[str, Any]]:
    if not output.is_file() or not sidecar.is_file():
        return {}
    try:
        previous = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return {}
    if previous.get("rules_version") != RULES_VERSION or previous.get("output_size") != output.stat().st_size:
        return {}
    return {row["path"]: row for row in previous.get("files", []) if row.get("status") == "ok"}


def analyze_tree(
    samples_path: Path,
    *,
    root: Path,
    workers: int,
    models: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write sample_analysis.jsonl (+ manifest) next to one samples.jsonl. Returns the summary."""
    started = time.time()
    manifest = json.loads(manifest_path(samples_path).read_text())
    scorers = manifest["scorers"]
    files = [row for row in manifest["files"] if row.get("parse_status") == "ok"]
    output = analysis_path(samples_path)
    sidecar = samples_path.with_name(OUTPUT_MANIFEST_NAME)
    previous = {} if force else _load_previous(output, sidecar)

    def same_log(row: dict[str, Any]) -> bool:
        old = previous.get(row["path"])
        return bool(old) and old["size"] == row["size"] and old["mtime_ns"] == row["mtime_ns"]

    print(f"{samples_path}: reading samples.jsonl...", flush=True)
    expected, kept_keys = samples_digests(
        samples_path, scorers, {row["path"] for row in files if not same_log(row)}
    )

    def cached(row: dict[str, Any]) -> bool:
        # Unchanged log, and preprocess kept the same answers of it as last time.
        return same_log(row) and tuple(previous[row["path"]]["digest"]) == expected.get(row["path"])

    def selected(row: dict[str, Any]) -> bool:
        return not models or any(pattern in str(row["model"]) for pattern in models)

    jobs = [
        (str(root), row["path"], str(row["model"]), str(row["task"]), scorers[str(row["task"])])
        for row in files
        if selected(row) and not cached(row)
    ]
    if any(job[1] not in kept_keys and job[1] in expected for job in jobs):
        # A cached log whose kept answers changed: collect its keys too.
        _, kept_keys = samples_digests(samples_path, scorers, {job[1] for job in jobs})
    print(f"{samples_path}: {len(files)} ok logs, {len(jobs)} to read, "
          f"{sum(1 for row in files if cached(row))} unchanged since the last run", flush=True)

    tmp = output.with_name(output.name + ".tmp")
    file_rows: list[dict[str, Any]] = []
    tags: collections.Counter[str] = collections.Counter()
    errors: list[tuple[str, str]] = []
    pending = {job[1] for job in jobs}
    executor = ProcessPoolExecutor(workers) if workers > 1 and jobs else None
    try:
        fresh = iter(executor.map(_analyze_file, jobs) if executor else map(_analyze_file, jobs))
        with open(tmp, "wb") as out, open(output if previous else os.devnull, "rb") as old:
            done = 0
            for row in files:
                path = row["path"]
                entry = {"path": path, "size": row["size"], "mtime_ns": row["mtime_ns"],
                         "model": row["model"], "task": row["task"]}
                if path in pending:
                    _, rows, error, seconds = next(fresh)
                    done += 1
                    if done % 25 == 0 or done == len(jobs):
                        print(f"  read {done}/{len(jobs)} logs ({time.time() - started:.0f}s)", flush=True)
                    if rows is None:
                        errors.append((path, error or ""))
                        file_rows.append({**entry, "status": "error", "error": error})
                        continue
                    rows, dropped = keep_kept_answers(rows, kept_keys.get(path, collections.Counter()))
                    data = b"".join(json.dumps(r, separators=(",", ":")).encode() + b"\n" for r in rows)
                    digest = answer_digest((r["sample_id"], r["epoch"], r["value"]) for r in rows)
                    file_tags = collections.Counter(r["tag"] for r in rows)
                    seconds = round(seconds, 2)
                elif cached(row):
                    old_row = previous[path]
                    old.seek(old_row["offset"])
                    data = old.read(old_row["length"])
                    digest = (old_row["digest"][0], old_row["digest"][1])
                    file_tags = collections.Counter(old_row["tags"])
                    seconds = old_row.get("seconds")
                    dropped = old_row.get("dropped", 0)
                else:
                    file_rows.append({**entry, "status": "skipped"})
                    continue
                offset = out.tell()
                out.write(data)
                tags.update(file_tags)
                file_rows.append({**entry, "status": "ok", "offset": offset, "length": len(data),
                                  "rows": digest[0], "digest": list(digest), "tags": dict(file_tags),
                                  "seconds": seconds, **({"dropped": dropped} if dropped else {})})
    finally:
        if executor:
            executor.shutdown()
    tmp.replace(output)

    mismatched = [
        {"path": row["path"], "analysis_rows": row["rows"], "samples_rows": expected.get(row["path"], (0, 0))[0]}
        for row in file_rows
        if row["status"] == "ok" and tuple(row["digest"]) != expected.get(row["path"])
    ]
    summary = {
        "rules_version": RULES_VERSION,
        "samples": str(samples_path),
        "output_size": output.stat().st_size,
        "rows": sum(row.get("rows", 0) for row in file_rows),
        "files_ok": sum(row["status"] == "ok" for row in file_rows),
        "files_error": len(errors),
        "files_skipped": sum(row["status"] == "skipped" for row in file_rows),
        "dropped_outside_reference": sum(row.get("dropped", 0) for row in file_rows),
        "mismatched_files": mismatched,
        "tags": {tag: tags[tag] for tag in TAGS if tags[tag]},
        "seconds": round(time.time() - started, 1),
        "files": file_rows,
    }
    sidecar.write_text(json.dumps(summary, indent=1) + "\n")
    _print_summary(summary, output, errors)
    return summary


def _print_summary(summary: dict[str, Any], output: Path, errors: list[tuple[str, str]]) -> None:
    rows = summary["rows"] or 1
    print(f"  wrote {output} ({summary['rows']:,} answers, {summary['output_size'] / 1e6:.0f} MB) "
          f"in {summary['seconds']}s")
    for tag, count in summary["tags"].items():
        print(f"    {tag:24s}{count:9,d}  {100 * count / rows:5.1f}%")
    if summary["dropped_outside_reference"]:
        print(f"  left out {summary['dropped_outside_reference']:,} answers that preprocess also left out "
              "(questions outside the task's reference question set)")
    if summary["files_skipped"]:
        print(f"  {summary['files_skipped']} logs skipped by --model and not in a previous run (not in the output)")
    for path, error in errors:
        print(f"  ERROR {path}: {error}", file=sys.stderr)
    for row in summary["mismatched_files"]:
        print(f"  MISMATCH {row['path']}: {row['analysis_rows']} analysis rows vs "
              f"{row['samples_rows']} samples.jsonl rows (or different scores)", file=sys.stderr)
    if not errors and not summary["mismatched_files"]:
        print("  every analyzed log matches samples.jsonl")


def analyze(
    input_paths: Annotated[Optional[list[Path]], typer.Argument(help="Preprocessed samples.jsonl files (default: tools/minify/data/samples.jsonl and every tools/minify/data/tranches/*/samples.jsonl)")] = None,
    root: Annotated[Path, typer.Option(help="Directory the log paths in the manifests are relative to (the repository root)")] = REPO_ROOT,
    workers: Annotated[int, typer.Option(help="Parallel worker processes")] = 4,
    model: Annotated[Optional[list[str]], typer.Option("--model", help="Only (re)read logs whose model contains this text; may repeat. Other logs are kept from the previous run if unchanged")] = None,
    force: Annotated[bool, typer.Option(help="Re-read every log even if unchanged since the last run")] = False,
) -> None:
    """Tag every answer in the preprocessed logs with how it went (passed, wrong, cut off, looping, blocked, ...).

    Writes sample_analysis.jsonl next to each samples.jsonl. Run after preprocess and before export-web.
    """
    paths = input_paths or default_inputs()
    if not paths:
        raise typer.BadParameter("no samples.jsonl found; run preprocess first")
    failed = False
    for path in paths:
        summary = analyze_tree(path, root=root, workers=workers, models=model, force=force)
        failed |= bool(summary["files_error"] or summary["mismatched_files"])
    if failed:
        raise typer.Exit(1)


app = typer.Typer(help="Tag every answer in the preprocessed logs with how it went.")
app.command()(analyze)

if __name__ == "__main__":
    app()
