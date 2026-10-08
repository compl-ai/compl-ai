import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from zipfile import BadZipFile
from zipfile import ZipFile

import ijson  # type: ignore[import-untyped]
from inspect_ai.log import read_eval_log
from inspect_ai.log import read_eval_log_sample
from inspect_ai.log import read_eval_log_sample_summaries
from tqdm import tqdm

from complai.utils.scores import metric_source
from complai.utils.scores import select_score


RECORDS_SCHEMA = "complai.utils.log_parser-v1"
SUPPORTED_RECORDS_SCHEMAS = {RECORDS_SCHEMA}
PARSER_VERSION = "complai-core-inspect-v1"
TRUTHFULQA_PARSER_VERSION = "complai-core-inspect-v1-truthfulqa"
GPQA_PARSER_VERSION = "complai-core-inspect-v1-gpqa"
HLE_PARSER_VERSION = "complai-core-inspect-v1-hle"
HIJACKING_PARSER_VERSION = "complai-core-inspect-v1-hijacking"
MMLU_PARSER_VERSION = "complai-core-inspect-v1-mmlu"
STRONG_REJECT_PARSER_VERSION = "complai-core-inspect-v1-strong-reject-basis"
LOG_SUFFIXES = (".eval", ".eval.gz")
# strong_reject repeats its 60 base prompts once per jailbreak method, so log
# sample k is a variant of base prompt (k - 1) % 60 + 1. The base prompt is the
# basis sample: its variants share one ID and hash and are averaged like epochs.
STRONG_REJECT_BASE_PROMPTS = 60
VARIANT_TASKS = {"strong_reject"}
# Eval-level metadata key set by ``complai eval --subset``: such runs cover only
# subset items, so they are partial even without --limit or --sample-id.
SUBSET_METADATA_KEY = "complai_subset_id"

QUESTION_METADATA_KEYS = {
    "sensitive_attribute",
    "original_question",
    "topic",
    "category",
    "type",
    "proposition",
    "config",
    "prompt",
    "question",
    "context",
    "scenario",
    "behavior",
    "test_case",
    "example",
    "instruction",
    "text",
    "input",
    "source",
}
ANSWER_METADATA_KEYS = {
    "label",
    "original_answer",
    "ground_truth",
    "formatted_ground_truth",
}


@dataclass(frozen=True)
class PreprocessedRecords:
    """A JSONL response source and its sidecar manifest."""

    records_path: Path
    manifest_path: Path
    files: tuple[dict[str, Any], ...]
    digest: str
    scorers: dict[str, str]
    records: int

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        """Yield normalized records without rechecking source logs."""
        count = 0
        with self.records_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {self.records_path}"
                    ) from exc
                if not isinstance(row, dict):
                    raise TypeError(
                        f"Line {line_number} of {self.records_path} is not an object"
                    )
                count += 1
                yield row
        if count != self.records:
            raise ValueError(
                f"Record count mismatch for {self.records_path}: "
                f"manifest says {self.records}, found {count}"
            )

    @property
    def inventory(self) -> dict[str, Any]:
        """Return provenance stored with the preprocessed records."""
        statuses = Counter(str(row.get("parse_status", "")) for row in self.files)
        return {
            "schema_version": RECORDS_SCHEMA,
            "source": "preprocessed_jsonl",
            "digest": self.digest,
            "summary": {
                "discovered": len(self.files),
                "eligible": statuses["ok"],
                "excluded": statuses["excluded"],
                "deferred": statuses["deferred"],
                "superseded": statuses["superseded"],
                "failed": statuses["error"],
            },
            "files": list(self.files),
        }


def preprocess_logs(
    log_paths: list[Path],
    scorers: dict[str, str],
    output_path: Path,
    *,
    workers: int = 1,
) -> PreprocessedRecords:
    """Write compact response JSONL, a provenance manifest and ``metrics.json``.

    Each task's question set is defined by its newest full eval (the
    "reference"); records for questions outside it are dropped. When a model has
    several evals of a task, the one answering the most reference questions is
    kept and the others are marked ``superseded``. Every kept eval records its
    ``coverage`` of the reference. Benchmark metrics come from the kept eval.
    """
    if not scorers or any(not task or not scorer for task, scorer in scorers.items()):
        raise ValueError("scorers must be a non-empty task-to-scorer mapping")
    roots, paths = _discover_logs(log_paths)
    output_path = output_path.expanduser().resolve()
    if output_path.suffix != ".jsonl":
        raise ValueError("Preprocessed records output must end in .jsonl")
    manifest_path = records_manifest_path(output_path)
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Output already exists: {existing[0]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=output_path.parent) as scratch:
        scratch_dir = Path(scratch)
        parsed = _parse_logs(paths, scorers, scratch_dir, workers)
        file_rows = [file_row for file_row, _, _ in parsed]
        chosen, references = _select_evals(parsed)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", dir=output_path.parent
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        record_count = 0
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for index in chosen:
                    task = file_rows[index]["task"]
                    for line in _scratch_path(scratch_dir, index).open(encoding="utf-8"):
                        record = json.loads(line)
                        if _hash_key(record["content_hash"]) not in references[task]:
                            continue
                        handle.write(line)
                        digest.update(line.encode())
                        record_count += 1
                handle.flush()
                os.fsync(handle.fileno())

            manifest = {
                "schema_version": RECORDS_SCHEMA,
                "records": record_count,
                "records_sha256": digest.hexdigest(),
                "scorers": dict(sorted(scorers.items())),
                "inspect_version": importlib.metadata.version("inspect-ai"),
                "roots": [str(path) for path in roots],
                "files": file_rows,
            }
            manifest["input_digest"] = digest_json(manifest)
            _write_manifest(manifest_path, manifest)
            os.replace(temporary_path, output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    _write_benchmark_metrics(output_path.parent / "metrics.json", file_rows, chosen, scorers)
    return load_records(output_path)


def _parse_logs(
    paths: tuple[Path, ...], scorers: dict[str, str], scratch_dir: Path, workers: int
) -> list[tuple[dict[str, Any], frozenset[bytes], frozenset[bytes]]]:
    """Parse every log (in parallel when ``workers > 1``), in path order."""
    job = partial(_preprocess_to_scratch, scorers=scorers, scratch_dir=scratch_dir)
    jobs = list(enumerate(paths))
    progress = partial(tqdm, total=len(paths), desc="Preprocessing logs", unit="log")
    if workers <= 1:
        return list(progress(job(item) for item in jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(progress(pool.map(job, jobs)))


def _preprocess_to_scratch(
    item: tuple[int, Path], *, scorers: dict[str, str], scratch_dir: Path
) -> tuple[dict[str, Any], frozenset[bytes], frozenset[bytes]]:
    """Parse one log into a scratch JSONL file.

    Returns its file row plus the content-hash keys of answered and unanswered
    questions, where answered means the configured score is present.
    """
    index, path = item
    file_row, records = _preprocess_file(path, scorers)
    scorer = scorers.get(file_row["task"], "")
    answered: set[bytes] = set()
    unanswered: set[bytes] = set()
    with _scratch_path(scratch_dir, index).open("w", encoding="utf-8") as handle:
        for record in records:
            key = _hash_key(record["content_hash"])
            (answered if select_score(record["scores"], scorer) is not None else unanswered).add(key)
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    return file_row, frozenset(answered), frozenset(unanswered - answered)


def _select_evals(
    parsed: list[tuple[dict[str, Any], frozenset[bytes], frozenset[bytes]]],
) -> tuple[list[int], dict[str, frozenset[bytes]]]:
    """Pick one eval per (model, task) and each task's reference question set.

    Updates the file rows in place with ``coverage`` and, for evals that lose to
    a more complete one, ``parse_status="superseded"`` plus ``superseded_by``.
    Returns the chosen row indices in path order and the per-task references.
    """
    by_task: dict[str, list[int]] = {}
    for index, (file_row, _, _) in enumerate(parsed):
        if file_row["parse_status"] == "ok":
            by_task.setdefault(file_row["task"], []).append(index)

    def recency(index: int) -> tuple[str, str]:
        return parsed[index][0]["created"], parsed[index][0]["path"]

    references: dict[str, frozenset[bytes]] = {}
    chosen: list[int] = []
    for task, indices in by_task.items():
        full = [index for index in indices if not parsed[index][0]["partial"]]
        if full:
            newest = max(full, key=recency)
            references[task] = parsed[newest][1] | parsed[newest][2]
        else:
            references[task] = frozenset().union(*(parsed[index][1] for index in indices))
        reference = references[task]
        by_model: dict[str, list[int]] = {}
        for index in indices:
            file_row, answered, _ = parsed[index]
            file_row["coverage"] = len(answered & reference) / len(reference) if reference else 0.0
            by_model.setdefault(file_row["model"], []).append(index)
        for candidates in by_model.values():
            best = max(
                candidates, key=lambda index: (parsed[index][0]["coverage"], *recency(index))
            )
            chosen.append(best)
            for index in candidates:
                if index != best:
                    parsed[index][0]["parse_status"] = "superseded"
                    parsed[index][0]["superseded_by"] = parsed[best][0]["path"]
    return sorted(chosen), references


def _write_benchmark_metrics(
    path: Path, file_rows: list[dict[str, Any]], chosen: list[int], scorers: dict[str, str]
) -> None:
    """Write task -> model -> benchmark metrics of each kept eval's configured scorer."""
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[str] = []
    for index in chosen:
        row = file_rows[index]
        logged = row.get("metrics", {})
        source = metric_source(logged, scorers[row["task"]])
        if source is None:
            missing.append(f"{row['task']}/{row['model']}")
        metrics.setdefault(row["task"], {})[row["model"]] = logged[source] if source else {}
    if missing:
        print(f"WARNING: no benchmark metrics for the configured scorer in {len(missing)} eval(s): "
              + ", ".join(sorted(missing)[:10]) + (" ..." if len(missing) > 10 else ""))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def _scratch_path(scratch_dir: Path, index: int) -> Path:
    return scratch_dir / f"{index:06d}.jsonl"


def _hash_key(content_hash: str) -> bytes:
    """Compact set key for a hex content hash (first 128 bits)."""
    return bytes.fromhex(content_hash[:32])


def load_records(records_path: Path) -> PreprocessedRecords:
    """Load response records without inspecting their source logs."""
    records_path = records_path.expanduser().resolve()
    manifest_path = records_manifest_path(records_path)
    if not records_path.is_file():
        raise FileNotFoundError(f"Preprocessed records do not exist: {records_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Preprocessed records manifest does not exist: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid records manifest JSON: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") not in SUPPORTED_RECORDS_SCHEMAS
    ):
        raise ValueError(f"Unsupported records manifest: {manifest_path}")
    scorers = manifest.get("scorers")
    files = manifest.get("files")
    if not isinstance(scorers, dict) or not isinstance(files, list):
        raise TypeError(f"Malformed records manifest: {manifest_path}")

    return PreprocessedRecords(
        records_path=records_path,
        manifest_path=manifest_path,
        files=tuple(dict(row) for row in files),
        digest=str(manifest.get("input_digest", "")),
        scorers={str(task): str(scorer) for task, scorer in scorers.items()},
        records=int(manifest.get("records", 0)),
    )


def records_manifest_path(records_path: Path) -> Path:
    """Return the sidecar manifest path for a records JSONL file."""
    return records_path.with_suffix(".manifest.json")


def _preprocess_file(
    path: Path, scorers: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Preprocess one Inspect log."""
    try:
        metadata = _read_log_metadata(path)
        task = str(metadata["task"])
        if task not in scorers:
            parse_status = "deferred"
            samples: list[dict[str, Any]] = []
        elif not metadata["eligible"]:
            parse_status = "excluded"
            samples = []
        else:
            metadata, samples = _parse_log(path, metadata)
            parse_status = "ok"

        records = []
        for sample in samples:
            records.append(
                {
                    "file_path": str(path),
                    "run_id": metadata["run_id"],
                    "created": metadata["created"],
                    "model": metadata["model"],
                    "task": task,
                    "dataset": metadata["dataset"],
                    "sample_id": sample["sample_id"],
                    "epoch": sample["epoch"],
                    "scores": sample["scores"],
                    "content_hash": sample["content_hash"],
                    "question_hash": sample["question_hash"],
                }
            )
        stat = path.stat()
        file_row = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "parser_version": _parser_version(task),
            "inspect_version": importlib.metadata.version("inspect-ai"),
            "parse_status": parse_status,
            "run_id": metadata["run_id"],
            "status": metadata["status"],
            "model": metadata["model"],
            "task": task,
            "dataset": metadata["dataset"],
            "created": metadata["created"],
            "sample_count": metadata["sample_count"],
            "partial": metadata["partial"],
            "metrics": metadata.get("metrics", {}),
            "model_config": metadata.get("model_config", {}),
            "task_config": metadata.get("task_config", {}),
        }
        return file_row, records
    except Exception as exc:
        raise ValueError(f"Failed to preprocess {path}: {exc}") from exc


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest atomically."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _discover_logs(paths: list[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Resolve input roots and discover Inspect logs."""
    roots = tuple(sorted({path.expanduser().resolve() for path in paths}))
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Inspect log path does not exist: {root}")
        if root.is_file():
            if _is_log(root):
                found.add(root)
            else:
                raise ValueError(f"Unsupported Inspect log file: {root}")
        else:
            found.update(
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and _is_log(path)
            )
    if not found:
        raise ValueError("No .eval or .eval.gz files were found")
    return roots, tuple(sorted(found))



def _extract_metrics(results: Any) -> dict[str, dict[str, float]]:
    metrics = {}
    if results and getattr(results, "scores", None):
        for s in results.scores:
            if hasattr(s, "metrics") and s.metrics:
                m_dict = {}
                for k, v in s.metrics.items():
                    if hasattr(v, "value"):
                        try:
                            val = float(v.value)
                            m_dict[k] = val if math.isfinite(val) else None
                        except (ValueError, TypeError):
                            m_dict[k] = None
                metrics[s.name] = m_dict
    return metrics

def _read_log_metadata(path: Path) -> dict[str, Any]:
    """Read lightweight run metadata from an Inspect log header."""
    log = read_eval_log(str(path), header_only=True)
    spec = log.eval
    results = log.results
    total = getattr(results, "total_samples", None) if results is not None else None
    completed = (
        getattr(results, "completed_samples", None) if results is not None else None
    )
    status = str(log.status)
    dataset_spec = getattr(spec, "dataset", None)
    
    model_config = {}
    task_config = {}
    try:
        with ZipFile(path, 'r') as z:
            if 'header.json' in z.namelist():
                header = json.loads(z.read('header.json'))
                eval_data = header.get('eval', {})
                plan_data = header.get('plan', {})
                
                # Model level
                if 'model_generate_config' in eval_data:
                    model_config.update(eval_data['model_generate_config'])
                    
                # Task level
                if 'task_args' in eval_data:
                    task_config.update(eval_data['task_args'])
                if 'steps' in plan_data and len(plan_data['steps']) > 0:
                    params = plan_data['steps'][0].get('params_passed', {})
                    if isinstance(params, dict):
                        task_config.update(params)
    except Exception:
        pass

    return {
        "run_id": str(getattr(spec, "eval_id", "") or getattr(spec, "run_id", "")),
        "created": str(getattr(spec, "created", "")),
        "status": status,
        "model": str(getattr(spec, "model", "")),
        "task": str(getattr(spec, "task", "")),
        "dataset": str(
            getattr(dataset_spec, "name", None)
            or getattr(dataset_spec, "location", None)
            or "unknown_dataset"
        ),
        "sample_count": int(completed or 0),
        # Runs restricted with --limit, --sample-id or --subset cannot define a task's question set.
        "partial": bool(
            getattr(spec.config, "limit", None)
            or getattr(spec.config, "sample_id", None)
            or SUBSET_METADATA_KEY in (getattr(spec, "metadata", None) or {})
        ),
        "eligible": (
            status.lower() == "success" and total is not None and total == completed
        ),
        "metrics": _extract_metrics(results),
        "model_config": model_config,
        "task_config": task_config,
    }


def _parse_log(
    path: Path, metadata: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse normalized sample records without loading full traces."""
    metadata = metadata or _read_log_metadata(path)
    if not metadata["eligible"]:
        return metadata, []
    if metadata["task"] == "strong_reject" and (metadata.get("task_config") or {}).get("full"):
        raise ValueError(f"strong_reject with full=True is not supported: {path}")
    summaries = read_eval_log_sample_summaries(str(path))
    missing_choices = _read_missing_choices(path, summaries)
    samples = []
    for sample in summaries:
        epoch = int(sample.epoch or 1)
        choices = (
            sample.choices
            if sample.choices is not None
            else missing_choices.get((str(sample.id), epoch))
        )
        samples.append(
            {
                "sample_id": logical_sample_id(
                    str(metadata["task"]), sample.id, sample.metadata, sample.input
                ),
                "epoch": epoch,
                "scores": {
                    str(name): json_value(getattr(score, "value", None))
                    for name, score in (sample.scores or {}).items()
                },
                "content_hash": content_hash(
                    sample.input,
                    sample.target,
                    sample.metadata,
                    choices=choices,
                    task=str(metadata["task"]),
                    sample_id=sample.id,
                ),
                "question_hash": question_hash(
                    sample.input,
                    sample.metadata,
                    choices=choices,
                    task=str(metadata["task"]),
                    sample_id=sample.id,
                ),
            }
        )
    return metadata, samples


def _read_missing_choices(
    path: Path, samples: list[Any]
) -> dict[tuple[str, int], list[str]]:
    """Read only early choice fields omitted by historical summaries."""
    candidates = [
        sample
        for sample in samples
        if sample.choices is None and _looks_like_choice_target(sample.target)
    ]
    if not candidates or not path.is_file():
        return {}
    try:
        with ZipFile(path) as archive:
            choices = {}
            for sample in candidates:
                epoch = int(sample.epoch or 1)
                name = f"samples/{sample.id}_epoch_{epoch}.json"
                try:
                    with archive.open(name) as member:
                        for key, value in ijson.kvitems(member, ""):
                            if key == "choices":
                                if isinstance(value, list):
                                    choices[(str(sample.id), epoch)] = [
                                        str(choice) for choice in value
                                    ]
                                break
                except KeyError:
                    return _read_missing_choices_fully(path, candidates)
            return choices
    except BadZipFile:
        return _read_missing_choices_fully(path, candidates)


def _read_missing_choices_fully(
    path: Path, samples: list[Any]
) -> dict[tuple[str, int], list[str]]:
    """Fall back to Inspect for non-ZIP log formats."""
    choices = {}
    excluded = {
        "attachments",
        "events",
        "events_data",
        "messages",
        "model_usage",
        "output",
        "role_usage",
        "scores",
        "store",
        "timelines",
    }
    for sample in samples:
        epoch = int(sample.epoch or 1)
        full_sample = read_eval_log_sample(
            str(path), id=sample.id, epoch=epoch, exclude_fields=excluded
        )
        if full_sample.choices is not None:
            choices[(str(sample.id), epoch)] = list(full_sample.choices)
    return choices


def _looks_like_choice_target(target: Any) -> bool:
    """Return whether a target uses Inspect's letter-choice notation."""
    values = target if isinstance(target, list) else [target]
    return bool(values) and all(
        isinstance(value, str) and bool(re.fullmatch(r"[A-Z](?:[, ]*[A-Z])*", value))
        for value in values
    )


def _parser_version(task: str) -> str:
    """Return the parser recipe version for a task."""
    return {
        "truthfulqa": TRUTHFULQA_PARSER_VERSION,
        "gpqa_diamond": GPQA_PARSER_VERSION,
        "hle": HLE_PARSER_VERSION,
        "instruction_goal_hijacking": HIJACKING_PARSER_VERSION,
        "mmlu_pro": MMLU_PARSER_VERSION,
        "strong_reject": STRONG_REJECT_PARSER_VERSION,
    }.get(task, PARSER_VERSION)


def _is_log(path: Path) -> bool:
    """Return whether a path has a supported Inspect log suffix."""
    return any(path.name.lower().endswith(suffix) for suffix in LOG_SUFFIXES)


def question_hash(
    input_value: Any,
    metadata: Any,
    *,
    choices: list[str] | None = None,
    task: str = "",
    sample_id: Any = None,
) -> str:
    """Create a stable digest for a logical question."""
    values = metadata if isinstance(metadata, dict) else {}
    if task in VARIANT_TASKS:
        payload = {
            "basis": logical_sample_id(task, sample_id, metadata),
            "category": canonical_item_value(values.get("category")),
            "source": canonical_item_value(values.get("source")),
        }
    elif task == "instruction_goal_hijacking":
        payload = {
            "access_code": canonical_item_value(values.get("access_code")),
            "attack": canonical_item_value(values.get("attack")),
        }
    elif task == "fairllm":
        payload = {
            "director": canonical_item_value(values.get("director")),
        }
    else:
        stable_metadata = {
            key: values[key] for key in sorted(QUESTION_METADATA_KEYS) if key in values
        }
        payload = {
            "input": _canonical_task_value(input_value, task),
            "choices": _canonical_choices(choices, task),
            "metadata": canonical_item_value(stable_metadata),
        }
    return digest_json({"recipe": "complai-question-v2", **payload})


def content_hash(
    input_value: Any,
    target: Any,
    metadata: Any,
    *,
    choices: list[str] | None = None,
    task: str = "",
    sample_id: Any = None,
) -> str:
    """Create a stable digest for a question and its scoring contract."""
    values = metadata if isinstance(metadata, dict) else {}
    answer_metadata = (
        {}
        if choices
        else {key: values[key] for key in sorted(ANSWER_METADATA_KEYS) if key in values}
    )
    payload = {
        "recipe": "complai-scoring-v2",
        "question_hash": question_hash(
            input_value, metadata, choices=choices, task=task, sample_id=sample_id
        ),
        "target": _canonical_target(target, choices, task),
        "answer_metadata": canonical_item_value(answer_metadata),
    }
    return digest_json(payload)


def _canonical_choices(choices: list[str] | None, task: str) -> list[Any] | None:
    """Normalize choices without preserving their presentation order."""
    if choices is None:
        return None
    values = [_canonical_task_value(choice, task) for choice in choices]
    return sorted(values, key=lambda value: json.dumps(value, sort_keys=True))


def _canonical_target(target: Any, choices: list[str] | None, task: str) -> Any:
    """Resolve multiple-choice labels to their semantic answer values."""
    if not choices:
        return _canonical_task_value(target, task)
    targets = target if isinstance(target, list) else [target]
    indexes: list[int] = []
    for value in targets:
        if not isinstance(value, str):
            return _canonical_task_value(target, task)
        labels = [character for character in value if character not in {",", " "}]
        for label in labels:
            if label.isalpha():
                index = ord(label.upper()) - ord("A")
            elif label.isnumeric():
                index = 25 + int(label)
            else:
                return _canonical_task_value(target, task)
            if not 0 <= index < len(choices):
                return _canonical_task_value(target, task)
            indexes.append(index)
    answers = [_canonical_task_value(choices[index], task) for index in indexes]
    return sorted(answers, key=lambda value: json.dumps(value, sort_keys=True))


def _canonical_task_value(value: Any, task: str) -> Any:
    """Apply any task-specific content normalization."""
    return (
        canonical_mmlu_value(value)
        if task == "mmlu_pro"
        else canonical_item_value(value)
    )


def logical_sample_id(
    task: str, sample_id: Any, metadata: Any, input_value: Any = None
) -> str:
    """Return the stable logical sample identifier for a task."""
    if task == "truthfulqa" and isinstance(input_value, str):
        digest = hashlib.md5(input_value.encode()).hexdigest()[:8]
        return f"truthfulqa_{digest}"
    if task == "hle" and isinstance(metadata, dict) and metadata.get("uid"):
        return str(metadata["uid"])
    if task in VARIANT_TASKS:
        if sample_id is None:
            raise ValueError(f"{task} samples need an ID to find their base prompt")
        return str((int(sample_id) - 1) % STRONG_REJECT_BASE_PROMPTS + 1)
    return str(sample_id)


def json_value(value: Any) -> Any:
    """Convert an arbitrary value to JSON-compatible data."""
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_item_value(value: Any) -> Any:
    """Normalize item content for deterministic hashing."""
    encoded = json_value(value)
    if isinstance(encoded, list):
        return [canonical_item_value(item) for item in encoded]
    if isinstance(encoded, dict):
        is_message = "role" in encoded and "content" in encoded
        return {
            key: canonical_item_value(item)
            for key, item in sorted(encoded.items())
            if not (is_message and key == "id")
        }
    if isinstance(encoded, str):
        return " ".join(encoded.split())
    return encoded


def canonical_mmlu_value(value: Any) -> Any:
    """Normalize legacy MMLU text variants for stable hashing."""
    encoded = json_value(value)
    if isinstance(encoded, list):
        return [canonical_mmlu_value(item) for item in encoded]
    if isinstance(encoded, dict):
        return {
            key: canonical_mmlu_value(item) for key, item in sorted(encoded.items())
        }
    if not isinstance(encoded, str):
        return encoded
    text = unicodedata.normalize("NFKC", encoded.strip()).lower()
    text = text.replace("\x0bert_t", "at_t").replace("\\vert_t", "at_t")
    text = text.replace("\x0bert", "").replace("\\vert", "").replace("√", "surd")
    return re.sub(r"[^a-z0-9]+", "", text)


def digest_json(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON value."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


