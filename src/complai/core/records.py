import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile
from zipfile import ZipFile

import ijson  # type: ignore[import-untyped]
from inspect_ai.log import read_eval_log
from inspect_ai.log import read_eval_log_sample
from inspect_ai.log import read_eval_log_sample_summaries
from tqdm import tqdm


RECORDS_SCHEMA = "complai-core-records-v1"
SUPPORTED_RECORDS_SCHEMAS = {RECORDS_SCHEMA}
PARSER_VERSION = "complai-core-inspect-v1"
TRUTHFULQA_PARSER_VERSION = "complai-core-inspect-v1-truthfulqa"
GPQA_PARSER_VERSION = "complai-core-inspect-v1-gpqa"
HLE_PARSER_VERSION = "complai-core-inspect-v1-hle"
HIJACKING_PARSER_VERSION = "complai-core-inspect-v1-hijacking"
MMLU_PARSER_VERSION = "complai-core-inspect-v1-mmlu"
LOG_SUFFIXES = (".eval", ".eval.gz")

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
                "failed": statuses["error"],
            },
            "files": list(self.files),
        }


def preprocess_logs(
    log_paths: list[Path],
    scorers: dict[str, str],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> PreprocessedRecords:
    """Write compact response JSONL and a provenance manifest."""
    if not scorers or any(not task or not scorer for task, scorer in scorers.items()):
        raise ValueError("scorers must be a non-empty task-to-scorer mapping")
    roots, paths = _discover_logs(log_paths)
    output_path = output_path.expanduser().resolve()
    if output_path.suffix != ".jsonl":
        raise ValueError("Preprocessed records output must end in .jsonl")
    manifest_path = records_manifest_path(output_path)
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    file_rows: list[dict[str, Any]] = []
    record_count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for path in tqdm(
                paths, total=len(paths), desc="Preprocessing logs", unit="log"
            ):
                file_row, records = _preprocess_file(path, scorers)
                file_rows.append(file_row)
                for record in records:
                    encoded = (
                        json.dumps(
                            record,
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    handle.write(encoded)
                    digest.update(encoded.encode())
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

    return load_records(output_path)


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
        "eligible": (
            status.lower() == "success" and total is not None and total == completed
        ),
    }


def _parse_log(
    path: Path, metadata: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse normalized sample records without loading full traces."""
    metadata = metadata or _read_log_metadata(path)
    if not metadata["eligible"]:
        return metadata, []
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
                ),
                "question_hash": question_hash(
                    sample.input,
                    sample.metadata,
                    choices=choices,
                    task=str(metadata["task"]),
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
    }.get(task, PARSER_VERSION)


def _is_log(path: Path) -> bool:
    """Return whether a path has a supported Inspect log suffix."""
    return any(path.name.lower().endswith(suffix) for suffix in LOG_SUFFIXES)


def question_hash(
    input_value: Any, metadata: Any, *, choices: list[str] | None = None, task: str = ""
) -> str:
    """Create a stable digest for a logical question."""
    values = metadata if isinstance(metadata, dict) else {}
    if task == "instruction_goal_hijacking":
        payload = {
            "access_code": canonical_item_value(values.get("access_code")),
            "attack": canonical_item_value(values.get("attack")),
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
            input_value, metadata, choices=choices, task=task
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
    text = unicodedata.normalize("NFKC", encoded).lower()
    text = text.replace("\x0bert_t", "at_t").replace("\\vert_t", "at_t")
    text = text.replace("\x0bert", "").replace("\\vert", "").replace("√", "surd")
    return re.sub(r"[^a-z0-9]+", "", text)


def digest_json(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON value."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
