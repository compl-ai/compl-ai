"""Index Inspect evaluation logs in a reusable SQLite cache."""

import hashlib
import importlib.metadata
import json
import re
import sqlite3
import unicodedata
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from inspect_ai.log import read_eval_log_sample_summaries
from inspect_ai.log import read_eval_log_samples


PARSER_VERSION = "minify-inspect-v2"
TRUTHFULQA_PARSER_VERSION = "minify-inspect-v4-truthfulqa-stable-id"
CHOICE_PARSER_VERSION = "minify-inspect-v3-choice-content"
HLE_PARSER_VERSION = "minify-inspect-v3-hle-uid"
HIJACKING_PARSER_VERSION = "minify-inspect-v3-hijacking-source"
MMLU_PARSER_VERSION = "minify-inspect-v3-mmlu-content"
CHOICE_CONTENT_TASKS = {"gpqa_diamond", "truthfulqa"}
SCHEMA_VERSION = 1
LOG_SUFFIXES = (".eval", ".eval.gz")


@dataclass(frozen=True)
class InventorySummary:
    """Counts describing one log inventory refresh."""

    discovered: int
    cache_hits: int
    new: int
    changed: int
    removed: int
    excluded: int
    deferred: int
    failed: int


@dataclass(frozen=True)
class IndexedLogs:
    """Indexed inventory metadata and access to cached samples."""

    files: tuple[dict[str, Any], ...]
    summary: InventorySummary
    digest: str
    cache_path: Path
    scope_id: str
    tasks: tuple[str, ...]

    def iter_samples(self) -> Iterator[sqlite3.Row]:
        """Yield cached samples for the selected tasks."""
        task_clause = (
            f" AND samples.task IN ({','.join('?' for _ in self.tasks)})"
            if self.tasks
            else ""
        )
        connection = sqlite3.connect(self.cache_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                f"""
                SELECT samples.* FROM samples
                JOIN scope_files ON scope_files.path = samples.file_path
                WHERE scope_files.scope_id = ?{task_clause}
                """,
                (self.scope_id, *self.tasks),
            )
            yield from rows
        finally:
            connection.close()

    def sample_count(self) -> int:
        """Count cached samples for the selected tasks."""
        task_clause = (
            f" AND samples.task IN ({','.join('?' for _ in self.tasks)})"
            if self.tasks
            else ""
        )
        connection = sqlite3.connect(self.cache_path)
        try:
            row = connection.execute(
                f"""
                SELECT COUNT(*) FROM samples
                JOIN scope_files ON scope_files.path = samples.file_path
                WHERE scope_files.scope_id = ?{task_clause}
                """,
                (self.scope_id, *self.tasks),
            ).fetchone()
        finally:
            connection.close()
        return int(row[0]) if row is not None else 0


def discover_logs(paths: list[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Resolve input roots and discover supported Inspect logs."""
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


def index_logs(
    paths: list[Path],
    cache_path: Path,
    *,
    reindex: bool = False,
    tasks: Collection[str] | None = None,
    progress: Callable[[int, int, Path, str], None] | None = None,
) -> IndexedLogs:
    """Refresh the cache and return the current indexed inventory."""
    roots, files = discover_logs(paths)
    cache_path = cache_path.expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    inspect_version = importlib.metadata.version("inspect-ai")
    scope_id = _digest_json([str(path) for path in roots])
    counts = {
        "cache_hits": 0,
        "new": 0,
        "changed": 0,
        "excluded": 0,
        "deferred": 0,
        "failed": 0,
    }
    task_filter = set(tasks or ())

    with sqlite3.connect(cache_path, timeout=30.0) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        _create_schema(connection)
        previous = {
            row["path"]
            for row in connection.execute(
                "SELECT path FROM scope_files WHERE scope_id = ?", (scope_id,)
            )
        }
        current = {str(path) for path in files}
        removed = len(previous - current)

        for file_number, path in enumerate(files, start=1):
            stat = path.stat()
            cached = connection.execute(
                "SELECT * FROM files WHERE path = ?", (str(path),)
            ).fetchone()
            reusable = bool(
                not reindex
                and cached is not None
                and cached["size"] == stat.st_size
                and cached["mtime_ns"] == stat.st_mtime_ns
                and cached["parser_version"] == _parser_version(str(cached["task"]))
                and cached["inspect_version"] == inspect_version
                and cached["parse_status"] != "error"
                and not (
                    cached["parse_status"] == "deferred"
                    and (not task_filter or cached["task"] in task_filter)
                )
            )
            if reusable:
                counts["cache_hits"] += 1
                if task_filter and cached["task"] not in task_filter:
                    counts["deferred"] += 1
                elif cached["parse_status"] == "excluded":
                    counts["excluded"] += 1
                if progress is not None:
                    progress(file_number, len(files), path, "cache_hit")
                continue

            key = "new" if cached is None else "changed"
            counts[key] += 1
            samples: list[dict[str, Any]]
            try:
                metadata = _read_log_metadata(path)
                if task_filter and metadata["task"] not in task_filter and not reindex:
                    sha256 = ""
                    samples = []
                    parse_status = "deferred"
                    counts["deferred"] += 1
                else:
                    sha256 = _file_sha256(path)
                    metadata, samples = _parse_log(path, metadata)
                    parse_status = "ok" if metadata["eligible"] else "excluded"
                    if parse_status == "excluded":
                        counts["excluded"] += 1
                error = ""
            except Exception as exc:  # noqa: BLE001
                sha256 = ""
                metadata = _empty_metadata()
                samples = []
                parse_status = "error"
                error = f"{type(exc).__name__}: {exc}"[:4000]
                counts["failed"] += 1
            with connection:
                connection.execute(
                    "DELETE FROM samples WHERE file_path = ?", (str(path),)
                )
                connection.execute(
                    """
                    INSERT INTO files (
                        path, size, mtime_ns, sha256, parser_version, inspect_version,
                        parse_status, error, run_id, created, status, model, task,
                        dataset, sample_count, eligible
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        size=excluded.size, mtime_ns=excluded.mtime_ns,
                        sha256=excluded.sha256, parser_version=excluded.parser_version,
                        inspect_version=excluded.inspect_version,
                        parse_status=excluded.parse_status, error=excluded.error,
                        run_id=excluded.run_id, created=excluded.created,
                        status=excluded.status, model=excluded.model,
                        task=excluded.task, dataset=excluded.dataset,
                        sample_count=excluded.sample_count, eligible=excluded.eligible
                    """,
                    (
                        str(path),
                        stat.st_size,
                        stat.st_mtime_ns,
                        sha256,
                        _parser_version(str(metadata["task"])),
                        inspect_version,
                        parse_status,
                        error,
                        metadata["run_id"],
                        metadata["created"],
                        metadata["status"],
                        metadata["model"],
                        metadata["task"],
                        metadata["dataset"],
                        metadata["sample_count"],
                        int(metadata["eligible"]),
                    ),
                )
                if parse_status == "ok":
                    connection.executemany(
                        """
                        INSERT INTO samples (
                            file_path, run_id, created, model, task, dataset,
                            sample_id, epoch, scores_json, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                str(path),
                                metadata["run_id"],
                                metadata["created"],
                                metadata["model"],
                                metadata["task"],
                                metadata["dataset"],
                                sample["sample_id"],
                                sample["epoch"],
                                sample["scores_json"],
                                sample["content_hash"],
                            )
                            for sample in samples
                        ],
                    )
            if progress is not None:
                progress(file_number, len(files), path, parse_status)

        with connection:
            connection.execute(
                "INSERT OR REPLACE INTO scopes (scope_id, roots_json) VALUES (?, ?)",
                (scope_id, json.dumps([str(path) for path in roots])),
            )
            connection.execute(
                "DELETE FROM scope_files WHERE scope_id = ?", (scope_id,)
            )
            connection.executemany(
                "INSERT INTO scope_files (scope_id, path) VALUES (?, ?)",
                [(scope_id, str(path)) for path in files],
            )

        file_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT files.* FROM files
                JOIN scope_files ON scope_files.path = files.path
                WHERE scope_files.scope_id = ? ORDER BY files.path
                """,
                (scope_id,),
            )
        ]
        failures = [row for row in file_rows if row["parse_status"] == "error"]
        if failures:
            detail = "; ".join(f"{row['path']}: {row['error']}" for row in failures[:3])
            raise ValueError(
                f"Failed to index {len(failures)} Inspect log(s): {detail}"
            )
    summary = InventorySummary(
        discovered=len(files),
        cache_hits=counts["cache_hits"],
        new=counts["new"],
        changed=counts["changed"],
        removed=removed,
        excluded=counts["excluded"],
        deferred=counts["deferred"],
        failed=counts["failed"],
    )
    provenance = [
        {
            "path": row["path"],
            "size": row["size"],
            "mtime_ns": row["mtime_ns"],
            "sha256": (
                "" if task_filter and row["task"] not in task_filter else row["sha256"]
            ),
            "parser_version": row["parser_version"],
            "inspect_version": row["inspect_version"],
            "parse_status": (
                "deferred"
                if task_filter and row["task"] not in task_filter
                else row["parse_status"]
            ),
            "run_id": row["run_id"],
            "status": row["status"],
            "model": row["model"],
            "task": row["task"],
            "dataset": row["dataset"],
            "created": row["created"],
            "sample_count": row["sample_count"],
        }
        for row in file_rows
    ]
    return IndexedLogs(
        files=tuple(provenance),
        summary=summary,
        digest=_digest_json(provenance),
        cache_path=cache_path,
        scope_id=scope_id,
        tasks=tuple(sorted(task_filter)),
    )


def inventory_json(indexed: IndexedLogs) -> dict[str, Any]:
    """Serialize inventory metadata for an artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "task_parser_versions": {
            task: _parser_version(task) for task in sorted(CHOICE_CONTENT_TASKS)
        }
        | {
            "hle": HLE_PARSER_VERSION,
            "instruction_goal_hijacking": HIJACKING_PARSER_VERSION,
            "mmlu_pro": MMLU_PARSER_VERSION,
        },
        "digest": indexed.digest,
        "summary": asdict(indexed.summary),
        "files": list(indexed.files),
    }


def _create_schema(connection: sqlite3.Connection) -> None:
    """Create the SQLite cache tables and indexes."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            inspect_version TEXT NOT NULL,
            parse_status TEXT NOT NULL,
            error TEXT NOT NULL,
            run_id TEXT NOT NULL,
            created TEXT NOT NULL,
            status TEXT NOT NULL,
            model TEXT NOT NULL,
            task TEXT NOT NULL,
            dataset TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            eligible INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS samples (
            file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
            run_id TEXT NOT NULL,
            created TEXT NOT NULL,
            model TEXT NOT NULL,
            task TEXT NOT NULL,
            dataset TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            scores_json TEXT NOT NULL,
            content_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS samples_file ON samples(file_path);
        CREATE INDEX IF NOT EXISTS samples_task ON samples(task);
        CREATE TABLE IF NOT EXISTS scopes (
            scope_id TEXT PRIMARY KEY,
            roots_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scope_files (
            scope_id TEXT NOT NULL REFERENCES scopes(scope_id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            PRIMARY KEY (scope_id, path)
        );
        """
    )


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
    eligible = status.lower() == "success" and total is not None and total == completed
    dataset_spec = getattr(spec, "dataset", None)
    dataset = str(
        getattr(dataset_spec, "name", None)
        or getattr(dataset_spec, "location", None)
        or "unknown_dataset"
    )
    return {
        "run_id": str(getattr(spec, "eval_id", "") or getattr(spec, "run_id", "")),
        "created": str(getattr(spec, "created", "")),
        "status": status,
        "model": str(getattr(spec, "model", "")),
        "task": str(getattr(spec, "task", "")),
        "dataset": dataset,
        "sample_count": int(completed or 0),
        "eligible": eligible,
    }


def _parse_log(
    path: Path, metadata: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse normalized sample records without loading full traces."""
    metadata = metadata or _read_log_metadata(path)
    if not metadata["eligible"]:
        return metadata, []
    samples = []
    sample_rows: Iterable[Any]
    if metadata["task"] in CHOICE_CONTENT_TASKS:
        sample_rows = read_eval_log_samples(
            str(path),
            exclude_fields={
                "attachments",
                "events",
                "events_data",
                "files",
                "messages",
                "model_usage",
                "output",
                "role_usage",
                "setup",
                "store",
                "timelines",
            },
        )
    else:
        sample_rows = read_eval_log_sample_summaries(str(path))
    for sample in sample_rows:
        scores = {
            str(name): _json_value(getattr(score, "value", None))
            for name, score in (sample.scores or {}).items()
        }
        samples.append(
            {
                "sample_id": _logical_sample_id(
                    str(metadata["task"]), sample.id, sample.metadata, sample.input
                ),
                "epoch": int(sample.epoch or 1),
                "scores_json": json.dumps(
                    scores, sort_keys=True, separators=(",", ":")
                ),
                "content_hash": _content_hash(
                    sample.input,
                    sample.target,
                    sample.metadata,
                    sample.choices,
                    task=str(metadata["task"]),
                ),
            }
        )
    return metadata, samples


def _content_hash(
    input_value: Any, target: Any, metadata: Any, choices: Any = None, *, task: str = ""
) -> str:
    """Create a stable digest for an evaluation item's content."""
    stable_keys = {
        "id",
        "label",
        "sensitive_attribute",
        "original_question",
        "original_answer",
        "topic",
        "category",
        "task_id",
        "type",
        "proposition",
        "ground_truth",
        "formatted_ground_truth",
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
    values = metadata if isinstance(metadata, dict) else {}
    if task == "instruction_goal_hijacking":
        return _digest_json(
            {
                "recipe": "minify-hijacking-item-v1",
                "access_code": _canonical_item_value(values.get("access_code")),
                "attack": _canonical_item_value(values.get("attack")),
            }
        )
    if task == "mmlu_pro":
        input_value = _canonical_mmlu_value(input_value)
        # Older Inspect summaries omit choices; the stable dataset sample ID,
        # normalized question, and target remain available in every version.
        choices = None
    stable_metadata = {key: values[key] for key in sorted(stable_keys) if key in values}
    payload = {
        "recipe": "minify-item-v1",
        "input": _canonical_item_value(input_value),
        "target": _canonical_item_value(target),
        "metadata": _canonical_item_value(stable_metadata),
    }
    canonical_choice_content = _canonical_choice_content(choices, target)
    if canonical_choice_content is not None:
        payload.update(
            {
                "recipe": "minify-choice-item-v1",
                "choices": canonical_choice_content[0],
                "target": canonical_choice_content[1],
            }
        )
    return _digest_json(payload)


def _canonical_choice_content(
    choices: Any, target: Any
) -> tuple[list[Any], list[Any]] | None:
    """Canonicalize choice content independently of label order."""
    if not isinstance(choices, list) or not choices:
        return None
    labels = target if isinstance(target, list) else [target]
    if not labels or any(
        not isinstance(label, str)
        or len(label) != 1
        or not "A" <= label.upper() <= "Z"
        or ord(label.upper()) - ord("A") >= len(choices)
        for label in labels
    ):
        return None
    canonical_choices = [_canonical_item_value(choice) for choice in choices]
    key = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
    correct = [canonical_choices[ord(label.upper()) - ord("A")] for label in labels]
    return sorted(canonical_choices, key=key), sorted(correct, key=key)


def _parser_version(task: str) -> str:
    """Return the parser recipe version for a task."""
    if task == "truthfulqa":
        return TRUTHFULQA_PARSER_VERSION
    if task == "hle":
        return HLE_PARSER_VERSION
    if task == "instruction_goal_hijacking":
        return HIJACKING_PARSER_VERSION
    if task == "mmlu_pro":
        return MMLU_PARSER_VERSION
    return CHOICE_PARSER_VERSION if task in CHOICE_CONTENT_TASKS else PARSER_VERSION


def _logical_sample_id(
    task: str, sample_id: Any, metadata: Any, input_value: Any = None
) -> str:
    """Return the stable logical sample identifier for a task."""
    if task == "truthfulqa" and isinstance(input_value, str):
        digest = hashlib.md5(input_value.encode()).hexdigest()[:8]
        return f"truthfulqa_{digest}"
    if task == "hle" and isinstance(metadata, dict) and metadata.get("uid"):
        return str(metadata["uid"])
    return str(sample_id)


def _canonical_item_value(value: Any) -> Any:
    """Normalize item content for deterministic hashing."""
    encoded = _json_value(value)
    if isinstance(encoded, list):
        return [_canonical_item_value(item) for item in encoded]
    if isinstance(encoded, dict):
        is_message = "role" in encoded and "content" in encoded
        return {
            key: _canonical_item_value(item)
            for key, item in sorted(encoded.items())
            if not (is_message and key == "id")
        }
    return encoded


def _canonical_mmlu_value(value: Any) -> Any:
    """Normalize legacy MMLU text variants for stable hashing."""
    encoded = _json_value(value)
    if isinstance(encoded, list):
        return [_canonical_mmlu_value(item) for item in encoded]
    if isinstance(encoded, dict):
        return {
            key: _canonical_mmlu_value(item) for key, item in sorted(encoded.items())
        }
    if not isinstance(encoded, str):
        return encoded
    text = unicodedata.normalize("NFKC", encoded).lower()
    text = text.replace("\x0bert_t", "at_t").replace("\\vert_t", "at_t")
    text = text.replace("\x0bert", "").replace("\\vert", "").replace("√", "surd")
    return re.sub(r"[^a-z0-9]+", "", text)


def _json_value(value: Any) -> Any:
    """Convert an arbitrary value to JSON-compatible data."""
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _empty_metadata() -> dict[str, Any]:
    """Return blank metadata for a failed parse."""
    return {
        "run_id": "",
        "created": "",
        "status": "",
        "model": "",
        "task": "",
        "dataset": "",
        "sample_count": 0,
        "eligible": False,
    }


def _file_sha256(path: Path) -> str:
    """Compute a file's SHA-256 digest incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON value."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_log(path: Path) -> bool:
    """Return whether a path has a supported Inspect log suffix."""
    return any(path.name.lower().endswith(suffix) for suffix in LOG_SUFFIXES)
