import json
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset

from complai.core.index import _content_hash
from complai.core.index import _logical_sample_id


def read_eval_subset(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read a subset JSONL file grouped by task."""
    selected: dict[str, list[dict[str, Any]]] = {}
    item_ids: set[str] = set()
    with path.expanduser().open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            required = ("task", "sample_id", "item_id", "content_hash")
            if not isinstance(row, dict) or any(
                not isinstance(row.get(field), str) or not row[field]
                for field in required
            ):
                raise ValueError(
                    f"{path}:{line_number}: expected string fields "
                    + ", ".join(required)
                )
            task, dataset, sample_id = _parse_item_id(row["item_id"], path, line_number)
            if row["task"] != task or row["sample_id"] != sample_id:
                raise ValueError(f"{path}:{line_number}: item identity does not match")
            if "dataset" in row and row["dataset"] != dataset:
                raise ValueError(
                    f"{path}:{line_number}: dataset does not match item_id"
                )
            row["dataset"] = dataset
            if row["item_id"] in item_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate item_id {row['item_id']!r}"
                )
            item_ids.add(row["item_id"])
            selected.setdefault(task, []).append(row)
    if not selected:
        raise ValueError(f"Subset is empty: {path}")
    return selected


def apply_eval_subset(
    task_names: list[str], tasks: list[Task], selected: dict[str, list[dict[str, Any]]]
) -> None:
    """Filter and order task datasets to match a subset."""
    if len(task_names) != len(tasks) or set(task_names) != set(selected):
        raise ValueError("Requested tasks do not match the subset tasks")

    for task_name, task in zip(task_names, tasks, strict=True):
        dataset = task.dataset
        if dataset is None:
            raise ValueError(f"Task {task_name!r} has no dataset")
        samples = {
            _logical_sample_id(
                task_name, sample.id, sample.metadata, sample.input
            ): sample
            for sample in dataset
        }
        rows = selected[task_name]
        if {row["dataset"] for row in rows} != {dataset.name}:
            raise ValueError(f"Task {task_name!r} does not match the subset dataset")

        ordered = []
        for row in rows:
            sample = samples.get(row["sample_id"])
            if sample is None:
                raise ValueError(
                    f"Task {task_name!r} is missing sample {row['sample_id']!r}"
                )
            content_hash = _content_hash(
                sample.input,
                sample.target,
                sample.metadata,
                sample.choices,
                task=task_name,
            )
            if content_hash != row["content_hash"]:
                raise ValueError(f"Content mismatch for item {row['item_id']!r}")
            ordered.append(sample)
        task.dataset = MemoryDataset(
            ordered,
            name=dataset.name,
            location=dataset.location,
            shuffled=dataset.shuffled,
        )


def _parse_item_id(item_id: str, path: Path, line_number: int) -> tuple[str, str, str]:
    """Split and validate a stable item identifier."""
    parts = item_id.split("::", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"{path}:{line_number}: invalid item_id")
    return parts[0], parts[1], parts[2]
