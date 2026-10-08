"""Item labels: loading the raw labeling output, the taxonomy, and addressing items.

``load_labels`` is the single loader for ``tools/label/labels`` used by ``irt``
(writing secondary labels into params as reporting metadata) and the maintainer
tooling. ``load_subcategories`` reads the sub-categories from
``tools/label/taxonomy.csv``; ``irt`` writes them into params, so consumers of a
fitted params (``predict``) never need the label files or the taxonomy.
"""

from __future__ import annotations

import collections
import csv
import json
from dataclasses import dataclass
from pathlib import Path

UNKNOWN_LABEL = "unknown"
DEFAULT_LABELS_DIR = Path("tools/label/labels")
TAXONOMY_PATH = Path("tools/label/taxonomy.csv")


@dataclass(frozen=True)
class ItemLabel:
    primary: str
    secondary: str | None
    tags: tuple[str, ...]


# label-file stem (task or dataset basename) -> sample_id -> label
ItemLabels = dict[str, dict[str, ItemLabel]]


def split_item_id(item_id: str) -> tuple[str, str, str]:
    """Return ``(task, dataset, sample_id)`` from a ``task::dataset::sample_id`` item id."""
    task, dataset, sample_id = item_id.split("::", 2)
    return task, dataset, sample_id


def lookup_label(labels: ItemLabels, task: str, dataset: str, sample_id: str) -> ItemLabel | None:
    """Find an item's label, trying the dataset basename and then the task name as file key."""
    for key in (dataset.split("/")[-1], task):
        found = labels.get(key, {}).get(sample_id)
        if found is not None:
            return found
    return None


def load_labels(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    datasets_dir: Path | None = None,
    with_aliases: bool = True,
) -> ItemLabels:
    """Load ``<name>.jsonl`` label files plus ``<name>_patch.jsonl`` human overrides.

    Label files may key samples differently from the eval logs, so when the
    matching dataset file is available each label is also registered under the
    candidate ids found there (uid, task_id, 0- and 1-based row index). An alias
    never replaces a label file's own ``sample_id``: numeric ids often equal
    another sample's row index. Pass ``with_aliases=False`` to get one entry per
    labeled sample, keyed by the label file's own ``sample_id``.
    """
    if datasets_dir is None:
        datasets_dir = labels_dir.parent / "datasets"
    labels: ItemLabels = {}
    if not labels_dir.exists():
        return labels
    for label_file in sorted(labels_dir.glob("*.jsonl")):
        if label_file.stem.endswith("_patch"):
            continue
        name = label_file.stem
        table: dict[str, ItemLabel] = {}
        for row in _read_jsonl(label_file):
            assigned = row.get("llm_assigned", {}) or {}
            secondary = assigned.get("secondary_labels") or []
            table[str(row.get("sample_id", ""))] = ItemLabel(
                primary=assigned.get("primary_label") or UNKNOWN_LABEL,
                secondary=secondary[0] if secondary else None,
                tags=tuple(assigned.get("tags") or []),
            )
        patch_file = label_file.with_name(f"{name}_patch.jsonl")
        if patch_file.exists():
            for row in _read_jsonl(patch_file):
                sample_id = str(row.get("sample_id", ""))
                current = table.get(sample_id) or ItemLabel(UNKNOWN_LABEL, None, ())
                secondary = current.secondary
                if "human_secondary_labels" in row:
                    secondary = (row["human_secondary_labels"] or [None])[0]
                elif "human_secondary_label" in row:
                    secondary = row["human_secondary_label"]
                table[sample_id] = ItemLabel(
                    primary=row.get("human_primary_label", current.primary),
                    secondary=secondary,
                    tags=tuple(row["human_tags"]) if "human_tags" in row else current.tags,
                )
        if with_aliases:
            for sample_id, aliases in _dataset_aliases(datasets_dir / label_file.name).items():
                if sample_id in table:
                    for alias in aliases:
                        table.setdefault(alias, table[sample_id])
        labels[name] = table
    return labels


@dataclass(frozen=True)
class Subcategories:
    """Reporting sub-categories, each contained in one index.

    ``by_label`` maps ``(index, secondary label)`` to the sub-category tag that
    label reports under within that index; ``definitions`` maps each tag to its
    taxonomy row (``label_name``, ``core_index``, ``description``).
    """

    by_label: dict[tuple[str, str], str]
    definitions: dict[str, dict[str, str]]

    def of(self, index: str, label: str) -> str:
        """The sub-category of an item with ``label`` in a task assigned to ``index``."""
        if label == UNKNOWN_LABEL:
            return UNKNOWN_LABEL
        try:
            return self.by_label[(index, label)]
        except KeyError:
            raise ValueError(
                f"Taxonomy has no {index} sub-category for label {label!r}; "
                f"set {index}_subcategory for it in {TAXONOMY_PATH} and refit"
            ) from None

    def to_params(self) -> dict[str, dict]:
        """The params ``labels.subcategories`` block: definitions and, per index, label -> sub-category."""
        by_index: dict[str, dict[str, str]] = collections.defaultdict(dict)
        for (index, label), tag in sorted(self.by_label.items()):
            by_index[index][label] = tag
        return {"definitions": dict(sorted(self.definitions.items())), "by_index": dict(by_index)}

    @classmethod
    def from_params(cls, block: dict[str, dict]) -> Subcategories:
        """Inverse of ``to_params``."""
        return cls(
            by_label={
                (index, label): tag
                for index, labels in block["by_index"].items()
                for label, tag in labels.items()
            },
            definitions=block["definitions"],
        )


def load_subcategories(path: Path = TAXONOMY_PATH) -> Subcategories:
    """Read the sub-category rows and the per-index ``<index>_subcategory`` columns of the taxonomy."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    definitions = {
        row["label_id"]: {key: row[key] for key in ("label_name", "core_index", "description")}
        for row in rows if row["label_type"] == "subcategory"
    }
    indices = {definition["core_index"] for definition in definitions.values()}
    by_label: dict[tuple[str, str], str] = {}
    for row in rows:
        if row["label_type"] == "subcategory":
            continue
        for index in indices:
            tag = row.get(f"{index}_subcategory", "")
            if not tag:
                continue
            if definitions.get(tag, {}).get("core_index") != index:
                raise ValueError(f"{row['label_id']}: {tag!r} is not a {index} sub-category")
            by_label[(index, row["label_id"])] = tag
    return Subcategories(by_label=by_label, definitions=definitions)


def _dataset_aliases(dataset_file: Path) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = collections.defaultdict(list)
    if not dataset_file.exists():
        return aliases
    for index, row in enumerate(_read_jsonl(dataset_file)):
        sample_id = str(row.get("sample_id", ""))
        meta = row.get("metadata", {}) or {}
        for candidate in (
            sample_id, str(meta.get("uid", "")), str(meta.get("task_id", "")),
            str(index), str(index + 1),
        ):
            if candidate:
                aliases[sample_id].append(candidate)
    return aliases


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
