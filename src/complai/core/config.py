from pathlib import Path

import yaml


DEFAULT_SCORER_CONFIG = Path(__file__).with_name("default_scorers.yaml")


def load_scorers(path: Path | None = None) -> dict[str, str]:
    """Load a task-to-scorer mapping from YAML or JSON."""
    path = path or DEFAULT_SCORER_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML or JSON configuration: {path}") from exc
    tasks = raw.get("tasks") if isinstance(raw, dict) else None
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Configuration must contain a non-empty 'tasks' mapping")
    if any(
        not isinstance(task, str)
        or not task
        or not isinstance(scorer, str)
        or not scorer
        for task, scorer in tasks.items()
    ):
        raise ValueError("Every tasks entry must map a task string to a scorer string")

    return dict(tasks)
