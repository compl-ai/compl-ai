import json
from pathlib import Path

DEFAULT_SUBSET_CONFIG = Path(__file__).with_name("subset_config.json")

def _load_config(path: Path | None = None) -> dict:
    path = path or DEFAULT_SUBSET_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {path}") from exc
    
    tasks = raw.get("tasks") if isinstance(raw, dict) else None
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Configuration must contain a non-empty 'tasks' mapping")
        
    return tasks

def load_scorers(path: Path | None = None) -> dict[str, str]:
    """Load a task-to-scorer mapping for the preprocessor."""
    tasks = _load_config(path)
    scorers = {}
    for task, info in tasks.items():
        if isinstance(info, dict) and "scorer" in info:
            scorers[task] = info["scorer"]
        elif isinstance(info, str):
            # Fallback if someone passes an old flat dict
            scorers[task] = info
    return scorers

def load_indices(path: Path | None = None) -> dict[str, str]:
    """Load the task-to-index assignment; every task must name its index."""
    tasks = _load_config(path)
    missing = sorted(task for task, info in tasks.items() if not isinstance(info, dict) or not info.get("index"))
    if missing:
        raise ValueError(f"Configuration tasks without an 'index': {', '.join(missing)}")
    return {task: info["index"] for task, info in tasks.items()}

def get_primary_metrics(path: Path | None = None) -> dict[str, str]:
    """Load a task-to-primary-metric mapping for evaluation."""
    tasks = _load_config(path)
    metrics = {}
    for task, info in tasks.items():
        if isinstance(info, dict) and "primary_metric" in info:
            metrics[task] = info["primary_metric"]
        else:
            # Fallback for old configs
            metrics[task] = "accuracy"
    return metrics
