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

def get_task_allocations(path: Path | None = None) -> dict[str, str]:
    """Load a task-to-allocation-strategy mapping for the subset allocator."""
    tasks = _load_config(path)
    allocations = {}
    for task, info in tasks.items():
        if isinstance(info, dict):
            allocations[task] = info.get("allocation", "default")
        else:
            allocations[task] = "default"
    return allocations

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
