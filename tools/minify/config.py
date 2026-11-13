from pathlib import Path

from complai.utils.scores import load_config


def load_scorers(path: Path | None = None) -> dict[str, str]:
    """Load the task-to-scorer mapping of the current release for the preprocessor."""
    return {task: info["scorer"] for task, info in load_config(path).items()}

def load_indices(path: Path | None = None) -> dict[str, str]:
    """Load the task-to-index assignment of the current release."""
    return {task: info["index"] for task, info in load_config(path).items()}

def get_primary_metrics(path: Path | None = None) -> dict[str, str]:
    """Load a task-to-primary-metric mapping for evaluation."""
    return {
        task: info.get("primary_metric", "accuracy")
        for task, info in load_config(path).items()
    }
