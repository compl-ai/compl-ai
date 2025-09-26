from datetime import datetime
from pathlib import Path
from typing import Callable

import typer
from inspect_ai import list_tasks
from inspect_ai import TaskInfo
from rich import print


def get_complai_tasks(
    filter: Callable[[TaskInfo], bool] | None = None,
) -> list[TaskInfo]:
    """
    Get all available tasks implemented in COMPL-AI.

    Args:
        filter (Callable[[TaskInfo], bool]): Filter to apply to the tasks.

    Returns:
        list[TaskInfo]: List of all available tasks.
    """
    return list_tasks(
        absolute=True, root_dir=Path(__file__).parent.parent / "tasks", filter=filter
    )


def get_task_infos_from_task_names(
    task_names: list[str], tasks_to_skip: list[str]
) -> list[TaskInfo]:
    """
    Get TaskInfo objects from a list of task names.

    Args:
        tasks_names (list[str]): List of task names to retrieve TaskInfo for. If empty,
            all available tasks are returned.
        tasks_to_skip (list[str]): List of task names to skip.

    Raises:
        typer.BadParameter: If one of the task names is not valid.

    Returns:
        list[TaskInfo]: List of TaskInfo objects for the specified tasks.
    """

    def filter(task: TaskInfo) -> bool:
        if task.name in tasks_to_skip:
            tasks_to_skip.remove(task.name)
            return False

        return not task_names or task.name in task_names

    tasks_to_run = get_complai_tasks(filter=filter)

    # Warn about tasks that were not skipped
    if tasks_to_skip:
        print(
            f"[yellow]WARNING[/yellow] The following task(s) were not skipped: {', '.join(tasks_to_skip)}"
        )

    # Warn about tasks that were not found
    not_found = set(task_names) - set(task.name for task in tasks_to_run)
    if not_found:
        print(
            f"[yellow]WARNING[/yellow] The following task(s) were not found: {', '.join(not_found)}"
        )

    return tasks_to_run


def parse_tasks(tasks: str | None) -> list[str]:
    """
    Parse task string into list of task names.

    Args:
        tasks (str | None): Comma-separated list of task names. If None, returns an empty list.

    Returns:
        list[str]: List of task names.
    """
    task_names = tasks.split(",") if tasks is not None else []

    return task_names


def get_task_infos(tasks: str | None, tasks_to_skip: str | None) -> list[TaskInfo]:
    """
    Get TaskInfo objects for the specified tasks.

    Args:
        tasks (str | None): Comma-separated list of task names. If None, all available tasks are returned.
        tasks_to_skip (str | None): Comma-separated list of task names to skip. If None, no tasks are skipped.

    Returns:
        list[TaskInfo]: List of TaskInfo objects for the specified tasks.
    """
    task_names = parse_tasks(tasks)
    tasks_to_skip_names = parse_tasks(tasks_to_skip)

    return get_task_infos_from_task_names(task_names, tasks_to_skip_names)


def patch_display_results() -> None:
    """
    Replace "inspect eval-retry" with "complai eval-retry" in display results.
    Should be called before invoking inspect_ai.eval_retry().
    """
    try:
        from inspect_ai._display.core import results

        original_task_interrupted = results.task_interrupted

        def custom_task_interrupted(profile, samples_completed):  # type: ignore
            result = original_task_interrupted(profile, samples_completed)

            if isinstance(result, str):
                return result.replace("inspect eval-retry", "complai eval-retry")
            elif hasattr(result, "_text") and isinstance(result._text, list):
                for i, segment in enumerate(result._text):
                    if isinstance(segment, tuple) and len(segment) >= 1:
                        text = segment[0]
                        if isinstance(text, str) and "inspect eval-retry" in text:
                            new_text = text.replace(
                                "inspect eval-retry", "complai eval-retry"
                            )
                            result._text[i] = (new_text,) + segment[1:]
                return result

        results.task_interrupted = custom_task_interrupted

    except (ImportError, AttributeError):
        pass


def get_log_dir(model: str) -> str:
    """
    Get the log directory for the model. If no log directory is provided,
    use model and timestamp to create a unique log directory name.
    """
    model_name = "_".join(model.split("/")[-2:])
    timestamp = (
        datetime.now().astimezone().isoformat(timespec="seconds").replace(":", "-")
    )

    return f"./logs/{model_name}_{timestamp}"


def bool_or_float(value: str) -> bool | float:
    """Parses a string into a boolean or a float."""
    if value.lower() in ("true", "t", "1"):
        return True
    if value.lower() in ("false", "f", "0"):
        return False
    try:
        return float(value)
    except ValueError:
        raise typer.BadParameter("Could not parse value as a boolean or a float")


def parse_samples_limit(limit: str | None) -> int | tuple[int, int] | None:
    if limit is not None:
        if "-" not in limit:
            return int(limit)
        else:
            limit_split = [int(r) for r in limit.split("-")]
            return (limit_split[0] - 1, limit_split[1])
    else:
        return None


def parse_sample_id(sample_id: str | None) -> list[str] | None:
    if sample_id is not None:
        return [id.strip() for id in sample_id.split(",")]
    else:
        return None
