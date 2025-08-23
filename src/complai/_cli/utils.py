from pathlib import Path
from typing import Any

import typer
import yaml
from inspect_ai import list_tasks
from inspect_ai import TaskInfo


def get_complai_tasks() -> list[TaskInfo]:
    """
    Get all available tasks implemented in COMPL-AI.

    Returns:
        list[TaskInfo]: List of all available tasks.
    """
    return list_tasks(absolute=True, root_dir=Path(__file__).parent.parent / "tasks")


def get_task_infos_from_task_names(task_names: list[str]) -> list[TaskInfo]:
    """
    Get TaskInfo objects from a list of task names.

    Args:
        tasks_names (list[str]): List of task names to retrieve TaskInfo for.

    Raises:
        typer.BadParameter: If one of the task names is not valid.

    Returns:
        list[TaskInfo]: List of TaskInfo objects for the specified tasks.
    """
    # Get all available tasks
    available_tasks = {task.name: task for task in get_complai_tasks()}
    if not task_names:
        return list(available_tasks.values())

    # Validate task names and get TaskInfo objects
    tasks_to_run: list[TaskInfo] = []
    for task_name in task_names:
        if task_name not in available_tasks:
            raise typer.BadParameter(
                f"Task '{task_name}' is not a valid task. Available tasks: {', '.join(available_tasks.keys())}"
            )
        task = available_tasks[task_name]
        tasks_to_run.append(task)

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


def get_task_infos(tasks: str | None) -> list[TaskInfo]:
    """
    Get TaskInfo objects for the specified tasks.

    Args:
        tasks (str | None): Comma-separated list of task names. If None, all available tasks are returned.

    Returns:
        list[TaskInfo]: List of TaskInfo objects for the specified tasks.
    """
    task_names = parse_tasks(tasks)

    return get_task_infos_from_task_names(task_names)


def parse_cli_args(
    args: tuple[str] | list[str] | None, force_str: bool = False
) -> dict[str, Any]:
    params: dict[str, Any] = dict()
    if args:
        for arg in list(args):
            parts = arg.split("=")
            if len(parts) > 1:
                key = parts[0].replace("-", "_")
                value = yaml.safe_load("=".join(parts[1:]))
                if isinstance(value, str):
                    value = value.split(",")
                    value = value if len(value) > 1 else value[0]
                params[key] = str(value) if force_str else value
    return params


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
