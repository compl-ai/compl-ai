from pathlib import Path

import typer
from inspect_ai import list_tasks
from inspect_ai import TaskInfo


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
    available_tasks = {
        task.name: task
        for task in list_tasks(root_dir=Path(__file__).parent.parent / "tasks")
    }
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
