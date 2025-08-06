from inspect_ai import list_tasks
from inspect_ai import TaskInfo
from rich import print


def list_command() -> None:
    """List all available tasks."""
    # Get all tasks
    tasks = list_tasks()
    if not tasks:
        print("No tasks available.")
        return

    # Group tasks by technical requirement
    tasks_by_requirement: dict[str, list[TaskInfo]] = {}
    for task in tasks:
        requirement = task.attribs["technical_requirement"]
        if requirement not in tasks_by_requirement:
            tasks_by_requirement[requirement] = []
        tasks_by_requirement[requirement].append(task)

    # Print tasks grouped by technical requirement
    for requirement, tasks in tasks_by_requirement.items():
        print(f"[bold]{requirement}[/bold]")
        print("  " + ",".join(task.name for task in tasks))
