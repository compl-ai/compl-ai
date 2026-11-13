from rich import print

from complai._cli.utils import get_complai_tasks
from complai.utils.scores import load_config
from complai.utils.scores import read_config


def list_command() -> None:
    """List all available tasks, grouped by index of the current Core release."""
    names = sorted(task.name for task in get_complai_tasks())
    if not names:
        print("No tasks available.")
        return

    release = load_config()
    by_index: dict[str, list[str]] = {
        index: [] for index in ("capability", "reliability", "safety")
    }
    for name in names:
        if name in release:
            by_index.setdefault(release[name]["index"], []).append(name)

    print(f"[dim]Core release: {read_config()['current_subset']}[/dim]")
    for index, tasks in by_index.items():
        if not tasks:
            continue
        print(f"[bold]{index.capitalize()}[/bold]")
        print("  " + ", ".join(tasks))

    other = [name for name in names if name not in release]
    if other:
        print("[bold]Not in Core[/bold] [dim](run with --full -t <task>)[/dim]")
        print("  " + ", ".join(other))
