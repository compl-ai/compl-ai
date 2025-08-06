import typer
from inspect_ai import eval
from inspect_ai import TaskInfo
from rich import print
from typing_extensions import Annotated

from complai._cli.utils import get_task_infos


def eval_command(
    model: Annotated[
        str, typer.Option("-m", help="Model to evaluate")
    ] = "vllm/HuggingFaceTB/SmolLM2-135M-Instruct",
    tasks: Annotated[
        str | None,
        typer.Option(
            "-t",
            help="Comma-separated list of tasks to run. If not provided, all COMPL-AI tasks are run.",
        ),
    ] = None,
    log_dir: Annotated[str, typer.Option(help="Directory to save logs to.")] = "logs/",
    limit: Annotated[
        int | None, typer.Option(help="Limit the number of samples per task.")
    ] = None,
    max_connections: Annotated[
        int,
        typer.Option(
            "-c",
            "--max-connections",
            help="Maximum number of concurrent connections to Model provider (defaults to 10)",
        ),
    ] = 10,
) -> None:
    """Run evals."""
    print(f"Running evals with model: [bold]{model}[/bold]")

    # Get TaskInfo objects from task names
    task_infos: list[TaskInfo] = get_task_infos(tasks)

    eval(
        tasks=task_infos,
        model=model,
        log_dir=log_dir,
        limit=limit,
        max_connections=max_connections,
    )
