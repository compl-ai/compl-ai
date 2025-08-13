import typer
from inspect_ai import eval
from inspect_ai import TaskInfo
from rich import print
from typing_extensions import Annotated

from complai._cli.utils import get_task_infos


def eval_command(
    tasks: Annotated[
        str | None,
        typer.Option(
            "-t",
            help="Comma-separated list of tasks to run. If not provided, all COMPL-AI tasks are run.",
            envvar="COMPLAI_TASKS",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "-m",
            "--model",
            help="Model to evaluate. Use the [Inspect](https://inspect.aisi.org.uk/) syntax for specifying models. See [inspect.aisi.org.uk/models](https://inspect.aisi.org.uk/models.html) and [inspect.aisi.org.uk/providers](https://inspect.aisi.org.uk/providers.html) for details.",
            envvar="COMPLAI_MODEL",
        ),
    ] = "vllm/HuggingFaceTB/SmolLM2-135M-Instruct",
    log_dir: Annotated[
        str, typer.Option(help="Directory to save logs to.", envvar="COMPLAI_LOG_DIR")
    ] = "./logs",
    limit: Annotated[
        int | None,
        typer.Option(
            help="Limit the number of samples per task.", envvar="COMPLAI_LIMIT"
        ),
    ] = None,
    max_connections: Annotated[
        int,
        typer.Option(
            help="Maximum number of concurrent connections to Model provider.",
            envvar="COMPLAI_MAX_CONNECTIONS",
        ),
    ] = 64,
    retry_on_error: Annotated[
        int,
        typer.Option(
            help="Number of times to retry on error.", envvar="COMPLAI_RETRY_ON_ERROR"
        ),
    ] = 0,
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
        retry_on_error=retry_on_error,
    )
