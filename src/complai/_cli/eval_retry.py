import typer
from inspect_ai import eval_retry
from typing_extensions import Annotated

from complai._cli.utils import get_complai_tasks


def eval_retry_command(
    log_files: Annotated[
        list[str], typer.Argument(help="Log file(s) for task(s) to retry.")
    ],
    log_dir: Annotated[
        str, typer.Option(help="Directory to save logs to.", envvar="COMPLAI_LOG_DIR")
    ] = "./logs",
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
    """Retry interrupted evals."""
    get_complai_tasks()

    eval_retry(
        tasks=log_files,
        log_dir=log_dir,
        max_connections=max_connections,
        retry_on_error=retry_on_error,
    )
