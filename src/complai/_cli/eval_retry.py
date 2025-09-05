import typer
from inspect_ai import eval_retry
from typing_extensions import Annotated

from complai._cli.eval import LogFormat
from complai._cli.eval import LoggingLevel
from complai._cli.utils import patch_display_results


def eval_retry_command(
    log_files: Annotated[
        list[str], typer.Argument(help="Log file(s) for task(s) to retry.")
    ],
    log_level: Annotated[
        LoggingLevel,
        typer.Option(
            help="Python logger level for console.", envvar="COMPLAI_LOG_LEVEL"
        ),
    ] = LoggingLevel.WARNING,
    log_dir: Annotated[
        str, typer.Option(help="Directory to save logs to.", envvar="COMPLAI_LOG_DIR")
    ] = "./logs",
    log_samples: Annotated[
        bool, typer.Option(help="Log sample details", envvar="COMPLAI_LOG_SAMPLES")
    ] = True,
    log_buffer: Annotated[
        int | None,
        typer.Option(
            help="Number of samples to buffer before writing log file. If not specified, an appropriate default for the format and filesystem is chosen (10 for most cases, 100 for JSON logs on remote filesystems).",
            envvar="COMPLAI_LOG_BUFFER",
        ),
    ] = None,
    log_format: Annotated[
        LogFormat,
        typer.Option(help="Format for writing log files.", envvar="COMPLAI_LOG_FORMAT"),
    ] = LogFormat.EVAL,
    max_connections: Annotated[
        int,
        typer.Option(
            help="Maximum number of concurrent connections to model provider.",
            envvar="COMPLAI_MAX_CONNECTIONS",
        ),
    ] = 10,
    max_samples: Annotated[
        int | None,
        typer.Option(
            help="Maximum number of samples to run in parallel (default is `--max-connections`).",
            envvar="COMPLAI_MAX_SAMPLES",
        ),
    ] = None,
    max_subprocesses: Annotated[
        int | None,
        typer.Option(
            help="Maximum number of subprocesses to run in parallel (default is `os.cpu_count()`).",
            envvar="COMPLAI_MAX_SUBPROCESSES",
        ),
    ] = None,
    max_sandboxes: Annotated[
        int | None,
        typer.Option(
            help="Maximum number of sandboxes (per-provider) to run in parallel (default is `2 * os.cpu_count()`).",
            envvar="COMPLAI_MAX_SANDBOXES",
        ),
    ] = None,
    max_tasks: Annotated[
        int,
        typer.Option(
            help="Maximum number of tasks to run in parallel.",
            envvar="COMPLAI_MAX_TASKS",
        ),
    ] = 1,
    fail_on_error: Annotated[
        int | None,
        typer.Option(
            help="Threshold of sample errors to tolerate (by default, evals fail when any error occurs). Value between 0 to 1 to set a proportion; value greater than 1 to set a count.",
            envvar="COMPLAI_FAIL_ON_ERROR",
        ),
    ] = None,
    continue_on_fail: Annotated[
        bool,
        typer.Option(
            "--no-fail-on-error",
            help="Do not fail the eval if errors occur within samples (instead, continue running other samples)",
            envvar="COMPLAI_CONTINUE_ON_FAIL",
        ),
    ] = False,
    retry_on_error: Annotated[
        int,
        typer.Option(
            help="Number of times to retry on error.", envvar="COMPLAI_RETRY_ON_ERROR"
        ),
    ] = 0,
) -> None:
    """Retry interrupted tasks."""
    # Apply display monkey patch
    patch_display_results()

    eval_retry(
        tasks=log_files,
        max_connections=max_connections,
        max_samples=max_samples,
        max_subprocesses=max_subprocesses,
        max_sandboxes=max_sandboxes,
        max_tasks=max_tasks,
        log_level=log_level.value if log_level else None,
        log_dir=log_dir,
        log_samples=log_samples,
        log_buffer=log_buffer,
        log_format=log_format.value if log_format else None,
        fail_on_error=fail_on_error,
        continue_on_fail=continue_on_fail,
        retry_on_error=retry_on_error,
    )
