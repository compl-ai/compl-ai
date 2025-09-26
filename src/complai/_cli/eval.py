from enum import Enum
from typing import cast

import typer
from inspect_ai import eval_set
from inspect_ai import TaskInfo
from inspect_ai._cli.util import parse_cli_config
from rich import print
from typing_extensions import Annotated

from complai._cli.utils import bool_or_float
from complai._cli.utils import get_log_dir
from complai._cli.utils import get_task_infos
from complai._cli.utils import parse_sample_id
from complai._cli.utils import parse_samples_limit
from complai._cli.utils import patch_display_results


class LoggingLevel(str, Enum):
    """Logging level."""

    DEBUG = "debug"
    TRACE = "trace"
    HTTP = "http"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogFormat(str, Enum):
    """Output format for logs."""

    EVAL = "eval"
    JSON = "json"


class ReasoningEffortLevel(str, Enum):
    """Reasoning effort level."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def eval_command(
    model: Annotated[
        str,
        typer.Argument(
            help="Model to evaluate. Use the [Inspect](https://inspect.aisi.org.uk/) syntax for specifying models: provider/model_name (e.g. openai/gpt-5-nano or vllm/Qwen/Qwen3-1.7B). See [inspect.aisi.org.uk/models](https://inspect.aisi.org.uk/models.html) and [inspect.aisi.org.uk/providers](https://inspect.aisi.org.uk/providers.html) for details and supported providers.",
            envvar="COMPLAI_MODEL",
        ),
    ],
    tasks: Annotated[
        str | None,
        typer.Option(
            "-t",
            "--tasks",
            help="Comma-separated list of tasks to run. If not provided, all COMPL-AI tasks are run.",
            envvar="COMPLAI_TASKS",
        ),
    ] = None,
    tasks_to_skip: Annotated[
        str | None,
        typer.Option(
            "--skip",
            "--tasks-to-skip",
            help="Comma-separated list of tasks to skip.",
            envvar="COMPLAI_TASKS_TO_SKIP",
        ),
    ] = None,
    task_args: Annotated[
        list[str],
        typer.Option(
            "-T",
            "--task-arg",
            help="One or more task arguments (e.g. `-T arg1=value1,arg2=value2`).",
        ),
    ] = [],
    task_config: Annotated[
        str | None,
        typer.Option("--task-config", help="Task arguments file (JSON or YAML)."),
    ] = None,
    model_base_url: Annotated[
        str | None,
        typer.Option(
            help="Base URL for for model API.", envvar="COMPLAI_MODEL_BASE_URL"
        ),
    ] = None,
    model_args: Annotated[
        list[str],
        typer.Option(
            "-M",
            "--model-arg",
            help="One or more model arguments (e.g. `-M arg1=value1,arg2=value2`).",
        ),
    ] = [],
    model_config: Annotated[
        str | None,
        typer.Option("--model-config", help="Model arguments file (JSON or YAML)."),
    ] = None,
    retry_attempts: Annotated[
        int | None,
        typer.Option(
            help="Maximum number of retry attempts before giving up.",
            envvar="COMPLAI_RETRY_ATTEMPTS",
        ),
    ] = 3,
    retry_wait: Annotated[
        float,
        typer.Option(
            help="Time to wait between attempts, increased exponentially. (30 results in waits of 30, 60, 120, 240, etc.). Wait time per-retry will in no case by longer than 1 hour.",
            envvar="COMPLAI_RETRY_WAIT",
        ),
    ] = 30,
    retry_connections: Annotated[
        float,
        typer.Option(
            help="Reduce max_connections at this rate with each retry (1.0 results in no reduction).",
            envvar="COMPLAI_RETRY_CONNECTIONS",
        ),
    ] = 1.0,
    retry_cleanup: Annotated[
        bool,
        typer.Option(
            help="Cleanup failed log files after retries.",
            envvar="COMPLAI_RETRY_CLEANUP",
        ),
    ] = True,
    limit: Annotated[
        str | None,
        typer.Option(
            "-l",
            "--limit",
            help="Limit the number of samples per task, e.g. 10 or 10-20.",
            envvar="COMPLAI_LIMIT",
        ),
    ] = None,
    sample_id: Annotated[
        str | None,
        typer.Option(
            help="Evaluate a specific sample (e.g. 44) or list of samples (e.g. 44,63,91).",
            envvar="COMPLAI_SAMPLE_ID",
        ),
    ] = None,
    sample_shuffle: Annotated[
        int | None,
        typer.Option(
            help="Shuffle order of samples (pass a seed).",
            envvar="COMPLAI_SAMPLE_SHUFFLE",
        ),
    ] = None,
    epochs: Annotated[
        int,
        typer.Option(
            help="Number of times to repeat each sample.", envvar="COMPLAI_EPOCHS"
        ),
    ] = 1,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            help="Maximum number of tokens that can be generated in a model completion.",
            envvar="COMPLAI_MAX_TOKENS",
        ),
    ] = 4096,
    temperature: Annotated[
        float | None,
        typer.Option(help="Model temperature.", envvar="COMPLAI_TEMPERATURE"),
    ] = None,
    top_p: Annotated[
        float | None, typer.Option(help="Model top-p.", envvar="COMPLAI_TOP_P")
    ] = None,
    top_k: Annotated[
        int | None,
        typer.Option(
            help="Randomly sample the next word from the top_k most likely next words. Anthropic, Google, HuggingFace, and vLLM only.",
            envvar="COMPLAI_TOP_K",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            help="Random seed. OpenAI, Google, Groq, Mistral, HuggingFace, and vLLM only.",
            envvar="COMPLAI_SEED",
        ),
    ] = None,
    reasoning_effort: Annotated[
        ReasoningEffortLevel | None,
        typer.Option(
            help="Constrains effort on reasoning for reasoning models. Open AI o-series and gpt-5 models only.",
            envvar="COMPLAI_REASONING_EFFORT",
        ),
    ] = None,
    reasoning_tokens: Annotated[
        int | None,
        typer.Option(
            help="Maximum number of tokens to use for reasoning. Anthropic Claude models only.",
            envvar="COMPLAI_REASONING_TOKENS",
        ),
    ] = None,
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
            help="Maximum number of sandboxes (per-provider) to run in parallel.",
            envvar="COMPLAI_MAX_SANDBOXES",
        ),
    ] = None,
    max_tasks: Annotated[
        int | None,
        typer.Option(
            help="Maximum number of tasks to run in parallel.",
            envvar="COMPLAI_MAX_TASKS",
        ),
    ] = 2,
    log_level: Annotated[
        LoggingLevel,
        typer.Option(
            help="Python logger level for console.", envvar="COMPLAI_LOG_LEVEL"
        ),
    ] = LoggingLevel.WARNING,
    log_dir: Annotated[
        str | None,
        typer.Option(help="Directory to save logs to.", envvar="COMPLAI_LOG_DIR"),
    ] = None,
    log_samples: Annotated[
        bool, typer.Option(help="Log sample details", envvar="COMPLAI_LOG_SAMPLES")
    ] = True,
    log_buffer: Annotated[
        int | None,
        typer.Option(
            help="Number of samples to buffer before writing log file.",
            envvar="COMPLAI_LOG_BUFFER",
        ),
    ] = 1000,
    log_format: Annotated[
        LogFormat,
        typer.Option(help="Format for writing log files.", envvar="COMPLAI_LOG_FORMAT"),
    ] = LogFormat.EVAL,
    fail_on_error: Annotated[
        str,
        typer.Option(
            help="If True, fail tasks on first sample error; If False, never fail tasks on sample errors; Value between 0 and 1 to fail tasks if a proportion of total samples fails. Value greater than 1 to fail tasks if a count of samples fails.",
            envvar="COMPLAI_FAIL_ON_ERROR",
            parser=bool_or_float,
        ),
    ] = "True",
    continue_on_fail: Annotated[
        bool,
        typer.Option(
            help="If False, fail tasks immediately when the fail_on_error condition is met. If True, continue running the failing task and only fail at the end of the task if the fail_on_error condition is met.",
            envvar="COMPLAI_CONTINUE_ON_FAIL",
        ),
    ] = False,
    retry_on_error: Annotated[
        int,
        typer.Option(
            help="Number of times to retry samples if they encounter errors.",
            envvar="COMPLAI_RETRY_ON_ERROR",
        ),
    ] = 0,
    debug_errors: Annotated[
        bool,
        typer.Option(
            help="Raise task errors (rather than logging them) so they can be debugged.",
            envvar="COMPLAI_DEBUG_ERRORS",
        ),
    ] = False,
    message_limit: Annotated[
        int | None,
        typer.Option(
            help="Limit on total messages used for each sample.",
            envvar="COMPLAI_MESSAGE_LIMIT",
        ),
    ] = None,
    token_limit: Annotated[
        int | None,
        typer.Option(
            help="Limit on total tokens used for each sample.",
            envvar="COMPLAI_TOKEN_LIMIT",
        ),
    ] = None,
    time_limit: Annotated[
        int | None,
        typer.Option(
            help="Limit on total running time for each sample.",
            envvar="COMPLAI_TIME_LIMIT",
        ),
    ] = None,
    working_limit: Annotated[
        int | None,
        typer.Option(
            help="Limit on total working time (model generation, tool calls, etc.) for each sample.",
            envvar="COMPLAI_WORKING_LIMIT",
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug", help="Enable full stack traces.", envvar="COMPLAI_DEBUG"
        ),
    ] = False,
) -> None:
    """Run tasks."""
    # Get TaskInfo objects from task names
    task_infos: list[TaskInfo] = get_task_infos(tasks, tasks_to_skip)

    # Apply display monkey patch
    patch_display_results()

    # Parse args
    parsed_task_args = parse_cli_config(task_args, task_config)
    parsed_model_args = parse_cli_config(model_args, model_config)
    parsed_limit = parse_samples_limit(limit)
    parsed_sample_id = parse_sample_id(sample_id)

    # Define log directory
    if log_dir is None:
        log_dir = get_log_dir(model)

    try:
        print("Starting evals...")
        eval_set(
            model=model,
            tasks=task_infos,
            log_dir=log_dir,
            retry_attempts=retry_attempts,
            retry_wait=retry_wait,
            retry_connections=retry_connections,
            retry_cleanup=retry_cleanup,
            task_args=parsed_task_args,
            model_base_url=model_base_url,
            model_args=parsed_model_args,
            limit=parsed_limit,
            sample_id=parsed_sample_id,
            sample_shuffle=sample_shuffle,
            epochs=epochs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            reasoning_effort=reasoning_effort.value if reasoning_effort else None,
            reasoning_tokens=reasoning_tokens,
            max_connections=max_connections,
            max_samples=max_samples,
            max_subprocesses=max_subprocesses,
            max_sandboxes=max_sandboxes,
            max_tasks=max_tasks,
            log_level=log_level.value if log_level else None,
            log_samples=log_samples,
            log_buffer=log_buffer,
            log_format=log_format.value if log_format else None,
            fail_on_error=cast(bool | float, fail_on_error),
            continue_on_fail=continue_on_fail,
            retry_on_error=retry_on_error,
            debug_errors=debug_errors,
            message_limit=message_limit,
            token_limit=token_limit,
            time_limit=time_limit,
            working_limit=working_limit,
        )
    except Exception as e:
        if debug:
            raise  # Show full stack trace
        else:
            import sys

            error_msg = (
                f"\n[bold][bright_red]{e.__class__.__name__}:[/bright_red][/bold] {e}"
                "\n\nRun with [bold][cyan]--debug[/cyan][/bold] flag for full stack trace."
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)
