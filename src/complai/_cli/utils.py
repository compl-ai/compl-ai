import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import Callable

import typer
import yaml
from inspect_ai import list_tasks
from inspect_ai import TaskInfo
from inspect_ai._eval.loader import load_task_spec
from inspect_ai._eval.task import Task
from inspect_ai._util.config import resolve_args
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
    task_names: list[str], tasks_to_skip: list[str], task_filter: dict[str, str] | None
) -> list[TaskInfo]:
    """
    Get TaskInfo objects from a list of task names.

    Args:
        tasks_names (list[str]): List of task names to retrieve TaskInfo for. If empty,
            all available tasks are returned.
        tasks_to_skip (list[str]): List of task names to skip.
        task_filter (dict[str, str] | None): Filter tasks by attribute (e.g. `-f technical_requirement='Capabilities, Performance, and Limitations'`).

    Raises:
        typer.BadParameter: If one of the task names is not valid.

    Returns:
        list[TaskInfo]: List of TaskInfo objects for the specified tasks.
    """

    def filter(task: TaskInfo) -> bool:
        if task.name in tasks_to_skip:
            tasks_to_skip.remove(task.name)
            return False

        if task_filter and not any(
            task.attribs.get(key) == value for key, value in task_filter.items()
        ):
            return False

        return not task_names or task.name in task_names

    tasks_to_run = get_complai_tasks(filter=filter)

    # Put human_eval first if it is in the list
    # https://github.com/UKGovernmentBEIS/inspect_ai/issues/2446
    humaneval = next((task for task in tasks_to_run if task.name == "humaneval"), None)
    if humaneval:
        tasks_to_run.remove(humaneval)
        tasks_to_run.insert(0, humaneval)

    # Warn about tasks that were not skipped
    if tasks_to_skip:
        print(
            f"[yellow][WARNING][/yellow] The following task(s) were not skipped: {', '.join(tasks_to_skip)}"
        )

    # Raise error if tasks were not found or skipped
    not_found = set(task_names) - set(task.name for task in tasks_to_run)
    if not_found:
        raise typer.BadParameter(
            f"The following task(s) were skipped or not found: {', '.join(not_found)}"
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


def get_task_infos(
    tasks: str | None, tasks_to_skip: str | None, task_filter: dict[str, str] | None
) -> list[TaskInfo]:
    """
    Get TaskInfo objects for the specified tasks.

    Args:
        tasks (str | None): Comma-separated list of task names. If None, all available tasks are returned.
        tasks_to_skip (str | None): Comma-separated list of task names to skip. If None, no tasks are skipped.
        task_filter (dict[str, str] | None): Filter tasks by attribute (e.g. `-f technical_requirement='Capabilities, Performance, and Limitations'`).

    Returns:
        list[TaskInfo]: List of TaskInfo objects for the specified tasks.
    """
    task_names = parse_tasks(tasks)
    tasks_to_skip_names = parse_tasks(tasks_to_skip)

    return get_task_infos_from_task_names(task_names, tasks_to_skip_names, task_filter)


def parse_task_args(
    task_args: list[str], task_config: str | None
) -> dict[str, dict[str, Any]]:
    """
    Parse per-task arguments from CLI and config file. CLI args take precedence over config file.

    Args:
        task_args: List of CLI task args in format "task_name:key=value"
        task_config: Path to config file (JSON or YAML)

    Returns:
        Dict mapping task_name -> args dict
    """
    result: dict[str, dict[str, Any]] = {}  # task_name -> {key: value}

    # Parse config file
    if task_config is not None:
        config = resolve_args(task_config)
        if any(not isinstance(value, dict) for value in config.values()):
            raise typer.BadParameter(
                "Task config file must specify a mapping of task names to argument dictionaries. See `config/default_config.yaml` for an example of a valid config file."
            )
        result = config

    # Parse CLI args (override config file)
    for arg in task_args:
        if ":" not in arg or "=" not in arg:
            raise typer.BadParameter(
                f"Task argument must be in format 'task_name:key=value', got '{arg}'"
            )

        # Split on first colon to get task_name and key=value part
        task_name, key_value = arg.split(":", 1)

        # Split on first equals to get key and value
        key, value_str = key_value.split("=", 1)

        # Parse value
        try:
            value = yaml.safe_load(value_str)
        except yaml.YAMLError as e:
            raise typer.BadParameter(
                f"Could not parse value '{value_str}' in argument '{arg}': {e}"
            )

        # Handle comma-separated values
        if isinstance(value, str):
            value_list = value.split(",")
            value = value_list if len(value_list) > 1 else value_list[0]

        # Initialize task dict if not present
        if task_name not in result:
            result[task_name] = {}

        # Set the value (CLI overrides config)
        result[task_name][key] = value

    return result


def instantiate_task_from_info(task_info: TaskInfo, task_args: dict = {}) -> Task:
    """
    Instantiate a Task from a TaskInfo object and task args.

    Args:
        task_info: TaskInfo object.
        task_args: Dictionary mapping task names to their specific args.

    Returns:
        Task: Instantiated Task object.
    """
    task_spec = f"{task_info.file}@{task_info.name}"

    return load_task_spec(task_spec, task_args)[0]


def instantiate_tasks_from_infos(
    task_infos: list[TaskInfo], task_args: dict[str, dict[str, Any]] = {}
) -> list[Task]:
    """
    Instantiate tasks from TaskInfo objects with per-task args.

    Args:
        task_infos: List of TaskInfo objects.
        task_args: Dictionary mapping task names to their specific args.

    Returns:
        list[Task]: List of instantiated Task objects.
    """
    tasks = []

    # Warn about args for tasks that aren't in the list of tasks to run
    tasks_to_run = {info.name for info in task_infos}
    extra_task_args = set(task_args.keys()) - tasks_to_run
    if extra_task_args:
        print(
            f"[yellow][WARNING][/yellow] Task arguments provided for tasks not being run: {', '.join(extra_task_args)}"
        )

    # Instantiate tasks
    print(f"Loading {len(task_infos)} task{'' if len(task_infos) == 1 else 's'}...")
    start_time = perf_counter()

    for i, info in enumerate(task_infos):
        task_start = perf_counter()
        args = task_args.get(info.name, {})
        task = instantiate_task_from_info(info, args)
        tasks.append(task)
        print(
            f"  ✓ {info.name}: {perf_counter() - task_start:.2f}s ({i + 1}/{len(task_infos)})"
        )

    total_time = perf_counter() - start_time
    print(f"Total loading time: {total_time:.2f}s")

    return tasks


def validate_model_args(
    model: str, task_infos: list[TaskInfo], model_args: dict[str, Any]
) -> None:
    """Validate model arguments."""
    if "swe_bench_verified" in {task.name for task in task_infos} and (
        model.startswith("vllm/") or model.startswith("sglang/")
    ):
        if "tool_call_parser" not in model_args:
            raise typer.BadParameter(
                "Tool call parser is required when running SWE-Bench with vLLM/SGLang backend. "
                "If running your own server, specify the tool call parser in your server launch command and provide a dummy value to complai. "
                "If not running your own server, specify the tool call parser as a model argument to complai: "
                "'-M tool-call-parser=<parser>'"
            )
        if model.startswith("vllm/"):
            model_args["enable_auto_tool_choice"] = None


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


def get_log_dir(model: str, base_log_dir: str | None = "./logs/") -> str:
    """
    Get the log directory for the model. If no log directory is provided,
    use model and timestamp to create a unique log directory name.
    """
    model_name = "_".join(model.split("/")[-2:])
    timestamp = (
        datetime.now().astimezone().isoformat(timespec="seconds").replace(":", "-")
    )

    return f"{base_log_dir}/{model_name}_{timestamp}"


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


@contextmanager
def error_handler(debug: bool) -> Iterator:
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        if debug:
            raise
        else:
            error_msg = (
                f"\n[bold][bright_red]{e.__class__.__name__}:[/bright_red][/bold] {e}"
                "\n\nRun with [bold][cyan]--debug[/cyan][/bold] flag for full stack trace."
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)
