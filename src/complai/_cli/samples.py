import json
from pathlib import Path
from typing import Any

import typer
from inspect_ai import Task
from inspect_ai import TaskInfo
from inspect_ai._eval.loader import load_task_spec
from inspect_ai.model import ChatMessage
from typing_extensions import Annotated

from complai._cli.utils import error_handler
from complai._cli.utils import get_task_infos
from complai._cli.utils import instantiate_tasks_from_infos
from complai._cli.utils import parse_samples_limit
from complai._cli.utils import parse_task_args


def _input_json(input: str | list[ChatMessage]) -> str | list[dict[str, Any]]:
    if isinstance(input, str):
        return input
    return [
        message.model_dump(mode="json", exclude_none=True, exclude={"id"})
        for message in input
    ]


def collect_task_samples(
    task_name: str, task: Task, limit: int | tuple[int, int] | None = None
) -> list[dict[str, Any]]:
    start, stop = (0, limit) if isinstance(limit, int) else limit or (0, None)
    return [
        {
            "task": task_name,
            "sample_id": sample.id if sample.id is not None else index + 1,
            "input": _input_json(sample.input),
        }
        for index, sample in enumerate(task.dataset.samples)
        if index >= start and (stop is None or index < stop)
    ]


def instantiate_task_specs(
    specs: list[str], task_args: dict[str, dict[str, Any]]
) -> list[tuple[str, Task]]:
    tasks = []
    for spec in specs:
        short_name = spec.rsplit("/", 1)[-1].split("@", 1)[-1]
        for task in load_task_spec(
            spec, task_args.get(spec, task_args.get(short_name, {}))
        ):
            tasks.append((task.name or short_name, task))
    return tasks


def samples_command(
    output: Annotated[Path, typer.Argument(help="JSONL file to write.")],
    tasks: Annotated[
        str | None,
        typer.Option(
            "-t",
            "--tasks",
            help="Comma-separated list of tasks. Defaults to all tasks.",
        ),
    ] = None,
    tasks_to_skip: Annotated[
        str | None,
        typer.Option("-s", "--skip", help="Comma-separated list of tasks to skip."),
    ] = None,
    task_spec: Annotated[
        list[str],
        typer.Option(
            "--task-spec",
            help="Inspect task spec, e.g. inspect_evals/arc_challenge. Repeatable.",
        ),
    ] = [],
    task_args: Annotated[
        list[str],
        typer.Option(
            "-T", "--task-arg", help="Task argument in task_name:key=value format."
        ),
    ] = [],
    task_config: Annotated[
        str | None,
        typer.Option(
            "-tc", "--task-config", help="Task arguments file (JSON or YAML)."
        ),
    ] = None,
    limit: Annotated[
        str | None,
        typer.Option("-l", "--limit", help="Samples per task, e.g. 10 or 10-20."),
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Write samples as JSONL."""
    task_infos: list[TaskInfo] = (
        get_task_infos(tasks, tasks_to_skip, None)
        if tasks is not None or not task_spec
        else []
    )

    with error_handler(debug):
        parsed_task_args = parse_task_args(task_args, task_config)
        loaded_tasks = (
            instantiate_tasks_from_infos(task_infos, parsed_task_args)
            if task_infos
            else []
        )
        named_tasks = list(zip((info.name for info in task_infos), loaded_tasks))
        named_tasks += instantiate_task_specs(task_spec, parsed_task_args)
        parsed_limit = parse_samples_limit(limit)
        with output.open("w") as file:
            for task_name, task in named_tasks:
                for record in collect_task_samples(task_name, task, parsed_limit):
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {output}")
