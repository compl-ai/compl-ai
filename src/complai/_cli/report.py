import math
import re
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Annotated
from typing import Any

import typer
from inspect_ai.log import EvalLog
from inspect_ai.log import EvalSampleSummary
from inspect_ai.log import list_eval_logs
from inspect_ai.log import read_eval_log
from inspect_ai.log import read_eval_log_sample
from inspect_ai.log import read_eval_log_sample_summaries
from rich import print
from rich.table import Table

from complai._cli.utils import default_subset_dir
from complai._cli.utils import error_handler
from complai._cli.utils import read_eval_subset
from complai.utils.log_parser import logical_sample_id
from complai.utils.log_parser import SMOKE_METADATA_KEY
from complai.utils.log_parser import SUBSET_METADATA_KEY
from complai.utils.scores import load_config
from complai.utils.scores import metric_source
from complai.utils.scores import primary_metric


# Stop reasons meaning the answer was cut off, e.g. reasoning used up the budget.
TRUNCATED_STOP_REASONS = ("max_tokens", "model_length")


@dataclass
class TaskReport:
    model: str
    task: str
    run: str
    status: str
    log_error: str | None
    total: int
    completed: int
    errors: Counter[str] = field(default_factory=Counter)
    limits: Counter[str] = field(default_factory=Counter)
    truncated: int = 0
    unscored: int = 0
    metric: str = ""
    tokens: str = ""
    covered: int | None = None
    subset_items: int | None = None

    @property
    def clean(self) -> bool:
        return (
            self.status == "success"
            and self.completed == self.total
            and not self.errors
            and not self.limits
            and not self.truncated
            and not self.unscored
        )


def _error_text(message: str) -> str:
    """Shorten an error message so that repeats of the same error group together."""
    lines = message.strip().splitlines()
    text = lines[-1] if lines else "(no message)"
    text = re.sub(r"\b(0x)?[0-9a-f]{8,}\b", "<id>", text.split(" by ")[0])
    return text[:160]


def _may_be_truncated(
    sample: EvalSampleSummary, model: str, max_tokens: int | None
) -> bool:
    """Whether a sample used enough output tokens to have hit max_tokens.

    Reading full samples is slow (some tasks store test cases in each sample), so
    only samples that may have hit the limit are read to check their stop reason.
    """
    if max_tokens is None:
        return True
    usage = sample.model_usage.get(model)
    return usage is not None and usage.output_tokens >= max_tokens


def _truncated(log_path: str, summary: EvalSampleSummary) -> bool:
    sample = read_eval_log_sample(
        log_path,
        id=summary.id,
        epoch=summary.epoch,
        exclude_fields={"events", "messages", "store", "attachments", "metadata"},
    )
    return bool(sample.output.choices) and (
        sample.output.stop_reason in TRUNCATED_STOP_REASONS
    )


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, dict):
        return any(_missing(item) for item in value.values())
    if isinstance(value, list):
        return any(_missing(item) for item in value)
    return False


def _main_metric(log: EvalLog, task: str, config: dict[str, dict[str, Any]]) -> str:
    """Show the main metric of the scorer that counts for the task (from config.json)."""
    if log.results is None or not log.results.scores:
        return ""
    scores = {score.name: score for score in log.results.scores}
    task_config = config.get(task, {})
    source = metric_source(scores, task_config.get("scorer", ""))
    score = scores[source] if source else log.results.scores[0]
    values = {name: metric.value for name, metric in score.metrics.items()}
    picked = primary_metric(values, task_config.get("primary_metric", []))
    if picked is None:
        return ""
    name, value = picked
    return f"{name}={value:.3f}"


def _tokens(log: EvalLog) -> str:
    usage = log.stats.model_usage.get(log.eval.model)
    if usage is None:
        return ""
    text = f"{_short(usage.input_tokens)}/{_short(usage.output_tokens)}"
    if usage.reasoning_tokens:
        text += f" ({_short(usage.reasoning_tokens)} reasoning)"
    return text


def task_report(
    log_path: str,
    subset: dict[str, list[dict[str, Any]]],
    config: dict[str, dict[str, Any]],
) -> TaskReport:
    """Summarise one eval log: completion, problems, main metric and coverage."""
    log = read_eval_log(log_path, header_only=True)
    task = log.eval.task.split("/")[-1]
    metadata = log.eval.metadata or {}
    run = (
        "smoke"
        if SMOKE_METADATA_KEY in metadata
        else "subset"
        if SUBSET_METADATA_KEY in metadata
        else "full"
    )
    report = TaskReport(
        model=log.eval.model,
        task=task,
        run=run,
        status=log.status,
        log_error=_error_text(log.error.message) if log.error else None,
        total=log.results.total_samples if log.results else 0,
        completed=log.results.completed_samples if log.results else 0,
        metric=_main_metric(log, task, config),
        tokens=_tokens(log),
    )
    max_tokens = log.eval.model_generate_config.max_tokens
    scored_ids = set()
    for sample in read_eval_log_sample_summaries(log_path):
        if sample.error is not None:
            report.errors[_error_text(sample.error)] += 1
        if sample.limit is not None:
            report.limits[sample.limit] += 1
        if _may_be_truncated(sample, log.eval.model, max_tokens) and _truncated(
            log_path, sample
        ):
            report.truncated += 1
        if not sample.scores or any(
            _missing(score.value) for score in sample.scores.values()
        ):
            report.unscored += 1
        elif task in subset:
            scored_ids.add(
                logical_sample_id(task, sample.id, sample.metadata, sample.input)
            )
    if task in subset:
        subset_ids = {row["sample_id"] for row in subset[task]}
        report.covered = len(scored_ids & subset_ids)
        report.subset_items = len(subset_ids)
    return report


def latest_logs(log_dir: Path) -> list[str]:
    """Return the latest log of each (model, task) in a directory."""
    latest: dict[tuple[str, str], tuple[str, str]] = {}
    for info in list_eval_logs(str(log_dir)):
        header = read_eval_log(info, header_only=True)
        key = (header.eval.model, header.eval.task)
        if key not in latest or header.eval.created > latest[key][0]:
            latest[key] = (header.eval.created, info.name)
    return [name for _, name in sorted(latest.values(), key=lambda item: item[1])]


def report_log_dir(log_dir: Path) -> None:
    """Print the report for the latest log of each (model, task) in a directory."""
    subset = read_eval_subset(default_subset_dir() / "subset.jsonl")
    config = load_config()
    reports = [task_report(path, subset, config) for path in latest_logs(log_dir)]
    if not reports:
        raise ValueError(f"No eval logs found in {log_dir}")
    reports.sort(key=lambda report: (report.model, report.task))
    print_report(reports, list(subset))


def _problem_counts(report: TaskReport) -> str:
    counts = {
        "error": sum(report.errors.values()),
        "limit": sum(report.limits.values()),
        "cut off": report.truncated,
        "unscored": report.unscored,
    }
    return ", ".join(f"{count} {name}" for name, count in counts.items() if count)


def _short(count: int) -> str:
    for size, suffix in ((1_000_000, "M"), (1_000, "k")):
        if count >= size:
            return f"{count / size:.1f}{suffix}"
    return str(count)


def print_report(reports: list[TaskReport], subset_tasks: list[str]) -> None:
    """Print a per-task table, then the problems found in each task."""
    for model in sorted({report.model for report in reports}):
        rows = [report for report in reports if report.model == model]
        runs = sorted({report.run for report in rows})
        table = Table(
            title=f"Eval report: {model} ({', '.join(runs)} run)", title_justify="left"
        )
        table.add_column("Task", no_wrap=True)
        for column in ("Status", "Samples", "Problems", "Coverage"):
            table.add_column(column)
        table.add_column("Metric", overflow="fold")
        table.add_column("Tokens in/out", justify="right")
        for report in rows:
            status_style = "green" if report.clean else "bold red"
            table.add_row(
                report.task if len(runs) == 1 else f"{report.task} ({report.run})",
                f"[{status_style}]{report.status}[/{status_style}]",
                f"{report.completed}/{report.total}",
                _problem_counts(report),
                f"{report.covered}/{report.subset_items}"
                if report.subset_items
                else "",
                report.metric,
                report.tokens,
            )
        print(table)

        if "subset" in runs:
            not_run = sorted(set(subset_tasks) - {report.task for report in rows})
            if not_run:
                print(
                    f"[yellow]Subset tasks not run yet: {', '.join(not_run)}[/yellow]"
                )
        problems = [report for report in rows if not report.clean]
        if not problems:
            print(f"[green]All {len(rows)} tasks completed cleanly.[/green]")
            continue
        print(
            f"[bold red]{len(problems)} of {len(rows)} tasks have problems:[/bold red]"
        )
        for report in problems:
            print(f"[bold]{report.task}[/bold]")
            if report.log_error:
                print(f"  run failed: {report.log_error}")
            for message, count in report.errors.most_common(5):
                print(f"  {count} sample error(s): {message}")
            for limit, count in report.limits.items():
                print(f"  {count} sample(s) hit the {limit} limit")
            if report.truncated:
                print(
                    f"  {report.truncated} answer(s) cut off at the token limit "
                    "(raise --max-tokens, or lower --reasoning-tokens)"
                )
            if report.unscored:
                print(
                    f"  {report.unscored} sample(s) have no score "
                    "(check the judge model and its API key)"
                )


def report_command(
    log_dir: Annotated[
        Path, typer.Argument(help="Directory with the eval logs of a run.")
    ],
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Report per-task status, errors, coverage and metrics for an eval run."""
    with error_handler(debug):
        report_log_dir(log_dir)
