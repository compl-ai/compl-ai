import os
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Literal

import httpx
import typer
from inspect_ai._cli.util import parse_cli_config
from inspect_ai.model import get_model
from rich import print
from rich.table import Table

from complai._cli.utils import error_handler
from complai._cli.utils import get_task_infos
from complai._cli.utils import parse_task_args
from complai._cli.utils import parse_tasks
from complai._cli.utils import select_subset_items


# Judge (grader) model arguments of each task and their defaults. Kept in sync with
# the task signatures by a test.
JUDGE_DEFAULTS: dict[str, dict[str, str]] = {
    "cab": {"judge_model": "openai/gpt-5-mini-2025-08-07"},
    "hle": {"grader_model": "openai/o3-mini-2025-01-31"},
    "mask": {
        "binary_judge_model": "openai/gpt-4o",
        "numeric_judge_model": "openai/o3-mini",
    },
    "self_check_consistency": {
        "argumentation_model": "openai/gpt-5-mini",
        "judge_model": "openai/gpt-5-mini",
    },
    "simpleqa_verified": {"grader_model": "openai/gpt-4.1-2025-04-14"},
    "strong_reject": {"judge": "openai/gpt-4o-mini"},
}

# OpenRouter equivalents of the default judges, used when only OPENROUTER_API_KEY
# is set. OpenRouter has no dated snapshots, so dated models map to their alias.
OPENROUTER_JUDGES: dict[str, str] = {
    "openai/gpt-4.1-2025-04-14": "openrouter/openai/gpt-4.1",
    "openai/o3-mini-2025-01-31": "openrouter/openai/o3-mini",
    "openai/o3-mini": "openrouter/openai/o3-mini",
    "openai/gpt-5-mini-2025-08-07": "openrouter/openai/gpt-5-mini",
    "openai/gpt-5-mini": "openrouter/openai/gpt-5-mini",
    "openai/gpt-4o": "openrouter/openai/gpt-4o",
    "openai/gpt-4o-mini": "openrouter/openai/gpt-4o-mini",
}

# Gated Hugging Face datasets: access must be requested on the dataset page.
GATED_DATASETS: dict[str, str] = {
    "hle": "cais/hle",
    "strong_reject": "compl-ai/strong_reject",
}

# Providers that start a local server unless a base URL is given.
SERVER_PROVIDERS = ("vllm", "sglang")
# Providers that run in-process and need no credentials.
LOCAL_PROVIDERS = ("hf", "mockllm", "none")


@dataclass
class CheckResult:
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str


def _one_line(exc: Exception) -> str:
    return " ".join(str(exc).replace("ERROR:", "").split())


def apply_judge_fallback(
    task_names: list[str], task_args: dict[str, dict[str, Any]]
) -> list[str]:
    """Switch default OpenAI judges to OpenRouter when only OpenRouter has a key.

    Judges set by the user in ``task_args`` are left alone. Updates ``task_args``
    in place and returns the switched judges as ``task:arg`` names.
    """
    if os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        return []
    switched = []
    for task in task_names:
        for arg, default in JUDGE_DEFAULTS.get(task, {}).items():
            if arg not in task_args.get(task, {}):
                task_args.setdefault(task, {})[arg] = OPENROUTER_JUDGES[default]
                switched.append(f"{task}:{arg}")
    return switched


def judge_models(
    task_names: list[str], task_args: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    """Map each judge model used by the tasks to the ``task:arg`` names using it."""
    judges: dict[str, list[str]] = {}
    for task in task_names:
        for arg, default in JUDGE_DEFAULTS.get(task, {}).items():
            model = task_args.get(task, {}).get(arg, default)
            judges.setdefault(model, []).append(f"{task}:{arg}")
    return judges


def _server_base_url(provider: str, model_base_url: str | None) -> str | None:
    return model_base_url or os.getenv(f"{provider.upper()}_BASE_URL")


def check_target_model(
    model: str, model_base_url: str | None, model_args: dict[str, Any]
) -> CheckResult:
    """Check the evaluated model can be reached, without generating anything."""
    provider, _, name = model.partition("/")
    if provider in LOCAL_PROVIDERS:
        return CheckResult(f"model {model}", "ok", "runs locally")
    if provider in SERVER_PROVIDERS:
        base_url = _server_base_url(provider, model_base_url)
        if base_url is None:
            return CheckResult(
                f"model {model}",
                "ok",
                f"a local {provider} server will be started "
                f"(set {provider.upper()}_BASE_URL to use a running server)",
            )
        return _check_server(model, provider, name, base_url)
    try:
        get_model(model, base_url=model_base_url, memoize=False, **model_args)
    except Exception as exc:  # noqa: BLE001 - any failure fails the check
        return CheckResult(f"model {model}", "fail", _one_line(exc))
    return CheckResult(f"model {model}", "ok", "credentials found")


def _check_server(model: str, provider: str, name: str, base_url: str) -> CheckResult:
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = base_url.rstrip("/") + "/models"
    try:
        response = httpx.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        served = [entry["id"] for entry in response.json().get("data", [])]
    except Exception as exc:  # noqa: BLE001 - any failure fails the check
        return CheckResult(
            f"model {model}", "fail", f"{url} unreachable: {_one_line(exc)}"
        )
    if name != "local" and name not in served:
        return CheckResult(
            f"model {model}",
            "fail",
            f"{base_url} serves {', '.join(served) or 'no models'}, not {name}",
        )
    return CheckResult(f"model {model}", "ok", f"served at {base_url}")


def check_judges(
    task_names: list[str], task_args: dict[str, dict[str, Any]]
) -> list[CheckResult]:
    """Check every judge model the tasks will call has credentials."""
    results = []
    for judge, used_by in judge_models(task_names, task_args).items():
        try:
            get_model(judge, memoize=False)
        except Exception as exc:  # noqa: BLE001 - any failure fails the check
            results.append(CheckResult(f"judge {judge}", "fail", _one_line(exc)))
        else:
            results.append(
                CheckResult(f"judge {judge}", "ok", f"for {', '.join(used_by)}")
            )
    return results


def check_gated_datasets(task_names: list[str]) -> list[CheckResult]:
    """Check access to gated Hugging Face datasets the tasks load."""
    from huggingface_hub import auth_check

    results = []
    for task in task_names:
        dataset = GATED_DATASETS.get(task)
        if dataset is None:
            continue
        try:
            auth_check(dataset, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001 - any failure fails the check
            results.append(
                CheckResult(
                    f"dataset {dataset}",
                    "fail",
                    f"no access ({exc.__class__.__name__}). Request access on "
                    f"https://huggingface.co/datasets/{dataset} and set HF_TOKEN.",
                )
            )
        else:
            results.append(CheckResult(f"dataset {dataset}", "ok", "access granted"))
    return results


def check_strong_reject_prompts(args: dict[str, Any]) -> CheckResult:
    """Check whether full-dataset strong_reject jailbreak prompts must be generated.

    The small dataset uses the gated reference prompts (see GATED_DATASETS); the
    full dataset has none, so its prompts come from the local cache or are generated.
    """
    from complai.tasks.strong_reject.jailbreaks import get_method_cache_path
    from complai.tasks.strong_reject.strong_reject import parse_jailbreak_methods

    methods = parse_jailbreak_methods(args.get("jailbreak_methods", "all"))
    missing = [m for m in methods if not get_method_cache_path(m, True).exists()]
    if not missing:
        return CheckResult("strong_reject prompts", "ok", "all methods cached")
    detail = (
        f"{len(missing)} of {len(methods)} jailbreak methods are not cached and "
        "will be generated with OpenAI models when the task loads; generated "
        "prompts differ between machines"
    )
    if not os.getenv("OPENAI_API_KEY"):
        return CheckResult(
            "strong_reject prompts", "fail", detail + ". Set OPENAI_API_KEY."
        )
    return CheckResult("strong_reject prompts", "warn", detail)


def run_checks(
    model: str,
    model_base_url: str | None,
    model_args: dict[str, Any],
    task_names: list[str],
    task_args: dict[str, dict[str, Any]],
) -> list[CheckResult]:
    """Check everything the tasks need to run and be graded, without generating."""
    results = [check_target_model(model, model_base_url, model_args)]
    results += check_judges(task_names, task_args)
    strong_reject_args = task_args.get("strong_reject", {})
    if bool(strong_reject_args.get("full", False)):
        results += check_gated_datasets([t for t in task_names if t != "strong_reject"])
        if "strong_reject" in task_names:
            results.append(check_strong_reject_prompts(strong_reject_args))
    else:
        results += check_gated_datasets(task_names)
    return results


def print_checks(results: list[CheckResult]) -> bool:
    """Print the check results and return whether none failed."""
    style = {"ok": "green", "warn": "yellow", "fail": "bold red"}
    table = Table(title="Pre-run checks", title_justify="left")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for result in results:
        table.add_row(
            result.name,
            f"[{style[result.status]}]{result.status}[/{style[result.status]}]",
            result.detail,
        )
    print(table)
    return all(result.status != "fail" for result in results)


def status_command(
    model: Annotated[
        str,
        typer.Argument(
            help="Model to check, as passed to `complai eval`.", envvar="COMPLAI_MODEL"
        ),
    ],
    tasks: Annotated[
        str | None,
        typer.Option(
            "--tasks",
            "-t",
            help="Comma-separated tasks. Defaults to the core subset tasks (all tasks with --full).",
        ),
    ] = None,
    full: Annotated[
        bool, typer.Option("--full", help="Check for the full COMPL-AI suite.")
    ] = False,
    task_args: Annotated[
        list[str] | None,
        typer.Option("-T", help="Task arguments, as for `complai eval`."),
    ] = None,
    task_config: Annotated[
        str | None,
        typer.Option("--task-config", help="Task config file, as for `complai eval`."),
    ] = None,
    model_base_url: Annotated[
        str | None, typer.Option("--model-base-url", help="Base URL of the model.")
    ] = None,
    model_args: Annotated[
        list[str] | None,
        typer.Option("-M", help="Model arguments, as for `complai eval`."),
    ] = None,
    model_config: Annotated[
        str | None, typer.Option("--model-config", help="Model config file.")
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable full stack traces.")
    ] = False,
) -> None:
    """Check credentials, servers and datasets needed to run and grade an eval.

    Generates nothing, so it costs no tokens. `complai eval` runs the same checks.
    """
    with error_handler(debug):
        if full:
            task_names = [info.name for info in get_task_infos(tasks, None, None)]
        else:
            task_names = list(select_subset_items(parse_tasks(tasks), [], None, {})[0])
        parsed_task_args = parse_task_args(task_args or [], task_config)
        for switched in apply_judge_fallback(task_names, parsed_task_args):
            print(f"Judge {switched} uses OpenRouter (no OPENAI_API_KEY set)")
        parsed_model_args = parse_cli_config(model_args or [], model_config)
        passed = print_checks(
            run_checks(
                model, model_base_url, parsed_model_args, task_names, parsed_task_args
            )
        )
    if not passed:
        raise typer.Exit(1)
