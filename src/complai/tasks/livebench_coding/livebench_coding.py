from datetime import datetime

from inspect_ai import Task
from inspect_ai import task
from inspect_evals.livebench import livebench


@task
def livebench_coding() -> Task:
    return livebench(livebench_release_date=datetime(2025, 1, 1), category="coding")
