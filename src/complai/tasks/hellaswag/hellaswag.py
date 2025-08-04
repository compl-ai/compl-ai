from inspect_ai import Task
from inspect_ai import task
from inspect_evals.hellaswag import hellaswag


@task(
    name="hellaswag", technical_requirement="Capabilities, Performance, and Limitations"
)
def decorated_hellaswag() -> Task:
    return hellaswag()
