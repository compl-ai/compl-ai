import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .problem_yaml import Problem, Result, ResultList, TestResults

# def eval_script(problem, index):
#     program = problem.prompt + problem.completions[index] + '\n' + problem.tests
#     eval_string_script(problem.language, program)


def eval_string_script(language, program):
    eval_script, file_ext = eval_script_python, ".py"
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=True) as f:
        f.write(program.encode("utf-8"))
        f.flush()
        result = eval_script(Path(f.name))
        # Only save the first 2K of output from the running program. Any futher
        # output is very likely an exceptionally long stack trace or a long
        # series of prints.
        if type(result["stdout"]) == bytes:
            result["stdout"] = result["stdout"].decode("utf-8", errors="ignore")
        if result["stdout"] is None:
            result["stdout"] = ""
        if result["stderr"] is None:
            result["stderr"] = ""
        if type(result["stderr"]) == bytes:
            result["stderr"] = result["stderr"].decode("utf-8", errors="ignore")
        assert type(result["stdout"]) == str
        assert type(result["stderr"]) == str
        return {
            "program": program,
            "stdout": result["stdout"].replace("!!int", "")[:2048],
            "stderr": result["stderr"][:2048],
            "exit_code": result["exit_code"],
            "status": result["status"],
        }


def eval_script_python(path: Path):
    output = None
    try:
        # Assumes exit-code 0 is all okay
        output = subprocess.run(
            ["python3", str(path)], encoding="utf-8", capture_output=True, timeout=5
        )
        returncode = -1
        if output.returncode == 0:
            status = "OK"
            returncode = output.returncode
        elif "SyntaxError" in output.stderr:
            status = "SyntaxError"
            returncode = output.returncode
        else:
            status = "Exception"
    except subprocess.TimeoutExpired as exc:
        status = "Timeout"
        returncode = -1
        output = exc

    return {
        "status": status,
        "exit_code": returncode,
        "stdout": str(output.stdout),
        "stderr": str(output.stderr),
    }


# def eval_in_thread(problem_yaml_path, index):
#     with open(problem_yaml_path) as f:
#         problem = Problem.load(f)
#     return eval_script(problem, index)
