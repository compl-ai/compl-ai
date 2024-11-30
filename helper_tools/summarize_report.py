#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
import datetime
import json
import os
import time
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate


class Status(Enum):
    OK = "\u2713"
    ERROR = "\u2717"
    RUNNING = "\u25F7"
    PENDING = "\u2219"


def clear() -> None:
    """Clears the console output."""
    if os.name == "nt":
        # for windows
        _ = os.system("cls")
    else:
        # for mac and linux(here, os.name is 'posix')
        _ = os.system("clear")


def _get_benchmark_status(benchmark_dir: Path) -> str:
    if not benchmark_dir.exists():
        return Status.PENDING.value
    if (benchmark_dir / "error.txt").exists():
        return Status.ERROR.value
    if (benchmark_dir / "final_result.json").exists():
        return Status.OK.value
    return Status.RUNNING.value


def _read_error_desc(benchmark_dir: Path) -> str:
    with open(benchmark_dir / "error.txt", "r") as f:
        lines = f.readlines()
        return lines[-1][:30] + "..."


def _get_benchmark_info(benchmark_dir: Path) -> dict:
    if not (benchmark_dir / "final_result.json").exists():
        if _get_benchmark_status(benchmark_dir) == Status.RUNNING.value:
            dir_creation_time = os.path.getctime(benchmark_dir / "config.json")
            return {
                # time since the directory was created
                "runtime": time.time()
                - dir_creation_time,
            }
        if _get_benchmark_status(benchmark_dir) == Status.ERROR.value:
            dir_creation_time = os.path.getctime(benchmark_dir / "config.json")
            error_creation_time = os.path.getctime(benchmark_dir / "error.txt")
            return {
                # time between
                "runtime": error_creation_time
                - dir_creation_time,
            }
        return {}

    with open(benchmark_dir / "final_result.json", "r") as f:
        return json.load(f)


# These are based on running gemma-2-2b-it on NVIDIA RTX 6000 Ada
# The values are in seconds (rounded up to a minute)
REFERENCE_RUNTIMES = {
    "ai2_reasoning": 60 * 29,
    "bbq": 60 * 1,
    "bold": 60 * 28,
    "boolq_contrast_robustness": 60 * 1,
    "calibration_big_bench": 60 * 1,
    "calibration_big_bench_i_know": 60 * 109,
    "decoding_trust": 60 * 1,
    "fairllm": 60 * 137,
    "forecast_consistency": 60 * 10,
    "hellaswag": 60 * 216,
    "human_deception": 60 * 1,
    "human_eval": 60 * 55,
    "imdb_contrast_robustness": 60 * 2,
    "instruction_goal_hijacking": 60 * 8,
    "memorization": 60 * 7,
    "mmlu": 60 * 145,
    "mmlu_robustness": 60 * 79,
    "multiturn_goal_hijacking": 60 * 20,
    "privacy": 60 * 3,
    "reddit_bias": 60 * 3,
    "self_check_consistency": 60 * 36,
    "toxicity": 60 * 94,
    "toxicity_advbench": 60 * 1,
    "truthful_qa_mc2": 60 * 7,
}


def human_time_duration(seconds: float) -> str:
    """Format seconds into a human-readable string, e.g., 1 hour, 23 min.

    Args:
        seconds (float): Number of seconds.
    """
    # drop seconds for better readability
    seconds = int(seconds)
    seconds += 60 - (seconds % 60)

    minutes = seconds // 60
    hours = minutes // 60
    if hours > 0:
        return f"{hours:3d} hr, {minutes % 60:2d} min"
    else:
        return f"{minutes % 60:2d} min"


def print_summary(parent_dir: str) -> None:
    entries = []
    total_runtime = 0
    runtime_scales = []
    for benchmark, ref_runtime in REFERENCE_RUNTIMES.items():
        benchmark_dir = Path(parent_dir, benchmark)
        status = _get_benchmark_status(benchmark_dir)

        info = _get_benchmark_info(benchmark_dir)
        runtime = info.get("runtime", None)
        category = info.get("category", "")

        score = ""
        if status == Status.OK.value:
            score = f"{info['FinalResult']['aggregate_score']:.2f}"
        if status == Status.ERROR.value:
            score = _read_error_desc(benchmark_dir)

        if runtime is not None and status == Status.OK.value:
            total_runtime += runtime
            runtime_scales.append(runtime / ref_runtime)  # type: ignore

        entry = {
            "category": category,
            "runtime": runtime,
            "status": status,
            "name": benchmark,
            "result": score,
        }
        entries.append(entry)

    runtime_scale = np.mean(runtime_scales) if runtime_scales else 1.0

    for entry, (benchmark, ref_runtime) in zip(entries, REFERENCE_RUNTIMES.items()):
        if entry["runtime"] is None:
            entry["runtime"] = "(estimated) " + human_time_duration(runtime_scale * ref_runtime)
        elif entry["status"] == Status.RUNNING.value:
            entry["runtime"] = (
                "(running) "
                + human_time_duration(entry["runtime"])
                + " /"
                + human_time_duration(runtime_scale * ref_runtime)
            )
        else:
            entry["runtime"] = human_time_duration(entry["runtime"])

    df = pd.DataFrame.from_records(
        entries, columns=["status", "name", "runtime", "result", "category"]
    )
    df.sort_values(by=["status", "category", "name"], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)

    print(
        tabulate(
            df,
            showindex=False,
            headers=df.columns,
            colalign=["center", "right", "right", "left", "left"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark runtimes from results.")
    parser.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="Path to the parent directory.",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        required=False,
        default=-1,
        help="Refresh interval in seconds.",
    )

    args = parser.parse_args()

    try:
        while True:
            if args.refresh > 0:
                clear()
            print(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "-", args.parent_dir)
            print()
            print_summary(args.parent_dir)
            print()
            print("Legend:")
            print(
                ", ".join(
                    [
                        f"└── {s.value} {s.name}"
                        for s in [Status.OK, Status.ERROR, Status.PENDING, Status.RUNNING]
                    ]
                )
            )
            print()
            print("Note:")
            print(
                "The expected runtimes are based on a reference evaluation and can vary significantly based on your hardware."
            )
            print()

            if args.refresh <= 0:
                break

            for i in range(args.refresh):
                print(f"\rRefresh in {args.refresh - i} seconds...", end="\r", flush=True)
                time.sleep(1)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
