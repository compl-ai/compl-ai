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

import json
import os
from pathlib import Path

from pydantic_core import from_json

from src.results.base_results_repository import BaseResultsRepository, RunInfo


def node_directories(results_directory) -> list[str]:
    """
    Retrieves a list of directories containing a 'config.json' file within the given results directory.

    Args:
        results_directory (str): The path to the results directories.

    Returns:
        list[str]: A list of directory paths containing a 'config.json' file.
    """

    node_dirs = []
    for root, dirs, files in os.walk(results_directory):
        if "config.json" in files:
            node_dirs.append(root)
    return node_dirs


def get_config(run_folder):
    with open(run_folder / "config.json", "r") as config_file:
        content = config_file.read()

    json_config = from_json(content)
    return json_config  # Config(**json_config)


def get_run_info(run_path: Path) -> RunInfo:
    config = get_config(run_path)

    results_file = run_path / "eval_results.json"

    results = {}

    try:
        with open(results_file, "r") as f:
            results = json.load(f)
    except Exception:
        pass

    if len(results) == 0:
        try:
            alt_res_file = run_path / "final_result.json"
            with open(alt_res_file, "r") as f:
                results = json.load(f)
        except Exception:
            pass

    # compatibility with old config format
    if "config" in config:
        config = config["config"]

    run_info = RunInfo(
        benchmark_name=(
            config["benchmark_configs"][0]["name"]
            if "name" in config["benchmark_configs"][0]
            else config["benchmark_configs"][0]["type"]
        ),
        model_name=config["model"]["name"],
        data=results,
        config=config,
    )
    return run_info


class FileResultsRepository(BaseResultsRepository):
    def __init__(self, results_dir: Path):
        super().__init__(results_dir)
        self.results_dir = results_dir

    def list(self) -> list[RunInfo]:
        node_dirs = node_directories(self.results_dir)
        run_infos = []
        for node_dir in node_dirs:
            try:
                run_infos.append(get_run_info(Path(node_dir)))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return run_infos
