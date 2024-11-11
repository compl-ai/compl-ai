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
import os.path
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic_core import from_json

from src.configs.base_config import Config
from src.results.base_connector import BaseConnector, BenchmarkInfo


def get_timed_run_folder(results_folder: Path, run_name: str, timestamp: bool = True) -> Path:
    """
    Creates a timed run folder based on the given run name and results folder.

    Args:
        results_folder (str): The path to the results folder.
        run_name (str): The name of the run.
        timestamp (bool): Whether to include the date in the results folder name.

    Returns:
        Path: The path to the created timed run folder.
    """
    if timestamp:
        # Get current date and time
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        run_name = run_name + "__" + datetime_string
    folder = results_folder / run_name
    folder.mkdir(parents=True, exist_ok=True)

    return folder


class FileConnector(BaseConnector):
    def __init__(
        self,
        benchmark_info: BenchmarkInfo,
        create_run: bool = True,
        run_folder: Optional[Path] = None,
        timestamp: bool = True,
        **kwargs
    ):
        """Creates folders and files if required by `create_run` flag."""

        super().__init__(benchmark_info, create_run, **kwargs)

        if not run_folder:
            raise ValueError("run_folder is required.")
        self.run_folder = run_folder

        self.run_id = self.get_run_id()
        self.create_run = create_run

        if create_run:
            self.run_folder = get_timed_run_folder(
                self.run_folder, benchmark_info.benchmark_type, timestamp=timestamp
            )

            # Set up output streams
            result_stream = open(self.run_folder / "eval_results.json", "w")
            prompt_stream = open(self.run_folder / "prompts.json", "w")
            prompt_idx_stream = open(self.run_folder / "prompts_idx.json", "w")

            self.result_stream = result_stream
            self.prompt_stream = prompt_stream
            self.prompt_idx_stream = prompt_idx_stream

    def _store_eval_results(self, evaluation_result):
        return self.result_stream.write(json.dumps(evaluation_result))

    def _store_final_result(self, final_result):
        with open(self.run_folder / "final_result.json", "w") as final_result_file:
            final_result_file.write(json.dumps(final_result))

    def results_exist(self) -> bool:
        return os.path.exists(self.run_folder / "final_result.json")

    def _store_prompt(self, prompt_entry) -> int:
        return self.prompt_stream.write(json.dumps(prompt_entry))

    def _store_prompt_idx(self, log_entry) -> int:
        return self.prompt_stream.write(json.dumps(log_entry))

    def __enter__(self) -> "FileConnector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.create_run:
            self.result_stream.close()
            self.prompt_stream.close()
            self.prompt_idx_stream.close()

    def store_config(self, config: Config) -> None:
        # Save config
        with open(self.run_folder / "config.json", "w") as config_file:
            config_file.write(config.model_dump_json())

    def get_config(self) -> Config:
        with open(self.run_folder / "config.json", "r") as config_file:
            content = config_file.read()

        json_config = from_json(content)
        return Config.model_validate(json_config)

    def log_error(self, exp: Exception) -> None:
        with open(self.run_folder / "error.txt", "w") as f:
            f.write(str(exp))
            f.write(traceback.format_exc())
