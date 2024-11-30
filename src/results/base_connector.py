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

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.configs.base_config import Config
from src.results.base_result import FinalResult


@dataclass
class BenchmarkInfo:
    benchmark_type: str
    category: str


class BaseConnector(ABC):
    """
    Base class for connectors that store evaluation results and log prompts for a specific benchmark.
    """

    def __init__(self, benchmark_info: BenchmarkInfo, create_run: bool = True, **kwargs):
        """
        Initializes a new instance of the BaseConnector class.

        Args:
            results_folder (Path): The folder where the evaluation results will be stored.
            config (Config): The configuration object containing benchmark-specific settings.
            create_run (bool, optional): Whether to create a new run or use an existing one. Defaults to True.
        """

        self.benchmark = benchmark_info.benchmark_type
        self.category = benchmark_info.category

        self.run_id = str(uuid.uuid4())

    def get_run_id(self):
        """
        Gets the unique identifier for the current run.

        Returns:
            str: The run ID.
        """
        return self.run_id

    @abstractmethod
    def _store_eval_results(self, evaluation_results):
        """
        Abstract method to store the evaluation results.

        Args:
            evaluation_results (dict): The evaluation results to be stored.
        """
        pass

    @abstractmethod
    def _store_prompt(self, prompt_entry):
        """
        Abstract method to store a prompt log entry.

        Args:
            prompt_entry (dict): The prompt log entry to be stored.
        """
        pass

    @abstractmethod
    def _store_prompt_idx(self, log_entry):
        """
        Abstract method to store a prompt index log entry.

        Args:
            log_entry (dict): The prompt index log entry to be stored.
        """
        pass

    @abstractmethod
    def _store_final_result(self, final_result):
        """
        Abstract method to store the final result.

        Args:
            final_result (dict): The final result to be stored.
        """
        pass

    def add_evaluation_result(self, benchmark_specific_info):
        """
        Adds an evaluation result to the connector.

        Args:
            benchmark_specific_info (dict): The benchmark-specific information containing the evaluation results to be stored.
        """

        evaluation_result = {
            "run_id": self.run_id,
            "time": time.time(),
            "category": self.category,
            "benchmark": self.benchmark,
            "benchmark_specific_info": benchmark_specific_info,
        }
        return self._store_eval_results(evaluation_result)

    def add_final_result(self, final_result: FinalResult, runtime: float):
        """
        Adds the final result to the underlying storage.

        Args:
            FinalResult (dict): The final result to be stored.
            runtime (float): The runtime of the benchmark.
        """

        final_result_info = final_result.model_dump()

        final_result_info = {
            "run_id": self.run_id,
            "time": time.time(),
            "runtime": runtime,
            "category": self.category,
            "benchmark": self.benchmark,
            "FinalResult": final_result_info,
        }
        return self._store_final_result(final_result_info)

    def log_prompt_answer(self, prompt, additional_info=None):
        """
        Logs a prompt answer.

        Args:
            prompt (str): The prompt text.
            additional_info (Any, optional): Additional information to be logged. Defaults to None.
        """

        log_entry = {
            "run_id": self.run_id,
            "time": time.time(),
            "category": self.category,
            "benchmark": self.benchmark,
            "prompt": prompt,
            "additional_info": additional_info,
        }
        return self._store_prompt(log_entry)

    def log_prompt_all_indices(self, all_indices, additional_info=None):
        """
        Logs all prompt indices.

        Args:
            all_indices (list): The list of prompt indices.
            additional_info (Any, optional): Additional information to be logged. Defaults to None.
        """

        log_entry = {
            "run_id": self.run_id,
            "time": time.time(),
            "category": self.category,
            "benchmark": self.benchmark,
            "prompt_indices": all_indices,
            "additional_info": additional_info,
        }

        return self._store_prompt_idx(log_entry)

    def log_prompt_idx(self, idx: int, additional_info=None):
        """
        Logs a prompt index.

        Args:
            idx (int): The prompt index.
            additional_info (Any, optional): Additional information to be logged. Defaults to None.
        """

        log_entry = {
            "run_id": self.run_id,
            "time": time.time(),
            "category": self.category,
            "benchmark": self.benchmark,
            "prompt_idx": idx,
            "additional_info": additional_info,
        }
        return self._store_prompt_idx(log_entry)

    @abstractmethod
    def get_config(self) -> Config:
        pass

    @abstractmethod
    def store_config(self, config: Config) -> None:
        pass

    @abstractmethod
    def log_error(self, exp: Exception) -> None:
        pass
