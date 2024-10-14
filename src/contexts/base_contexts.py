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

from abc import ABC, abstractmethod
from typing import List

from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.configs.base_model_config import ModelConfig
from src.data.base_data import BaseData
from src.metrics.base_metric import BaseMetric
from src.prompts.prompt_formatters import DefaultPromptFormatter
from src.results.base_connector import BaseConnector
from src.utils.log_manager import log_manager


class BenchmarkException(Exception):
    pass


class BaseContext(ABC):
    def construct_logger(self):
        # Create a logger
        logger_name = self.get_logger_name()
        logger = log_manager.get_new_logger(logger_name)
        self.logger = logger
        return logger

    @abstractmethod
    def get_logger_name(self) -> str:
        pass

    def get_logger(self):
        self.construct_logger()
        return self.logger


class BaseDataContext(BaseContext):
    """This is what each data context gets and contains methods to get all the relevant objects and information dynamically"""

    @abstractmethod
    def get_data_config(self) -> DataConfig:
        pass

    @abstractmethod
    def get_prompt_formatter(self) -> DefaultPromptFormatter:
        pass


class BaseBenchmarkContext(BaseContext):
    """This is what each benchmark gets and contains methods to get all the relevant objects and information dynamically"""

    @abstractmethod
    def get_benchmark_config(self) -> BenchmarkConfig:
        pass

    @abstractmethod
    def get_dataset(self) -> BaseData:
        pass

    @abstractmethod
    def get_prompt_formatter(self) -> DefaultPromptFormatter:
        pass

    @abstractmethod
    def get_data_handler(self) -> BaseConnector:
        pass

    @abstractmethod
    def get_model_config(self) -> ModelConfig:
        pass

    @abstractmethod
    def get_metrics(self) -> List[BaseMetric]:
        pass
