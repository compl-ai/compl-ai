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

from typing import List, cast

from config import dataset_registry
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.configs.base_model_config import ModelConfig
from src.contexts.base_contexts import BaseBenchmarkContext, BenchmarkException
from src.data.base_data import BaseData
from src.metrics.base_metric import BaseMetric
from src.models.proxy_model import ProxyModel
from src.prompts.prompt_formatters import DefaultPromptFormatter, PromptConfig
from src.results.base_connector import BaseConnector


class BenchmarkContext(BaseBenchmarkContext):
    """
    The BenchmarkContext class represents the context in which a benchmark is executed.
    It provides access to the base model, benchmark configuration, datasets, metrics, and data handler.

    Args:
        base_model (BaseModel): The base model used for the benchmark.
        benchmark_cls: The benchmark class.
        benchmark_config (BenchmarkConfig): The configuration for the benchmark.
        data_provider: The data provider for the benchmark.
        metrics: The metrics to be used for evaluation.
        data_handler: The data handler for the benchmark.

    Attributes:
        base_model (BaseModel): The base model used for the benchmark.
        benchmark_config (BenchmarkConfig): The configuration for the benchmark.
        benchmark_cls: The benchmark class.
        prompt_config (PromptConfig): The prompt configuration for the benchmark.
        datasets (dict[str, BaseData]): A dictionary of datasets available for the benchmark.
        dataset_configs (dict[str, DataConfig]): A dictionary of dataset configurations.
        handler: The data handler for the benchmark.
        metrics: The metrics to be used for evaluation.
        prompt_formatter (PromptFormatter): The prompt formatter for the benchmark.

    Methods:
        get_logger_name(): Returns the logger name for the benchmark.
        run(): Runs the benchmark and returns the results.
        add_dataset(dataset_config: DataConfig): Adds a dataset to the benchmark.
        get_dataset(): Returns the dataset used for the benchmark.
        get_metrics(): Returns the metrics used for evaluation.
        get_prompt_formatter(): Returns the prompt formatter for the benchmark.
        get_benchmark_config(): Returns the benchmark configuration.
        get_dataset_config(name: str): Returns the configuration for a specific dataset.
        get_data_handler(): Returns the data handler for the benchmark.
    """

    def __init__(
        self,
        model: ProxyModel,
        benchmark_cls,
        benchmark_config: BenchmarkConfig,
        data_provider=None,
        metrics=None,
        data_handler=None,
    ):
        super().__init__()

        self.model = model
        self.model_config = model.config
        self.benchmark_config = benchmark_config
        self.benchmark_cls = benchmark_cls
        self.prompt_config = benchmark_config.prompt_config or PromptConfig()
        self.datasets: dict[str, BaseData] = dict()
        self.dataset_configs: dict[str, DataConfig] = dict()

        self.handler = data_handler
        self.metrics = metrics

        chat_formatter = self.model.base_model.get_chat_formatter()
        self.prompt_formatter = DefaultPromptFormatter(
            self.prompt_config, chat_formatter=chat_formatter
        )

        # Initialize datasets the benchmark will have access to
        self.datasets["dataset"] = data_provider

    def get_logger_name(self) -> str:
        return f"benchmark-{self.benchmark_config.type}"

    def add_dataset(self, dataset_config: DataConfig):
        dataset_name = dataset_config.type
        data_logic_cls = dataset_registry.get_logic_cls(dataset_name)
        self.datasets[dataset_name] = cast(BaseData, data_logic_cls(dataset_config))
        self.dataset_configs[dataset_name] = dataset_config

    def get_dataset(self) -> BaseData:
        """Gets one dataset and its config, but throws an error if there are multiple ones"""
        values = self.datasets.values()
        if len(values) == 0:
            raise BenchmarkException("Benchmark context doesn't have registered any dataset")
        if len(values) != 1:
            raise BenchmarkException(
                "Benchmark context is aware of more than one dataset, therefore it cannot distinguish between them!"
            )
        return list(values)[0]

    def get_metrics(self) -> List[BaseMetric]:
        return self.metrics

    def get_prompt_formatter(self) -> DefaultPromptFormatter:
        return self.prompt_formatter

    def get_benchmark_config(self) -> BenchmarkConfig:
        return self.benchmark_config

    def get_dataset_config(self, name: str) -> DataConfig:
        return self.dataset_configs[name]

    def get_data_handler(self) -> BaseConnector:
        return self.handler

    def get_model_config(self) -> ModelConfig:
        return self.model_config
