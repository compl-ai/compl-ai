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
import logging
import time
from typing import Callable, Iterator, List, Optional

from pydantic import ValidationError

from src.benchmarks.base_benchmark import BaseBenchmark
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_config import Config
from src.contexts.benchmark_context import BenchmarkContext
from src.contexts.data_context import DataContext
from src.data.base_data import BaseData
from src.metrics.base_metric import BaseMetric
from src.models.base.base_model import BaseModel
from src.models.proxy_model import ProxyModel
from src.modifiers.base_modifier import BaseModifier, ModifierDataProvider
from src.modifiers.modifier_factory import get_modifier_from_config
from src.registry import BENCHMARK_PROCESSORS, registry
from src.results.base_connector import BaseConnector
from src.results.base_result import BaseResult, FinalResult, Results

log = logging.getLogger(__name__)


def get_json_results(results: List[BaseResult]) -> str:
    try:
        model_results = Results(results)  # type: ignore
        # This will trigger a validation error if wrong

        json_result = model_results.model_dump_json()

    except ValidationError:
        json_result = json.dumps(results)

    return json_result


class InternalBenchmarkRepresentation:
    """
    Represents an internal benchmark for evaluating a model's performance.

    Args:
        model (BaseModel): The model to be evaluated.
        config (BenchmarkConfig): The configuration for the benchmark.
        data_handler (optional): The data handler for processing the data. Defaults to None.

    Attributes:
        model (BaseModel): The model to be evaluated.
        config (BenchmarkConfig): The configuration for the benchmark.
        data_handler: The data handler for processing the data.
        benchmark (BaseBenchmark): The benchmark object.
        modifiers (List[BaseModifier]): The list of modifiers for the benchmark.
        data_providers (List[BaseData]): The list of data providers for the benchmark.
        metrics (List[BaseMetric]): The list of metrics for the benchmark.

    Methods:
        _get_data_providers: Retrieves the data providers from the configuration.
        _get_benchmark: Retrieves the benchmark class from the registry.
        _get_modifiers: Retrieves the modifiers from the configuration.
        _get_metrics: Retrieves the metrics from the configuration.
        benchmark_generator: Generates benchmarks using the data iterator.
        metric_generator: Generates metrics using the results iterator.
        modifier_generator: Generates modified data providers using the base data provider.
        data_generator: Generates data providers.
        to_results_iterator: Connects different components and returns the benchmark generator.

    """

    def __init__(self, model: BaseModel, config: BenchmarkConfig, data_handler=None):
        """
        Initializes the InternalBenchmarkRepresentation class.
        """

        self.model: BaseModel = model
        self.config: BenchmarkConfig = config
        self.data_handler = data_handler

        # Setup benchmarks
        self.benchmark: BaseBenchmark
        self._get_benchmark()

        # Setup modifiers
        self.modifiers: List[BaseModifier] = []
        self._get_modifiers()

        # Setup data providers
        self.data_providers: List[BaseData] = []
        self._get_data_providers()

        # Setup metrics
        self.metrics: List[BaseMetric]
        self._get_metrics()

        # Setup postprocessing for final aggregation
        postprocessor: Optional[Callable] = BENCHMARK_PROCESSORS[config.postprocessor.type]
        assert postprocessor is not None
        self.postprocessor = postprocessor

    def _get_data_providers(self):
        """
        Retrieves the data providers from the configuration.
        """

        if self.config.data_config:
            if isinstance(self.config.data_config, list):
                for data_config in self.config.data_config:
                    new_data_provider = registry.get("data").get_logic_cls(data_config.type)

                    data_context = DataContext(self.model, data_config)  # type: ignore
                    self.data_providers.append(new_data_provider(data_context))
            else:
                new_data_provider = registry.get("data").get_logic_cls(self.config.data_config.type)

                data_context = DataContext(self.model, self.config.data_config)  # type: ignore
                self.data_providers.append(new_data_provider(data_context))

    def _get_benchmark(self):
        cfg = self.config
        self.benchmark_cls = registry.get("benchmark").get_logic_cls(cfg.type)

    def _get_modifiers(self):
        cfg = self.config
        modifier_list = []
        for modfier_cfg in cfg.modifier_configs:
            modifier_list.append(get_modifier_from_config(modfier_cfg))  # type: ignore
        self.modifiers = modifier_list

    def _get_metrics(self):
        cfg = self.config
        metrics = []
        for metric_cfg in cfg.metric_configs:
            metric_cls = registry.get("metric").get_logic_cls(metric_cfg.type)
            metrics.append(metric_cls(metric_cfg))

        self.metrics = metrics

    def _log_intermediate_results(self, results: List[BaseResult]):
        """
        Logs intermediate results.

        Args:
            results (List[BaseResult]): The intermediate results to log.
        """

        self.json_results = get_json_results(results)
        self.results = results

        self.data_handler.add_evaluation_result(self.json_results)

    def benchmark_generator(self, data_iterator: Iterator[BaseData]):
        """
        Generates benchmark contexts for each data provider in the data iterator.

        Args:
            data_iterator (Iterator[BaseData]): An iterator that provides the data for benchmarking.

        Yields:
            BenchmarkContext: A benchmark context object for each data provider.
        """

        data_iterator_empty = True
        for data_provider in data_iterator:
            data_iterator_empty = False
            benchmark_context = BenchmarkContext(
                self.model,  # type: ignore
                self.benchmark_cls,
                self.config,
                data_provider=data_provider,
                metrics=self.metrics,
                data_handler=self.data_handler,
            )
            benchmark = self.benchmark_cls(benchmark_context)
            yield benchmark.eval_benchmark(self.model)

        if data_iterator_empty:
            benchmark_context = BenchmarkContext(
                self.model,  # type: ignore
                self.benchmark_cls,
                self.config,
                metrics=self.metrics,
                data_handler=self.data_handler,
            )
            benchmark = self.benchmark_cls(benchmark_context)
            yield benchmark.eval_benchmark(self.model)

    def modifier_generator(self, data_provider: BaseData):
        """
        Generates modified data providers using the base data provider.

        Args:
            data_provider (BaseData): The base data provider.

        Yields:
            ModifierDataProvider: The modified data provider.
        """

        for modifier in self.modifiers:
            # We are using the decorator pattern here to get a new object with the (almost) the same interface
            modified_data_provider = ModifierDataProvider(modifier, data_provider)
            yield modified_data_provider

    def data_generator(self):
        """
        Generates data providers.

        Yields:
            BaseData: The data provider.
        """

        for data_provider in self.data_providers:
            yield data_provider

            for modified_data_provider in self.modifier_generator(data_provider):
                yield modified_data_provider

    def postprocessing(self, results: dict) -> FinalResult:
        """
        Post-processing on te the results iterator.

        Args:
            results_iterator (Iterator[BaseResult]): The iterator for the benchmark results.

        Yields:
            BaseResult: The post-processed results.
        """

        return FinalResult(**self.postprocessor(results))

    def to_final_result(self) -> FinalResult:
        """
        Connects up different components and introduces intermediate connector objects.

        Returns:
            FinalResult: The final result of the benchmark.
        """

        data_generator = self.data_generator()
        benchmark_generator = self.benchmark_generator(data_generator)
        intermediate_results = list(benchmark_generator)

        json_results = get_json_results(intermediate_results)
        final_result = self.postprocessing({"benchmark_specific_info": json_results})

        return final_result


class Runner:
    """
    The Runner class is responsible for running the model and handling the results.

    Args:
        model (BaseModel): The model to be run.
        config (Config): The configuration for the model.
        result_handler (BaseConnector): The result handler for storing and processing the results.

    Attributes:
        config (Config): The configuration for the model.
        result_handler (BaseConnector): The result handler for storing and processing the results.
        model (ProxyModel): The proxy model used for running the actual model and handling the results.
        internal_repr (InternalBenchmarkRepresentation): The internal representation of the benchmark.

    """

    def __init__(self, model: BaseModel, config: Config, result_handler: BaseConnector):
        self.config: Config = config
        self.result_handler = result_handler
        self.model = ProxyModel(model, result_handler)

        self.internal_repr = InternalBenchmarkRepresentation(
            self.model, self.config.benchmark_configs[0], data_handler=self.result_handler
        )

    def run(self):
        """
        Runs the model and handles the results.
        """

        log.info("Running...")
        log.info("Config: {}".format(self.config))

        start = time.time()

        try:
            final_result = self.internal_repr.to_final_result()
        except Exception as e:
            self.result_handler.log_error(e)
            raise e

        runtime = time.time() - start

        self.result_handler.add_final_result(final_result, runtime=runtime)

        log.info("Done.")
