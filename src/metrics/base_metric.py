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

from src.configs.base_metric_config import MetricConfig
from src.results.base_result import BaseResult

# Build metrics according to https://huggingface.co/docs/datasets/dataset_script


class BaseMetric(ABC):
    def __init__(self, config: MetricConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float:
        """Evaluate the metric on the given inputs

        Returns:
            float: Metric value
        """

    @abstractmethod
    def add(self, *args, **kwargs):
        """Add results to the metric for evaluation"""

    @abstractmethod
    def add_batch(self, *args, **kwargs):
        """Add a batch to the metric for continuous evaluation"""

    @abstractmethod
    def compute(self) -> float:
        """Compute the metric value from the batches added"""

    @abstractmethod
    def compute_result(self, dict) -> BaseResult:
        pass
