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

import evaluate

from src.configs.base_metric_config import MetricConfig
from src.metrics.base_metric import BaseMetric
from src.results.base_result import BaseResult


class HFMetric(BaseMetric):
    def __init__(self, metric_config: MetricConfig):
        super().__init__(metric_config)
        self.metric_config = metric_config
        self.metric = evaluate.load(self.metric_config.name)

    def compute(self, *args, **kwargs) -> float:
        return self.metric.compute(*args, **kwargs)

    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)

    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> float:
        return self.compute(*args, **kwargs)

    def compute_result(self, results: dict) -> BaseResult:
        result = self.compute(**results)
        return BaseResult(name=self.metric.name, value=result, description=self.metric.description)
