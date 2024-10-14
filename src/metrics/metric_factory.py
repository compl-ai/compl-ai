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

import datasets

from src.configs.base_metric_config import MetricConfig
from src.metrics.base_metric import BaseMetric


def metric_class_from_hf(cls_name: str, hf_class: datasets.Metric):
    """Create a metric class from a Hugging Face metric class."""

    class WrappedClass(BaseMetric):
        def __init__(self, config: MetricConfig):
            super().__init__(config)
            self.metric = hf_class(config.name)

        def compute(self, *args, **kwargs) -> float:
            return self.metric.compute(*args, **kwargs)

        def add_batch(self, *args, **kwargs):
            return self.metric.add_batch(*args, **kwargs)

        def evaluate(self, *args, **kwargs) -> float:
            return self.compute(*args, **kwargs)

    WrappedClass.__name__ = "%s_%s" % (cls_name, datasets.Metric.__name__)
    return WrappedClass
