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

from typing import Any, Dict

from pydantic import BaseModel as PBM
from pydantic import Field, WrapValidator
from typing_extensions import Annotated

from src.configs.dynamic_config import metric_config_validator


class MetricConfig(PBM):
    metric_args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the metric"
    )

    name: str = Field(default="", description="Hugginface name of metric")
    type: str = Field(..., description="Name of the metric")


DynamicMetricConfig = Annotated[MetricConfig, WrapValidator(metric_config_validator)]
