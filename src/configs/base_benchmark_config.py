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

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel as PBM
from pydantic import ConfigDict, Field, WrapValidator
from typing_extensions import Annotated

from src.configs.base_data_config import DynamicDataConfig
from src.configs.base_metric_config import DynamicMetricConfig
from src.configs.base_modifier_config import ModifierConfig
from src.configs.dynamic_config import benchmark_config_validator
from src.prompts.prompt_formatters import HFPromptConfig


class BenchmarkProvider(Enum):
    HF = "hf"
    LOCAL = "local"
    JSON = "json"
    TSV = "tsv"
    TXT = "txt"


class BenchmarkDebugConfig(PBM):  # type: ignore
    cpu_mode: bool = Field(False, description="Whether to run in CPU mode for testing")


class PostProcessor(PBM):
    type: str = Field(..., description="Type of postprocessor to use")


class BenchmarkConfig(PBM):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    debug: Optional[Union[bool, BenchmarkDebugConfig]] = Field(
        default=None, description="Whether or not to run in debug mode"
    )

    prompt_config: Union[HFPromptConfig, None] = Field(
        None, description="Configuration of prompt formatting, also includes chat layout"
    )

    data_config: Union[List[DynamicDataConfig], DynamicDataConfig] = Field(
        union_mode="left_to_right", default_factory=list
    )
    name: str = Field(..., description="Name of the benchmark")
    type: str = Field(..., description="Type of benchmark class used")

    num_workers: int = Field(1, description="Number of workers to use for the benchmark")
    modifier_configs: List[ModifierConfig] = Field(
        default_factory=list,
        description="List of modifier configs to apply to the benchmark",
    )
    metric_configs: List[DynamicMetricConfig] = Field(
        default_factory=list,
        description="List of metric configs to run on the benchmark",
    )

    postprocessor: PostProcessor = Field(
        ..., description="Postprocessor to use for final aggregation"
    )

    def is_cpu_mode(self) -> bool:
        return bool(
            self.debug and isinstance(self.debug, BenchmarkDebugConfig) and self.debug.cpu_mode
        )


class MultipleChoiceBenchmarkConfig(BenchmarkConfig):
    checking_mode: str = Field(
        "all",
        description="Options are 'label' (checks if the model would choose the correct label - restricted to valid answer labels), 'full' (checks if the model generates exactly the correct string - restricted to valid answers), 'free' (checks if the model generates exactly the correct string - not restricted to valid answers),",
    )


DynamicBenchmarkConfig = Annotated[BenchmarkConfig, WrapValidator(benchmark_config_validator)]
