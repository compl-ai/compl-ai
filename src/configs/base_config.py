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

from pydantic import BaseModel as PBM
from pydantic import Field

import src.configs.base_benchmark_config as base_benchmark_config
from src.configs.base_model_config import ModelConfig


class Config(PBM):
    # This is the outermost config containing subconfigs for each benchmark as well as
    # IO and logging configs. The default values are set to None so that they can be
    # overridden by the user

    run_id: int | str = Field(..., description="ID for this run")
    model: ModelConfig  # Shouldn't be called model_config as it overwrites pydantic
    seed: int = Field(default=42, description="Seed to use for reproducibility")
    benchmark_configs: list[base_benchmark_config.DynamicBenchmarkConfig] = Field(
        default_factory=list, description="List of benchmark configs to run"
    )
