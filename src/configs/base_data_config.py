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

from typing import Annotated, Union

from pydantic import BaseModel as PBM
from pydantic import Field, WrapValidator

from src.configs.dynamic_config import data_config_validator
from src.prompts.prompt_formatters import HFPromptConfig


class DataConfig(PBM, extra="forbid", arbitrary_types_allowed=True):
    type: str
    debug: bool = Field(default=False, description="Whether to subset the dataset for debugging")
    subset_size: int = Field(default=1, description="Size of the subset for evaluation")
    prompt_config: Union[HFPromptConfig, None] = Field(
        None, description="Configuration of prompt formatting, also includes chat layout"
    )


DynamicDataConfig = Annotated[DataConfig, WrapValidator(data_config_validator)]
