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

from typing import List, Tuple, Union

from src.models.base.base_model import BaseModel, Input, Message
from src.results.base_connector import BaseConnector


class ProxyModel(BaseModel):
    """
    A proxy model that logs the prompt and answer pairs to the data handler.
    Based upon the proxy design pattern.
    If a method is not found in this implementation, it will be forwarded to the base model.
    """

    def __init__(self, base_model: BaseModel, data_handler: BaseConnector):
        self.base_model = base_model
        self.data_handler = data_handler

    @property
    def model(self):
        return self.base_model.model  # type: ignore

    @property
    def config(self):
        return self.base_model.config

    @property
    def tokenizer(self):
        if hasattr(self.base_model, "tokenizer"):
            return self.base_model.tokenizer
        else:
            return None

    @property
    def batch_size(self):
        return self.base_model.config.batch_size

    def perplexities(self, inputs: List[str]) -> List[float]:
        prompts = inputs
        perplexities = self.base_model.perplexities(inputs)  # type: ignore

        self.data_handler.log_prompt_answer(prompts, additional_info={"perplexities": perplexities})

        return perplexities

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        prompts = inputs
        loglikelihoods = self.base_model.loglikelihood(inputs)

        self.data_handler.log_prompt_answer(
            prompts, additional_info={"loglikelihoods": loglikelihoods}
        )

        return loglikelihoods

    def loglikelihood_rolling(self, inputs: List[str]) -> List[float]:
        results = self.base_model.loglikelihood_rolling(inputs)  # type: ignore

        self.data_handler.log_prompt_answer(inputs, additional_info={"loglikelihoods": results})

        return results

    def generate_until(self, inputs: List[Input]) -> List[str]:
        results = self.base_model.generate_until(inputs)  # type: ignore

        self.data_handler.log_prompt_answer(inputs, additional_info={"answers": results})

        return results

    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        results = self.base_model.generate(inputs, **kwargs)

        self.data_handler.log_prompt_answer(inputs, additional_info={"answers": results})

        return results

    def generate_system(self, messages: List[List[Message]], **kwargs) -> List[str]:
        results = self.base_model.generate_system(messages, **kwargs)

        self.data_handler.log_prompt_answer(messages, additional_info={"answers": results})

        return results

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            result = getattr(self.base_model, name)(*args, **kwargs)
            return result

        return wrapper
