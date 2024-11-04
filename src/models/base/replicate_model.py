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

import replicate

from src.configs.base_model_config import ModelConfig
from src.models.base.base_model import BaseModel, Message


class ReplicateModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self._config = config
        super().__init__(config)

        self._batch_size = config.batch_size
        self._add_special_tokens = config.add_special_tokens
        self._max_gen_toks = config.max_gen_toks
        self._max_length = config.max_length

    def get_answer(self, prompt) -> str:
        answer = ""

        # The meta/llama-2-70b-chat model can stream output as it's running.
        for event in replicate.stream(
            "meta/llama-2-70b-chat",
            input={
                "debug": True,
                "top_p": 0.95,
                "prompt": prompt,
                "temperature": 0.2,
                "max_new_tokens": 50,
                "min_new_tokens": -1,
                "prompt_template": "{prompt}",
            },
        ):
            answer += str(event)

        return answer

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """
        raise NotImplementedError("This API doesn't seem to return logits or scores")

    def generate_system(self, messages: List[List[Message]], **kwargs) -> List[str]:
        raise NotImplementedError("System messages are not yet implemented for ReplicateModel.")

    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            inputs (Union[str, List[str]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        inputs = inputs if isinstance(inputs, list) else [inputs]
        answers = [self.get_answer(inp) for inp in inputs]
        return answers
