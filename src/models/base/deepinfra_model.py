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

from openai import OpenAI

from src.configs.base_model_config import ModelConfig
from src.models.base.base_model import BaseModel


class DeepinfraModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self._config = config
        super().__init__(config)

        self.client = OpenAI(base_url="https://api.deepinfra.com/v1/openai")

        self.model_name = "meta-llama/Llama-2-70b-chat-hf"

        self.batch_size = config.batch_size
        self._add_special_tokens = config.add_special_tokens
        self._max_gen_toks = config.max_gen_toks
        self._max_length = config.max_length

    def get_answer(self, prompt):
        assert isinstance(prompt, str)

        response = self.client.completions.create(
            model=self.model_name, temperature=1, top_p=0.95, max_tokens=256, prompt=prompt
        )

        return {"text": response.choices[0].text, "logprob": response.choices[0].logprobs}

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """

        inputs = inputs if isinstance(inputs, list) else list(inputs)
        cat_inputs = [a + b for a, b in inputs]
        answers = [(self.get_answer(inp)["logprob"], False) for inp in cat_inputs]

        print(answers)
        return answers

    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            inputs (Union[str, List[str]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """

        print(type(inputs))

        inputs = inputs if isinstance(inputs, list) else list(inputs)
        answers = [self.get_answer(inp)["text"] for inp in inputs]
        return answers