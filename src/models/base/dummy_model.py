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

import itertools
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.configs.base_model_config import ModelConfig
from src.models.base.base_model import BaseModel


def load_answers(output_dir: Path):
    # Define the command to be executed
    command = f"cat {output_dir / 'prompts.json'} | jq -s .[].additional_info.answers | jq -s | jq 'map(select(. != null)) | flatten'"

    # Run the command and capture its output
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    # Extract the stdout output
    stdout_output = result.stdout

    # Parse the JSON output
    json_output = json.loads(stdout_output)
    return json_output

    # Read data from prompts.json
    # with open(output_dir / 'prompts.json', 'r') as file:
    #    # data = json.load(file)
    #    data = ijson.items(file, 'additonal_info.answers')
    #    data = list(data)
    #    breakpoint()

    # Extract answers from additional_info and flatten the list
    # answers = [answer for prompt in data for answer in prompt['additional_info']['answers']]

    # return answers


def load_loglikelihoods(output_dir: Path):
    # Define the command to be executed
    command = f"cat {output_dir / 'prompts.json'} | jq -s .[].additional_info.loglikelihoods | jq -s | jq .[] | jq -s 'reduce .[] as $item ([]; . + $item)'"
    # command = f"cat {output_dir / 'prompts.json'} | jq -s .[].additional_info.loglikelihoods | jq -s | jq flatten"

    # Run the command and capture its output
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    # Extract the stdout output
    stdout_output = result.stdout

    # Parse the JSON output
    json_output = json.loads(stdout_output)
    return json_output

    # Read data from prompts.json
    # with open('prompts.json', 'r') as file:
    #     # data = json.load(file)
    #     data = ijson.items(file, 'additional_info.loglikelihoods')
    #     breakpoint()

    # Extract answers from additional_info and flatten the list
    # answers = [answer for prompt in data for answer in prompt['additional_info']['loglikelihoods']]

    # return answers


def load_batch_size(output_dir: Path):
    command = f"cat {output_dir / 'config.json'} | jq '.model.batch_size'"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    # Extract the stdout output
    batch_size = result.stdout
    return batch_size


class DummyModel(BaseModel):
    def __init__(self, config: ModelConfig, answers_file: Optional[str] = None):
        self._config = config
        super().__init__(config)
        self.model = config.name
        self.batch_size = config.batch_size
        self._add_special_tokens = config.add_special_tokens
        self._max_gen_toks = config.max_gen_toks
        self._max_length = config.max_length
        self._batch_size = config.batch_size

        self.tokenizer = "dummy_tokenizer"
        self.answers = None
        self.loglikelihoods = None
        if answers_file:
            answers_path = Path(answers_file)
            self._batch_size = int(load_batch_size(answers_path).strip())
            self.answers = iter(load_answers(answers_path))
            self.loglikelihoods = iter(load_loglikelihoods(answers_path))

    def perplexities(self, inputs):
        len_inputs = len(inputs)
        return [0.1] * len_inputs

    def loglikelihood(
        self, context_continuations: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            context_continuations (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs
        """
        len_inputs = len(context_continuations)
        if not self.loglikelihoods:
            return [(-0.1, False)] * len_inputs
        else:
            return list(itertools.islice(self.loglikelihoods, len_inputs))

    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            inputs (Union[str, List[str]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        len_inputs = len(inputs)

        if not self.answers:
            return ["A" * 512] * len_inputs

        else:
            return list(itertools.islice(self.answers, len_inputs))
