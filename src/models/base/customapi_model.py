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

import requests
from tqdm import tqdm

from src.configs.base_model_config import ModelConfig
from src.models.base.base_model import BaseModel, Message

from .utils import chunks


class CustomAPIModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.url is None:
            raise ValueError("URL is required for CustomAPIModel")
        if config.endpoints is None:
            raise ValueError("Endpoints are required for CustomAPIModel")

        self.url = config.url
        self.endpoints = config.endpoints
        self.model = config.name
        self.batch_size = config.batch_size

    def detect_watermark(self, text: str) -> bool:
        """Detects whether a given text contains a watermark.

        Args:
            text (str): Text to check for watermark

        Returns:
            bool: True if watermark is detected, False otherwise
        """
        if "detect_watermark" not in self.endpoints:
            raise NotImplementedError(
                "Detect watermark endpoint not implemented for the given CustomAPIModel"
            )
        endpoint = self.url + self.endpoints["detect_watermark"]
        response = requests.post(endpoint, json={"queries": [text]})
        return response.json()["is_watermarked"][0]

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """
        raise NotImplementedError("Log likelihood not implemented for CustomAPIModel")

    def generate_system(self, messages: List[List[Message]], **kwargs) -> List[str]:
        raise NotImplementedError("Generate with system queries not implemented for CustomAPIModel")

    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            inputs (Union[str, List[str]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        if "generate" not in self.endpoints:
            raise NotImplementedError(
                "Generate endpoint not implemented for the given CustomAPIModel"
            )
        endpoint = self.url + self.endpoints["generate"]

        inputs = [inputs] if isinstance(inputs, str) else inputs
        responses = []

        for inputs_batch in tqdm(chunks(inputs, self.batch_size)):
            payload = {"queries": inputs_batch}
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                responses.extend(response.json()["responses"])
            else:
                print("Request failed:", response.text)
        return responses
