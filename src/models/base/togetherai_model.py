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

import asyncio
import logging
import math
import time
from typing import Any, List, Tuple, Union

from aiohttp import ClientSession
from tqdm import tqdm

from secret import TOGETHERAI_API_KEY
from src.configs.base_model_config import ModelConfig
from src.models.base.base_model import BaseModel, Message

from .utils import PromptStatistics, RateLimit, chunks

logging.getLogger("asyncio").setLevel(logging.WARNING)


# Requests per minute
RPM = 400


class TogetherAIModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.model = config.name
        PromptStatistics.set_model(config.name)
        self.batch_size = config.batch_size
        self.has_started = False
        self._init_generation_args(config)

    def _init_generation_args(self, config: ModelConfig) -> None:
        self.generation_args = {"max_tokens": config.max_gen_toks}
        if config.generation_args is not None:
            for key, value in config.generation_args.items():
                if key == "top_p":
                    self.generation_args["top_p"] = value
                elif key == "temperature":
                    self.generation_args["temperature"] = value
                elif key == "frequency_penalty":
                    raise ValueError("Unsupported argument 'frequency_penalty' for TogetherAI API.")
                elif key == "presence_penalty":
                    raise ValueError("Unsupported argument 'presence_penalty' for TogetherAI API.")
                elif key == "num_return_sequences":
                    raise ValueError(
                        "Unsupported argument 'num_return_sequences' for TogetherAI API."
                    )

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """
        raise NotImplementedError()

    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            inputs (Union[str, List[str]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        inputs = [inputs] if isinstance(inputs, str) else inputs

        return self.generate_system(
            [[{"role": "user", "content": content}] for content in inputs], **kwargs
        )

    def generate_system(self, messages: List[List[Message]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            messages (List[List[Message]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        if "max_length" in kwargs:
            self.generation_args["max_tokens"] = kwargs["max_length"]
        if "top_p" in kwargs:
            self.generation_args["top_p"] = kwargs["top_p"]
        if "temperature" in kwargs:
            self.generation_args["temperature"] = kwargs["temperature"]
        if "num_return_sequences" in kwargs:
            raise ValueError("Unsupported argument 'num_return_sequences' for TogetherAI API.")

        results = []

        for messages_batch in tqdm(
            chunks(messages, self.batch_size), total=math.ceil(len(messages) / self.batch_size)
        ):
            with RateLimit(rpm=RPM / self.batch_size):

                async def generate_async():
                    async with ClientSession() as session:
                        tasks = [
                            self._create_chat_completion(session, message)
                            for message in messages_batch
                        ]
                        responses = await asyncio.gather(*tasks)
                    return [
                        element["message"]["content"]
                        for response in responses
                        for element in response["choices"]
                    ]

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results.extend(loop.run_until_complete(generate_async()))
                loop.close()

        self.has_started = False

        return results

    async def _create_chat_completion(self, session: ClientSession, messages: List[Message]) -> Any:
        """Creates a chat completion for a message.

        Args:
            messages (dict): messages

        Returns:
            Any: OpenAI API response
        """
        retry_counter = 4

        content_messages = [
            {"role": message["role"], "parts": [{"text": message["content"]}]}
            for message in messages
        ]

        for _ in range(retry_counter):
            try:
                if not self.has_started:
                    self.first_time = time.time()
                    self.has_started = True
                async with session.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {TOGETHERAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": content_messages,
                        **self.generation_args,
                    },
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        PromptStatistics.log_response(response_json)
                        return response_json
                    else:
                        seconds_since_start = (time.time() - self.first_time) % 60
                        print(
                            f"Error: {response.status} at {time.strftime('%X')} with {60 - seconds_since_start} wait."
                        )
                        print(response)
                        await asyncio.sleep(60 - seconds_since_start)
            except Exception as e:
                print(f"Aiohttp Error: {e}")
                await asyncio.sleep(60)
        raise Exception("Error in TogetherAI API.")
