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
import itertools
import math
import time
from typing import Any, List, Tuple, Union

import numpy as np
from aiohttp import ClientSession
from tqdm import tqdm

from secret import OPENAI_API_KEY, OPENAI_ORG
from src.configs.base_model_config import ModelConfig
from src.models.base.base_model import BaseModel, ContexContinuations, Message
from src.models.dtypes.dtypes_openai import ChatCompletion

from .utils import PromptStatistics, RateLimit, chunks

# Requests per minute
RPM = 10000


class OpenAIModel(BaseModel):
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
                    self.generation_args["frequency_penalty"] = value
                elif key == "presence_penalty":
                    self.generation_args["presence_penalty"] = value
                elif key == "num_return_sequences":
                    self.generation_args["n"] = value

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """
        # If contexts are the same we don't need to recompute each time
        grouped_inputs = [
            ContexContinuations(k, [el[1] for el in g])
            for k, g in itertools.groupby(inputs, key=lambda x: x[0])
        ]

        if self.has_started is False:
            self.generation_args["logprobs"] = True
            self.generation_args["top_logprobs"] = 20
            self.generation_args["max_tokens"] = 5

        results: list = []

        for pair_batch in tqdm(chunks(grouped_inputs, self.batch_size), ncols=120, leave=False):
            with RateLimit(rpm=RPM / self.batch_size):

                async def generate_async():
                    async with ClientSession() as session:
                        tasks = [self._get_logprobs(session, message) for message in pair_batch]
                        responses = await asyncio.gather(*tasks)
                    return responses

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results.extend(loop.run_until_complete(generate_async()))
                loop.close()

        self.has_started = False

        return list(itertools.chain.from_iterable(results))

    def generate_system(self, messages: List[List[Message]], **kwargs) -> List[str]:
        if "max_length" in kwargs:
            self.generation_args["max_tokens"] = kwargs["max_length"]
        if "top_p" in kwargs:
            self.generation_args["top_p"] = kwargs["top_p"]
        if "temperature" in kwargs:
            self.generation_args["temperature"] = kwargs["temperature"]
        if "num_return_sequences" in kwargs:
            self.generation_args["n"] = kwargs["num_return_sequences"]

        results = []

        for messages_batch in tqdm(
            chunks(messages, self.batch_size),
            total=math.ceil(len(messages) / self.batch_size),
            ncols=120,
            leave=False,
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

    async def _create_chat_completion(self, session: ClientSession, messages: List[Message]) -> Any:
        """Creates a chat completion for a message.

        Args:
            messages (List[Message]): messages

        Returns:
            Any: OpenAI API response
        """
        retry_counter = 4
        for _ in range(retry_counter):
            try:
                if not self.has_started:
                    self.first_time = time.time()
                    self.has_started = True
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                        "OpenAI-Organization": OPENAI_ORG,
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
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
                        await asyncio.sleep(60 - seconds_since_start)
            except Exception as e:
                print(f"Aiohttp Error: {e}")
                await asyncio.sleep(60)
        raise Exception("Error in OpenAI API.")

    async def _get_logprobs(
        self, session: ClientSession, pair: ContexContinuations
    ) -> List[Tuple[float, bool]]:
        """Extracts log-probabilities from a given context and its continuations.

        Returns:
            Tuple[float, bool]: (log-likelihood, is-exact-match) pair
        """
        context = {"role": "user", "content": pair.context}

        # We only check if the main letter is correct i.e. A or B
        continuations = [c.lstrip()[0] for c in pair.continuations]

        retry_counter = 4
        for _ in range(retry_counter):
            try:
                if not self.has_started:
                    self.first_time = time.time()
                    self.has_started = True
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                        "OpenAI-Organization": OPENAI_ORG,
                    },
                    json={
                        "model": self.model,
                        "messages": [context],
                        **self.generation_args,
                    },
                ) as response:
                    if response.status == 200:
                        response = await response.json()
                        completion = ChatCompletion(**response)
                        answers = [(-np.inf, False)] * len(continuations)

                        for i, element in enumerate(completion.choices[0].logprobs.content[0].top_logprobs):  # type: ignore
                            if element.token in continuations and i == 0:
                                answers[continuations.index(element.token)] = (
                                    element.logprob,
                                    True,
                                )
                            elif element.token in continuations:
                                answers[continuations.index(element.token)] = (
                                    element.logprob,
                                    False,
                                )

                        return answers
                    else:
                        seconds_since_start = (time.time() - self.first_time) % 60
                        print(
                            f"Error: {response.status} at {time.strftime('%X')} with {60 - seconds_since_start} wait."
                        )
                        await asyncio.sleep(60 - seconds_since_start)
            except Exception as e:
                print(f"Aiohttp Error: {e}")
                await asyncio.sleep(60)
        raise Exception("Error in OpenAI API.")
