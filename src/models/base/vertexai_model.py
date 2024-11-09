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
import json
import logging
import math
import sys
import time
from typing import List, Tuple, Union

import google.auth
import google.auth.transport.requests
import numpy as np
from aiohttp import ClientSession
from tqdm import tqdm

from secret import VERTEX_AI_LOCATION, VERTEX_AI_PROJECT_ID
from src.configs.base_model_config import ModelConfig
from src.models.base.base_model import BaseModel, ContextContinuations, Message
from src.models.dtypes.dtypes_vertexai import ChatCompletion

from .utils import PromptStatistics, RateLimit, chunks

logging.getLogger("asyncio").setLevel(logging.WARNING)


# Requests per minute
RPM = 200


def _get_response_text(response: ChatCompletion) -> list[str]:
    if response.promptFeedback and response.promptFeedback.blockReason is not None:
        assert response.candidates is None or len(response.candidates) == 0, response
        return ["I cannot provide a response."]

    if response.candidates is None:
        logging.warning("No candidates and no promptFeedback in model response")
        return ["No Response"]

    texts = []
    for candidate in response.candidates:
        if candidate.finishReason in [
            "OTHER",
            "SAFETY",
            "PROHIBITED_CONTENT",
            "SPII",
            "BLOCKLIST",
            "RECITATION",
        ]:
            texts.append("I cannot provide a response.")
        if (
            candidate.content is None
            or candidate.content.parts is None
            or len(candidate.content.parts) == 0
        ):
            texts.append("I cannot provide a response.")
        else:
            try:
                texts.append(candidate.content.parts[0].text)
            except Exception as err:
                print(response)
                raise err
    return texts


class VertexAIModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # gemini-1.5-flash
        # gemini-1.5-flash-8b
        # gemini-1.5-pro
        # gemini-1.0-pro
        self.model = config.name
        PromptStatistics.set_model(config.name)
        self.batch_size = config.batch_size
        self.has_started = False
        self._init_generation_args(config)

        self.creds, project = google.auth.default()

        # creds.valid is False, and creds.token is None
        # Need to refresh credentials to populate those
        auth_req = google.auth.transport.requests.Request()
        self.creds.refresh(auth_req)

    @property
    def auth_token(self) -> str:
        if not self.creds.valid:
            auth_req = google.auth.transport.requests.Request()
            self.creds.refresh(auth_req)
        return self.creds.token

    @property
    def endpoint_url(self) -> str:
        return f"https://{VERTEX_AI_LOCATION}-aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT_ID}/locations/{VERTEX_AI_LOCATION}/publishers/google/models/{self.model}:generateContent"

    def _init_generation_args(self, config: ModelConfig) -> None:
        self.generation_args = {"maxOutputTokens": config.max_gen_toks}
        if config.generation_args is not None:
            for key, value in config.generation_args.items():
                if key == "top_p":
                    self.generation_args["topP"] = value
                elif key == "temperature":
                    self.generation_args["temperature"] = value
                elif key == "frequency_penalty":
                    self.generation_args["frequencyPenalty"] = value
                elif key == "presence_penalty":
                    self.generation_args["presencePenalty"] = value
                elif key == "num_return_sequences":
                    self.generation_args["candidateCount"] = max(1, min(8, value))

    def generate_system(self, messages: List[List[Message]], **kwargs) -> List[str]:
        if "max_length" in kwargs:
            self.generation_args["maxOutputTokens"] = kwargs["max_length"]
        if "top_p" in kwargs:
            self.generation_args["topP"] = kwargs["top_p"]
        if "temperature" in kwargs:
            self.generation_args["temperature"] = kwargs["temperature"]
        if "num_return_sequences" in kwargs:
            self.generation_args["candidateCount"] = max(1, min(8, kwargs["num_return_sequences"]))

        results = []

        for messages_batch in tqdm(
            chunks(messages, self.batch_size),
            total=math.ceil(len(messages) / self.batch_size),
            file=sys.stdout,
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
                    return list(
                        itertools.chain.from_iterable(
                            [_get_response_text(response) for response in responses]
                        )
                    )

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

    async def _create_chat_completion(
        self, session: ClientSession, messages: List[Message]
    ) -> ChatCompletion:
        """Creates a chat completion for a message.

        Args:
            messages (Message): message

        Returns:
            Any: Gemini API response
        """
        retry_counter = 4

        ROLES = {"system": "system", "user": "user", "assistant": "model"}

        content_messages = [
            {"role": ROLES[message["role"]], "parts": [{"text": message["content"]}]}
            for message in messages
            if message["role"] != "system"
        ]

        system_messages = [
            message["content"] for message in messages if message["role"] == "system"
        ]
        system_instructions = (
            {
                "systemInstruction": {
                    "role": "system",
                    "parts": [{"text": message} for message in system_messages],
                }
            }
            if system_messages
            else {}
        )

        for _ in range(retry_counter):
            try:
                if not self.has_started:
                    self.first_time = time.time()
                    self.has_started = True
                async with session.post(
                    self.endpoint_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.auth_token}",
                    },
                    json={
                        "contents": content_messages,
                        **system_instructions,
                        "safetySettings": [
                            {"category": category, "threshold": "BLOCK_NONE"}
                            for category in [
                                "HARM_CATEGORY_HARASSMENT",
                                "HARM_CATEGORY_HATE_SPEECH",
                                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "HARM_CATEGORY_CIVIC_INTEGRITY",
                            ]
                        ],
                        "generationConfig": {**self.generation_args},  # type: ignore
                    },
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        PromptStatistics.log_response(response_json, response_type="gemini")

                        try:
                            completion = ChatCompletion(**response_json)
                            return completion
                        except Exception as err:
                            print(json.dumps(response_json, indent=2))
                            raise err
                    else:
                        seconds_since_start = (time.time() - self.first_time) % 60
                        print(
                            f"Error: {response.status} at {time.strftime('%X')} with {60 - seconds_since_start} wait."
                        )
                        await asyncio.sleep(60 - seconds_since_start)
            except Exception as e:
                print(f"Aiohttp Error: {e}")
                await asyncio.sleep(60)
        raise Exception("Error in Gemini API.")

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """

        # If contexts are the same we don't need to recompute each time
        grouped_inputs = [
            ContextContinuations(k, [el[1] for el in g])
            for k, g in itertools.groupby(inputs, key=lambda x: x[0])
        ]

        if self.has_started is False:
            self.generation_args["responseLogprobs"] = True
            self.generation_args["logprobs"] = 5  # 5 is maximum for gemini
            self.generation_args["maxOutputTokens"] = 5
        results: list = []

        for pair_batch in tqdm(
            chunks(grouped_inputs, self.batch_size), file=sys.stdout, ncols=120, leave=False
        ):
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

        results = list(itertools.chain.from_iterable(results))
        return results

    async def _get_logprobs(
        self, session: ClientSession, pair: ContextContinuations
    ) -> List[Tuple[float, bool]]:
        """Extracts log-probabilities from a given context and its continuations.

        Returns:
            Tuple[float, bool]: (log-likelihood, is-exact-match) pair
        """
        # We only check if the main letter is correct i.e. A or B
        continuations = [c.lstrip()[0] for c in pair.continuations]

        retry_counter = 4
        for _ in range(retry_counter):
            try:
                if not self.has_started:
                    self.first_time = time.time()
                    self.has_started = True
                async with session.post(
                    self.endpoint_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.auth_token}",
                    },
                    json={
                        "contents": {"role": "user", "parts": [{"text": pair.context}]},
                        "safetySettings": [
                            {"category": category, "threshold": "BLOCK_NONE"}
                            for category in [
                                "HARM_CATEGORY_HARASSMENT",
                                "HARM_CATEGORY_HATE_SPEECH",
                                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "HARM_CATEGORY_CIVIC_INTEGRITY",
                            ]
                        ],
                        "generationConfig": {**self.generation_args},
                    },
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        PromptStatistics.log_response(response_json, response_type="gemini")
                        completion = ChatCompletion(**response_json)

                        answers = [(-np.inf, False)] * len(continuations)
                        for i, element in enumerate(completion.candidates[0].logprobsResult.topCandidates[0].candidates):  # type: ignore
                            if element.token in continuations and i == 0:
                                answers[continuations.index(element.token)] = (
                                    element.logProbability,
                                    True,
                                )
                            elif element.token in continuations:
                                answers[continuations.index(element.token)] = (
                                    element.logProbability,
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
        raise Exception("Error in VertexAI API.")
