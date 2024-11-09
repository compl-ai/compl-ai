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
import sys
import time
from typing import Any, List, Tuple, Union

from aiohttp import ClientSession
from tqdm import tqdm

from secret import GEMINI_API_KEY
from src.configs.base_model_config import ModelConfig
from src.models.base.base_model import BaseModel, Message

from .utils import PromptStatistics, RateLimit, chunks

logging.getLogger("asyncio").setLevel(logging.WARNING)


# Requests per minute
RPM = 1000


def _get_response_text(response: dict) -> str:
    if "promptFeedback" in response and "blockReason" in response["promptFeedback"]:
        return "I cannot provide a response."
    candidate = response["candidates"][0]
    if candidate["finishReason"] == "OTHER":
        return "I cannot provide a response."

    try:
        return candidate["content"]["parts"][0]["text"]
    except Exception as err:
        print(response)
        raise err


class GoogleAIModel(BaseModel):
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

    def _init_generation_args(self, config: ModelConfig) -> None:
        self.generation_args = {"maxOutputTokens": config.max_gen_toks}
        if config.generation_args is not None:
            for key, value in config.generation_args.items():
                if key == "top_p":
                    self.generation_args["topP"] = value
                elif key == "temperature":
                    self.generation_args["temperature"] = value
                elif key == "frequency_penalty":
                    raise ValueError("Unsupported argument 'frequency_penalty' for GoogleAI API.")
                elif key == "presence_penalty":
                    raise ValueError("Unsupported argument 'presence_penalty' for GoogleAI API.")
                elif key == "num_return_sequences":
                    raise ValueError(
                        "Unsupported argument 'num_return_sequences' for GoogleAI API."
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
            self.generation_args["maxOutputTokens"] = kwargs["max_length"]
        if "top_p" in kwargs:
            self.generation_args["topP"] = kwargs["top_p"]
        if "temperature" in kwargs:
            self.generation_args["temperature"] = kwargs["temperature"]
        if "num_return_sequences" in kwargs:
            raise ValueError("Unsupported argument 'num_return_sequences' for GoogleAI API.")

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
                    return [_get_response_text(response) for response in responses]

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
            Any: Gemini API response
        """
        retry_counter = 4

        ROLES = {"system": "system", "user": "user", "assistant": "model"}

        content_messages = [
            {"role": ROLES[message["role"]], "parts": [{"text": message["content"]}]}
            for message in messages
            if message["role"] != "system"
        ]

        content = {"contents": content_messages}
        system_messages = [
            message["content"] for message in messages if message["role"] == "system"
        ]
        if system_messages:
            content["system_instruction"] = {  # type: ignore
                "parts": [{"text": message} for message in system_messages],
            }

        for _ in range(retry_counter):
            try:
                if not self.has_started:
                    self.first_time = time.time()
                    self.has_started = True
                async with session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={GEMINI_API_KEY}",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        **content,
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
        raise Exception("Error in Gemini API.")
