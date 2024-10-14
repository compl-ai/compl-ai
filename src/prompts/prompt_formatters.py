# mypy: disable-error-code="no-redef"

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


from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Union

import jinja2
from pydantic import BaseModel as PBM
from pydantic import Field

from src.prompts.prompt_base import BaseChatFormatter


class HFPromptConfig(PBM):
    tokenizer_name: Optional[str] = Field(None, description="Name of the tokenizer to use")
    add_generation_prompt: bool = Field(
        default=True,
        description="Whether to add a generation prompt to the end of the chat template",
    )


# For compatibility with old code usage
PromptConfig = HFPromptConfig


class Role(IntEnum):
    SYSTEM = 2
    ASSISTANT = 1
    USER = 0


def str_to_role(role: str) -> Role:
    mapping = {"user": Role.USER, "system": Role.SYSTEM, "assistant": Role.ASSISTANT}
    return mapping[role]


def role_to_str(role: Role) -> str:
    mapping = {Role.USER: "user", Role.SYSTEM: "system", Role.ASSISTANT: "assistant"}
    return mapping[role]


@dataclass
class ConversationEntry:
    message: str
    role: Role

    def __post_init__(self):
        self.role = role_to_str(self.role)  # type: ignore


@dataclass
class ConversationPrompt:
    messages: list[ConversationEntry]


def conv_prompt_to_list(self) -> list[dict]:
    return [{"role": msg.role, "content": msg.message} for msg in self.messages]


class DefaultPromptFormatter:
    """
    DefaultPromptFormatter is a class that formats prompts for different prompt types.

    Args:
        prompt_config (HFPromptConfig): The configuration for the prompt.
        chat_formatter (BaseChatFormatter): The chat formatter to apply to the prompt.
    """

    def __init__(
        self,
        prompt_config: HFPromptConfig,
        chat_formatter: BaseChatFormatter,
    ):
        self.prompt_config = prompt_config
        self.chat_formatter = chat_formatter

    def get_chat_formatter(self):
        return self.chat_formatter

    def get_prompt_config(self):
        return self.prompt_config

    def format_chat_prompt(
        self,
        conversation_prompt: Union[List[Dict[str, str]], ConversationPrompt],
        add_generation_prompt: bool = False,
    ) -> str:  # noqa F811
        # Format expected by HF library
        if isinstance(conversation_prompt, ConversationPrompt):
            chat: list[dict[str, str]] = [{"role": msg.role, "content": msg.message} for msg in conversation_prompt.messages]  # type: ignore
        else:
            chat = conversation_prompt
        assembled_chat: str
        generation_prompt = getattr(self.prompt_config, "add_generation_prompt", False)
        add_generation_prompt = generation_prompt or add_generation_prompt
        try:
            assembled_chat = self.chat_formatter.apply_chat_template(chat, add_generation_prompt=add_generation_prompt)  # type: ignore

        except jinja2.exceptions.TemplateError:
            assert chat[0]["role"] == "system"
            # Dirty hack, exception is very likely caused by system role, merge it with user role
            system_msg = chat[0]
            user_msg = chat[1]
            rest_chat = chat[2:] if len(chat) > 2 else []
            user_msg["content"] = system_msg["content"] + "\n\n" + user_msg["content"]
            new_chat = [user_msg] + rest_chat

            assembled_chat = self.chat_formatter.apply_chat_template(  # type: ignore
                new_chat, add_generation_prompt=add_generation_prompt
            )

        return assembled_chat
