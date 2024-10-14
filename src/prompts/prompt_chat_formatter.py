#    Copyright 2018- The Hugging Face team. All rights reserved.
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
#
#    Source: https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1684
#    Modifications: Compl-AI Team

from functools import lru_cache
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer

from src.prompts.prompt_base import BaseChatFormatter
from src.prompts.prompt_conversation import Conversation
from src.prompts.prompt_formatters import HFPromptConfig


# https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1684
class ChatFormatter(BaseChatFormatter):
    def __init__(
        self,
        chat_template: str,
        special_tokens_map: dict[str, str] = {},
        add_generation_prompt: Optional[bool] = None,
    ):
        self.chat_template = chat_template
        self.special_tokens_map = special_tokens_map
        self.add_generation_prompt = add_generation_prompt

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], Conversation],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
        determine the format and control tokens to use when converting.

        Args:
            conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]], "Conversation"]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            chat_template (str, *optional*): A Jinja template to use for this conversion. If
                this is not passed, the model's default chat template will be used instead.
            add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                the start of an assistant message. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
            **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

        Returns:
            `Union[List[str], str]`: A string or list of strings representing the input conversations.
        """
        if self.add_generation_prompt:
            add_generation_prompt = add_generation_prompt or self.add_generation_prompt

        # Compilation function uses a cache to avoid recompiling the same template
        compiled_template = self._compile_jinja_template(self.chat_template)

        conversations: Union[List[List[Dict[str, str]]], List[Conversation]]
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            assert not isinstance(conversation, Conversation)
            conversations = conversation  # type: ignore
            is_batched = True
        else:
            conversations = [conversation]  # type: ignore
            is_batched = False

        rendered = []
        template_kwargs = {
            **self.special_tokens_map,
            **kwargs,
        }  # kwargs overwrite special tokens if both are present
        for chat in conversations:
            if hasattr(chat, "messages"):
                # Indicates it's a Conversation object
                chat = chat.messages
            rendered_chat = compiled_template.render(
                messages=chat, add_generation_prompt=add_generation_prompt, **template_kwargs
            )
            rendered.append(rendered_chat)

        if not is_batched:
            rendered = rendered[0]

        return rendered

    @lru_cache
    def _compile_jinja_template(self, chat_template):
        try:
            import jinja2  # noqa: F401
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ImportError("apply_chat_template requires jinja2 to be installed.")

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)


# This class serves as null object to signal that there wasn't any model specific template.
# Based on the null pattern.
class DefaultChatFormatter(ChatFormatter):
    pass


class DummyChatFormatter(BaseChatFormatter):
    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], Conversation],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
        conversations: Union[List[List[Dict[str, str]]], List[Conversation]]
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            assert not isinstance(conversation, Conversation)
            conversations = conversation  # type: ignore
            is_batched = True
        else:
            conversations = [conversation]  # type: ignore
            is_batched = False

        rendered_chats = []
        for chat in conversations:
            if hasattr(chat, "messages"):
                # Indicates it's a Conversation object
                chat = chat.messages
            rendered_chat = ""
            for message in chat:
                rendered_chat += message["content"]
            rendered_chats.append(rendered_chat)

        if not is_batched:
            rendered_chats = rendered_chats[0]  # type: ignore

        return rendered_chats


class ChatFormatterLight(BaseChatFormatter):
    def __init__(self, config: Union[HFPromptConfig, None] = None):
        """A light version of the chat formatter that uses only Huggingface tokenizers

        Args:
            config (Union[HFPromptConfig, None], optional): Everything that is not a HFPromptConfig is not formatted. Defaults to None.
        """
        self.config = config
        self.tokenizer = None
        self.add_generation_prompt = None
        if config:
            self.tokenizer = config.tokenizer_name
            self.add_generation_prompt = config.add_generation_prompt

        self.chat_formatter: BaseChatFormatter
        if self.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
            self.chat_formatter = ChatFormatter(
                tokenizer.chat_template, tokenizer.special_tokens_map, self.add_generation_prompt
            )
        else:
            self.chat_formatter = DummyChatFormatter()

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], Conversation],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
        return self.chat_formatter.apply_chat_template(
            conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
