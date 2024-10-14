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
#    Source: https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/pipelines/conversational.py#L22
#    Modifications: Compl-AI Team

import logging
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/pipelines/conversational.py#L22
class Conversation:
    """
    Utility class containing a conversation and its history. The conversation contains several utility functions to manage the addition of new user
    inputs and generated model responses.

    Arguments:
        messages (Union[str, List[Dict[str, str]]], *optional*):
            The initial messages to start the conversation, either a string, or a list of dicts containing "role" and
            "content" keys. If a string is passed, it is interpreted as a single message with the "user" role.
        conversation_id (`uuid.UUID`, *optional*):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.

    Usage:

    ```python
    conversation = Conversation("Going to the movies tonight - any suggestions?")
    conversation.add_message({"role": "assistant", "content": "The Big lebowski."})
    conversation.add_message({"role": "user", "content": "Is it good?"})
    ```"""

    def __init__(self, messages: List[Dict[str, str]], conversation_id: Optional[uuid.UUID] = None):
        if not conversation_id:
            conversation_id = uuid.uuid4()

        self.uuid = conversation_id
        self.messages = messages

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.uuid == other.uuid or self.messages == other.messages

    def add_message(self, message: Dict[str, str]):
        if not set(message.keys()) == {"role", "content"}:
            raise ValueError("Message should contain only 'role' and 'content' keys!")
        if message["role"] not in ("user", "assistant", "system"):
            raise ValueError("Only 'user', 'assistant' and 'system' roles are supported for now!")
        self.messages.append(message)

    def __iter__(self):
        for message in self.messages:
            yield message

    def __getitem__(self, item):
        return self.messages[item]

    def __setitem__(self, key, value):
        self.messages[key] = value

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        """
        Generates a string representation of the conversation.

        Returns:
            `str`:

        Example:
            Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user: Going to the movies tonight - any suggestions?
            bot: The Big Lebowski
        """
        output = f"Conversation id: {self.uuid}\n"
        for message in self.messages:
            output += f"{message['role']}: {message['content']}\n"
        return output
