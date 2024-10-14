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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from src.prompts.prompt_conversation import Conversation


class BaseChatFormatter(ABC):
    """Base class for chat formatters."""

    @abstractmethod
    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], Conversation],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        **kwargs
    ) -> Union[str, List[str]]:
        pass
