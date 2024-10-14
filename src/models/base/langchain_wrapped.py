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

from typing import Any, List, Optional

from base_model import BaseModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class WrappedLLM(LLM):
    """
    Create wrapper to interface with Langchain library.
    According to: https://python.langchain.com/docs/modules/model_io/models/llms/custom_llm
    """

    base_model: BaseModel

    @property
    def _llm_type(self) -> str:
        return "custom"

    def __init__(self, base_model: BaseModel):
        self.base_model = base_model

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.base_model.generate(prompt)
