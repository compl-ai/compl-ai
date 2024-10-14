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

from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseDataContext
from src.models.proxy_model import ProxyModel
from src.prompts.prompt_formatters import DefaultPromptFormatter, PromptConfig


class DataContext(BaseDataContext):
    """This is what each benchmark gets and contains methods to get all the relevant objects and information dynamically"""

    def __init__(self, model: ProxyModel, config: DataConfig) -> None:
        self.model = model
        self.config = config

        self.prompt_config = self.config.prompt_config or PromptConfig()
        chat_formatter = self.model.base_model.get_chat_formatter()
        self.prompt_formatter = DefaultPromptFormatter(
            self.prompt_config, chat_formatter=chat_formatter
        )

    def get_data_config(self) -> DataConfig:
        return self.config

    def get_prompt_formatter(self) -> DefaultPromptFormatter:
        return self.prompt_formatter

    def get_logger_name(self) -> str:
        return f"data-{self.config.type}"
