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

import importlib.resources
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from llm_rules import Message, Role, models, scenarios
from llm_rules.models.base import BaseModel as LLMRulesBaseModel
from llm_rules.models.base import build_prompt
from llm_rules.scenarios import SCENARIOS
from pydantic import BaseModel as PBM
from pydantic import Field, ValidationError, field_validator, model_validator
from tqdm import tqdm

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel
from src.models.base.hf_model import HFLM
from src.models.base.utils import PromptStatistics
from src.models.proxy_model import ProxyModel
from src.prompts.prompt_formatters import DefaultPromptFormatter


def messages_to_hf_format(messages: List[Message]) -> List[Dict[str, str]]:
    def map_type(role: Role) -> str:
        if role == Role.SYSTEM:
            return "system"
        elif role == Role.USER:
            return "user"
        elif role == Role.ASSISTANT:
            return "assistant"
        assert False, f"Unknown role '{role}'"

    return [{"role": map_type(m.role), "content": m.content} for m in messages]


def get_data(data_str):
    """
    Retrieves the contents of a data file.

    Args:
        data_str (str): The name of the data file to retrieve.

    Returns:
        str: The contents of the data file.
    """
    contents = importlib.resources.files("llm_rules").joinpath(Path("data") / data_str).read_text()
    return contents


class ModelWrapper(LLMRulesBaseModel):
    def __init__(self, model):
        self.model = model

    def __call__(self, messages: List[Message], api_key: Optional[str] = None):
        """
        Generate a response based on the given messages.

        Args:
            messages (List[Message]): A list of messages exchanged in the conversation.
            api_key (Optional[str]): An optional API key.

        Returns:
            str: The generated response.

        """

        prompt = build_prompt(messages)
        result = self.model.generate(prompt)
        return result


class LanguageGameDataConfig(DataConfig):
    scenario: Optional[str] = Field(default="SimonSays")

    @field_validator("scenario")
    def validate_scenario(cls, val):
        if val not in SCENARIOS.keys():
            raise ValidationError("Scenario doesn't exist")
        return val


class LLMRulesConf(PBM):
    scenario: Optional[str] = Field(default="SimonSays")
    category: Optional[str] = Field(default=None)
    use_system_instructions: bool = Field(
        True, description="Whether to rely on the default system instruction"
    )
    system_message: Optional[str] = Field(default=None)
    prefix_message: Optional[str] = Field(default=None)
    suffix_message: Optional[str] = Field(default=None)

    @field_validator("system_message")
    def validate_system_message(cls, val):
        if val not in models.SYSTEM_MESSAGES.keys():
            raise ValidationError("Chosen system_message doesn't exist!")

    @model_validator(mode="after")
    def validate_conf(self):
        """Either scenario or system_message must be set"""
        if self.scenario not in SCENARIOS:
            raise ValidationError("No scenario provided!")
        return self


class LanguageGameConfig(BenchmarkConfig):
    llm_rules_conf: LLMRulesConf = Field(default_factory=LLMRulesConf)


class LanguageGameData(BaseData):
    def __init__(self, data_context: BaseDataContext):
        data_config = data_context.get_data_config()
        assert isinstance(data_config, LanguageGameDataConfig)

        self.subset_size = data_config.subset_size
        self.category = data_config.scenario
        self.data_config = data_config

    def get_data(self):
        """
        Retrieves data from a JSON file based on the specified category.
        A link with the correct hash is used to ensure reproducibility.

        Returns:
            list: A list of dictionaries representing the data records.
        """

        path = Path(__file__).resolve().parent / "llm_rules_data"
        data = pd.read_json(
            path / f"manual/{self.category}.jsonl",
            lines=True,
        )

        if self.data_config.debug:
            return data.to_dict(orient="records")[: self.subset_size]
        else:
            return data.to_dict(orient="records")

    def get_scenario_class(self):
        """
        Returns the scenario class based on the category.
        This is part of the llm-rules library.

        Returns:
            The scenario class corresponding to the category.
        """
        return SCENARIOS[self.category]


class LanguageGame(BaseBenchmark):
    def __init__(self, context: BaseBenchmarkContext):
        super().__init__(context)
        self.context = context
        self.config = cast(LanguageGameConfig, context.get_benchmark_config())
        self.llm_rules_conf = self.config.llm_rules_conf
        data_provider = context.get_dataset()
        assert isinstance(data_provider, LanguageGameData)
        self.data_provider = data_provider
        self.chat_formatter = context.get_prompt_formatter()

    def run(self, model: BaseModel) -> Dict[str, Any]:
        """
        Runs the benchmark using the provided model.
        For each conversation in the dataset,
        the model evaluates the conversation using the provided scenario.

        Args:
            model (BaseModel): The model to be evaluated.

        Returns:
            Dict[str, any]: A dictionary containing the results of the benchmark.
                The dictionary has two keys:
                    - 'valid_runs': The percentage of valid runs.
                    - 'reasons': A dictionary containing the count of each reason for failure.
        """

        valid_runs: list[bool] = []
        data = self.data_provider.get_data()

        print("Multiturn Goal Hijacking:")
        print(f"└── {self.data_provider.category}: ", len(data))

        PromptStatistics.reset()

        reasons = {}
        for row in tqdm(data, ncols=120, file=sys.stdout):
            messages = Message.unserialize(row["messages"])
            results = evaluate_single(
                self.data_provider.get_scenario_class()(param_dict=row["params"]),
                messages,
                model,  # type: ignore
                self.llm_rules_conf.use_system_instructions,
                self.llm_rules_conf.system_message,
                self.llm_rules_conf.prefix_message,
                self.llm_rules_conf.suffix_message,
                self.chat_formatter,
            )
            reason = results[0].reason
            if reason not in reasons:
                reasons[reason] = 0
            else:
                reasons[reason] += 1

            valid_runs.append(results[0].passed)

        PromptStatistics.dump("Multiturn Goal Hijacking")

        return {"valid_runs": sum(valid_runs) / len(valid_runs), "reasons": reasons}


# Taken from https://github.com/normster/llm_rules/blob/main/evaluate.py
def evaluate_single(
    scenario: scenarios.BaseScenario,
    user_messages: List[Message],
    model: ProxyModel,
    use_system_instructions: bool,
    system_message: Optional[str],
    prefix_message: Optional[str],
    suffix: Optional[str],
    prompt_formatter: DefaultPromptFormatter,
):
    """
    Evaluates a single conversation scenario from llm-rules.
    For each user message in the conversation,
    the model generates a response and the scenario evaluates the conversation.
    If the scenario fails, the function returns the result and the list of messages in the conversation.
    On the other hand, if the scenario passes, the loop continues with the
    next user message in the conversation.

    Args:
        scenario (scenarios.BaseScenario): The conversation scenario to evaluate.
        user_messages (List[Message]): List of user messages in the conversation.
        model (models.BaseModel): The model used for generating assistant responses.
        use_system_instructions (bool): Flag indicating whether to use system instructions.
        system_message (Optional[str]): The system message to use if `use_system_instructions` is True.
        prefix_message (Optional[str]): The prefix message to add to the prompt.
        suffix (Optional[str]): The suffix to add to the first user message.

    Returns:
        Tuple[Result, List[Message]]: A tuple containing the evaluation result and the list of messages in the conversation.
    """

    full_prompt = scenario.prompt
    if prefix_message:
        full_prompt = models.SYSTEM_MESSAGES[prefix_message] + "\n" + full_prompt

    if use_system_instructions:
        messages = [Message(Role.SYSTEM, full_prompt)]
    else:
        system_message = models.SYSTEM_MESSAGES[system_message]
        messages = [
            Message(Role.SYSTEM, system_message),
            Message(Role.USER, full_prompt),
            Message(Role.ASSISTANT, scenario.initial_response),
        ]

    for i, m in enumerate(user_messages):
        if i == 0 and suffix:
            m.content += " " + suffix  # gcg suffixes are applied with a preceding space
        messages.append(m)

        messages_hf = messages_to_hf_format(messages)
        if isinstance(model.base_model, HFLM):
            chat = prompt_formatter.format_chat_prompt(messages_hf)
            response = model.generate(chat)
        else:
            response = model.generate_system([messages_hf])  # type: ignore
        content = response[0].strip()

        messages.append(Message(Role.ASSISTANT, content))

        result = scenario.evaluate(messages, use_system_instructions)

        if not result.passed:
            return result, messages

    return result, messages
