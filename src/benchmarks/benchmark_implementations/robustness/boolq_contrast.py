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

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData
from src.data.hf_data import HFDataConfig, WithKShot
from src.prompts.prompt_chat_formatter import ChatFormatterLight


class BoolQContrastData(BaseData):
    def __init__(self, data_context: BaseDataContext) -> None:
        data_config = data_context.get_data_config()
        assert isinstance(data_config, HFDataConfig)
        self.config = data_config
        self.prompt_formatter = ChatFormatterLight()

    def retrieve_data(self, is_for_kshot=False) -> dict:
        path = Path(__file__).resolve().parent / "boolq_data"
        with open(path / "boolq_perturbed.json", "r") as f:
            data = json.load(f)

        return data["data"][1:]

    def normalize_input(self, input: Dict[str, Any]) -> Optional[Dict]:
        # Extract original question and answer
        original_question = input["question"]
        original_question = original_question.strip().capitalize()
        assert original_question[-1] != "?"
        original_question += "?"
        original_answer = input["answer"]
        # Extract contrasting questions and answers
        questions_list = [
            item["perturbed_q"]
            for item in input["perturbed_questions"]
            if original_answer != item["answer"]
        ]
        answers_list = [
            item["answer"]
            for item in input["perturbed_questions"]
            if original_answer != item["answer"]
        ]
        if not questions_list or questions_list[0] == "":
            return None
        # Sample contrast question and answer
        index: int = random.choice(range(len(questions_list)))
        contrast_question: str = questions_list[index]
        contrast_answer: str = answers_list[index]
        contrast_question = contrast_question.strip().capitalize()
        assert contrast_question[-1] != "?"
        # Assemble context
        context = self.prompt_formatter.apply_chat_template(
            [
                {
                    "role": "assistant",
                    "content": "You are a helpful assistant. For each snippet of text, predict the Sentiment by selecting from the option 'A' or 'B'. The answer should be exactly 'A' or 'B'.",
                },
                {
                    "role": "user",
                    "content": input["paragraph"]
                    + "\n"
                    + "Question: "
                    + original_question
                    + "\n"
                    + "Answer: ",
                },
                {"role": "assistant", "content": original_answer},
                {
                    "role": "user",
                    "content": "\n" + "Question: " + contrast_question + "\n" + "Answer: ",
                },
            ]
        )

        return {
            "input": context,
            "labels": 1 if contrast_answer == "TRUE" else 0,
            "choices": ["FALSE", "TRUE"],
        }

    def normalize_data(self, data):
        return [self.normalize_input(row) for row in data if self.normalize_input(row) is not None]

    def debug_pre_process_data(self, data):
        # def gen_from_iterable_dataset(iterable_ds):
        #        yield from iterable_ds

        if isinstance(data, list):
            return data[: self.config.subset_size]

    def get_data(self) -> WithKShot:
        normal = self.retrieve_data()

        normal = self.normalize_data(normal)
        if self.config.debug:
            normal = self.debug_pre_process_data(normal)
            k_shot = None

        if self.config.k_shot_split:
            k_shot = self.retrieve_data(is_for_kshot=True)

            k_shot = self.normalize_data(k_shot)[: self.config.k_shot]

        return WithKShot(normal=normal, k_shot=k_shot)  # type: ignore
