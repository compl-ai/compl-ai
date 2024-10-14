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

from typing import Iterable, TypedDict

from datasets import load_dataset
from pydantic import Field

from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData


class LmsysDataConfig(DataConfig):
    language: str = Field("English", description="Which language to select from the dataset")
    toxic_category: str = Field(
        "hate/threatening",
        description="Which category of toxic content to focus on, see dataset page for further categories",
    )


class LmsysData(BaseData):
    def __init__(self, data_context: BaseDataContext):
        # super().__init__(data_config)
        data_config = data_context.get_data_config()
        assert isinstance(data_config, LmsysDataConfig)
        self.data_config: LmsysDataConfig = data_config
        self.subset_size: int = data_config.subset_size

    def get_citation(self):
        """
        @misc{zheng2023lmsyschat1m,
        title={LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset},
        author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Tianle Li and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zhuohan Li and Zi Lin and Eric. P Xing and Joseph E. Gonzalez and Ion Stoica and Hao Zhang},
        year={2023},
        eprint={2309.11998},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
        }
        """

    class Messages(TypedDict):
        content: str
        role: str

    def generate_toxic_prompt(self, messages: list[Messages], is_bad: list[bool]) -> str:
        complete_prompt = ""
        for message, is_bad_m in zip(messages, is_bad):
            complete_prompt += message["content"]
            if message["role"] == "user" and is_bad_m:
                return complete_prompt

        raise RuntimeError("Data seems to be invalid!")

    def get_bad(self, row: dict):
        is_bad = [
            el["categories"][self.data_config.toxic_category] for el in row["openai_moderation"]
        ]
        messages = row["conversation"]
        toxic_prompt = self.generate_toxic_prompt(messages, is_bad)
        return {"prompt": toxic_prompt}

    def get_prompts(self) -> Iterable[str]:
        dataset_name = "lmsys/lmsys-chat-1m"
        dataset = load_dataset(dataset_name)

        def query_language(example):
            return example["language"] == self.data_config.language

        dataset_english = dataset.filter(query_language)

        def query_harmful(row):
            return row["openai_moderation"][0]["categories"][self.data_config.toxic_category]

        dataset_harmful = dataset_english.filter(query_harmful)

        dataset_harmful = dataset_harmful.map(self.get_bad)
        prompts = dataset_harmful["train"]["prompt"]

        if self.data_config.debug:
            return prompts[: self.subset_size]
        else:
            return prompts

    def get_data(self):
        return self.get_prompts()
