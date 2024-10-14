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

import pandas as pd
from datasets import Dataset

from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData
from src.prompts.prompt_formatters import PromptConfig


class TensortrustDataConfig(DataConfig):
    pass


class TensortrustData(BaseData):
    def __init__(self, data_context: BaseDataContext):
        self.data_context = data_context
        self.config = data_context.get_data_config()
        self.subset_size = self.config.subset_size
        self.prompt_config = self.config.prompt_config or PromptConfig()
        self.prompt_formatter = data_context.get_prompt_formatter()

    def get_citation(self):
        """
        @misc{toyer2023tensor,
            title={{Tensor Trust}: Interpretable Prompt Injection Attacks from an Online Game},
            author={Toyer, Sam and Watkins, Olivia and Mendes, Ethan Adrian and Svegliato, Justin and Bailey, Luke and Wang, Tiffany and Ong, Isaac and Elmaaroufi, Karim and Abbeel, Pieter and Darrell, Trevor and Ritter, Alan and Russell, Stuart},
            year={2023},
            journal={arXiv preprint arXiv:2311.01011},
            url={https://arxiv.org/pdf/2311.01011.pdf}
        }
        """

    def get_prompts_and_access_codes(self) -> Dataset:
        """
        Retrieves prompts and access codes from the hijacking dataset.

        Returns:
            Dataset: A dataset containing prompts and access codes.
                Prompts are formatted using appropriate chat template.
        """

        hijacking_dataset = pd.read_json(
            "https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/main/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl",
            lines=True,
        ).set_index("sample_id")

        print(len(hijacking_dataset.index))

        def generate_prompt(row):
            """
            Take the pre_prompt, the attack attempt and the post_prompt and create new prompt
            """
            row |= {
                "prompt": self.prompt_formatter.format_chat_prompt(
                    [
                        {
                            "role": "user",
                            "content": row["pre_prompt"] + row["attack"] + row["post_prompt"],
                        }
                    ]
                ),
                "access_code": row["access_code"],
            }
            return row

        prompt_data = hijacking_dataset.to_dict(orient="records")
        prompt_data = [generate_prompt(row) for row in prompt_data]

        if self.config.debug:
            return prompt_data[: self.subset_size]
        else:
            return prompt_data
