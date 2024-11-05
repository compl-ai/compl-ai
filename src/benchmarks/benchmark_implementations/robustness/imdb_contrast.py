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
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData
from src.data.hf_data import HFDataConfig, WithKShot
from src.prompts.prompt_chat_formatter import ChatFormatterLight


class IMDBContrastData(BaseData):
    DATA_FILES = [
        "dev_contrast.tsv",
        "dev_original.tsv",
        "test_contrast.tsv",
        "test_original.tsv",
    ]

    def __init__(self, data_context: BaseDataContext):
        data_config = data_context.get_data_config()
        assert isinstance(data_config, HFDataConfig)
        self.config = data_config
        self.prompt_formatter = ChatFormatterLight()

    def retrieve_data(self, is_for_kshot=False) -> dict:
        data = {}
        for name in self.DATA_FILES:
            path = Path(__file__).resolve().parent / "imdb_data"
            sub_data = pd.read_csv(path / name, sep="\t")
            data[name] = sub_data.to_dict(orient="records")

        return data

    def preprocess_data(self, data: dict) -> List[dict]:
        preprocessed_data: list = []

        def pair_and_transform(data: dict, original_file: str, contrast_file: str) -> list:
            output: list = []
            for original, contrast in zip(data[original_file], data[contrast_file]):
                output.append(
                    {
                        "Sentiment_Original": original["Sentiment"],
                        "Text_Original": original["Text"],
                        "Sentiment_Contrast": contrast["Sentiment"],
                        "Text_Contrast": contrast["Text"],
                    }
                )
            return output

        preprocessed_data.extend(pair_and_transform(data, "dev_original.tsv", "dev_contrast.tsv"))
        preprocessed_data.extend(pair_and_transform(data, "dev_original.tsv", "dev_contrast.tsv"))
        return preprocessed_data

    def normalize_input(self, input: Dict[str, Any]) -> Dict:
        context: str = self.prompt_formatter.apply_chat_template(  # type: ignore
            [
                {
                    "role": "assistant",
                    "content": "You are a helpful assistant. For each snippet of text, predict the Sentiment by selecting from the option 'A' or 'B'. The answer should be exactly 'A' or 'B'.",
                },
                {"role": "user", "content": input["Text_Original"] + "\n" + "Sentiment: "},
                {
                    "role": "assistant",
                    "content": input["Sentiment_Original"],
                },
                {"role": "user", "content": "\n\n" + input["Text_Contrast"] + "\n" + "Sentiment: "},
            ]
        )

        labels: int = 1 if input["Sentiment_Contrast"] == "Positive" else 0

        return {
            "input": context,
            "labels": labels,
            "choices": ["Negative", "Positive"],
        }

    def normalize_data(self, data):
        return [self.normalize_input(row) for row in data]

    def debug_pre_process_data(self, data):
        # def gen_from_iterable_dataset(iterable_ds):
        #        yield from iterable_ds

        if isinstance(data, list):
            return data[: self.config.subset_size]

    def get_data(self) -> WithKShot:
        normal = self.retrieve_data()
        normal = self.preprocess_data(normal)  # type: ignore

        normal = self.normalize_data(normal)
        if self.config.debug:
            normal = self.debug_pre_process_data(normal)
        k_shot = None

        if self.config.k_shot_split:
            k_shot = self.retrieve_data(is_for_kshot=True)
            k_shot = self.preprocess_data(k_shot)  # type: ignore

            k_shot = self.normalize_data(k_shot)[: self.config.k_shot]

        return WithKShot(normal=normal, k_shot=k_shot)  # type: ignore
