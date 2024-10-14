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

import os
from dataclasses import dataclass
from typing import List, Optional

import datasets
from datasets import Dataset, DatasetDict, IterableDataset
from pydantic import Field

from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData


class HFDataConfig(DataConfig):
    path: str = Field(description="Name of the huggingface dataset by which it can be retrieved")
    name: Optional[str] = Field(
        None, description="Category of the benchmark - Commonly used for HF benchmarks"
    )
    split: str = Field(
        "test",
        description="Split of the benchmark - Commonly used for HF benchmarks",
    )
    k_shot_split: Optional[str] = Field(None, description="Split to use for k_shots")
    k_shot: Optional[int] = Field(3, description="How many k_shots to use")


@dataclass
class WithKShot:
    normal: Dataset
    k_shot: Optional[List] = Field(None, description="Contains k_shot dataset")


class HFData(BaseData):
    def __init__(self, data_context: BaseDataContext):
        super().__init__(data_context)
        config = data_context.get_data_config()
        assert isinstance(config, HFDataConfig)
        self.config = config

    def get_citation(self):
        """No citation since this is a general mechanism"""

    def _get_hf_dataset(self, is_for_kshot=False) -> DatasetDict:
        """Get a HF dataset

        Args:
            config (BenchmarkConfig): Config to use
            splits (List[str]): Splits to use

        Raises:
            ValueError: If the dataset is not supported/found

        Returns:
            datasets.Dataset: Corresponding dataset
        """

        config = self.config
        splits = config.split

        streaming_mode = True  # not is_for_kshot

        # Split has to be a str
        split: str = splits[0] if isinstance(splits, list) else splits  # noqa: F841

        # In this case, we get a different split to make sure
        # the final questions didn't already occur in the k-shots examples
        if is_for_kshot:
            assert self.config.k_shot_split
            splits = self.config.k_shot_split

        dataset_dict = datasets.load_dataset(
            config.path,
            name=config.name,
            split=splits,
            streaming=streaming_mode,
            trust_remote_code=True,
        )

        return dataset_dict

    def normalize_data(self, data):
        return data

    def debug_pre_process_data(self, data):
        # def gen_from_iterable_dataset(iterable_ds):
        #        yield from iterable_ds
        if self.config.debug:
            if isinstance(data, IterableDataset):
                seed = os.getenv("PL_GLOBAL_SEED") or 42
                # Only reshuffle is subset size is big enough
                if self.config.subset_size > 10:
                    data = data.shuffle(seed=int(seed))
                return data.take(self.config.subset_size)
            elif isinstance(data, Dataset):
                seed = os.getenv("PL_GLOBAL_SEED") or 42
                data = data.shuffle(seed=seed)
                num_rows = data.num_rows
                return data.select(range(min(num_rows, self.config.subset_size)))
            else:
                raise NotImplementedError

        else:
            if isinstance(data, IterableDataset):
                return data
            elif isinstance(data, Dataset):
                num_rows = data.num_rows
                return data.select(range(num_rows))
            else:
                raise NotImplementedError

    def get_data(self) -> WithKShot:
        normal = self._get_hf_dataset()
        normal = self.normalize_data(normal)
        normal = self.debug_pre_process_data(normal)
        k_shot = None

        if self.config.k_shot_split:
            k_shot = self._get_hf_dataset(is_for_kshot=True)
            k_shot = list(self.normalize_data(k_shot).take(self.config.k_shot))

        return WithKShot(normal=normal, k_shot=k_shot)
