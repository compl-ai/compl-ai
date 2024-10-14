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
from typing import Any, Dict, Iterable, Optional

from datasets import Dataset

from src.configs.base_modifier_config import ModifierConfig
from src.data.base_data import BaseData
from src.data.hf_data import WithKShot
from src.utils.general import fill_in_defaults


class BaseModifier(ABC):
    """
    Base class for modifiers.

    This class defines the common interface and behavior for all modifiers.

    Attributes:
        params (Dict[str, Any]): The parameters the modifier can actually modify.

    Methods:
        __init__(self, config: Optional[ModifierConfig] = None): Initializes the modifier with the given configuration.
        default_params(self) -> Dict[str, Any]: Returns the default parameters for the modifier.
        perturb(self, text: str) -> str: Applies the modification to the given text.
        modify(self, *args): Modifies the first argument using the perturb method.
        modify_dataset(self, dataset: Dataset) -> Dataset: Modifies the input dataset using the perturb method.
        modify_iterator(self, iter_data: Iterable) -> Iterable: Modifies the input iterator using the perturb method.
    """

    def __init__(self, config: Optional[ModifierConfig] = None):
        if config:
            self.params = fill_in_defaults(config.params, self.default_params)
        else:
            self.params = self.default_params

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def perturb(self, text: str) -> str:
        pass

    def modify(self, *args):
        return self.perturb(args[0])

    def modify_dataset(self, dataset: Dataset) -> Dataset:
        # Avoid circular import
        from src.utils.batch_utils import map_batch_wrapper

        return map_batch_wrapper(dataset, self.modify, only_input=True)

    def modify_iterator(self, iter_data: Iterable) -> Iterable:
        for data_row in iter_data:
            yield self.modify(data_row)


class ModifierDataProvider(BaseData):
    """
    A class that provides modified data based on a given modifier and old data provider.

    Args:
        modifier (BaseModifier): The modifier to be applied to the data.
        old_data_provider (BaseData): The old data provider that contains the original data.

    Attributes:
        old_data (BaseData): The original data obtained from the old data provider.
        modifier (BaseModifier): The modifier to be applied to the data.

    Methods:
        get_input(data): Returns an iterator that yields the input values from the given data.
        get_adjusted(data): Returns an iterator that yields the adjusted data by applying the modifier to the input values.
        get_data(): Returns the modified data.

    """

    def __init__(self, modifier: BaseModifier, old_data_provider: BaseData):
        assert hasattr(
            old_data_provider, "get_data"
        ), "Old data provider must have a get_data method."
        self.old_data = old_data_provider.get_data()
        self.modifier = modifier

    def get_adjusted(self, data: Dataset):
        def modify_input(row: dict) -> dict:
            row["question"] = self.modifier.perturb(row["question"])
            return row

        data = data.map(modify_input)
        return data

    def get_data(self):
        normal_perturbed = self.get_adjusted(self.old_data.normal)
        k_shots_perturbed = self.get_adjusted(self.old_data.k_shot)

        return WithKShot(normal=normal_perturbed, k_shot=k_shots_perturbed)
