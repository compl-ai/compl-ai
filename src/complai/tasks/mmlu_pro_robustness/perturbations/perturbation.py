# Copyright 2025 LatticeFlow AG
# Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any


def match_case(source_word: str, target_word: str) -> str:
    """Return a version of the target_word where the case matches the source_word.

    Args:
        source_word: The source word.
        target_word: The target word.

    Returns:
        Target word where the case matches the source word
    """
    # Check for all lower case source_word
    if all(letter.islower() for letter in source_word):
        return target_word.lower()
    # Check for all caps source_word
    if all(letter.isupper() for letter in source_word):
        return target_word.upper()
    # Check for capital source_word
    if source_word and source_word[0].isupper():
        return target_word.capitalize()
    return target_word


def fill_in_defaults(
    custom_dict: dict[str, Any], default_dict: dict[str, Any]
) -> dict[str, Any]:
    """Fill in missing keys in custom dict from default dict.

    Args:
        custom_dict: Some dict, possibly from a config.
        default_dict: A dict with default values.

    Returns:
        The custom dict but with missing default values added
    """
    for key, value in default_dict.items():
        if key not in custom_dict:
            custom_dict[key] = value
    return custom_dict


class Perturbation(ABC):
    """Base class for perturbations.

    This class defines the common interface and behavior for all perturbations.

    Attributes:
        params: The parameters the perturbation can actually modify.
    """

    name: str

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        if params:
            self.params = fill_in_defaults(params, self.default_params)
        else:
            self.params = self.default_params

    @property
    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def perturb(self, text: str) -> str:
        pass

    def perturb_batch(self, texts: list[str]) -> list[str]:
        return [self.perturb(text) for text in texts]
