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

import random
import re
from typing import Any

from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import Perturbation


class SpacePerturbation(Perturbation):
    """A simple perturbation that replaces existing spaces with 0-max_spaces spaces.

    As a result, it can potentially merge words."""

    name: str = "space"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.max_spaces = self.params["max_spaces"]
        self.randomize = self.params["randomize"]

    @property
    def default_params(self) -> dict[str, Any]:
        return {"max_spaces": 3, "randomize": True}

    def perturb(self, text: str) -> str:
        # Replace each space with a random number of spaces
        if self.randomize:
            return re.sub(
                r" +",
                lambda x: " " * random.randint(0, self.max_spaces),  # nosec B311
                text,  # nosec B311
            )
        else:
            return text.replace(" ", " " * self.max_spaces)
