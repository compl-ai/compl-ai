# Copyright 2024 LatticeFlow AG
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

from typing import Any

from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import Perturbation


class LowerCasePerturbation(Perturbation):
    """Simple perturbation turning input and references into lowercase."""

    name: str = "lowercase"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

    @property
    def default_params(self) -> dict[str, Any]:
        return {}

    def perturb(self, text: str) -> str:
        return text.lower()
