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

import json
import random
import re
from pathlib import Path
from typing import Any

from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import match_case
from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import Perturbation


# The implementation below is based on the following list of common misspellings:
# https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines
class MisspellingPerturbation(Perturbation):
    """Replaces words with common misspellings, from a list of common misspellings.

    Perturbation example:

    **Input:**
        Already, the new product is not available.

    **Output:**
        Aready, the new product is not availible. # spellchecker:disable-line
    """

    name: str = "misspelling"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the misspelling perturbation.

        Args:
            prob: Probability between [0,1] of perturbing a word to a
                common misspelling (if we have a common misspelling for the word).
        """
        super().__init__(params)
        self.prob: float = self.params["prob"]
        self.correct_to_misspelling = self.load_misspellings_data()
        self.misspelling_pattern = re.compile(
            r"\b({})\b".format("|".join(self.correct_to_misspelling.keys()))
        )

    def load_misspellings_data(self) -> dict[str, list[str]]:
        misspellings_file = Path(__file__).parent.parent / "correct_to_misspelling.json"

        with open(misspellings_file, "r") as f:
            correct_to_misspelling: dict[str, list[str]] = json.load(f)
        return correct_to_misspelling

    @property
    def default_params(self) -> dict[str, Any]:
        return {"prob": 1.0}

    def perturb(self, text: str) -> str:
        def misspell(match: re.Match) -> str:
            word = match.group(1)
            if random.random() < self.prob:  # nosec B311
                misspelled_word = random.choice(self.correct_to_misspelling[word])  # nosec B311
                return match_case(word, misspelled_word)
            else:
                return word

        return self.misspelling_pattern.sub(misspell, text)
