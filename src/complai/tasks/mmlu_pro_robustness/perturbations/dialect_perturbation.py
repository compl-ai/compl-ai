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
from typing import Any

from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import match_case
from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import Perturbation
from complai.tasks.utils.constants import CACHE_DIR
from complai.tasks.utils.download import ensure_dialect_mapping


class DialectPerturbation(Perturbation):
    """Individual fairness perturbation for dialect."""

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "dialect"

    should_perturb_references: bool = True

    """ Dictionary mapping dialects to one another """
    SAE = "SAE"  # American standard english
    AAVE = "AAVE"  # Afro-american vernacular english

    FILENAME = f"{SAE}_to_{AAVE}_mapping.json"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        """Initialize the dialect perturbation.

        If mapping_file_path is not provided, (source_class, target_class)
        should be ("SAE", "AAVE").

        Args:
            prob: Probability of substituting a word in the original class with
                a word in the target class given that a substitution is
                available.
            source_class: The source dialect that will be substituted with
                the target dialect. Case-insensitive.
            target_class: The target dialect.
            mapping_file_path: The absolute path to a file containing the
                word mappings from the source dialect to the target dialect in
                a json format. The json dictionary must be of type
                dict[str, List[str]]. Otherwise, the default dictionary in
                self.MAPPING_DICTS for the provided source and target classes
                will be used, if available.
        """
        # Assign parameters to instance variables
        assert 0 <= self.params["prob"] <= 1
        self.prob = self.params["prob"]
        self.source_class: str = self.params["source_class"].upper()
        self.target_class: str = self.params["target_class"].upper()

        self.mapping_dict: dict[str, list[str]] = self.load_mapping_dict()

        # Pattern capturing any occurrence of the given words in the text, surrounded by characters other than
        # alphanumeric characters and '-'. We use re.escape since the words in our dictionary may
        # contain special RegEx characters.
        words = [re.escape(w) for w in self.mapping_dict.keys()]
        words_string = "|".join(words)
        self.pattern = f"[^\\w-]({words_string})[^\\w-]"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "prob": 1.0,
            "source_class": "SAE",
            "target_class": "AAVE",
            "mapping_file_path": None,
        }

    def load_mapping_dict(self) -> dict[str, list[str]]:
        mapping_file = CACHE_DIR / self.FILENAME

        # Ensure the file exists
        ensure_dialect_mapping(mapping_file)

        with open(mapping_file, "r") as f:
            return json.load(f)

    def perturb(self, text: str) -> str:
        """Substitute the source dialect in text with the target dialect."""

        # Substitution function
        def sub_func(m: re.Match) -> str:
            match_str = m.group(0)  # The full match (e.g. " With ", " With,", " With.")
            word = m.group(1)  # Captured group (e.g. "With")
            if random.random() < self.prob:  # nosec B311
                synonyms = self.mapping_dict[word.lower()]
                synonym = random.choice(synonyms)  # nosec B311 Synonym (e.g. "wit")
                synonym = match_case(
                    word, synonym
                )  # Synonym with matching case (e.g. "Wit")
                match_str = match_str.replace(
                    word, synonym
                )  # Synonym placed in the matching group (e.g. " Wit ", " Wit,", " Wit.")
            return match_str

        # Execute the RegEx
        return re.sub(self.pattern, sub_func, text, flags=re.IGNORECASE)
