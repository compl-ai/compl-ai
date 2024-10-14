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

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.configs.base_modifier_config import ModifierConfig
from src.modifiers.base_modifier import BaseModifier
from src.utils.general import ensure_file_downloaded, match_case


class DialectPerturbation(BaseModifier):
    """Individual fairness perturbation for dialect."""

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "dialect"

    should_perturb_references: bool = True

    """ Output path to store external files and folders """
    OUTPUT_PATH = os.path.join("benchmark_data", "perturbations", name)

    """ Dictionary mapping dialects to one another """
    SAE = "SAE"
    AAVE = "AAVE"

    """ Dictionary containing the URIs for the dialect mapping dictionaries

    Keys are tuples of the form (source_class, target_class), such as
    ("SAE", "AAVE"). Mapping dictionaries are from the sources listed below,
    converted to JSON and stored in Google Cloud Storage.

        (1) SAE to AAVE dictionary is from Ziems et al. (2022)

                Paper: https://arxiv.org/abs/2204.03031
                GitHub: https://github.com/GT-SALT/value/

    """
    MAPPING_DICT_URIS = {
        (SAE, AAVE): (
            "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
            "augmentations/dialect_perturbation/SAE_to_AAVE_mapping.json"
        )
    }

    def __init__(
        self,
        config: Optional[ModifierConfig] = None,
    ):
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
                Dict[str, List[str]]. Otherwise, the default dictionary in
                self.MAPPING_DICTS for the provided source and target classes
                will be used, if available.
        """
        # TODO: Update path so it is not hard-coded to benchmark_output
        # https://github.com/stanford-crfm/benchmarking/issues/493
        super().__init__(config)
        self.output_path: str = self.OUTPUT_PATH
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Assign parameters to instance variables
        assert 0 <= self.params["prob"] <= 1
        self.prob = self.params["prob"]
        self.source_class: str = self.params["source_class"].upper()
        self.target_class: str = self.params["target_class"].upper()

        if self.params["mapping_file_path"]:
            self.mapping_file_path: str = self.params["mapping_file_path"]
        else:
            self.mapping_file_path = self.retrieve_mapping_dict()
        self.mapping_dict: Dict[str, List[str]] = self.load_mapping_dict()

        # Pattern capturing any occurence of the given words in the text, surrounded by characters other than
        # alphanumeric characters and '-'. We use re.escape since the words in our dictionary may
        # contain special RegEx characters.
        words = [re.escape(w) for w in self.mapping_dict.keys()]
        words_string = "|".join(words)
        self.pattern = f"[^\\w-]({words_string})[^\\w-]"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "prob": 1.0,
            "source_class": "SAE",
            "target_class": "AAVE",
            "mapping_file_path": None,
        }

    def retrieve_mapping_dict(self) -> str:
        """Download the mapping dict for self.source_class to self.target_class, if available."""
        mapping_tuple = (self.source_class, self.target_class)
        if mapping_tuple not in self.MAPPING_DICT_URIS:
            msg = f"""The mapping from the source class {self.source_class} to the
                      target class {self.target_class} isn't available in {self.MAPPING_DICT_URIS}.
                   """
            raise ValueError(msg)
        file_name = f"{self.source_class}_to_{self.target_class}_mapping.json"
        target_file_path: str = os.path.join(self.output_path, file_name)
        ensure_file_downloaded(
            source_url=self.MAPPING_DICT_URIS[mapping_tuple], target_path=target_file_path
        )
        return target_file_path

    def load_mapping_dict(self) -> Dict[str, List[str]]:
        """Load the mapping dict."""
        with open(self.mapping_file_path, "r") as f:
            return json.load(f)

    def perturb(self, text: str) -> str:
        """Substitute the source dialect in text with the target dialect with probability self.prob."""

        # Substitution function
        def sub_func(m: re.Match):
            match_str = m.group(0)  # The full match (e.g. " With ", " With,", " With.")
            word = m.group(1)  # Captured group (e.g. "With")
            if random.uniform(0, 1) < self.prob:
                synonyms = self.mapping_dict[word.lower()]
                synonym = random.choice(synonyms)  # Synonym (e.g. "wit")
                synonym = match_case(word, synonym)  # Synoynm with matching case (e.g. "Wit")
                match_str = match_str.replace(
                    word, synonym
                )  # Synonym placed in the matching group (e.g. " Wit ", " Wit,", " Wit.")
            return match_str

        # Execute the RegEx
        return re.sub(self.pattern, sub_func, text, flags=re.IGNORECASE)
