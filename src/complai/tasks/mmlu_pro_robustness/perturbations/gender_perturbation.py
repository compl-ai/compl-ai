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


""" Gender term mappings """
GENDER_TERM_MAPPINGS: list[tuple[str, ...]] = [
    # List based on Garg et al. (2018)
    ("child", "daughter", "son"),
    ("children", "daughters", "sons"),
    ("parent", "mother", "father"),
    ("parents", "mothers", "fathers"),
    ("kiddo", "girl", "boy"),
    ("kiddos", "girls", "boys"),
    ("person", "woman", "man"),
    ("people", "women", "men"),
    ("sibling", "sister", "brother"),
    ("siblings", "sisters", "brothers"),
    ("nibling", "niece", "nephew"),
    ("niblings", "nieces", "nephews"),
    # List based on Bolukbasi et al. (2016)
    ("monarch", "queen", "king"),
    ("monarchs", "queens", "kings"),
    ("server", "waitress", "waiter"),
    ("servers", "waitresses", "waiters"),
    # Our additions
    ("parent", "mom", "dad"),
    ("parents", "moms", "dads"),
    ("stepchild", "stepdaughter", "stepson"),
    ("stepchildren", "stepdaughters", "stepsons"),
    ("stepparent", "stepmother", "stepfather"),
    ("stepparents", "stepmothers", "stepfathers"),
    ("stepparent", "stepmom", "stepdad"),
    ("stepparents", "stepmoms", "stepdads"),
    ("grandchild", "granddaughter", "grandson"),
    ("grandchildren", "granddaughters", "grandsons"),
    ("grandparent", "grandmother", "grandfather"),
    ("grandparents", "grandmothers", "grandfathers"),
    ("grandparent", "grandma", "granddad"),
    ("grandparents", "grandmas", "granddads"),
    ("human", "female", "male"),
    ("humans", "females", "males"),
]

""" Gender pronoun mappings """
# The overlaps between the pairs cause our replacements to be wrong in certain
# cases (direct pronouns vs. indirect pronouns). In these cases, we keep the
# first match instead of making our decision using a POS tagger for simplicity
GENDER_PRONOUN_MAPPINGS: list[tuple[str, ...]] = [
    # List from Lauscher et. al. 2022
    ("they", "she", "he"),
    ("them", "her", "him"),
    ("their", "her", "his"),
    ("theirs", "hers", "his"),
    ("themselves", "herself", "himself"),
]


class GenderPerturbation(Perturbation):
    """Individual fairness perturbation for gender terms and pronouns."""

    should_perturb_references: bool = True

    """ Genders defined by default """
    NEUTRAL = "neutral"
    FEMALE = "female"
    MALE = "male"
    GENDERS = [NEUTRAL, FEMALE, MALE]

    """ Modes """
    GENDER_TERM = "terms"
    GENDER_PRONOUN = "pronouns"
    MODES = [GENDER_TERM, GENDER_PRONOUN]
    MODE_TO_MAPPINGS = {
        GENDER_TERM: GENDER_TERM_MAPPINGS,
        GENDER_PRONOUN: GENDER_PRONOUN_MAPPINGS,
    }

    name: str = "gender"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        """Initialize the gender perturbation.

        Args:
            mode: The mode of the gender perturbation, must be one of
                "terms" or "pronouns".
            prob: Probability of substituting a word in the source class with
                a word in the target class given that a substitution is
                available.
            source_class: The source gender that will be substituted with
                the target gender. If mapping_file_path is provided, the source
                class must be one of the genders in it. If not, it must be
                exactly one of `male`, `female`, and `neutral. Case-insensitive.
            target_class: Same as the source class, but for the target gender.
            mapping_file_path: The absolute path to a file containing the
                word mappings from the source gender to the target gender in
                a json format. The json dictionary must be of type
                List[List[str, ...]]. It is assumed that 0th index of the inner
                lists correspond to the 0th gender, 1st index to 1st gender
                and so on. All word cases are lowered.
                If mapping_file_path is None, the default dictionary in
                self.MODE_TO_MAPPINGS for the provided source and target classes
                will be used, if available.
            mapping_file_genders: The genders in the mapping supplied in the
                mapping_file_path. The inner lists read from mapping_file_path
                should have the same length as the mapping_file_genders. The
                order of the genders is assumed to reflect the order in the
                mapping_file_path. Must not be None if mapping_file_path
                is set. All word cases are lowered.
            bidirectional: Whether we should apply the perturbation in both
                directions. If we need to perturb a word, we first check if it
                is in list of source_class words, and replace it with the
                corresponding target_class word if so. If the word isn't in the
                source_class words, we check if it is in the target_class words,
                and replace it with the corresponding source_class word if so.
        """
        # Assign parameters to instance variables
        assert self.params["mode"] in self.MODES
        self.mode = self.params["mode"]

        assert 0 <= self.params["prob"] <= 1
        self.prob = self.params["prob"]

        self.source_class: str = self.params["source_class"].lower()
        self.target_class: str = self.params["target_class"].lower()
        self.mapping_file_path: str | None = self.params["mapping_file_path"]
        self.bidirectional: bool = self.params["bidirectional"]

        # Get mappings and self.genders
        mappings: list[tuple[str, ...]] = self.MODE_TO_MAPPINGS[self.mode]
        self.genders = self.GENDERS
        if self.mapping_file_path and self.params["mapping_file_genders"]:
            mappings = self.load_mappings(self.mapping_file_path)
            self.genders = [g.lower() for g in self.params["mapping_file_genders"]]
        assert (
            mappings
            and self.source_class in self.genders
            and self.target_class in self.genders
        )
        assert all([len(m) == len(self.genders) for m in mappings])

        # Get source and target words
        gender_to_ind: dict[str, int] = {
            gender: ind for ind, gender in enumerate(self.genders)
        }
        word_lists = list(zip(*mappings))
        self.source_words: list[str] = list(
            word_lists[gender_to_ind[self.source_class]]
        )
        self.target_words: list[str] = list(
            word_lists[gender_to_ind[self.target_class]]
        )

        # Get word_synonym_pairs
        self.word_synonym_pairs = list(zip(self.source_words, self.target_words))

        # If self.bidirectional flag is set, extend the pairs list
        if self.bidirectional:
            new_pairs = list(zip(self.target_words, self.source_words))
            self.word_synonym_pairs.extend(new_pairs)

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "mode": "terms",
            "prob": 1.0,
            "source_class": "male",
            "target_class": "female",
            "mapping_file_path": None,
            "mapping_file_genders": None,
            "bidirectional": False,
        }

    @staticmethod
    def load_mappings(mapping_file_path: str) -> list[tuple[str, ...]]:
        """Load mappings as a list."""
        with open(mapping_file_path, "r") as f:
            loaded_json = json.load(f)
            return [tuple([str(e).lower() for e in t]) for t in loaded_json]

    def substitute_word(self, text: str, word: str, synonym: str) -> str:
        """Substitute the occurrences of word in text with its synonym with self.probability"""
        # Pattern capturing any occurrence of given word in the text, surrounded by non-alphanumeric characters
        pattern = f"[^\\w]({word})[^\\w]"

        # Substitution function
        def sub_func(m: re.Match) -> str:
            match_str = m.group(
                0
            )  # The full match (e.g. " Man ", " Man,", " Man.", "-Man.")
            match_word = m.group(1)  # Captured group (e.g. "Man")
            if random.uniform(0, 1) < self.prob:  # nosec B311
                syn = match_case(
                    match_word, synonym
                )  # Synonym with matching case (e.g. "Woman")
                match_str = match_str.replace(
                    match_word, syn
                )  # Synonym placed in the matching group (e.g. " Woman ", " Woman,", " Woman.", "-Woman")
            return match_str

        # Execute the RegEx
        return re.sub(pattern, sub_func, text, flags=re.IGNORECASE)

    def perturb(self, text: str) -> str:
        """Perform the perturbations on the provided text."""
        # Substitute the words
        for word, synonym in self.word_synonym_pairs:
            text = self.substitute_word(text, word, synonym)

        return text
