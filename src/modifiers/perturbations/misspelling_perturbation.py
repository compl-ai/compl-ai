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
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.configs.base_modifier_config import ModifierConfig
from src.modifiers.base_modifier import BaseModifier
from src.utils.general import match_case


# The implementation below is based on the following list of common misspellings:
# https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines
class MisspellingPerturbation(BaseModifier):
    """
    Replaces words randomly with common misspellings, from a list of common misspellings.

    Perturbation example:

    **Input:**
        Already, the new product is not available.

    **Output:**
        Aready, the new product is not availible.
    """

    def __init__(self, config: Optional[ModifierConfig] = None):
        """Initialize the misspelling perturbation.

        Args:
            prob (float): probability between [0,1] of perturbing a word to a
                common misspelling (if we have a common misspelling for the word)
        """
        super().__init__(config)
        self.prob: float = self.params["prob"]
        misspellings_file = (
            Path(__file__).resolve().expanduser().parent / "correct_to_misspelling.json"
        )
        with open(misspellings_file, "r") as f:
            self.correct_to_misspelling: Dict[str, List[str]] = json.load(f)
        self.mispelling_pattern = re.compile(
            r"\b({})\b".format("|".join(self.correct_to_misspelling.keys()))
        )

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"prob": 1.0}

    def perturb(self, text: str) -> str:
        def mispell(match: re.Match) -> str:
            word = match.group(1)
            if random.random() < self.prob:
                mispelled_word = str(random.choice(self.correct_to_misspelling[word]))
                return match_case(word, mispelled_word)
            else:
                return word

        return self.mispelling_pattern.sub(mispell, text)
