# MIT License
#
# Copyright (c) 2021 GEM-benchmark
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Source: https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/butter_fingers_perturbation
# Modifications: Compl-AI Team

import random
from typing import Any, Dict, Optional

from src.configs.base_modifier_config import ModifierConfig
from src.modifiers.base_modifier import BaseModifier


class TyposPerturbation(BaseModifier):
    """
    Typos. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation

    Replaces each random letters with nearby keys on a querty keyboard.
    We modified the keyboard mapping compared to the NL-augmenter augmentations so that: a) only distance-1 keys are
    used for replacement, b) the original letter is no longer an option, c) removed special characters (e.g., commas).

    Perturbation example:

    **Input:**
        After their marriage, she started a close collaboration with Karvelas.

    **Output:**
        Aftrr theif marriage, she started a close collaboration with Karcelas.
    """

    def __init__(self, config: Optional[ModifierConfig] = None):
        super().__init__(config)
        self.prob: float = self.params["prob"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"prob": 0.1}

    def perturb(self, text: str) -> str:
        key_approx = {}

        key_approx["q"] = "was"
        key_approx["w"] = "qesad"
        key_approx["e"] = "wsdfr"
        key_approx["r"] = "edfgt"
        key_approx["t"] = "rfghy"
        key_approx["y"] = "tghju"
        key_approx["u"] = "yhjki"
        key_approx["i"] = "ujklo"
        key_approx["o"] = "iklp"
        key_approx["p"] = "ol"

        key_approx["a"] = "qwsz"
        key_approx["s"] = "weadzx"
        key_approx["d"] = "erfcxs"
        key_approx["f"] = "rtgvcd"
        key_approx["g"] = "tyhbvf"
        key_approx["h"] = "yujnbg"
        key_approx["j"] = "uikmnh"
        key_approx["k"] = "iolmj"
        key_approx["l"] = "opk"

        key_approx["z"] = "asx"
        key_approx["x"] = "sdcz"
        key_approx["c"] = "dfvx"
        key_approx["v"] = "fgbc"
        key_approx["b"] = "ghnv"
        key_approx["n"] = "hjmb"
        key_approx["m"] = "jkn"

        perturbed_texts = ""
        for letter in text:
            lcletter = letter.lower()
            if lcletter not in key_approx.keys():
                new_letter = lcletter
            else:
                if random.random() < self.prob:
                    new_letter = random.choice(list(key_approx[lcletter]))
                else:
                    new_letter = lcletter
            # go back to original case
            if not lcletter == letter:
                new_letter = new_letter.upper()
            perturbed_texts += new_letter
        return perturbed_texts
