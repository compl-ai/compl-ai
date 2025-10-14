# MIT License
#
# Copyright 2025 LatticeFlow AG
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
from __future__ import annotations

import random
from typing import Any

from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import Perturbation


class TyposPerturbation(Perturbation):
    """Introduces Typos.

    For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation

    Replaces each random letters with nearby keys on a querty keyboard.
    We modified the keyboard mapping compared to the NL-augmenter augmentations so
    that: a) only distance-1 keys are used for replacement, b) the original letter is
    no longer an option, c) removed special characters (e.g., commas).

    Perturbation example:

    **Input:**
        After their marriage, she started a close collaboration with Karvelas.

    **Output:**
        Aftrr theif marriage, she started a close collaboration with Karcelas. # spellchecker:disable-line
    """

    name: str = "typos"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.prob: float = self.params["prob"]
        self.key_approx = {}

        self.key_approx["q"] = "was"
        self.key_approx["w"] = "qesad"
        self.key_approx["e"] = "wsdfr"
        self.key_approx["r"] = "edfgt"
        self.key_approx["t"] = "rfghy"
        self.key_approx["y"] = "tghju"
        self.key_approx["u"] = "yhjki"
        self.key_approx["i"] = "ujklo"
        self.key_approx["o"] = "iklp"
        self.key_approx["p"] = "ol"

        self.key_approx["a"] = "qwsz"
        self.key_approx["s"] = "weadzx"
        self.key_approx["d"] = "erfcxs"
        self.key_approx["f"] = "rtgvcd"
        self.key_approx["g"] = "tyhbvf"
        self.key_approx["h"] = "yujnbg"
        self.key_approx["j"] = "uikmnh"
        self.key_approx["k"] = "iolmj"
        self.key_approx["l"] = "opk"

        self.key_approx["z"] = "asx"
        self.key_approx["x"] = "sdcz"
        self.key_approx["c"] = "dfvx"
        self.key_approx["v"] = "fgbc"
        self.key_approx["b"] = "ghnv"
        self.key_approx["n"] = "hjmb"
        self.key_approx["m"] = "jkn"

    @property
    def default_params(self) -> dict[str, Any]:
        return {"prob": 0.1}

    def perturb(self, text: str) -> str:
        perturbed_texts = ""
        for letter in text:
            lcletter = letter.lower()
            if lcletter not in self.key_approx.keys():
                new_letter = lcletter
            else:
                if random.random() < self.prob:  # nosec B311
                    new_letter = random.choice(list(self.key_approx[lcletter]))  # nosec B311
                else:
                    new_letter = lcletter
            # go back to original case
            if not lcletter == letter:
                new_letter = new_letter.upper()
            perturbed_texts += new_letter
        return perturbed_texts
