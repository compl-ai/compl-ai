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
# Source: https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/filler_word_augmentation/transformation.py
# Modifications: Compl-AI Team


import random
from typing import Any, Dict, Optional

from src.configs.base_modifier_config import ModifierConfig
from src.modifiers.base_modifier import BaseModifier

# Speaker opinion/mental state phrases
# Taken from Kovatchev et al. (2021)
SPEAKER_PHRASES = [
    "I think",
    "I believe",
    "I mean",
    "I guess",
    "that is",
    "I assume",
    "I feel",
    "In my opinion",
    "I would say",
]

# Words and phrases indicating uncertainty
# Taken from Kovatchev et al. (2021)
UNCERTAIN_PHRASES = ["maybe", "perhaps", "probably", "possibly", "most likely"]

# Filler words that should preserve the meaning of the phrase
# Taken from Laserna et al. (2014)
FILL_PHRASE = ["uhm", "umm", "ahh", "err", "actually", "obviously", "naturally", "like", "you know"]


class FillerWordsPerturbation(BaseModifier):
    """
    Randomly inserts filler words and phrases in the sentence.
    Perturbation example:

    **Input:**
        The quick brown fox jumps over the lazy dog.

    **Output:**
        The quick brown fox jumps over probably the lazy dog.

    """

    def __init__(self, config: Optional[ModifierConfig] = None):
        super().__init__(config)
        self.prob = self.params["prob"]
        self.max_num_insert = self.params["max_num_insert"]
        self.uncertain_ph = self.params["uncertain_ph"]
        self.fill_ph = self.params["fill_ph"]
        self.speaker_ph = self.params["speaker_ph"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "prob": 0.3,
            "max_num_insert": None,
            "speaker_ph": False,
            "uncertain_ph": True,
            "fill_ph": True,
        }

    @staticmethod
    def gen_filled_text(
        text,
        prob: float,
        max_num_insert=1,
        speaker_ph=True,
        uncertain_ph=True,
        fill_ph=True,
    ):
        all_fillers = []
        if speaker_ph:
            all_fillers.extend(SPEAKER_PHRASES)
        if uncertain_ph:
            all_fillers.extend(UNCERTAIN_PHRASES)
        if fill_ph:
            all_fillers.extend(FILL_PHRASE)

        insert_count = 0
        perturbed_words = []
        for index, word in enumerate(text.split(" ")):
            if (
                (max_num_insert is None or insert_count < max_num_insert)
                and random.random() <= prob
            ) and index != 0:
                random_filler = random.choice(all_fillers)
                perturbed_words.append(random_filler)
                insert_count += 1
            perturbed_words.append(word)

        perturbed_text = " ".join(perturbed_words)

        return perturbed_text

    def perturb(self, text: str) -> str:
        return self.gen_filled_text(
            text,
            self.prob,
            self.max_num_insert,
            self.speaker_ph,
            self.uncertain_ph,
            self.fill_ph,
        )
