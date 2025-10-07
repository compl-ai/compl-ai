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
# Source: https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/synonym_substitution/transformation.py
# Modifications: Compl-AI Team
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from platformdirs import user_cache_dir

from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import match_case
from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import Perturbation
from complai.tasks.mmlu_pro_robustness.utils import ensure_nltk_wordnet
from complai.tasks.mmlu_pro_robustness.utils import ensure_wordnet_synonyms


if TYPE_CHECKING:
    import spacy.tokens


class SynonymPerturbation(Perturbation):
    """Replaces words with their synonyms.

    For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/synonym_substitution/transformation.py

    This perturbation adds noise to a text source by randomly inserting synonyms of randomly selected
    words excluding punctuations and stopwords.
    The space of synonyms depends on WordNet and could be limited. The transformation might introduce
    non-grammatical segments.

    Perturbation example:

    **Input:**
        This was a good movie, would watch again.

    **Output:**
        This was a dependable movie, would determine again.
    """

    name: str = "synonym"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.prob: float = self.params["prob"]
        self.spacy_model, self.wordnet_synonyms = self.load_data()

    def load_data(self) -> tuple[Any, dict[str, list[str]]]:
        import nltk
        import spacy

        # Set up cache paths
        cache_dir = Path(user_cache_dir("complai"))
        nltk_data_dir = cache_dir / "nltk_data"
        synonyms_path = cache_dir / "wordnet_synonyms.json"

        # Ensure all data is downloaded
        ensure_nltk_wordnet(nltk_data_dir)
        ensure_wordnet_synonyms(synonyms_path)

        # Add NLTK data path
        nltk.data.path.append(str(nltk_data_dir))

        # Load spaCy model
        spacy_model = spacy.load("en_core_web_sm")
        with open(synonyms_path) as f:
            wordnet_synonyms: dict[str, list[str]] = json.load(f)
        return spacy_model, wordnet_synonyms

    @property
    def default_params(self) -> dict[str, Any]:
        return {"prob": 1.0}

    def get_wordnet_synonyms(self, word: str, wordnet_pos: str) -> list[str]:
        """Returns a list of synonyms for the base forms of the given word."""
        from nltk.corpus import wordnet

        synonyms = []
        for base in wordnet._morphy(word.lower(), wordnet_pos):
            synonyms.extend(self.wordnet_synonyms.get(f"{base}:{wordnet_pos}", []))
        return list(dict.fromkeys([s for s in synonyms if s != word.lower()]))

    def get_synonym_replacement(self, token: spacy.tokens.Token) -> str:
        spacy_to_wordnet_pos = {"VERB": "v", "NOUN": "n", "ADV": "r", "ADJ": "s"}

        word = token.text
        wordnet_pos = spacy_to_wordnet_pos.get(token.pos_)

        if wordnet_pos:
            synonyms = self.get_wordnet_synonyms(word, wordnet_pos)
            if synonyms and random.uniform(0, 1) < self.prob:  # nosec B311
                return match_case(word, random.choice(synonyms))  # nosec B311

        return word

    def perturb(self, text: str) -> str:
        doc = self.spacy_model(text)
        return "".join(
            self.get_synonym_replacement(token) + token.whitespace_ for token in doc
        )

    def perturb_batch(self, texts: list[str]) -> list[str]:
        docs = self.spacy_model.pipe(texts)
        return [
            "".join(
                self.get_synonym_replacement(token) + token.whitespace_ for token in doc
            )
            for doc in docs
        ]
