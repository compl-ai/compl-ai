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
# Source: https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/synonym_substitution/transformation.py
# Modifications: Compl-AI Team

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import nltk
import spacy
from nltk.corpus import wordnet

from src.configs.base_modifier_config import ModifierConfig
from src.modifiers.base_modifier import BaseModifier
from src.utils.general import ensure_file_downloaded, match_case


class SynonymPerturbation(BaseModifier):
    """
    Synonyms. For implementation details, see
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
    # For downloading wordnet_synonyms.json
    FILE_NAME: str = "wordnet_synonyms.json"
    SOURCE_URI: str = (
        "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
        "augmentations/synonym_perturbation/wordnet_synonyms.json"
    )

    def __init__(self, config: Optional[ModifierConfig] = None):
        super().__init__(config)
        # Assign parameters to instance variables
        self.prob: float = self.params["prob"]

        # Initialize the model with spaCy: https://spacy.io/models/en
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")  # type: ignore
            self.spacy_model = spacy.load("en_core_web_sm")

        output_dir = os.path.join("benchmark_data", "perturbations", self.name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        nltk.data.path.append(output_dir)
        try:
            # We cannot use wordnet.synsets directly since it's not thread-safe. So we copy the synsets to
            # wordnet_synonyms.json and use that in combination with _morphy (as done in the original wordnet.synsets).
            wordnet.ensure_loaded()
        except LookupError:
            if not os.path.exists(os.path.join(output_dir, "corpora/wordnet")):
                nltk.download("wordnet", download_dir=output_dir)
            if not os.path.exists(os.path.join(output_dir, "corpora/omw-1.4")):
                nltk.download("omw-1.4", download_dir=output_dir)
        wordnet.ensure_loaded()

        target_path = os.path.join(output_dir, self.FILE_NAME)
        ensure_file_downloaded(source_url=self.SOURCE_URI, target_path=target_path)
        with open(target_path) as f:
            self.wordnet_synonyms: Dict[str, List[str]] = json.load(f)

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"prob": 1.0}

    def perturb(self, text: str) -> str:
        spacy_to_wordnet_pos = {
            "VERB": "v",
            "NOUN": "n",
            "ADV": "r",
            "ADJ": "s",
        }

        doc = self.spacy_model(text)

        perturbed_text = ""

        for token in doc:
            word = token.text
            wordnet_pos = spacy_to_wordnet_pos.get(token.pos_)
            synonyms = []
            if wordnet_pos:
                for base in wordnet._morphy(
                    word.lower(), wordnet_pos
                ):  # _morphy returns the base form of a word
                    synonyms.extend(self.wordnet_synonyms.get(f"{base}:{wordnet_pos}", []))
            synonyms = [s for s in synonyms if s != word.lower()]
            synonyms = list(
                dict.fromkeys(synonyms)
            )  # Make the list unique while preserving the order
            if synonyms and random.uniform(0, 1) < self.prob:
                synonym = random.choice(synonyms)
                word = match_case(word, synonym)
            perturbed_text += word + token.whitespace_

        return perturbed_text
