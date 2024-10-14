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

from typing import Dict, Type

from src.configs.base_modifier_config import ModifierConfig
from src.modifiers.base_modifier import BaseModifier
from src.modifiers.perturbations.contraction_expansion_perturbation import (
    ContractionPerturbation,
    ExpansionPerturbation,
)
from src.modifiers.perturbations.dialect_perturbation import DialectPerturbation
from src.modifiers.perturbations.filler_words_perturbation import (
    FillerWordsPerturbation,
)
from src.modifiers.perturbations.gender_perturbation import GenderPerturbation
from src.modifiers.perturbations.lowercase_perturbation import LowerCasePerturbation
from src.modifiers.perturbations.misspelling_perturbation import MisspellingPerturbation
from src.modifiers.perturbations.paraphrase_perturbation import ParaphrasePerturbation
from src.modifiers.perturbations.space_perturbation import SpacePerturbation
from src.modifiers.perturbations.synonym_perturbation import SynonymPerturbation
from src.modifiers.perturbations.typos_perturbation import TyposPerturbation

MODIFIER_MAP: Dict[str, Type[BaseModifier]] = {
    "dialect": DialectPerturbation,
    "typos": TyposPerturbation,
    "contraction": ContractionPerturbation,
    "expansion": ExpansionPerturbation,
    "misspelling": MisspellingPerturbation,
    "lowercase": LowerCasePerturbation,
    "filler-words": FillerWordsPerturbation,
    "spaces": SpacePerturbation,
    "gender": GenderPerturbation,
    "synonym": SynonymPerturbation,
    "paraphrase": ParaphrasePerturbation,
}


def get_modifier_from_config(config: ModifierConfig) -> BaseModifier:
    if config.name in MODIFIER_MAP:
        return MODIFIER_MAP[config.name](config)
    else:
        raise NotImplementedError("your modifier is not implemented yet")
