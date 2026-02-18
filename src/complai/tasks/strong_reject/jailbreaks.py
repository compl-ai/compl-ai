import json
from pathlib import Path
from typing import Literal

from strong_reject.jailbreaks import registered_jailbreaks as strong_reject_jailbreaks

from complai.constants import CACHE_DIR


STRONG_REJECT_CACHE_DIR = CACHE_DIR / "strong_reject"


JailbreakMethod = Literal[
    "auto_payload_splitting",
    "auto_obfuscation",
    "rot_13",
    "disemvowel",
    "pair",
    "pap_evidence_based_persuasion",
    "pap_expert_endorsement",
    "pap_misrepresentation",
    "pap_authority_endorsement",
    "pap_logical_appeal",
    ## These require Google Translate API
    # "translation_hmong",
    # "translation_scotts_gaelic",
    # "translation_guarani",
    # "translation_zulu",
    "gcg_transfer_harmbench",
    "gcg_transfer_universal_attacks",
    "combination_3",
    "combination_2",
    "few_shot_json",
    "dev_mode_v2",
    "dev_mode_with_rant",
    "wikipedia_with_title",
    "distractors",
    "wikipedia",
    "style_injection_json",
    "style_injection_short",
    "refusal_suppression",
    "prefix_injection",
    "distractors_negated",
    "poems",
    "base64",
    "base64_raw",
    "base64_input_only",
    "base64_output_only",
    "none",
    "evil_confidant",
    "aim",
    "bon",
    "renellm",
]


def get_jailbreak_prompt(
    method: JailbreakMethod, forbidden_prompt: str, cache: dict[str, str] | None
) -> tuple[str, bool]:
    if cache is not None and forbidden_prompt in cache:
        return cache[forbidden_prompt], False

    jailbroken_prompt = strong_reject_jailbreaks[method](forbidden_prompt)
    return jailbroken_prompt, True


def get_method_cache_path(method: JailbreakMethod) -> Path:
    return STRONG_REJECT_CACHE_DIR / f"{method}.json"


def load_method_cache(method: JailbreakMethod) -> dict[str, str] | None:
    cache_path = get_method_cache_path(method)
    try:
        with open(cache_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def save_method_cache(method: JailbreakMethod, cache: dict[str, str]) -> None:
    STRONG_REJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_method_cache_path(method)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
