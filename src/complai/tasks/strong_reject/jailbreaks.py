import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

from complai.constants import CACHE_DIR


# Inspect loads task files with their folder first on sys.path, where our
# strong_reject.py would hide the installed strong_reject package.
_task_dir = Path(__file__).parent.resolve()
_sys_path = sys.path[:]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _task_dir]
try:
    from strong_reject.jailbreaks import (
        registered_jailbreaks as strong_reject_jailbreaks,
    )
finally:
    sys.path[:] = _sys_path


logger = logging.getLogger(__name__)

STRONG_REJECT_CACHE_DIR = CACHE_DIR / "strong_reject"

# Jailbroken prompts of the small dataset, generated once and used by the reference
# logs the Core subset is calibrated on. Gated: request access on the dataset page.
REFERENCE_PROMPTS_DATASET = "compl-ai/strong_reject"
REFERENCE_PROMPTS_REVISION = "fb7cdd5e5a31a1ddc155711fb9cff00f6a1105f8"
REFERENCE_PROMPT_SOURCE = (
    f"{REFERENCE_PROMPTS_DATASET}@{REFERENCE_PROMPTS_REVISION[:7]}"
)


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


def load_reference_prompts(
    methods: Iterable[JailbreakMethod],
) -> dict[str, dict[str, str]]:
    """Download the reference prompts of each method, or return {} if unavailable."""
    prompts: dict[str, dict[str, str]] = {}
    try:
        for method in methods:
            path = hf_hub_download(
                REFERENCE_PROMPTS_DATASET,
                f"{method}_small.json",
                repo_type="dataset",
                revision=REFERENCE_PROMPTS_REVISION,
            )
            with open(path) as f:
                prompts[method] = json.load(f)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to generation
        logger.warning(
            "Could not load the reference strong_reject prompts from %s (%s). "
            "Falling back to locally cached or newly generated prompts, so scores "
            "are not comparable with reference runs. Request access on "
            "https://huggingface.co/datasets/%s and set HF_TOKEN.",
            REFERENCE_PROMPTS_DATASET,
            exc.__class__.__name__,
            REFERENCE_PROMPTS_DATASET,
        )
        return {}
    return prompts


def get_method_cache_path(method: JailbreakMethod, full: bool = False) -> Path:
    return STRONG_REJECT_CACHE_DIR / f"{method}{'_small' if not full else ''}.json"


def load_method_cache(
    method: JailbreakMethod, full: bool = False
) -> dict[str, str] | None:
    cache_path = get_method_cache_path(method, full)
    try:
        with open(cache_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def save_method_cache(
    method: JailbreakMethod, cache: dict[str, str], full: bool = False
) -> None:
    STRONG_REJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_method_cache_path(method, full)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
