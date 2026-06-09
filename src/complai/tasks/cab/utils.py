import re
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError


class BiasDetails(BaseModel):
    description: str
    reasoning: str
    bias_score: str = Field(pattern="^[1-5]$")
    relevance_reasoning: str
    relevance_score: str = Field(pattern="^[1-5]$")
    generality_reasoning: str
    generality_score: str = Field(pattern="^[1-5]$")
    refusal_reasoning: str
    refusal_score: str = Field(pattern="^[0-1]$")


class GenderBiasEvaluation(BaseModel):
    reasoning: str
    gender_bias: BiasDetails


class RaceBiasEvaluation(BaseModel):
    reasoning: str
    race_bias: BiasDetails


class ReligionBiasEvaluation(BaseModel):
    reasoning: str
    religion_bias: BiasDetails


BiasEvaluation = GenderBiasEvaluation | RaceBiasEvaluation | ReligionBiasEvaluation


@dataclass
class Persona:
    name: str


@dataclass
class Thread:
    question: str
    answer: str


@dataclass
class Conversation:
    persona: Persona
    threads: list[Thread]


def substitute_placeholder(text: str) -> dict[str, str]:
    """
    Substitute {{option1/option2/...}} placeholders with each variant.
    Returns dict of demographic_key -> substituted_text, e.g.
    {"male": "I'm male...", "female": "I'm female..."}

    Adapted from https://github.com/eth-sri/cab/blob/main/src/models/local_replace.py#L9
    """
    pattern = r"\{\{\s*([^/}]+(?:\s*/\s*[^/}]+)*)\s*\}\}"

    og_matches = re.findall(pattern, text)

    if not og_matches:
        return {"default": text}

    matches = [match.split("/") for match in og_matches]

    # Handle suffix propagation (e.g., "male/female friends" -> "male friends", "female friends")
    formatted_matches = []
    for match_group in matches:
        new_group = [elem.strip() for elem in match_group]
        lengths = [len(e.split()) for e in new_group]

        if (
            all(l == lengths[0] for l in lengths[:-1])
            and len(lengths) > 1
            and lengths[-1] == lengths[0] + 1
        ):
            suffix = new_group[-1].split()[-1]
            new_group = [e + " " + suffix for e in new_group[:-1]] + [new_group[-1]]

        formatted_matches.append(new_group)

    # Verify consistency across all placeholders
    num_variants = len(formatted_matches[0])
    if not all(len(m) == num_variants for m in formatted_matches):
        raise ValueError(f"Inconsistent placeholder options in: {text}")

    completions: dict[str, str] = {}
    for i in range(num_variants):
        variant = text
        for j, og_match in enumerate(og_matches):
            variant = re.sub(
                rf"\{{{{{re.escape(og_match)}}}}}",
                formatted_matches[j][i],
                variant,
                count=1,
            )
        # Use first word of first match as demographic key
        key = formatted_matches[0][i].split()[0].lower()
        completions[key] = variant

    return completions


def complete_truncated_json(completion: str) -> str | None:
    """Conservatively append missing closing delimiters to truncated JSON."""
    completion = completion.rstrip()
    stack: list[str] = []
    in_string = False
    escaped = False

    for character in completion:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            expected = "{" if character == "}" else "["
            if not stack or stack.pop() != expected:
                return None

    if in_string or not stack:
        return None

    closing_delimiters = {"{": "}", "[": "]"}
    return completion + "".join(closing_delimiters[item] for item in reversed(stack))


def parse_bias_evaluation(
    evaluation_model: BiasEvaluation, completion: str
) -> tuple[BiasEvaluation, bool]:
    try:
        return evaluation_model.model_validate_json(completion), False
    except ValidationError as original_error:
        repaired_completion = complete_truncated_json(completion)
        if repaired_completion is None:
            raise

        try:
            return evaluation_model.model_validate_json(repaired_completion), True
        except ValidationError as repaired_error:
            raise repaired_error from original_error


def format_validation_error(error: ValidationError) -> str:
    return "; ".join(
        (f"{'.'.join(str(part) for part in detail['loc'])}: " if detail["loc"] else "")
        + detail["msg"]
        for detail in error.errors(include_input=False, include_url=False)
    )
