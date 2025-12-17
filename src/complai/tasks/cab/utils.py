import re
from dataclasses import dataclass


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


def substitute_placeholder(text: str) -> list[tuple[str, str]]:
    """
    Substitute {{option1/option2/...}} placeholders with each variant.
    Returns list of (substituted_text, demographic_key) tuples, e.g.
    [("I'm male...", "male"), ("I'm female...", "female")]

    Source: https://github.com/eth-sri/cab/blob/main/src/models/local_replace.py#L9
    """
    pattern = r"\{\{\s*([^/}]+(?:\s*/\s*[^/}]+)*)\s*\}\}"

    og_matches = re.findall(pattern, text)

    if not og_matches:
        return [(text, "default")]

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

    completions: list[tuple[str, str]] = []
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
        completions.append((variant, key))

    return completions
