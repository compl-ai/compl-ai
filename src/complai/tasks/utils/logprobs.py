import numpy as np
from inspect_ai.model import ChatCompletionChoice


def num_nonzero_probs(logprobs: list[float]) -> int:
    """
    Returns the number of logprobs that are not -np.inf.

    Args:
        logprobs: Input list of logprobs.
    """
    return sum(value != -np.inf for value in logprobs)


def get_logprobs_first_token(
    choice: ChatCompletionChoice, tokens: list[str], strip_spaces: bool | None = None
) -> list[float]:
    """
    Get the logprobs for all tokens in tokens at the first position of
    the model output. If a token is not in the model output, we assign it a logprob of
    -infinity.

    Args:
        choice: The model outputs with logprobs.
        tokens: The tokens for which to get logprobs.
        strip_spaces: Whether to strip spaces from logprob tokens.
    """
    return get_logprobs(choice, tokens, strip_spaces=strip_spaces, position=0)


def get_logprobs_last_tokens(
    choice: ChatCompletionChoice, tokens: list[str], last_k: int = 5
) -> list[float]:
    """
    Get the logprobs for each token in `tokens` at the last k positions of the
    model output. If a token is not in the model output, we assign it a logprob of
    -infinity.

    Extracting logprobs at the end of a completion for a given token is tricky since
    we do not know which tokenizer was used. There can be tokens for punctuation or
    tokens can include leading spaces. Here, we consider the final last_k tokens and
    return the logprobs for the position for which we can extract the most amount of
    non-zero probs for the given `tokens`.

    Args:
        choice: The model outputs with logprobs.
        tokens: The tokens for which to get logprobs.
        last_k: The last k positions to check.
    """
    if choice.logprobs is None or not choice.logprobs.content:
        raise ValueError("Logprobs must be provided.")

    num_tokens = len(choice.logprobs.content)
    best_logprobs = get_logprobs(
        choice, tokens, strip_spaces=True, position=num_tokens - 1
    )
    for position in reversed(range(num_tokens - last_k, num_tokens)):
        # In case of ties, prefer extraction from the later tokens.
        logprobs = get_logprobs(choice, tokens, strip_spaces=True, position=position)
        if num_nonzero_probs(logprobs) > num_nonzero_probs(best_logprobs):
            best_logprobs = logprobs

    return best_logprobs


def get_logprobs(
    choice: ChatCompletionChoice,
    tokens: list[str],
    strip_spaces: bool | None = None,
    position: int = 0,
) -> list[float]:
    """
    Get the logprobs for all the `tokens` at the given `position` in the model output.

    If a token is not in the provided logprobs, we assign it a logprob of -infinity.

    Args:
        choice: The model outputs. They must include logprobs.
        tokens: The tokens for which to check the logprobs.
        strip_spaces: Whether to strip spaces from logprob tokens.
        position: The position for which to extract logprobs.
    """
    if choice.logprobs is None or not choice.logprobs.content:
        raise ValueError("Logprobs must be provided.")

    if position >= len(choice.logprobs.content):
        raise ValueError(
            f"Logprobs requested for token at '{position}' position, but only '{len(choice.logprobs.content)}' tokens found."
        )

    top_logprobs = choice.logprobs.content[position].top_logprobs
    if top_logprobs is None:
        raise ValueError(
            f"Logprobs for position {position} are not available in the choice."
        )

    # NOTE: This piece is used to extract the probability a token in the
    #  response. These tokens are typically 'A', 'B', etc., but can also be ' A', ' B'
    #  if the model is a bit weird (e.g. Qwen) or if it is not the first position.
    #
    # What the logic below tries to do is align with what's in the continuation:
    # - if the continuation has no spaces, then it's safe to strip also from the output
    #   tokens.
    # - if the continuation has spaces, then we should not touch the tokens.
    strip_spaces = strip_spaces or all(value == value.strip() for value in tokens)
    if strip_spaces:
        for logprob in top_logprobs:
            # \u0120 is used in LLaMA/GPT-style models to denote space.
            # It can be returned by the model if tokens are not postprocessed
            # (e.g., when using vllm serve).
            logprob.token = logprob.token.replace("\u0120", " ").strip()

    token_to_logprobs: dict[str, float] = {}
    for logprob in top_logprobs:
        token = logprob.token
        if token in token_to_logprobs and token_to_logprobs[token] > logprob.logprob:
            # some models return the same token multiple times
            # in this case, we keep the token with the highest logprob
            continue
        token_to_logprobs[token] = logprob.logprob

    return [token_to_logprobs.get(token, -np.inf) for token in tokens]
