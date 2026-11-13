"""Score OpenRouter content-policy blocks as refusals instead of sample errors.

OpenRouter reports a provider-side request block (e.g. Gemini ``PROHIBITED_CONTENT``) as an
error body with ``metadata.error_type == "content_policy_violation"``; Inspect's
OpenRouter provider raises it as ``OpenRouterError``, which fails the sample (and,
with the default fail_on_error, the whole task). Inspect's native Google and OpenAI
providers instead return an empty output with ``stop_reason="content_filter"``; this
patch does the same for ``openrouter/`` (and ``openrouter-batch/``) models. Patching
the existing provider keeps the model name in logs unchanged.

A Gemini *response* block ("Gemini blocked the response: <REASON>") arrives as a 502
``provider_unavailable``, which Inspect retries indefinitely; it is mapped to an empty
output with the stop reason the native Google provider uses for that finish reason.
"""

import functools
from typing import Any

from inspect_ai.model import ModelOutput
from inspect_ai.model import StopReason
from inspect_ai.model._openai import OpenAIResponseError
from inspect_ai.model._providers.openrouter import OpenRouterAPI
from inspect_ai.model._providers.openrouter import OpenRouterError


CONTENT_POLICY_ERROR_TYPE = "content_policy_violation"
# OpenRouter turns a Gemini *response* block into a retryable 502, which never succeeds.
GEMINI_BLOCKED_RESPONSE = "Gemini blocked the response: "
# Gemini finish reasons Inspect's native Google provider maps to content_filter (others: unknown).
GEMINI_FILTER_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
}


def is_content_policy_block(ex: OpenRouterError) -> bool:
    return (ex.response.get("metadata") or {}).get(
        "error_type"
    ) == CONTENT_POLICY_ERROR_TYPE


def blocked_response_stop_reason(ex: OpenAIResponseError) -> StopReason | None:
    """Stop reason for a blocked Gemini response, None if ``ex`` is not one."""
    if ex.code != "server_error" or not ex.message.startswith(GEMINI_BLOCKED_RESPONSE):
        return None
    reason = ex.message.removeprefix(GEMINI_BLOCKED_RESPONSE).strip()
    return "content_filter" if reason in GEMINI_FILTER_REASONS else "unknown"


def _patch() -> None:
    generate = OpenRouterAPI.generate
    if getattr(generate, "_complai_content_filter", False):
        return

    @functools.wraps(generate)
    async def patched(self: OpenRouterAPI, *args: Any, **kwargs: Any) -> Any:
        try:
            return await generate(self, *args, **kwargs)
        except OpenRouterError as ex:
            if not is_content_policy_block(ex):
                raise
            return ModelOutput.from_content(
                self.model_name, content="", stop_reason="content_filter"
            )
        except OpenAIResponseError as ex:
            stop_reason = blocked_response_stop_reason(ex)
            if stop_reason is None:
                raise
            return ModelOutput.from_content(
                self.model_name, content="", stop_reason=stop_reason
            )

    patched._complai_content_filter = True  # type: ignore[attr-defined]
    OpenRouterAPI.generate = patched  # type: ignore[method-assign]


_patch()
