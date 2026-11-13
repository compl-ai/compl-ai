import asyncio
from typing import Any

import pytest
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model
from inspect_ai.model import ModelOutput
from inspect_ai.model._openai import OpenAIResponseError
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model._providers.openrouter import OpenRouterError

import complai.providers.openrouter_content_filter  # noqa: F401  (applies the patch)


def raising_generate(error: dict[str, Any]) -> Any:
    async def generate(self, input, tools, tool_choice, config):  # type: ignore[no-untyped-def]
        raise OpenRouterError(error)  # type: ignore[arg-type]

    return generate


def run_generate(monkeypatch: pytest.MonkeyPatch, error: dict[str, Any]) -> Any:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(OpenAICompatibleAPI, "generate", raising_generate(error))
    model = get_model("openrouter/google/gemini-4-argon")
    return asyncio.run(model.api.generate([], [], "auto", GenerateConfig()))


def test_content_policy_violation_becomes_content_filter_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenRouter reports a provider-side block (e.g. Gemini PROHIBITED_CONTENT) as an
    # error body; it is a refusal, so it must be scored like one rather than fail the sample.
    blocked = {
        "message": "Gemini blocked the request: PROHIBITED_CONTENT",
        "code": 403,
        "metadata": {
            "error_type": "content_policy_violation",
            "provider_code": "PROHIBITED_CONTENT",
        },
    }

    output = run_generate(monkeypatch, blocked)

    assert isinstance(output, ModelOutput)
    assert output.stop_reason == "content_filter"
    assert output.completion == ""


def raising_server_error(message: str) -> Any:
    async def generate(self, input, tools, tool_choice, config):  # type: ignore[no-untyped-def]
        raise OpenAIResponseError("server_error", message)

    return generate


def run_server_error(monkeypatch: pytest.MonkeyPatch, message: str) -> Any:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(OpenAICompatibleAPI, "generate", raising_server_error(message))
    model = get_model("openrouter/google/gemini-4-argon")
    return asyncio.run(model.api.generate([], [], "auto", GenerateConfig()))


@pytest.mark.parametrize(
    ("reason", "stop_reason"),
    [
        ("OTHER", "unknown"),
        ("SAFETY", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
    ],
)
def test_blocked_gemini_response_becomes_output_like_native_google(
    monkeypatch: pytest.MonkeyPatch, reason: str, stop_reason: str
) -> None:
    # OpenRouter reports a Gemini response block as a 502 "provider_unavailable", which
    # Inspect retries forever; the block is deterministic, so it must end as an output.
    output = run_server_error(monkeypatch, f"Gemini blocked the response: {reason}")

    assert isinstance(output, ModelOutput)
    assert output.stop_reason == stop_reason
    assert output.completion == ""


def test_genuine_server_errors_still_raise_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(OpenAIResponseError):
        run_server_error(monkeypatch, "Internal Server Error")


def test_other_openrouter_errors_still_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    key_limit = {"message": "Key limit exceeded (total limit).", "code": 403}

    # (OpenRouterError's str() is empty without metadata, so check the response itself)
    with pytest.raises(OpenRouterError) as raised:
        run_generate(monkeypatch, key_limit)
    assert raised.value.response == key_limit
