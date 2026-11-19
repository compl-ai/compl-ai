from typing import Any

from inspect_ai.model import ModelOutput
from inspect_ai.model._openai import openai_handle_bad_request
from inspect_ai.model._providers._openai_batch import OpenAIBatcher
from openai import AsyncOpenAI
from openai import BadRequestError
from openai.types.chat import ChatCompletion

import complai.providers.openai_batch_errors  # noqa: F401  (applies the patch)


class FakeBatcher:
    """Just the attributes OpenAIBatcher._parse_jsonl_line uses."""

    _response_cls = ChatCompletion
    _openai_client = AsyncOpenAI(api_key="test-key")


def parse(line: dict[str, Any]) -> tuple[str, Any]:
    return OpenAIBatcher._parse_jsonl_line(FakeBatcher(), line)  # type: ignore[arg-type]


# shape of a real gpt-6.1-terra batch result line (2026-10-26)
REFUSED = {
    "id": "batch_req_1",
    "custom_id": "req-1",
    "response": {
        "status_code": 400,
        "request_id": "abc",
        "body": {
            "error": {
                "message": "This content was flagged for possible cybersecurity risk.",
                "type": "invalid_request",
                "param": None,
                "code": "cyber_policy",
            }
        },
    },
    "error": None,
}


def test_http_error_inside_a_result_line_becomes_an_api_error() -> None:
    # Inspect validated the error body as a response, which raised and failed the task
    request_id, result = parse(REFUSED)

    assert request_id == "req-1"
    assert isinstance(result, BadRequestError)
    assert result.code == "cyber_policy"


def test_policy_refusal_is_then_scored_like_the_sync_api_does() -> None:
    _, result = parse(REFUSED)

    output = openai_handle_bad_request("gpt-6.1-terra", result)

    assert isinstance(output, ModelOutput)
    assert output.stop_reason == "content_filter"


def test_successful_lines_are_parsed_as_before() -> None:
    body = {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ANSWER: B"},
                "finish_reason": "stop",
            }
        ],
    }
    line = {
        "custom_id": "req-2",
        "response": {"status_code": 200, "body": body},
        "error": None,
    }

    request_id, result = parse(line)

    assert request_id == "req-2"
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "ANSWER: B"
