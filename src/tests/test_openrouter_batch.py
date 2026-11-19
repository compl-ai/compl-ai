import asyncio
from typing import Any

import anyio
import pytest
from inspect_ai.model import BatchConfig
from inspect_ai.model import get_model
from inspect_ai.model._providers.util.batch import Batch
from inspect_ai.model._providers.util.batch import BatchRequest
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletion

from complai.providers.openrouter_batch import batch_model_name
from complai.providers.openrouter_batch import OpenRouterBatchAPI
from complai.providers.openrouter_batch import OpenRouterBatcher


class FakeClient:
    """Records low-level calls and returns canned JSON responses."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, Any]] = []

    async def post(self, path: str, *, cast_to: type, body: Any = None) -> Any:
        self.calls.append(("POST", path, body))
        return self.responses[path]

    async def get(self, path: str, *, cast_to: type) -> Any:
        self.calls.append(("GET", path, None))
        return self.responses[path]


def make_batcher(client: FakeClient) -> OpenRouterBatcher:
    return OpenRouterBatcher(
        client,  # type: ignore[arg-type]
        "deepseek/deepseek-v4.1-flash:batch",
        BatchConfig(),
        {},  # type: ignore[typeddict-item]
    )


def make_request(custom_id: str, body: dict[str, Any]) -> BatchRequest[ChatCompletion]:
    send, _ = anyio.create_memory_object_stream[ChatCompletion | Exception](1)
    return BatchRequest(request=body, result_stream=send, custom_id=custom_id)


def make_batch(custom_ids: list[str]) -> Batch[ChatCompletion]:
    return Batch(
        id="batch_1", requests={cid: make_request(cid, {}) for cid in custom_ids}
    )


def completion_body(content: str) -> dict[str, Any]:
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek/deepseek-v4.1-flash:batch",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def batch_object(status: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": "batch_1",
        "status": status,
        "created_at": 1700000000,
        "request_counts": {"total": 2, "completed": 1, "failed": 0},
        "results": None,
        "error": None,
        **extra,
    }


def test_batch_model_name_appends_suffix_once() -> None:
    assert (
        batch_model_name("deepseek/deepseek-v4.1-flash")
        == "deepseek/deepseek-v4.1-flash:batch"
    )
    assert (
        batch_model_name("deepseek/deepseek-v4.1-flash:batch")
        == "deepseek/deepseek-v4.1-flash:batch"
    )


def test_create_batch_posts_inline_requests_in_required_key_order() -> None:
    client = FakeClient({"/batches": batch_object("validating")})
    batcher = make_batcher(client)
    request = {
        "model": "deepseek/deepseek-v4.1-flash:batch",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "stream": False,
    }

    batch_id = asyncio.run(batcher._create_batch([make_request("a", request)]))

    assert batch_id == "batch_1"
    method, path, body = client.calls[0]
    assert (method, path) == ("POST", "/batches")
    # OpenRouter stream-parses the body and rejects `requests` before the header fields.
    assert list(body) == ["endpoint", "model", "requests"]
    assert body["endpoint"] == "/v1/chat/completions"
    assert body["model"] == "deepseek/deepseek-v4.1-flash:batch"
    assert body["requests"] == [
        {
            "custom_id": "a",
            "body": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        }
    ]


def test_create_batch_flattens_client_only_request_keys() -> None:
    client = FakeClient({"/batches": batch_object("validating")})
    request = {
        "model": "deepseek/deepseek-v4.1-flash:batch",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": NOT_GIVEN,
        "extra_headers": {"x-irid": "abc"},
        "extra_body": {"reasoning": {"effort": "high"}},
    }

    asyncio.run(make_batcher(client)._create_batch([make_request("a", request)]))

    # reasoning settings live in extra_body for the sync client; batch bodies need them top-level
    assert client.calls[0][2]["requests"][0]["body"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"effort": "high"},
    }


def test_check_batch_in_progress_returns_no_completion_info() -> None:
    client = FakeClient({"/batches/batch_1": batch_object("in_progress")})

    result = asyncio.run(make_batcher(client)._check_batch(make_batch(["a", "b"])))

    assert result.completion_info is None
    assert (result.completed_count, result.failed_count) == (1, 0)
    assert result.created_at == 1700000000


def test_check_batch_completed_returns_inline_results() -> None:
    results = [
        {
            "custom_id": "a",
            "response": {"status_code": 200, "body": completion_body("x")},
        }
    ]
    client = FakeClient(
        {"/batches/batch_1": batch_object("completed", results=results)}
    )

    result = asyncio.run(make_batcher(client)._check_batch(make_batch(["a"])))

    assert result.completion_info == results


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_check_batch_terminal_failure_raises_with_error_message(status: str) -> None:
    error = {"message": "base64/data-URI images are rejected"}
    client = FakeClient({"/batches/batch_1": batch_object(status, error=error)})

    with pytest.raises(RuntimeError, match="data-URI images are rejected"):
        asyncio.run(make_batcher(client)._check_batch(make_batch(["a"])))


def test_handle_batch_result_maps_successes_and_failures_by_custom_id() -> None:
    results = [
        {
            "custom_id": "b",
            "response": {
                "status_code": 400,
                "body": {"error": {"message": "bad request"}},
            },
        },
        {
            "custom_id": "a",
            "response": {"status_code": 200, "body": completion_body("ANSWER: B")},
        },
    ]
    batcher = make_batcher(FakeClient({}))

    mapped = asyncio.run(
        batcher._handle_batch_result(make_batch(["a", "b", "c"]), results)
    )

    assert isinstance(mapped["a"], ChatCompletion)
    assert mapped["a"].choices[0].message.content == "ANSWER: B"
    assert isinstance(mapped["b"], Exception) and "bad request" in str(mapped["b"])
    # a request missing from the results must still fail rather than hang
    assert isinstance(mapped["c"], Exception)


def test_handle_batch_result_accepts_batch_service_tier() -> None:
    # Anthropic batch results report service_tier="batch", which the openai model rejects.
    body = completion_body("ANSWER: B") | {"service_tier": "batch"}
    results = [{"custom_id": "a", "response": {"status_code": 200, "body": body}}]

    mapped = asyncio.run(
        make_batcher(FakeClient({}))._handle_batch_result(make_batch(["a"]), results)
    )

    assert isinstance(mapped["a"], ChatCompletion)
    assert mapped["a"].choices[0].message.content == "ANSWER: B"


def test_handle_batch_result_isolates_unparseable_results() -> None:
    # one malformed result must fail only its own request, not the whole (paid) batch
    results = [
        {
            "custom_id": "a",
            "response": {"status_code": 200, "body": {"choices": "not-a-list"}},
        },
        {
            "custom_id": "b",
            "response": {"status_code": 200, "body": completion_body("ANSWER: C")},
        },
    ]

    mapped = asyncio.run(
        make_batcher(FakeClient({}))._handle_batch_result(
            make_batch(["a", "b"]), results
        )
    )

    assert isinstance(mapped["a"], Exception)
    assert isinstance(mapped["b"], ChatCompletion)


def test_handle_batch_result_maps_blocked_output_to_content_filter_completion() -> None:
    # The sync API returns a blocked output as a normal completion (finish_reason
    # content_filter); the batch API reports it as a per-request error instead.
    results = [
        {
            "custom_id": "a",
            "error": {
                "type": "invalid_request_error",
                "message": "Output blocked by content filtering policy",
            },
            "response": None,
        }
    ]

    mapped = asyncio.run(
        make_batcher(FakeClient({}))._handle_batch_result(make_batch(["a"]), results)
    )

    assert isinstance(mapped["a"], ChatCompletion)
    assert mapped["a"].choices[0].finish_reason == "content_filter"
    assert not mapped["a"].choices[0].message.content


def test_provider_is_registered_and_targets_batch_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    model = get_model("openrouter-batch/deepseek/deepseek-v4.1-flash")

    assert isinstance(model.api, OpenRouterBatchAPI)
    assert model.api.service_model_name().endswith("deepseek/deepseek-v4.1-flash:batch")
