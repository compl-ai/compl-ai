"""OpenRouter Batch API provider for Inspect (``openrouter-batch/<org>/<model>``).

Requests are collected by Inspect's batcher and submitted to OpenRouter's
``POST /batches`` endpoint as one inline batch, then polled until done. The
model id gets OpenRouter's ``:batch`` suffix. Batches accept only public image
URLs, so tasks with inline (base64) images, such as mmmu_pro, fail.
"""

from typing import Any
from typing import cast

from inspect_ai.model import BatchConfig
from inspect_ai.model import GenerateConfig
from inspect_ai.model import modelapi
from inspect_ai.model._generate_config import normalized_batch_config
from inspect_ai.model._model import log_model_retry
from inspect_ai.model._providers.openrouter import OpenRouterAPI
from inspect_ai.model._providers.util.batch import Batch
from inspect_ai.model._providers.util.batch import BatchCheckResult
from inspect_ai.model._providers.util.batch import Batcher
from inspect_ai.model._providers.util.batch import BatchRequest
from inspect_ai.model._retry import model_retry_config
from inspect_ai.model._retry import ModelRetryConfig
from openai import AsyncOpenAI
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletion


BATCH_SUFFIX = ":batch"
# OpenRouter does not document per-batch limits; keep batches comfortably small.
MAX_BATCH_REQUESTS = 10_000
MAX_BATCH_SIZE_MB = 100
# Keys of the sync client's create() call that are not part of the request body.
CLIENT_ONLY_KEYS = {
    "model",
    "stream",
    "extra_body",
    "extra_headers",
    "extra_query",
    "timeout",
}
PENDING_STATUSES = {"validating", "in_progress", "finalizing"}
# Batch API error for a blocked output (the sync API returns finish_reason content_filter).
CONTENT_FILTER_MESSAGE = "Output blocked by content filtering policy"


def batch_model_name(model_name: str) -> str:
    """OpenRouter model id of the model's batch variant."""
    return (
        model_name if model_name.endswith(BATCH_SUFFIX) else model_name + BATCH_SUFFIX
    )


def request_body(request: dict[str, Any]) -> dict[str, Any]:
    """Turn sync client create() kwargs into a batch request body."""
    body = {
        key: value
        for key, value in request.items()
        if key not in CLIENT_ONLY_KEYS and value is not NOT_GIVEN
    }
    return body | (request.get("extra_body") or {})


def parse_completion(body: dict[str, Any]) -> ChatCompletion:
    """Validate a batch result body; drops service_tier ("batch" is not an openai literal)."""
    return ChatCompletion.model_validate(
        {k: v for k, v in body.items() if k != "service_tier"}
    )


def content_filter_completion(model: str) -> ChatCompletion:
    """Empty completion for a blocked output, as the sync API would return it."""
    return ChatCompletion.model_validate(
        {
            "id": "content-filter",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "content_filter",
                }
            ],
        }
    )


class OpenRouterBatcher(Batcher[ChatCompletion, list[dict[str, Any]]]):
    """Inspect batcher backed by OpenRouter's inline Batch API."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        config: BatchConfig,
        retry_config: ModelRetryConfig,
    ) -> None:
        super().__init__(config, retry_config, MAX_BATCH_REQUESTS, MAX_BATCH_SIZE_MB)
        self._client = client
        self._model = model

    async def _create_batch(self, batch: list[BatchRequest[ChatCompletion]]) -> str:
        # OpenRouter stream-parses the body: endpoint and model must precede requests.
        body = {
            "endpoint": "/v1/chat/completions",
            "model": self._model,
            "requests": [
                {"custom_id": request.custom_id, "body": request_body(request.request)}
                for request in batch
            ],
        }
        response = cast(
            dict[str, Any],
            await self._client.post("/batches", cast_to=object, body=body),
        )
        return str(response["id"])

    async def _check_batch(
        self, batch: Batch[ChatCompletion]
    ) -> BatchCheckResult[list[dict[str, Any]]]:
        info = cast(
            dict[str, Any],
            await self._client.get(f"/batches/{batch.id}", cast_to=object),
        )
        status = info["status"]
        if status not in PENDING_STATUSES and status != "completed":
            raise RuntimeError(
                f"OpenRouter batch {batch.id} {status}: {info.get('error')}"
            )
        counts = info.get("request_counts") or {}
        return BatchCheckResult(
            completed_count=counts.get("completed", 0),
            failed_count=counts.get("failed", 0),
            created_at=info["created_at"],
            completion_info=info["results"] if status == "completed" else None,
        )

    async def _handle_batch_result(
        self, batch: Batch[ChatCompletion], completion_info: list[dict[str, Any]]
    ) -> dict[str, ChatCompletion | Exception]:
        results: dict[str, ChatCompletion | Exception] = {
            custom_id: RuntimeError(
                f"No result for request {custom_id} in batch {batch.id}"
            )
            for custom_id in batch.requests
        }
        for result in completion_info:
            response = result.get("response") or {}
            if response.get("status_code") == 200:
                # Parse per result: a bad one must not fail the whole (already paid) batch.
                try:
                    results[result["custom_id"]] = parse_completion(response["body"])
                except Exception as ex:
                    results[result["custom_id"]] = ex
            elif CONTENT_FILTER_MESSAGE in str(
                (result.get("error") or {}).get("message", "")
            ):
                results[result["custom_id"]] = content_filter_completion(self._model)
            else:
                results[result["custom_id"]] = RuntimeError(
                    f"OpenRouter batch request failed: {response.get('body') or result.get('error')}"
                )
        return results


class OpenRouterBatchAPI(OpenRouterAPI):
    """OpenRouter provider that sends every completion through the Batch API."""

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(batch_model_name(model_name), *args, **kwargs)
        self._batcher: OpenRouterBatcher | None = None

    async def _generate_completion(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ChatCompletion:
        if self._batcher is None:
            self._batcher = OpenRouterBatcher(
                self.client,
                self.service_model_name(),
                normalized_batch_config(config.batch) or BatchConfig(),
                model_retry_config(
                    self.model_name,
                    config.max_retries,
                    config.timeout,
                    self.should_retry,
                    lambda ex: None,
                    log_model_retry,
                ),
            )
        return await self._batcher.generate_for_request(request)


@modelapi(name="openrouter-batch")
def openrouter_batch() -> type[OpenRouterBatchAPI]:
    return OpenRouterBatchAPI
