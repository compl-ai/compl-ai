"""Turn HTTP errors inside OpenAI batch result lines into API errors.

A batch result line can succeed at the batch level (``"error": null``) while its
``response`` carries an HTTP error, e.g. a policy refusal::

    {"response": {"status_code": 400, "body": {"error": {"code": "cyber_policy", ...}}}, "error": null}

Inspect's ``OpenAIBatcher`` validates such a body as a model response, which raises
and fails the sample (and with fail_on_error the task). Returning the matching
``openai`` status error instead sends it down the same path as a synchronous request:
``openai_handle_bad_request`` maps policy refusals (``cyber_policy``,
``content_policy_violation``, ...) to an empty ``content_filter`` output.
"""

from typing import Any

import httpx
from inspect_ai.model._providers._openai_batch import OpenAIBatcher


def _patch() -> None:
    parse = OpenAIBatcher._parse_jsonl_line
    if getattr(parse, "_complai_http_errors", False):
        return

    def patched(self: OpenAIBatcher, line_data: dict[str, Any]) -> tuple[str, Any]:
        response = line_data.get("response") or {}
        status = response.get("status_code") or 200
        if line_data.get("error") is None and status >= 400:
            error = self._openai_client._make_status_error_from_response(  # pyright: ignore[reportPrivateUsage]
                httpx.Response(
                    status_code=status,
                    json=response.get("body"),
                    request=httpx.Request("POST", "https://api.openai.com/v1/batches"),
                )
            )
            return str(line_data.get("custom_id", "")), error
        return parse(self, line_data)

    patched._complai_http_errors = True  # type: ignore[attr-defined]
    OpenAIBatcher._parse_jsonl_line = patched  # type: ignore[method-assign]


_patch()
