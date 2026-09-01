"""A thin client for the OpenAI-compatible `/chat/completions` endpoint.

"OpenAI-compatible" is a de-facto standard: OpenAI, Ollama, vLLM, LM Studio, llama.cpp, and
most gateways all speak it. Pointing `base_url` at a local server is enough to run this
whole project offline and for free.

The only job here is turning HTTP into two small dataclasses. Everything the agent needs to
decide -- whether to act or answer -- comes from `Message.tool_calls` being empty or not.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from .config import LLMConfig
from .errors import LLMError

#: Status codes worth sending the same request again. Listed rather than written as `>= 500`
#: because most 5xx are not transient in the way that matters: a 501 from a server that does
#: not implement tool calling will say 501 forever, and retrying only delays the error.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

#: Seconds to wait before attempt 2, 3, ... Doubling, and deliberately not jittered: jitter
#: exists to stop a fleet of clients retrying in lockstep, and there is one client here.
BACKOFF_SECONDS = (1.0, 2.0, 4.0)

#: A hostile or misconfigured `Retry-After` should not be able to hang the CLI for an hour.
MAX_RETRY_AFTER_SECONDS = 30.0


@dataclass(frozen=True)
class ToolCall:
    """One tool the model wants run, with its arguments already parsed from JSON."""

    id: str  # echoed back in the tool result so the model can match them up
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """What the model said: some text, some tool calls, or both."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_history_entry(self) -> dict[str, Any]:
        """Render this message back into the wire format, to append to the conversation.

        The model must see its own tool calls in the history, or the `tool` results that
        follow refer to nothing and the API rejects the next request. Note the round trip:
        arguments arrive as a JSON string, we parse them for the tool, and we re-serialise
        them here.
        """
        entry: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return entry


class SupportsComplete(Protocol):
    """Everything the agent needs from a model client: one method.

    `Agent` is annotated with this rather than with `LLMClient`, which makes the seam
    visible in the types -- swapping in another backend means writing one `complete`, not
    subclassing anything. The test suite has always relied on this (`FakeLLMClient` is not
    an `LLMClient`); the protocol just stops that from being a secret.

    Note the boundary honestly: `messages` are OpenAI-shaped dicts, and `Agent.run` builds
    them itself. A provider with a different message shape needs edits to the loop as well
    as a new client -- see EXTENDING.md.
    """

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Message: ...


class LLMClient:
    """Sends one chat completion request and parses one response."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Message:
        """Ask the model what to do next, given the conversation so far."""
        url = f"{self.config.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            # "auto" lets the model choose between calling a tool and answering. That choice
            # is precisely the branch the ReAct loop turns on, so never force it.
            payload["tool_choice"] = "auto"

        response, attempts = self._post_with_retries(url, payload)

        if response.status_code >= 400:
            # The body usually says exactly what is wrong (bad key, unknown model, context
            # length), so pass it through rather than hiding it behind the status code.
            tried = "" if attempts == 1 else f" after {attempts} attempts"
            raise LLMError(
                f"{url} returned HTTP {response.status_code}{tried}: {response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"{url} did not return JSON: {response.text[:200]!r}") from exc

        return _parse_response(data)

    def _post_with_retries(
        self, url: str, payload: dict[str, Any]
    ) -> tuple[requests.Response, int]:
        """POST, trying again on the failures that are worth trying again.

        Returns the last response and how many attempts it took. Note it *returns* a failing
        response rather than raising on one: the caller already turns a 4xx/5xx into an
        `LLMError` carrying the server's own explanation, and retries must not make a failure
        less diagnosable than it was before.

        Only two exception types are retried. `requests.RequestException` -- what this used to
        catch wholesale -- also covers `MissingSchema`, `InvalidURL` and `TooManyRedirects`,
        which are typos in `base_url`. Retrying a typo three times makes the error slower and
        no clearer.

        Retrying a POST is normally a question ("did the server already do the work?"), and it
        is safe here only because a chat completion has no server-side effect. The tools run on
        this machine, after the response comes back.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: requests.RequestException | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            response = None
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.config.timeout_seconds
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
            else:
                if response.status_code not in RETRY_STATUS:
                    return response, attempt  # success, or a failure retrying cannot fix
                last_error = None

            if attempt == self.config.max_attempts:
                break
            time.sleep(self._delay_before(attempt, response))

        if response is not None:
            return response, self.config.max_attempts  # out of attempts, still a real reply

        raise LLMError(
            f"Could not reach {url} after {_plural_attempts(self.config.max_attempts)}: "
            f"{last_error}"
        ) from last_error

    def _delay_before(self, attempt: int, response: requests.Response | None) -> float:
        """How long to wait before the next attempt: the server's answer if it gave one."""
        if response is not None and response.status_code == 429:
            # A 429 usually comes with the server telling you exactly how long to wait, and
            # guessing shorter than that just earns another 429. Only honour an integer, and
            # cap it: a misconfigured header should not hang the CLI for an hour.
            seconds = _parse_retry_after(response.headers.get("Retry-After"))
            if seconds is not None:
                return min(seconds, MAX_RETRY_AFTER_SECONDS)
        return BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS)) - 1]


def _plural_attempts(count: int) -> str:
    return "1 attempt" if count == 1 else f"{count} attempts"


def _parse_retry_after(value: str | None) -> float | None:
    """`Retry-After` as whole seconds, or None. The HTTP-date form is not worth the code."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _parse_response(data: dict[str, Any]) -> Message:
    choices = data.get("choices")
    if not choices:
        raise LLMError(f"Response contained no choices: {json.dumps(data)[:300]}")

    message = choices[0].get("message") or {}
    raw_calls = message.get("tool_calls") or []
    return Message(
        content=message.get("content"),
        # The index is only used to synthesise an id when the server omits one -- see below.
        tool_calls=tuple(_parse_tool_call(raw, index) for index, raw in enumerate(raw_calls)),
    )


def _parse_tool_call(raw: dict[str, Any], index: int = 0) -> ToolCall:
    function = raw.get("function") or {}
    name = function.get("name")
    if not name:
        raise LLMError(f"Tool call had no function name: {json.dumps(raw)[:200]}")

    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        # The spec says arguments are a JSON *string*. Smaller local models regularly emit
        # something that is not valid JSON, and a clear error beats a confusing crash.
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model sent invalid JSON arguments for {name}: {arguments!r}") from exc
    if not isinstance(arguments, dict):
        raise LLMError(f"Arguments for {name} were not an object: {arguments!r}")

    # Some servers omit the id. The agent only needs it to pair a result with its call, so any
    # unique string will do -- but it does have to be unique. Naming it after the tool alone was
    # not: a model that asks to read two files in one turn produces two calls named `read_file`,
    # so both results came back tagged `call_read_file` and nothing could say which was which.
    # The position in the message is what makes them distinct; the name is kept because a
    # visibly synthetic id is easier to recognise in a transcript than a bare number.
    return ToolCall(id=raw.get("id") or f"call_{index}_{name}", name=name, arguments=arguments)
