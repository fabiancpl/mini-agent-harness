"""A thin client for the OpenAI-compatible `/chat/completions` endpoint.

"OpenAI-compatible" is a de-facto standard: OpenAI, Ollama, vLLM, LM Studio, llama.cpp, and
most gateways all speak it. Pointing `base_url` at a local server is enough to run this
whole project offline and for free.

The only job here is turning HTTP into two small dataclasses. Everything the agent needs to
decide -- whether to act or answer -- comes from `Message.tool_calls` being empty or not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from .config import LLMConfig
from .errors import LLMError


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

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach {url}: {exc}") from exc

        if response.status_code >= 400:
            # The body usually says exactly what is wrong (bad key, unknown model, context
            # length), so pass it through rather than hiding it behind the status code.
            raise LLMError(f"{url} returned HTTP {response.status_code}: {response.text[:500]}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"{url} did not return JSON: {response.text[:200]!r}") from exc

        return _parse_response(data)


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
