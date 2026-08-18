"""Shared fixtures.

Deliberately tiny: `tmp_path` plus a couple of factories cover the whole suite. No mocking
library -- fakes you can read beat mocks you have to decode.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from mini_agent.llm import Message, ToolCall
from mini_agent.sandbox import Sandbox

API_KEY_ENV = "MINI_AGENT_TEST_KEY"

#: The smallest config that loads. Tests deep-copy this and mutate the copy.
BASE_CONFIG: dict[str, Any] = {
    "llm": {
        "base_url": "https://example.test/v1",
        "model": "test-model",
        "api_key_env": API_KEY_ENV,
    },
    "agent": {"root_path": "./workspace"},
}


def base_config() -> dict[str, Any]:
    """A fresh, mutable copy of the minimal valid config."""
    return copy.deepcopy(BASE_CONFIG)


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a config mapping to disk and return its path, with the API key env var set."""
    monkeypatch.setenv(API_KEY_ENV, "sk-test-key")

    def _write(data: dict[str, Any] | None = None, *, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(base_config() if data is None else data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    """A sandbox rooted at an empty directory inside tmp_path.

    The root is a *subdirectory* of tmp_path on purpose: it leaves somewhere outside the
    root but inside the test's scratch space for escape attempts to aim at.
    """
    root = tmp_path / "root"
    root.mkdir()
    return Sandbox(root)


@pytest.fixture
def outside_file(tmp_path: Path) -> Path:
    """A file that exists next to the sandbox root, but outside it. Must stay unreachable."""
    secret = tmp_path / "outside.txt"
    secret.write_text("top secret\n", encoding="utf-8")
    return secret


# --- driving the agent without a network ---------------------------------------------------


class FakeLLMClient:
    """An `LLMClient` stand-in that replays a script of prepared messages.

    Writing the model's side of the conversation by hand is what makes the loop testable:
    every branch (answer immediately, call a tool, batch two calls, never stop) is just a
    different script. It also records what it was sent, so tests can check the history the
    agent built.
    """

    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.received: list[list[dict[str, Any]]] = []  # conversation as of each request
        self.received_tools: list[list[dict[str, Any]] | None] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Message:
        # Deep-copied because the agent keeps mutating the same list afterwards.
        self.received.append(copy.deepcopy(messages))
        self.received_tools.append(tools)
        if not self.responses:
            raise AssertionError("The agent asked for more responses than the test scripted")
        return self.responses.pop(0)


def answer(text: str) -> Message:
    """A model message with no tool calls: the loop treats this as the final answer."""
    return Message(content=text)


def acts(thought: str | None, *calls: tuple[str, dict[str, Any]]) -> Message:
    """A model message that reasons and then calls one or more tools.

    `thought` may be None: models routinely return tool calls with no accompanying text,
    and the CLI has to render that step without a blank line where the reasoning would be.
    """
    return Message(
        content=thought,
        tool_calls=tuple(
            ToolCall(id=f"call_{index}", name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(calls)
        ),
    )
