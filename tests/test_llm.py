"""Tests for the OpenAI-compatible client.

The network is never touched: `requests.post` is monkeypatched, which also lets the tests
assert on exactly what would have gone over the wire.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
import requests

from mini_agent import llm as llm_module
from mini_agent.config import LLMConfig
from mini_agent.errors import LLMError
from mini_agent.llm import LLMClient, Message, ToolCall

CONFIG = LLMConfig(
    base_url="https://example.test/v1",
    model="test-model",
    api_key="sk-test-key",
    temperature=0.25,
    max_tokens=512,
    timeout_seconds=7,
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {"name": "read_file", "description": "Read a file.", "parameters": {}},
    }
]


class FakeResponse:
    def __init__(
        self,
        payload: Any = None,
        *,
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload)
        self.headers = headers or {}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


@pytest.fixture
def post(monkeypatch: pytest.MonkeyPatch):
    """Replace requests.post; return a recorder holding the call and the reply to give."""

    class Recorder:
        def __init__(self) -> None:
            self.url: str | None = None
            self.kwargs: dict[str, Any] = {}
            self.response = FakeResponse(chat_response("ok"))
            self.error: Exception | None = None
            #: Optional script for the retry tests: one entry consumed per call, each either a
            #: FakeResponse to return or an exception to raise. `response`/`error` still work
            #: unchanged when this is empty, which is what every other test in here uses.
            self.replies: list[Any] = []
            self.calls = 0

        def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
            self.url, self.kwargs = url, kwargs
            self.calls += 1
            if self.replies:
                reply = self.replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply
            if self.error is not None:
                raise self.error
            return self.response

        @property
        def payload(self) -> dict[str, Any]:
            return self.kwargs["json"]

    recorder = Recorder()
    monkeypatch.setattr(llm_module.requests, "post", recorder)
    return recorder


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch):
    """Replace the `time` module *inside llm.py* so retries never really sleep.

    Autouse on purpose: a suite that takes three real seconds to prove a connection failure is
    a suite people stop running. Replacing the name in llm.py's namespace mirrors how
    `requests.post` is already faked, needs no production parameter that exists only for tests,
    and lets a test assert the exact backoff sequence rather than merely tolerate it.
    """

    class FakeClock:
        def __init__(self) -> None:
            self.slept: list[float] = []

        def sleep(self, seconds: float) -> None:
            self.slept.append(seconds)

    fake = FakeClock()
    monkeypatch.setattr(llm_module, "time", fake)
    return fake


def chat_response(content: str | None, tool_calls: list[dict] | None = None) -> dict[str, Any]:
    """A realistic /chat/completions body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"id": "chatcmpl-1", "choices": [{"index": 0, "message": message}]}


def function_call(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# --- the request ---------------------------------------------------------------------------


def test_posts_to_the_chat_completions_endpoint(post) -> None:
    LLMClient(CONFIG).complete([{"role": "user", "content": "hi"}])

    assert post.url == "https://example.test/v1/chat/completions"


def test_sends_the_api_key_as_a_bearer_token(post) -> None:
    LLMClient(CONFIG).complete([{"role": "user", "content": "hi"}])

    assert post.kwargs["headers"]["Authorization"] == "Bearer sk-test-key"


def test_sends_the_model_and_sampling_settings_from_the_config(post) -> None:
    LLMClient(CONFIG).complete([{"role": "user", "content": "hi"}])

    assert post.payload["model"] == "test-model"
    assert post.payload["temperature"] == 0.25
    assert post.payload["max_tokens"] == 512
    assert post.kwargs["timeout"] == 7


def test_sends_the_conversation_verbatim(post) -> None:
    messages = [{"role": "system", "content": "be careful"}, {"role": "user", "content": "hi"}]

    LLMClient(CONFIG).complete(messages)

    assert post.payload["messages"] == messages


def test_advertises_tools_and_lets_the_model_choose(post) -> None:
    LLMClient(CONFIG).complete([{"role": "user", "content": "hi"}], TOOL_SCHEMAS)

    assert post.payload["tools"] == TOOL_SCHEMAS
    # "auto" is required: forcing a tool would remove the model's ability to answer, which
    # is exactly the branch that ends the ReAct loop.
    assert post.payload["tool_choice"] == "auto"


def test_omits_the_tools_key_when_there_are_none(post) -> None:
    LLMClient(CONFIG).complete([{"role": "user", "content": "hi"}], [])

    assert "tools" not in post.payload
    assert "tool_choice" not in post.payload


# --- parsing a reply -----------------------------------------------------------------------


def test_parses_a_plain_answer(post) -> None:
    post.response = FakeResponse(chat_response("All done."))

    message = LLMClient(CONFIG).complete([])

    assert message == Message(content="All done.", tool_calls=())


def test_parses_a_tool_call(post) -> None:
    post.response = FakeResponse(
        chat_response("Reading it.", [function_call("read_file", '{"path": "a.txt"}')])
    )

    message = LLMClient(CONFIG).complete([])

    assert message.content == "Reading it."
    assert message.tool_calls == (
        ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"}),
    )


def test_parses_several_tool_calls_in_one_message(post) -> None:
    post.response = FakeResponse(
        chat_response(
            None,
            [
                function_call("list_directory", "{}", "call_a"),
                function_call("read_file", '{"path": "a.txt"}', "call_b"),
            ],
        )
    )

    message = LLMClient(CONFIG).complete([])

    assert [call.name for call in message.tool_calls] == ["list_directory", "read_file"]
    assert message.content is None


def test_parses_empty_arguments(post) -> None:
    post.response = FakeResponse(chat_response(None, [function_call("list_directory", "")]))

    assert LLMClient(CONFIG).complete([]).tool_calls[0].arguments == {}


def test_accepts_arguments_sent_as_an_object(post) -> None:
    # Off-spec, but several local servers do it. Accepting it costs one isinstance check.
    function = {"name": "read_file", "arguments": {"path": "a"}}
    call = {"id": "c", "type": "function", "function": function}
    post.response = FakeResponse(chat_response(None, [call]))

    assert LLMClient(CONFIG).complete([]).tool_calls[0].arguments == {"path": "a"}


def test_substitutes_an_id_when_the_server_omits_one(post) -> None:
    call = {"type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    post.response = FakeResponse(chat_response(None, [call]))

    assert LLMClient(CONFIG).complete([]).tool_calls[0].id == "call_0_read_file"


def test_substituted_ids_are_unique_within_one_message(post) -> None:
    """Two calls to the same tool must not end up sharing an id.

    The id is the only thing pairing a tool result with the call that asked for it. Naming the
    substitute after the tool alone meant a model reading two files in one turn produced two
    results both tagged `call_read_file`, and the pairing became a guess -- for the rest of the
    conversation, since the ambiguity is written into the history.
    """
    calls = [
        {"type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
        {"type": "function", "function": {"name": "read_file", "arguments": '{"path": "b"}'}},
    ]
    post.response = FakeResponse(chat_response(None, calls))

    ids = [call.id for call in LLMClient(CONFIG).complete([]).tool_calls]

    assert ids == ["call_0_read_file", "call_1_read_file"]
    assert len(set(ids)) == len(ids)


def test_a_server_supplied_id_is_always_preferred(post) -> None:
    # The substitute is a fallback, not a rewrite: if the server sent an id, it is the one the
    # server expects back on the tool result.
    calls = [
        {"id": "abc", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        {"type": "function", "function": {"name": "read_file", "arguments": "{}"}},
    ]
    post.response = FakeResponse(chat_response(None, calls))

    assert [call.id for call in LLMClient(CONFIG).complete([]).tool_calls] == [
        "abc",
        "call_1_read_file",
    ]


# --- failures ------------------------------------------------------------------------------


def test_reports_an_http_error_with_the_response_body(post) -> None:
    # The body says what is actually wrong -- bad key, unknown model, context too long.
    post.response = FakeResponse(None, status_code=401, text='{"error": "invalid api key"}')

    with pytest.raises(LLMError, match="invalid api key"):
        LLMClient(CONFIG).complete([])


def test_reports_a_connection_failure(post) -> None:
    post.error = requests.ConnectionError("connection refused")

    with pytest.raises(LLMError, match="Could not reach"):
        LLMClient(CONFIG).complete([])


def test_reports_a_non_json_response(post) -> None:
    post.response = FakeResponse(None, text="<html>502 Bad Gateway</html>")

    with pytest.raises(LLMError, match="did not return JSON"):
        LLMClient(CONFIG).complete([])


def test_reports_a_response_without_choices(post) -> None:
    post.response = FakeResponse({"id": "chatcmpl-1"})

    with pytest.raises(LLMError, match="no choices"):
        LLMClient(CONFIG).complete([])


def test_reports_invalid_json_arguments_with_the_raw_text(post) -> None:
    # A real failure mode of smaller models; the raw text is what makes it diagnosable.
    post.response = FakeResponse(chat_response(None, [function_call("read_file", "{path: a.txt")]))

    with pytest.raises(LLMError, match="invalid JSON arguments"):
        LLMClient(CONFIG).complete([])


def test_reports_arguments_that_are_not_an_object(post) -> None:
    post.response = FakeResponse(chat_response(None, [function_call("read_file", '"a.txt"')]))

    with pytest.raises(LLMError, match="not an object"):
        LLMClient(CONFIG).complete([])


def test_reports_a_tool_call_without_a_name(post) -> None:
    raw_call = {"id": "c", "function": {"arguments": "{}"}}
    post.response = FakeResponse(chat_response(None, [raw_call]))

    with pytest.raises(LLMError, match="no function name"):
        LLMClient(CONFIG).complete([])


# --- rendering a message back into history --------------------------------------------------


def test_history_entry_for_a_plain_answer() -> None:
    assert Message(content="done").to_history_entry() == {"role": "assistant", "content": "done"}


def test_history_entry_reserialises_tool_call_arguments() -> None:
    message = Message(
        content="Reading.",
        tool_calls=(ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"}),),
    )

    entry = message.to_history_entry()

    assert entry["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
        }
    ]


def test_history_entry_round_trips_through_the_parser() -> None:
    # What we send back must parse to what we received, or the conversation drifts.
    original = Message(
        content="Looking.",
        tool_calls=(ToolCall(id="call_1", name="find_files", arguments={"pattern": "*.py"}),),
    )

    reparsed = llm_module._parse_response({"choices": [{"message": original.to_history_entry()}]})

    assert reparsed == original


def test_base_url_with_a_trailing_slash_would_not_double_up(post) -> None:
    # config.load_config strips it, but the client should also compose a clean URL.
    LLMClient(LLMConfig(base_url="https://example.test/v1", model="m", api_key="k")).complete([])

    assert "//chat" not in post.url


# --- retries -------------------------------------------------------------------------------
#
# The `clock` fixture records what would have been slept, so these assert the backoff schedule
# instead of tolerating it. `post.replies` scripts one entry per attempt.


def test_a_transient_failure_is_retried_and_succeeds(post, clock) -> None:
    post.replies = [
        requests.ConnectionError("connection reset"),
        FakeResponse(chat_response("recovered")),
    ]

    message = LLMClient(CONFIG).complete([])

    assert message.content == "recovered"
    assert post.calls == 2
    assert clock.slept == [1.0]


def test_retries_stop_at_max_attempts(post, clock) -> None:
    post.replies = [requests.ConnectionError("nope")] * 3

    with pytest.raises(LLMError, match="after 3 attempts"):
        LLMClient(CONFIG).complete([])

    assert post.calls == 3
    assert clock.slept == [1.0, 2.0]  # waits between attempts, never after the last


def test_a_timeout_is_transient_too(post, clock) -> None:
    post.replies = [requests.Timeout("read timed out"), FakeResponse(chat_response("ok"))]

    assert LLMClient(CONFIG).complete([]).content == "ok"
    assert post.calls == 2


def test_a_server_error_is_retried(post, clock) -> None:
    post.replies = [
        FakeResponse(None, status_code=503, text="upstream busy"),
        FakeResponse(chat_response("ok")),
    ]

    assert LLMClient(CONFIG).complete([]).content == "ok"
    assert post.calls == 2


def test_an_exhausted_server_error_still_reports_the_server_body(post, clock) -> None:
    """Retrying must not make a failure less diagnosable than it was before.

    The body is where the server says what is actually wrong, so the last response is returned
    rather than swallowed, and the normal HTTP error is raised from it.
    """
    post.replies = [FakeResponse(None, status_code=503, text="upstream busy")] * 3

    with pytest.raises(LLMError, match="upstream busy"):
        LLMClient(CONFIG).complete([])

    assert post.calls == 3


def test_a_bad_api_key_is_not_retried(post, clock) -> None:
    # 401 will say 401 forever. Retrying it makes the error slower and no clearer.
    post.replies = [FakeResponse(None, status_code=401, text='{"error": "invalid api key"}')]

    with pytest.raises(LLMError, match="invalid api key"):
        LLMClient(CONFIG).complete([])

    assert post.calls == 1
    assert clock.slept == []


def test_a_malformed_url_is_not_retried(post, clock) -> None:
    # MissingSchema is a typo in base_url, not a transient fault. It used to be caught by the
    # blanket `except requests.RequestException`, which would now retry it three times.
    post.replies = [requests.exceptions.MissingSchema("no scheme")]

    with pytest.raises(requests.exceptions.MissingSchema):
        LLMClient(CONFIG).complete([])

    assert post.calls == 1


def test_max_attempts_of_one_disables_retrying(post, clock) -> None:
    post.replies = [requests.ConnectionError("nope")]

    with pytest.raises(LLMError, match="after 1 attempt"):
        LLMClient(replace(CONFIG, max_attempts=1)).complete([])

    assert post.calls == 1
    assert clock.slept == []


def test_a_429_waits_as_long_as_the_server_asked(post, clock) -> None:
    post.replies = [
        FakeResponse(None, status_code=429, text="slow down", headers={"Retry-After": "5"}),
        FakeResponse(chat_response("ok")),
    ]

    assert LLMClient(CONFIG).complete([]).content == "ok"
    assert clock.slept == [5.0]  # the server's number, not the default 1.0


def test_an_absurd_retry_after_is_capped(post, clock) -> None:
    # A misconfigured or hostile header must not hang the CLI for an hour.
    post.replies = [
        FakeResponse(None, status_code=429, text="slow down", headers={"Retry-After": "99999"}),
        FakeResponse(chat_response("ok")),
    ]

    LLMClient(CONFIG).complete([])

    assert clock.slept == [30.0]


def test_an_unparseable_retry_after_falls_back_to_the_backoff(post, clock) -> None:
    # The HTTP-date form of Retry-After is legal and not worth parsing; ignore what we cannot
    # read rather than crashing on it.
    post.replies = [
        FakeResponse(
            None,
            status_code=429,
            text="slow",
            headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
        ),
        FakeResponse(chat_response("ok")),
    ]

    LLMClient(CONFIG).complete([])

    assert clock.slept == [1.0]
