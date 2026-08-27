"""Tests for the ReAct loop.

Driven by `FakeLLMClient`: each test writes the model's side of the conversation by hand, so
every branch of the loop is just a different script. Real tools run against a real temporary
sandbox, so these double as end-to-end tests of everything below the network.
"""

from __future__ import annotations

import pytest
from conftest import FakeLLMClient, acts, answer  # pytest puts tests/ on sys.path

from mini_agent.agent import DEFAULT_SYSTEM_PROMPT, Agent, Step
from mini_agent.errors import LLMError
from mini_agent.registry import ToolRegistry
from mini_agent.sandbox import Sandbox
from mini_agent.tools import build_registry


@pytest.fixture
def registry(sandbox: Sandbox) -> ToolRegistry:
    (sandbox.root / "notes.txt").write_text("hello\n", encoding="utf-8")
    return build_registry(sandbox)


def make_agent(responses, registry: ToolRegistry, **kwargs) -> tuple[Agent, FakeLLMClient]:
    llm = FakeLLMClient(responses)
    return Agent(llm, registry, **kwargs), llm


# --- terminating ----------------------------------------------------------------------------


def test_answers_immediately_when_no_tool_is_needed(registry: ToolRegistry) -> None:
    agent, _ = make_agent([answer("Nothing to do.")], registry)

    result = agent.run("say hello")

    assert result.answer == "Nothing to do."
    assert result.stopped_early is False
    assert result.tool_calls_made == 0


def test_acts_once_and_then_answers(registry: ToolRegistry) -> None:
    agent, _ = make_agent(
        [
            acts("Let me look at the file.", ("read_file", {"path": "notes.txt"})),
            answer("The file says hello."),
        ],
        registry,
    )

    result = agent.run("what is in notes.txt?")

    assert result.answer == "The file says hello."
    assert len(result.steps) == 2
    assert "hello" in result.steps[0].observations[0].content


def test_runs_several_steps_in_order(registry: ToolRegistry) -> None:
    agent, _ = make_agent(
        [
            acts("First, look around.", ("list_directory", {})),
            acts("Now read it.", ("read_file", {"path": "notes.txt"})),
            acts("Add a file.", ("write_file", {"path": "new.txt", "content": "x\n"})),
            answer("Done."),
        ],
        registry,
    )

    result = agent.run("explore and write")

    assert [step.number for step in result.steps] == [1, 2, 3, 4]
    assert [step.thought for step in result.steps][-1] == "Done."


def test_the_final_step_records_the_answer_as_its_thought(registry: ToolRegistry) -> None:
    agent, _ = make_agent([answer("All finished.")], registry)

    result = agent.run("task")

    assert result.steps[-1] == Step(number=1, thought="All finished.", observations=())


def test_the_tools_actually_change_the_sandbox(registry: ToolRegistry, sandbox: Sandbox) -> None:
    agent, _ = make_agent(
        [
            acts("Writing.", ("write_file", {"path": "out/report.md", "content": "# Report\n"})),
            answer("Written."),
        ],
        registry,
    )

    agent.run("write a report")

    assert (sandbox.root / "out" / "report.md").read_text(encoding="utf-8") == "# Report\n"


def test_the_tools_compose_into_a_safe_refactor(registry: ToolRegistry, sandbox: Sandbox) -> None:
    # The workflow the tool set is designed around: find it, back it up, then change it.
    # Nothing is ever destroyed, because the backup exists before the edit happens.
    agent, _ = make_agent(
        [
            acts("Where is it?", ("search_text", {"pattern": "hello"})),
            acts("Back it up first.", ("copy", {"source": "notes.txt", "destination": "notes.bak"})),
            acts(
                "Now change it.",
                ("edit_file", {"path": "notes.txt", "old_text": "hello", "new_text": "goodbye"}),
            ),
            answer("Backed up to notes.bak and updated notes.txt."),
        ],
        registry,
    )

    result = agent.run("change hello to goodbye, safely")

    assert "notes.txt:1: hello" in result.steps[0].observations[0].content
    assert (sandbox.root / "notes.bak").read_text(encoding="utf-8") == "hello\n"
    assert (sandbox.root / "notes.txt").read_text(encoding="utf-8") == "goodbye\n"
    assert not any(observation.failed for step in result.steps for observation in step.observations)


def test_a_refused_overwrite_becomes_an_observation(registry: ToolRegistry, sandbox: Sandbox) -> None:
    # The model tries to clobber a file, is refused, and adapts -- the run survives and the
    # file it aimed at is still there.
    (sandbox.root / "keep.txt").write_text("precious\n", encoding="utf-8")
    agent, _ = make_agent(
        [
            acts("Renaming.", ("move", {"source": "notes.txt", "destination": "keep.txt"})),
            answer("That name was taken, so I left everything alone."),
        ],
        registry,
    )

    result = agent.run("rename notes.txt to keep.txt")

    assert result.steps[0].observations[0].failed is True
    assert "already exists" in result.steps[0].observations[0].content
    assert (sandbox.root / "keep.txt").read_text(encoding="utf-8") == "precious\n"
    assert result.answer == "That name was taken, so I left everything alone."


def test_batches_several_tool_calls_from_one_message(registry: ToolRegistry) -> None:
    agent, _ = make_agent(
        [
            acts(
                "Two things at once.",
                ("list_directory", {}),
                ("read_file", {"path": "notes.txt"}),
            ),
            answer("Both done."),
        ],
        registry,
    )

    result = agent.run("do two things")

    assert len(result.steps[0].observations) == 2
    assert result.tool_calls_made == 2


# --- the step limit ---------------------------------------------------------------------------


def test_stops_at_max_steps_without_inventing_an_answer(registry: ToolRegistry) -> None:
    # A model stuck in a loop must not have its last partial thought reported as success.
    agent, _ = make_agent([acts("Again.", ("list_directory", {}))] * 3, registry, max_steps=3)

    result = agent.run("loop forever")

    assert result.answer is None
    assert result.stopped_early is True
    assert len(result.steps) == 3


def test_max_steps_counts_model_turns(registry: ToolRegistry) -> None:
    agent, llm = make_agent([acts("Again.", ("list_directory", {}))] * 5, registry, max_steps=2)

    agent.run("loop")

    assert len(llm.received) == 2  # the model is asked exactly max_steps times


def test_answering_on_the_last_allowed_step_still_counts_as_success(
    registry: ToolRegistry,
) -> None:
    agent, _ = make_agent(
        [acts("Looking.", ("list_directory", {})), answer("Found it.")], registry, max_steps=2
    )

    result = agent.run("look")

    assert result.answer == "Found it."
    assert result.stopped_early is False


# --- recovering from the model's mistakes -------------------------------------------------------


def test_an_unknown_tool_becomes_an_observation_and_the_loop_continues(
    registry: ToolRegistry,
) -> None:
    agent, _ = make_agent(
        [acts("Deleting it.", ("delete_file", {"path": "notes.txt"})), answer("I cannot do that.")],
        registry,
    )

    result = agent.run("delete notes.txt")

    observation = result.steps[0].observations[0]
    assert observation.failed is True
    assert "Unknown tool" in observation.content
    assert result.answer == "I cannot do that."  # the model recovered instead of crashing


def test_a_tool_error_becomes_an_observation(registry: ToolRegistry) -> None:
    agent, _ = make_agent(
        [acts("Reading.", ("read_file", {"path": "missing.txt"})), answer("It is not there.")],
        registry,
    )

    result = agent.run("read missing.txt")

    assert result.steps[0].observations[0].failed is True
    assert "No such file" in result.steps[0].observations[0].content


def test_an_escape_attempt_becomes_an_observation(registry: ToolRegistry) -> None:
    # The refusal reaches the model as text, so it can adjust -- and the run survives it.
    agent, _ = make_agent(
        [
            acts("Peeking outside.", ("read_file", {"path": "../../etc/passwd"})),
            answer("That is outside my workspace."),
        ],
        registry,
    )

    result = agent.run("read /etc/passwd")

    assert "outside the sandbox root" in result.steps[0].observations[0].content
    assert result.answer == "That is outside my workspace."


def test_bad_arguments_become_an_observation(registry: ToolRegistry) -> None:
    agent, _ = make_agent(
        [acts("Reading.", ("read_file", {"file": "notes.txt"})), answer("Retrying next time.")],
        registry,
    )

    result = agent.run("read notes.txt")

    assert "Bad arguments" in result.steps[0].observations[0].content


def test_an_llm_error_is_not_swallowed(registry: ToolRegistry) -> None:
    # Tool failures are recoverable; a broken endpoint is not, and must surface.
    class BrokenClient:
        def complete(self, messages, tools=None):
            raise LLMError("endpoint is down")

    with pytest.raises(LLMError):
        Agent(BrokenClient(), registry).run("anything")


# --- the conversation the agent builds --------------------------------------------------------


def test_the_first_request_is_the_system_prompt_and_the_task(registry: ToolRegistry) -> None:
    agent, llm = make_agent([answer("ok")], registry)

    agent.run("summarise the workspace")

    assert llm.received[0] == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "summarise the workspace"},
    ]


def test_a_custom_system_prompt_replaces_the_default(registry: ToolRegistry) -> None:
    agent, llm = make_agent([answer("ok")], registry, system_prompt="Be terse.")

    agent.run("task")

    assert llm.received[0][0] == {"role": "system", "content": "Be terse."}


def test_tool_schemas_are_sent_on_every_request(registry: ToolRegistry) -> None:
    agent, llm = make_agent(
        [acts("Looking.", ("list_directory", {})), answer("ok")], registry
    )

    agent.run("task")

    for tools in llm.received_tools:
        assert [entry["function"]["name"] for entry in tools] == registry.names()


def test_the_history_pairs_each_tool_result_with_its_call(registry: ToolRegistry) -> None:
    # If the tool_call_id does not match, the API rejects the next request outright.
    agent, llm = make_agent(
        [acts("Reading.", ("read_file", {"path": "notes.txt"})), answer("done")], registry
    )

    agent.run("task")

    second_request = llm.received[1]
    assistant, tool_result = second_request[2], second_request[3]
    assert assistant["role"] == "assistant"
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == assistant["tool_calls"][0]["id"]


def test_the_history_grows_by_one_assistant_and_one_tool_message_per_call(
    registry: ToolRegistry,
) -> None:
    agent, llm = make_agent(
        [
            acts("Two.", ("list_directory", {}), ("read_file", {"path": "notes.txt"})),
            answer("done"),
        ],
        registry,
    )

    agent.run("task")

    # system + user, then the assistant message and one result per tool call.
    assert [message["role"] for message in llm.received[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]


def test_the_observation_text_is_what_the_model_is_shown(registry: ToolRegistry) -> None:
    agent, llm = make_agent(
        [acts("Reading.", ("read_file", {"path": "notes.txt"})), answer("done")], registry
    )

    result = agent.run("task")

    assert llm.received[1][-1]["content"] == result.steps[0].observations[0].content


# --- observability ------------------------------------------------------------------------------


def test_on_step_is_called_once_per_step_as_it_completes(registry: ToolRegistry) -> None:
    seen: list[Step] = []
    agent, _ = make_agent(
        [acts("Looking.", ("list_directory", {})), answer("done")],
        registry,
        on_step=seen.append,
    )

    result = agent.run("task")

    assert [step.number for step in seen] == [1, 2]
    assert seen == list(result.steps)


def test_the_loop_itself_prints_nothing(registry: ToolRegistry, capsys) -> None:
    # Keeping I/O in the CLI is what lets the agent be used as a library, and tested quietly.
    agent, _ = make_agent(
        [acts("Looking.", ("list_directory", {})), answer("done")], registry
    )

    agent.run("task")

    assert capsys.readouterr().out == ""


# --- the transcript ---------------------------------------------------------------------------


def test_the_result_carries_the_whole_conversation(registry: ToolRegistry) -> None:
    agent, _ = make_agent(
        [acts("Looking.", ("list_directory", {})), answer("All done.")], registry
    )

    result = agent.run("what is here?")

    assert [message["role"] for message in result.messages] == [
        "system",
        "user",
        "assistant",  # the thought plus the tool call
        "tool",  # the observation
        "assistant",  # the final answer
    ]
    assert result.messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert result.messages[1]["content"] == "what is here?"
    assert result.messages[-1]["content"] == "All done."


def test_the_transcript_is_what_the_client_was_actually_sent(registry: ToolRegistry) -> None:
    """The point of exposing it is fidelity, so check it against the client's own record."""
    agent, llm = make_agent(
        [acts("Looking.", ("list_directory", {})), answer("All done.")], registry
    )

    result = agent.run("task")

    # The last request carried everything except the answer that ended the run.
    assert list(result.messages[:-1]) == llm.received[-1]


def test_the_transcript_survives_running_out_of_steps(registry: ToolRegistry) -> None:
    # This is the run you most want to read afterwards, so it must not come back empty.
    agent, _ = make_agent([acts("Again.", ("list_directory", {}))] * 3, registry, max_steps=3)

    result = agent.run("task")

    assert result.stopped_early
    assert [message["role"] for message in result.messages] == [
        "system", "user",
        "assistant", "tool",
        "assistant", "tool",
        "assistant", "tool",
    ]


def test_exposing_the_transcript_did_not_make_the_loop_stateful(registry: ToolRegistry) -> None:
    """Each run must still start from the system prompt -- PLAN.md §11 depends on it."""
    agent, _ = make_agent([answer("first"), answer("second")], registry)

    first = agent.run("one")
    second = agent.run("two")

    assert [message["content"] for message in second.messages[1:]] == ["two", "second"]
    assert len(second.messages) == len(first.messages)
