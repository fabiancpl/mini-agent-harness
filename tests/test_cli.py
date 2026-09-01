"""Tests for the command line.

`main` takes argv and an `llm_factory` and returns an exit code, so a whole session -- REPL
included -- runs inside a test with a scripted model and no network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mini_agent.cli import build_parser, main
from mini_agent.llm import Message, Usage

from conftest import (  # pytest puts tests/ on sys.path
    FakeLLMClient,
    acts,
    answer,
    base_config,
)

# --- helpers --------------------------------------------------------------------------------


def fake_llm(*responses):
    """An llm_factory that ignores the config and replays these messages."""
    return lambda config: FakeLLMClient(list(responses))


@pytest.fixture
def cli_config(write_config, tmp_path: Path) -> Path:
    """A config whose workspace contains one file, ready to be run against."""
    path = write_config()
    (tmp_path / "workspace").mkdir(exist_ok=True)
    (tmp_path / "workspace" / "notes.txt").write_text("hello\n", encoding="utf-8")
    return path


# --- argument parsing -------------------------------------------------------------------------


def test_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.config == "config.yaml"
    assert args.task is None
    assert args.verbose is False


def test_parses_every_flag() -> None:
    args = build_parser().parse_args(
        ["--config", "c.yaml", "--task", "do it", "--root", "/tmp/w",
         "--max-steps", "3", "--verbose"]
    )

    assert (args.config, args.task, args.root, args.max_steps) == ("c.yaml", "do it", "/tmp/w", 3)
    assert args.verbose is True


# --- one-shot mode ----------------------------------------------------------------------------


def test_runs_a_single_task_and_reports_the_answer(cli_config: Path, capsys) -> None:
    code = main(
        ["--config", str(cli_config), "--task", "read the notes"],
        llm_factory=fake_llm(
            acts("Reading.", ("read_file", {"path": "notes.txt"})),
            answer("The notes say hello."),
        ),
    )

    assert code == 0
    assert "The notes say hello." in capsys.readouterr().out


def test_prints_a_banner_with_the_workspace_model_and_tools(cli_config: Path, capsys) -> None:
    main(["--config", str(cli_config), "--task", "hi"], llm_factory=fake_llm(answer("hi")))

    output = capsys.readouterr().out
    assert "workspace:" in output
    assert "test-model" in output
    assert "read_file" in output


def test_the_agent_really_edits_the_workspace(cli_config: Path, tmp_path: Path) -> None:
    main(
        ["--config", str(cli_config), "--task", "write a file"],
        llm_factory=fake_llm(
            acts("Writing.", ("write_file", {"path": "out.txt", "content": "written\n"})),
            answer("Done."),
        ),
    )

    assert (tmp_path / "workspace" / "out.txt").read_text(encoding="utf-8") == "written\n"


def test_quiet_mode_prints_one_line_per_tool_call(cli_config: Path, capsys) -> None:
    main(
        ["--config", str(cli_config), "--task", "look"],
        llm_factory=fake_llm(acts("Looking.", ("list_directory", {})), answer("Done.")),
    )

    output = capsys.readouterr().out
    assert "- [1] list_directory()" in output
    assert "Looking." not in output  # reasoning is verbose-only


def test_verbose_mode_shows_the_thought_and_the_observation(cli_config: Path, capsys) -> None:
    main(
        ["--config", str(cli_config), "--task", "look", "--verbose"],
        llm_factory=fake_llm(acts("Looking around.", ("list_directory", {})), answer("Done.")),
    )

    output = capsys.readouterr().out
    assert "step 1" in output
    assert "Looking around." in output
    assert "notes.txt" in output  # the observation itself


def test_verbose_mode_handles_a_step_with_no_reasoning(cli_config: Path, capsys) -> None:
    # Models often return tool calls with content: None. The step must still render, and
    # without an empty line standing in for the reasoning that was never there.
    main(
        ["--config", str(cli_config), "--task", "look", "--verbose"],
        llm_factory=fake_llm(acts(None, ("list_directory", {})), answer("Done.")),
    )

    output = capsys.readouterr().out
    assert "step 1" in output
    assert "  -> list_directory()" in output
    assert "\n\n  ->" not in output  # no blank line where the thought would have gone


def test_failed_tool_calls_are_marked(cli_config: Path, capsys) -> None:
    main(
        ["--config", str(cli_config), "--task", "escape"],
        llm_factory=fake_llm(
            acts("Peeking.", ("read_file", {"path": "../../etc/passwd"})),
            answer("Refused."),
        ),
    )

    assert "! [1] read_file(" in capsys.readouterr().out


def test_long_arguments_are_truncated_on_screen(cli_config: Path, capsys) -> None:
    main(
        ["--config", str(cli_config), "--task", "write"],
        llm_factory=fake_llm(
            acts("Writing.", ("write_file", {"path": "big.txt", "content": "x" * 5000})),
            answer("Done."),
        ),
    )

    output = capsys.readouterr().out
    assert "more)" in output
    assert "x" * 200 not in output


def test_running_out_of_steps_is_reported_and_exits_non_zero(cli_config: Path, capsys) -> None:
    code = main(
        ["--config", str(cli_config), "--task", "loop", "--max-steps", "2"],
        llm_factory=fake_llm(*[acts("Again.", ("list_directory", {}))] * 2),
    )

    assert code == 1
    assert "Stopped after 2 steps" in capsys.readouterr().out


# --- overrides ---------------------------------------------------------------------------------


def test_root_override_wins_over_the_config(cli_config: Path, tmp_path: Path, capsys) -> None:
    elsewhere = tmp_path / "other_workspace"

    main(
        ["--config", str(cli_config), "--root", str(elsewhere), "--task", "write"],
        llm_factory=fake_llm(
            acts("Writing.", ("write_file", {"path": "here.txt", "content": "x\n"})),
            answer("Done."),
        ),
    )

    assert (elsewhere / "here.txt").is_file()
    assert not (tmp_path / "workspace" / "here.txt").exists()


def test_root_override_creates_the_directory(cli_config: Path, tmp_path: Path) -> None:
    new_root = tmp_path / "made" / "up"

    main(
        ["--config", str(cli_config), "--root", str(new_root), "--task", "hi"],
        llm_factory=fake_llm(answer("hi")),
    )

    assert new_root.is_dir()


def test_rejects_a_root_override_that_is_a_file(cli_config: Path, tmp_path: Path, capsys) -> None:
    a_file = tmp_path / "afile.txt"
    a_file.write_text("x", encoding="utf-8")

    code = main(
        ["--config", str(cli_config), "--root", str(a_file), "--task", "hi"],
        llm_factory=fake_llm(answer("hi")),
    )

    assert code == 1
    assert "not a directory" in capsys.readouterr().err


def test_max_steps_override_wins_over_the_config(cli_config: Path, capsys) -> None:
    code = main(
        ["--config", str(cli_config), "--task", "loop", "--max-steps", "1"],
        llm_factory=fake_llm(acts("Again.", ("list_directory", {}))),
    )

    assert code == 1
    assert "Stopped after 1 step" in capsys.readouterr().out


def test_rejects_a_max_steps_below_one(cli_config: Path, capsys) -> None:
    code = main(
        ["--config", str(cli_config), "--task", "hi", "--max-steps", "0"],
        llm_factory=fake_llm(answer("hi")),
    )

    assert code == 1
    assert "at least 1" in capsys.readouterr().err


def test_the_tools_allow_list_from_the_config_is_applied(write_config, capsys) -> None:
    from conftest import base_config

    data = base_config()
    data["tools"] = {"enabled": ["list_directory"]}

    main(["--config", str(write_config(data)), "--task", "hi"], llm_factory=fake_llm(answer("hi")))

    output = capsys.readouterr().out
    # The count is the point: a tool you registered but left out of the allow-list is
    # otherwise invisible, which is the trap this banner exists to spring.
    assert "tools:     1 of 10 registered — list_directory\n" in output
    assert "write_file" not in output


def test_the_banner_omits_the_count_when_every_tool_is_enabled(write_config, capsys) -> None:
    # No allow-list means nothing is being withheld, so "10 of 10" would be noise.
    main(["--config", str(write_config()), "--task", "hi"], llm_factory=fake_llm(answer("hi")))

    output = capsys.readouterr().out
    assert "tools:     list_directory" in output
    assert "registered" not in output


# --- configuration errors ------------------------------------------------------------------------


def test_a_missing_config_file_exits_one_with_a_single_line(tmp_path: Path, capsys) -> None:
    code = main(["--config", str(tmp_path / "nope.yaml"), "--task", "hi"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err  # a stack trace would blame our code, not theirs
    assert captured.out == ""


def test_an_unknown_tool_in_the_config_exits_one(write_config, capsys) -> None:
    from conftest import base_config

    data = base_config()
    data["tools"] = {"enabled": ["delete_everything"]}

    code = main(["--config", str(write_config(data)), "--task", "hi"], llm_factory=fake_llm())

    assert code == 1
    assert "do not exist" in capsys.readouterr().err


def test_an_llm_failure_during_a_run_exits_one(cli_config: Path, capsys) -> None:
    from mini_agent.errors import LLMError

    class BrokenClient:
        def complete(self, messages, tools=None):
            raise LLMError("endpoint is down")

    code = main(
        ["--config", str(cli_config), "--task", "hi"],
        llm_factory=lambda config: BrokenClient(),
    )

    assert code == 1
    assert "endpoint is down" in capsys.readouterr().err


# --- the REPL ---------------------------------------------------------------------------------


def feed(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    """Script stdin for the REPL, one input() call per line."""
    remaining = iter(lines)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None  # as if the user pressed Ctrl-D

    monkeypatch.setattr("builtins.input", fake_input)


def test_the_repl_runs_a_task_then_exits_on_command(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    feed(monkeypatch, "what is here?", "exit")

    code = main(["--config", str(cli_config)], llm_factory=fake_llm(answer("Just notes.txt.")))

    assert code == 0
    assert "Just notes.txt." in capsys.readouterr().out


def test_the_repl_exits_on_end_of_input(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed(monkeypatch)  # immediate Ctrl-D

    assert main(["--config", str(cli_config)], llm_factory=fake_llm()) == 0


def test_the_repl_ignores_blank_lines(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    feed(monkeypatch, "", "   ", "quit")

    assert main(["--config", str(cli_config)], llm_factory=fake_llm()) == 0


def test_the_repl_survives_an_llm_error_and_keeps_going(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from mini_agent.errors import LLMError

    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                raise LLMError("temporary outage")
            return answer("Second time worked.")

    feed(monkeypatch, "first task", "second task", "exit")

    code = main(["--config", str(cli_config)], llm_factory=lambda config: FlakyClient())

    captured = capsys.readouterr()
    assert code == 0
    assert "temporary outage" in captured.err
    assert "Second time worked." in captured.out  # the session continued


def test_ctrl_c_during_a_run_abandons_it_but_keeps_the_session(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    class ImpatientUser:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt  # the user pressed Ctrl-C mid-run
            return answer("Finished this time.")

    feed(monkeypatch, "a long task", "a short task", "exit")

    code = main(["--config", str(cli_config)], llm_factory=lambda config: ImpatientUser())

    captured = capsys.readouterr().out
    assert code == 0
    assert "(interrupted)" in captured
    assert "Finished this time." in captured


def test_run_exits_with_the_code_main_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    from mini_agent import cli

    monkeypatch.setattr(cli, "main", lambda: 3)

    with pytest.raises(SystemExit) as excinfo:
        cli.run()

    assert excinfo.value.code == 3


def test_the_module_is_runnable_with_python_dash_m() -> None:
    # The documented entry point, exercised the way a reader will actually type it.
    result = subprocess.run(
        [sys.executable, "-m", "mini_agent", "--help"], capture_output=True, text=True, check=True
    )

    assert "ReAct agent" in result.stdout


def test_a_missing_config_exits_one_from_the_real_entry_point(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mini_agent",
         "--config", str(tmp_path / "nope.yaml"), "--task", "x"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stderr.startswith("error: ")


def test_the_repl_says_up_front_that_turns_share_a_conversation(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # Both halves matter: that turns accumulate, and that `reset` is how you stop them.
    feed(monkeypatch, "exit")

    main(["--config", str(cli_config)], llm_factory=fake_llm())

    out = capsys.readouterr().out
    assert "Turns share one conversation" in out
    assert "reset" in out


def test_the_repl_carries_the_conversation_between_turns(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inverse of the 0.2.0 guard, rewritten deliberately rather than repaired.

    This is the release in one assertion: the second request carries the first exchange.
    """
    client = FakeLLMClient([answer("one"), answer("two")])
    feed(monkeypatch, "first", "second", "exit")

    main(["--config", str(cli_config)], llm_factory=lambda config: client)

    assert [message["content"] for message in client.received[1]][1:] == [
        "first",
        "one",
        "second",
    ]


def test_reset_forgets_the_conversation(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    client = FakeLLMClient([answer("one"), answer("two")])
    feed(monkeypatch, "first", "reset", "second", "exit")

    main(["--config", str(cli_config)], llm_factory=lambda config: client)

    # The second request starts over: system prompt and the new task, nothing else.
    assert len(client.received[1]) == 2
    assert client.received[1][1] == {"role": "user", "content": "second"}
    assert "Forgotten" in capsys.readouterr().out


def test_reset_does_not_count_as_a_task(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Typing `reset` must not spend a model call, the way a blank line does not.
    client = FakeLLMClient([answer("one")])
    feed(monkeypatch, "reset", "first", "exit")

    main(["--config", str(cli_config)], llm_factory=lambda config: client)

    assert len(client.received) == 1


# --- --dump-transcript -------------------------------------------------------------------------


def test_dump_transcript_writes_the_conversation_as_json(
    cli_config: Path, tmp_path: Path, capsys
) -> None:
    destination = tmp_path / "transcript.json"

    code = main(
        ["--config", str(cli_config), "--task", "look", "--dump-transcript", str(destination)],
        llm_factory=fake_llm(acts("Looking.", ("list_directory", {})), answer("Done.")),
    )

    assert code == 0
    transcript = json.loads(destination.read_text(encoding="utf-8"))
    assert [message["role"] for message in transcript] == [
        "system", "user", "assistant", "tool", "assistant"
    ]
    # The tool result must carry the id of the call it answers, or the API rejects the turn.
    assert transcript[3]["tool_call_id"] == transcript[2]["tool_calls"][0]["id"]
    assert f"(transcript written to {destination})" in capsys.readouterr().out


def test_no_transcript_is_written_without_the_flag(cli_config: Path, tmp_path: Path) -> None:
    main(["--config", str(cli_config), "--task", "look"], llm_factory=fake_llm(answer("Done.")))

    assert list(tmp_path.glob("*.json")) == []


def test_dump_transcript_creates_a_missing_parent_directory(
    cli_config: Path, tmp_path: Path
) -> None:
    # `--dump-transcript dumps/run.json` is how anyone keeps transcripts out of the way,
    # and the directory does not exist the first time.
    destination = tmp_path / "dumps" / "nested" / "transcript.json"

    code = main(
        ["--config", str(cli_config), "--task", "look", "--dump-transcript", str(destination)],
        llm_factory=fake_llm(answer("Done.")),
    )

    assert code == 0
    assert json.loads(destination.read_text(encoding="utf-8"))[0]["role"] == "system"


def test_an_unwritable_transcript_path_does_not_lose_the_run(
    cli_config: Path, tmp_path: Path, capsys
) -> None:
    # A file where the directory should be: no amount of mkdir makes this path writable.
    # (Being unreadable to *this* user would not do -- the suite may run as root.)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    destination = blocker / "t.json"

    # The answer is already on screen; a failed dump must not turn a good run into an error.
    code = main(
        ["--config", str(cli_config), "--task", "look", "--dump-transcript", str(destination)],
        llm_factory=fake_llm(answer("Done.")),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Done." in captured.out
    assert "could not write transcript" in captured.err


# --- context reporting -----------------------------------------------------------------------
#
# The whole of 0.3.0's answer to context growth: report it, warn about it, never act on it.


def answer_with_usage(text: str, total: int) -> Message:
    return Message(content=text, usage=Usage(prompt_tokens=total - 5, completion_tokens=5,
                                             total_tokens=total))


@pytest.fixture
def windowed_config(write_config, tmp_path: Path) -> Path:
    """A config that knows its model's context window, so percentages can be printed."""
    data = base_config()
    data["llm"]["context_window"] = 8000
    path = write_config(data)
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return path


def test_reports_tokens_against_the_window(windowed_config: Path, capsys) -> None:
    main(
        ["--config", str(windowed_config), "--task", "hi"],
        llm_factory=lambda config: FakeLLMClient([answer_with_usage("done", 1000)]),
    )

    assert "context: 1,000 of 8,000 tokens (12%)" in capsys.readouterr().out


def test_warns_when_the_window_is_nearly_full(
    windowed_config: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    client = FakeLLMClient([answer_with_usage("done", 7000)])
    feed(monkeypatch, "first", "exit")

    main(["--config", str(windowed_config)], llm_factory=lambda config: client)

    out = capsys.readouterr().out
    assert "running out of room" in out
    assert "reset" in out


def test_does_not_warn_with_room_to_spare(
    windowed_config: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    client = FakeLLMClient([answer_with_usage("done", 100)])
    feed(monkeypatch, "first", "exit")

    main(["--config", str(windowed_config)], llm_factory=lambda config: client)

    assert "running out of room" not in capsys.readouterr().out


def test_reports_exact_counts_when_the_server_sends_no_usage(cli_config: Path, capsys) -> None:
    # No estimate, no invented percentage: two true numbers and a note saying why.
    main(
        ["--config", str(cli_config), "--task", "hi"],
        llm_factory=fake_llm(answer("done")),
    )

    out = capsys.readouterr().out
    assert "reports no token usage" in out
    assert "messages" in out and "characters" in out


def test_reports_tokens_without_a_percentage_when_the_window_is_unknown(
    write_config, tmp_path: Path, capsys
) -> None:
    data = base_config()  # no llm.context_window
    (tmp_path / "workspace").mkdir(exist_ok=True)

    main(
        ["--config", str(write_config(data)), "--task", "hi"],
        llm_factory=lambda config: FakeLLMClient([answer_with_usage("done", 1234)]),
    )

    out = capsys.readouterr().out
    assert "context: 1,234 tokens" in out
    assert "%" not in out
