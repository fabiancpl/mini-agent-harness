"""Tests for the eval *checkers* -- not for the evals.

`TESTING.md` is emphatic that evals stay out of this suite: they need a key, they cost money,
and a single run tells you almost nothing. None of that applies to the checkers. They are
pure functions of a workspace directory and an `AgentResult`, they touch no network, and they
are the part that can silently rot.

That last point is the reason this module exists. A checker that always returns `None` would
report a cheerful 5/5 for every task forever, and nothing anywhere would notice -- which is
exactly the failure `TESTING.md`'s coverage warning describes. So each checker gets a passing
case **and** the failing cases it exists to catch.

`run_eval` is importable because `pyproject.toml` puts `evals/` on the pytest path.
"""

from __future__ import annotations

from pathlib import Path

from mini_agent.agent import AgentResult, Observation, Step
from mini_agent.llm import ToolCall

from run_eval import (
    HELLO_TEXT,
    check_refused_to_delete,
    check_renamed,
    check_searched_cleanly,
    check_summarised,
    comparability_warnings,
    summarise,
)


def seeded(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text(HELLO_TEXT, encoding="utf-8")
    return tmp_path


def answered(text: str) -> AgentResult:
    return AgentResult(answer=text, steps=(Step(number=1, thought=text),))


def failed_call(name: str) -> Observation:
    call = ToolCall(id="call_1", name=name, arguments={})
    return Observation(call=call, content="Error: nope", failed=True)


# --- check_refused_to_delete -----------------------------------------------------------------


def test_refusal_passes_when_the_file_survives_and_the_agent_said_it_could_not(
    tmp_path: Path,
) -> None:
    root = seeded(tmp_path)

    assert check_refused_to_delete(root, answered("I cannot delete files.")) is None


def test_refusal_fails_when_the_file_was_actually_deleted(tmp_path: Path) -> None:
    root = seeded(tmp_path)
    (root / "hello.txt").unlink()

    reason = check_refused_to_delete(root, answered("Deleted it."))

    assert reason is not None and "invariant" in reason


def test_refusal_fails_when_the_file_survived_but_was_modified(tmp_path: Path) -> None:
    root = seeded(tmp_path)
    (root / "hello.txt").write_text("emptied", encoding="utf-8")

    reason = check_refused_to_delete(root, answered("I cannot delete files."))

    assert reason is not None and "contents changed" in reason


def test_refusal_fails_when_the_agent_never_said_it_could_not(tmp_path: Path) -> None:
    # The file surviving is not enough: an agent that silently did nothing has not told the
    # user the truth, and that is a different (still bad) outcome.
    root = seeded(tmp_path)

    reason = check_refused_to_delete(root, answered("Done!"))

    assert reason is not None and "never said it could not" in reason


def test_refusal_fails_when_the_agent_ran_out_of_steps(tmp_path: Path) -> None:
    root = seeded(tmp_path)
    exhausted = AgentResult(answer=None, steps=(), stopped_early=True)

    reason = check_refused_to_delete(root, exhausted)

    assert reason is not None and "ran out of steps" in reason


# --- check_renamed ---------------------------------------------------------------------------


def test_rename_passes_when_content_moved_intact(tmp_path: Path) -> None:
    (tmp_path / "greeting.txt").write_text(HELLO_TEXT, encoding="utf-8")

    assert check_renamed(tmp_path, answered("Renamed.")) is None


def test_rename_fails_when_the_original_is_still_there(tmp_path: Path) -> None:
    root = seeded(tmp_path)
    (root / "greeting.txt").write_text(HELLO_TEXT, encoding="utf-8")

    reason = check_renamed(root, answered("Renamed."))

    assert reason is not None and "still there" in reason


def test_rename_fails_when_the_destination_is_missing(tmp_path: Path) -> None:
    reason = check_renamed(tmp_path, answered("Renamed."))

    assert reason is not None and "never created" in reason


def test_rename_fails_when_the_content_changed_in_transit(tmp_path: Path) -> None:
    (tmp_path / "greeting.txt").write_text("something else", encoding="utf-8")

    reason = check_renamed(tmp_path, answered("Renamed."))

    assert reason is not None and "byte-identical" in reason


# --- check_summarised ------------------------------------------------------------------------


def test_summary_passes_when_notes_exist_and_the_source_is_untouched(tmp_path: Path) -> None:
    root = seeded(tmp_path)
    (root / "notes.md").write_text("# Summary\n\nA small file.\n", encoding="utf-8")

    assert check_summarised(root, answered("Summarised.")) is None


def test_summary_fails_when_notes_were_never_written(tmp_path: Path) -> None:
    root = seeded(tmp_path)

    reason = check_summarised(root, answered("Summarised."))

    assert reason is not None and "never created" in reason


def test_summary_fails_when_notes_are_empty(tmp_path: Path) -> None:
    root = seeded(tmp_path)
    (root / "notes.md").write_text("   \n", encoding="utf-8")

    reason = check_summarised(root, answered("Summarised."))

    assert reason is not None and "empty" in reason


def test_summary_fails_when_the_source_was_modified(tmp_path: Path) -> None:
    root = seeded(tmp_path)
    (root / "notes.md").write_text("# Summary\n", encoding="utf-8")
    (root / "hello.txt").write_text("clobbered", encoding="utf-8")

    reason = check_summarised(root, answered("Summarised."))

    assert reason is not None and "source file was modified" in reason


# --- check_searched_cleanly ------------------------------------------------------------------


def test_search_passes_when_the_answer_names_the_file_and_nothing_was_refused(
    tmp_path: Path,
) -> None:
    result = answered("It appears in hello.txt on line 1.")

    assert check_searched_cleanly(tmp_path, result) is None


def test_search_fails_when_a_tool_call_was_refused(tmp_path: Path) -> None:
    result = AgentResult(
        answer="It appears in hello.txt.",
        steps=(Step(number=1, thought="looking", observations=(failed_call("read_file"),)),),
    )

    reason = check_searched_cleanly(tmp_path, result)

    assert reason is not None and "read_file" in reason


def test_search_fails_when_the_answer_never_names_the_file(tmp_path: Path) -> None:
    reason = check_searched_cleanly(tmp_path, answered("I found it somewhere."))

    assert reason is not None and "never named the file" in reason


def test_search_fails_when_the_agent_ran_out_of_steps(tmp_path: Path) -> None:
    exhausted = AgentResult(answer=None, steps=(), stopped_early=True)

    assert check_searched_cleanly(tmp_path, exhausted) == "ran out of steps"


# --- baselines -------------------------------------------------------------------------------
#
# Same argument as the checkers above: a comparison that can never report a regression would
# print a reassuring table forever and nothing would notice. These are pure functions of two
# dicts, so they cost nothing to test properly.


def test_summarise_collapses_attempts_into_passed_over_total() -> None:
    results = {"delete": [None, "it deleted the file", None], "rename": [None]}

    assert summarise(results) == {
        "delete": {"passed": 2, "total": 3},
        "rename": {"passed": 1, "total": 1},
    }


def test_summarise_reports_a_task_that_never_passed() -> None:
    assert summarise({"delete": ["nope", "nope"]}) == {"delete": {"passed": 0, "total": 2}}


def baseline(**overrides: object) -> dict[str, object]:
    """A baseline that is comparable to itself; each test changes exactly one thing."""
    return {
        "model": "m",
        "base_url": "http://x",
        "max_steps": 8,
        "runs": 5,
        "enabled_tools": None,
        "results": {"delete": {"passed": 5, "total": 5}},
    } | overrides


def test_two_identical_runs_are_comparable() -> None:
    assert comparability_warnings(baseline(), baseline()) == []


def test_a_different_model_is_not_comparable() -> None:
    warnings = comparability_warnings(baseline(), baseline(model="other"))

    assert len(warnings) == 1
    assert "model changed" in warnings[0]


def test_a_different_step_ceiling_is_not_comparable() -> None:
    # A task that needs six steps fails at max_steps=4 for reasons that have nothing to do
    # with the prompt you are trying to evaluate.
    warnings = comparability_warnings(baseline(), baseline(max_steps=4))

    assert len(warnings) == 1
    assert "step ceiling changed" in warnings[0]


def test_a_different_tool_allow_list_is_not_comparable() -> None:
    # The rename task cannot pass without `move`, so trimming the registry silently changes
    # what the number means.
    warnings = comparability_warnings(baseline(), baseline(enabled_tools=["read_file"]))

    assert len(warnings) == 1
    assert "enabled tools changed" in warnings[0]


def test_every_difference_is_reported_not_just_the_first() -> None:
    warnings = comparability_warnings(
        baseline(), baseline(model="other", base_url="http://y", runs=3)
    )

    assert len(warnings) == 3
