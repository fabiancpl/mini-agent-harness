"""Tests for `read_file`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.errors import PathOutsideRootError, ToolError
from mini_agent.sandbox import Sandbox
from mini_agent.tools.common import MAX_FILE_BYTES
from mini_agent.tools.read import read_file


@pytest.fixture
def poem(sandbox: Sandbox) -> Sandbox:
    (sandbox.root / "poem.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    return sandbox


# --- reading -------------------------------------------------------------------------------


def test_returns_numbered_lines_with_a_header(poem: Sandbox) -> None:
    assert read_file(poem, "poem.txt").splitlines() == [
        "poem.txt (lines 1-3 of 3):",
        "1  alpha",
        "2  beta",
        "3  gamma",
    ]


def test_reads_a_nested_file(sandbox: Sandbox) -> None:
    (sandbox.root / "a").mkdir()
    (sandbox.root / "a" / "b.txt").write_text("deep\n", encoding="utf-8")

    assert "deep" in read_file(sandbox, "a/b.txt")


def test_reports_an_empty_file(sandbox: Sandbox) -> None:
    (sandbox.root / "empty.txt").write_text("", encoding="utf-8")

    assert read_file(sandbox, "empty.txt") == "empty.txt is empty (0 bytes)"


def test_preserves_content_exactly(sandbox: Sandbox) -> None:
    (sandbox.root / "code.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    assert "    return 1" in read_file(sandbox, "code.py")  # indentation survives


# --- windowing -----------------------------------------------------------------------------


def test_starts_at_the_requested_line(poem: Sandbox) -> None:
    assert read_file(poem, "poem.txt", start_line=2).splitlines() == [
        "poem.txt (lines 2-3 of 3):",
        "2  beta",
        "3  gamma",
    ]


def test_limits_the_number_of_lines_and_says_how_to_continue(poem: Sandbox) -> None:
    lines = read_file(poem, "poem.txt", max_lines=2).splitlines()

    assert lines[0] == "poem.txt (lines 1-2 of 3):"
    assert lines[-1] == "... 1 more line. Call again with start_line=3."


def test_paging_through_a_file_reaches_the_end(poem: Sandbox) -> None:
    second_page = read_file(poem, "poem.txt", start_line=3, max_lines=2)

    assert "more lines" not in second_page  # no continuation note on the last page


def test_line_numbers_are_right_aligned(sandbox: Sandbox) -> None:
    (sandbox.root / "many.txt").write_text("\n".join(str(n) for n in range(1, 11)), "utf-8")

    lines = read_file(sandbox, "many.txt").splitlines()

    assert lines[1] == " 1  1"  # padded to the width of the widest number, '10'
    assert lines[10] == "10  10"


def test_rejects_a_start_line_past_the_end(poem: Sandbox) -> None:
    with pytest.raises(ToolError, match="only 3 lines"):
        read_file(poem, "poem.txt", start_line=9)


@pytest.mark.parametrize("start_line", [0, -1])
def test_rejects_a_start_line_below_one(poem: Sandbox, start_line: int) -> None:
    with pytest.raises(ToolError, match="start_line"):
        read_file(poem, "poem.txt", start_line=start_line)


def test_rejects_a_max_lines_below_one(poem: Sandbox) -> None:
    with pytest.raises(ToolError, match="max_lines"):
        read_file(poem, "poem.txt", max_lines=0)


# --- refusals ------------------------------------------------------------------------------


def test_rejects_a_missing_file(sandbox: Sandbox) -> None:
    with pytest.raises(ToolError, match="No such file"):
        read_file(sandbox, "nope.txt")


def test_rejects_a_directory(sandbox: Sandbox) -> None:
    (sandbox.root / "docs").mkdir()

    with pytest.raises(ToolError, match="list_directory"):  # points at the right tool
        read_file(sandbox, "docs")


def test_rejects_a_file_over_the_size_limit(sandbox: Sandbox) -> None:
    (sandbox.root / "big.txt").write_bytes(b"x" * (MAX_FILE_BYTES + 1))

    with pytest.raises(ToolError, match="byte limit"):
        read_file(sandbox, "big.txt")


def test_rejects_a_binary_file(sandbox: Sandbox) -> None:
    (sandbox.root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

    with pytest.raises(ToolError, match="not a UTF-8 text file"):
        read_file(sandbox, "image.png")


def test_refuses_to_read_outside_the_root(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        read_file(sandbox, "../outside.txt")


def test_refuses_to_read_through_a_symlink_out_of_the_root(
    sandbox: Sandbox, outside_file: Path
) -> None:
    (sandbox.root / "shortcut.txt").symlink_to(outside_file)

    with pytest.raises(PathOutsideRootError):
        read_file(sandbox, "shortcut.txt")
