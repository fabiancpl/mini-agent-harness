"""Tests for `create_directory`, `write_file`, and `edit_file`.

Plus the negative space: these three are the *only* ways the agent can change the disk, and
none of them can remove anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.errors import PathOutsideRootError, ToolError
from mini_agent.sandbox import Sandbox
from mini_agent.tools.read import read_file
from mini_agent.tools.write import create_directory, edit_file, write_file

# --- create_directory -----------------------------------------------------------------------


def test_creates_a_directory(sandbox: Sandbox) -> None:
    assert create_directory(sandbox, "docs") == "Created directory: docs"
    assert (sandbox.root / "docs").is_dir()


def test_creates_missing_parent_directories(sandbox: Sandbox) -> None:
    create_directory(sandbox, "a/b/c")

    assert (sandbox.root / "a" / "b" / "c").is_dir()


def test_creating_an_existing_directory_is_not_an_error(sandbox: Sandbox) -> None:
    create_directory(sandbox, "docs")

    assert create_directory(sandbox, "docs") == "Directory already exists: docs"


def test_rejects_creating_a_directory_where_a_file_exists(sandbox: Sandbox) -> None:
    (sandbox.root / "docs").write_text("i am a file\n", encoding="utf-8")

    with pytest.raises(ToolError, match="a file is already there"):
        create_directory(sandbox, "docs")


def test_create_directory_refuses_to_escape(sandbox: Sandbox) -> None:
    with pytest.raises(PathOutsideRootError):
        create_directory(sandbox, "../escaped")


# --- write_file -----------------------------------------------------------------------------


def test_writes_a_new_file(sandbox: Sandbox) -> None:
    result = write_file(sandbox, "notes.txt", "hello\nworld\n")

    assert (sandbox.root / "notes.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert result == "Created notes.txt (12 characters, 2 lines)"


def test_writing_creates_missing_parent_directories(sandbox: Sandbox) -> None:
    write_file(sandbox, "a/b/deep.txt", "content\n")

    assert (sandbox.root / "a" / "b" / "deep.txt").is_file()


def test_writing_an_existing_file_replaces_it_entirely(sandbox: Sandbox) -> None:
    write_file(sandbox, "notes.txt", "first\n")

    result = write_file(sandbox, "notes.txt", "second\n")

    assert (sandbox.root / "notes.txt").read_text(encoding="utf-8") == "second\n"
    assert result.startswith("Overwrote")  # the model is told it destroyed the old contents


def test_writes_an_empty_file(sandbox: Sandbox) -> None:
    write_file(sandbox, "empty.txt", "")

    assert (sandbox.root / "empty.txt").read_text(encoding="utf-8") == ""


def test_rejects_writing_over_a_directory(sandbox: Sandbox) -> None:
    (sandbox.root / "docs").mkdir()

    with pytest.raises(ToolError, match="is a directory"):
        write_file(sandbox, "docs", "content")


def test_write_file_refuses_to_escape(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        write_file(sandbox, "../outside.txt", "overwritten!")

    assert outside_file.read_text(encoding="utf-8") == "top secret\n"  # untouched


def test_write_file_refuses_an_absolute_path_outside_the_root(sandbox: Sandbox) -> None:
    with pytest.raises(PathOutsideRootError):
        write_file(sandbox, "/tmp/mini-agent-should-not-exist.txt", "nope")


def test_write_then_read_round_trips(sandbox: Sandbox) -> None:
    write_file(sandbox, "round.txt", "alpha\nbeta\n")

    assert read_file(sandbox, "round.txt").splitlines()[1:] == ["1  alpha", "2  beta"]


# --- edit_file ------------------------------------------------------------------------------


@pytest.fixture
def greeting(sandbox: Sandbox) -> Sandbox:
    (sandbox.root / "hello.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    return sandbox


def test_replaces_an_exact_unique_string(greeting: Sandbox) -> None:
    edit_file(greeting, "hello.py", "'hi'", "'hello'")

    assert (greeting.root / "hello.py").read_text(encoding="utf-8") == (
        "def hello():\n    return 'hello'\n"
    )


def test_reports_what_it_changed(greeting: Sandbox) -> None:
    assert edit_file(greeting, "hello.py", "'hi'", "'hello'") == (
        "Edited hello.py: replaced 4 characters with 7."
    )


def test_replaces_a_multi_line_string(greeting: Sandbox) -> None:
    edit_file(greeting, "hello.py", "def hello():\n    return 'hi'", "def hello():\n    pass")

    assert "pass" in (greeting.root / "hello.py").read_text(encoding="utf-8")


def test_an_empty_new_text_removes_the_matched_text_but_keeps_the_file(greeting: Sandbox) -> None:
    edit_file(greeting, "hello.py", "    return 'hi'\n", "")

    assert (greeting.root / "hello.py").read_text(encoding="utf-8") == "def hello():\n"
    assert (greeting.root / "hello.py").is_file()  # emptying is possible, deleting is not


def test_rejects_text_that_is_not_in_the_file(greeting: Sandbox) -> None:
    with pytest.raises(ToolError, match="must match the file exactly"):
        edit_file(greeting, "hello.py", "return 'bye'", "return 'hello'")


def test_rejects_an_ambiguous_edit(sandbox: Sandbox) -> None:
    # Silently editing the first of several matches would eventually pick the wrong one.
    (sandbox.root / "dup.txt").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")

    with pytest.raises(ToolError, match="appears 2 times"):
        edit_file(sandbox, "dup.txt", "x = 1", "x = 3")


def test_an_ambiguous_edit_leaves_the_file_untouched(sandbox: Sandbox) -> None:
    original = "x = 1\ny = 2\nx = 1\n"
    (sandbox.root / "dup.txt").write_text(original, encoding="utf-8")

    with pytest.raises(ToolError):
        edit_file(sandbox, "dup.txt", "x = 1", "x = 3")

    assert (sandbox.root / "dup.txt").read_text(encoding="utf-8") == original


def test_rejects_an_empty_old_text(greeting: Sandbox) -> None:
    with pytest.raises(ToolError, match="must not be empty"):
        edit_file(greeting, "hello.py", "", "injected")


def test_rejects_editing_a_missing_file(sandbox: Sandbox) -> None:
    with pytest.raises(ToolError, match="write_file"):  # points at the right tool
        edit_file(sandbox, "nope.py", "a", "b")


def test_rejects_editing_a_directory(sandbox: Sandbox) -> None:
    (sandbox.root / "docs").mkdir()

    with pytest.raises(ToolError, match="is a directory"):
        edit_file(sandbox, "docs", "a", "b")


def test_rejects_editing_a_binary_file(sandbox: Sandbox) -> None:
    (sandbox.root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

    with pytest.raises(ToolError, match="not a UTF-8 text file"):
        edit_file(sandbox, "image.png", "PNG", "GIF")


def test_edit_file_refuses_to_escape(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        edit_file(sandbox, "../outside.txt", "top secret", "leaked")

    assert outside_file.read_text(encoding="utf-8") == "top secret\n"  # untouched


def test_edit_file_refuses_to_follow_a_symlink_out_of_the_root(
    sandbox: Sandbox, outside_file: Path
) -> None:
    (sandbox.root / "shortcut.txt").symlink_to(outside_file)

    with pytest.raises(PathOutsideRootError):
        edit_file(sandbox, "shortcut.txt", "top secret", "leaked")

    assert outside_file.read_text(encoding="utf-8") == "top secret\n"


# --- filesystem failures reach the model as text, not as a crash ---------------------------


def test_create_directory_reports_an_os_error(sandbox: Sandbox) -> None:
    # 'notes.txt' is a file, so it cannot have children. The OS says NotADirectoryError;
    # the model needs to hear it as an observation it can act on.
    (sandbox.root / "notes.txt").write_text("x\n", encoding="utf-8")

    with pytest.raises(ToolError, match="Could not create directory"):
        create_directory(sandbox, "notes.txt/child")


def test_write_file_reports_an_os_error(sandbox: Sandbox) -> None:
    (sandbox.root / "notes.txt").write_text("x\n", encoding="utf-8")

    with pytest.raises(ToolError, match="Could not write"):
        write_file(sandbox, "notes.txt/child.txt", "content")


def test_edit_file_reports_an_os_error(
    greeting: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failing write is hard to arrange honestly (and impossible as root), so simulate the
    # disk-full / permission-denied case directly.
    def explode(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", explode)

    with pytest.raises(ToolError, match="Could not write"):
        edit_file(greeting, "hello.py", "'hi'", "'hello'")
