"""Tests for the six tools that change the workspace.

Plus the negative space: these six are the *only* ways the agent can change the disk, and
none of them can make existing content unreachable. `move` and `copy` earn that by refusing
an existing destination, which is the property most of their tests are about.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.errors import PathOutsideRootError, ToolError
from mini_agent.sandbox import Sandbox
from mini_agent.tools.read import read_file
from mini_agent.tools.write import (
    append_to_file,
    copy,
    create_directory,
    edit_file,
    move,
    write_file,
)

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


# --- append_to_file ---------------------------------------------------------------------------


def test_appends_to_an_existing_file(sandbox: Sandbox) -> None:
    (sandbox.root / "log.md").write_text("- first\n", encoding="utf-8")

    result = append_to_file(sandbox, "log.md", "- second\n")

    assert (sandbox.root / "log.md").read_text(encoding="utf-8") == "- first\n- second\n"
    assert result == "Appended 9 characters to log.md (now 2 lines)"


def test_appending_never_touches_what_was_already_there(sandbox: Sandbox) -> None:
    # The defining property: this is the one mutation that cannot destroy anything.
    original = "important data\n"
    (sandbox.root / "log.md").write_text(original, encoding="utf-8")

    append_to_file(sandbox, "log.md", "extra\n")

    assert (sandbox.root / "log.md").read_text(encoding="utf-8").startswith(original)


def test_appending_creates_the_file_when_it_is_missing(sandbox: Sandbox) -> None:
    append_to_file(sandbox, "new.md", "hello\n")

    assert (sandbox.root / "new.md").read_text(encoding="utf-8") == "hello\n"


def test_appending_creates_missing_parent_directories(sandbox: Sandbox) -> None:
    append_to_file(sandbox, "a/b/log.md", "x\n")

    assert (sandbox.root / "a" / "b" / "log.md").is_file()


def test_appending_nothing_leaves_the_file_unchanged(sandbox: Sandbox) -> None:
    (sandbox.root / "log.md").write_text("unchanged\n", encoding="utf-8")

    append_to_file(sandbox, "log.md", "")

    assert (sandbox.root / "log.md").read_text(encoding="utf-8") == "unchanged\n"


def test_rejects_appending_to_a_directory(sandbox: Sandbox) -> None:
    (sandbox.root / "docs").mkdir()

    with pytest.raises(ToolError, match="is a directory"):
        append_to_file(sandbox, "docs", "text")


def test_append_refuses_to_escape(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        append_to_file(sandbox, "../outside.txt", "leaked")

    assert outside_file.read_text(encoding="utf-8") == "top secret\n"


def test_append_reports_an_os_error(sandbox: Sandbox) -> None:
    (sandbox.root / "notes.txt").write_text("x\n", encoding="utf-8")

    with pytest.raises(ToolError, match="Could not append"):
        append_to_file(sandbox, "notes.txt/child.txt", "text")


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


# --- move -------------------------------------------------------------------------------------


@pytest.fixture
def project(sandbox: Sandbox) -> Sandbox:
    """A file and a small directory tree to relocate."""
    (sandbox.root / "draft.md").write_text("# Draft\n", encoding="utf-8")
    (sandbox.root / "src").mkdir()
    (sandbox.root / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (sandbox.root / "src" / "util.py").write_text("x = 1\n", encoding="utf-8")
    return sandbox


def test_renames_a_file(project: Sandbox) -> None:
    result = move(project, "draft.md", "final.md")

    assert (project.root / "final.md").read_text(encoding="utf-8") == "# Draft\n"
    assert not (project.root / "draft.md").exists()
    assert result == "Moved draft.md to final.md"


def test_moves_a_file_into_a_subdirectory(project: Sandbox) -> None:
    move(project, "draft.md", "src/draft.md")

    assert (project.root / "src" / "draft.md").is_file()


def test_moving_creates_missing_parent_directories(project: Sandbox) -> None:
    move(project, "draft.md", "a/b/draft.md")

    assert (project.root / "a" / "b" / "draft.md").is_file()


def test_moves_a_directory_with_everything_in_it(project: Sandbox) -> None:
    move(project, "src", "lib")

    assert (project.root / "lib" / "main.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert (project.root / "lib" / "util.py").is_file()
    assert not (project.root / "src").exists()


def test_a_move_can_be_undone_by_moving_back(project: Sandbox) -> None:
    # The property that makes move non-destructive: it has an exact inverse.
    move(project, "draft.md", "somewhere/else.md")
    move(project, "somewhere/else.md", "draft.md")

    assert (project.root / "draft.md").read_text(encoding="utf-8") == "# Draft\n"


def test_refuses_to_move_onto_an_existing_file(project: Sandbox) -> None:
    (project.root / "final.md").write_text("do not lose me\n", encoding="utf-8")

    with pytest.raises(ToolError, match="already exists"):
        move(project, "draft.md", "final.md")


def test_refuses_to_move_onto_an_existing_directory(project: Sandbox) -> None:
    with pytest.raises(ToolError, match="already exists"):
        move(project, "draft.md", "src")


def test_a_refused_move_leaves_both_paths_untouched(project: Sandbox) -> None:
    (project.root / "final.md").write_text("do not lose me\n", encoding="utf-8")

    with pytest.raises(ToolError):
        move(project, "draft.md", "final.md")

    assert (project.root / "draft.md").read_text(encoding="utf-8") == "# Draft\n"
    assert (project.root / "final.md").read_text(encoding="utf-8") == "do not lose me\n"


def test_refuses_to_move_something_that_does_not_exist(sandbox: Sandbox) -> None:
    with pytest.raises(ToolError, match="No such file or directory to move"):
        move(sandbox, "ghost.md", "elsewhere.md")


def test_refuses_to_move_a_directory_into_itself(project: Sandbox) -> None:
    with pytest.raises(ToolError, match="into itself"):
        move(project, "src", "src/nested")


def test_move_refuses_an_escaping_source(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        move(sandbox, "../outside.txt", "stolen.txt")

    assert outside_file.is_file()


def test_move_refuses_an_escaping_destination(project: Sandbox, tmp_path: Path) -> None:
    # Both ends of a relocation are confined; checking only the source would be a hole.
    with pytest.raises(PathOutsideRootError):
        move(project, "draft.md", "../smuggled.md")

    assert not (tmp_path / "smuggled.md").exists()
    assert (project.root / "draft.md").is_file()


def test_move_reports_an_os_error(project: Sandbox) -> None:
    with pytest.raises(ToolError, match="Could not move"):
        move(project, "src", "draft.md/inside")


# --- copy -------------------------------------------------------------------------------------


def test_copies_a_file(project: Sandbox) -> None:
    result = copy(project, "draft.md", "draft.backup.md")

    assert (project.root / "draft.backup.md").read_text(encoding="utf-8") == "# Draft\n"
    assert result == "Copied draft.md to draft.backup.md"


def test_copying_leaves_the_source_in_place(project: Sandbox) -> None:
    copy(project, "draft.md", "draft.backup.md")

    assert (project.root / "draft.md").read_text(encoding="utf-8") == "# Draft\n"


def test_copies_a_directory_recursively(project: Sandbox) -> None:
    result = copy(project, "src", "src_v2")

    assert (project.root / "src_v2" / "main.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert (project.root / "src_v2" / "util.py").is_file()
    assert (project.root / "src" / "main.py").is_file()  # original untouched
    assert result == "Copied src to src_v2 (2 files)"


def test_copying_creates_missing_parent_directories(project: Sandbox) -> None:
    copy(project, "draft.md", "backups/2026/draft.md")

    assert (project.root / "backups" / "2026" / "draft.md").is_file()


def test_refuses_to_copy_onto_an_existing_path(project: Sandbox) -> None:
    (project.root / "final.md").write_text("do not lose me\n", encoding="utf-8")

    with pytest.raises(ToolError, match="already exists"):
        copy(project, "draft.md", "final.md")

    assert (project.root / "final.md").read_text(encoding="utf-8") == "do not lose me\n"


def test_refuses_to_copy_something_that_does_not_exist(sandbox: Sandbox) -> None:
    with pytest.raises(ToolError, match="No such file or directory to copy"):
        copy(sandbox, "ghost.md", "elsewhere.md")


def test_refuses_to_copy_a_directory_into_itself(project: Sandbox) -> None:
    with pytest.raises(ToolError, match="into itself"):
        copy(project, "src", "src/nested")


def test_copy_refuses_an_escaping_source(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        copy(sandbox, "../outside.txt", "stolen.txt")

    assert not (sandbox.root / "stolen.txt").exists()


def test_copy_refuses_an_escaping_destination(project: Sandbox, tmp_path: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        copy(project, "draft.md", "../smuggled.md")

    assert not (tmp_path / "smuggled.md").exists()


def test_copy_reports_an_os_error(project: Sandbox) -> None:
    with pytest.raises(ToolError, match="Could not copy"):
        copy(project, "src", "draft.md/inside")


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
