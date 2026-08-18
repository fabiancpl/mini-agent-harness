"""Tests for `list_directory`, `find_files`, and `search_text`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.errors import PathOutsideRootError, ToolError
from mini_agent.sandbox import Sandbox
from mini_agent.tools.common import MAX_FILE_BYTES
from mini_agent.tools.navigate import (
    MAX_LINE_CHARS,
    MAX_MATCHES,
    MAX_TEXT_MATCHES,
    find_files,
    list_directory,
    search_text,
)


@pytest.fixture
def populated(sandbox: Sandbox) -> Sandbox:
    """A small tree: two directories, three files, one nested."""
    (sandbox.root / "src").mkdir()
    (sandbox.root / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (sandbox.root / "docs").mkdir()
    (sandbox.root / "README.md").write_text("# readme\n", encoding="utf-8")
    (sandbox.root / "notes.txt").write_text("notes\n", encoding="utf-8")
    return sandbox


# --- list_directory ------------------------------------------------------------------------


def test_lists_directories_first_then_files_alphabetically(populated: Sandbox) -> None:
    output = list_directory(populated, ".")

    assert output.splitlines() == [
        ". (4 entries):",
        "docs/",
        "src/",
        "notes.txt",
        "README.md",
    ]


def test_lists_a_subdirectory(populated: Sandbox) -> None:
    assert list_directory(populated, "src").splitlines() == ["src (1 entry):", "main.py"]


def test_defaults_to_the_root(populated: Sandbox) -> None:
    assert list_directory(populated) == list_directory(populated, ".")


def test_reports_an_empty_directory(sandbox: Sandbox) -> None:
    assert list_directory(sandbox, ".") == ". is empty"


def test_list_directory_rejects_a_missing_path(sandbox: Sandbox) -> None:
    with pytest.raises(ToolError, match="No such directory"):
        list_directory(sandbox, "nope")


def test_list_directory_rejects_a_file(populated: Sandbox) -> None:
    with pytest.raises(ToolError, match="read_file"):  # the error suggests the right tool
        list_directory(populated, "README.md")


def test_list_directory_never_shows_a_host_absolute_path(populated: Sandbox) -> None:
    assert str(populated.root) not in list_directory(populated, "src")


def test_list_directory_refuses_to_escape(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        list_directory(sandbox, "..")


# --- find_files ----------------------------------------------------------------------------


def test_finds_files_recursively_by_extension(populated: Sandbox) -> None:
    output = find_files(populated, "*.py")

    assert output.splitlines() == ["1 match for '*.py':", str(Path("src") / "main.py")]


def test_finds_several_matches_sorted(populated: Sandbox) -> None:
    lines = find_files(populated, "*.md").splitlines()

    assert lines[0] == "1 match for '*.md':"
    assert lines[1:] == ["README.md"]


def test_searches_only_under_the_given_path(populated: Sandbox) -> None:
    assert "No files matching" in find_files(populated, "*.md", "src")


def test_reports_no_matches_clearly(populated: Sandbox) -> None:
    assert find_files(populated, "*.rs") == "No files matching '*.rs' under ."


def test_marks_matched_directories_with_a_slash(populated: Sandbox) -> None:
    assert "docs/" in find_files(populated, "docs")


def test_truncates_a_very_large_result_set_and_says_so(sandbox: Sandbox) -> None:
    for index in range(MAX_MATCHES + 5):
        (sandbox.root / f"file{index:03d}.txt").write_text("x", encoding="utf-8")

    lines = find_files(sandbox, "*.txt").splitlines()

    assert lines[0] == f"{MAX_MATCHES + 5} matches for '*.txt':"
    assert len(lines) == MAX_MATCHES + 2  # header + capped matches + the truncation note
    assert "5 more matches not shown" in lines[-1]


def test_find_files_rejects_a_missing_search_root(sandbox: Sandbox) -> None:
    with pytest.raises(ToolError, match="No such directory"):
        find_files(sandbox, "*.py", "nope")


def test_find_files_refuses_to_escape(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        find_files(sandbox, "*.txt", "../")


def test_find_files_does_not_descend_into_a_symlinked_directory(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "secret.txt").write_text("nope\n", encoding="utf-8")
    (sandbox.root / "link").symlink_to(tmp_path / "elsewhere")

    assert "secret.txt" not in find_files(sandbox, "*.txt")


def test_find_files_filters_out_a_symlink_to_a_file_outside_the_root(
    sandbox: Sandbox, outside_file: Path
) -> None:
    """The case the containment filter actually exists for.

    The test above passes for a reason that has nothing to do with our code: `rglob` simply
    does not descend into symlinked directories. A symlink to a *file* is different -- rglob
    yields it by name, so only the explicit `is_relative_to(sandbox.root)` check keeps a
    path pointing outside the root from being named to the model. Remove that check and
    this test is the one that fails.
    """
    (sandbox.root / "shortcut.txt").symlink_to(outside_file)

    assert find_files(sandbox, "*.txt") == "No files matching '*.txt' under ."


# --- search_text ----------------------------------------------------------------------------


@pytest.fixture
def sources(sandbox: Sandbox) -> Sandbox:
    """A tiny codebase to grep through."""
    (sandbox.root / "app.py").write_text("import os\n\ndef main():\n    return 1\n", "utf-8")
    (sandbox.root / "lib").mkdir()
    (sandbox.root / "lib" / "util.py").write_text("def helper():\n    return main()\n", "utf-8")
    (sandbox.root / "notes.md").write_text("call main() to start\n", encoding="utf-8")
    return sandbox


def test_reports_matches_as_path_line_text(sources: Sandbox) -> None:
    assert search_text(sources, "def main").splitlines() == [
        "1 match for 'def main':",
        "app.py:3: def main():",
    ]


def test_finds_matches_across_several_files(sources: Sandbox) -> None:
    lines = search_text(sources, "main").splitlines()

    assert lines[0] == "3 matches for 'main':"
    assert lines[1:] == [
        "app.py:3: def main():",
        str(Path("lib") / "util.py") + ":2:     return main()",
        "notes.md:1: call main() to start",
    ]


def test_reports_no_matches_clearly(sources: Sandbox) -> None:
    assert search_text(sources, "nonexistent") == "No matches for 'nonexistent' under ."


def test_the_pattern_is_a_regular_expression(sources: Sandbox) -> None:
    lines = search_text(sources, r"^def (main|helper)").splitlines()

    assert lines[0] == "2 matches for '^def (main|helper)':"


def test_rejects_an_invalid_regular_expression(sources: Sandbox) -> None:
    # Recoverable: the model reads this and writes a valid pattern next step.
    with pytest.raises(ToolError, match="not a valid regular expression"):
        search_text(sources, "def (main")


def test_file_pattern_restricts_which_files_are_searched(sources: Sandbox) -> None:
    output = search_text(sources, "main", file_pattern="*.md")

    assert "notes.md" in output
    assert "app.py" not in output


def test_path_narrows_the_search_to_a_subtree(sources: Sandbox) -> None:
    output = search_text(sources, "main", path="lib")

    assert "util.py" in output
    assert "app.py" not in output


def test_skips_binary_files_instead_of_failing(sources: Sandbox) -> None:
    # A single PNG in the tree must not make the whole search useless.
    (sources.root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfemain")

    output = search_text(sources, "main")

    assert "logo.png" not in output
    assert "app.py:3: def main():" in output


def test_skips_oversized_files_instead_of_failing(sources: Sandbox) -> None:
    (sources.root / "huge.py").write_bytes(b"x" * (MAX_FILE_BYTES + 1))

    assert "huge.py" not in search_text(sources, "main")


def test_reports_how_many_files_were_skipped(sources: Sandbox) -> None:
    (sources.root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

    assert "(1 unreadable file skipped)" in search_text(sources, "main")


def test_mentions_skipped_files_even_when_nothing_matched(sandbox: Sandbox) -> None:
    (sandbox.root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

    assert "1 unreadable file skipped" in search_text(sandbox, "anything")


def test_truncates_a_very_long_matched_line(sandbox: Sandbox) -> None:
    # One minified bundle should not consume the whole context window.
    (sandbox.root / "bundle.js").write_text("var x=" + "a" * 5000 + ";\n", encoding="utf-8")

    line = search_text(sandbox, "var x").splitlines()[1]

    assert line.endswith("...")
    assert len(line) < MAX_LINE_CHARS + 60  # 'path:line: ' prefix plus the ellipsis


def test_caps_the_number_of_matches_and_says_so(sandbox: Sandbox) -> None:
    body = "\n".join(f"line {index} needle" for index in range(MAX_TEXT_MATCHES + 7))
    (sandbox.root / "many.txt").write_text(body, encoding="utf-8")

    lines = search_text(sandbox, "needle").splitlines()

    assert lines[0] == f"{MAX_TEXT_MATCHES + 7} matches for 'needle':"
    assert len(lines) == MAX_TEXT_MATCHES + 2  # header + capped matches + truncation note
    assert "7 more matches not shown" in lines[-1]


def test_search_text_rejects_a_missing_search_root(sandbox: Sandbox) -> None:
    with pytest.raises(ToolError, match="No such directory"):
        search_text(sandbox, "anything", path="nope")


def test_search_text_refuses_to_escape(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        search_text(sandbox, "secret", path="..")


def test_search_text_does_not_descend_into_a_symlinked_directory(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "secret.txt").write_text("hunter2 is the password\n", encoding="utf-8")
    (sandbox.root / "link").symlink_to(tmp_path / "elsewhere")

    output = search_text(sandbox, "password")

    # Neither the outside file's name nor a line of its contents may reach the model.
    assert "secret.txt" not in output
    assert "hunter2" not in output


def test_search_text_will_not_read_a_symlink_to_a_file_outside_the_root(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    """The counterpart to the find_files case, and the one that reads real content.

    A symlinked directory is skipped by `is_file()` before the containment check is even
    reached, so that test proves nothing about the check. A symlink to a file *is* a file,
    so `is_relative_to(sandbox.root)` is the only thing standing between the model and the
    contents of an arbitrary file on the host.
    """
    (tmp_path / "outside_secrets.txt").write_text("swordfish is the password\n", encoding="utf-8")
    (sandbox.root / "shortcut.txt").symlink_to(tmp_path / "outside_secrets.txt")

    output = search_text(sandbox, "password")

    assert "swordfish" not in output
    assert "shortcut.txt" not in output
