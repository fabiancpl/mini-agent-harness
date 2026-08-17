"""Tests for `list_directory` and `find_files`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.errors import PathOutsideRootError, ToolError
from mini_agent.sandbox import Sandbox
from mini_agent.tools.navigate import MAX_MATCHES, find_files, list_directory


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


def test_find_files_does_not_report_matches_reached_through_a_symlink(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    # A glob can walk into a symlinked directory; the results are filtered back through the
    # sandbox so no path outside the root is ever named to the model.
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "secret.txt").write_text("nope\n", encoding="utf-8")
    (sandbox.root / "link").symlink_to(tmp_path / "elsewhere")

    assert "secret.txt" not in find_files(sandbox, "*.txt")
