"""Tests for the EXTENDING.md walkthrough tool.

`CLAUDE.md` asks three things of every new tool: the happy path, an input error, and an
escape attempt. This module is what that looks like in practice -- it is as much part of the
walkthrough as the tool itself, since "and now write these three tests" is easier to follow
when the tests exist to be copied.

`word_count_tool` is importable because `pyproject.toml` puts `examples/` on the pytest path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.errors import PathOutsideRootError, ToolError
from mini_agent.registry import ToolRegistry
from mini_agent.sandbox import Sandbox

from word_count_tool import make_tools, word_count

# --- the happy path -------------------------------------------------------------------------


def test_counts_words_lines_and_characters(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello world\nsecond line\n", encoding="utf-8")

    observation = word_count(Sandbox(tmp_path), "note.txt")

    assert observation == "note.txt: 4 words, 2 lines, 24 characters"


def test_uses_singular_wording_for_a_count_of_one(tmp_path: Path) -> None:
    # The observation is prose the model reads, so "1 words" would be a (small) bug.
    (tmp_path / "tiny.txt").write_text("hello", encoding="utf-8")

    assert word_count(Sandbox(tmp_path), "tiny.txt") == "tiny.txt: 1 word, 1 line, 5 characters"


# --- input errors become ToolErrors the model can act on ------------------------------------


def test_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="No such file"):
        word_count(Sandbox(tmp_path), "absent.txt")


def test_rejects_a_directory(tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()

    with pytest.raises(ToolError, match="is a directory"):
        word_count(Sandbox(tmp_path), "folder")


def test_rejects_a_binary_file_via_the_shared_loader(tmp_path: Path) -> None:
    # Delegating to load_text is what buys this behaviour; assert it actually arrives.
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")

    with pytest.raises(ToolError, match="not a UTF-8 text file"):
        word_count(Sandbox(tmp_path), "blob.bin")


# --- the escape attempt ---------------------------------------------------------------------


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")

    with pytest.raises(PathOutsideRootError):
        word_count(Sandbox(root), "../secret.txt")


def test_rejects_a_symlink_pointing_outside_the_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("outside", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)

    with pytest.raises(PathOutsideRootError):
        word_count(Sandbox(root), "link.txt")


# --- and it survives the trip through the registry ------------------------------------------


def test_the_registry_accepts_the_tool_and_invokes_it(tmp_path: Path) -> None:
    """The schema and the handler signature have to agree, or `invoke` cannot bind."""
    (tmp_path / "note.txt").write_text("one two three\n", encoding="utf-8")
    registry = ToolRegistry()
    for tool in make_tools(Sandbox(tmp_path)):
        registry.register(tool)

    assert registry.invoke("word_count", {"path": "note.txt"}).startswith("note.txt: 3 words")


def test_a_bad_argument_name_is_a_recoverable_tool_error(tmp_path: Path) -> None:
    registry = ToolRegistry()
    for tool in make_tools(Sandbox(tmp_path)):
        registry.register(tool)

    with pytest.raises(ToolError, match="Bad arguments for word_count"):
        registry.invoke("word_count", {"file": "note.txt"})
