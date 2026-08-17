"""Tests for the security boundary.

If any test in this file starts failing, assume the sandbox is broken and not that the test
is wrong. These cases are the reason the project exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.errors import PathOutsideRootError
from mini_agent.sandbox import Sandbox

# --- paths that should be allowed ---------------------------------------------------------


def test_resolves_a_simple_relative_path(sandbox: Sandbox) -> None:
    assert sandbox.resolve("notes.txt") == sandbox.root / "notes.txt"


def test_resolves_a_nested_relative_path(sandbox: Sandbox) -> None:
    assert sandbox.resolve("a/b/c.txt") == sandbox.root / "a" / "b" / "c.txt"


def test_resolves_dot_to_the_root(sandbox: Sandbox) -> None:
    assert sandbox.resolve(".") == sandbox.root


def test_resolves_empty_string_to_the_root(sandbox: Sandbox) -> None:
    assert sandbox.resolve("") == sandbox.root


def test_resolves_explicit_current_directory_prefix(sandbox: Sandbox) -> None:
    assert sandbox.resolve("./x/../y.txt") == sandbox.root / "y.txt"


def test_allows_an_absolute_path_that_is_inside_the_root(sandbox: Sandbox) -> None:
    inside = str(sandbox.root / "inside.txt")
    assert sandbox.resolve(inside) == sandbox.root / "inside.txt"


def test_allows_a_path_that_does_not_exist_yet(sandbox: Sandbox) -> None:
    # write_file and create_directory resolve paths before creating them.
    assert sandbox.resolve("brand/new/file.txt").name == "file.txt"


def test_allows_traversal_that_stays_inside_the_root(sandbox: Sandbox) -> None:
    assert sandbox.resolve("a/b/../c.txt") == sandbox.root / "a" / "c.txt"


# --- paths that must be refused -----------------------------------------------------------


def test_rejects_parent_traversal(sandbox: Sandbox) -> None:
    with pytest.raises(PathOutsideRootError):
        sandbox.resolve("../../etc/passwd")


def test_rejects_a_single_step_out(sandbox: Sandbox, outside_file: Path) -> None:
    with pytest.raises(PathOutsideRootError):
        sandbox.resolve("../outside.txt")


def test_rejects_an_absolute_path_outside_the_root(sandbox: Sandbox) -> None:
    with pytest.raises(PathOutsideRootError):
        sandbox.resolve("/etc/passwd")


def test_allows_traversal_that_leaves_and_comes_back(sandbox: Sandbox) -> None:
    # '../root/ok.txt' normalises back inside, so it is allowed. The rule is about the
    # destination, not about whether '..' appears in the string -- which is the whole
    # reason we resolve first and judge second.
    assert sandbox.resolve("../root/ok.txt") == sandbox.root / "ok.txt"


def test_rejects_a_sibling_directory_with_the_root_as_a_name_prefix(tmp_path: Path) -> None:
    # 'root-evil' starts with the string 'root'. A prefix check on strings would wrongly
    # allow it; is_relative_to compares path components, so it does not.
    (tmp_path / "root").mkdir()
    (tmp_path / "root-evil").mkdir()
    sandbox = Sandbox(tmp_path / "root")
    with pytest.raises(PathOutsideRootError):
        sandbox.resolve("../root-evil/file.txt")


# --- symlinks: the case string-inspection cannot catch ------------------------------------


def test_rejects_a_symlink_pointing_outside_the_root(sandbox: Sandbox, outside_file: Path) -> None:
    (sandbox.root / "escape").symlink_to(outside_file)
    with pytest.raises(PathOutsideRootError):
        sandbox.resolve("escape")


def test_rejects_a_path_traversing_a_symlinked_directory(sandbox: Sandbox, tmp_path: Path) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "secret.txt").write_text("nope\n", encoding="utf-8")
    (sandbox.root / "link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(PathOutsideRootError):
        sandbox.resolve("link/secret.txt")


def test_allows_a_symlink_pointing_inside_the_root(sandbox: Sandbox) -> None:
    target = sandbox.root / "real.txt"
    target.write_text("fine\n", encoding="utf-8")
    (sandbox.root / "alias").symlink_to(target)
    assert sandbox.resolve("alias") == target


# --- construction and display -------------------------------------------------------------


def test_root_is_stored_resolved(tmp_path: Path) -> None:
    # A root reached through a symlink still compares correctly against resolved candidates.
    real = tmp_path / "real_root"
    real.mkdir()
    (tmp_path / "linked_root").symlink_to(real)
    sandbox = Sandbox(tmp_path / "linked_root")
    assert sandbox.root == real.resolve()
    assert sandbox.resolve("f.txt") == real.resolve() / "f.txt"


def test_relative_renders_paths_without_the_host_prefix(sandbox: Sandbox) -> None:
    assert sandbox.relative(sandbox.root / "a" / "b.txt") == str(Path("a") / "b.txt")


def test_relative_renders_the_root_itself_as_dot(sandbox: Sandbox) -> None:
    assert sandbox.relative(sandbox.root) == "."


def test_error_message_does_not_leak_the_host_path(sandbox: Sandbox) -> None:
    with pytest.raises(PathOutsideRootError) as excinfo:
        sandbox.resolve("/etc/passwd")
    # The model sees this text; it should explain the rule, not the machine's layout.
    assert str(sandbox.root) not in str(excinfo.value)
