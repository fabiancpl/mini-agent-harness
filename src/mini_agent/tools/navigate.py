"""Tools for finding your way around: `list_directory` and `find_files`."""

from __future__ import annotations

from functools import partial

from ..errors import ToolError
from ..registry import Tool
from ..sandbox import Sandbox
from .common import plural

#: Cap on `find_files` results. A glob like '*' in a node_modules-sized tree would otherwise
#: return tens of thousands of lines and blow out the context window.
MAX_MATCHES = 200


def list_directory(sandbox: Sandbox, path: str = ".") -> str:
    """List one directory, directories first, then files, each group alphabetical."""
    target = sandbox.resolve(path)
    if not target.exists():
        raise ToolError(f"No such directory: {path!r}")
    if not target.is_dir():
        raise ToolError(f"{path!r} is a file, not a directory. Use read_file to read it.")

    # sorted() with a tuple key: False sorts before True, so directories come first.
    entries = sorted(target.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))
    if not entries:
        return f"{sandbox.relative(target)} is empty"

    # Symlinks are listed by name even when they point outside the root. Following one is
    # what gets refused, by Sandbox.resolve, at the moment a tool is asked to read it.
    lines = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries]
    header = f"{sandbox.relative(target)} ({plural(len(entries), 'entry', 'entries')}):"
    return "\n".join([header, *lines])


def find_files(sandbox: Sandbox, pattern: str, path: str = ".") -> str:
    """Search a directory tree for names matching a glob pattern."""
    target = sandbox.resolve(path)
    if not target.is_dir():
        raise ToolError(f"No such directory to search: {path!r}")

    matches = []
    for match in sorted(target.rglob(pattern)):
        # rglob does not follow symlinked directories, but a matched entry can still *be* a
        # symlink out of the tree. Re-checking through the sandbox keeps the one rule in one
        # place: no path leaves the root, no matter which code path produced it.
        if match.resolve().is_relative_to(sandbox.root):
            matches.append(match)

    if not matches:
        return f"No files matching {pattern!r} under {sandbox.relative(target)}"

    truncated = matches[:MAX_MATCHES]
    lines = [
        f"{sandbox.relative(match)}/" if match.is_dir() else sandbox.relative(match)
        for match in truncated
    ]
    if len(matches) > MAX_MATCHES:
        hidden = plural(len(matches) - MAX_MATCHES, "more match", "more matches")
        lines.append(f"... {hidden} not shown, narrow the pattern")
    header = f"{plural(len(matches), 'match', 'matches')} for {pattern!r}:"
    return "\n".join([header, *lines])


def make_tools(sandbox: Sandbox) -> list[Tool]:
    """Build the navigation tools, each bound to this sandbox.

    `partial` supplies the sandbox up front, so the handler the model calls takes only the
    arguments described in its schema.
    """
    return [
        Tool(
            name="list_directory",
            description=(
                "List the contents of a directory. Directories are shown with a trailing "
                "slash. Use '.' for the root of the workspace. Start here when you do not "
                "know what files exist."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list, relative to the workspace root.",
                        "default": ".",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=partial(list_directory, sandbox),
        ),
        Tool(
            name="find_files",
            description=(
                "Recursively find files and directories whose name matches a glob pattern, "
                "such as '*.py' or 'test_*'. Faster than listing directories one by one "
                "when you know roughly what you are looking for."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match against names, e.g. '*.md'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search under, relative to the root.",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=partial(find_files, sandbox),
        ),
    ]
