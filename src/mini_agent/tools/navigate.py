"""Tools for finding things: `list_directory`, `find_files`, and `search_text`.

The last two are the pair worth understanding together. `find_files` searches file *names*
("where is the config?"); `search_text` searches file *contents* ("where is this function
called?"). An agent without the second one has to read files one at a time to find anything,
which spends its whole step budget on looking rather than doing.
"""

from __future__ import annotations

import re
from functools import partial

from ..errors import ToolError
from ..registry import Tool
from ..sandbox import Sandbox
from .common import load_text, plural

#: Cap on `find_files` results. A glob like '*' in a node_modules-sized tree would otherwise
#: return tens of thousands of lines and blow out the context window.
MAX_MATCHES = 200

#: Cap on `search_text` results -- lower than MAX_MATCHES because each hit is a whole line of
#: source, not a short path.
MAX_TEXT_MATCHES = 100

#: Individual matched lines are trimmed to this. A minified bundle can be one 400 KB line.
MAX_LINE_CHARS = 200


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


def search_text(
    sandbox: Sandbox, pattern: str, path: str = ".", file_pattern: str = "*"
) -> str:
    """Search file contents for a regular expression, grep-style."""
    try:
        expression = re.compile(pattern)
    except re.error as exc:
        # Recoverable: the model reads this and writes a valid pattern on the next step.
        raise ToolError(f"{pattern!r} is not a valid regular expression: {exc}") from exc

    target = sandbox.resolve(path)
    if not target.is_dir():
        raise ToolError(f"No such directory to search: {path!r}")

    matches: list[str] = []
    skipped = 0
    for candidate in sorted(target.rglob(file_pattern)):
        if not candidate.is_file() or not candidate.resolve().is_relative_to(sandbox.root):
            continue
        try:
            text = load_text(candidate, sandbox.relative(candidate))
        except ToolError:
            # Binary or oversized. Unlike read_file, where the model named one specific
            # file, a tree search must step over what it cannot read: failing the whole
            # call because one PNG exists would make the tool useless in a real project.
            skipped += 1
            continue

        for number, line in enumerate(text.splitlines(), start=1):
            if expression.search(line):
                trimmed = line if len(line) <= MAX_LINE_CHARS else f"{line[:MAX_LINE_CHARS]}..."
                matches.append(f"{sandbox.relative(candidate)}:{number}: {trimmed}")

    footer = f" ({plural(skipped, 'unreadable file')} skipped)" if skipped else ""
    if not matches:
        return f"No matches for {pattern!r} under {sandbox.relative(target)}{footer}"

    lines = matches[:MAX_TEXT_MATCHES]
    if len(matches) > MAX_TEXT_MATCHES:
        hidden = plural(len(matches) - MAX_TEXT_MATCHES, "more match", "more matches")
        lines.append(f"... {hidden} not shown, narrow the pattern")
    header = f"{plural(len(matches), 'match', 'matches')} for {pattern!r}{footer}:"
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
        Tool(
            name="search_text",
            description=(
                "Search inside files for a regular expression and return matching lines as "
                "'path:line: text'. Use this to find where something is written or used; "
                "use find_files when you are looking for a file by its name instead. "
                "Binary and very large files are skipped automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for, e.g. 'def \\w+_file'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search under, relative to the root.",
                        "default": ".",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Only search files whose name matches this glob.",
                        "default": "*",
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=partial(search_text, sandbox),
        ),
    ]
