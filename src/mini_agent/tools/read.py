"""The reading tool.

Line numbers in the output are not decoration: they give the model a way to ask for the next
window of a long file, and a vocabulary for talking about where an edit should go.
"""

from __future__ import annotations

from functools import partial

from ..errors import ToolError
from ..registry import Tool
from ..sandbox import Sandbox
from .common import load_text, plural

#: How many lines one call returns unless the model asks for fewer.
DEFAULT_MAX_LINES = 400


def read_file(
    sandbox: Sandbox, path: str, start_line: int = 1, max_lines: int = DEFAULT_MAX_LINES
) -> str:
    """Return a numbered window of a text file."""
    if start_line < 1:
        raise ToolError(f"start_line must be 1 or greater, got {start_line}")
    if max_lines < 1:
        raise ToolError(f"max_lines must be 1 or greater, got {max_lines}")

    target = sandbox.resolve(path)
    if not target.exists():
        raise ToolError(f"No such file: {path!r}")
    if target.is_dir():
        raise ToolError(f"{path!r} is a directory. Use list_directory to see what is in it.")

    text = load_text(target, path)
    if not text:
        return f"{sandbox.relative(target)} is empty (0 bytes)"

    lines = text.splitlines()
    if start_line > len(lines):
        raise ToolError(
            f"{path!r} has only {len(lines)} lines, so start_line={start_line} is past the end."
        )

    first = start_line - 1
    window = lines[first : first + max_lines]
    # Right-align the numbers on the widest one so the code itself stays visually aligned.
    width = len(str(first + len(window)))
    body = "\n".join(f"{first + offset + 1:>{width}}  {line}" for offset, line in enumerate(window))

    last = first + len(window)
    header = f"{sandbox.relative(target)} (lines {start_line}-{last} of {len(lines)}):"
    if last < len(lines):
        remaining = plural(len(lines) - last, "more line")
        return "\n".join(
            [header, body, f"... {remaining}. Call again with start_line={last + 1}."]
        )
    return "\n".join([header, body])


def make_tools(sandbox: Sandbox) -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file and return its lines with line numbers. Long files "
                "come back in windows; use start_line to page through the rest. Always read "
                "a file before editing it, so your edit matches the real text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to read, relative to the workspace root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to return, counting from 1.",
                        "default": 1,
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "How many lines to return at most.",
                        "default": DEFAULT_MAX_LINES,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=partial(read_file, sandbox),
        )
    ]
