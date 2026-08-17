"""The creating and modifying tools: `create_directory`, `write_file`, `edit_file`.

Note what is not in this file. There is no `delete_file`, no `remove_directory`, no `move`,
no `rename`. The agent cannot destroy your work because that ability was never written --
not because a rule in the prompt asks it politely to refrain. A prompt can be argued with;
a function that does not exist cannot be called.

`edit_file` can replace a file's contents with nothing, which is the closest this harness
gets to destruction. The file itself always survives, and so does its history if the
workspace is under version control -- which is the recommended way to run this.
"""

from __future__ import annotations

from functools import partial

from ..errors import ToolError
from ..registry import Tool
from ..sandbox import Sandbox
from .common import load_text, plural


def create_directory(sandbox: Sandbox, path: str) -> str:
    """Create a directory, including any missing parents."""
    target = sandbox.resolve(path)
    if target.is_dir():
        # Idempotent on purpose: a model that re-creates a directory has not made a mistake
        # worth spending a step correcting.
        return f"Directory already exists: {sandbox.relative(target)}"
    if target.exists():
        raise ToolError(f"Cannot create a directory at {path!r}: a file is already there.")

    try:
        target.mkdir(parents=True)
    except OSError as exc:
        raise ToolError(f"Could not create directory {path!r}: {exc.strerror}") from exc
    return f"Created directory: {sandbox.relative(target)}"


def write_file(sandbox: Sandbox, path: str, content: str) -> str:
    """Create a file, or replace an existing one completely."""
    target = sandbox.resolve(path)
    if target.is_dir():
        raise ToolError(f"{path!r} is a directory, so it cannot be written to as a file.")

    existed = target.exists()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"Could not write {path!r}: {exc.strerror}") from exc

    verb = "Overwrote" if existed else "Created"
    size = f"{plural(len(content), 'character')}, {plural(len(content.splitlines()), 'line')}"
    return f"{verb} {sandbox.relative(target)} ({size})"


def edit_file(sandbox: Sandbox, path: str, old_text: str, new_text: str) -> str:
    """Replace one exact, unique occurrence of `old_text` with `new_text`."""
    if not old_text:
        raise ToolError("old_text must not be empty. Use write_file to replace a whole file.")

    target = sandbox.resolve(path)
    if not target.exists():
        raise ToolError(f"No such file to edit: {path!r}. Use write_file to create it.")
    if target.is_dir():
        raise ToolError(f"{path!r} is a directory, not a file.")

    text = load_text(target, path)
    occurrences = text.count(old_text)
    if occurrences == 0:
        raise ToolError(
            f"old_text was not found in {path!r}. It must match the file exactly, including "
            f"whitespace and indentation. Read the file again and copy the text from it."
        )
    if occurrences > 1:
        # Refusing an ambiguous edit is the point: picking one silently would eventually
        # pick the wrong one, and the model would have no way to tell.
        raise ToolError(
            f"old_text appears {occurrences} times in {path!r}, so the edit is ambiguous. "
            f"Include the surrounding lines to identify exactly one location."
        )

    try:
        target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"Could not write {path!r}: {exc.strerror}") from exc
    return (
        f"Edited {sandbox.relative(target)}: replaced {len(old_text)} characters "
        f"with {len(new_text)}."
    )


def make_tools(sandbox: Sandbox) -> list[Tool]:
    return [
        Tool(
            name="create_directory",
            description=(
                "Create a directory, including any missing parent directories. Succeeds "
                "quietly if it already exists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to create, relative to the workspace root.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=partial(create_directory, sandbox),
        ),
        Tool(
            name="write_file",
            description=(
                "Write a file, creating it and any missing parent directories. If the file "
                "already exists its entire contents are replaced, so prefer edit_file for "
                "changing part of an existing file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to write, relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new contents of the file.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=partial(write_file, sandbox),
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace one exact occurrence of old_text with new_text in an existing "
                "file. old_text must match the file character for character and must "
                "appear exactly once, so include enough surrounding context to be unique. "
                "Read the file first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to edit, relative to the workspace root.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find. Must occur exactly once.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to put in its place. May be empty to remove it.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            handler=partial(edit_file, sandbox),
        ),
    ]
