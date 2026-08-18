"""Everything that changes the workspace.

The rule these tools are built around:

    No operation may make existing content unreachable.

That is a claim about *effects*, not about names, and it is the honest version. An earlier
draft of this file said "the agent cannot delete anything", but `write_file` overwrites --
content loss was already possible, and the slogan was promising more than the code did.

Read the tools as a ladder of how much they can destroy:

    append_to_file   grows a file                  nothing, ever -- provably additive
    edit_file        replaces one exact span       that span
    write_file       replaces the whole file       the whole previous content
    move / copy      relocates or duplicates       nothing: refused if the destination exists

`move` and `copy` deserve the explanation, because at first glance they look destructive.
They are not, and the difference from deletion is worth being precise about:

    move(a, b)  with b free      -> content intact at b, undone by move(b, a)
    move(a, b)  with b existing  -> b's old content gone, no inverse
    delete(a)                    -> a's content gone, no inverse

So relocation is destructive in exactly one case, and that case is a precondition you can
check *before* acting -- which is what `_resolve_relocation` does. Deletion has no such
case; it is destructive by definition. That is why there is no `delete_file`, no
`remove_directory`, no `unlink`, and no shell tool anywhere in this package. The capability
was never written, and a function that does not exist cannot be argued into existence by a
cleverly worded prompt.

One documented limitation: `Sandbox.resolve` follows symlinks, so moving or copying a
symlink acts on the file it points at, not on the link. `edit_file` has always behaved this
way. It is safe -- a link pointing outside the root is refused by the sandbox either way --
but it is a surprise worth knowing about.

Running the workspace under version control remains the recommended safety net for the
content loss that `write_file` and `edit_file` can legitimately cause.
"""

from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

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


def append_to_file(sandbox: Sandbox, path: str, content: str) -> str:
    """Add text to the end of a file, creating it if it does not exist yet.

    The only mutation in this package that cannot destroy anything: it never looks at what
    is already in the file, so there is no case where existing content is replaced.
    """
    target = sandbox.resolve(path)
    if target.is_dir():
        raise ToolError(f"{path!r} is a directory, so text cannot be appended to it.")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # "a" opens for appending and creates the file if missing, so there is no separate
        # "does it exist yet" branch to get wrong.
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        total_lines = len(target.read_text(encoding="utf-8").splitlines())
    except OSError as exc:
        raise ToolError(f"Could not append to {path!r}: {exc.strerror}") from exc

    appended = plural(len(content), "character")
    return f"Appended {appended} to {sandbox.relative(target)} (now {plural(total_lines, 'line')})"


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


def _resolve_relocation(
    sandbox: Sandbox, source: str, destination: str, verb: str
) -> tuple[Path, Path]:
    """Validate a source/destination pair for `move` or `copy`.

    Shared because the two tools need exactly the same guarantees. `verb` only shapes the
    wording of the errors, which the model reads.
    """
    # Both paths, not just the source. A relocation has two ends, and leaving the
    # destination unchecked would be a hole straight out of the sandbox.
    source_path = sandbox.resolve(source)
    destination_path = sandbox.resolve(destination)

    if not source_path.exists():
        raise ToolError(f"No such file or directory to {verb}: {source!r}")

    # The no-clobber rule. This single check is what keeps every relocation reversible, and
    # so what keeps "nothing becomes unreachable" true rather than aspirational.
    if destination_path.exists():
        raise ToolError(
            f"{destination!r} already exists. Moving onto it would destroy its contents. "
            f"Choose a path that is free, or use edit_file or write_file to change it "
            f"in place."
        )

    # Putting a directory inside itself. The three things this prevents are all different:
    #
    #   move("src", "src/nested")        rename() fails with a bare EINVAL, "Invalid
    #                                    argument", which tells the model nothing.
    #   copy("src", "src/nested")        copytree *succeeds*, silently leaving a copy of
    #                                    src inside src -- almost never what was meant.
    #   copy("src", "src/deep/here")     copytree walks into what it is writing and dies
    #                                    with RecursionError.
    #
    # That last one is why this check is load-bearing rather than cosmetic: RecursionError
    # is not an OSError, so the handler below would not catch it, and it is not a
    # HarnessError either, so the agent loop would not turn it into an observation. It
    # would take down the whole run. Refusing up front keeps the failure recoverable.
    if source_path.is_dir() and destination_path.is_relative_to(source_path):
        raise ToolError(f"Cannot {verb} {source!r} into itself ({destination!r} is inside it).")

    return source_path, destination_path


def move(sandbox: Sandbox, source: str, destination: str) -> str:
    """Move or rename a file or directory. Refuses to overwrite anything."""
    source_path, destination_path = _resolve_relocation(sandbox, source, destination, "move")

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(destination_path)
    except OSError as exc:
        raise ToolError(f"Could not move {source!r} to {destination!r}: {exc.strerror}") from exc

    return f"Moved {sandbox.relative(source_path)} to {sandbox.relative(destination_path)}"


def copy(sandbox: Sandbox, source: str, destination: str) -> str:
    """Duplicate a file or directory. Refuses to overwrite anything."""
    source_path, destination_path = _resolve_relocation(sandbox, source, destination, "copy")

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            # copytree refuses an existing destination by itself, which happens to be the
            # same rule _resolve_relocation just enforced.
            shutil.copytree(source_path, destination_path)
            copied = plural(sum(1 for path in destination_path.rglob("*") if path.is_file()), "file")
            detail = f" ({copied})"
        else:
            shutil.copy2(source_path, destination_path)  # copy2 keeps the modification time
            detail = ""
    except OSError as exc:
        raise ToolError(f"Could not copy {source!r} to {destination!r}: {exc.strerror}") from exc

    return f"Copied {sandbox.relative(source_path)} to {sandbox.relative(destination_path)}{detail}"


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
            name="append_to_file",
            description=(
                "Add text to the end of a file, creating it if it does not exist. Never "
                "changes what is already in the file, so prefer this over write_file when "
                "you are adding to something rather than replacing it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to append to, relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to add at the end. Include a trailing newline.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=partial(append_to_file, sandbox),
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
        Tool(
            name="move",
            description=(
                "Move or rename a file or directory. The destination is the complete new "
                "path, not a folder to put the source inside: to move 'a.txt' into 'docs', "
                "pass 'docs/a.txt'. Refuses if the destination already exists, so it can "
                "never overwrite anything -- pick a free path instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Existing file or directory, relative to the root.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "The complete new path. Must not already exist.",
                    },
                },
                "required": ["source", "destination"],
                "additionalProperties": False,
            },
            handler=partial(move, sandbox),
        ),
        Tool(
            name="copy",
            description=(
                "Copy a file, or a directory and everything in it. The destination is the "
                "complete new path, not a folder to put the copy inside. Refuses if the "
                "destination already exists. Useful for keeping a backup before making a "
                "risky change to a file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Existing file or directory, relative to the root.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "The complete new path. Must not already exist.",
                    },
                },
                "required": ["source", "destination"],
                "additionalProperties": False,
            },
            handler=partial(copy, sandbox),
        ),
    ]
