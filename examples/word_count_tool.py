"""A worked example: the eleventh tool, built step by step in EXTENDING.md.

This file is the *finished* code from that walkthrough. It is deliberately **not** wired
into `mini_agent.tools.build_registry` -- wiring it in is the last step the reader does
themselves, and leaving it out keeps the agent's real capability list at the ten tools the
README documents.

It is still real, imported, tested code (see `tests/test_word_count_example.py`), which is
the point: a fenced code block in a markdown file drifts out of step with the harness it
describes, and nothing notices. This cannot.
"""

from __future__ import annotations

from functools import partial

from mini_agent.errors import ToolError
from mini_agent.registry import Tool
from mini_agent.sandbox import Sandbox
from mini_agent.tools.common import load_text, plural


def word_count(sandbox: Sandbox, path: str) -> str:
    """Count the words, lines, and characters in one UTF-8 text file.

    The shape every handler shares: take a `Sandbox` first, take the model's arguments as
    keywords after it, and return the string the model will read as its observation.
    """
    # Resolve before touching anything. `path` came from the model and means nothing until
    # the sandbox has proved it lands inside the root. No handler may skip this line.
    target = sandbox.resolve(path)

    # Say what is wrong *and* what to do instead: the model reads this and gets one cheap
    # chance to correct itself before spending another step.
    if not target.exists():
        raise ToolError(f"No such file: {path!r}. Use list_directory to see what is there.")
    if target.is_dir():
        raise ToolError(f"{path!r} is a directory, not a file. Use list_directory instead.")

    # `load_text` already enforces the 1 MiB cap and the UTF-8 check, and raises ToolError
    # with a message written for the model. Reuse it rather than calling read_text here --
    # every tool that reads a file should fail the same way for the same reasons.
    text = load_text(target, path)

    return (
        f"{sandbox.relative(target)}: {plural(len(text.split()), 'word')}, "
        f"{plural(len(text.splitlines()), 'line')}, {plural(len(text), 'character')}"
    )


def make_tools(sandbox: Sandbox) -> list[Tool]:
    """Describe the handler to the model. Same shape as every `make_tools` in `tools/`."""
    return [
        Tool(
            name="word_count",
            description=(
                "Count the words, lines, and characters in a UTF-8 text file. Use this "
                "when you need the size of a file's contents but not the contents "
                "themselves; it is much cheaper than reading the whole file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to measure, relative to the workspace root.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            # `partial` binds the sandbox, leaving a callable whose remaining signature is
            # exactly the JSON schema above. That correspondence is what lets the registry
            # bind the model's arguments and reject bad ones before the handler runs.
            handler=partial(word_count, sandbox),
        )
    ]
