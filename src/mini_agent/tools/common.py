"""Helpers shared by more than one tool module."""

from __future__ import annotations

from pathlib import Path

from ..errors import ToolError

#: Refuse to load anything bigger than this. A 50 MB file would not fit in the context
#: window anyway, and reading it would just burn the run's step budget.
MAX_FILE_BYTES = 1024 * 1024  # 1 MiB


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """Render '1 entry' / '3 entries'. Observations are prose the model reads; keep it clean."""
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {word}"


def load_text(target: Path, user_path: str) -> str:
    """Read a file as UTF-8 text, or raise a `ToolError` the model can act on.

    `target` must already have come out of `Sandbox.resolve`. `user_path` is the string the
    model supplied, quoted back in errors so it recognises what it asked for.
    """
    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ToolError(
            f"{user_path!r} is {size} bytes, over the {MAX_FILE_BYTES} byte limit. "
            f"Read part of it with read_file's start_line and max_lines arguments."
        )
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(
            f"{user_path!r} is not a UTF-8 text file (it looks binary) and cannot be read."
        ) from None
