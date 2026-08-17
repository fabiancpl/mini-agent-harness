"""The security boundary.

Everything else in this project trusts one method, `Sandbox.resolve`. It takes a path the
model made up and returns a real path proven to live inside the root folder, or raises.

No tool may call `open()`, `mkdir()`, or `iterdir()` on a raw argument from the model. If it
did, the boundary would have a hole and the rest of the design would not matter.
"""

from __future__ import annotations

from pathlib import Path

from .errors import PathOutsideRootError


class Sandbox:
    """Confines every file operation to one directory tree."""

    def __init__(self, root: Path | str) -> None:
        # Resolve the root once, here, so every later comparison is real-path vs real-path.
        # This matters when the root itself is reached through a symlink (a common setup on
        # macOS, where /tmp is a link to /private/tmp): comparing a resolved candidate
        # against an unresolved root would reject perfectly legitimate paths.
        self.root = Path(root).resolve()

    def resolve(self, user_path: str) -> Path:
        """Turn a model-supplied path into an absolute path guaranteed inside the root.

        Three steps, and the order is the entire point:

        1. Join onto the root. Note that `Path("/root") / "/etc/passwd"` is `/etc/passwd` --
           an absolute argument silently replaces the root. That is not a bug we work around
           here; step 3 catches it, which is why the check must come last.
        2. `.resolve()` normalises `..` segments *and follows symlinks*, so a link pointing
           at /etc becomes /etc before we judge it. Resolution is what makes the check
           honest: inspecting the string for ".." is the classic broken version, because a
           symlink contains no ".." at all.
        3. Only now ask whether the real destination is inside the root.

        `strict=False` (the default) means the path need not exist yet -- necessary, since
        `write_file` and `create_directory` resolve paths they are about to create.
        """
        candidate = (self.root / user_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise PathOutsideRootError(
                f"Path {user_path!r} resolves outside the sandbox root and was refused. "
                f"Use paths relative to the root, such as 'notes.txt' or 'src/main.py'."
            )
        return candidate

    def relative(self, path: Path) -> str:
        """Render a resolved path for the model: sandbox-relative, root itself as '.'.

        Observations never contain host absolute paths, so the model cannot learn (or leak)
        anything about the machine's layout outside its root.
        """
        relative = path.relative_to(self.root)
        return str(relative) if relative.parts else "."

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Sandbox(root={self.root})"
