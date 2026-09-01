"""A small, readable ReAct agent harness with sandboxed file-system tools."""

from __future__ import annotations

# Keep this in step with `version` in pyproject.toml. It is a second copy of the number, which
# is a real cost, and 0.2.0 paid it: pyproject was bumped and this line was not, so the package
# reported 0.1.0 for a whole release. `importlib.metadata` would remove the copy but read the
# *installed* version, which lags the source in an editable checkout -- a bump would then fail
# the suite until someone re-ran `uv sync`, which is a worse trap than the one it fixes.
# So: keep the literal, and let test_the_reported_version_matches_pyproject enforce the pair.
__version__ = "0.3.0"
