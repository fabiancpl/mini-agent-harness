"""Scaffolding checks: the package imports, and the error hierarchy is one tree.

The CLI catches `HarnessError` to turn every expected failure into one clear line, so a new
exception that forgets to inherit from it would escape as a traceback.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import mini_agent
from mini_agent import errors

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_package_exposes_a_version() -> None:
    assert mini_agent.__version__


def test_the_reported_version_matches_pyproject() -> None:
    """One source of truth, checked.

    `__version__` used to be a second hand-written copy of the number, and it sat a whole
    release behind: pyproject said 0.2.0 while the package reported 0.1.0. The old test only
    asserted the string was truthy, so it passed throughout. Compare them instead.
    """
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]

    assert mini_agent.__version__ == declared


def test_every_error_derives_from_harness_error() -> None:
    for name in ("ConfigError", "PathOutsideRootError", "RegistryError", "ToolError", "LLMError"):
        assert issubclass(getattr(errors, name), errors.HarnessError)
