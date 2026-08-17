"""Scaffolding checks: the package imports, and the error hierarchy is one tree.

The CLI catches `HarnessError` to turn every expected failure into one clear line, so a new
exception that forgets to inherit from it would escape as a traceback.
"""

from __future__ import annotations

import mini_agent
from mini_agent import errors


def test_package_exposes_a_version() -> None:
    assert mini_agent.__version__


def test_every_error_derives_from_harness_error() -> None:
    for name in ("ConfigError", "PathOutsideRootError", "RegistryError", "ToolError", "LLMError"):
        assert issubclass(getattr(errors, name), errors.HarnessError)
