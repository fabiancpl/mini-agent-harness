"""Every error this harness raises on purpose.

One shallow hierarchy so the CLI can catch `HarnessError` and print a single clear line,
while the ReAct loop catches the narrower `ToolError` and hands it back to the model as an
observation instead of crashing the run.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all expected failures. Anything else is a genuine bug."""


class ConfigError(HarnessError):
    """The configuration file is missing, malformed, or inconsistent."""


class PathOutsideRootError(HarnessError):
    """A tool was given a path that resolves outside the sandbox root."""


class RegistryError(HarnessError):
    """A tool was registered twice, or one was requested that does not exist."""


class ToolError(HarnessError):
    """A tool ran but could not do what was asked (missing file, bad argument, …).

    The loop turns this into an observation, so the message is read by the model: say what
    went wrong and, where useful, what to do instead.
    """


class LLMError(HarnessError):
    """The model endpoint returned an error, or a response we could not parse."""
