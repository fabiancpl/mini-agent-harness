"""The tool implementations, and the single place where they are wired into a registry.

`build_registry` is one explicit function with no auto-discovery and no plugin scanning: to
know exactly what an agent can do, you read this file and nothing else. Adding a capability
must be a visible edit here, which is what makes the safety properties reviewable.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import RegistryError
from ..registry import ToolRegistry
from ..sandbox import Sandbox
from . import navigate, read, write


def build_registry(sandbox: Sandbox, enabled: Sequence[str] | None = None) -> ToolRegistry:
    """Build the registry for one sandbox.

    `enabled` is the allow-list from the config file; `None` means every tool. Unknown names
    are an error rather than a silent skip -- a typo in the config should not quietly leave
    the agent without a capability you believe you granted it.
    """
    registry = ToolRegistry()
    all_tools = [
        *navigate.make_tools(sandbox),
        *read.make_tools(sandbox),
        *write.make_tools(sandbox),
    ]
    for tool in all_tools:
        registry.register(tool)

    if enabled is None:
        return registry

    unknown = [name for name in enabled if name not in registry]
    if unknown:
        raise RegistryError(
            f"tools.enabled names tools that do not exist: {unknown}. "
            f"Available tools: {', '.join(registry.names())}"
        )
    return registry.subset(enabled)
