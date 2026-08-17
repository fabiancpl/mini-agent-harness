"""What a tool *is*, and the collection the agent is allowed to call.

The registry is the agent's complete list of capabilities. If something is not registered
here, the agent cannot do it -- there is no fallback path, no shell, no `eval`. That is why
the absence of a delete tool is a real guarantee and not a promise.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import RegistryError, ToolError


@dataclass(frozen=True)
class Tool:
    """One capability, plus everything the model needs to know to call it."""

    #: Function name the model will emit. Snake case, verb first.
    name: str
    #: Written *for the model*: what it does, when to use it, and its limits.
    description: str
    #: JSON Schema for the arguments, passed to the API verbatim.
    parameters: dict[str, Any]
    #: Takes the validated arguments, returns the observation text the model will read.
    handler: Callable[..., str]

    def to_schema(self) -> dict[str, Any]:
        """Render as one entry of the OpenAI `tools` array."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """An ordered, name-addressed collection of tools."""

    def __init__(self) -> None:
        # A plain dict: insertion-ordered, which keeps the order tools are advertised in
        # stable and therefore keeps model behaviour reproducible.
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise RegistryError(f"A tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            # This message is shown to the model when it hallucinates a tool, so it lists
            # the real options: the model can usually correct itself on the next step.
            raise RegistryError(
                f"Unknown tool {name!r}. Available tools: {', '.join(self.names()) or '(none)'}"
            )
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def to_schemas(self) -> list[dict[str, Any]]:
        """The `tools` payload sent with every request."""
        return [tool.to_schema() for tool in self._tools.values()]

    def subset(self, names: Sequence[str]) -> ToolRegistry:
        """A new registry with only `names`, in the order given."""
        selected = ToolRegistry()
        for name in names:
            selected.register(self.get(name))
        return selected

    def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool with model-supplied arguments and return its observation.

        Arguments arrive as whatever JSON the model produced, so they may be missing,
        extra, or misspelled. Binding them to the handler's signature *before* calling
        turns that into a `ToolError` the model can read and retry -- if we simply
        splatted them in, a bad argument would surface as a `TypeError` and take down the
        whole run over a fixable mistake.
        """
        tool = self.get(name)
        try:
            bound = inspect.signature(tool.handler).bind(**arguments)
        except TypeError as exc:
            expected = sorted(tool.parameters.get("properties", {}))
            raise ToolError(f"Bad arguments for {name}: {exc}. Expected: {expected}") from exc
        return tool.handler(*bound.args, **bound.kwargs)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
