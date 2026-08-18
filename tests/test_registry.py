"""Tests for `Tool`, `ToolRegistry`, and `build_registry`."""

from __future__ import annotations

import pytest

from mini_agent.errors import RegistryError, ToolError
from mini_agent.registry import Tool, ToolRegistry
from mini_agent.sandbox import Sandbox
from mini_agent.tools import build_registry

ALL_TOOL_NAMES = [
    "list_directory",
    "find_files",
    "search_text",
    "read_file",
    "create_directory",
    "write_file",
    "append_to_file",
    "edit_file",
    "move",
    "copy",
]


def make_tool(name: str = "echo", **overrides) -> Tool:
    defaults = dict(
        name=name,
        description="Echo a message back.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        handler=lambda message: f"echo: {message}",
    )
    return Tool(**{**defaults, **overrides})


# --- registration -------------------------------------------------------------------------


def test_registers_and_retrieves_a_tool() -> None:
    registry = ToolRegistry()
    tool = make_tool()
    registry.register(tool)

    assert registry.get("echo") is tool
    assert "echo" in registry
    assert len(registry) == 1


def test_rejects_a_duplicate_name() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())

    with pytest.raises(RegistryError, match="already registered"):
        registry.register(make_tool())


def test_unknown_tool_error_lists_the_real_options() -> None:
    # The model reads this message, so it must be enough to self-correct from.
    registry = ToolRegistry()
    registry.register(make_tool("read_file"))

    with pytest.raises(RegistryError, match="read_file"):
        registry.get("raed_file")


def test_names_preserve_registration_order() -> None:
    registry = ToolRegistry()
    for name in ("c", "a", "b"):
        registry.register(make_tool(name))

    assert registry.names() == ["c", "a", "b"]


# --- the OpenAI schema contract -----------------------------------------------------------


def test_to_schema_matches_the_openai_function_shape() -> None:
    tool = make_tool()

    assert tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo a message back.",
            "parameters": tool.parameters,
        },
    }


def test_to_schemas_returns_one_entry_per_tool_in_order() -> None:
    registry = ToolRegistry()
    registry.register(make_tool("first"))
    registry.register(make_tool("second"))

    assert [entry["function"]["name"] for entry in registry.to_schemas()] == ["first", "second"]


# --- invoking ------------------------------------------------------------------------------


def test_invoke_calls_the_handler_with_the_models_arguments() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())

    assert registry.invoke("echo", {"message": "hi"}) == "echo: hi"


def test_invoke_turns_a_missing_argument_into_a_tool_error() -> None:
    # A ToolError becomes an observation and the model gets another turn; a raw TypeError
    # would end the whole run over a mistake the model could have fixed.
    registry = ToolRegistry()
    registry.register(make_tool())

    with pytest.raises(ToolError, match="Bad arguments"):
        registry.invoke("echo", {})


def test_invoke_turns_an_unexpected_argument_into_a_tool_error() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())

    with pytest.raises(ToolError) as excinfo:
        registry.invoke("echo", {"message": "hi", "colour": "blue"})

    assert "message" in str(excinfo.value)  # the error names the arguments that are expected


def test_invoke_propagates_an_unknown_tool_name() -> None:
    with pytest.raises(RegistryError):
        ToolRegistry().invoke("nope", {})


# --- subsetting ----------------------------------------------------------------------------


def test_subset_keeps_only_the_named_tools_in_the_given_order() -> None:
    registry = ToolRegistry()
    for name in ("a", "b", "c"):
        registry.register(make_tool(name))

    assert registry.subset(["c", "a"]).names() == ["c", "a"]


def test_subset_does_not_modify_the_original() -> None:
    registry = ToolRegistry()
    registry.register(make_tool("a"))
    registry.register(make_tool("b"))

    registry.subset(["a"])

    assert registry.names() == ["a", "b"]


# --- build_registry -------------------------------------------------------------------------


def test_build_registry_registers_every_tool_by_default(sandbox: Sandbox) -> None:
    assert build_registry(sandbox).names() == ALL_TOOL_NAMES


def test_build_registry_honours_the_allow_list_and_its_order(sandbox: Sandbox) -> None:
    registry = build_registry(sandbox, ["read_file", "list_directory"])

    assert registry.names() == ["read_file", "list_directory"]
    assert "write_file" not in registry


def test_build_registry_rejects_an_unknown_tool_name(sandbox: Sandbox) -> None:
    with pytest.raises(RegistryError, match="delete_file"):
        build_registry(sandbox, ["read_file", "delete_file"])


def test_every_built_tool_has_a_description_and_a_schema(sandbox: Sandbox) -> None:
    for tool in build_registry(sandbox):
        assert tool.description.strip(), f"{tool.name} has no description for the model"
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters


def test_built_tools_are_bound_to_the_sandbox(sandbox: Sandbox) -> None:
    # The handler takes only the model-facing arguments; the sandbox is already bound in.
    registry = build_registry(sandbox)

    assert registry.invoke("list_directory", {}) == f"{sandbox.relative(sandbox.root)} is empty"


# --- the capability that must not exist -----------------------------------------------------

# Note what is *not* here: 'rename' and 'move'. Relocating something preserves its content
# and has an exact inverse, so it does not violate the invariant -- provided it refuses to
# overwrite, which `test_relocation_tools_refuse_to_overwrite` below is what actually checks.
FORBIDDEN_NAME_PARTS = ["delete", "remove", "rm", "unlink", "rmdir", "trash", "exec", "shell", "eval"]


@pytest.mark.parametrize("forbidden", FORBIDDEN_NAME_PARTS)
def test_no_obviously_destructive_tool_is_registered(sandbox: Sandbox, forbidden: str) -> None:
    """A lint against accidental additions -- not the guarantee itself.

    Be clear about what this can and cannot do. It matches on *names*, so a tool called
    `tidy_workspace` that unlinked files would sail straight past it. The real guarantee is
    structural: `build_registry` is the single place tools are wired, so reviewing one
    function tells you everything the agent can do.

    What this test buys is a tripwire. Adding a capability that changes the safety story of
    the project should be a deliberate, documented decision, and failing here forces the
    person doing it to notice.
    """
    for name in build_registry(sandbox).names():
        assert forbidden not in name.split("_"), f"{name!r} looks destructive"


@pytest.mark.parametrize("tool_name", ["move", "copy"])
def test_relocation_tools_refuse_to_overwrite(sandbox: Sandbox, tool_name: str) -> None:
    """The actual invariant, checked through the wired registry: nothing becomes unreachable.

    `move` and `copy` are the two tools that could destroy content without being named
    anything alarming -- both would silently clobber the destination if they used the raw
    stdlib calls. This asserts the guard is in place on the tools the agent really gets.
    """
    (sandbox.root / "source.txt").write_text("new\n", encoding="utf-8")
    (sandbox.root / "occupied.txt").write_text("must survive\n", encoding="utf-8")
    registry = build_registry(sandbox)

    with pytest.raises(ToolError, match="already exists"):
        registry.invoke(tool_name, {"source": "source.txt", "destination": "occupied.txt"})

    assert (sandbox.root / "occupied.txt").read_text(encoding="utf-8") == "must survive\n"
