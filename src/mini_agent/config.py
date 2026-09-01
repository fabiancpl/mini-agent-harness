"""Load `config.yaml` into validated, frozen dataclasses.

Everything is checked once, at startup, so a typo fails immediately with a sentence you can
act on -- rather than at step 7 of an agent run, as a `KeyError` from somewhere deep.

The validation here is deliberately explicit and repetitive instead of schema-driven: you
can read it top to bottom and know exactly what a valid config is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError


@dataclass(frozen=True)
class LLMConfig:
    """How to reach the model. Any OpenAI-compatible server works."""

    base_url: str
    model: str
    api_key: str  # the resolved secret, read from the environment -- never stored in YAML
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 60
    #: Total attempts for one request, not extra tries after the first: 1 disables retrying.
    max_attempts: int = 3


@dataclass(frozen=True)
class AgentConfig:
    """How the agent behaves."""

    root_path: Path  # the sandbox root, already absolute and guaranteed to exist
    max_steps: int = 12
    system_prompt: str | None = None  # already read from `system_prompt_file`; None = built-in


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    agent: AgentConfig
    enabled_tools: tuple[str, ...] | None  # None = every registered tool
    source: Path  # where this config came from, for error messages


def load_config(path: str | Path) -> Config:
    """Read, validate, and resolve a YAML config file.

    Raises `ConfigError` -- and only `ConfigError` -- for anything wrong with the file.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}. Copy config.example.yaml to config.yaml."
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    _reject_unknown_keys("the top level", raw, {"llm", "agent", "tools"})

    # Paths in the file are relative to the file itself, not to the shell's working
    # directory, so a config keeps working no matter where you run the CLI from.
    config_dir = config_path.parent

    return Config(
        llm=_load_llm(_require_section(raw, "llm")),
        agent=_load_agent(_require_section(raw, "agent"), config_dir),
        enabled_tools=_load_enabled_tools(raw.get("tools")),
        source=config_path,
    )


def _load_llm(section: dict[str, Any]) -> LLMConfig:
    _reject_unknown_keys(
        "section 'llm'",
        section,
        {
            "base_url",
            "model",
            "api_key_env",
            "temperature",
            "max_tokens",
            "timeout_seconds",
            "max_attempts",
        },
    )

    # The file stores the *name* of an environment variable, so the secret itself stays out
    # of the repo, out of your shell history, and out of any config you paste into a chat.
    api_key_env = str(_require(section, "api_key_env", "llm"))
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ConfigError(
            f"Environment variable {api_key_env!r} (llm.api_key_env) is unset or empty. "
            f"Export it before running: export {api_key_env}=..."
        )

    return LLMConfig(
        # Trailing slashes would produce '…/v1//chat/completions' on some servers.
        base_url=str(_require(section, "base_url", "llm")).rstrip("/"),
        model=str(_require(section, "model", "llm")),
        api_key=api_key,
        temperature=_as_float(section.get("temperature", 0.0), "llm.temperature"),
        max_tokens=_as_int(section.get("max_tokens", 2048), "llm.max_tokens", minimum=1),
        timeout_seconds=_as_int(
            section.get("timeout_seconds", 60), "llm.timeout_seconds", minimum=1
        ),
        max_attempts=_as_int(section.get("max_attempts", 3), "llm.max_attempts", minimum=1),
    )


def _load_agent(section: dict[str, Any], config_dir: Path) -> AgentConfig:
    _reject_unknown_keys(
        "section 'agent'", section, {"root_path", "max_steps", "system_prompt_file"}
    )

    root_path = _relative_to_config(str(_require(section, "root_path", "agent")), config_dir)
    if root_path.exists() and not root_path.is_dir():
        raise ConfigError(f"agent.root_path exists but is not a directory: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

    system_prompt = None
    prompt_file = section.get("system_prompt_file")
    if prompt_file is not None:
        prompt_path = _relative_to_config(str(prompt_file), config_dir)
        if not prompt_path.is_file():
            raise ConfigError(f"agent.system_prompt_file not found: {prompt_path}")
        system_prompt = prompt_path.read_text(encoding="utf-8")

    return AgentConfig(
        root_path=root_path,
        max_steps=_as_int(section.get("max_steps", 12), "agent.max_steps", minimum=1),
        system_prompt=system_prompt,
    )


def _load_enabled_tools(section: Any) -> tuple[str, ...] | None:
    """Return the allow-list, or None meaning 'every registered tool'.

    Tool *names* are not checked here -- config.py deliberately knows nothing about which
    tools exist. `tools.build_registry` owns that and reports unknown names.
    """
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ConfigError("Section 'tools:' must be a mapping")
    _reject_unknown_keys("section 'tools'", section, {"enabled"})

    enabled = section.get("enabled")
    if enabled is None:
        return None
    if not isinstance(enabled, list) or not all(isinstance(name, str) for name in enabled):
        raise ConfigError("tools.enabled must be a list of tool names")
    if not enabled:
        raise ConfigError(
            "tools.enabled is empty. Remove the section to enable every tool, "
            "or list the ones you want."
        )
    duplicates = sorted({name for name in enabled if enabled.count(name) > 1})
    if duplicates:
        raise ConfigError(f"Duplicate tool name(s) in tools.enabled: {duplicates}")
    return tuple(enabled)


# --- small validation helpers -------------------------------------------------------------


def _relative_to_config(value: str, config_dir: Path) -> Path:
    """Resolve a path from the config against the config's own directory.

    An absolute value replaces `config_dir` under pathlib's `/` semantics, which is exactly
    what we want: absolute paths in the config are honoured as written.
    """
    return (config_dir / value).resolve()


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if section is None:
        raise ConfigError(f"Missing required section '{name}:'")
    if not isinstance(section, dict):
        raise ConfigError(f"Section '{name}:' must be a mapping of keys to values")
    return section


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if section.get(key) is None:
        raise ConfigError(f"Missing required key '{key}' in section '{where}:'")
    return section[key]


def _reject_unknown_keys(where: str, mapping: dict[str, Any], allowed: set[str]) -> None:
    """Fail on unrecognised keys, so a misspelled option is never silently ignored."""
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {unknown} in {where}. Allowed keys: {sorted(allowed)}"
        )


def _as_int(value: Any, where: str, *, minimum: int) -> int:
    # bool is a subclass of int in Python, and `max_steps: true` should not mean 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be a whole number, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{where} must be at least {minimum}, got {value}")
    return value


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    return float(value)
