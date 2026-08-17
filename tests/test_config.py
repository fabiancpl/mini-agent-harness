"""Tests for loading and validating config.yaml.

The theme: every mistake a reader could plausibly make in the YAML file produces a
`ConfigError` with a sentence explaining it, at startup, and never a silent default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.config import load_config
from mini_agent.errors import ConfigError

from conftest import API_KEY_ENV, base_config  # pytest puts tests/ on sys.path

# --- the happy path -----------------------------------------------------------------------


def test_loads_a_minimal_config(write_config) -> None:
    config = load_config(write_config())

    assert config.llm.base_url == "https://example.test/v1"
    assert config.llm.model == "test-model"
    assert config.llm.api_key == "sk-test-key"
    assert config.enabled_tools is None  # no tools section = every registered tool


def test_applies_documented_defaults(write_config) -> None:
    config = load_config(write_config())

    assert config.llm.temperature == 0.0
    assert config.llm.max_tokens == 2048
    assert config.llm.timeout_seconds == 60
    assert config.agent.max_steps == 12
    assert config.agent.system_prompt is None


def test_reads_explicit_values(write_config) -> None:
    data = base_config()
    data["llm"].update(temperature=0.7, max_tokens=100, timeout_seconds=5)
    data["agent"]["max_steps"] = 3

    config = load_config(write_config(data))

    assert (config.llm.temperature, config.llm.max_tokens, config.llm.timeout_seconds) == (
        0.7,
        100,
        5,
    )
    assert config.agent.max_steps == 3


def test_strips_a_trailing_slash_from_base_url(write_config) -> None:
    data = base_config()
    data["llm"]["base_url"] = "https://example.test/v1/"

    assert load_config(write_config(data)).llm.base_url == "https://example.test/v1"


def test_the_shipped_example_config_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    # config.example.yaml is the first thing a reader copies; it must actually load.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    example = Path(__file__).resolve().parent.parent / "config.example.yaml"

    config = load_config(example)

    assert config.enabled_tools is not None
    assert config.agent.root_path.is_dir()


# --- the API key comes from the environment, never from the file --------------------------


def test_resolves_the_api_key_from_the_named_environment_variable(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "sk-from-env")

    assert load_config(write_config()).llm.api_key == "sk-from-env"


def test_rejects_a_missing_api_key_variable(write_config, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_config()
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.raises(ConfigError, match=API_KEY_ENV):
        load_config(path)


def test_rejects_an_empty_api_key_variable(write_config, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_config()
    monkeypatch.setenv(API_KEY_ENV, "")

    with pytest.raises(ConfigError, match="unset or empty"):
        load_config(path)


# --- paths resolve against the config file, not the shell ---------------------------------


def test_root_path_resolves_relative_to_the_config_file(write_config, tmp_path: Path) -> None:
    path = write_config(name="nested/config.yaml")

    config = load_config(path)

    assert config.agent.root_path == tmp_path / "nested" / "workspace"


def test_root_path_is_independent_of_the_working_directory(
    write_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_config(path).agent.root_path == tmp_path / "workspace"


def test_an_absolute_root_path_is_honoured(write_config, tmp_path: Path) -> None:
    data = base_config()
    data["agent"]["root_path"] = str(tmp_path / "somewhere_else")

    assert load_config(write_config(data)).agent.root_path == tmp_path / "somewhere_else"


def test_creates_the_root_directory_if_it_is_missing(write_config, tmp_path: Path) -> None:
    config = load_config(write_config())

    assert config.agent.root_path.is_dir()
    assert config.agent.root_path == tmp_path / "workspace"


def test_rejects_a_root_path_that_is_a_file(write_config, tmp_path: Path) -> None:
    (tmp_path / "workspace").write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not a directory"):
        load_config(write_config())


def test_loads_a_system_prompt_file_relative_to_the_config(write_config, tmp_path: Path) -> None:
    (tmp_path / "prompt.txt").write_text("You are a careful agent.\n", encoding="utf-8")
    data = base_config()
    data["agent"]["system_prompt_file"] = "./prompt.txt"

    assert load_config(write_config(data)).agent.system_prompt == "You are a careful agent.\n"


def test_rejects_a_missing_system_prompt_file(write_config) -> None:
    data = base_config()
    data["agent"]["system_prompt_file"] = "./nope.txt"

    with pytest.raises(ConfigError, match="system_prompt_file"):
        load_config(write_config(data))


# --- malformed files ----------------------------------------------------------------------


def test_rejects_a_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("llm: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_rejects_yaml_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


@pytest.mark.parametrize("section", ["llm", "agent"])
def test_rejects_a_missing_required_section(write_config, section: str) -> None:
    data = base_config()
    del data[section]

    with pytest.raises(ConfigError, match=f"'{section}:'"):
        load_config(write_config(data))


@pytest.mark.parametrize(
    ("section", "key"),
    [("llm", "base_url"), ("llm", "model"), ("llm", "api_key_env"), ("agent", "root_path")],
)
def test_rejects_a_missing_required_key(write_config, section: str, key: str) -> None:
    data = base_config()
    del data[section][key]

    with pytest.raises(ConfigError, match=key):
        load_config(write_config(data))


def test_rejects_a_section_that_is_not_a_mapping(write_config) -> None:
    data = base_config()
    data["agent"] = "./workspace"

    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(write_config(data))


# --- typos are caught, never ignored ------------------------------------------------------


def test_rejects_an_unknown_top_level_section(write_config) -> None:
    data = base_config()
    data["agnet"] = {}

    with pytest.raises(ConfigError, match="agnet"):
        load_config(write_config(data))


def test_rejects_an_unknown_key_inside_a_section(write_config) -> None:
    # A misspelled option silently doing nothing is the worst possible outcome: the user
    # believes they set max_steps and the agent quietly uses the default.
    data = base_config()
    data["agent"]["max_step"] = 3

    with pytest.raises(ConfigError, match="max_step"):
        load_config(write_config(data))


def test_rejects_an_api_key_written_directly_into_the_file(write_config) -> None:
    # api_key is not a valid key -- only api_key_env is. This keeps secrets out of the repo.
    data = base_config()
    data["llm"]["api_key"] = "sk-oops-committed-to-git"

    with pytest.raises(ConfigError, match="api_key"):
        load_config(write_config(data))


# --- value types --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["twelve", 2.5, None, True])
def test_rejects_a_non_integer_max_steps(write_config, value) -> None:
    data = base_config()
    data["agent"]["max_steps"] = value

    with pytest.raises(ConfigError, match="max_steps"):
        load_config(write_config(data))


def test_rejects_a_max_steps_below_one(write_config) -> None:
    data = base_config()
    data["agent"]["max_steps"] = 0

    with pytest.raises(ConfigError, match="at least 1"):
        load_config(write_config(data))


def test_rejects_a_non_numeric_temperature(write_config) -> None:
    data = base_config()
    data["llm"]["temperature"] = "hot"

    with pytest.raises(ConfigError, match="temperature"):
        load_config(write_config(data))


def test_accepts_an_integer_temperature_as_a_float(write_config) -> None:
    data = base_config()
    data["llm"]["temperature"] = 1

    config = load_config(write_config(data))

    assert isinstance(config.llm.temperature, float)
    assert config.llm.temperature == 1.0


# --- the tools allow-list -----------------------------------------------------------------


def test_reads_the_enabled_tools_in_order(write_config) -> None:
    data = base_config()
    data["tools"] = {"enabled": ["read_file", "list_directory"]}

    assert load_config(write_config(data)).enabled_tools == ("read_file", "list_directory")


def test_an_omitted_tools_section_means_every_tool(write_config) -> None:
    assert load_config(write_config()).enabled_tools is None


def test_rejects_an_empty_enabled_list(write_config) -> None:
    data = base_config()
    data["tools"] = {"enabled": []}

    with pytest.raises(ConfigError, match="empty"):
        load_config(write_config(data))


def test_rejects_duplicate_tool_names(write_config) -> None:
    data = base_config()
    data["tools"] = {"enabled": ["read_file", "read_file"]}

    with pytest.raises(ConfigError, match="Duplicate"):
        load_config(write_config(data))


def test_rejects_a_non_list_enabled_value(write_config) -> None:
    data = base_config()
    data["tools"] = {"enabled": "read_file"}

    with pytest.raises(ConfigError, match="list of tool names"):
        load_config(write_config(data))


def test_rejects_a_tools_section_that_is_not_a_mapping(write_config) -> None:
    data = base_config()
    data["tools"] = ["read_file"]  # forgot the 'enabled:' key

    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(write_config(data))


def test_rejects_an_unknown_key_in_the_tools_section(write_config) -> None:
    data = base_config()
    data["tools"] = {"enabled": ["read_file"], "disabled": ["write_file"]}

    with pytest.raises(ConfigError, match="disabled"):
        load_config(write_config(data))


def test_an_empty_tools_section_means_every_tool(write_config) -> None:
    data = base_config()
    data["tools"] = {}

    assert load_config(write_config(data)).enabled_tools is None


def test_does_not_validate_tool_names_here(write_config) -> None:
    # config.py knows nothing about which tools exist; build_registry reports unknown names.
    data = base_config()
    data["tools"] = {"enabled": ["delete_everything"]}

    assert load_config(write_config(data)).enabled_tools == ("delete_everything",)
