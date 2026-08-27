# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-20

First release: a complete, readable ReAct agent harness that navigates and edits files
inside one sandboxed folder. Built for teaching — the whole thing is meant to be read in an
afternoon, so it favours obvious code over clever code and explains its design decisions in
comments.

### Added

- **ReAct loop** (`agent.py`) — think → act → observe → repeat, bounded by `max_steps`,
  using native tool calling rather than scraping `Thought:`/`Action:` text. Tool failures
  become observations the model can recover from; only genuine bugs propagate.
- **Sandbox** (`sandbox.py`) — the security boundary. Joins a model-supplied path onto the
  root, fully resolves it (normalising `..` *and* following symlinks), then checks
  containment. Resolving before judging is what makes the check honest.
- **Ten tools** (`tools/`) — `list_directory`, `find_files`, `search_text`, `read_file`,
  `create_directory`, `write_file`, `append_to_file`, `edit_file`, `move`, `copy`. Wired in
  one explicit function, `build_registry`.
- **Tool registry** (`registry.py`) — binds model-supplied arguments to handler signatures
  before calling, so a malformed argument is a recoverable `ToolError` rather than a
  `TypeError` that ends the run.
- **OpenAI-compatible client** (`llm.py`) — works against OpenAI, Ollama, vLLM, LM Studio,
  or any gateway speaking `/chat/completions`.
- **YAML configuration** (`config.py`) — validated once at startup into frozen dataclasses.
  Unknown keys are errors, never silently ignored. The API key is named, not stored.
- **CLI** (`cli.py`) — one-shot `--task`, an interactive REPL, `--verbose` traces, and
  `--root`/`--max-steps` overrides. Every expected failure prints one line, never a
  traceback.
- **Offline demo** (`examples/offline_demo.py`) — runs the whole stack over a real socket
  with **no API key**, scripted to show both safety refusals and the model recovering.
- **Documentation** — `README.md` (setup, providers, troubleshooting), `PLAN.md` (design and
  contracts), `TESTING.md` (test layers, what "end-to-end" means for an agent, why evals stay
  out of the suite), `CLAUDE.md` (invariants and conventions).

### Safety properties

The design rule, stated about effects rather than names:

> **No operation may make existing content unreachable.**

- No delete, remove, unlink, rename-over, or shell tool exists. The capability was never
  built, so it cannot be argued into existence by a prompt.
- `move` and `copy` refuse an existing destination, which keeps every relocation invertible.
- `write_file` and `edit_file` *can* destroy content — that is acknowledged rather than
  papered over, and is why running the workspace under version control is recommended.
- Nothing escapes the sandbox root: `..` traversal, absolute paths, and symlinks pointing
  outside are all refused, on both ends of `move` and `copy`.

### Tested

286 tests, 99% line coverage with no partial branches, ~2 seconds, no network beyond
localhost. Three guards that sat at full coverage while the case they exist for never ran
were found and covered; see the warning at the end of `TESTING.md`.

### Known limitations

- **No conversational memory between tasks.** Each REPL turn starts a fresh conversation.
  The workspace persists, so the agent can rediscover its own work without remembering it.
- **Folders accumulate** — there is no `rmdir` and nothing empties a directory.
- **No evals.** Running against a real provider is a measurement, not a test, and is
  deliberately deferred. `TESTING.md` has the reasoning and a sketch.
- Out of scope by design: streaming, retries/backoff, token accounting, parallel tool
  execution, sub-agents, MCP, shell execution.

[0.1.0]: https://github.com/fabiancpl/mini-agent-harness/releases/tag/v0.1.0
