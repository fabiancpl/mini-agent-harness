# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **`--dump-transcript` creates the directory it is asked to write into.** `--dump-transcript
  dumps/run.json` failed with `No such file or directory` unless `dumps/` already existed,
  which is the first thing anyone types to keep transcripts out of the way. Only missing
  parents are created; a dump that still cannot be written is reported and the run's answer
  is kept, as before.

## [0.2.0] — 2026-08-27

The "forkable" release. 0.1.0 was built to be read; an evaluation of it found that reading
worked and *extending* did not — a config trap silently dropped newly added tools, nothing
documented the extension seam, and a finished run threw away the conversation that produced
it. This release closes those three and builds the evals that `TESTING.md` had sketched.

No new agent capability: the agent still has exactly ten tools and still cannot delete
anything.

### Added

- **`EXTENDING.md`** — the missing walkthrough. The tool contract, adding a tool end to end,
  swapping the model client, reading a transcript, measuring a prompt change, and the one
  invariant not to cross. The worked example is real tested code
  (`examples/word_count_tool.py`), not a fenced block that can rot.
- **`evals/run_eval.py`** — opt-in measurement against a real model, four tasks with property
  checkers, printing a success-rate table. Never collected by `pytest`; exits 0 even on
  failure, because a measurement is not a gate. The checkers *are* unit-tested — one that can
  never fail would report 5/5 forever and nothing would notice.
- **`AgentResult.messages`** — the verbatim conversation a run sent, exposed for inspection.
- **`--dump-transcript PATH`** — writes that conversation as JSON.
- **`SupportsComplete`** — a protocol for the model client, so the seam the tests always used
  is visible in the types. `py.typed` ships the hints to consumers.
- **`ruff`** in the dev extra, configured to the conventions `CLAUDE.md` already documents.

### Changed

- **`config.example.yaml` ships `tools.enabled` commented out**, so a copied config enables
  every registered tool and a newly added one appears without a second edit. The explicit
  list was a trap: `build_registry` rejects unknown names but says nothing about registered
  names left off the list, so a correctly written tool silently failed to load.
- **The startup banner prints `10 of 11 registered`** whenever an allow-list is trimming the
  registry, so the same mistake is visible if a hand-written config recreates it.

### Upheld

`PLAN.md` §11 was not reopened: still no token accounting, retries, streaming, conversational
memory, parallel tool execution, sub-agents, MCP, or shell execution. `AgentResult.messages`
is not memory — the loop still builds its history inside `run` and every task still starts
from the system prompt, which `test_exposing_the_transcript_did_not_make_the_loop_stateful`
enforces. "Make the REPL remember previous turns" stays a reader exercise, because meeting
context growth yourself is the point of it. See §12.

### Known limitations

- Unchanged from 0.1.0: no conversational memory between tasks, folders accumulate, and
  streaming/retries/token accounting remain out of scope by design.
- The OpenAI message shape is still built inline in `Agent.run`, so a provider with a
  different shape needs edits to the loop and not just a new client. Named explicitly in
  `EXTENDING.md` rather than left to be discovered.

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
  deliberately deferred. `TESTING.md` has the reasoning and a sketch. *(Built in 0.2.0.)*
- Out of scope by design: streaming, retries/backoff, token accounting, parallel tool
  execution, sub-agents, MCP, shell execution.

[0.2.0]: https://github.com/fabiancpl/mini-agent-harness/releases/tag/v0.2.0
[0.1.0]: https://github.com/fabiancpl/mini-agent-harness/releases/tag/v0.1.0
