# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

`mini-agent-harness` is a small, readable **ReAct agent harness**: an LLM loop that can
navigate and edit files inside one sandboxed folder, driven from a CLI and configured
from a YAML file.

It is **teaching code**. The audience is someone who wants to read the whole thing in an
afternoon and understand exactly how an agent loop works. That goal outranks performance,
feature coverage, and defensive robustness.

- **Clarity over robustness.** Prefer 20 obvious lines to 5 clever ones. If a trade-off
  arises between "handles every edge case" and "a reader immediately sees why this works",
  pick the reader. Handle the edge case with a clear error instead.
- **No hidden magic.** No metaclasses, no dynamic imports, no plugin auto-discovery, no
  decorators that rewrite behaviour. Tools are registered explicitly in one function.
- **Stdlib first.** Third-party dependencies are `pyyaml`, `requests`, and `pytest`. Do not
  add more without a strong reason.
- **Comment the *why*.** The code says what it does; comments explain the design decision
  behind it (especially in `sandbox.py` and `agent.py`).

## Commands

```bash
uv sync --extra dev                 # install (or: pip install -e ".[dev]")

pytest                              # run the full test suite
pytest -q tests/test_sandbox.py     # run one module
pytest --cov=mini_agent             # coverage

cp config.example.yaml config.yaml  # create a local config (gitignored)
export OPENAI_API_KEY=sk-...        # the key is read from the env, never from the file

python -m mini_agent --help
python -m mini_agent                              # interactive REPL
python -m mini_agent --task "summarise README.md" # one-shot
python -m mini_agent --config config.yaml --verbose
```

## Architecture

The dependency direction is strictly one-way: `cli → agent → registry → tools → sandbox`.
Nothing lower ever imports something higher, which is what keeps every layer unit-testable
in isolation.

| Module | Responsibility |
| --- | --- |
| `config.py` | Load + validate YAML into frozen dataclasses. Resolves the API key from the environment. |
| `sandbox.py` | The security boundary. Turns an agent-supplied path string into a real path proven to live under the root. |
| `registry.py` | `Tool` (name, description, JSON-Schema params, handler) and `ToolRegistry` (register / get / list / export OpenAI schemas). |
| `tools/` | The actual tool implementations. Each takes a `Sandbox` plus keyword arguments and returns a string observation. |
| `llm.py` | Thin OpenAI-compatible `/chat/completions` client. One method: `complete(messages, tools) -> Message`. |
| `agent.py` | The ReAct loop: think → call tool → observe → repeat, until a final answer or `max_steps`. |
| `cli.py` | Argument parsing, the REPL, and printing the trace. No business logic. |

See `PLAN.md` for the full design, tool contracts, and the implementation checklist.

## Invariants — do not break these

These are the point of the project. Every one of them has a dedicated test; if you change
behaviour here, the tests must be updated deliberately, never "fixed" to pass.

1. **Every path an agent supplies passes through `Sandbox.resolve()` before any filesystem
   call.** No tool may call `open()`, `Path.mkdir()`, or `os.listdir()` on a raw argument.
2. **Nothing escapes the root.** `..` traversal, absolute paths outside the root, and
   symlinks pointing outside the root are all rejected with `PathOutsideRootError`.
   Resolution happens *before* the access, not after.
3. **No deletion is possible.** There is no delete/remove/rename tool in the registry, and
   no tool truncates a directory. The capability is absent by construction, not blocked by
   a check. Do not add one.
4. **Tool failures are observations, never crashes.** A `ToolError` is caught by the loop
   and fed back to the model as the observation text so it can recover. Only genuinely
   unexpected exceptions propagate.
5. **The loop is bounded.** `max_steps` always terminates the loop, and the agent reports
   that it ran out of steps rather than pretending to answer.
6. **Secrets stay out of the repo.** The config file holds `api_key_env`, a variable *name*.
   `config.yaml`, `.env`, and `workspace/` are gitignored.

## Conventions

- Python 3.12, 4-space indent, ~100 column lines, double quotes.
- Type hints on every public function; `from __future__ import annotations` at the top.
- Dataclasses for data (`frozen=True` for config), plain functions for behaviour. Classes
  only where there is real state: `Sandbox`, `ToolRegistry`, `LLMClient`, `Agent`.
- Custom exceptions live in `errors.py` and all derive from `HarnessError`.
- Tool handlers return a **string** — it is what the model literally sees next. Write these
  strings for an LLM reader: short, factual, no ANSI colour, no emoji.

## Testing

- `pytest`, plain `assert`, no mocking library — the `tmp_path` fixture and small fakes
  (`FakeLLMClient`) are enough and are easier to read.
- The network is never touched by the suite. `llm.py` is tested by monkeypatching
  `requests.post`; `agent.py` is tested with a scripted fake client.
- Every test names the behaviour it protects: `test_read_file_rejects_parent_traversal`.
- New tool ⇒ new test module covering the happy path, an input error, and an escape attempt.
