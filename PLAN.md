# Implementation Plan

Design and build order for `mini-agent-harness`. Read `CLAUDE.md` first for the project's
guiding principles; this document is the concrete specification.

## 1. Goal

A minimal but complete agent harness:

- a **ReAct loop** (reason → act → observe → repeat) over an OpenAI-compatible LLM API,
- a **CLI** for interactive and one-shot use,
- **YAML configuration** for the endpoint, model, API key source, and agent limits,
- a **tool registry** giving the agent file-system capabilities that are *safe by
  construction*: it can navigate, search, read, create, modify, and reorganise inside one
  root folder, but no tool can make existing content unreachable, and none can touch a path
  outside that root.

## 2. Repository layout

```
mini-agent-harness/
├── CLAUDE.md               # working guidance + invariants
├── EXTENDING.md            # adding a tool, swapping the client, where the lines are
├── PLAN.md                 # this file
├── README.md               # quickstart, real-endpoint setup, troubleshooting
├── TESTING.md              # test layers, what e2e means here, why evals stay out
├── examples/
│   ├── offline_demo.py     # the whole stack over a real socket, no API key
│   └── word_count_tool.py  # the EXTENDING.md walkthrough, as real tested code
├── evals/
│   └── run_eval.py         # opt-in measurement against a real model; never in pytest
├── pyproject.toml          # package metadata, deps, pytest config
├── config.example.yaml     # documented template; copy to config.yaml
├── src/mini_agent/
│   ├── __init__.py
│   ├── __main__.py         # enables `python -m mini_agent`
│   ├── errors.py           # HarnessError hierarchy
│   ├── config.py           # YAML → frozen dataclasses
│   ├── sandbox.py          # the path security boundary
│   ├── registry.py         # Tool + ToolRegistry
│   ├── tools/
│   │   ├── __init__.py     # build_registry(sandbox, allowed) — the ONE place tools are wired
│   │   ├── navigate.py     # list_directory, find_files, search_text
│   │   ├── read.py         # read_file
│   │   └── write.py        # create_directory, write_file, append_to_file,
│   │                       # edit_file, move, copy
│   ├── llm.py              # OpenAI-compatible chat client
│   ├── agent.py            # the ReAct loop
│   └── cli.py              # argparse + REPL + trace printing
└── tests/                  # one module per source module
```

## 3. Configuration

`config.yaml` (copied from `config.example.yaml`). Loaded by `config.py` into frozen
dataclasses so a typo fails at startup with a clear message rather than at step 7 of a run.

```yaml
llm:
  base_url: "https://api.openai.com/v1"   # any OpenAI-compatible server: Ollama, vLLM, LM Studio…
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"           # a variable NAME — the secret never lives in the file
  temperature: 0.0
  max_tokens: 2048
  timeout_seconds: 60

agent:
  root_path: "./workspace"                # the sandbox; every tool path resolves under it
  max_steps: 12                           # hard bound on the ReAct loop
  system_prompt_file: null                # optional path to override the built-in prompt

tools:
  enabled:                                # registry is the menu, this is the order
    - list_directory                      # omit the whole section to enable everything;
    - find_files                          # list only the first four for a read-only agent
    - search_text
    - read_file
    - create_directory
    - write_file
    - append_to_file
    - edit_file
    - move
    - copy
```

Rules:

- `api_key_env` names an environment variable; `load_config` reads it and raises
  `ConfigError` if it is unset. This keeps secrets out of git and demonstrates the pattern.
  `api_key` is not a valid key, so a secret pasted into the file fails loudly.
- Unknown top-level sections or unknown keys ⇒ `ConfigError` listing the offending key. A
  misspelled option must never be silently ignored.
- Omitting `tools:` (or `tools.enabled`) enables every registered tool. `config.py`
  deliberately does not know which tools exist, so an unknown *name* is caught one layer
  down by `build_registry`, as a `RegistryError`.
- `root_path` and `system_prompt_file` resolve relative to the **config file's** directory,
  so a config is portable and does not depend on the shell's cwd. The root is created if
  missing. (A `--root` on the command line is relative to the shell instead — the least
  surprising rule for each source.)

## 4. The security boundary: `Sandbox`

One class, one job. Everything else in the project trusts it.

```python
class Sandbox:
    def __init__(self, root: Path) -> None: ...      # stores root.resolve()
    def resolve(self, user_path: str) -> Path: ...   # -> real path, guaranteed inside root
    def relative(self, path: Path) -> str: ...       # -> display path, for observations
```

`resolve` is deliberately three steps:

1. Join the argument onto the root (`self.root / user_path`). An absolute argument replaces
   the root under `pathlib` semantics, which is exactly why step 3 exists.
2. `.resolve()` — normalises `..` segments **and follows symlinks**, so a symlink pointing
   at `/etc` becomes `/etc` before we judge it.
3. `is_relative_to(self.root)` — if false, raise `PathOutsideRootError`.

Checking *after* resolution is the whole trick: string-inspecting for `".."` is the classic
broken version, and a symlink defeats it. Observations report sandbox-relative paths only,
so the model never learns the host's absolute layout.

Cases the tests pin down: `"notes.txt"`, `"a/b/c.txt"`, `"."`, `"./x"`, `"../../etc/passwd"`,
`"/etc/passwd"`, an absolute path that *is* inside the root, a symlink to outside, a symlink
to inside, and the root itself.

## 5. Tool registry

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str          # written for the model: what it does and when to use it
    parameters: dict          # JSON Schema, sent verbatim as the function schema
    handler: Callable[..., str]

class ToolRegistry:
    def register(self, tool: Tool) -> None      # duplicate name ⇒ RegistryError
    def get(self, name: str) -> Tool            # unknown name ⇒ RegistryError listing the real ones
    def names(self) -> list[str]
    def to_schemas(self) -> list[dict]          # OpenAI `tools` array
    def subset(self, names) -> ToolRegistry     # the allow-list, in the order given
    def invoke(self, name, arguments) -> str    # bad arguments ⇒ ToolError, not TypeError
```

`invoke` binds the model's arguments to the handler's signature *before* calling it. Model
arguments are untrusted JSON: without that step a missing or misspelled key surfaces as a
`TypeError` and kills the run over a mistake the model could have fixed on the next step.

`tools/__init__.py::build_registry(sandbox, enabled)` is the single wiring point: it
constructs every tool, keeps those in `enabled`, and preserves that order.

### Tool contracts

Every handler takes the `Sandbox` plus keyword arguments straight from the model's JSON, and
returns a plain-text observation. Bad input raises `ToolError`, which the loop converts into
an observation so the model can retry.

| Tool | Parameters | Returns / raises |
| --- | --- | --- |
| `list_directory` | `path` (default `"."`) | One entry per line, directories suffixed `/`, sorted dirs-then-files. `ToolError` if missing or not a directory. |
| `find_files` | `pattern`, `path` (default `"."`) | Recursive glob (`**/`), sandbox-relative paths, capped at 200 hits with a truncation note. |
| `read_file` | `path`, `start_line` (default 1), `max_lines` (default 400) | Numbered lines. `ToolError` if missing, a directory, larger than 1 MiB, or not UTF-8 decodable. |
| `create_directory` | `path` | Creates parents; succeeding on an existing directory is fine (idempotent). `ToolError` if a *file* is already there. |
| `write_file` | `path`, `content` | Creates or fully overwrites; creates parent directories. `ToolError` if the path is a directory. |
| `search_text` | `pattern`, `path` (default `"."`), `file_pattern` (default `"*"`) | Regex search of file contents, returning `path:line: text`. Lines trimmed to 200 chars, capped at 100 hits. Binary and oversized files are **skipped and counted**, not errors — unlike `read_file`, a tree search must step over what it cannot read. `ToolError` on an invalid regex. |
| `append_to_file` | `path`, `content` | Appends, creating the file and parents if missing. The only mutation that cannot destroy anything. `ToolError` if the path is a directory. |
| `move` | `source`, `destination` | Renames or relocates a file or directory. `destination` is the complete new path, never a folder to drop into. |
| `copy` | `source`, `destination` | Same contract as `move`, but the source stays. Directories copy recursively. |

`move` and `copy` share `_resolve_relocation`, which enforces, in order: both paths inside
the root (a relocation has two ends — checking only the source would be a hole); the source
exists; **the destination does not exist**; and the destination is not inside the source.

That last check is load-bearing, not cosmetic. `move("src", "src/nested")` fails with a bare
`EINVAL`; `copy("src", "src/nested")` *succeeds*, silently leaving a copy of `src` inside
itself; and `copy("src", "src/deep/here")` walks into what it is writing and raises
`RecursionError` — which is neither an `OSError` (so `copy`'s handler misses it) nor a
`HarnessError` (so the agent loop will not turn it into an observation), and would therefore
take down the whole run. Refusing up front keeps all three recoverable.

### Files and folders

Six of the ten tools accept a directory: `list_directory`, `find_files`, `search_text` (as a
search root — it only reads inside files, but walks folders recursively), `create_directory`,
`move`, and `copy`. The four file-content tools — `read_file`, `write_file`,
`append_to_file`, `edit_file` — refuse one, with an error naming the tool that does the job,
so the model recovers in a single step rather than guessing.

Two consequences worth stating plainly:

- **Folders accumulate.** There is no `rmdir` and no tool that empties a directory, so an
  agent that creates `a/b/c/d` leaves it there forever. This is the price of the invariant.
- **`create_directory` is idempotent while `move`/`copy` hard-refuse an existing
  destination.** Deliberate — re-creating a folder costs nothing and is not worth a step to
  correct, whereas overwriting one destroys it — but the asymmetry is real and a reader will
  notice it.

`move` is also the one operation that can make a folder *name* disappear: after
`move("docs", "guide")` a listing no longer shows `docs`. Nothing became unreachable — every
file is under `guide/`, and `move("guide", "docs")` restores it exactly — but it is the
exception to "paths are only ever added".

**The invariant:** *no operation may make existing content unreachable.* Stated about
effects rather than names, because `write_file` already overwrites — the promise is that
nothing becomes **unrecoverable**. `move(a, b)` with `b` free is undone by `move(b, a)`;
`move` onto an existing `b` has no inverse, so it is refused. Deletion has no such
precondition, which is why there is no delete, remove, unlink, or shell tool. A capability
you never grant cannot be talked out of you by a jailbreak.

## 6. LLM client

`llm.py` speaks OpenAI-compatible `POST {base_url}/chat/completions` via `requests`, so the
same code works against OpenAI, Ollama, vLLM, LM Studio, or a gateway.

```python
@dataclass(frozen=True)
class ToolCall:  id: str; name: str; arguments: dict
@dataclass(frozen=True)
class Message:   content: str | None; tool_calls: list[ToolCall]

class LLMClient:
    def complete(self, messages: list[dict], tools: list[dict]) -> Message: ...
```

It sends `tools` + `tool_choice: "auto"` and parses `choices[0].message`. Malformed JSON in
`tool_calls.function.arguments` raises `LLMError` with the raw text attached — a real failure
mode of small local models, worth showing rather than hiding. Non-2xx ⇒ `LLMError` including
the response body.

## 7. The ReAct loop

`agent.py` is the centrepiece and should read like the paper's pseudocode.

```
messages = [system, user(task)]
for step in 1..max_steps:
    message = llm.complete(messages, registry.to_schemas())
    record message.content as the THOUGHT
    if not message.tool_calls:
        return final answer = message.content
    append the assistant message
    for each tool_call:                     # models may batch several
        observation = run tool, or the ToolError text
        append {role: "tool", tool_call_id, content: observation}
return "stopped after max_steps"
```

Design notes:

- **Native tool calling, not text parsing.** Classic ReAct scrapes `Thought:/Action:` out of
  free text; every production harness uses the API's structured tool calls instead. We keep
  the ReAct *shape* — the system prompt asks the model to state its reasoning in `content`
  before acting, so each step still has a visible Thought — while parsing structured JSON.
  Comment this trade-off in the code; it is one of the things a reader most needs explained.
- Each iteration is recorded in a `Step` dataclass (`number`, `thought`, `observations`) and
  returned in `AgentResult` alongside the final answer. The CLI prints it; tests assert
  against it. The loop itself never prints — an optional `on_step` callback is how the CLI
  shows progress live, which keeps all I/O in one file and the agent usable as a library.
- `ToolError` and unknown tool names become observations. The model gets a chance to fix its
  own mistake, which is most of what makes an agent feel capable.

## 8. CLI

`python -m mini_agent [--config PATH] [--task TEXT] [--root PATH] [--max-steps N] [--verbose]`

- No `--task` ⇒ REPL: read a task, run the agent, print the answer, repeat. `exit`/`quit`/
  Ctrl-D leaves; Ctrl-C aborts the current run, not the session.
- `--task` ⇒ run once, print, exit (exit code 0 on success, 1 on `HarnessError`).
- `--verbose` prints the full trace per step: thought, tool + arguments, observation
  (truncated). Default prints a one-line-per-step summary.
- `--root` / `--max-steps` override the config, showing the usual precedence
  CLI > config > default.
- `main(argv, llm_factory=LLMClient)` returns an exit code rather than calling `sys.exit`,
  and takes both parameters so a test can drive a whole session — REPL included — with a
  scripted client and no network. All configuration errors print one clear line, no
  traceback. Exit code 1 also covers running out of steps, so scripts can detect it.

## 9. Test matrix

`pytest`, no mocking library — `tmp_path` and small fakes only. Target: every invariant in
`CLAUDE.md` has at least one test that fails if the invariant is removed.

**As built: 286 tests, 99% line coverage and no partial branches** (`__main__.py` is covered
by a subprocess test that `coverage` cannot see into; every other module is at 100% under
`--cov-branch`). Runs in ~2s, no network.

A caution the suite earned the hard way: full coverage is not proof that a guard is tested.
Three checks here were green while the case they exist for never ran — the containment
filters in `find_files` and `search_text` were only ever exercised against symlinked
*directories*, which `rglob` declines to descend into anyway, and the self-nesting check in
`_resolve_relocation` was only exercised in the form that fails harmlessly. Each now has a
test that fails when the guard is deleted, which is the property that actually matters.

| Module | Covers |
| --- | --- |
| `test_config.py` | Valid load; defaults; missing file; malformed YAML; unknown key; missing `api_key_env` var; unknown tool name in `enabled`; `root_path` resolved relative to the config file; root auto-created. |
| `test_sandbox.py` | The ten path cases from §4; `relative()` output; root auto-resolution. |
| `test_registry.py` | Register/get; duplicate name; unknown name; `to_schemas()` shape matches the OpenAI contract; `invoke` turns bad arguments into `ToolError`; `build_registry` honours `enabled` and its order; a name-matching **lint** against destructive tool names (documented as a tripwire, not the guarantee); and the real check — **`move` and `copy`, invoked through the wired registry, refuse to overwrite**. |
| `test_tools_navigate.py` | Listing order and `/` markers; empty directory; missing path; file-not-directory; glob matching and the 200-hit cap; `search_text` formatting, regex support, invalid regex, `file_pattern` and `path` narrowing, **skipping binary and oversized files**, the skipped count, long-line trimming, the 100-hit cap; escape attempts on all three tools, including that a symlink out of the root leaks neither a filename nor a line of content. |
| `test_tools_read.py` | Content and line numbering; `start_line`/`max_lines` windowing; past-EOF; missing file; directory; oversize file; binary file; escape attempt. |
| `test_tools_write.py` | Create nested directory; idempotent re-create; file-in-the-way; write creates parents, overwrites, refuses a directory; append preserves prior content, creates when missing, refuses a directory; edit replaces exactly once, refuses missing/ambiguous `old_text`; `move` renames, moves directories with contents, **round-trips**, and refuses an existing destination, a missing source, and a directory into itself; `copy` duplicates files and trees and **leaves the source unchanged**; escape attempts on every tool and on **both** ends of `move`/`copy`; `OSError` becomes a `ToolError` everywhere. |
| `test_llm.py` | Request URL, auth header, and payload shape; `tool_choice: auto`; parsing a plain answer, one tool call, and batched calls; arguments sent as an object; missing call id; malformed arguments JSON, non-JSON body, HTTP error, connection failure ⇒ `LLMError`; `to_history_entry` round-trips through the parser. |
| `test_agent.py` | Answer with no tool call; one tool call then answer; multi-step; batched tool calls in one message; unknown tool, `ToolError`, escape attempt, and bad arguments each ⇒ observation and the loop continues; `LLMError` propagates; `max_steps` exhaustion reports `answer=None`; history shape (`tool` messages carry the matching `tool_call_id`); `on_step` fires once per step; the loop prints nothing. |
| `test_cli.py` | Arg parsing and precedence; one-shot run with a fake client; quiet vs `--verbose` output; failed calls marked; long arguments truncated on screen; `--root`/`--max-steps` overrides; REPL exit, Ctrl-D, blank lines, error recovery, per-task conversation isolation, Ctrl-C mid-run; `HarnessError` ⇒ exit code 1, one line, no traceback; `python -m mini_agent` runs as a subprocess. |

## 10. Build order

Bottom-up, so every layer is tested before anything depends on it.

- [x] **Step 0 — Scaffolding.** `pyproject.toml`, package skeleton, `config.example.yaml`,
      README, gitignore additions.
- [x] **Step 1 — Foundations.** `errors.py`, `config.py`, `sandbox.py` + their tests.
- [x] **Step 2 — Tools.** `registry.py`, `tools/*` + their tests. Fully usable and testable
      with no LLM involved.
- [x] **Step 3 — Model access.** `llm.py` + tests against a monkeypatched `requests.post`.
- [x] **Step 4 — The loop.** `agent.py` + tests driven by `FakeLLMClient`.
- [x] **Step 5 — Interface.** `cli.py`, `__main__.py` + tests.
- [x] **Step 6 — Polish.** Docs reconciled with the code; full suite green; verified
      end-to-end over real HTTP against a stub OpenAI-compatible server (the agent listed a
      workspace, wrote a file, and was refused a path outside the root).

## 11. Explicitly out of scope

Conversational memory of any kind — **between turns of one REPL session as well as between
CLI invocations**. `Agent.run` builds its `messages` list locally, so every task starts from
the system prompt. Each run is therefore independently bounded, and the workspace covers most
of what continuity would buy: a file written by one task is still there for the next. Making
it stateful is listed as a reader exercise, because doing so immediately raises context-window
management, which is a much larger subject than this project wants to open.

Also out of scope: streaming responses, retries/backoff, token
accounting, parallel tool execution, sub-agents, MCP, and shell execution. Each would add
real code for little teaching value here. The natural extensions are listed in the README so
a reader knows where to take it next.

**Evals against a real provider** were deferred rather than rejected, and were **built in
0.2.0** as `evals/run_eval.py`. They are the one thing that can measure prompt changes or
surface failure modes a script cannot invent, but they cost money, cannot be deterministic,
and stay out of the test suite. `TESTING.md` has the reasoning and how to run them.

Also considered and declined: a larger HTTP-level test suite (connection refused, timeouts,
401 handling over a real socket). `test_llm.py` already covers all of those against a
monkeypatched `requests.post`, in a form a reader can follow in seconds. Re-testing them
over a threaded server would add daemon threads, `shutdown()`, and port binding — concepts
that teach HTTP test infrastructure rather than agent building, which is the trade this
project exists to refuse. `examples/offline_demo.py` covers the wire once, and earns its
place by being useful for something else.

## 12. What 0.2.0 changed, and what it deliberately did not

0.2.0 came out of an evaluation of 0.1.0 against a specific question: can a developer pick
this up, understand agent foundations, and extend it for their own purposes? The reading part
held up. The extending part had three gaps, and the release closes them — a config trap that
silently dropped newly added tools, no walkthrough for the extension seam, and no way to see
the conversation a run actually produced. `EXTENDING.md` is the new front door for all three.

**§11 above was upheld in full.** Nothing in it was reopened: no token accounting, no
retries, no streaming, no conversational memory. Two additions sit near the line and are
worth naming, because both look at first glance like §11 violations and neither is:

- **`AgentResult.messages` is not conversational memory.** The list is still built inside
  `run` and handed out afterwards; the next run still starts from the system prompt. What
  changed is that a finished run can be *read*, not that anything carries over. Every
  bound §11 relies on is intact, and `test_exposing_the_transcript_did_not_make_the_loop_stateful`
  exists to keep it that way.
- **"Make the REPL remember previous turns" stays an exercise.** It is the most valuable
  exercise in the project, precisely because doing it forces the reader to meet context
  growth and decide what to do when the window fills. Shipping it would take that away.

The evals were the one deferred item that became real; see §11 and `TESTING.md`.
