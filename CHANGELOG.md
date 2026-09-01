# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

**Released entries are never edited again** -- not to note that a limitation was later fixed,
not to soften a judgement that turned out wrong. Each one is the record of what shipped and
what was believed at the time, and a record whose past moves is not a record. What changed is
described in the entry for the release that changed it; what is true *now* lives in
`README.md`, and what is in or out of scope lives in `PLAN.md`. Correcting a factual error --
a wrong date, a miscounted figure -- is a different thing and is fine.

## [Unreleased]

## [0.3.0] — 2026-09-01

The "session" release. 0.1.0 was built to be read and 0.2.0 to be forked; both shipped with
the same hole, and it is the first one anybody hits. You type a second thing into the REPL and
the agent has forgotten the first.

`PLAN.md` §11 called that out of scope and pointed at the workspace instead — a file written by
one task is still there for the next — which is true, and is not the same thing. It covers
*"add a section to notes.md"*. It cannot cover *"do that again, but in French"*, because there
is no "that".

**Why the exercise argument did not survive.** 0.2.0 argued that shipping memory would take
away the most valuable exercise in the project. The mistake was in what "it" meant. Holding
`messages` on the instance is three lines, and the README already spelled them out; nobody
learns anything by typing them. The valuable half is the *second* half — what do you do when
the window fills? So this release ships the three lines and keeps the question. The harness
now tells you exactly how full the context is and gives you `reset`. It will never drop or
summarise a message on your behalf, and compaction is still yours to design — against a
session that now exists, which makes it a better exercise than it was.

Retries and token accounting are here for the same reason, not as a grab bag. Memory changes
what a failure costs, and it removes the bound that made accounting pointless.

Still no new agent capability: ten tools, no delete, one sandboxed root.

### Added

- **Conversation memory within a session.** The REPL's turns share one conversation, so
  follow-ups that refer to earlier ones work. `reset` forgets it and starts from the system
  prompt; the banner says so. `Agent.reset()` is the API.
- **Context reporting.** Every turn prints what the conversation costs — `context: 4,210 of
  32,768 tokens (13%)` — and warns past 75%, pointing at `reset`. `Message.usage` and
  `AgentResult.usage` expose the server's own count. New optional `llm.context_window`.
- **Retries with backoff** (`llm.max_attempts`, default 3). Connection failures, timeouts,
  429 and the transient 5xx are retried 1s then 2s apart; `Retry-After` is honoured when it
  parses, capped at 30s. A bad key or a malformed URL still fails on the first try.
- **Eval baselines.** `--save-baseline` / `--compare` write and diff `evals/baselines/*.json`,
  recording model, endpoint, commit, run count, step ceiling and enabled tools, and refusing
  to imply a comparison when those differ. A `followup` eval task measures the memory feature
  itself, which a scripted fake cannot.

### Changed

- **`AgentResult.messages` is now the whole session**, not one run. `--dump-transcript` gets
  better for free: the file it overwrites each turn is a complete transcript rather than only
  the last exchange.
- **`max_steps` is per-task, and that distinction now matters.** It bounds a run; nothing
  bounds a session except `reset`. Noted in `CLAUDE.md` invariant 5.

### Fixed

- **`--dump-transcript` creates the directory it is asked to write into.** `--dump-transcript
  dumps/run.json` failed with `No such file or directory` unless `dumps/` already existed,
  which is the first thing anyone types to keep transcripts out of the way. Only missing
  parents are created; a dump that still cannot be written is reported and the run's answer
  is kept, as before.
- **Synthesized tool-call ids no longer collide.** When a server omits an id, the client made
  one from the tool name alone, so two calls to the same tool in one turn shared it and their
  results could not be told apart. Now `call_0_read_file`. Latent before this release, since
  the ambiguity died with the run; permanent once it is written into a session's history.
- **`__version__` no longer drifts from `pyproject.toml`.** It said 0.1.0 through the whole of
  0.2.0, because the only test asserted the string was truthy. It is now compared against
  `pyproject.toml`. `uv.lock` had the same problem and is regenerated.
- **`dumps/` is gitignored**, so following the transcript documentation no longer dirties the
  working tree.

### Upheld

Still out of scope, and §11 now says why: **persistence** of any kind — no session files, no
`--resume`, no named sessions; a conversation dies with the process. And **automatic context
management** — no trimming, no summarisation, no compaction. Nothing leaves a conversation
unless you say so. Also unchanged: no streaming, no parallel tool execution, no sub-agents, no
MCP, no shell execution.

One alternative was considered and rejected: `run(task, history=...)`, with the REPL owning the
list. It makes commit-on-success automatic, since an exception never reaches the caller's
assignment. But `Agent.reset()` is a better API than a parameter every caller must thread, and
a `history` argument invites callers to hand in a malformed list — precisely the thing the loop
works hard to make unreachable.

### Known limitations

- **A session grows without bound.** That is the deliberate half of this release: the harness
  measures it and warns, and the decision stays yours. A very long conversation will
  eventually be refused by the provider, and `reset` is the answer.
- Conversation memory does not survive the process, so `--task` invocations are still
  independent of each other.
- Unchanged from 0.2.0: the OpenAI message shape is still built inline in `Agent.run`, so a
  provider with a different shape needs edits to the loop and not just a new client.

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

## [0.1.0] — 2026-08-27

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

[0.3.0]: https://github.com/fabiancpl/mini-agent-harness/releases/tag/v0.3.0
[0.2.0]: https://github.com/fabiancpl/mini-agent-harness/releases/tag/v0.2.0
[0.1.0]: https://github.com/fabiancpl/mini-agent-harness/releases/tag/v0.1.0
