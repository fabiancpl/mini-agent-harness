# mini-agent-harness

A small, readable **ReAct agent harness**: an LLM loop that can navigate and edit files
inside one sandboxed folder, driven from a CLI and configured from a YAML file.

It is built for teaching. The whole thing is meant to be read in an afternoon, so it favours
obvious code over clever code, and explains its design decisions in comments.

```
                    ┌──────────────────────────────────────────┐
   you ──task──▶    │  ReAct loop   think → act → observe → …   │
                    └───────┬──────────────────────┬───────────┘
                            │                      │
                     tool schemas            tool call
                            ▼                      ▼
                    ┌───────────────┐     ┌──────────────────┐
                    │  LLM endpoint │     │  tool registry   │
                    │ (OpenAI-      │     │  list · find     │
                    │  compatible)  │     │  search · read   │
                    └───────────────┘     │  create · write  │
                                          │  append · edit   │
                                          │  move · copy     │
                                          └────────┬─────────┘
                                                   │ every path
                                                   ▼
                                          ┌──────────────────┐
                                          │  Sandbox (root)  │  ← nothing escapes,
                                          └──────────────────┘    nothing deletes
```

## Status

[`PLAN.md`](PLAN.md) is the design document, [`CLAUDE.md`](CLAUDE.md) holds the invariants
and conventions; [`TESTING.md`](TESTING.md) the unit test and e2e testing strategy.
**[`EXTENDING.md`](EXTENDING.md) is where to go once you want to change something** — adding
a tool, swapping the model client, and which lines not to cross.

## Quickstart — no API key needed

```bash
uv sync --extra dev            # or: pip install -e ".[dev]"

python examples/offline_demo.py
pytest                         # the test suite; it never touches the network
```

The demo stands in for the model: it starts a tiny HTTP server on localhost that speaks the
OpenAI `/chat/completions` protocol and replays a fixed script of replies, then points the
real CLI at it. Everything except the model's *decisions* is real — a real socket, real JSON
on the wire, the real config loader, the real sandbox, real files on disk.

It is scripted to hit both safety properties: at step 4 the model invents a `delete_file`
tool that does not exist, and at step 5 it tries to overwrite a file. Both are refused, both
refusals come back as observations, and the model adapts and finishes. Nothing is lost.

See [`examples/offline_demo.py`](examples/offline_demo.py) — it is about 60 lines of scripted
conversation and is meant to be read.

## What a run looks like

```
$ python -m mini_agent --task "Summarise the workspace, then try to read /etc/passwd"
workspace: /home/you/mini-agent-harness/workspace
model:     gpt-4o-mini at https://api.openai.com/v1
tools:     list_directory, find_files, search_text, read_file, create_directory,
           write_file, append_to_file, edit_file, move, copy

  - [1] list_directory(path='.')
  - [2] write_file(path='summary.md', content='# Summary\n\nOne file: hello.txt\n')
  ! [3] read_file(path='../../etc/passwd')

I listed the workspace, wrote summary.md, and was refused access outside the root.
```

`-` is a tool call that worked, `!` one that was refused. Step 3 is the sandbox doing its
job: the refusal goes back to the model as an observation, so it adapts and finishes instead
of crashing. Add `--verbose` to see the reasoning and full observation behind each step.

## Connect a real model

Two things are required: a **config file**, and the **API key in an environment variable**.
A custom system prompt is *optional* — leave `system_prompt_file: null` and the built-in
prompt in `agent.py` is used.

**1. Create the config.** `config.yaml` is gitignored, so your endpoint and settings stay
local:

```bash
cp config.example.yaml config.yaml
```

Edit the three values that matter:

```yaml
llm:
  base_url: "https://api.openai.com/v1"   # include /v1, no trailing slash
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"           # the NAME of the variable, not the key
```

**2. Export the key.** The config stores a variable *name*, never the secret, so nothing
sensitive can end up in git:

```bash
export OPENAI_API_KEY=sk-...
```

**3. Run it.**

```bash
python -m mini_agent --task "list the files here and tell me what this workspace is"
python -m mini_agent                      # interactive REPL, one conversation per session
python -m mini_agent --verbose            # show reasoning and full observations
python -m mini_agent --root ./sandbox     # override the workspace for one run
python -m mini_agent --task "…" --dump-transcript run.json   # the raw conversation
```

`--dump-transcript` writes the exact messages the API saw — system prompt, every assistant
turn with its tool calls, every tool result. Reading one is the fastest way to stop thinking
of an agent as magic: it is a list of dictionaries and a `for` loop. From code, the same
thing is `result.messages`.

### What the agent remembers

**One REPL session is one conversation.** The second thing you type builds on the first, so
follow-ups work the way you would expect:

```
task> summarise hello.txt into notes.md
task> now do the same for draft.txt
task> what did you just do?
```

| Scope | Remembered? |
| --- | --- |
| Within one task, across ReAct steps | **Yes** — step 7 sees every thought, tool call, and observation from steps 1–6 |
| Between tasks in one REPL session | **Yes** — until you type `reset` |
| Between CLI invocations | **No** — the conversation lives in the process and dies with it |
| The workspace on disk | **Always** |

There are two different memories here and it is worth keeping them apart. **Conversational**
memory is the transcript, and it is gone when the process exits. **Environmental** memory is
the workspace, and it outlives everything — which is why a fresh `--task` invocation can still
pick up where the last one left off, by reading the files rather than by remembering them.

**Nothing is ever dropped behind your back.** A conversation grows until you end it, and the
harness tells you how big it is getting:

```
  context: 12,480 of 32,768 tokens (38%)
```

Set `llm.context_window` to get the percentage; without it you still get the token count. When
the server does not report token usage at all, you get exact message and character counts
instead — never an estimate, because on this workload (tool schemas, JSON arguments, directory
listings) a character-based guess reads low exactly when the window is filling up.

Past 75% it says so and suggests `reset`, which forgets the conversation and starts again from
the system prompt. That is the only way anything leaves the history. There is no automatic
trimming and no summarisation, on purpose — deciding what to throw away is the interesting
part, and it is left to you. See "Where to take it next".

The design note is on `Agent.run` in `src/mini_agent/agent.py`, and the part worth reading is
why the loop works on a *copy* of the conversation and commits it only when a turn finishes:
a turn that dies halfway would otherwise leave an assistant message whose tool calls have no
results, and every later request in the session would be rejected because of it.

### Providers

Anything OpenAI-compatible works. `base_url` and `model` are the only differences:

| Provider | `base_url` | Notes |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` is cheap and calls tools reliably |
| Ollama | `http://localhost:11434/v1` | Free and local. Use a tool-calling model such as `qwen2.5` or `llama3.1`. The key is unused, but export any placeholder |
| vLLM | `http://localhost:8000/v1` | Launch with `--enable-auto-tool-choice` |
| LM Studio | `http://localhost:1234/v1` | Enable the local server in the UI |
| OpenRouter | `https://openrouter.ai/api/v1` | Model names are namespaced, e.g. `openai/gpt-4o-mini` |

Small local models are the interesting stress test: they forget arguments, emit invalid JSON,
and invent tools. Every one of those paths is handled and turned back into an observation —
watching it happen with `--verbose` teaches more than any amount of reading.

### Writing your own system prompt

Optional. To override the built-in one:

```yaml
agent:
  system_prompt_file: "./prompts/my_prompt.txt"   # resolved relative to config.yaml
```

The built-in prompt is `DEFAULT_SYSTEM_PROMPT` in `src/mini_agent/agent.py` — start by
copying it. Two things in it are load-bearing:

- **"Before each tool call, briefly say what you are doing and why."** This is what keeps the
  *reasoning* half of ReAct visible. Drop it and the model still works, but every step's
  thought comes back empty and the trace becomes unreadable.
- **"When the task is done, reply with a short summary and no tool call."** This is the loop's
  exit condition. A model that never stops calling tools runs until `max_steps`.

### Tuning

| Setting | Effect |
| --- | --- |
| `agent.max_steps` | Hard bound on **one task**, and your per-task cost ceiling. 12 is generous for small tasks. It does not bound a session: turns accumulate until you `reset` |
| `agent.root_path` | The sandbox. Relative paths resolve against `config.yaml`, not your shell |
| `llm.temperature` | `0.0` keeps tool calling deterministic and debuggable |
| `llm.context_window` | Optional. Lets the CLI show how full the conversation is, as a percentage, and warn past 75% |
| `llm.max_attempts` | Total tries per request, default 3. Only transient faults are retried — a bad key or a typo'd URL fails at once. `1` turns retrying off |
| `tools.enabled` | Ships commented out, which enables every registered tool. Uncomment it for a strict allow-list — the first four alone give you a **read-only agent** that can explore and explain but never touch anything. A tool you register but leave off the list is not loaded; the banner prints `10 of 11 registered` when that is happening |

### When something goes wrong

Every expected failure prints one line beginning `error:` — no traceback, because a stack
trace would describe *this* code when the problem is in your setup.

| Message | Cause |
| --- | --- |
| `Config file not found: config.yaml` | Run `cp config.example.yaml config.yaml` |
| `Environment variable 'X' (llm.api_key_env) is unset or empty` | `export X=...` in the shell you are running from |
| `Unknown key(s) ['...'] in section 'agent'` | A typo. Unknown keys are rejected rather than silently ignored |
| `returned HTTP 401: ...` | Bad or missing key. The provider's own message follows |
| `returned HTTP 404: ...` | Usually `base_url` missing `/v1`, or a model name the provider does not know |
| `Could not reach http://...` | Nothing is listening — check your local server is running |
| `did not return JSON` | A proxy or error page came back instead of the API. The first 200 characters are shown |
| `Model sent invalid JSON arguments for X` | The model produced malformed tool arguments. Common with small local models; try a larger one or a tool-calling fine-tune |
| `Stopped after N steps without finishing` | Hit `max_steps`. Raise it, or give a smaller task. Exit code is 1 |

## What the agent can do

| Capability | Tool |
| --- | --- |
| Navigate | `list_directory`, `find_files` |
| Search contents | `search_text` (regex, grep-style) |
| Read | `read_file` |
| Create | `create_directory`, `write_file` |
| Modify | `append_to_file`, `edit_file` (exact string replacement) |
| Reorganise | `move`, `copy` |

## What it cannot do, and why that matters

The rule the whole design is built around:

> **No operation may make existing content unreachable.**

That is a claim about *effects*, not about names, which is what makes it checkable. Sorted
by how much damage each tool can do:

| Tool | Effect | Can it destroy content? |
| --- | --- | --- |
| `append_to_file` | grows a file | **No** — it never reads what is already there |
| `edit_file` | replaces one exact span | Yes, that span |
| `write_file` | replaces the whole file | Yes, the previous contents |
| `move`, `copy` | relocate or duplicate | **No** — refused when the destination exists |
| *(deletion)* | — | *no such tool exists* |

`move` looks destructive and is not, and the distinction is the interesting part:

```
move(a, b)  with b free      →  content intact at b, undone by move(b, a)
move(a, b)  with b existing  →  b's old content gone, no inverse
delete(a)                    →  a's content gone, no inverse
```

Relocation is destructive in exactly **one** case, and that case is a precondition you can
check *before* acting — so `move` and `copy` refuse an existing destination and tell the
model to pick a free path. Deletion has no such case; it is destructive by definition. So
there is no delete, remove, unlink, or shell tool anywhere, and the agent cannot be talked
into one: a function that does not exist cannot be called.

This applies to folders too. `create_directory`, `move`, and `copy` all work on them
(`copy` recursively), and `list_directory`, `find_files`, and `search_text` explore them —
but there is no `rmdir` and nothing that empties a directory. So **folders accumulate**: an
agent that creates `a/b/c/d` leaves it there forever. That is the price of the invariant,
and worth knowing before pointing one at a real project.

The other three limits:

- **It cannot leave the root folder.** Every path the model supplies is joined onto the
  sandbox root, fully resolved (which normalises `..` *and* follows symlinks), and only then
  checked to be inside the root. `../../etc/passwd`, `/etc/passwd`, and a symlink pointing
  out all fail the same way. `move` and `copy` check *both* of their paths.
- **It cannot run forever on one task.** `agent.max_steps` bounds the loop, and it says so
  when it runs out rather than inventing an answer. A *session* is not bounded: the
  conversation grows until you type `reset`, which is why the CLI reports its size after
  every turn.
- **It cannot run shell commands.** No execution tool exists.

That layering — *capability absent* beats *capability denied* — is the main lesson of the
project. See `sandbox.py` and the docstring at the top of `tools/write.py` for the details.

## Repository map

| Path | What lives there |
| --- | --- |
| `src/mini_agent/config.py` | YAML → validated frozen dataclasses |
| `src/mini_agent/sandbox.py` | The path security boundary |
| `src/mini_agent/registry.py` | `Tool` and `ToolRegistry` |
| `src/mini_agent/tools/` | The ten tool implementations, wired in `__init__.py` |
| `src/mini_agent/llm.py` | OpenAI-compatible chat client |
| `src/mini_agent/agent.py` | The ReAct loop |
| `src/mini_agent/cli.py` | Argument parsing and the REPL |
| `tests/` | One test module per source module |
| `examples/offline_demo.py` | The whole stack over a real socket, with no API key |
| `examples/word_count_tool.py` | The `EXTENDING.md` walkthrough, as real tested code |
| `evals/run_eval.py` | Opt-in measurement against a real model. Never run by `pytest` |
| `EXTENDING.md` | Adding a tool, swapping the model client, and the lines not to cross |
| `TESTING.md` | Test layers, what "end-to-end" means for an agent, why evals stay out of the suite |
| `CHANGELOG.md` | What each release contains, and the known limitations of the current one |

## Where to take it next

**[`EXTENDING.md`](EXTENDING.md) is the practical guide** — the tool contract, a worked
example you can copy, how to swap the model client, and the one invariant you must not break.
Start there.

Good exercises once you have read the code: add a confirmation prompt before writes; add a
`trash` tool that moves into `.trash/` instead of deleting, and make the search tools ignore
it (soft delete gives the agent the *affordance* of removal without the power of
destruction); **decide what to do when the context window fills** — the harness will tell you
it is filling and then do nothing about it, so this one is yours; stream the model's tokens;
give the agent a sub-agent for read-only exploration; expose the same tools over MCP.

That third one stays an exercise on purpose, and 0.3.0 made it a better one. Holding the
conversation on the `Agent` was three lines and is now done; what was never done for you is
the hard half. Drop the oldest turns and you will find you cannot drop an assistant message
without the tool results that answer it. Summarise instead and you have to decide what a
summary of a tool call even is, and pay a model call to make one. Start from `agent.messages`
and the token count the CLI already prints.

## License

MIT — see [`LICENSE`](LICENSE).
