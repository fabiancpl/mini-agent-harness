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
                    │  compatible)  │     │  read · create   │
                    └───────────────┘     │  write · edit    │
                                          └────────┬─────────┘
                                                   │ every path
                                                   ▼
                                          ┌──────────────────┐
                                          │  Sandbox (root)  │  ← nothing escapes,
                                          └──────────────────┘    nothing deletes
```

## Status

Complete and tested: 227 tests, 99% coverage, ~2 seconds, no network.
[`PLAN.md`](PLAN.md) is the design document; [`CLAUDE.md`](CLAUDE.md) holds the invariants
and conventions.

## Quickstart

```bash
uv sync --extra dev            # or: pip install -e ".[dev]"

cp config.example.yaml config.yaml
export OPENAI_API_KEY=sk-...   # the name of this variable is what config.yaml stores

python -m mini_agent --task "list the files here and summarise README.md"
python -m mini_agent           # interactive REPL
pytest                         # the test suite; it never touches the network
```

Any OpenAI-compatible endpoint works — point `llm.base_url` at Ollama, vLLM, or LM Studio
to run the whole thing locally for free.

## What a run looks like

```
$ python -m mini_agent --task "Summarise the workspace, then try to read /etc/passwd"
workspace: /home/you/mini-agent-harness/workspace
model:     gpt-4o-mini at https://api.openai.com/v1
tools:     list_directory, find_files, read_file, create_directory, write_file, edit_file

  - [1] list_directory(path='.')
  - [2] write_file(path='summary.md', content='# Summary\n\nOne file: hello.txt\n')
  ! [3] read_file(path='../../etc/passwd')

I listed the workspace, wrote summary.md, and was refused access outside the root.
```

`-` is a tool call that worked, `!` one that was refused. Step 3 is the sandbox doing its
job: the refusal goes back to the model as an observation, so it adapts and finishes instead
of crashing. Add `--verbose` to see the reasoning and full observation behind each step.

## What the agent can do

| Capability | Tool |
| --- | --- |
| Navigate | `list_directory`, `find_files` |
| Read | `read_file` |
| Create | `create_directory`, `write_file` |
| Modify | `edit_file` (exact string replacement) |

## What it cannot do, and why that matters

- **It cannot delete anything.** There is no delete, remove, rename, or move tool in the
  registry. The capability is absent by construction — not blocked by a rule in the prompt
  that a determined user could argue their way past.
- **It cannot leave the root folder.** Every path the model supplies is joined onto the
  sandbox root, fully resolved (which normalises `..` *and* follows symlinks), and only then
  checked to be inside the root. `../../etc/passwd`, `/etc/passwd`, and a symlink pointing
  out all fail the same way.
- **It cannot run forever.** `agent.max_steps` bounds the loop, and it says so when it runs
  out rather than inventing an answer.
- **It cannot run shell commands.** No execution tool exists.

That layering — *capability absent* beats *capability denied* — is the main lesson of the
project. See `sandbox.py` for the details.

## Repository map

| Path | What lives there |
| --- | --- |
| `src/mini_agent/config.py` | YAML → validated frozen dataclasses |
| `src/mini_agent/sandbox.py` | The path security boundary |
| `src/mini_agent/registry.py` | `Tool` and `ToolRegistry` |
| `src/mini_agent/tools/` | The six tool implementations, wired in `__init__.py` |
| `src/mini_agent/llm.py` | OpenAI-compatible chat client |
| `src/mini_agent/agent.py` | The ReAct loop |
| `src/mini_agent/cli.py` | Argument parsing and the REPL |
| `tests/` | One test module per source module |

## Where to take it next

Good exercises once you have read the code: add a `search_text` (grep) tool; add a
confirmation prompt before writes; persist the conversation between CLI sessions; stream
the model's tokens; add retries with backoff; give the agent a sub-agent for read-only
exploration; expose the same tools over MCP.

## License

MIT — see [`LICENSE`](LICENSE).
