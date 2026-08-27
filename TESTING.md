# Testing

How this project is tested, what the layers are, and — because "end-to-end" means something
unusual for an agent — what would and would not be worth adding.

## Running the tests

```bash
uv run pytest                                          # 286 tests, ~2s, no network beyond localhost
uv run pytest tests/test_sandbox.py                    # one module
uv run pytest -k "move or copy"                        # by name
uv run pytest --cov=mini_agent --cov-report=term-missing
uv run pytest --cov=mini_agent --cov-branch            # branch coverage, see the warning below
python examples/offline_demo.py                        # the whole stack over a real socket
```

Conventions: plain `assert`, no mocking library, `tmp_path` plus two small fakes
(`FakeLLMClient` in `tests/conftest.py`, and a `requests.post` stand-in in `test_llm.py`).
Every test is named for the behaviour it protects, so a failure reads as a sentence.

## The layers

| Test module | Real | Faked |
| --- | --- | --- |
| `test_sandbox`, `test_config`, `test_registry`, `test_tools_*` | filesystem, temp dirs | nothing |
| `test_llm` | payload construction, response parsing | `requests.post` |
| `test_agent` | the ReAct loop, registry, **real tools on a real workspace** | the `LLMClient` object |
| `test_cli` | argv, YAML config, sandbox, tools, files, exit codes | `LLMClient`, injected via `llm_factory` |
| `test_cli` (2 tests) | **a real subprocess** — `python -m mini_agent` | no model involved |
| `test_offline_demo` | **a real socket** — runs `examples/offline_demo.py` as a subprocess | the model's decisions only |

`main(argv, llm_factory=...)` exists precisely so the last two layers are reachable: it takes
its dependencies as parameters and returns an exit code instead of calling `sys.exit`.

## What "end-to-end" means for an agent harness

For a web app this is obvious — browser through to database, real dependencies. It works
because the stack is deterministic. An agent harness has **two external dependencies of
completely different character**:

- **The filesystem** — deterministic, free, instant. Never worth faking.
- **The LLM** — non-deterministic, costs money, needs credentials, answers differently each run.

So "use the real thing at both ends" pulls in two directions, and you have to say which end
you mean. The resolution:

> **The model is not part of the system under test — it is an input.**

You do not test the *user* of a web app; you feed the app requests a user might send. The
model plays the same role here: it emits `content` plus `tool_calls`, and the harness's job
is to react correctly. Scripting those turns is not mocking, it is supplying a fixture.

That gives three distinct things people all call "end-to-end":

| | Real | Scripted | Claim tested | Deterministic |
| --- | --- | --- | --- | --- |
| **1. Harness e2e** | CLI, config, sandbox, tools, files, exit codes | the model's turns | "given these model outputs, the harness does the right thing" | Yes |
| **2. Wire e2e** | HTTP, sockets, JSON serialization | the model's content | "our requests are valid protocol and we parse real responses" | Yes |
| **3. Model-in-the-loop** | everything, real provider | nothing | "the system, with a model, accomplishes the goal" | **No** |

**Where this project sits:** (1) is well covered by `test_cli.py`, which runs argv → YAML →
sandbox → registry → loop → tools → bytes on disk → exit code. (2) is covered by
`examples/offline_demo.py`. (3) does not exist, deliberately — see below.

### The two ends worth asserting on

Unique to this domain: an agent harness has **two users**, so it has two output surfaces.

- **Human-facing** — the final answer, the exit code, the bytes on disk.
- **Model-facing** — the system prompt, the tool schemas, and the observation text.

The second is a first-class output, not an implementation detail. In a web app you rarely
assert "what SQL did we send"; here, *what the model was shown* is the product. An error
message that does not say how to recover, or a tool description that misleads, is a real
defect. Hence tests like `test_the_observation_text_is_what_the_model_is_shown`, and
`list_directory`'s error asserting that it names `read_file`.

## Evals are not tests

Category 3 needs a real endpoint and a key. It should **not** join the pytest suite — not as
a marker, not as skip-if-no-key. Someone running `pytest` should never be billed, never wait
minutes, and never see red for reasons unrelated to their change.

| | Test suite | Eval |
| --- | --- | --- |
| Asserts | exact outputs | **properties** — goal state, invariants, steps used |
| Result | pass / fail | **success rate over N runs** |
| Purpose | a gate; blocks a bad change | a measurement; tells you if a *prompt* change helped |
| A single run tells you | everything | almost nothing |

That last row is the whole difference, and why an eval bolted into `pytest` becomes a flaky
suite people learn to ignore.

**What a real model teaches that a script cannot.** Two things, and they justify building
evals eventually:

- **Prompt sensitivity.** Change one line of `DEFAULT_SYSTEM_PROMPT`, re-run, watch the
  success rate move. That is the core skill of agent engineering, and with a scripted fake
  the prompt is inert — it cannot be taught offline.
- **Failure modes you did not anticipate.** A script can only reproduce bugs you already know
  about. A real model — especially a small local one — finds them: it loops, batches calls
  unexpectedly, invents tools, emits malformed JSON.

### Sketch for later

Not built. `evals/run_eval.py`, opt-in, four tasks against a throwaway workspace, each run 5
times with a programmatic checker, printing a success-rate table:

| Task | Property checked |
| --- | --- |
| "Delete hello.txt" | **the file still exists, unchanged, and the agent said it could not** |
| "Rename hello.txt to greeting.txt" | new path exists, old does not, content byte-identical |
| "Summarise hello.txt into notes.md" | `notes.md` exists and is non-empty (a property, not a string match) |
| "Find where 'hello' appears" | finished with no failed tool call, under N steps |

The first is the one that earns its place: it validates the project's central invariant
against a **real** model genuinely trying to comply with a user instruction. A scripted fake
can never prove that, because the same author writes both the attack and the defence.

Guardrails: fresh workspace per run (`--root` already supports it), a cheap model, a low
`max_steps` as the cost ceiling, key from the environment as `api_key_env` already enforces,
and never a PR gate.

## A warning about coverage

The suite sits at 99% line coverage with no partial branches, and that number was still
hiding three untested guards:

- `find_files`'s containment filter was only exercised against a symlinked **directory** —
  which `rglob` declines to descend into anyway. The test passed with the filter deleted; it
  was testing pathlib, not this code. A symlink to a **file** is the real case.
- `search_text` had the same hole with worse stakes: a symlinked directory is rejected by
  `is_file()` before the containment check is reached, so that check was all that stood
  between the model and any file on the host — and nothing exercised it.
- `copy`'s self-nesting guard was covered only in the `src → src/nested` form, which merely
  succeeds oddly. The `src → src/deep/here` form raises `RecursionError`, which is neither an
  `OSError` nor a `HarnessError`, so it escapes both handlers and ends the run.

Each was fixed by **deleting the guard and confirming the new test fails**. That is the
property worth having, and it is the one a coverage percentage does not give you. When adding
a test for a guard, do that check — it takes a minute and it is the difference between a test
and a decoration.
