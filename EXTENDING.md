# Extending the harness

`README.md` explains what this project does. This file explains how to make it do something
else — how to add a tool, how to swap the model client, and where the lines are that you
should not cross without knowing you are crossing them.

Read `src/mini_agent/agent.py` first if you have not. It is 175 lines and it is the whole
loop; everything below assumes you have seen it.

## The shape of a tool

A tool is four things bundled in a frozen dataclass (`src/mini_agent/registry.py`):

| Field | What it is |
| --- | --- |
| `name` | the function name the model emits — snake case, verb first |
| `description` | written **for the model**: what it does, when to use it, its limits |
| `parameters` | JSON Schema, passed to the API verbatim |
| `handler` | takes the arguments, returns the observation string the model reads |

Two of those deserve more than a table row.

**The handler returns a string, and that string is what the model literally sees next.** It
is not a log line and not a return value someone will parse — it is the next thing in the
conversation. Write it the way you would write a note to a colleague who cannot see your
screen: short, factual, no ANSI colour, no emoji. `"Created notes.md (24 characters, 2
lines)"` is a good observation. `"OK"` is not, because the model cannot tell what happened.

**The description is prompt engineering, not documentation.** It is the only thing standing
between the model and calling your tool at the wrong moment. Say when *not* to use it.

## Adding a tool, end to end

The worked example is real code: [`examples/word_count_tool.py`](examples/word_count_tool.py),
tested by [`tests/test_word_count_example.py`](tests/test_word_count_example.py). It is
deliberately **not** wired into the registry — step 4 is yours to do. Read the two files
alongside this section.

### 1. Write the handler

```python
def word_count(sandbox: Sandbox, path: str) -> str:
    target = sandbox.resolve(path)          # <- always first, no exceptions

    if not target.exists():
        raise ToolError(f"No such file: {path!r}. Use list_directory to see what is there.")
    if target.is_dir():
        raise ToolError(f"{path!r} is a directory, not a file. Use list_directory instead.")

    text = load_text(target, path)          # shared size + UTF-8 guards
    return f"{sandbox.relative(target)}: {plural(len(text.split()), 'word')}, …"
```

Four rules, in order of how much trouble breaking them causes:

1. **`sandbox.resolve()` before any filesystem call.** The path came from a model. It means
   nothing until the sandbox has proved it lands inside the root. Never `open()`,
   `Path.mkdir()`, or `os.listdir()` a raw argument — that is invariant 1 in `CLAUDE.md`, and
   it is the one that makes the other guarantees worth anything.
2. **Take the `Sandbox` as the first positional argument.** `make_tools` binds it with
   `functools.partial`, which leaves a callable whose remaining signature is exactly your
   JSON schema. That correspondence is what lets `ToolRegistry.invoke` bind the model's
   arguments and reject bad ones *before* your code runs.
3. **Raise `ToolError` for anything the model could fix.** The loop turns it into an
   observation and the model gets another step. Say what went wrong *and* what to do instead.
4. **Reuse `tools/common.py`.** `load_text` gives you the 1 MiB cap and the UTF-8 check;
   `plural` keeps observations readable. Every tool that reads a file should fail the same
   way for the same reasons.

Return paths through `sandbox.relative(target)`, never the absolute path — observations must
not leak the host's layout.

### 2. Describe it to the model

```python
Tool(
    name="word_count",
    description=(
        "Count the words, lines, and characters in a UTF-8 text file. Use this when you "
        "need the size of a file's contents but not the contents themselves; it is much "
        "cheaper than reading the whole file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to measure, relative to the root."},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    handler=partial(word_count, sandbox),
)
```

Keep `additionalProperties: False`. It is what stops a model inventing an argument and
getting a confusing failure three steps later.

### 3. Put it in a `make_tools`

Every module in `src/mini_agent/tools/` exports `make_tools(sandbox) -> list[Tool]`. Add
yours to the module where it belongs — `navigate.py` finds things, `read.py` reads one,
`write.py` changes them — or start a new module if it is genuinely a new category.

### 4. Wire it up

`build_registry` in `src/mini_agent/tools/__init__.py` is the **single** place tools become
real. There is no auto-discovery and no plugin scan, on purpose: to know exactly what an
agent can do, you read one function.

```python
    all_tools = [
        *navigate.make_tools(sandbox),
        *read.make_tools(sandbox),
        *write.make_tools(sandbox),
        *word_count_tool.make_tools(sandbox),   # <- your line
    ]
```

**Then check `config.yaml`.** If it has a `tools.enabled:` list, that list is a strict
allow-list and your new tool is not in it — so it will not load, and nothing will say so.
`config.example.yaml` ships with the list commented out precisely so this does not bite you,
but a config you wrote earlier may still have one. The startup banner prints
`tools: 10 of 11 registered` whenever a list is trimming the registry; that line is your
clue.

### 5. Test it

`CLAUDE.md` asks for three cases, and `tests/test_word_count_example.py` is a copyable
template for all three:

- **the happy path** — the observation string is exactly what you expect;
- **an input error** — a missing file or a directory raises `ToolError`;
- **an escape attempt** — `../secret.txt` and a symlink pointing outside both raise
  `PathOutsideRootError`.

Then do the thing `TESTING.md` insists on: **delete the guard and confirm your test fails.**
Three guards in this repo sat at 100% coverage while the case they existed for never ran. A
test you have not seen fail is a decoration.

## Swapping the model client

`Agent` takes anything satisfying `SupportsComplete` (`src/mini_agent/llm.py`) — one method:

```python
def complete(self, messages, tools=None) -> Message: ...
```

Return a `Message` with `tool_calls` empty to end the run, or populated to keep it going.
That is the entire contract; the test suite has always driven the loop with a `FakeLLMClient`
that is not an `LLMClient` at all.

**The honest caveat.** `messages` are OpenAI-shaped dicts, and `Agent.run` builds them itself
— it appends `{"role": "tool", "tool_call_id": …, "content": …}` inline. So a provider whose
API has a different message shape (Anthropic's native API, for instance) needs edits to the
loop as well as a new client. Writing a client that translates in both directions is the
cleaner fix and a genuinely instructive exercise.

Two smaller notes. `Message.usage` is optional — return `None` and the CLI reports message and
character counts instead of tokens, which is exactly what happens against servers that do not
send a `usage` block. And retries live in `LLMClient`, not in the loop, so a client of your own
owns its own retry policy; `Agent` will let anything your `complete` raises escape the run.

## Reading what actually happened

Two things worth knowing before you debug an agent:

```bash
python -m mini_agent --task "…" --verbose            # reasoning + observations, live
python -m mini_agent --task "…" --dump-transcript t.json
```

`--dump-transcript` writes the exact conversation the API saw: the system prompt, every
assistant turn with its tool calls, every tool result. From code it is `result.messages`.
Reading one is the fastest way to stop thinking of an agent as magic — it is a list of
dictionaries and a `for` loop.

In the REPL it is the whole *session*, not one turn, since 0.3.0 made turns share a
conversation. The file is rewritten after each turn and each rewrite is complete, so the last
one is everything that happened.

## Measuring a prompt change

Editing `DEFAULT_SYSTEM_PROMPT` and eyeballing one run tells you nothing; models are
stochastic. `evals/run_eval.py` runs each task N times against a real endpoint and prints a
success rate, and can write that rate down so the next run has something to compare against:

```bash
python evals/run_eval.py --config config.yaml --runs 5
python evals/run_eval.py --save-baseline          # before the change
python evals/run_eval.py --compare                # after it, side by side
```

A baseline records the model, endpoint, step ceiling and enabled tools alongside the numbers,
and `--compare` refuses to imply a comparison when any of those moved — change the model and
the rates are measuring two different experiments.

It is not a test and never joins `pytest` — see `TESTING.md` for why that distinction
matters. Adding a task is a `(prompt, seed, checker)` triple in `TASKS`; the checker returns
`None` when the property held or a sentence explaining what went wrong. Set `follow_up` to ask
a second question in the same conversation, which is how the `followup` task measures whether
memory actually works. **Unit-test your
checker** in `tests/test_eval_checkers.py`: one that can never fail reports a cheerful 5/5
forever and nothing notices.

## The lines you should not cross

Extending is unrestricted with one exception, and it is the point of the project:

> **No operation may make existing content unreachable.**

That is a claim about *effects*, not names. `write_file` overwrites, so content loss is
already possible; what must never happen is content becoming *unrecoverable*. `move` and
`copy` satisfy it by refusing an existing destination, which keeps every relocation
invertible. Deletion has no such precondition, so there is no delete, remove, unlink, or
shell tool anywhere — and `test_no_obviously_destructive_tool_is_registered` is a
name-matching lint against accidental additions, not the guarantee itself. The guarantee is
that `build_registry` is one explicit wiring point you have to edit on purpose.

If you want the agent to be able to clear things away, the interesting move is a **`trash`
tool** that moves into `.trash/` instead of unlinking, with the search tools taught to ignore
that folder. The agent gets the *affordance* of removal without the power of destruction, and
every "deletion" stays invertible. That is the exercise this design is pointing at.

The other invariants are in `CLAUDE.md`. Each has a dedicated test. If you change one, update
its test deliberately — never "fix" it to pass.
