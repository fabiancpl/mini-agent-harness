"""The ReAct loop.

ReAct (Yao et al., 2022, "ReAct: Synergizing Reasoning and Acting in Language Models") is
the idea that an LLM solves multi-step problems better when it alternates between reasoning
about what to do and taking an action whose result it can observe:

    Thought -> Action -> Observation -> Thought -> ... -> Answer

`Agent.run` below is that sentence as code, and it is the shortest interesting file in the
project. The loop ends when the model answers without calling a tool, or when it runs out of
steps -- and in the second case it says so rather than inventing an answer.

*On the shape of it.* The original paper had the model emit `Thought:` / `Action:` lines as
plain text, which the harness scraped with a regular expression. Every model API now exposes
structured tool calling instead, so that is what we use: the model returns a parsed
`tool_calls` list and no parsing can go wrong. We keep the reasoning half of ReAct by asking
the model, in the system prompt, to say what it is doing in `content` before each call --
so every step still has a visible Thought, which is what makes an agent debuggable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import HarnessError
from .llm import SupportsComplete, ToolCall, Usage
from .registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """\
You are a careful file-system assistant working inside a single workspace folder.

How to work:
- Before each tool call, briefly say what you are doing and why. One or two sentences.
- Explore before you act: list or find files rather than guessing at names.
- Read a file before editing it, so the text you replace matches it exactly.
- When the task is done, reply with a short plain-text summary and no tool call. That is
  how you finish.

What you can do:
- Every path you give is relative to the workspace root, for example 'notes.txt' or
  'src/main.py'. Paths outside the workspace are refused.
- You can create folders and files, and modify existing files.
- You cannot delete or rename anything, and you cannot run commands. If a task needs that,
  say so plainly instead of trying to work around it.
- If a tool returns an error, read it: it usually says exactly how to fix the call.\
"""


@dataclass(frozen=True)
class Observation:
    """The result of one tool call, as the model will read it."""

    call: ToolCall
    content: str
    failed: bool = False  # True when the tool refused; the model gets to try again


@dataclass(frozen=True)
class Step:
    """One turn of the loop: what the model thought, and what happened when it acted."""

    number: int
    thought: str | None
    observations: tuple[Observation, ...] = ()


@dataclass(frozen=True)
class AgentResult:
    """Everything one `run` produced. The CLI prints it; the tests assert on it."""

    answer: str | None  # None if the loop hit its step limit
    steps: tuple[Step, ...] = ()
    stopped_early: bool = False
    #: The exact conversation sent to the model, in wire format: the system prompt, every
    #: task, every assistant turn with its tool calls, and every tool result. `steps` is
    #: the readable summary of a run; this is the literal thing the API saw, which is what
    #: you want when a run went wrong or when you are learning what an agent really is.
    #:
    #: Since 0.3.0 this is the whole *session*, not one run: the second run in a session
    #: returns its own turn and the first one. The dicts are the same objects the agent is
    #: still using, so read them and do not edit them.
    messages: tuple[dict[str, Any], ...] = ()
    #: What the *last* request cost, as the server counted it, or None if it did not say.
    #:
    #: The last one is the interesting one: `total_tokens` is prompt plus completion, and the
    #: completion has since been appended to the conversation. So it is the floor for what the
    #: next request will cost -- the number that predicts, rather than the one that reminisces.
    usage: Usage | None = None

    @property
    def tool_calls_made(self) -> int:
        return sum(len(step.observations) for step in self.steps)


class Agent:
    """Runs one task to completion, or to the step limit."""

    def __init__(
        self,
        llm: SupportsComplete,
        registry: ToolRegistry,
        *,
        max_steps: int = 12,
        system_prompt: str | None = None,
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        # Called as each step finishes, so the CLI can show progress live. The loop itself
        # never prints: keeping I/O out of it is what makes it straightforward to test.
        self.on_step = on_step
        # The conversation so far. `run` works on a *copy* and assigns back only when it
        # returns, so a turn that raises leaves this exactly as it was -- see `run`.
        self.messages: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        """Forget the conversation and start again from the system prompt.

        The system prompt is read from the attribute rather than a stashed copy, so setting
        `agent.system_prompt` and calling this is a working way to try a different prompt.
        """
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def run(self, task: str) -> AgentResult:
        """Think, act, observe, repeat.

        **Runs share one conversation.** The task is appended to `self.messages`, so asking a
        second question in the same session continues the first: "do that again, but in
        French" has something to refer back to. `reset()` starts over.

        The interesting part is the copy on the next line. The loop works on its own list and
        assigns it back only on the two paths that *return*, so a turn that raises -- a
        dropped connection, Ctrl-C, a bug -- leaves `self.messages` exactly as it was. That
        matters more than it looks: the loop appends an assistant message with tool calls and
        then the results answering them, and a history left holding the first without the
        second is malformed. Every later request would be rejected, for the rest of the
        session, because of one blip. Committing only on success makes that unreachable.

        A shallow copy is the right one: no message dict is ever mutated after it is
        appended, so the copy only has to stop the *list* being shared. Deep-copying would
        imply mutation is expected somewhere, which would be a lie about how this works. It
        does mean the dicts in a returned `AgentResult.messages` are the same objects as the
        ones here -- read them, do not edit them.

        One honest asymmetry, because it will look like a bug otherwise. `on_step` fires as
        each step finishes, so a turn that fails midway has already *printed* steps the model
        will not remember. The files those steps wrote are still on disk too. Environmental
        memory and conversational memory diverge under failure, and the workspace is the one
        telling the truth about what happened.

        Nothing here bounds the conversation. `max_steps` bounds a single run; a session grows
        until the window fills or someone calls `reset`.
        """
        # See the docstring: work on a copy, commit only when we return.
        messages: list[dict[str, Any]] = list(self.messages)
        messages.append({"role": "user", "content": task})
        steps: list[Step] = []

        for number in range(1, self.max_steps + 1):
            message = self.llm.complete(messages, self.registry.to_schemas())

            # No tool calls means the model is answering: that is the exit condition.
            if not message.tool_calls:
                step = Step(number=number, thought=message.content)
                steps.append(step)
                self._announce(step)
                # Record the answer in the history too. Nothing is sent after this, but a
                # transcript that stopped just before the answer would be a strange thing
                # to hand someone trying to read the conversation.
                messages.append(message.to_history_entry())
                self.messages = messages  # commit: this turn finished cleanly
                return AgentResult(
                    answer=message.content or "",
                    steps=tuple(steps),
                    messages=tuple(messages),
                    usage=message.usage,
                )

            # The model must see its own tool calls before the results that answer them.
            messages.append(message.to_history_entry())

            observations = []
            for call in message.tool_calls:  # a model may batch several calls in one turn
                observation = self._observe(call)
                observations.append(observation)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": observation.content}
                )

            step = Step(number=number, thought=message.content, observations=tuple(observations))
            steps.append(step)
            self._announce(step)

        # Out of steps. Reporting that honestly matters: a harness that returned its last
        # partial thought as "the answer" would look like it succeeded when it did not.
        #
        # This commits too, and the history is sound: each iteration appends the assistant
        # turn and all of its tool results together, so running out of steps always leaves a
        # complete exchange. The next turn continues from an unfinished task, which is
        # usually what you want -- "keep going" is the obvious follow-up. `reset` is there
        # for when it is not.
        self.messages = messages
        return AgentResult(
            answer=None,
            steps=tuple(steps),
            stopped_early=True,
            messages=tuple(messages),
            usage=message.usage,
        )

    def _observe(self, call: ToolCall) -> Observation:
        """Run one tool call, turning any expected failure into readable text.

        Every `HarnessError` -- an unknown tool, a bad argument, a missing file, a path
        outside the sandbox -- becomes an observation instead of ending the run. That is
        deliberate: the model reads the error and usually fixes it on the next step, and
        recovering from its own mistakes is most of what makes an agent feel capable.
        Anything that is *not* a HarnessError is a real bug in this code, so it propagates.
        """
        try:
            return Observation(call=call, content=self.registry.invoke(call.name, call.arguments))
        except HarnessError as exc:
            return Observation(call=call, content=f"Error: {exc}", failed=True)

    def _announce(self, step: Step) -> None:
        if self.on_step is not None:
            self.on_step(step)
