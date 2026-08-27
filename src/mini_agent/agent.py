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
from .llm import SupportsComplete, ToolCall
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
    #: The exact conversation sent to the model, in wire format: the system prompt, the
    #: task, every assistant turn with its tool calls, and every tool result. `steps` is
    #: the readable summary of a run; this is the literal thing the API saw, which is what
    #: you want when a run went wrong or when you are learning what an agent really is.
    #:
    #: Exposing it does NOT make the loop stateful -- it is built fresh inside `run` and
    #: handed out afterwards, so the next run still starts from the system prompt.
    messages: tuple[dict[str, Any], ...] = ()

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

    def run(self, task: str) -> AgentResult:
        """Think, act, observe, repeat.

        **Each run is a fresh conversation.** `messages` is local to this method, so nothing
        carries over between calls: ask the REPL two questions in a row and the second one
        starts from the system prompt again, with no memory of the first.

        That is deliberate. Every run is then independently bounded -- context cannot grow
        without limit across a long session, and one confused task cannot poison the next.
        The cost is that follow-ups phrased against the conversation ("do that again, but in
        French") have nothing to refer back to.

        The workspace is the exception, and in practice it covers most of the gap. Files
        written by one task are still on disk for the next one, so the agent can *rediscover*
        what it did even though it does not remember doing it. "Summarise hello.txt into
        notes.md" followed by "add a section to notes.md" works fine; the second run just
        reads the file.

        Making this stateful is a good exercise: hold `messages` on the instance instead of
        here, and you immediately meet the two problems every real agent has -- unbounded
        context growth, and what to do when the window fills up.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
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
                return AgentResult(
                    answer=message.content or "",
                    steps=tuple(steps),
                    messages=tuple(messages),
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
        return AgentResult(
            answer=None, steps=tuple(steps), stopped_early=True, messages=tuple(messages)
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
