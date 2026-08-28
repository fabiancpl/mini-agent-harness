"""The command line: argument parsing, the REPL, and printing a run.

All of this project's input and output lives here. `Agent` never prints, which is what lets
it be used as a library and tested in silence -- so every `print` in the codebase is below.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Callable
from pathlib import Path

from .agent import Agent, AgentResult, Step
from .config import Config, LLMConfig, load_config
from .errors import ConfigError, HarnessError
from .llm import LLMClient, SupportsComplete
from .sandbox import Sandbox
from .tools import build_registry

DEFAULT_CONFIG_PATH = "config.yaml"
#: Observations are trimmed on screen only. The model always sees the whole thing.
MAX_OBSERVATION_CHARS = 400
#: Long arguments (a whole file's content, say) are trimmed the same way.
MAX_ARGUMENT_CHARS = 60

EXIT_OK = 0
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="A small ReAct agent that reads and edits files inside one folder.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to the YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--task",
        help="run this one task and exit; omit it to start an interactive session",
    )
    parser.add_argument("--root", help="override agent.root_path from the config")
    parser.add_argument("--max-steps", type=int, help="override agent.max_steps from the config")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show each step's reasoning, tool arguments, and observation",
    )
    parser.add_argument(
        "--dump-transcript",
        metavar="PATH",
        help="write the exact messages exchanged with the model to PATH as JSON",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    llm_factory: Callable[[LLMConfig], SupportsComplete] = LLMClient,
) -> int:
    """Entry point. Returns a process exit code instead of calling sys.exit.

    `llm_factory` exists so the tests can drive the whole CLI with a scripted fake client
    and never touch the network. Production always uses the default. Its return type is the
    `SupportsComplete` protocol rather than `LLMClient`, which is what says out loud that
    any object with a matching `complete` will do.
    """
    args = build_parser().parse_args(argv)

    try:
        config = _apply_overrides(load_config(args.config), args)
        sandbox = Sandbox(config.agent.root_path)
        registry = build_registry(sandbox, config.enabled_tools)
        # Built a second time with no allow-list, only to count what *could* have been
        # enabled. It constructs dataclasses and touches no filesystem, and two obvious
        # lines beat threading a count back out of build_registry.
        total_registered = len(build_registry(sandbox))
    except HarnessError as exc:
        # Configuration problems get one clear line, never a traceback: a stack trace tells
        # the reader about our code when the mistake is in their file.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    agent = Agent(
        llm=llm_factory(config.llm),
        registry=registry,
        max_steps=config.agent.max_steps,
        system_prompt=config.agent.system_prompt,
        on_step=lambda step: _print_step(step, verbose=args.verbose),
    )

    _print_banner(config, registry.names(), total_registered)
    if args.task:
        return _run_once(agent, args.task, args.dump_transcript)
    return _repl(agent, args.dump_transcript)


def run() -> None:
    """Console-script wrapper: `mini-agent` on the command line."""
    sys.exit(main())


# --- the two modes --------------------------------------------------------------------------


def _run_once(agent: Agent, task: str, dump_path: str | None = None) -> int:
    try:
        result = agent.run(task)
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _print_result(result)
    _dump_transcript(result, dump_path)
    # Running out of steps is a failure: scripts calling this should be able to tell.
    return EXIT_ERROR if result.stopped_early else EXIT_OK


def _repl(agent: Agent, dump_path: str | None = None) -> int:
    # Say up front that turns are independent. It is the first thing people are surprised
    # by, and the surprise is much cheaper to prevent here than to explain afterwards.
    print("Type a task, or 'exit' to quit.")
    print("Each task starts a fresh conversation; the workspace persists between them.\n")
    while True:
        try:
            task = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D or Ctrl-C at the prompt
            print()
            return EXIT_OK

        if not task:
            continue
        if task in {"exit", "quit"}:
            return EXIT_OK

        try:
            result = agent.run(task)
            _print_result(result)
            # Each turn is a separate conversation, so each turn overwrites the file. There
            # is no combined transcript to write: there was never a combined conversation.
            _dump_transcript(result, dump_path)
        except KeyboardInterrupt:
            # Ctrl-C abandons the current run but keeps the session alive.
            print("\n(interrupted)\n")
        except HarnessError as exc:
            print(f"error: {exc}\n", file=sys.stderr)


# --- printing --------------------------------------------------------------------------------


def _print_banner(config: Config, tool_names: list[str], total_registered: int) -> None:
    print(f"workspace: {config.agent.root_path}")
    print(f"model:     {config.llm.model} at {config.llm.base_url}")
    # When tools.enabled is trimming the registry, say so. A tool you registered but forgot
    # to add to the allow-list is otherwise invisible, and that is the easiest mistake to
    # make when extending the harness.
    trimmed = len(tool_names) < total_registered
    count = f"{len(tool_names)} of {total_registered} registered — " if trimmed else ""
    print(f"tools:     {count}{', '.join(tool_names)}")
    print()


def _dump_transcript(result: AgentResult, path: str | None) -> None:
    """Write the exact messages exchanged with the model, when --dump-transcript asked.

    This is the whole conversation as the API saw it: the system prompt, every assistant
    turn with its tool calls, and every tool result. Reading one is the fastest way to
    understand what an agent actually *is*, which is why it is a flag and not a code edit.
    """
    if path is None:
        return
    try:
        destination = Path(path)
        # `--dump-transcript dumps/run.json` is the obvious way to keep transcripts out of
        # the way, so create the directory rather than failing on a path the user clearly
        # meant. Nothing is overwritten: only the missing parents are made.
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.messages, indent=2), encoding="utf-8")
    except OSError as exc:
        # A failed dump must not lose the run that produced it: the answer is already on
        # screen, so report the problem and carry on.
        print(f"error: could not write transcript to {path}: {exc.strerror}", file=sys.stderr)
        return
    print(f"(transcript written to {path})\n")


def _print_step(step: Step, *, verbose: bool) -> None:
    # The final step has no tool calls -- its thought *is* the answer, and `_print_result`
    # shows that. Printing it here too would say the same thing twice.
    if not step.observations:
        return
    if verbose:
        _print_step_verbose(step)
        return
    # Quiet mode: one line per tool call, enough to follow along without scrolling.
    for observation in step.observations:
        marker = "!" if observation.failed else "-"
        arguments = _format_arguments(observation.call.arguments)
        print(f"  {marker} [{step.number}] {observation.call.name}({arguments})")


def _print_step_verbose(step: Step) -> None:
    print(f"--- step {step.number} " + "-" * 40)
    if step.thought:
        print(step.thought)
    for observation in step.observations:
        print(f"  -> {observation.call.name}({_format_arguments(observation.call.arguments)})")
        for line in _truncate(observation.content, MAX_OBSERVATION_CHARS).splitlines():
            print(f"     {line}")
    print()


def _print_result(result: AgentResult) -> None:
    if result.stopped_early:
        steps = "step" if len(result.steps) == 1 else "steps"
        print(
            f"\nStopped after {len(result.steps)} {steps} without finishing. "
            f"Raise agent.max_steps, or give a smaller task.\n"
        )
        return
    print(f"\n{result.answer}\n")


def _format_arguments(arguments: dict) -> str:
    return ", ".join(
        f"{name}={_truncate(str(value), MAX_ARGUMENT_CHARS)!r}" for name, value in arguments.items()
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... (+{len(text) - limit} more)"


# --- overrides ----------------------------------------------------------------------------------


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Let command-line flags win over the file: CLI > config > default."""
    agent_config = config.agent

    if args.root is not None:
        # A CLI path is relative to where you are standing, unlike one in the config file,
        # which is relative to the file. Both are the least surprising rule for their source.
        root = Path(args.root).resolve()
        if root.exists() and not root.is_dir():
            raise ConfigError(f"--root exists but is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        agent_config = dataclasses.replace(agent_config, root_path=root)

    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ConfigError(f"--max-steps must be at least 1, got {args.max_steps}")
        agent_config = dataclasses.replace(agent_config, max_steps=args.max_steps)

    return dataclasses.replace(config, agent=agent_config)
