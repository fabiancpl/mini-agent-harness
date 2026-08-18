#!/usr/bin/env python3
"""Watch the agent work, with no API key, no network access, and nothing to configure.

    python examples/offline_demo.py

Getting an API key is the biggest thing standing between reading this project and seeing
it run, and it teaches you nothing about agents. So this script stands in for the model:
it starts a tiny HTTP server on localhost that speaks the OpenAI `/chat/completions`
protocol and replays a fixed script of replies, then points the real CLI at it.

Everything except the model's *decisions* is real -- a real socket, real JSON on the wire,
the real config loader, the real sandbox, the real tools, real files on disk. That makes
this both a demo and a check that the HTTP layer works end to end.

The script is written to show the two safety properties the project is built around:

    step 4   the model invents `delete_file`      -> there is no such tool
    step 5   the model tries to overwrite a file  -> refused, nothing is lost

In both cases the refusal comes back as an observation, the model adapts, and the run
finishes normally. That recovery is most of what makes an agent feel capable.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Lets you run this straight from a fresh clone, before `uv sync` or `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_agent.cli import main  # noqa: E402  (import must follow the path setup above)

TASK = "Find the greeting, keep a backup, tidy up the file names, and note what you did."


def says(text: str, *calls: tuple[str, dict]) -> dict:
    """One scripted assistant message: some reasoning, then zero or more tool calls."""
    message: dict = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for index, (name, arguments) in enumerate(calls)
        ]
    return message


#: The model's side of the conversation, one entry per turn. The harness cannot tell these
#: from a real model's replies -- they arrive over the same socket in the same JSON.
SCRIPT = [
    says(
        "First let me find where the greeting actually lives.",
        ("search_text", {"pattern": "hello"}),
    ),
    says(
        "Before changing anything, I'll keep a copy I can fall back on.",
        ("copy", {"source": "hello.txt", "destination": "backup/hello.txt"}),
    ),
    says(
        "Now I'll give the original a clearer name.",
        ("move", {"source": "hello.txt", "destination": "greeting.txt"}),
    ),
    says(
        "The old draft is redundant now, so I'll delete it.",
        ("delete_file", {"path": "draft.txt"}),  # <- no such tool exists
    ),
    says(
        "No delete tool. I'll fold the draft into the greeting instead by moving it there.",
        ("move", {"source": "draft.txt", "destination": "greeting.txt"}),  # <- would clobber
    ),
    says(
        "Refused, because that would destroy the greeting. I'll append a note instead.",
        ("append_to_file", {"path": "greeting.txt", "content": "\n(draft kept separately)\n"}),
    ),
    says(
        "Done. I found the greeting in hello.txt, copied it to backup/hello.txt, renamed it "
        "to greeting.txt, and appended a note. I could not delete the draft -- no delete "
        "tool exists -- and I was refused when I tried to overwrite the greeting, so "
        "draft.txt is still there untouched."
    ),
]


class StubModel(BaseHTTPRequestHandler):
    """A fake OpenAI-compatible endpoint that replays SCRIPT, one entry per request."""

    turn = 0

    def do_POST(self) -> None:  # noqa: N802  (the name is fixed by http.server)
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        print(
            f"    [stub model] turn {StubModel.turn + 1}: "
            f"sent {len(request['messages'])} messages and {len(request.get('tools', []))} "
            f"tool schemas, auth {self.headers['Authorization'][:11]}..."
        )

        body = json.dumps({"choices": [{"index": 0, "message": SCRIPT[StubModel.turn]}]}).encode()
        StubModel.turn += 1

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence http.server's own request logging; the line above is friendlier."""


def seed_workspace(directory: Path) -> None:
    """Give the agent something to work on."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hello.txt").write_text("hello world\n", encoding="utf-8")
    (directory / "draft.txt").write_text("an old draft\n", encoding="utf-8")


def write_config(path: Path, base_url: str, workspace: Path) -> None:
    """Write a config.yaml pointing at the stub server. Identical in shape to a real one."""
    path.write_text(
        f"""\
llm:
  base_url: "{base_url}"
  model: "stub-model"
  api_key_env: "MINI_AGENT_DEMO_KEY"

agent:
  root_path: "{workspace}"
  max_steps: 10
""",
        encoding="utf-8",
    )


def show_tree(workspace: Path) -> None:
    print("\nThe workspace on disk afterwards:\n")
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            first_line = path.read_text(encoding="utf-8").splitlines()[:1]
            preview = f"  ->  {first_line[0]}" if first_line else ""
            print(f"    {path.relative_to(workspace)}{preview}")


def run() -> int:
    # A throwaway directory, so re-running is always a clean slate and this script never has
    # to delete anything. (It could -- it is an ordinary program. The *agent* is the thing
    # that cannot, and that difference is the point of the project.)
    scratch = Path(tempfile.mkdtemp(prefix="mini-agent-demo-"))
    workspace = scratch / "workspace"
    seed_workspace(workspace)

    # Port 0 asks the OS for any free port, so this never collides with anything you have
    # running. The daemon thread means the server dies with the script, whatever happens.
    server = HTTPServer(("127.0.0.1", 0), StubModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"

    config = scratch / "config.yaml"
    write_config(config, base_url, workspace)

    print(__doc__.split("\n\n")[0])
    print(f"\nStub model listening on {base_url}")
    print(f"Workspace: {workspace}")
    print(f"Task: {TASK!r}\n")
    print("=" * 78)

    try:
        # The real entry point -- the same function `python -m mini_agent` calls.
        exit_code = main(["--config", str(config), "--task", TASK, "--verbose"])
    finally:
        server.shutdown()

    print("=" * 78)
    show_tree(workspace)
    print(
        "\nNote what survived: draft.txt is still here, because no tool can delete it, and "
        "\ngreeting.txt kept its contents, because move refused to overwrite it."
        f"\n\nExit code: {exit_code}. Explore the workspace at:\n    {workspace}\n"
    )
    return exit_code


if __name__ == "__main__":
    # The demo asserts nothing and needs no key: the environment variable named by the
    # config just has to exist, because the harness insists on a key rather than silently
    # sending an empty one.
    import os

    os.environ.setdefault("MINI_AGENT_DEMO_KEY", "not-a-real-key")
    sys.exit(run())
