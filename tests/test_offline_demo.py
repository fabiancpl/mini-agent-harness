"""Runs `examples/offline_demo.py` as a real subprocess.

This is the project's only wire-level test: the demo starts an HTTP server on a real socket
and drives the real CLI against it, so JSON serialisation, the auth header, URL composition,
and response parsing are all exercised for real rather than monkeypatched.

It is one test rather than a suite. `test_llm.py` already covers the HTTP failure modes in a
form that is far easier to read, so the only thing left to prove here is that the whole stack
fits together over a socket -- and that the example a reader is told to run still works, which
is what stops it rotting when a tool signature changes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "examples" / "offline_demo.py"


def test_the_offline_demo_runs_and_shows_both_refusals() -> None:
    result = subprocess.run(
        [sys.executable, str(DEMO)], capture_output=True, text=True, timeout=120
    )
    output = result.stdout

    assert result.returncode == 0, f"demo failed:\n{result.stdout}\n{result.stderr}"

    # The agent got through every step, over a real socket, using the real tools.
    assert "search_text(pattern='hello')" in output
    assert "Copied hello.txt to backup/hello.txt" in output
    assert "Moved hello.txt to greeting.txt" in output

    # The two refusals the demo exists to show, and the recovery after each.
    assert "Unknown tool 'delete_file'" in output
    assert "already exists. Moving onto it would destroy its contents" in output
    assert "Appended 25 characters to greeting.txt" in output  # it carried on afterwards

    # And the point of it all: nothing was lost.
    assert "draft.txt" in output
    assert "greeting.txt" in output
