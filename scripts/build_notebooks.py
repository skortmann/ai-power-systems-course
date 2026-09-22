"""Build and execute the tutorial notebooks.

The notebooks in ``tutorials/`` are the deliverable and are committed with their
outputs. This script regenerates them from the paired jupytext sources in
``tutorials/_sources/`` (percent format), which are what a reviewer can sensibly
read a diff of.

Run it as it stands to rebuild everything::

    uv run python scripts/build_notebooks.py

To rebuild a subset or change the configuration, either edit the constants
below, or import :func:`build` and call it with keyword arguments::

    from build_notebooks import build
    build(only=("03_", "06_"))      # just those two
    build(fast=True)                # the reduced configuration CI uses
    build(do_execute=False)         # convert only, no execution

Notebook execution is CPU-bound and single-threaded at the notebook level: run
one build at a time, or the runtimes quoted in the README stop meaning anything.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

# --- configuration -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "tutorials" / "_sources"
OUTPUT_DIR = PROJECT_ROOT / "tutorials"

#: Which notebooks to build, matched as substrings of the filename
#: (e.g. ``("03_", "06_")``). An empty tuple means "every source found".
ONLY: tuple[str, ...] = ()

#: Execute after conversion, which is what embeds the outputs. Set False for a
#: fast syntax-only rebuild.
EXECUTE = True

#: Run the reduced configuration: tiny models, few epochs, small samples. This
#: is what CI uses; leave it False to produce the notebooks students receive.
FAST = False

#: Per-notebook execution timeout, in seconds.
TIMEOUT = 3600

# -----------------------------------------------------------------------------


def convert(source: Path, target: Path) -> nbformat.NotebookNode:
    """Percent-format ``.py`` -> notebook object, written to ``target``."""
    notebook = jupytext.read(source, fmt="py:percent")
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    nbformat.write(notebook, target)
    return notebook


def execute(path: Path, timeout: int = TIMEOUT) -> tuple[bool, str, float]:
    """Run a notebook top to bottom in place. Returns ``(ok, message, seconds)``."""
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    start = time.perf_counter()
    try:
        client.execute()
        ok, message = True, "ok"
    except Exception as exc:  # noqa: BLE001 - we want the message, whatever it is
        ok, message = False, f"{type(exc).__name__}: {str(exc)[:600]}"
    elapsed = time.perf_counter() - start
    nbformat.write(notebook, path)
    return ok, message, elapsed


def build(
    only: tuple[str, ...] = ONLY,
    do_execute: bool = EXECUTE,
    fast: bool = FAST,
    timeout: int = TIMEOUT,
) -> int:
    """Convert (and optionally execute) the tutorials. Returns the failure count.

    ``fast`` sets ``AI_POWER_COURSE_FAST`` for the kernels, which every notebook
    reads through :func:`ai_power_course.config.scaled`.
    """
    os.environ["AI_POWER_COURSE_FAST"] = "1" if fast else "0"
    sources = sorted(SOURCE_DIR.glob("*.py"))
    if only:
        sources = [s for s in sources if any(token in s.name for token in only)]
    if not sources:
        raise SystemExit(f"no notebook sources matched {only!r} in {SOURCE_DIR}")

    failures = 0
    for source in sources:
        target = OUTPUT_DIR / f"{source.stem}.ipynb"
        convert(source, target)
        if not do_execute:
            print(f"converted  {target.name}")
            continue
        ok, message, elapsed = execute(target, timeout=timeout)
        status = "ok  " if ok else "FAIL"
        print(f"{status} {target.name:44s} {elapsed:7.1f}s  {'' if ok else message}")
        failures += 0 if ok else 1
    return failures


if __name__ == "__main__":
    raise SystemExit(build())
