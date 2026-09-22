"""Generate ``tutorials/exercise.ipynb`` and ``tutorials/solution.ipynb``.

Both notebooks come from the same source — :mod:`ai_power_course.exercises` —
so the student's task text and the instructor's solution cannot drift apart.
The only differences between the two files are which code cell is emitted
(scaffold or worked implementation) and whether the solution explanation is
present.

Run it as-is::

    uv run python scripts/build_exercise_notebooks.py

or import it and pass different arguments::

    from scripts.build_exercise_notebooks import build
    build(do_execute=False)

Cell tags make the structure machine-checkable, which is what
``tests/test_exercises.py`` verifies:

``["setup", "chapter-NN"]``   a chapter's data preparation, runs in both files
``["prompt", "TASK-NN-MM"]``  the task statement, byte-identical in both files
``["task", "TASK-NN-MM"]``    the code cell, scaffold or solution
``["check", "TASK-NN-MM"]``   the self-check, present in both files
``["answer", "TASK-NN-MM"]``  a reflection answer or its blank template
"""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import nbformat

from ai_power_course.config import PROJECT_ROOT, fast_mode
from ai_power_course.exercises import CHAPTERS, STAR, Chapter, Task, all_tasks

# --------------------------------------------------------------------------
# Configuration — edit these, there are no command-line flags.
# --------------------------------------------------------------------------

#: Which notebooks to build. Both are generated from the same source.
BUILD_EXERCISE = True
BUILD_SOLUTION = True

#: Execute ``solution.ipynb`` after generating it. The exercise notebook is
#: never executed in full: its task cells are deliberately incomplete.
EXECUTE = True

#: Seconds before a single cell is abandoned.
TIMEOUT = 1800

#: Where the notebooks are written.
OUTPUT_DIR = PROJECT_ROOT / "tutorials"

# --------------------------------------------------------------------------

EXERCISE_PATH = OUTPUT_DIR / "exercise.ipynb"
SOLUTION_PATH = OUTPUT_DIR / "solution.ipynb"


def _code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip("\n"))


def _markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip("\n"))


def _tag(cell: nbformat.NotebookNode, *tags: str) -> nbformat.NotebookNode:
    cell.metadata["tags"] = list(tags)
    return cell


def _header(solution: bool) -> list[nbformat.NotebookNode]:
    """The title, the instructions and the difficulty key."""
    if solution:
        title = "# Solutions — AI for Power & Energy Systems"
        warning = (
            "> **Instructor / self-check version.** Every task below is worked "
            "through in full, with an explanation of *why* the implementation "
            "looks the way it does.\n"
            ">\n"
            "> If you are a student: attempt the exercise first. Reading a "
            "solution feels like learning and is not — the gap between "
            "recognising a correct answer and producing one is the entire "
            "skill this course is trying to build."
        )
    else:
        title = "# Exercises — AI for Power & Energy Systems"
        warning = (
            "> **How to use this notebook.** Each task has a scaffold: `# TODO` "
            "markers, `____` blanks and function signatures with a docstring but "
            "no body. Fill them in.\n"
            ">\n"
            "> Most tasks end with a **self-check** cell that runs a few "
            "assertions. A passing check means the shapes and the obvious "
            "invariants are right; it does not mean the reasoning is. The "
            "analysis and reflection tasks have no check at all, because the "
            "thing being assessed is the argument.\n"
            ">\n"
            "> **Do not open `solution.ipynb` until you have attempted the "
            "corresponding exercise.**"
        )

    counts: dict[str, int] = {}
    for task in all_tasks():
        counts[task.kind] = counts.get(task.kind, 0) + 1

    kind_rows = "\n".join(
        f"| **{label}** ({counts.get(kind, 0)}) | {blurb} |"
        for kind, label, blurb in (
            ("coding", "Coding Task",
             "Implement something. The scaffold shows the shape; you write the body."),
            ("analysis", "Analysis Task",
             "Run an experiment and interpret the numbers. Partly code, mostly judgement."),
            ("reflection", "Reflection Question",
             "Write an argument. No code. Worth discussing with someone."),
        )
    )

    overview_rows = "\n".join(
        f"| {chapter.number:02d} | {chapter.title} | "
        f"[`{chapter.tutorial}`]({chapter.tutorial}) | {len(chapter.tasks)} | "
        f"{''.join(STAR[max(t.difficulty for t in chapter.tasks)])} |"
        for chapter in CHAPTERS
    )

    template = textwrap.dedent("""
        {title}

        {warning}

        Ten chapters, one per tutorial, {n_tasks} tasks in total.

        Work them in order. Ideas carry forward — chapter 06 assumes you derived
        attention in chapter 05, chapter 10 assumes you masked something in
        chapter 04 — but *code* dependencies stay inside a chapter, so you can run
        one chapter without having done the previous one. Any task that reuses an
        earlier task's code says which, in its own prompt.

        ### Task types

        | Type | What it asks for |
        | --- | --- |
        {kind_rows}

        ### Difficulty

        | | Meaning |
        | --- | --- |
        | ★ | Direct application of something the tutorial showed. |
        | ★★ | Requires combining two ideas, or getting an index right. |
        | ★★★ | Open-ended, or the correct answer is not the obvious one. |

        ### Chapters

        | # | Topic | Builds on | Tasks | Hardest |
        | --- | --- | --- | --- | --- |
        {overview_rows}

        ### Before you start

        The environment is the course's own: `uv sync`, then select the
        `ai-power-systems-course` kernel. Chapters 08 and 09 download pretrained
        models the first time they run; set `AI_POWER_COURSE_OFFLINE=1` to skip
        those cells and `AI_POWER_COURSE_FAST=1` to shrink every training budget.
        """)

    return [
        _markdown(
            template.format(
                title=title,
                warning=warning,
                n_tasks=len(all_tasks()),
                kind_rows=kind_rows,
                overview_rows=overview_rows,
            )
        ),
        _tag(
            _code("""
            # Shared imports and the reproducibility seed. Every chapter below
            # re-imports what it needs, so chapters can also be run in isolation.
            import numpy as np
            import pandas as pd
            import torch

            from ai_power_course.config import fast_mode, offline_mode, set_seed
            from ai_power_course.plotting import use_course_style

            use_course_style()
            set_seed()

            print(f"reduced (fast) configuration: {fast_mode()} | offline: {offline_mode()}")
            print("Every result below is seeded. If a number differs from a neighbour's,")
            print("it is because one of you changed something — which is the point.")
            """),
            "setup",
            "global",
        ),
    ]


def _task_cells(task: Task, solution: bool) -> list[nbformat.NotebookNode]:
    """The cells for one task. The prompt is identical in both notebooks."""
    cells = [_tag(_markdown(task.prompt_markdown()), "prompt", task.task_id)]

    if task.kind == "reflection":
        body = task.answer if solution else task.answer_template
        cells.append(_tag(_markdown(body), "answer", task.task_id))
        return cells

    source = task.solution_code if solution else task.exercise_code
    cells.append(_tag(_code(source), "task", task.task_id))

    if task.check_code:
        cells.append(_tag(_code(task.check_code), "check", task.task_id))

    if solution and task.explanation:
        cells.append(
            _tag(
                _markdown(f"**Why this works.**\n\n{textwrap.dedent(task.explanation).strip()}"),
                "explanation",
                task.task_id,
            )
        )
    return cells


def _chapter_cells(chapter: Chapter, solution: bool) -> list[nbformat.NotebookNode]:
    cells = [
        _markdown("---"),
        _markdown(chapter.intro_markdown()),
        _tag(_code(chapter.setup_code), "setup", f"chapter-{chapter.number:02d}"),
    ]
    for task in chapter.tasks:
        cells.extend(_task_cells(task, solution))
    return cells


def _footer(solution: bool) -> list[nbformat.NotebookNode]:
    if solution:
        closing = """
        ## That is the whole track

        Forty-four tasks from a linear regression on lagged load to a
        message-passing encoder pretrained on a population of grids. Two things
        are worth saying at the end.

        **The worked answers here are not the only correct ones.** Several tasks —
        3.4, 9.4, and the final challenge above — have defensible alternatives, and
        a student who reaches a different conclusion *with the numbers to support
        it* has done the exercise better than one who reproduced this text.

        **The last question is still open.** Nobody knows what the tokens of a grid
        foundation model should be. The worked example in task 10.6 is one
        consistent design; it is not the answer, and it says so. If a student
        argues convincingly that branches beat buses, that is a research
        contribution, not a wrong answer.
        """
    else:
        closing = """
        ## Finished?

        Some things worth doing before you look at the solutions.

        1. **Re-read your reflection answers.** They are the ones that are hard to
           mark yourself and the ones most worth discussing with a supervisor or a
           colleague.
        2. **Find a task where your number disagrees with a neighbour's** and work
           out which of you changed something. Seeds are fixed throughout; a
           difference always has a cause.
        3. **Then** open `solution.ipynb`, and compare *reasoning*, not just
           output. Several tasks have more than one defensible answer.

        The last task has no solution in any meaningful sense. What the tokens of a
        grid foundation model should be is an open research question, and the
        worked example in the solution notebook is one design among many.
        """
    return [_markdown("---"), _markdown(closing)]


def _notebook(solution: bool) -> nbformat.NotebookNode:
    cells = _header(solution)
    for chapter in CHAPTERS:
        cells.extend(_chapter_cells(chapter, solution))
    cells.extend(_footer(solution))

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    return notebook


def _execute(notebook: nbformat.NotebookNode, path: Path, timeout: int) -> None:
    """Run every cell, and report the first failure with its traceback."""
    from nbclient import NotebookClient

    started = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()
    print(f"  executed in {time.perf_counter() - started:.0f}s")


def verify_exercise_setup(timeout: int = TIMEOUT) -> int:
    """Execute only the setup cells of ``exercise.ipynb``.

    The task cells are incomplete by design, so the notebook as a whole cannot
    run. The setup cells must, though: they are what a student needs before
    attempting anything, and the invariant that makes them runnable is that a
    chapter's setup never reads a variable a task was asked to create.

    Returns the number of cells executed.
    """
    from nbclient import NotebookClient

    notebook = _notebook(solution=False)
    setup_only = nbformat.v4.new_notebook(
        cells=[c for c in notebook.cells if "setup" in c.metadata.get("tags", [])],
        metadata=notebook.metadata,
    )
    print(f"verifying {len(setup_only.cells)} setup cells of exercise.ipynb")
    started = time.perf_counter()
    NotebookClient(
        setup_only,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(OUTPUT_DIR)}},
        allow_errors=False,
    ).execute()
    print(f"  all setup cells ran in {time.perf_counter() - started:.0f}s")
    return len(setup_only.cells)


def build(
    *,
    build_exercise: bool = BUILD_EXERCISE,
    build_solution: bool = BUILD_SOLUTION,
    do_execute: bool = EXECUTE,
    timeout: int = TIMEOUT,
) -> list[Path]:
    """Generate the notebooks. Returns the paths written."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if build_exercise:
        notebook = _notebook(solution=False)
        # The exercise notebook is never executed: its task cells are incomplete
        # by construction. tests/test_exercises.py runs its setup cells instead.
        nbformat.write(notebook, EXERCISE_PATH)
        print(f"wrote {EXERCISE_PATH.relative_to(PROJECT_ROOT)} "
              f"({len(notebook.cells)} cells, task cells not executed by design)")
        written.append(EXERCISE_PATH)
        if do_execute:
            verify_exercise_setup(timeout)

    if build_solution:
        notebook = _notebook(solution=True)
        print(f"building {SOLUTION_PATH.relative_to(PROJECT_ROOT)} "
              f"({len(notebook.cells)} cells)"
              f"{' — reduced configuration' if fast_mode() else ''}")
        if do_execute:
            _execute(notebook, SOLUTION_PATH, timeout)
        nbformat.write(notebook, SOLUTION_PATH)
        print(f"wrote {SOLUTION_PATH.relative_to(PROJECT_ROOT)}")
        written.append(SOLUTION_PATH)

    return written


if __name__ == "__main__":
    build()
