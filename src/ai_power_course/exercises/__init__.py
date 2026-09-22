"""The single source of truth for the student exercise track.

``tutorials/exercise.ipynb`` and ``tutorials/solution.ipynb`` are *generated*
from the chapter definitions here by ``scripts/build_exercise_notebooks.py``.
Neither notebook is edited by hand: the task text, numbering, ordering, setup
code and self-checks exist once, so the instructor's solutions cannot quietly
drift away from what the students were asked to do. ``tests/test_exercises.py``
enforces that.

Why this lives in the package rather than a top-level ``course_content/``
directory: it has to be importable by the build script and by the test suite,
and the project already has a src layout. Putting it here means no path
manipulation and no second packaging story.

Structure
---------
:class:`Chapter` holds the tasks for one tutorial plus the setup code that
prepares its data. :class:`Task` holds one task in both variants — the
scaffolded version the student receives and the worked version the instructor
gets — together with the prose that frames it.

The rule that keeps ``exercise.ipynb`` runnable: **a chapter's setup code may
never depend on a variable a student was asked to create.** Setup builds what it
needs from :mod:`ai_power_course`, so every setup cell executes cleanly even in
a notebook where no task has been attempted.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "Task",
    "Chapter",
    "CHAPTERS",
    "all_tasks",
    "TaskKind",
    "STAR",
]

TaskKind = Literal["coding", "analysis", "reflection"]

STAR = {1: "★", 2: "★★", 3: "★★★"}

_KIND_LABEL: dict[str, str] = {
    "coding": "Coding Task",
    "analysis": "Analysis Task",
    "reflection": "Reflection Question",
}


def _sort_key(number: str) -> tuple[int, int]:
    """Sortable (chapter, index) for a human-facing task number like "5.1"."""
    chapter, index = number.split(".")
    return int(chapter), int(index)


def _clean(text: str) -> str:
    """Dedent a triple-quoted block and strip the leading/trailing blank line."""
    return textwrap.dedent(text).strip("\n")


@dataclass(frozen=True)
class Task:
    """One task, in both its student and instructor form.

    ``number`` is the human-facing "5.1"; ``task_id`` is derived from it and is
    what the synchronization test matches on.
    """

    number: str
    title: str
    kind: TaskKind
    difficulty: int
    background: str
    instruction: str
    #: Shown as a bullet list under "Requirements". Keep them checkable.
    requirements: tuple[str, ...] = ()
    #: One to three nudges. A hint that gives away the answer is not a hint.
    hints: tuple[str, ...] = ()
    #: What the student should *observe*, not the answer itself.
    expected: str = ""
    #: Scaffolded code for the student. Empty for reflection tasks.
    exercise_code: str = ""
    #: The worked implementation. Must run.
    solution_code: str = ""
    #: Lightweight self-check shown in *both* notebooks, after the code cell.
    check_code: str = ""
    #: Solution-only prose explaining why the implementation works.
    explanation: str = ""
    #: Markdown template the student fills in (reflection tasks).
    answer_template: str = ""
    #: The worked answer (reflection tasks, solution notebook only).
    answer: str = ""
    #: Task numbers this one reuses names from, e.g. ``("5.1",)``. Rendered in
    #: the prompt so a student who skipped ahead knows what they are missing.
    depends_on: tuple[str, ...] = ()
    optional: bool = False

    def __post_init__(self) -> None:
        if self.difficulty not in STAR:
            raise ValueError(f"task {self.number}: difficulty must be 1, 2 or 3")
        for dependency in self.depends_on:
            if _sort_key(dependency) >= _sort_key(self.number):
                raise ValueError(
                    f"task {self.number} depends on {dependency}, which is not earlier"
                )
        if self.kind == "reflection":
            if not self.answer:
                raise ValueError(f"task {self.number}: a reflection needs an answer")
        elif not self.solution_code:
            raise ValueError(f"task {self.number}: a {self.kind} task needs solution code")

    @property
    def chapter_number(self) -> int:
        return int(self.number.split(".")[0])

    @property
    def task_id(self) -> str:
        chapter, index = self.number.split(".")
        return f"TASK-{int(chapter):02d}-{int(index):02d}"

    @property
    def stars(self) -> str:
        return STAR[self.difficulty] + (" Optional" if self.optional else "")

    @property
    def heading(self) -> str:
        return f"{_KIND_LABEL[self.kind]} {self.number} {self.stars} — {self.title}"

    def prompt_markdown(self) -> str:
        """The task statement, identical in both notebooks."""
        parts = [f"### {self.heading}", "", _clean(self.background), ""]
        if self.depends_on:
            needed = ", ".join(self.depends_on)
            parts += [
                f"*Reuses what you wrote in {needed} — do those first.*",
                "",
            ]
        parts += ["**Your task.** " + _clean(self.instruction), ""]
        if self.requirements:
            parts.append("**Requirements**")
            parts.append("")
            parts += [f"- {r}" for r in self.requirements]
            parts.append("")
        if self.hints:
            parts.append("<details><summary><b>Hints</b></summary>")
            parts.append("")
            parts += [f"- {h}" for h in self.hints]
            parts.append("")
            parts.append("</details>")
            parts.append("")
        if self.expected:
            parts += ["**Expected outcome.** " + _clean(self.expected), ""]
        return "\n".join(parts).rstrip() + "\n"


@dataclass(frozen=True)
class Chapter:
    """One exercise chapter, corresponding to one tutorial."""

    number: int
    title: str
    tutorial: str
    intro: str
    #: Runs in *both* notebooks and must never depend on student output.
    setup_code: str
    tasks: tuple[Task, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for task in self.tasks:
            if task.chapter_number != self.number:
                raise ValueError(
                    f"task {task.number} is filed under chapter {self.number}"
                )
        numbers = [t.number for t in self.tasks]
        if numbers != sorted(numbers, key=_sort_key):
            raise ValueError(f"chapter {self.number}: tasks are out of order")

    def intro_markdown(self) -> str:
        return (
            f"## Exercise {self.number:02d} — {self.title}\n\n"
            f"*Builds on [`{self.tutorial}`]({self.tutorial}).*\n\n"
            f"{_clean(self.intro)}\n"
        )


def _load_chapters() -> tuple[Chapter, ...]:
    from .part1 import CHAPTERS as EARLY
    from .part2 import CHAPTERS as LATE

    chapters = (*EARLY, *LATE)
    if [c.number for c in chapters] != list(range(1, len(chapters) + 1)):
        raise ValueError("chapters must be numbered 1..N without gaps")
    return chapters


CHAPTERS: tuple[Chapter, ...] = _load_chapters()


def all_tasks() -> tuple[Task, ...]:
    """Every task, in notebook order."""
    return tuple(task for chapter in CHAPTERS for task in chapter.tasks)
