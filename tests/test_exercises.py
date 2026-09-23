"""The exercise and solution notebooks must stay in lockstep.

Two notebooks generated from one source can still drift if the generator is
wrong, so these tests check the *generated artefacts* rather than the source:
matching task ids in matching order, byte-identical prompts, a solution for
every exercise, and no scaffolding left behind in the solution.

The expensive check — that ``solution.ipynb`` actually runs — belongs in CI and
lives in ``.github/workflows/ci.yml``, not here.
"""

from __future__ import annotations

import re

import nbformat
import pytest

from ai_power_course.config import PROJECT_ROOT
from ai_power_course.exercises import CHAPTERS, STAR, all_tasks

EXERCISE_PATH = PROJECT_ROOT / "tutorials" / "exercise.ipynb"
SOLUTION_PATH = PROJECT_ROOT / "tutorials" / "solution.ipynb"

#: Markers that mean "the student has to fill this in". None may survive into
#: the solution notebook.
SCAFFOLD_PATTERNS = (
    re.compile(r"#\s*TODO"),
    re.compile(r"_{4,}"),
    re.compile(r"\bNotImplementedError\b"),
    re.compile(r"^\s*pass\s*$", re.MULTILINE),
)


def _load(path):
    if not path.exists():
        pytest.skip(f"{path.name} has not been generated yet")
    return nbformat.read(path, as_version=4)


@pytest.fixture(scope="module")
def exercise():
    return _load(EXERCISE_PATH)


@pytest.fixture(scope="module")
def solution():
    return _load(SOLUTION_PATH)


def _tagged(notebook, tag):
    """Cells carrying ``tag``, in notebook order."""
    return [c for c in notebook.cells if tag in c.metadata.get("tags", [])]


def _ids(notebook, tag):
    """The task ids of cells carrying ``tag``, in notebook order."""
    out = []
    for cell in _tagged(notebook, tag):
        out.extend(t for t in cell.metadata["tags"] if t.startswith("TASK-"))
    return out


# --------------------------------------------------------------------- source


def test_task_ids_are_unique_and_well_formed():
    ids = [task.task_id for task in all_tasks()]
    assert len(ids) == len(set(ids)), "duplicate task id"
    for task_id in ids:
        assert re.fullmatch(r"TASK-\d{2}-\d{2}", task_id), task_id


def test_task_count_is_within_the_intended_range():
    """Size the CORE course separately from any optional chapter.

    Counting them together would let an advanced chapter quietly absorb the
    budget the ten required chapters are supposed to fit into -- the bound
    exists to stop the onboarding course growing, and an optional extra is not
    part of that course.
    """
    core = [c for c in CHAPTERS if not c.optional]
    core_total = sum(len(c.tasks) for c in core)
    assert len(core) == 10, f"expected ten core chapters, found {len(core)}"
    assert 35 <= core_total <= 45, (
        f"{core_total} core tasks is outside the intended range"
    )
    for chapter in CHAPTERS:
        upper = 8 if chapter.optional else 6
        assert 3 <= len(chapter.tasks) <= upper, (
            f"chapter {chapter.number} has {len(chapter.tasks)} tasks"
        )


def test_every_chapter_points_at_an_existing_tutorial():
    for chapter in CHAPTERS:
        target = PROJECT_ROOT / "tutorials" / chapter.tutorial
        assert target.exists(), f"chapter {chapter.number} references {chapter.tutorial}"


def test_every_task_is_complete():
    for task in all_tasks():
        assert task.background.strip(), f"{task.task_id} has no background"
        assert task.instruction.strip(), f"{task.task_id} has no instruction"
        assert task.difficulty in STAR, f"{task.task_id} has a bad difficulty"
        if task.kind == "reflection":
            assert task.answer_template.strip(), f"{task.task_id} has no answer template"
            assert task.answer.strip(), f"{task.task_id} has no answer"
        else:
            assert task.exercise_code.strip(), f"{task.task_id} has no scaffold"
            assert task.solution_code.strip(), f"{task.task_id} has no solution"


def test_every_coding_scaffold_actually_scaffolds():
    """A scaffold with nothing to fill in is not an exercise."""
    for task in all_tasks():
        if task.kind == "reflection":
            continue
        assert any(p.search(task.exercise_code) for p in SCAFFOLD_PATTERNS), (
            f"{task.task_id}: the scaffold has no TODO, blank or stub to fill in"
        )


def test_difficulty_does_not_decrease_across_the_track():
    """Chapter 10 must not be easier than chapter 01."""
    peaks = [max(t.difficulty for t in c.tasks) for c in CHAPTERS]
    assert peaks[-1] >= peaks[0], "the track should not get easier"
    assert max(peaks[5:]) == 3, "the second half should contain three-star tasks"


def test_setup_code_never_depends_on_student_output():
    """The invariant that keeps exercise.ipynb runnable.

    A chapter's setup may only use names it defines itself or imports. If it
    referenced a variable a task was asked to create, the exercise notebook's
    setup cells would fail for any student who had not done that task.
    """
    for chapter in CHAPTERS:
        produced = set()
        for task in chapter.tasks:
            for line in task.solution_code.splitlines():
                match = re.match(r"^\s{0,4}(\w+)\s*=", line)
                if match:
                    produced.add(match.group(1))
                match = re.match(r"^\s{0,4}(?:def|class)\s+(\w+)", line)
                if match:
                    produced.add(match.group(1))
        setup_names = set(re.findall(r"\b[a-zA-Z_]\w*\b", chapter.setup_code))
        defined_in_setup = set(re.findall(r"^\s*(\w+)\s*=", chapter.setup_code, re.MULTILINE))
        defined_in_setup |= set(
            re.findall(r"^\s*(?:def|class)\s+(\w+)", chapter.setup_code, re.MULTILINE)
        )
        defined_in_setup |= set(re.findall(r"import\s+([\w, ]+)", chapter.setup_code))
        leaked = (produced & setup_names) - defined_in_setup
        assert not leaked, (
            f"chapter {chapter.number} setup uses student-produced names: {sorted(leaked)}"
        )


# --------------------------------------------------- generated notebooks


def test_both_notebooks_exist(exercise, solution):
    assert exercise.cells and solution.cells


def test_task_ids_match_and_are_in_the_same_order(exercise, solution):
    expected = [task.task_id for task in all_tasks()]
    assert _ids(exercise, "prompt") == expected
    assert _ids(solution, "prompt") == expected


def test_every_exercise_has_a_solution(exercise, solution):
    exercise_tasks = set(_ids(exercise, "task")) | set(_ids(exercise, "answer"))
    solution_tasks = set(_ids(solution, "task")) | set(_ids(solution, "answer"))
    assert exercise_tasks == solution_tasks, (
        f"only in exercise: {sorted(exercise_tasks - solution_tasks)}; "
        f"only in solution: {sorted(solution_tasks - exercise_tasks)}"
    )


def test_prompts_are_byte_identical(exercise, solution):
    """The student and the instructor must be reading the same question."""
    for a, b in zip(_tagged(exercise, "prompt"), _tagged(solution, "prompt"), strict=True):
        assert a.source == b.source, f"prompt differs for {a.metadata['tags']}"


def test_self_checks_appear_in_both_notebooks(exercise, solution):
    assert _ids(exercise, "check") == _ids(solution, "check")
    assert _ids(exercise, "check"), "no self-checks were emitted at all"


def test_setup_cells_are_identical(exercise, solution):
    a = [c.source for c in _tagged(exercise, "setup")]
    b = [c.source for c in _tagged(solution, "setup")]
    assert a == b, "the two notebooks prepare their data differently"


def test_no_scaffolding_survives_into_the_solution(solution):
    for cell in solution.cells:
        if cell.cell_type != "code":
            continue
        tags = cell.metadata.get("tags", [])
        for pattern in SCAFFOLD_PATTERNS:
            match = pattern.search(cell.source)
            assert match is None, (
                f"solution cell {tags} still contains {match.group(0)!r}"
            )


def test_the_exercise_notebook_leaks_no_solutions(exercise, solution):
    """Every task code cell must actually differ between the two notebooks."""
    by_id = {}
    for cell in _tagged(solution, "task"):
        task_id = next(t for t in cell.metadata["tags"] if t.startswith("TASK-"))
        by_id[task_id] = cell.source
    for cell in _tagged(exercise, "task"):
        task_id = next(t for t in cell.metadata["tags"] if t.startswith("TASK-"))
        assert cell.source != by_id[task_id], f"{task_id}: the scaffold IS the solution"


def test_explanations_are_solution_only(exercise, solution):
    assert not _tagged(exercise, "explanation"), "the exercise notebook leaks explanations"
    coding = {t.task_id for t in all_tasks() if t.kind != "reflection" and t.explanation}
    assert set(_ids(solution, "explanation")) == coding


def test_reflection_answers_differ_from_their_templates(exercise, solution):
    reflections = {t.task_id for t in all_tasks() if t.kind == "reflection"}
    assert set(_ids(exercise, "answer")) == reflections
    templates = {
        next(t for t in c.metadata["tags"] if t.startswith("TASK-")): c.source
        for c in _tagged(exercise, "answer")
    }
    for cell in _tagged(solution, "answer"):
        task_id = next(t for t in cell.metadata["tags"] if t.startswith("TASK-"))
        assert len(cell.source) > len(templates[task_id]), (
            f"{task_id}: the worked answer is no longer than the blank template"
        )


def test_the_exercise_notebook_warns_against_reading_the_solution(exercise):
    text = "\n".join(c.source for c in exercise.cells if c.cell_type == "markdown")
    assert "solution.ipynb" in text
    assert "attempted" in text.lower()


def test_the_final_challenge_is_present_and_open_ended(exercise, solution):
    """The capstone of the CORE course, which is chapter 10's last task.

    Pinned to the core course rather than to `all_tasks()[-1]`, so that adding
    an optional advanced chapter does not silently move what "the final
    challenge" means.
    """
    core_tasks = [
        task
        for chapter in CHAPTERS
        if not chapter.optional
        for task in chapter.tasks
    ]
    final = core_tasks[-1]
    assert final.task_id == "TASK-10-06"
    assert final.kind == "reflection"
    assert "token" in final.background.lower()
    solution_text = "\n".join(c.source for c in _tagged(solution, "answer"))
    assert "not the answer" in solution_text, (
        "the worked example must not present itself as the answer"
    )


@pytest.mark.parametrize("path", [EXERCISE_PATH, SOLUTION_PATH])
def test_no_absolute_local_paths_are_embedded(path):
    if not path.exists():
        pytest.skip(f"{path.name} has not been generated yet")
    text = path.read_text(encoding="utf-8")
    assert "/home/" not in text, f"{path.name} embeds an absolute home path"
    assert "C:\\Users" not in text


def test_the_solution_notebook_ran_without_errors(solution):
    """Generated with execution on, so no cell may carry an error output."""
    executed = [c for c in solution.cells if c.cell_type == "code" and c.get("outputs")]
    if not executed:
        pytest.skip("solution.ipynb was generated without executing it")
    for cell in solution.cells:
        for output in cell.get("outputs", []):
            assert output.output_type != "error", (
                f"{cell.metadata.get('tags')}: "
                f"{output.get('ename')}: {output.get('evalue')}"
            )


def test_dependencies_point_backwards_and_exist():
    """A task may only reuse names from tasks a student has already reached."""
    known = {task.number for task in all_tasks()}
    order = {task.number: i for i, task in enumerate(all_tasks())}
    for task in all_tasks():
        for dependency in task.depends_on:
            assert dependency in known, f"{task.task_id} depends on unknown {dependency}"
            assert order[dependency] < order[task.number], (
                f"{task.task_id} depends on {dependency}, which comes later"
            )


def test_dependencies_are_stated_in_the_prompt(exercise):
    """A student who skipped ahead must be told what they are missing."""
    prompts = {
        next(t for t in c.metadata["tags"] if t.startswith("TASK-")): c.source
        for c in _tagged(exercise, "prompt")
    }
    for task in all_tasks():
        if not task.depends_on:
            continue
        text = prompts[task.task_id]
        for dependency in task.depends_on:
            assert dependency in text, (
                f"{task.task_id} does not mention its dependency on {dependency}"
            )


@pytest.mark.parametrize("which", ["exercise", "solution"])
def test_markdown_cells_are_not_accidentally_indented(which, exercise, solution):
    """Four leading spaces turn a whole markdown cell into a code block.

    It happened: a multi-line substitution into a triple-quoted template lands
    flush left, so ``textwrap.dedent`` finds no common prefix and strips
    nothing. Nothing errors and the notebook renders as a wall of monospace.
    """
    notebook = exercise if which == "exercise" else solution
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        indented = [
            line for line in cell.source.split("\n")
            if line.startswith("    ") and line.strip()
        ]
        assert not indented, (
            f"{which} cell {index} {cell.metadata.get('tags', [])} is indented "
            f"and will render as a code block: {indented[0][:60]!r}"
        )


@pytest.mark.parametrize("which", ["exercise", "solution"])
def test_markdown_tables_are_well_formed(which, exercise, solution):
    """Rows of one table must agree on their column count, or it renders as prose.

    A cell may hold several tables, so contiguous runs of rows are checked
    separately rather than the cell as a whole.
    """
    notebook = exercise if which == "exercise" else solution
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        block: list[str] = []
        for line in [*cell.source.split("\n"), ""]:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                block.append(stripped)
                continue
            if len(block) > 1:
                widths = {row.count("|") for row in block}
                assert len(widths) == 1, (
                    f"{which} cell {index} {cell.metadata.get('tags', [])} has a "
                    f"table whose rows disagree on column count {sorted(widths)}: "
                    f"{block[0][:60]!r}"
                )
            block = []


#: Where an abstraction would do the exact thing the task exists to teach.
FORBIDDEN_IN_SOLUTION = {
    # Qualified forms only: tasks 5.1 and 6.1 legitimately DEFINE functions of
    # their own with these names.
    "5.1": ("MultiheadAttention", "torch.nn.functional.scaled_dot_product_attention",
            "F.scaled_dot_product_attention", "import torch"),
    "5.3": ("MultiheadAttention", "F.scaled_dot_product_attention"),
    "6.1": ("MultiheadAttention", "torch.nn.functional.scaled_dot_product_attention",
            "F.scaled_dot_product_attention"),
    "6.3": ("nn.TransformerEncoderLayer", "nn.Transformer("),
    "10.2": ("torch_geometric", "MessagePassingLayer", "GridEncoder("),
}


def test_the_lesson_is_never_delegated_to_a_library():
    """No task may solve itself with the abstraction it is meant to explain.

    The self-check cells DO compare against the library version — that is the
    right use of it, and it comes after the student has written their own.
    """
    by_number = {task.number: task for task in all_tasks()}
    for number, banned in FORBIDDEN_IN_SOLUTION.items():
        code = by_number[number].solution_code
        for token in banned:
            assert token not in code, (
                f"task {number} solves itself with {token!r}, which is the lesson"
            )


def test_attention_is_verified_against_the_library_afterwards():
    """Having written it, a student should confirm it against the real thing."""
    by_number = {task.number: task for task in all_tasks()}
    assert "scaled_dot_product_attention" in by_number["5.1"].check_code
    assert "scaled_dot_product_attention" in by_number["6.1"].check_code
