"""Whole-project checks: imports, corpus, diagrams, results store, notebooks."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ai_power_course.config import (  # noqa: E402
    PROJECT_ROOT,
    fast_mode,
    scaled,
    set_seed,
    torch_device,
)

TUTORIAL_DIR = PROJECT_ROOT / "tutorials"
EXPECTED_NOTEBOOKS = [
    "01_classical_machine_learning",
    "02_neural_networks",
    "03_rnns_and_lstms",
    "04_representation_learning",
    "05_attention",
    "06_transformers",
    "07_language_models",
    "08_pretrained_llms_and_adaptation",
    "09_foundation_models_beyond_llms",
    "10_grid_foundation_models",
]

MODULES = [
    "ai_power_course",
    "ai_power_course.config",
    "ai_power_course.corpus",
    "ai_power_course.data",
    "ai_power_course.diagrams",
    "ai_power_course.metrics",
    "ai_power_course.plotting",
    "ai_power_course.results",
    "ai_power_course.synthetic",
    "ai_power_course.grid.graphs",
    "ai_power_course.grid.networks",
    "ai_power_course.grid.physics",
    "ai_power_course.grid.sampling",
    "ai_power_course.models.attention",
    "ai_power_course.models.baselines",
    "ai_power_course.models.forecasters",
    "ai_power_course.models.gnn",
    "ai_power_course.models.representation",
    "ai_power_course.models.tinygpt",
    "ai_power_course.models.training",
]


# --- imports and config -------------------------------------------------------


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_every_public_name_in_dunder_all_exists():
    for name in MODULES:
        module = importlib.import_module(name)
        for symbol in getattr(module, "__all__", []):
            assert hasattr(module, symbol), f"{name}.{symbol} is exported but missing"


def test_set_seed_is_reproducible():
    import numpy as np
    import torch

    set_seed(123)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(123)
    b = (np.random.rand(3), torch.rand(3))
    np.testing.assert_allclose(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_fast_mode_switch(monkeypatch):
    monkeypatch.delenv("AI_POWER_COURSE_FAST", raising=False)
    assert fast_mode() is False
    assert scaled(full=50, fast=2) == 50
    monkeypatch.setenv("AI_POWER_COURSE_FAST", "1")
    assert fast_mode() is True
    assert scaled(full=50, fast=2) == 2


def test_torch_device_is_valid():
    assert torch_device() in {"cpu", "cuda"}


# --- corpus -------------------------------------------------------------------


def test_corpus_generation_is_deterministic():
    from ai_power_course.corpus import generate_corpus

    assert generate_corpus(n_logs=20, n_reports=3, n_assets=5) == generate_corpus(
        n_logs=20, n_reports=3, n_assets=5
    )


def test_corpus_has_enough_structure_for_a_tiny_model():
    from ai_power_course.corpus import corpus_path, write_corpus

    path = corpus_path()
    if not path.exists():
        write_corpus()
    text = path.read_text(encoding="utf-8")
    assert len(text) > 100_000
    assert 40 < len(set(text)) < 120, "a character model needs a small vocabulary"
    assert "SYNTHETIC OPERATIONAL RECORDS" in text, "provenance must be stated in the file"


def test_corpus_magnitudes_are_physically_sensible():
    """A 20 kV feeder must not be described as carrying hundreds of megawatts."""
    import re

    from ai_power_course.corpus import corpus_path, write_corpus

    path = corpus_path()
    if not path.exists():
        write_corpus()
    text = path.read_text(encoding="utf-8")
    limits = {20: 40, 30: 60, 110: 250, 220: 600, 380: 1400}
    for level, maximum in limits.items():
        loads = [float(m) for m in re.findall(rf"(?<!\d){level} kV \| load ([\d.]+) MW", text)]
        assert loads, f"no entries found for {level} kV"
        assert max(loads) <= maximum, f"{level} kV entry carries {max(loads)} MW"


def test_event_dataset_is_balanced_and_coherent():
    import re

    import numpy as np

    from ai_power_course.corpus import EVENT_CLASSES, generate_event_dataset

    texts, labels = generate_event_dataset(n_per_class=50)
    assert len(texts) == 50 * len(EVENT_CLASSES)
    assert (np.bincount(labels) == 50).all()
    # No sentence may claim a voltage was "raised" to an undervoltage, or "fell"
    # to an overvoltage.
    for text in texts:
        assert not (("raised" in text) and re.search(r"to 0\.\d+ per unit", text))
        assert not re.search(r"(fell|sagged) to 1\.\d+ per unit", text)


# --- diagrams -----------------------------------------------------------------


def test_every_diagram_renders():
    from ai_power_course import diagrams
    from ai_power_course.plotting import use_course_style

    use_course_style()
    for name in diagrams.__all__:
        figure = getattr(diagrams, name)()
        assert figure is not None, name
        plt.close(figure)


def test_plotting_helpers_run():
    import numpy as np
    import pandas as pd

    from ai_power_course.plotting import (
        plot_attention,
        plot_embedding,
        plot_error_by_hour,
        plot_forecast,
        plot_learning_curve,
        plot_metric_comparison,
        plot_prediction_scatter,
    )

    rng = np.random.default_rng(0)
    truth = rng.normal(size=48)
    prediction = truth + rng.normal(scale=0.1, size=48)
    plot_forecast(truth, {"model": prediction})
    plot_error_by_hour(truth, {"model": prediction}, hours=np.arange(48) % 24)
    plot_prediction_scatter(truth, prediction)
    plot_learning_curve({"train": [1.0, 0.5, 0.3]})
    plot_embedding(rng.normal(size=(30, 2)), rng.normal(size=30))
    weights = rng.random((8, 8))
    plot_attention(weights / weights.sum(axis=1, keepdims=True), tick_step=2)
    plot_metric_comparison(pd.DataFrame({"MAE": [1.0, 2.0]}, index=["a", "b"]))
    plt.close("all")


# --- results store ------------------------------------------------------------


def test_leaderboard_roundtrip(tmp_path):
    from ai_power_course.results import Leaderboard, ResultEntry

    path = tmp_path / "leaderboard.json"
    board = Leaderboard(path)
    board.add(ResultEntry(model="A", tutorial=1, task="t", metrics={"MAE": 2.0}))
    board.add(ResultEntry(model="B", tutorial=2, task="t", metrics={"MAE": 1.0}))
    assert json.loads(path.read_text())

    reloaded = Leaderboard(path)
    frame = reloaded.to_frame(task="t")
    assert len(frame) == 2
    assert frame.MAE.min() == 1.0

    # Re-recording the same model replaces rather than duplicates.
    reloaded.add(ResultEntry(model="A", tutorial=1, task="t", metrics={"MAE": 0.5}))
    assert len(Leaderboard(path).to_frame(task="t")) == 2

    reloaded.clear(tutorial=1)
    assert len(Leaderboard(path).to_frame(task="t")) == 1


def test_empty_leaderboard_returns_an_empty_frame(tmp_path):
    from ai_power_course.results import Leaderboard

    assert Leaderboard(tmp_path / "none.json").to_frame().empty


# --- notebooks ----------------------------------------------------------------


def test_all_ten_notebook_sources_exist():
    sources = TUTORIAL_DIR / "_sources"
    missing = [name for name in EXPECTED_NOTEBOOKS if not (sources / f"{name}.py").exists()]
    assert not missing, f"missing notebook sources: {missing}"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_source_is_valid_python(name):
    """Catches syntax errors without paying for execution."""
    source = TUTORIAL_DIR / "_sources" / f"{name}.py"
    if not source.exists():
        pytest.skip(f"{name} not written yet")
    ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_built_notebook_is_well_formed(name):
    """A built notebook must be valid JSON, have cells, and contain no errors."""
    import nbformat

    path = TUTORIAL_DIR / f"{name}.ipynb"
    if not path.exists():
        pytest.skip(f"{name}.ipynb has not been built")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) > 10, "a tutorial should have more than ten cells"

    errors = [
        output
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, f"{name} contains {len(errors)} error output(s)"


def test_notebooks_follow_the_house_structure():
    """Every tutorial carries the same pedagogical skeleton."""
    required = ["Why this matters", "Learning objectives", "Exercises",
                "Key takeaways", "Further reading"]
    sources = TUTORIAL_DIR / "_sources"
    for name in EXPECTED_NOTEBOOKS:
        source = sources / f"{name}.py"
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8")
        missing = [heading for heading in required if heading not in text]
        assert not missing, f"{name} is missing sections: {missing}"


def test_repository_has_its_documentation():
    for relative in [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "data/README.md",
        "docs/literature.md",
        "docs/glossary.md",
        "docs/ai_timeline.md",
        "docs/foundation_models.md",
        "docs/course_overview.md",
        "docs/instructor_guide.md",
    ]:
        assert (PROJECT_ROOT / relative).exists(), f"missing {relative}"


def test_no_secrets_are_committed():
    """Guard against an accidentally pasted credential.

    The patterns match the *shape* of real keys, not just their prefix: a naive
    "sk-" substring search fires on every occurrence of "mask-" and "task-",
    which is how this test failed the first time it was written.
    """
    import re

    patterns = {
        "OpenAI key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
        "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
        "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    }
    roots = [PROJECT_ROOT / "src", PROJECT_ROOT / "tutorials" / "_sources",
             PROJECT_ROOT / "scripts", PROJECT_ROOT / "tests"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in patterns.items():
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {label}")
    assert not offenders, offenders


def test_sample_data_is_small_enough_to_commit():
    sample = PROJECT_ROOT / "data" / "sample"
    for path in sample.iterdir():
        if path.is_file():
            size_mb = path.stat().st_size / 1e6
            assert size_mb < 5, f"{path.name} is {size_mb:.1f} MB — too large to commit"


def test_no_model_weights_are_committed():
    for pattern in ("*.pt", "*.pth", "*.safetensors", "*.bin", "*.ckpt"):
        found = [p for p in Path(PROJECT_ROOT).rglob(pattern)
                 if ".venv" not in p.parts and "cache" not in p.parts]
        assert not found, f"model weights should not be committed: {found}"
