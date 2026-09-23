"""Project-wide paths, random seeds and the classroom/CI execution switch.

Everything in this course that could be non-deterministic goes through here, so
a student can reproduce a number from the notebook exactly and a CI run can
shrink every experiment with a single environment variable.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "SAMPLE_DIR",
    "CACHE_DIR",
    "ARTIFACT_DIR",
    "SEED",
    "fast_mode",
    "scaled",
    "set_seed",
    "torch_device",
]

# --- Paths -------------------------------------------------------------------
# src/ai_power_course/config.py -> src/ai_power_course -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

# --- Reproducibility ---------------------------------------------------------
SEED = 20260101
"""Single course-wide seed. Notebooks call :func:`set_seed` before every model."""


def set_seed(seed: int = SEED) -> int:
    """Seed Python, NumPy and (if installed) PyTorch. Returns the seed used.

    Call this immediately before constructing *and* before training a model:
    weight initialisation and mini-batch shuffling both consume randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CPU maths costs a little speed and buys reproducible numbers.
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Pin the CUDA path too. The course runs on CPU, where the line above is
    # sufficient and two full training runs were verified bit-identical, but
    # `torch_device()` will hand back a GPU when one exists and cuDNN picks
    # algorithms nondeterministically by default.
    #
    # One limit worth stating rather than papering over: some cuBLAS kernels
    # additionally need CUBLAS_WORKSPACE_CONFIG, and it is read when the CUDA
    # context is created, so setting it from here would be too late to take
    # effect. Export it before starting the process if you need bitwise
    # determinism on a GPU:
    #
    #     CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run jupyter lab
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


# --- Classroom mode vs. CI mode ----------------------------------------------
def fast_mode() -> bool:
    """True when the notebooks should run their reduced ("CI") configuration.

    Set ``AI_POWER_COURSE_FAST=1`` to enable. CI uses it to execute all ten
    notebooks in minutes; students leave it unset and get the full experiments.
    """
    return os.environ.get("AI_POWER_COURSE_FAST", "").strip().lower() in {"1", "true", "yes"}


def scaled(full: int, fast: int) -> int:
    """Pick ``fast`` under :func:`fast_mode`, otherwise ``full``.

    Used for epoch counts, dataset sizes and sample counts so that a notebook
    reads ``epochs = scaled(full=30, fast=2)`` instead of hiding an ``if``.
    """
    return fast if fast_mode() else full


def offline_mode() -> bool:
    """True when notebooks must not reach the network (no model/data downloads).

    Set ``AI_POWER_COURSE_OFFLINE=1``. Sections that need a Hugging Face
    download check this and fall back to a tiny locally constructed stand-in.
    """
    return os.environ.get("AI_POWER_COURSE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def torch_device() -> str:
    """Return ``"cuda"`` if a GPU is usable, otherwise ``"cpu"``.

    The course never *requires* a GPU; optional sections use this helper so the
    same cell runs in both places.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
