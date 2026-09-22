"""Reusable code for *From Machine Learning to Foundation Models*.

The notebooks are the course; this package holds the parts that would otherwise
be copy-pasted between ten of them — the dataset, the metrics, the training
loop, the model definitions and the grid tooling.

Deliberately *not* in here: the algorithms the course is about. Gradient
descent, backpropagation, attention and masked pretraining are written out in
the notebooks where students can see them, even where a library function
exists. The rule is: if moving code here would hide the idea being taught, it
stays in the notebook.

Typical use inside a notebook::

    from ai_power_course.config import set_seed, fast_mode
    from ai_power_course.data import load_energy_data, time_split
    from ai_power_course.metrics import point_metrics
    from ai_power_course.plotting import use_course_style
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "config",
    "data",
    "metrics",
    "plotting",
    "results",
    "synthetic",
    "diagrams",
]
