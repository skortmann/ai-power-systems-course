"""Shared plotting helpers and a consistent visual style for the course.

These are the plots that appear in more than one notebook (forecast overlays,
error profiles, embedding scatters, attention heatmaps). One-off illustrative
plots stay inline in the notebook that needs them — students should see the
matplotlib, not a wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "use_course_style",
    "COLORS",
    "plot_forecast",
    "plot_error_by_hour",
    "plot_prediction_scatter",
    "plot_learning_curve",
    "plot_embedding",
    "plot_attention",
    "plot_metric_comparison",
]

#: A small, colour-blind-safe palette (Okabe-Ito). Reused everywhere so the
#: same model keeps the same colour across ten notebooks.
COLORS: dict[str, str] = {
    "truth": "#000000",
    "baseline": "#999999",
    "linear": "#E69F00",
    "tree": "#009E73",
    "mlp": "#56B4E9",
    "rnn": "#0072B2",
    "transformer": "#D55E00",
    "foundation": "#CC79A7",
    "accent": "#F0E442",
}

_MODEL_COLOR_ORDER = [
    COLORS["baseline"], COLORS["linear"], COLORS["tree"], COLORS["mlp"],
    COLORS["rnn"], COLORS["transformer"], COLORS["foundation"], COLORS["accent"],
]


def use_course_style() -> None:
    """Apply the course's matplotlib defaults. Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.figsize": (9.0, 3.6),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "font.size": 10,
            "axes.prop_cycle": mpl.cycler(color=_MODEL_COLOR_ORDER),
        }
    )


def plot_forecast(
    truth: pd.Series | np.ndarray,
    predictions: Mapping[str, np.ndarray | pd.Series],
    index: pd.DatetimeIndex | None = None,
    title: str = "Forecast vs. observation",
    ylabel: str = "Load [MW]",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay one or more forecasts on the truth over a time window."""
    if ax is None:
        _, ax = plt.subplots()
    x = index if index is not None else getattr(truth, "index", np.arange(len(truth)))
    ax.plot(x, np.asarray(truth).reshape(-1), color=COLORS["truth"], label="observed", lw=2.0)
    for i, (name, values) in enumerate(predictions.items()):
        ax.plot(
            x,
            np.asarray(values).reshape(-1),
            label=name,
            lw=1.4,
            alpha=0.9,
            color=_MODEL_COLOR_ORDER[(i + 1) % len(_MODEL_COLOR_ORDER)],
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=min(4, len(predictions) + 1))
    return ax


def plot_error_by_hour(
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    hours: np.ndarray,
    title: str = "Mean absolute error by hour of day",
    ylabel: str = "MAE [MW]",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Where in the day does each model fail? Aggregate errors are averages of
    very different regimes; this plot is the first step of failure analysis."""
    if ax is None:
        _, ax = plt.subplots()
    truth = np.asarray(truth).reshape(-1)
    hours = np.asarray(hours).reshape(-1)
    for i, (name, values) in enumerate(predictions.items()):
        errors = np.abs(truth - np.asarray(values).reshape(-1))
        profile = pd.Series(errors).groupby(hours).mean()
        ax.plot(
            profile.index,
            profile.to_numpy(),
            marker="o",
            ms=3,
            label=name,
            color=_MODEL_COLOR_ORDER[(i + 1) % len(_MODEL_COLOR_ORDER)],
        )
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(ncol=3)
    return ax


def plot_prediction_scatter(
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "Predicted vs. observed",
    unit: str = "MW",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter with the 45-degree line: reveals bias and variance at a glance."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    truth = np.asarray(truth).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    ax.scatter(truth, prediction, s=5, alpha=0.25, color=COLORS["rnn"], edgecolors="none")
    lo = float(min(truth.min(), prediction.min()))
    hi = float(max(truth.max(), prediction.max()))
    ax.plot([lo, hi], [lo, hi], color=COLORS["truth"], lw=1.0, ls="--", label="perfect")
    ax.set_xlabel(f"observed [{unit}]")
    ax.set_ylabel(f"predicted [{unit}]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    return ax


def plot_learning_curve(
    history: Mapping[str, Sequence[float]],
    title: str = "Training history",
    ylabel: str = "loss",
    log_y: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot train/validation curves. The gap between them *is* overfitting."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, values in history.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=name, marker="o", ms=2.5)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_embedding(
    coords: np.ndarray,
    color_by: np.ndarray,
    title: str = "Learned representation",
    label: str = "",
    cmap: str = "twilight",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """2-D scatter of an embedding, coloured by a variable the encoder never saw.

    If unlabelled pretraining has learned something real, structure in this
    plot will line up with a physical variable (season, hour, PV share).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))
    coords = np.asarray(coords)
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=np.asarray(color_by), s=9, cmap=cmap, alpha=0.85
    )
    plt.colorbar(scatter, ax=ax, label=label)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_title(title)
    return ax


def plot_attention(
    weights: np.ndarray,
    title: str = "Attention weights",
    xlabel: str = "key position (attended to)",
    ylabel: str = "query position",
    tick_step: int = 12,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of an attention matrix, row-normalised as softmax leaves it.

    Read a row as: "when producing output at this position, how much weight did
    the model put on each input position?" Interpret with care — tutorial 05
    explains why this is not an explanation of the model's decision.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.2))
    weights = np.asarray(weights)
    image = ax.imshow(weights, aspect="auto", cmap="magma", origin="upper")
    plt.colorbar(image, ax=ax, label="weight")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(0, weights.shape[1], tick_step))
    ax.set_yticks(range(0, weights.shape[0], tick_step))
    return ax


def plot_metric_comparison(
    frame: pd.DataFrame,
    metric: str = "MAE",
    title: str | None = None,
    lower_is_better: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar chart of one metric across models (the leaderboard view)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 0.42 * len(frame) + 1.2))
    series = frame[metric].sort_values(ascending=not lower_is_better)
    labels = [
        f"{idx[1]} (T{idx[0]:02d})" if isinstance(idx, tuple) else str(idx)
        for idx in series.index
    ]
    colors = [_MODEL_COLOR_ORDER[i % len(_MODEL_COLOR_ORDER)] for i in range(len(series))]
    ax.barh(labels, series.to_numpy(), color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    direction = "lower" if lower_is_better else "higher"
    ax.set_title(title or f"{metric} by model ({direction} is better)")
    for y, value in enumerate(series.to_numpy()):
        ax.text(value, y, f" {value:,.1f}", va="center", fontsize=8)
    ax.grid(axis="y", visible=False)
    return ax
