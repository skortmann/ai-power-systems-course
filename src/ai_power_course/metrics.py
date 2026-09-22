"""Evaluation: point forecasts, probabilistic forecasts, and physical plausibility.

The course insists on three separate questions:

1. *Is the number close?*  -> :func:`mae`, :func:`rmse`, :func:`r2`, :func:`mape`
2. *Is the uncertainty honest?* -> :func:`pinball_loss`, :func:`crps_from_quantiles`,
   :func:`coverage`
3. *Could this happen in a real power system?* -> :func:`physical_violations`

A model can win on (1) and fail catastrophically on (3) — predicting PV
generation at midnight, negative load, or a 40 GW ramp in one hour. Tutorials
01, 06, 09 and 10 all report both.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "mae",
    "rmse",
    "mape",
    "smape",
    "r2",
    "nrmse",
    "bias",
    "skill_score",
    "point_metrics",
    "pinball_loss",
    "crps_from_quantiles",
    "coverage",
    "probabilistic_metrics",
    "classification_metrics",
    "physical_violations",
]

ArrayLike = np.ndarray | Sequence[float] | pd.Series


def _as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot score an empty series")
    return true, pred


# --- point forecasts ---------------------------------------------------------


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error, in the units of the target."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error. Penalises large errors more than :func:`mae`."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed error. Positive means the model over-predicts on average."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.mean(pred - true))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (%).

    Undefined near zero, so it is meaningless for PV (half the values are 0)
    and for prices (which cross zero). Use it for load, not for everything.
    """
    true, pred = _as_pair(y_true, y_pred)
    return float(100.0 * np.mean(np.abs((true - pred) / np.maximum(np.abs(true), eps))))


def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-8) -> float:
    """Symmetric MAPE (%), bounded at 200 and better behaved near zero."""
    true, pred = _as_pair(y_true, y_pred)
    denom = np.maximum((np.abs(true) + np.abs(pred)) / 2.0, eps)
    return float(100.0 * np.mean(np.abs(true - pred) / denom))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination against the *mean* predictor.

    Beware on time series: beating the mean is trivial when the series has a
    daily cycle. A high R^2 says nothing about beating persistence.
    """
    true, pred = _as_pair(y_true, y_pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def nrmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """RMSE normalised by the mean of the truth (%). Comparable across grids."""
    true, pred = _as_pair(y_true, y_pred)
    return float(100.0 * rmse(true, pred) / np.mean(np.abs(true)))


def skill_score(
    y_true: ArrayLike, y_pred: ArrayLike, y_reference: ArrayLike, metric=rmse
) -> float:
    """Fractional improvement over a reference forecast: ``1 - m(pred)/m(ref)``.

    1.0 is perfect, 0.0 matches the reference, negative is *worse* than the
    reference. This is the number that actually matters in forecasting, and the
    reference should be persistence or a seasonal naive model — not the mean.
    """
    model = metric(y_true, y_pred)
    reference = metric(y_true, y_reference)
    return float(1.0 - model / reference) if reference > 0 else float("nan")


def point_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, include_mape: bool = True
) -> dict[str, float]:
    """The standard point-forecast bundle used throughout the course."""
    out = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }
    if include_mape:
        out["MAPE_%"] = mape(y_true, y_pred)
    return out


# --- probabilistic forecasts -------------------------------------------------


def pinball_loss(y_true: ArrayLike, y_quantile: ArrayLike, quantile: float) -> float:
    """Quantile (pinball) loss for a single quantile level.

    Asymmetric on purpose: for ``quantile=0.9`` under-prediction is punished
    nine times harder than over-prediction, which is exactly what forces the
    forecast to sit at the 90th percentile.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    true, pred = _as_pair(y_true, y_quantile)
    delta = true - pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


def crps_from_quantiles(
    y_true: ArrayLike, quantile_forecasts: np.ndarray, quantile_levels: Sequence[float]
) -> float:
    """Approximate CRPS as the mean pinball loss over a quantile grid.

    The continuous ranked probability score integrates the Brier score over all
    thresholds; averaging the pinball loss over an evenly spaced quantile grid
    is the standard discrete approximation (and is what most forecasting
    benchmarks report). Exact in the limit of a dense grid.

    ``quantile_forecasts`` has shape ``(n_samples, n_quantiles)``.
    """
    true = np.asarray(y_true, dtype=float).reshape(-1)
    forecasts = np.asarray(quantile_forecasts, dtype=float)
    if forecasts.ndim != 2:
        raise ValueError("quantile_forecasts must be 2-D (n_samples, n_quantiles)")
    if forecasts.shape != (true.size, len(quantile_levels)):
        raise ValueError(
            f"expected quantile_forecasts of shape {(true.size, len(quantile_levels))}, "
            f"got {forecasts.shape}"
        )
    losses = [
        pinball_loss(true, forecasts[:, i], q) for i, q in enumerate(quantile_levels)
    ]
    # Factor 2 makes the grid-average agree with CRPS for a dense uniform grid.
    return float(2.0 * np.mean(losses))


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Empirical share of observations inside ``[lower, upper]``.

    A nominal 80% interval that covers 55% of the truth is over-confident; one
    that covers 99% is useless but honest. Calibration is not accuracy.
    """
    true = np.asarray(y_true, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    return float(np.mean((true >= lo) & (true <= hi)))


def probabilistic_metrics(
    y_true: ArrayLike,
    quantile_forecasts: np.ndarray,
    quantile_levels: Sequence[float],
) -> dict[str, float]:
    """CRPS, the median's MAE, and the coverage of the widest interval."""
    levels = list(quantile_levels)
    forecasts = np.asarray(quantile_forecasts, dtype=float)
    median_index = int(np.argmin(np.abs(np.asarray(levels) - 0.5)))
    lo_index, hi_index = int(np.argmin(levels)), int(np.argmax(levels))
    nominal = levels[hi_index] - levels[lo_index]
    return {
        "CRPS": crps_from_quantiles(y_true, forecasts, levels),
        "MAE_median": mae(y_true, forecasts[:, median_index]),
        f"Coverage_{nominal:.0%}": coverage(
            y_true, forecasts[:, lo_index], forecasts[:, hi_index]
        ),
    }


# --- classification ----------------------------------------------------------


def classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, positive_label: int = 1
) -> dict[str, float]:
    """Accuracy, precision, recall and F1 for a binary problem.

    Written out rather than imported so students can see that accuracy is
    almost useless on the imbalanced problems power systems produce: if 3% of
    hours have an overload, predicting "never" scores 97%.
    """
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("shape mismatch between y_true and y_pred")
    tp = float(np.sum((pred == positive_label) & (true == positive_label)))
    fp = float(np.sum((pred == positive_label) & (true != positive_label)))
    fn = float(np.sum((pred != positive_label) & (true == positive_label)))
    tn = float(np.sum((pred != positive_label) & (true != positive_label)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Accuracy": (tp + tn) / true.size,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PositiveRate": float(np.mean(true == positive_label)),
    }


# --- physical plausibility ---------------------------------------------------


def physical_violations(
    prediction: ArrayLike,
    *,
    lower_bound: float | None = 0.0,
    upper_bound: float | None = None,
    zero_mask: ArrayLike | None = None,
    max_ramp: float | None = None,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Count the ways a forecast is physically impossible.

    Parameters
    ----------
    lower_bound, upper_bound:
        Hard limits, e.g. ``0`` and the installed capacity. Load and generation
        cannot be negative; PV cannot exceed its inverter rating.
    zero_mask:
        Boolean array marking steps where the quantity *must* be exactly zero —
        for PV, every hour with the sun below the horizon.
    max_ramp:
        Largest admissible change between consecutive steps, in the target's
        units. A national PV fleet cannot swing 30 GW in one hour.

    Returns a dict of violation *rates* (share of steps) plus the worst
    magnitude, so the numbers are comparable across datasets.
    """
    values = np.asarray(prediction, dtype=float).reshape(-1)
    out: dict[str, float] = {}

    if lower_bound is not None:
        below = values < lower_bound - tolerance
        out["below_min_rate"] = float(np.mean(below))
        out["worst_below_min"] = float(np.min(values) - lower_bound) if below.any() else 0.0
    if upper_bound is not None:
        above = values > upper_bound + tolerance
        out["above_max_rate"] = float(np.mean(above))
        out["worst_above_max"] = float(np.max(values) - upper_bound) if above.any() else 0.0
    if zero_mask is not None:
        mask = np.asarray(zero_mask, dtype=bool).reshape(-1)
        if mask.shape != values.shape:
            raise ValueError("zero_mask must match the prediction shape")
        nonzero = mask & (np.abs(values) > tolerance)
        out["nonzero_when_impossible_rate"] = (
            float(np.mean(nonzero[mask])) if mask.any() else 0.0
        )
        out["worst_impossible_value"] = (
            float(np.max(np.abs(values[mask]))) if mask.any() else 0.0
        )
    if max_ramp is not None:
        ramps = np.abs(np.diff(values))
        out["ramp_violation_rate"] = float(np.mean(ramps > max_ramp + tolerance))
        out["worst_ramp"] = float(np.max(ramps)) if ramps.size else 0.0
    return out
