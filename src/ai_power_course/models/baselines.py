"""Baselines. Read this module before believing any neural network result.

A forecast is only as good as what it beats. In energy forecasting the bar is
not "better than the mean" — the mean is a terrible predictor of a series with
a daily cycle. The bar is persistence and seasonal-naive, and a surprising
number of published deep-learning results fail to clear it.

Alignment convention
--------------------
Every function here takes the **forecast origins** and returns a
:class:`~pandas.Series` indexed by those origins, matching
:func:`ai_power_course.data.make_supervised`: row ``t`` is the moment the
forecast is made, and the value is the prediction for ``t + horizon``.

This is worth stating loudly because getting it wrong is silent. A baseline
indexed by *target* timestamp and then reindexed onto origins is off by
``horizon`` steps — it still runs, it still produces a plausible-looking error,
and it makes every model in the comparison look better than it is. The
functions below refuse to use a value from after the origin rather than
trusting the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "persistence",
    "seasonal_naive",
    "climatology",
    "baseline_suite",
]


def _lookup(history: pd.Series, origins: pd.DatetimeIndex, offset_steps: int) -> pd.Series:
    """Value of ``history`` ``offset_steps`` before each origin, indexed by origin.

    ``offset_steps`` must be non-negative: a baseline may only use the past.
    """
    if offset_steps < 0:
        raise ValueError(
            f"offset_steps={offset_steps} would read the future; a baseline may only "
            "use values at or before the forecast origin"
        )
    positions = history.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("every origin must be present in the history series")
    source = positions - offset_steps
    values = np.full(len(origins), np.nan)
    valid = source >= 0
    values[valid] = history.to_numpy()[source[valid]]
    return pd.Series(values, index=origins)


def persistence(history: pd.Series, origins: pd.DatetimeIndex, horizon: int = 1) -> pd.Series:
    """"Whatever it is now, it will still be that": ``y_hat[t + h] = y[t]``.

    The strongest trivial baseline at short horizons and the one deep models
    most often fail to beat on the first attempt. Note that it does not depend
    on ``horizon`` — the last observed value is the last observed value — which
    is exactly why its skill collapses as the horizon grows.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return _lookup(history, origins, 0).rename("persistence")


def seasonal_naive(
    history: pd.Series, origins: pd.DatetimeIndex, horizon: int = 1, season: int = 24
) -> pd.Series:
    """"The same point in the last cycle": ``y_hat[t + h] = y[t + h - season]``.

    Requires ``season >= horizon``, otherwise the value needed is itself in the
    future. The function raises rather than silently leaking, because the leak
    produces a spectacular and entirely fake result.

    Watch the identity this makes explicit: on hourly data at ``horizon=24``,
    ``season=24`` gives ``y[t]`` — which is :func:`persistence`. The two are the
    same forecast at that horizon and only at that horizon.
    """
    if season < horizon:
        raise ValueError(
            f"seasonal_naive with season={season} < horizon={horizon} would use unobserved "
            "data; use a multiple of the season that is at least the horizon"
        )
    return _lookup(history, origins, season - horizon).rename(f"seasonal_naive_{season}")


def climatology(
    train: pd.Series,
    origins: pd.DatetimeIndex,
    horizon: int = 1,
    by: tuple[str, ...] = ("hour", "dayofweek"),
) -> pd.Series:
    """Training-set average for the calendar position of the *predicted* hour.

    A static profile carrying no recent information at all. It answers: how much
    of my skill comes from knowing what is happening *now*, and how much from
    knowing what a Tuesday in January looks like?

    Note that the lookup uses the target timestamp (``origin + horizon``), since
    that is the hour being predicted — but the result is indexed by origin like
    every other baseline here. The calendar is known arbitrarily far ahead, so
    this is not a leak.
    """
    groups = [getattr(train.index, attribute) for attribute in by]
    table = train.groupby(groups).mean()
    step = origins.freq or (origins[1] - origins[0])
    targets = origins + horizon * step
    keys = list(zip(*[getattr(targets, attribute) for attribute in by], strict=True))
    fallback = float(train.mean())
    return pd.Series(
        [float(table.get(key, fallback)) for key in keys], index=origins, name="climatology"
    )


def baseline_suite(
    history: pd.Series,
    origins: pd.DatetimeIndex,
    horizon: int = 24,
    train: pd.Series | None = None,
    seasons: tuple[int, ...] = (24, 168),
) -> dict[str, pd.Series]:
    """Every baseline at once, all indexed by ``origins``.

    ``history`` must cover the origins and enough of the past to fill the
    longest season; ``train`` (defaulting to ``history``) is what the
    climatology profile is averaged over and should exclude the test period.

    Baselines that come out numerically identical are emitted once, under a name
    that says so — listing the same idea twice would make a comparison table
    look like it holds more evidence than it does.
    """
    train = history if train is None else train
    candidates: list[tuple[str, pd.Series]] = [
        (f"persistence (h={horizon})", persistence(history, origins, horizon)),
    ]
    for season in seasons:
        usable = season * int(np.ceil(horizon / season))
        candidates.append(
            (f"seasonal naive ({usable} h)", seasonal_naive(history, origins, horizon, usable))
        )
    candidates.append(
        ("climatology (hour x weekday)", climatology(train, origins, horizon))
    )

    aligned: dict[str, pd.Series] = {}
    for name, forecast in candidates:
        duplicate = next(
            (other for other, kept in aligned.items()
             if np.allclose(kept.to_numpy(), forecast.to_numpy(), equal_nan=True)),
            None,
        )
        if duplicate is not None:
            aligned[f"{duplicate} = {name}"] = aligned.pop(duplicate)
        else:
            aligned[name] = forecast
    return aligned
