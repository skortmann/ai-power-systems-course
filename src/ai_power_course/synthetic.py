"""Deterministic synthetic European power-system time series.

Why this module exists
----------------------
The course needs a dataset that (a) can be committed to the repository and run
in CI, (b) is legally unencumbered, and (c) has enough real structure that the
modelling story actually works: annual and daily seasonality, weather-driven
renewables, a weekday/weekend split, holidays, and a merit-order price.

Every series here is **simulated**, not measured. It is built from physically
motivated components (solar geometry, a wind turbine power curve, a
temperature-dependent load model, a residual-load price stack) so that the
lessons transfer, but it must never be described as measurement data. The
notebooks print the provenance banner from :mod:`ai_power_course.data` for
exactly this reason.

For the real thing, :func:`ai_power_course.data.download_opsd` fetches the Open
Power System Data time-series package; the course compares the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["SyntheticConfig", "generate_energy_frame", "GERMAN_HOLIDAYS"]

# Fixed public holidays that visibly depress German load, plus the moving
# Easter/Whitsun dates for the simulated years. Kept explicit and small rather
# than pulling in a holidays dependency for six lines of effect.
GERMAN_HOLIDAYS: tuple[str, ...] = (
    # New Year / Labour day / Unity day / Christmas, every year
    "01-01", "05-01", "10-03", "12-24", "12-25", "12-26", "12-31",
)

_MOVING_HOLIDAYS: dict[int, tuple[str, ...]] = {
    # Good Friday, Easter Monday, Ascension, Whit Monday
    2016: ("2016-03-25", "2016-03-28", "2016-05-05", "2016-05-16"),
    2017: ("2017-04-14", "2017-04-17", "2017-05-25", "2017-06-05"),
    2018: ("2018-03-30", "2018-04-02", "2018-05-10", "2018-05-21"),
    2019: ("2019-04-19", "2019-04-22", "2019-05-30", "2019-06-10"),
    2020: ("2020-04-10", "2020-04-13", "2020-05-21", "2020-06-01"),
    2021: ("2021-04-02", "2021-04-05", "2021-05-13", "2021-05-24"),
    2022: ("2022-04-15", "2022-04-18", "2022-05-26", "2022-06-06"),
}


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs for the simulated system. Defaults roughly mimic Germany's scale."""

    start: str = "2017-01-01"
    end: str = "2021-01-01"
    freq: str = "h"
    latitude_deg: float = 51.0
    #: Installed capacities in MW (order-of-magnitude realistic for DE ~2019).
    pv_capacity_mw: float = 45_000.0
    wind_capacity_mw: float = 55_000.0
    #: Load model, MW.
    load_base_mw: float = 54_000.0
    load_temp_heating_mw_per_k: float = 560.0
    load_temp_cooling_mw_per_k: float = 180.0
    #: Price stack, EUR/MWh.
    price_intercept: float = 8.0
    price_slope_per_gw: float = 1.05
    noise_scale: float = 1.0
    seed: int = 20260101


# --- building blocks ---------------------------------------------------------


def _ou_process(n: int, rng: np.random.Generator, theta: float, std: float) -> np.ndarray:
    """Mean-reverting (Ornstein-Uhlenbeck / AR(1)) noise, in its stationary state.

    Weather is persistent: a cloudy hour is very likely followed by a cloudy
    hour. White noise would make the renewable series trivially unpredictable
    and would destroy the point of tutorials 03, 05 and 06.

    Parameters
    ----------
    theta:
        Mean-reversion rate per step. The correlation time is ``1 / theta``
        hours, so ``theta=0.035`` is a weather system lasting about a day.
    std:
        Stationary standard deviation of the process, in the units of the
        quantity it perturbs. Parameterising by ``std`` rather than by the
        innovation scale keeps the physical magnitudes obvious.
    """
    alpha = np.exp(-theta)
    x = np.empty(n)
    x[0] = rng.normal(0.0, std)
    innovations = rng.normal(0.0, std * np.sqrt(1.0 - alpha**2), size=n)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + innovations[i]
    return x


def _clear_sky_index(index: pd.DatetimeIndex, latitude_deg: float) -> np.ndarray:
    """Normalised clear-sky irradiance in [0, 1] from simple solar geometry.

    Declination from the day of year, hour angle from solar time; the cosine of
    the zenith angle is clipped at zero so the series is exactly zero at night.
    That hard zero matters pedagogically: a model that predicts PV at 02:00 is
    physically wrong regardless of its RMSE (tutorial 01 and 09 both use this).
    """
    day_of_year = index.dayofyear.to_numpy()
    # Solar time offset for central Europe (UTC index, ~15 deg E -> +1 h).
    solar_hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0 + 1.0
    declination = np.radians(23.45) * np.sin(2.0 * np.pi * (284.0 + day_of_year) / 365.0)
    hour_angle = np.radians(15.0 * (solar_hour - 12.0))
    lat = np.radians(latitude_deg)
    cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(
        hour_angle
    )
    clear_sky = np.clip(cos_zenith, 0.0, None)
    # Snap the twilight sliver to exactly zero. Without this, an hour with the
    # sun a hair above the horizon carries a clear-sky index of ~1e-4 and a few
    # hundred kW of PV -- which survives as a non-zero value while the index
    # itself rounds to 0.000 in the stored file. The dataset would then contradict
    # itself, and the "PV is exactly zero at night" property that tutorials 01
    # and 09 rely on would be false in the data before any model touched it.
    return np.where(clear_sky < 1e-3, 0.0, clear_sky)


def _wind_power_curve(speed_ms: np.ndarray) -> np.ndarray:
    """Aggregate wind fleet power curve, normalised to [0, 1].

    A single turbine has a sharp cut-out at 25 m/s; a whole fleet spread over a
    country does not, because the wind speed differs across sites. We therefore
    use a smoothed cubic region and a soft high-wind roll-off.
    """
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    p = np.zeros_like(speed_ms)
    ramp = (speed_ms >= cut_in) & (speed_ms < rated)
    p[ramp] = ((speed_ms[ramp] - cut_in) / (rated - cut_in)) ** 3
    p[(speed_ms >= rated) & (speed_ms < cut_out)] = 1.0
    roll_off = speed_ms >= cut_out
    p[roll_off] = np.clip(1.0 - (speed_ms[roll_off] - cut_out) / 5.0, 0.0, 1.0)
    return np.clip(p, 0.0, 1.0)


def _holiday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    fixed = np.isin(np.asarray(index.strftime("%m-%d")), np.asarray(GERMAN_HOLIDAYS))
    moving_dates = sorted({d for year in _MOVING_HOLIDAYS.values() for d in year})
    moving = np.isin(np.asarray(index.strftime("%Y-%m-%d")), np.asarray(moving_dates))
    return fixed | moving


# --- public API --------------------------------------------------------------


def generate_energy_frame(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Build the course's synthetic hourly energy dataset.

    Returns a :class:`~pandas.DataFrame` indexed by a UTC ``DatetimeIndex`` with
    columns:

    ``load_mw``
        System load. Temperature-dependent, with weekday/weekend and holiday
        effects and a double daily peak.
    ``pv_mw``, ``wind_mw``
        Renewable in-feed. Exactly zero for PV at night.
    ``residual_load_mw``
        ``load_mw - pv_mw - wind_mw``; the quantity the conventional fleet and
        the market actually have to serve.
    ``price_eur_mwh``
        Day-ahead price from a convex merit-order stack on residual load. Goes
        negative in high-renewable, low-load hours, as it does in reality.
    ``temperature_c``, ``wind_speed_ms``, ``clear_sky_index``
        Weather drivers, usable as covariates.
    ``is_night``
        True exactly where the sun is below the horizon, and therefore exactly
        where ``pv_mw`` is zero. Use this rather than comparing the rounded
        ``clear_sky_index`` against zero.
    ``is_weekend``, ``is_holiday``
        Calendar flags.

    The result is fully determined by ``config`` — same config, same numbers.
    """
    cfg = config or SyntheticConfig()
    index = pd.date_range(
        cfg.start, cfg.end, freq=cfg.freq, inclusive="left", tz="UTC", name="time"
    )
    n = len(index)
    rng = np.random.default_rng(cfg.seed)

    hour = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    is_weekend = index.dayofweek.to_numpy() >= 5
    is_holiday = _holiday_mask(index)
    off_day = is_weekend | is_holiday

    # --- weather -------------------------------------------------------------
    # Temperature: annual cycle (min early February), daily cycle, persistent noise.
    annual_temp = 9.5 - 9.0 * np.cos(2.0 * np.pi * (day_of_year - 20.0) / 365.25)
    daily_temp = 3.2 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    temp_noise = _ou_process(n, rng, theta=0.035, std=3.4 * cfg.noise_scale)
    temperature = annual_temp + daily_temp + temp_noise

    # Wind speed: windier in winter, persistent, strictly positive.
    wind_annual = 7.2 - 1.6 * np.cos(2.0 * np.pi * (day_of_year - 15.0) / 365.25)
    wind_noise = _ou_process(n, rng, theta=0.02, std=2.6 * cfg.noise_scale)
    wind_speed = np.clip(wind_annual + wind_noise, 0.0, None)

    # Cloudiness: persistent multiplicative attenuation of the clear-sky index.
    clear_sky = _clear_sky_index(index, cfg.latitude_deg)
    cloud = 1.0 / (1.0 + np.exp(-_ou_process(n, rng, theta=0.05, std=1.5 * cfg.noise_scale)))
    cloud = 0.25 + 0.75 * cloud  # never fully dark during the day

    # --- generation ----------------------------------------------------------
    is_night = clear_sky <= 0.0
    pv = cfg.pv_capacity_mw * 0.92 * clear_sky * cloud
    pv = np.where(is_night, 0.0, pv)  # hard physical zero at night
    wind = cfg.wind_capacity_mw * 0.95 * _wind_power_curve(wind_speed)

    # --- load ----------------------------------------------------------------
    # Double daily peak (morning ~08:00, evening ~19:00), damped at weekends.
    daily_shape = (
        0.115 * np.sin(2.0 * np.pi * (hour - 4.0) / 24.0)
        + 0.075 * np.sin(4.0 * np.pi * (hour - 1.5) / 24.0)
    )
    weekday_factor = np.where(off_day, 0.86, 1.0)
    # Heating below 15 degC, cooling above 22 degC.
    heating = np.clip(15.0 - temperature, 0.0, None) * cfg.load_temp_heating_mw_per_k
    cooling = np.clip(temperature - 22.0, 0.0, None) * cfg.load_temp_cooling_mw_per_k
    # Slow multi-year drift (efficiency gains vs. electrification).
    years = (index - index[0]).days.to_numpy() / 365.25
    drift = 1.0 - 0.006 * years
    load_noise = _ou_process(n, rng, theta=0.25, std=900.0 * cfg.noise_scale)
    load = (
        (cfg.load_base_mw * (1.0 + daily_shape) * weekday_factor + heating + cooling) * drift
        + load_noise
    )
    load = np.clip(load, 1_000.0, None)

    residual = load - pv - wind

    # --- price ---------------------------------------------------------------
    # Convex merit order: cheap when residual load is low, steep when it is high.
    residual_gw = residual / 1_000.0
    price = (
        cfg.price_intercept
        + cfg.price_slope_per_gw * residual_gw
        + 0.022 * np.clip(residual_gw - 45.0, 0.0, None) ** 2
        - 26.0 * np.clip(-residual_gw / 12.0, 0.0, None)  # negative-price regime
        + rng.normal(0.0, 4.5 * cfg.noise_scale, size=n)
    )

    frame = pd.DataFrame(
        {
            "load_mw": load,
            "pv_mw": pv,
            "wind_mw": wind,
            "residual_load_mw": residual,
            "price_eur_mwh": price,
            "temperature_c": temperature,
            "wind_speed_ms": wind_speed,
            "clear_sky_index": clear_sky,
            "is_night": is_night,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
        },
        index=index,
    )
    return frame.round(3)
