"""Dataset generation, loading, splitting and — above all — leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_power_course.data import (
    DEFAULT_LAGS,
    CourseDataset,
    add_calendar_features,
    add_lag_features,
    load_energy_data,
    make_supervised,
    make_windows,
    time_split,
)
from ai_power_course.synthetic import SyntheticConfig, generate_energy_frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_energy_data(source="synthetic").frame


# --- the generator ------------------------------------------------------------


def test_synthetic_is_deterministic():
    a = generate_energy_frame(SyntheticConfig(end="2017-04-01"))
    b = generate_energy_frame(SyntheticConfig(end="2017-04-01"))
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_is_physically_plausible(frame):
    assert (frame.load_mw > 0).all(), "load must be positive"
    assert (frame.pv_mw >= 0).all(), "PV generation cannot be negative"
    assert (frame.wind_mw >= 0).all(), "wind generation cannot be negative"

    # PV is exactly zero whenever the sun is below the horizon. This is the
    # property tutorials 01 and 09 build their physical-validity argument on.
    night = frame.is_night.to_numpy()
    assert night.any(), "the dataset should contain night hours"
    # The rounded clear-sky index must agree with the night flag, or a notebook
    # filtering on one will silently disagree with a notebook using the other.
    np.testing.assert_array_equal(night, (frame.clear_sky_index <= 0.0).to_numpy())
    assert frame.pv_mw.to_numpy()[night].max() == 0.0

    # Capacity factors and annual energies in a believable range for Germany.
    years = (frame.index[-1] - frame.index[0]).days / 365.25
    assert 400 < frame.load_mw.sum() / 1e6 / years < 560
    assert 0.08 < frame.pv_mw.mean() / 45_000 < 0.15
    assert 0.15 < frame.wind_mw.mean() / 55_000 < 0.30
    assert -30 < frame.temperature_c.min() < 5
    assert 25 < frame.temperature_c.max() < 45


def test_residual_load_is_consistent(frame):
    # The stored file is rounded to three decimals, so allow one rounding unit.
    expected = frame.load_mw - frame.pv_mw - frame.wind_mw
    np.testing.assert_allclose(frame.residual_load_mw, expected, atol=2e-3)


def test_dataset_carries_its_provenance():
    dataset = load_energy_data(source="synthetic")
    assert isinstance(dataset, CourseDataset)
    assert dataset.provenance.is_measurement is False
    assert "SIMULATED" in dataset.provenance.banner()
    assert "not measurements" in dataset.provenance.banner()


def test_dataset_rejects_an_unsorted_index(frame):
    with pytest.raises(ValueError, match="chronologically sorted"):
        CourseDataset(frame=frame.iloc[::-1], provenance=load_energy_data().provenance)


# --- splitting ----------------------------------------------------------------


def test_time_split_is_chronological_and_disjoint(frame):
    train, valid, test = time_split(frame)
    assert train.index.max() < valid.index.min() < valid.index.max() < test.index.min()
    assert len(train) + len(valid) + len(test) == len(frame)
    assert not train.index.intersection(test.index).size


def test_time_split_rejects_an_empty_part(frame):
    with pytest.raises(ValueError, match="empty"):
        time_split(frame, train_end="2030-01-01", valid_end="2031-01-01")


# --- leakage ------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 6, 24, 48])
def test_no_feature_uses_the_future(frame, horizon):
    """The property the whole course depends on, checked against the raw series."""
    part = frame.iloc[:5000]
    X, y = make_supervised(part, horizon=horizon)
    series = part.load_mw
    positions = series.index.get_indexer(X.index)

    for lag in DEFAULT_LAGS:
        column = f"load_mw_lag{lag}"
        expected = series.to_numpy()[positions - lag]
        np.testing.assert_allclose(X[column].to_numpy(), expected, rtol=1e-9)

    expected_target = series.to_numpy()[positions + horizon]
    np.testing.assert_allclose(y.to_numpy(), expected_target, rtol=1e-9)


def test_lag_features_reject_negative_offsets(frame):
    with pytest.raises(ValueError, match="non-negative"):
        add_lag_features(frame.iloc[:500], lags=(-1, 0))


def test_rolling_features_end_at_the_origin(frame):
    part = frame.iloc[:400]
    featured = add_lag_features(part, lags=(0,), rolling=(24,))
    position = 200
    expected = part.load_mw.iloc[position - 23 : position + 1].mean()
    assert np.isclose(featured["load_mw_roll24_mean"].iloc[position], expected)


def test_supervised_table_has_no_missing_values(frame):
    X, y = make_supervised(frame.iloc[:3000], horizon=24, exogenous=("temperature_c",))
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(X) == len(y)
    assert X.index.equals(y.index)


def test_calendar_features_are_cyclical(frame):
    featured = add_calendar_features(frame.iloc[:200])
    np.testing.assert_allclose(
        featured.hour_sin**2 + featured.hour_cos**2, 1.0, atol=1e-9
    )


# --- windowing ----------------------------------------------------------------


def test_make_windows_alignment(frame):
    values = frame.load_mw.iloc[:600]
    x, y = make_windows(values, context=48, horizon=12)
    assert x.shape == (len(values) - 48 - 12 + 1, 48, 1)
    assert y.shape == (len(values) - 48 - 12 + 1, 12)
    # Window i covers values[i : i+48]; its target is values[i+48 : i+60].
    np.testing.assert_allclose(x[3, :, 0], values.to_numpy()[3:51], rtol=1e-6)
    np.testing.assert_allclose(y[3], values.to_numpy()[51:63], rtol=1e-6)


def test_make_windows_with_covariates(frame):
    part = frame.iloc[:400]
    x, _ = make_windows(part.load_mw, context=24, horizon=6,
                        covariates=part[["temperature_c", "wind_speed_ms"]])
    assert x.shape[2] == 3
    np.testing.assert_allclose(x[0, :, 1], part.temperature_c.to_numpy()[:24], rtol=1e-6)


def test_make_windows_rejects_a_too_short_series(frame):
    with pytest.raises(ValueError, match="too short"):
        make_windows(frame.load_mw.iloc[:10], context=48, horizon=12)


def test_opsd_loader_fails_helpfully_without_a_download(monkeypatch, tmp_path):
    import ai_power_course.data as data_module

    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        data_module._load_opsd()
