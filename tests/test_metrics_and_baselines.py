"""Metrics, physical checks, and the baselines everything is measured against."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_power_course.data import load_energy_data, make_supervised, time_split
from ai_power_course.metrics import (
    classification_metrics,
    coverage,
    crps_from_quantiles,
    mae,
    physical_violations,
    pinball_loss,
    point_metrics,
    r2,
    rmse,
    skill_score,
)
from ai_power_course.models.baselines import (
    baseline_suite,
    climatology,
    persistence,
    seasonal_naive,
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_energy_data(source="synthetic").frame


# --- point metrics ------------------------------------------------------------


def test_metrics_of_a_perfect_forecast():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(truth, truth) == 0.0
    assert rmse(truth, truth) == 0.0
    assert r2(truth, truth) == 1.0


def test_rmse_punishes_outliers_more_than_mae():
    """Same MAE, very different RMSE — the reason to report both."""
    truth = np.zeros(100)
    spread = np.full(100, 0.1)          # every hour off by a little
    concentrated = np.zeros(100)
    concentrated[0] = 10.0              # one hour off by a lot
    assert mae(truth, spread) == pytest.approx(mae(truth, concentrated))
    assert rmse(truth, concentrated) > 5 * rmse(truth, spread)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        mae(np.zeros(4), np.zeros(5))


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="empty"):
        mae(np.array([]), np.array([]))


def test_skill_score_signs():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    reference = truth + 2.0
    better, worse = truth + 1.0, truth + 4.0
    assert skill_score(truth, better, reference) > 0
    assert skill_score(truth, worse, reference) < 0
    assert skill_score(truth, reference, reference) == pytest.approx(0.0)


def test_point_metrics_bundle_keys():
    truth = np.linspace(1, 10, 20)
    assert set(point_metrics(truth, truth + 0.1)) == {"MAE", "RMSE", "R2", "Bias", "MAPE_%"}


# --- probabilistic ------------------------------------------------------------


def test_pinball_loss_is_asymmetric():
    truth = np.array([10.0])
    # A high quantile should punish under-prediction much harder.
    under, over = pinball_loss(truth, np.array([8.0]), 0.9), pinball_loss(
        truth, np.array([12.0]), 0.9
    )
    assert under > over
    assert pinball_loss(truth, np.array([8.0]), 0.5) == pytest.approx(
        pinball_loss(truth, np.array([12.0]), 0.5)
    )


def test_pinball_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match="strictly between"):
        pinball_loss(np.array([1.0]), np.array([1.0]), 1.0)


def test_crps_is_zero_for_a_perfect_deterministic_forecast():
    truth = np.array([5.0, 7.0])
    levels = [0.1, 0.5, 0.9]
    forecasts = np.tile(truth.reshape(-1, 1), (1, len(levels)))
    assert crps_from_quantiles(truth, forecasts, levels) == pytest.approx(0.0)


def test_crps_prefers_a_sharper_calibrated_forecast():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, 2000)
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    from scipy.stats import norm

    sharp = np.tile(norm.ppf(levels, loc=0, scale=1.0), (2000, 1))
    wide = np.tile(norm.ppf(levels, loc=0, scale=3.0), (2000, 1))
    assert crps_from_quantiles(truth, sharp, levels) < crps_from_quantiles(truth, wide, levels)


def test_coverage_matches_the_nominal_level():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, 20000)
    from scipy.stats import norm

    lower = np.full_like(truth, norm.ppf(0.1))
    upper = np.full_like(truth, norm.ppf(0.9))
    assert coverage(truth, lower, upper) == pytest.approx(0.8, abs=0.02)


# --- classification -----------------------------------------------------------


def test_classification_metrics_on_an_imbalanced_problem():
    truth = np.zeros(100, dtype=int)
    truth[:3] = 1
    never = np.zeros(100, dtype=int)
    metrics = classification_metrics(truth, never)
    assert metrics["Accuracy"] == pytest.approx(0.97)
    assert metrics["Recall"] == 0.0
    assert metrics["F1"] == 0.0
    assert metrics["PositiveRate"] == pytest.approx(0.03)


# --- physical checks ----------------------------------------------------------


def test_physical_violations_detects_negative_generation():
    prediction = np.array([-5.0, 10.0, 20.0, -1.0])
    report = physical_violations(prediction, lower_bound=0.0)
    assert report["below_min_rate"] == 0.5
    assert report["worst_below_min"] == pytest.approx(-5.0)


def test_physical_violations_detects_night_time_pv():
    prediction = np.array([0.0, 0.0, 3.0, 100.0])
    night = np.array([True, True, True, False])
    report = physical_violations(prediction, zero_mask=night)
    assert report["nonzero_when_impossible_rate"] == pytest.approx(1 / 3)
    assert report["worst_impossible_value"] == pytest.approx(3.0)


def test_physical_violations_detects_impossible_ramps():
    prediction = np.array([0.0, 1.0, 2.0, 50.0])
    report = physical_violations(prediction, max_ramp=5.0)
    assert report["ramp_violation_rate"] == pytest.approx(1 / 3)
    assert report["worst_ramp"] == pytest.approx(48.0)


def test_a_clean_forecast_reports_no_violations():
    report = physical_violations(np.array([1.0, 2.0, 3.0]), lower_bound=0.0, max_ramp=10.0)
    assert report["below_min_rate"] == 0.0
    assert report["ramp_violation_rate"] == 0.0


# --- baselines: the alignment that used to be wrong ---------------------------


def test_persistence_predicts_the_value_at_the_origin(frame):
    part = frame.iloc[:2000]
    _, y = make_supervised(part, horizon=24)
    forecast = persistence(part.load_mw, y.index, horizon=24)
    positions = part.load_mw.index.get_indexer(y.index)
    np.testing.assert_allclose(forecast.to_numpy(), part.load_mw.to_numpy()[positions])


def test_seasonal_naive_alignment(frame):
    part = frame.iloc[:2000]
    _, y = make_supervised(part, horizon=24)
    forecast = seasonal_naive(part.load_mw, y.index, horizon=24, season=168)
    positions = part.load_mw.index.get_indexer(y.index)
    np.testing.assert_allclose(
        forecast.to_numpy(), part.load_mw.to_numpy()[positions - (168 - 24)]
    )


def test_seasonal_naive_refuses_to_leak(frame):
    part = frame.iloc[:2000]
    _, y = make_supervised(part, horizon=48)
    with pytest.raises(ValueError, match="unobserved"):
        seasonal_naive(part.load_mw, y.index, horizon=48, season=24)


def test_persistence_equals_daily_seasonal_naive_at_horizon_24(frame):
    """The identity the baseline suite exists to stop people double-counting."""
    part = frame.iloc[:2000]
    _, y = make_supervised(part, horizon=24)
    a = persistence(part.load_mw, y.index, horizon=24)
    b = seasonal_naive(part.load_mw, y.index, horizon=24, season=24)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_baseline_suite_merges_duplicates_and_is_aligned(frame):
    part = frame.iloc[:4000]
    train, _, _ = time_split(part, train_end=part.index[2000].strftime("%Y-%m-%d"),
                             valid_end=part.index[3000].strftime("%Y-%m-%d"))
    _, y = make_supervised(part, horizon=24)
    suite = baseline_suite(part.load_mw, y.index, horizon=24, train=train.load_mw)

    assert any("=" in name for name in suite), "duplicate baselines should be merged"
    for name, values in suite.items():
        assert values.index.equals(y.index), f"{name} is not aligned to the origins"
        assert values.notna().all(), f"{name} has gaps"


def test_climatology_uses_the_target_calendar_slot(frame):
    part = frame.iloc[:3000]
    _, y = make_supervised(part, horizon=24)
    forecast = climatology(part.load_mw.iloc[:2000], y.index[:100], horizon=24)
    assert forecast.index.equals(y.index[:100])
    assert forecast.notna().all()
    # A static profile should be far worse than the truth but in the right range.
    assert 0.5 < forecast.mean() / part.load_mw.mean() < 1.5


def test_learned_models_should_beat_the_baselines(frame):
    """A smoke test of the whole benchmark: gradient boosting must clear the bar."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    train, _, test = time_split(frame)
    X_train, y_train = make_supervised(train, horizon=24)
    X_test, y_test = make_supervised(test, horizon=24)
    model = HistGradientBoostingRegressor(max_iter=80, random_state=0).fit(X_train, y_train)

    suite = baseline_suite(frame.load_mw, y_test.index, horizon=24, train=train.load_mw)
    best_baseline = min(mae(y_test, values) for values in suite.values())
    assert mae(y_test, model.predict(X_test)) < best_baseline


def test_leaderboard_surfaces_the_reference_each_row_was_scored_against():
    """Skill is only comparable between rows scored against the same baseline.

    The feature-based tutorials evaluate on `make_supervised` forecast origins
    and the sequence tutorials on `make_windows` windows, which begin 168 hours
    later. Same series, same test period, different samples -- so the seasonal
    naive they each score against differs, and the Skill column cannot be read
    straight down as a ranking. The table has to SAY that rather than imply
    otherwise, which is what Reference_RMSE is for.
    """
    from ai_power_course.results import leaderboard_table, reference_groups

    frame = leaderboard_table()
    if frame.empty or "Skill" not in frame.columns:
        pytest.skip("no leaderboard entries recorded yet")

    assert "Reference_RMSE" in frame.columns, (
        "the leaderboard must expose which baseline each row was scored against"
    )
    scored = frame["Reference_RMSE"].dropna()
    assert not scored.empty

    # Recovered reference must reproduce the stored skill.
    for (_tutorial, model), reference in scored.items():
        row = frame.loc[(_tutorial, model)]
        implied = 1.0 - row["RMSE"] / reference
        assert implied == pytest.approx(row["Skill"], abs=1e-3), (
            f"{model}: Reference_RMSE does not reproduce the stored Skill"
        )

    # And the table must sort by MAE, not by Skill, precisely because more than
    # one reference is in play.
    groups = reference_groups()
    if len(groups) > 1:
        mae = frame["MAE"].to_numpy()
        assert (mae[:-1] <= mae[1:] + 1e-9).all(), (
            "with several reference baselines present the leaderboard must be "
            "ordered by MAE, which is comparable, rather than by Skill, which is not"
        )
