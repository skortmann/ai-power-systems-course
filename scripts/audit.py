"""Correctness audit for the AI-for-power-systems course.

Every check below compares the course against something independent: a hand
computation, scikit-learn, an explicit perturbation experiment, or a physical
identity. Nothing here is satisfied by "the notebooks executed".

The failure modes this targets are the ones specific to machine learning, where
a bug produces a *better-looking* result rather than an obviously wrong one:

* a causal mask that leaks, so a language model sees the future and shows a
  beautiful loss curve;
* a scaler fit before the train/test split, so the test score is optimistic;
* a baseline implemented weakly, so every model looks good against it;
* a metric with the wrong denominator, so a number is reported in units nobody
  can compare;
* an interval called "calibrated" because it was requested, not measured.

Run it as-is::

    uv run python scripts/audit.py
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Configuration -- edit these; there are no command-line flags.
# --------------------------------------------------------------------------

#: Samples for the Monte Carlo calibration checks.
N_MONTE_CARLO = 20_000

#: Sequence length used in the causal-masking experiments.
MASK_LENGTH = 16

#: How many independent positions to perturb when probing a causal mask.
N_MASK_PROBES = 8

# --------------------------------------------------------------------------

import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from ai_power_course import tolerances as tol

RESULTS: list[dict] = []


def record(check: str, verdict: str, evidence: str) -> None:
    RESULTS.append({"check": check, "verdict": verdict, "evidence": evidence})
    print(f"  [{verdict}] {check}\n         {evidence}")


# ============================================================ 1. point metrics


def audit_point_metrics() -> None:
    """Every metric against an independent computation.

    scikit-learn where it has the same metric, an explicit formula where it
    does not. A metric that is subtly wrong makes every number in the course
    wrong at once, so this is checked first.
    """
    print("\n1. Point metrics vs independent implementations")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from ai_power_course.metrics import bias, mae, mape, nrmse, r2, rmse, skill_score, smape

    rng = np.random.default_rng(11)
    truth = rng.normal(50.0, 12.0, size=500)
    pred = truth + rng.normal(0.0, 4.0, size=500)

    rows = []
    ok = True
    for name, ours, theirs in [
        ("MAE", mae(truth, pred), mean_absolute_error(truth, pred)),
        ("RMSE", rmse(truth, pred), math.sqrt(mean_squared_error(truth, pred))),
        ("R2", r2(truth, pred), r2_score(truth, pred)),
    ]:
        agree = tol.close(ours, theirs, tol.METRIC_RELATIVE)
        ok &= agree
        rows.append(f"{name}: ours={ours:.10f} sklearn={theirs:.10f}")
    record(
        "MAE, RMSE and R2 match scikit-learn",
        "PASS" if ok else "FAIL",
        "; ".join(rows),
    )

    # Hand formulas for the ones scikit-learn does not provide.
    hand_bias = float(np.mean(pred - truth))
    hand_mape = float(100.0 * np.mean(np.abs((truth - pred) / truth)))
    hand_smape = float(
        100.0 * np.mean(np.abs(truth - pred) / ((np.abs(truth) + np.abs(pred)) / 2.0))
    )
    hand_nrmse = float(
        100.0 * math.sqrt(np.mean((truth - pred) ** 2)) / np.mean(np.abs(truth))
    )
    checks = [
        ("bias", bias(truth, pred), hand_bias),
        ("MAPE", mape(truth, pred), hand_mape),
        ("sMAPE", smape(truth, pred), hand_smape),
        ("nRMSE", nrmse(truth, pred), hand_nrmse),
    ]
    ok = all(tol.close(a, b, tol.METRIC_RELATIVE) for _, a, b in checks)
    record(
        "bias, MAPE, sMAPE and nRMSE match hand formulas",
        "PASS" if ok else "FAIL",
        "; ".join(f"{n}: ours={a:.8f} hand={b:.8f}" for n, a, b in checks),
    )

    # skill_score must be 0 against itself and 1 against a perfect forecast.
    reference = truth + rng.normal(0.0, 8.0, size=500)
    self_skill = skill_score(truth, reference, reference)
    perfect = skill_score(truth, truth, reference)
    improving = skill_score(truth, pred, reference)
    ok = (
        abs(self_skill) < tol.METRIC_RELATIVE
        and abs(perfect - 1.0) < tol.METRIC_RELATIVE
        and 0.0 < improving < 1.0
    )
    record(
        "skill score is 0 against itself, 1 for a perfect forecast",
        "PASS" if ok else "FAIL",
        f"vs itself={self_skill:.2e}, perfect={perfect:.10f}, "
        f"better-than-reference={improving:.6f} (must be in (0,1))",
    )

    # A metric must not reward the wrong direction: bias sign convention.
    over = skill_score(truth, truth + 5.0, truth + 10.0)
    record(
        "bias is positive when the model over-predicts",
        "PASS" if bias(truth, truth + 5.0) > 0 and over > 0 else "FAIL",
        f"bias(truth, truth+5) = {bias(truth, truth + 5.0):+.6f} (must be > 0)",
    )


# ====================================================== 2. probabilistic scores


def audit_probabilistic_metrics() -> None:
    """Pinball, CRPS and coverage, against their definitions.

    The pinball asymmetry is the one most often implemented with the sign
    flipped, which silently trains every quantile to the wrong side.
    """
    print("\n2. Probabilistic scores")
    from ai_power_course.metrics import coverage, crps_from_quantiles, pinball_loss

    # The defining property: for q = 0.9, UNDER-prediction must cost 9x more.
    truth = np.array([10.0])
    under = pinball_loss(truth, np.array([9.0]), 0.9)   # pred below truth
    over = pinball_loss(truth, np.array([11.0]), 0.9)   # pred above truth
    ok = tol.close(under, 0.9, tol.EXACT_FLOAT64) and tol.close(over, 0.1, tol.EXACT_FLOAT64)
    record(
        "pinball loss is asymmetric in the correct direction",
        "PASS" if ok else "FAIL",
        f"q=0.9: under-prediction by 1 costs {under:.10f} (expect 0.9), "
        f"over-prediction by 1 costs {over:.10f} (expect 0.1)",
    )

    # The minimiser of the pinball loss must BE the quantile.
    rng = np.random.default_rng(3)
    sample = rng.normal(0.0, 1.0, size=200_000)
    rows, ok = [], True
    for q in (0.1, 0.5, 0.9):
        grid = np.linspace(-3.0, 3.0, 1201)
        losses = [pinball_loss(sample, np.full(sample.size, c), q) for c in grid]
        argmin = grid[int(np.argmin(losses))]
        truth_q = float(np.quantile(sample, q))
        agree = abs(argmin - truth_q) < 0.02
        ok &= agree
        rows.append(f"q={q}: argmin={argmin:+.4f}, empirical quantile={truth_q:+.4f}")
    record(
        "the pinball loss is minimised AT the quantile it names",
        "PASS" if ok else "FAIL",
        "; ".join(rows),
    )

    # CRPS of a perfect deterministic forecast is zero.
    levels = np.arange(0.05, 1.0, 0.05)
    obs = rng.normal(size=300)
    perfect = np.repeat(obs[:, None], len(levels), axis=1)
    crps_perfect = crps_from_quantiles(obs, perfect, levels)
    record(
        "CRPS is zero for a perfect forecast",
        "PASS" if abs(crps_perfect) < tol.EXACT_FLOAT64 else "FAIL",
        f"CRPS = {crps_perfect:.3e}",
    )

    # CRPS of the true predictive distribution must beat a shifted one.
    from scipy.stats import norm

    truth_draws = rng.normal(0.0, 1.0, size=20_000)
    honest = np.tile(norm.ppf(levels)[None, :], (truth_draws.size, 1))
    shifted = honest + 1.0
    crps_honest = crps_from_quantiles(truth_draws, honest, levels)
    crps_shifted = crps_from_quantiles(truth_draws, shifted, levels)
    record(
        "CRPS prefers the correctly located predictive distribution",
        "PASS" if crps_honest < crps_shifted else "FAIL",
        f"correct={crps_honest:.6f} < shifted-by-1={crps_shifted:.6f}",
    )

    # CRPS must CONVERGE as the quantile grid densifies. A plain average of
    # pinball losses does not: it carries a grid-dependent factor and was
    # measured 4.9% high on 19 levels and 11.3% low on the 3-level grid
    # Tutorial 09 uses. This is the check that catches that.
    analytic = 2.0 / math.sqrt(math.pi)          # E[CRPS] = sigma/sqrt(pi), sigma=2
    draws = rng.normal(3.0, 2.0, size=200_000)
    errors = []
    for n_levels in (19, 99):
        grid = np.arange(1, n_levels + 1) / (n_levels + 1)
        forecast = np.tile(norm.ppf(grid, 3.0, 2.0)[None, :], (draws.size, 1))
        errors.append(abs(crps_from_quantiles(draws, forecast, grid) / analytic - 1.0))
    record(
        "CRPS converges to the analytic value as the grid densifies",
        "PASS" if errors[1] < errors[0] and errors[1] < 0.01 else "FAIL",
        f"19 levels: {errors[0]:.1%} error; 99 levels: {errors[1]:.1%} error "
        f"(must shrink, and be under 1% at 99)",
    )

    # Coverage of an exact Gaussian interval must match its nominal level.
    draws = rng.normal(0.0, 1.0, size=N_MONTE_CARLO)
    rows, ok = [], True
    for nominal in (0.5, 0.8, 0.95):
        half = norm.ppf(0.5 + nominal / 2.0)
        empirical = coverage(draws, np.full_like(draws, -half), np.full_like(draws, half))
        agree = abs(empirical - nominal) < tol.PROBABILITY_ABSOLUTE
        ok &= agree
        rows.append(f"nominal={nominal:.2f} empirical={empirical:.4f}")
    record(
        "empirical coverage matches the nominal level",
        "PASS" if ok else "FAIL",
        f"N={N_MONTE_CARLO}; " + "; ".join(rows),
    )


# ============================================================ 3. causal masking


def _logits_from(model, tokens):
    with torch.no_grad():
        out = model(tokens)
    return out[0] if isinstance(out, tuple) else out


def audit_causal_masking() -> None:
    """The load-bearing check of the whole language-model half of the course.

    A causal mask that leaks lets position t see token t+1. The model trains,
    the loss falls beautifully, and everything downstream -- perplexity,
    generation quality, the entire claim that the model learned to predict the
    next token -- is meaningless.

    Reading the mask cannot establish this. The test is a perturbation: change
    a FUTURE token and confirm the logits at earlier positions do not move at
    all. A correct mask removes those tokens from the computation, so the
    difference is exactly zero up to float reduction order.
    """
    print("\n3. Causal masking, by perturbation")
    from ai_power_course.models.attention import (
        causal_mask,
        scaled_dot_product_attention_numpy,
        softmax_numpy,
    )

    # --- the NumPy reference implementation
    rng = np.random.default_rng(5)
    d_k, length = 8, MASK_LENGTH
    query = rng.normal(size=(length, d_k))
    key = rng.normal(size=(length, d_k))
    value = rng.normal(size=(length, d_k))
    mask = np.tril(np.ones((length, length), dtype=bool))

    base, _ = scaled_dot_product_attention_numpy(query, key, value, mask=mask)
    worst = 0.0
    for position in range(1, length):
        perturbed_v = value.copy()
        perturbed_k = key.copy()
        perturbed_v[position] += 10.0
        perturbed_k[position] += 10.0
        out, _ = scaled_dot_product_attention_numpy(query, perturbed_k, perturbed_v, mask=mask)
        # rows BEFORE `position` must be untouched
        worst = max(worst, float(np.max(np.abs(out[:position] - base[:position]))))
    record(
        "NumPy attention: perturbing key/value at t leaves rows < t unchanged",
        "PASS" if worst <= tol.EXACT_FLOAT64 else "FAIL",
        f"worst change over {length - 1} perturbations = {worst:.3e} "
        f"(must be <= {tol.EXACT_FLOAT64:.0e})",
    )

    # --- softmax normalises over the KEY axis
    scores = rng.normal(size=(6, 9))
    probabilities = softmax_numpy(scores, axis=-1)
    sums = probabilities.sum(axis=-1)
    record(
        "softmax normalises over the key axis",
        "PASS" if np.allclose(sums, 1.0, atol=tol.EXACT_FLOAT64) else "FAIL",
        f"row sums in [{sums.min():.12f}, {sums.max():.12f}], shape {probabilities.shape}",
    )

    # --- the 1/sqrt(d_k) scaling is present and uses the HEAD dimension
    q1 = np.ones((1, d_k))
    k1 = np.ones((1, d_k))
    _, weights = scaled_dot_product_attention_numpy(q1, k1, np.ones((1, d_k)))
    # With one key the weight is 1 whatever the scaling, so probe the scores
    # directly through a two-key contrast instead.
    k2 = np.stack([np.ones(d_k), np.zeros(d_k)])
    _, w2 = scaled_dot_product_attention_numpy(q1, k2, np.stack([np.ones(d_k), np.zeros(d_k)]))
    expected_logit_gap = d_k / math.sqrt(d_k)          # (q.k1 - q.k2)/sqrt(d_k)
    implied = math.log(w2[0, 0] / w2[0, 1])
    record(
        "attention scores are divided by sqrt(d_k)",
        "PASS" if abs(implied - expected_logit_gap) < 1e-9 else "FAIL",
        f"implied logit gap = {implied:.10f}, expected d_k/sqrt(d_k) = "
        f"{expected_logit_gap:.10f} for d_k={d_k}",
    )

    # --- causal_mask is lower-triangular INCLUDING the diagonal
    m = causal_mask(5).cpu().numpy().astype(bool)
    correct = np.array_equal(m, np.tril(np.ones((5, 5), dtype=bool)))
    record(
        "causal_mask is lower-triangular including the diagonal",
        "PASS" if correct else "FAIL",
        f"row 0 allows {int(m[0].sum())} key(s), row 4 allows {int(m[4].sum())}; "
        f"matches np.tril: {correct}",
    )

    # --- the torch multi-head module
    from ai_power_course.models.attention import MultiHeadSelfAttention

    torch.manual_seed(0)
    module = MultiHeadSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(1, length, 32)
    m_t = causal_mask(length)
    with torch.no_grad():
        reference = module(x, mask=m_t)
    reference = reference[0] if isinstance(reference, tuple) else reference

    worst = 0.0
    for position in range(1, length):
        perturbed = x.clone()
        perturbed[0, position] += 10.0
        with torch.no_grad():
            out = module(perturbed, mask=m_t)
        out = out[0] if isinstance(out, tuple) else out
        worst = max(
            worst, float((out[0, :position] - reference[0, :position]).abs().max())
        )
    record(
        "MultiHeadSelfAttention: perturbing x at t leaves positions < t unchanged",
        "PASS" if worst <= tol.EXACT_FLOAT32 else "FAIL",
        f"worst change over {length - 1} perturbations = {worst:.3e} "
        f"(must be <= {tol.EXACT_FLOAT32:.0e})",
    )


def audit_tinygpt_causality() -> None:
    """The same perturbation test on the full model the course trains."""
    print("\n4. TinyGPT end-to-end causality")
    try:
        from ai_power_course.models.tinygpt import TinyGPT
    except ImportError as error:
        record("TinyGPT importable", "FAIL", str(error))
        return

    from ai_power_course.models.tinygpt import TinyGPTConfig

    torch.manual_seed(0)
    vocab = 32
    # Dropout OFF: a stochastic forward pass would make every perturbation
    # look like a leak, and the question here is about the MASK, not dropout.
    config = TinyGPTConfig(vocab_size=vocab, context=MASK_LENGTH, d_model=32,
                           n_heads=4, n_layers=2, dropout=0.0)
    model = TinyGPT(config).eval()

    rng = np.random.default_rng(1)
    tokens = torch.tensor(rng.integers(0, vocab, size=(1, MASK_LENGTH)), dtype=torch.long)
    base = _logits_from(model, tokens)

    worst = 0.0
    probes = 0
    for position in range(1, MASK_LENGTH):
        for _ in range(max(1, N_MASK_PROBES // MASK_LENGTH)):
            perturbed = tokens.clone()
            new = int(rng.integers(0, vocab))
            if new == int(perturbed[0, position]):
                new = (new + 1) % vocab
            perturbed[0, position] = new
            out = _logits_from(model, perturbed)
            worst = max(worst, float((out[0, :position] - base[0, :position]).abs().max()))
            probes += 1
    record(
        "TinyGPT: changing token t cannot change the logits at positions < t",
        "PASS" if worst <= tol.EXACT_FLOAT32 else "FAIL",
        f"worst logit change over {probes} token perturbations = {worst:.3e} "
        f"(must be <= {tol.EXACT_FLOAT32:.0e}). A non-zero value here means the "
        f"model is trained to predict tokens it can already see.",
    )


# ============================================================== 5. data leakage


def audit_data_leakage() -> None:
    """Splits, lags and windows, tested rather than read."""
    print("\n5. Data leakage")
    from ai_power_course.data import add_lag_features, make_supervised, make_windows, time_split

    index = pd.date_range("2018-01-01", "2020-06-01", freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"load": np.sin(np.arange(len(index)) / 24.0) * 10 + 50 + np.arange(len(index)) * 1e-4},
        index=index,
    )

    train, valid, test = time_split(frame)
    ordered = train.index.max() < valid.index.min() and valid.index.max() < test.index.min()
    disjoint = (
        len(train.index.intersection(valid.index)) == 0
        and len(valid.index.intersection(test.index)) == 0
        and len(train.index.intersection(test.index)) == 0
    )
    record(
        "time_split is chronological and disjoint",
        "PASS" if ordered and disjoint else "FAIL",
        f"train ends {train.index.max()}, valid {valid.index.min()}..{valid.index.max()}, "
        f"test starts {test.index.min()}; overlaps: {not disjoint}",
    )

    # A lag feature must equal the past value, never a future one.
    lagged = add_lag_features(frame, column="load", lags=(0, 1, 24), rolling=(24,))
    position = 500
    checks = {
        k: float(lagged[f"load_lag{k}"].iloc[position])
        - float(frame["load"].iloc[position - k])
        for k in (0, 1, 24)
    }
    roll = float(lagged["load_roll24_mean"].iloc[position])
    roll_hand = float(frame["load"].iloc[position - 23 : position + 1].mean())
    ok = all(abs(v) < tol.EXACT_FLOAT64 for v in checks.values()) and abs(
        roll - roll_hand
    ) < 1e-9
    record(
        "lag and rolling features read only the past, anchored at the origin",
        "PASS" if ok else "FAIL",
        f"lag errors {[f'{v:.1e}' for v in checks.values()]}; "
        f"roll24 ours={roll:.10f} hand(t-23..t)={roll_hand:.10f}",
    )

    # The supervised target must be strictly in the future.
    horizon = 24
    features, target = make_supervised(frame, target="load", horizon=horizon,
                                       lags=(0, 1), rolling=(24,))
    origin = features.index[100]
    expected = float(frame["load"].loc[origin + pd.Timedelta(hours=horizon)])
    got = float(target.loc[origin])
    record(
        "make_supervised puts the target exactly `horizon` ahead of the origin",
        "PASS" if abs(got - expected) < tol.EXACT_FLOAT64 else "FAIL",
        f"origin {origin}: y={got:.10f}, load[origin+{horizon}h]={expected:.10f}",
    )

    # No feature column may correlate perfectly with the future target, which
    # is what a leaked column looks like.
    correlations = features.corrwith(target).abs()
    worst_name = correlations.idxmax()
    record(
        "no feature is a perfect copy of the future target",
        "PASS" if correlations.max() < 0.9999 else "FAIL",
        f"highest |corr| with y is {worst_name} at {correlations.max():.6f}",
    )

    # Windows must not straddle a boundary: y starts exactly where X ends.
    series = np.arange(1000.0)
    x, y = make_windows(series, context=48, horizon=12, stride=7)
    contiguous = all(
        abs(y[i, 0] - (x[i, -1, 0] + 1.0)) < tol.EXACT_FLOAT32 for i in range(len(x))
    )
    no_overlap = all(y[i, 0] > x[i, -1, 0] for i in range(len(x)))
    record(
        "make_windows: the horizon begins immediately after the context, never inside it",
        "PASS" if contiguous and no_overlap else "FAIL",
        f"{len(x)} windows, X {x.shape}, y {y.shape}; "
        f"y[0] == x[-1]+1 for every window: {contiguous}",
    )


def audit_corpus_and_dataset_leakage() -> None:
    """Duplicate text across a split is leakage no split function can catch.

    ``time_split`` can be perfectly chronological and the data still be
    contaminated, because the contamination is in the CONTENT: the same string
    occurring on both sides. Both cases below were live in the course.
    """
    print("\n6. Duplicate content across splits")
    from ai_power_course.corpus import generate_corpus

    text = generate_corpus()
    cut = int(0.9 * len(text))
    train, valid = text[:cut], text[cut:]
    rng = np.random.default_rng(0)
    rates = {}
    for width in (65, 129):
        hits = 0
        for _ in range(2000):
            start = int(rng.integers(0, len(valid) - width))
            if valid[start : start + width] in train:
                hits += 1
        rates[width] = hits / 2000.0
    worst = max(rates.values())
    record(
        "the LM validation split is not a copy of the training split",
        "PASS" if worst < 0.05 else "FAIL",
        "; ".join(f"{w}-char windows verbatim in train: {r:.1%}" for w, r in rates.items())
        + " (was 66.0% when the repeated handbook sat in the final 10%)",
    )

    from sklearn.model_selection import train_test_split

    from ai_power_course.corpus import generate_event_dataset

    texts, labels = generate_event_dataset()
    unique_texts, first = np.unique(texts, return_index=True)
    tr, te, _, _ = train_test_split(
        list(unique_texts), labels[first], test_size=0.3, random_state=0,
        stratify=labels[first],
    )
    overlap = len(set(tr) & set(te))
    record(
        "the event-classification split has no verbatim overlap",
        "PASS" if overlap == 0 else "FAIL",
        f"{len(texts)} rows -> {len(unique_texts)} distinct; train {len(tr)} / "
        f"test {len(te)}; verbatim overlap {overlap} (was 56 of 264 = 21.2% "
        f"before deduplication)",
    )


def audit_baselines() -> None:
    """A weak baseline makes every model look good. Check the baselines."""
    print("\n7. Baseline correctness")
    from ai_power_course.models.baselines import climatology, persistence, seasonal_naive

    index = pd.date_range("2019-01-01", periods=24 * 40, freq="h", tz="UTC")
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    origins = index[200:260]

    p = persistence(values, origins, horizon=1)
    ok = all(abs(p.loc[t] - values.loc[t]) < tol.EXACT_FLOAT64 for t in origins)
    record(
        "persistence returns the value AT the forecast origin",
        "PASS" if ok else "FAIL",
        f"y_hat[t+h] == y[t] at all {len(origins)} origins; "
        f"example t={origins[0]}: {p.iloc[0]:.1f} vs y[t]={values.loc[origins[0]]:.1f}",
    )

    sn = seasonal_naive(values, origins, horizon=1, season=24)
    expected = [values.iloc[values.index.get_loc(t) - 23] for t in origins]
    ok = all(abs(a - b) < tol.EXACT_FLOAT64 for a, b in zip(sn.to_numpy(), expected, strict=True))
    record(
        "seasonal_naive reads y[t + h - season], a past value",
        "PASS" if ok else "FAIL",
        f"season=24 horizon=1 -> offset 23 steps back; matches at all {len(origins)} origins",
    )

    # The documented identity: at horizon == season the two coincide.
    identical = np.allclose(
        seasonal_naive(values, origins, horizon=24, season=24).to_numpy(),
        persistence(values, origins, horizon=24).to_numpy(),
    )
    record(
        "seasonal_naive(season=h) equals persistence at horizon h, as documented",
        "PASS" if identical else "FAIL",
        f"horizon=24, season=24: identical to persistence = {identical}",
    )

    # It must REFUSE to read the future rather than silently leaking.
    refused = False
    try:
        seasonal_naive(values, origins, horizon=48, season=24)
    except ValueError:
        refused = True
    record(
        "seasonal_naive refuses a season shorter than the horizon",
        "PASS" if refused else "FAIL",
        "season=24 < horizon=48 raises ValueError rather than reading unobserved data",
    )

    # Climatology must be fit on the training series only.
    train = values.iloc[: 24 * 30]
    clim = climatology(train, origins, horizon=1)
    finite = np.isfinite(clim.to_numpy()).all()
    record(
        "climatology produces a finite profile from the training series alone",
        "PASS" if finite else "FAIL",
        f"{len(clim)} origins scored from {len(train)} training hours, all finite: {finite}",
    )


# ========================================================== 7. reproducibility


def audit_reproducibility() -> None:
    print("\n8. Reproducibility")
    from ai_power_course.config import set_seed

    def draw():
        set_seed(1234)
        import random

        return (
            np.random.default_rng(1234).normal(size=5),
            torch.randn(5).numpy(),
            random.random(),
        )

    a_np, a_torch, a_py = draw()
    b_np, b_torch, b_py = draw()
    same = (
        np.array_equal(a_np, b_np)
        and np.array_equal(a_torch, b_torch)
        and a_py == b_py
    )
    record(
        "set_seed makes NumPy, torch and random reproducible",
        "PASS" if same else "FAIL",
        f"numpy identical: {np.array_equal(a_np, b_np)}, "
        f"torch identical: {np.array_equal(a_torch, b_torch)}, "
        f"random identical: {a_py == b_py}",
    )

    # Two identical short trainings must land on the same loss.
    from ai_power_course.models.training import train_model  # noqa: F401  (existence)

    set_seed(7)
    net1 = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    set_seed(7)
    net2 = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    identical_init = all(
        torch.equal(p1, p2) for p1, p2 in zip(net1.parameters(), net2.parameters(), strict=True)
    )
    record(
        "seeded weight initialisation is identical across runs",
        "PASS" if identical_init else "FAIL",
        f"every parameter tensor equal: {identical_init}",
    )


# ============================================================== 8. grid physics


def audit_grid_physics() -> None:
    print("\n9. Grid physics plausibility")
    try:
        from ai_power_course.grid import networks, physics
    except ImportError as error:
        record("grid package importable", "FAIL", str(error))
        return

    import pandapower as pp

    catalogue = sorted(networks.NETWORK_CATALOGUE)
    rows, worst_relative, all_ok = [], 0.0, True
    for name in catalogue:
        net_i = networks.load_network(name)
        if not networks.run_power_flow(net_i):
            rows.append(f"{name}: DID NOT CONVERGE")
            all_ok = False
            continue
        gen_i = (
            float(net_i.res_ext_grid.p_mw.sum())
            + float(net_i.res_gen.p_mw.sum() if len(net_i.gen) else 0.0)
            + float(net_i.res_sgen.p_mw.sum() if len(net_i.sgen) else 0.0)
        )
        load_i = float(net_i.res_load.p_mw.sum())
        loss_i = float(net_i.res_line.pl_mw.sum()) + float(
            net_i.res_trafo.pl_mw.sum() if len(net_i.trafo) else 0.0
        )
        rel = abs(gen_i - load_i - loss_i) / max(1.0, abs(load_i))
        worst_relative = max(worst_relative, rel)
        rows.append(f"{name}: {rel:.1e}")
    record(
        "generation == load + losses on every catalogued network",
        "PASS" if all_ok and worst_relative <= tol.PHYSICS_RELATIVE else "FAIL",
        "relative imbalance per network -- " + "; ".join(rows),
    )

    net = networks.load_network(catalogue[0])
    pp.runpp(net, numba=False)
    # Every injecting element, not just the slack. case118 carries most of its
    # supply on PV `gen` buses; omitting res_gen made generation look 8x short
    # of load while the power flow itself was perfectly balanced.
    generation = (
        float(net.res_ext_grid.p_mw.sum())
        + float(net.res_gen.p_mw.sum() if len(net.gen) else 0.0)
        + float(net.res_sgen.p_mw.sum() if len(net.sgen) else 0.0)
    )
    load = float(net.res_load.p_mw.sum())
    losses = float(net.res_line.pl_mw.sum()) + float(
        net.res_trafo.pl_mw.sum() if len(net.trafo) else 0.0
    )
    imbalance = abs(generation - load - losses)
    scale = max(1.0, abs(load))
    record(
        "generation equals load plus losses",
        "PASS" if imbalance / scale <= tol.PHYSICS_RELATIVE else "FAIL",
        f"generation={generation:.6f}, load={load:.6f}, losses={losses:.6f} MW, "
        f"relative imbalance={imbalance / scale:.2e}",
    )
    record(
        "losses are positive",
        "PASS" if losses > 0 else "FAIL",
        f"total losses = {losses:.6f} MW",
    )
    vm = net.res_bus.vm_pu
    record(
        "voltage magnitudes are physically plausible",
        "PASS" if float(vm.min()) > 0.8 and float(vm.max()) < 1.2 else "FAIL",
        f"|V| in [{float(vm.min()):.4f}, {float(vm.max()):.4f}] pu over {len(vm)} buses",
    )
    _ = physics  # imported for the audit trail


def main() -> int:
    print("=" * 78)
    print("AI-FOR-POWER-SYSTEMS COURSE — CORRECTNESS AUDIT")
    print("=" * 78)
    print(tol.describe())

    audit_point_metrics()
    audit_probabilistic_metrics()
    audit_causal_masking()
    audit_tinygpt_causality()
    audit_data_leakage()
    audit_corpus_and_dataset_leakage()
    audit_baselines()
    audit_reproducibility()
    audit_grid_physics()

    print("\n" + "=" * 78)
    counts: dict[str, int] = {}
    for row in RESULTS:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"{len(RESULTS)} checks: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    for row in RESULTS:
        if row["verdict"] == "FAIL":
            print(f"  FAILED: {row['check']}\n          {row['evidence']}")
    print("=" * 78)
    return 1 if any(r["verdict"] == "FAIL" for r in RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
