"""Exercise chapters 01-05: classical ML through attention.

Every ``setup_code`` block here is self-contained — it rebuilds what the chapter
needs from :mod:`ai_power_course` rather than from a variable a student was
asked to produce. That is what lets ``exercise.ipynb`` run its setup cells even
when no task has been attempted.
"""

from __future__ import annotations

from . import Chapter, Task

# --------------------------------------------------------------------------- 01

CHAPTER_01 = Chapter(
    number=1,
    title="Classical Machine Learning",
    tutorial="01_classical_machine_learning.ipynb",
    intro="""
    Before any neural network, the discipline: framing the problem, splitting
    the data so the split cannot lie to you, and building a baseline worth
    beating.

    The running problem for the whole exercise track is **day-ahead system load
    forecasting** — predict the load 24 hours after the moment the forecast is
    made. Every later chapter attacks the same problem with a different tool, so
    the numbers are comparable throughout.
    """,
    setup_code="""
    from ai_power_course.config import set_seed
    from ai_power_course.data import load_energy_data, make_supervised, time_split
    from ai_power_course.metrics import mae, rmse

    set_seed()
    dataset = load_energy_data()
    print(dataset.describe())

    frame = dataset.frame
    HORIZON = 24
    train_raw, valid_raw, test_raw = time_split(frame)
    print(f"\\ntrain {len(train_raw):,} h | validation {len(valid_raw):,} h | "
          f"test {len(test_raw):,} h")
    """,
    tasks=(
        Task(
            number="1.1",
            title="Inspect the dataset and find its structure",
            kind="analysis",
            difficulty=1,
            background="""
            A forecasting model is a bet about structure. Before choosing one,
            find out what structure is actually there. This dataset is hourly and
            has three seasonalities and a weather driver; you should be able to
            see all four.
            """,
            instruction="""
            Produce a two-panel figure: the mean load by hour of day (weekday
            versus weekend on the same axes) and the mean load by month. Then
            print the annual energy in TWh.
            """,
            requirements=(
                "Use `frame`, which is already loaded.",
                "Both panels need axis labels and a title.",
                "Group with pandas rather than looping over rows.",
            ),
            hints=(
                "`frame.index` is a tz-aware `DatetimeIndex`, so `frame.index.hour` "
                "and `frame.index.month` are available directly.",
                "`frame.groupby([a, b]).load_mw.mean().unstack()` gives you a frame "
                "with one column per group.",
            ),
            expected="""
            A double daily peak, a visibly lower weekend profile, and a winter
            maximum. The annual energy should be on the order of a few hundred
            TWh — this dataset is scaled to resemble Germany.
            """,
            exercise_code="""
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

            # TODO 1: mean load by hour of day, split into weekday and weekend.
            #         Hint: group by [frame.index.dayofweek >= 5, frame.index.hour]
            hourly = ____
            axes[0].plot(____, ____, label="weekday")
            axes[0].plot(____, ____, label="weekend")
            axes[0].set_xlabel(____)
            axes[0].set_ylabel(____)
            axes[0].set_title(____)
            axes[0].legend()

            # TODO 2: mean load by month
            monthly = ____
            axes[1].plot(monthly.index, monthly.to_numpy(), marker="o")
            axes[1].set_xlabel(____)
            axes[1].set_ylabel(____)
            axes[1].set_title(____)

            fig.tight_layout()
            plt.show()

            # TODO 3: annual energy in TWh (the data covers several whole years)
            years = (frame.index[-1] - frame.index[0]).days / 365.25
            annual_twh = ____
            print(f"annual load energy: {annual_twh:.0f} TWh/a")
            """,
            solution_code="""
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

            hourly = (
                frame.groupby([frame.index.dayofweek >= 5, frame.index.hour])
                .load_mw.mean()
                .unstack(0)
            )
            axes[0].plot(hourly.index, hourly[False], label="weekday")
            axes[0].plot(hourly.index, hourly[True], label="weekend")
            axes[0].set_xlabel("hour of day (UTC)")
            axes[0].set_ylabel("mean load [MW]")
            axes[0].set_title("Daily profile")
            axes[0].legend()

            monthly = frame.groupby(frame.index.month).load_mw.mean()
            axes[1].plot(monthly.index, monthly.to_numpy(), marker="o")
            axes[1].set_xlabel("month")
            axes[1].set_ylabel("mean load [MW]")
            axes[1].set_title("Annual cycle")

            fig.tight_layout()
            plt.show()

            years = (frame.index[-1] - frame.index[0]).days / 365.25
            annual_twh = frame.load_mw.sum() / 1e6 / years
            print(f"annual load energy: {annual_twh:.0f} TWh/a")
            print(f"weekday/weekend ratio at 08:00: "
                  f"{hourly.loc[8, False] / hourly.loc[8, True]:.2f}")
            """,
            explanation="""
            `groupby` on two keys then `unstack(0)` turns the boolean weekend key
            into two columns, which is what lets both profiles go on one axis
            without a loop.

            Dividing the summed MW by `1e6` gives TWh because the data is hourly:
            one MW held for one hour is one MWh. If the resolution were 15
            minutes you would need a factor of 4, and forgetting it is one of the
            most common unit errors in energy analysis.
            """,
        ),
        Task(
            number="1.2",
            title="Build a leakage-free temporal split",
            kind="coding",
            difficulty=1,
            background="""
            A random train/test split of a time series lets the model interpolate
            between neighbouring hours it has already memorised. The score goes
            up, the model does not get better, and nothing warns you.

            Tutorial 01 measured the damage: a shuffled split flattered a
            1-nearest-neighbour model by 30%.
            """,
            instruction="""
            Build the supervised tables for all three splits using
            `make_supervised`, then verify by hand that no feature uses
            information from after the forecast origin.
            """,
            requirements=(
                "Produce `X_train, y_train, X_valid, y_valid, X_test, y_test`.",
                "Use `horizon=HORIZON` and pass `temperature_c` as an exogenous feature.",
                "The verification must compare against the raw series, not against "
                "another function from the same module.",
            ),
            hints=(
                "`make_supervised(part, horizon=..., exogenous=(...))` returns `(X, y)`.",
                "Row `t` is the forecast origin: `load_mw_lag0` is the load *at* `t`, "
                "and `y` is the load at `t + horizon`.",
            ),
            expected="""
            Three aligned tables with no missing values, and three assertions that
            pass. If an assertion fails, the features and the target are
            misaligned — fix that before training anything.
            """,
            exercise_code="""
            import numpy as np

            EXOGENOUS = ("temperature_c",)

            # TODO 1: build the three supervised tables
            X_train, y_train = ____
            X_valid, y_valid = ____
            X_test, y_test = ____

            print(f"X_train {X_train.shape} | X_valid {X_valid.shape} | X_test {X_test.shape}")

            # TODO 2: verify the alignment against the raw series.
            #         Pick one origin and check lag0, lag24 and the target.
            probe = X_train.index[1000]
            series = train_raw.load_mw
            position = series.index.get_loc(probe)

            assert np.isclose(X_train.loc[probe, "load_mw_lag0"], ____)
            assert np.isclose(X_train.loc[probe, "load_mw_lag24"], ____)
            assert np.isclose(y_train.loc[probe], ____)
            print("alignment verified against the raw series")
            """,
            solution_code="""
            import numpy as np

            EXOGENOUS = ("temperature_c",)

            X_train, y_train = make_supervised(train_raw, horizon=HORIZON, exogenous=EXOGENOUS)
            X_valid, y_valid = make_supervised(valid_raw, horizon=HORIZON, exogenous=EXOGENOUS)
            X_test, y_test = make_supervised(test_raw, horizon=HORIZON, exogenous=EXOGENOUS)

            print(f"X_train {X_train.shape} | X_valid {X_valid.shape} | X_test {X_test.shape}")

            probe = X_train.index[1000]
            series = train_raw.load_mw
            position = series.index.get_loc(probe)

            assert np.isclose(X_train.loc[probe, "load_mw_lag0"], series.iloc[position])
            assert np.isclose(X_train.loc[probe, "load_mw_lag24"], series.iloc[position - 24])
            assert np.isclose(y_train.loc[probe], series.iloc[position + HORIZON])
            print("alignment verified against the raw series")
            """,
            check_code="""
            assert not X_train.isna().any().any(), "features contain NaN"
            assert len(X_train) == len(y_train), "X and y are not aligned"
            assert X_train.index.max() < X_test.index.min(), "the splits overlap in time"
            print("Basic checks passed.")
            """,
            explanation="""
            The convention that makes this work is that lags count from the
            *forecast origin*, not from the timestamp being predicted. `lag0` is
            the most recent observation the forecaster actually has.

            Counting from the predicted timestamp is the classic leak: at a
            24-hour horizon, "the value one hour before the target" is not
            available when the forecast is made. The feature table would still
            build, the model would still train, and the error would be
            impossively good.
            """,
        ),
        Task(
            number="1.3",
            depends_on=("1.2",),
            title="Persistence and seasonal-naive baselines",
            kind="coding",
            difficulty=1,
            background="""
            A forecast is only as good as what it beats, and "better than the
            mean" is not an achievement on a series with a daily cycle.

            Note an identity that catches people out: on hourly data at a
            24-hour horizon, persistence and a 24-hour seasonal naive are the
            *same* forecast.
            """,
            instruction="""
            Implement a persistence forecast — the predicted load at the target
            time equals the load observed at the forecast origin — and a weekly
            seasonal naive. Score both.
            """,
            requirements=(
                "Both must return a `pandas.Series` indexed by the forecast "
                "origins in `y_test.index`.",
                "Neither may use a value from after the origin.",
                "Report MAE and RMSE for each.",
            ),
            hints=(
                "`frame.load_mw.index.get_indexer(origins)` converts timestamps to "
                "integer positions you can offset.",
                "For a weekly naive at horizon h, the value you want sits "
                "`168 - h` steps before the origin.",
            ),
            expected="""
            The weekly naive should clearly beat persistence: knowing the weekday
            matters more at a 24-hour horizon than knowing the last value. Both
            should be far better than predicting the training mean.
            """,
            exercise_code="""
            import pandas as pd

            history = frame.load_mw
            origins = y_test.index
            positions = history.index.get_indexer(origins)

            def persistence_forecast(history, origins, horizon):
                \"\"\"Predict y[t + horizon] as the value observed at t.\"\"\"
                # TODO: return a Series indexed by `origins`
                raise NotImplementedError

            def weekly_naive_forecast(history, origins, horizon, season=168):
                \"\"\"Predict y[t + horizon] as y[t + horizon - season].\"\"\"
                # TODO: work out the offset from the origin, then index
                raise NotImplementedError

            baseline = {
                "persistence": persistence_forecast(history, origins, HORIZON),
                "seasonal naive (168 h)": weekly_naive_forecast(history, origins, HORIZON),
                "training mean": pd.Series(y_train.mean(), index=origins),
            }
            for name, prediction in baseline.items():
                print(f"{name:24s} MAE {mae(y_test, prediction):8,.1f} MW   "
                      f"RMSE {rmse(y_test, prediction):8,.1f} MW")
            """,
            solution_code="""
            import pandas as pd

            history = frame.load_mw
            origins = y_test.index
            positions = history.index.get_indexer(origins)

            def persistence_forecast(history, origins, horizon):
                \"\"\"Predict y[t + horizon] as the value observed at t.\"\"\"
                positions = history.index.get_indexer(origins)
                return pd.Series(history.to_numpy()[positions], index=origins)

            def weekly_naive_forecast(history, origins, horizon, season=168):
                \"\"\"Predict y[t + horizon] as y[t + horizon - season].\"\"\"
                if season < horizon:
                    raise ValueError("season must be at least the horizon, or this leaks")
                positions = history.index.get_indexer(origins)
                return pd.Series(history.to_numpy()[positions - (season - horizon)], index=origins)

            baseline = {
                "persistence": persistence_forecast(history, origins, HORIZON),
                "seasonal naive (168 h)": weekly_naive_forecast(history, origins, HORIZON),
                "training mean": pd.Series(y_train.mean(), index=origins),
            }
            for name, prediction in baseline.items():
                print(f"{name:24s} MAE {mae(y_test, prediction):8,.1f} MW   "
                      f"RMSE {rmse(y_test, prediction):8,.1f} MW")
            """,
            check_code="""
            for _name, _p in baseline.items():
                assert _p.index.equals(y_test.index), f"{_name} is not aligned to the origins"
                assert _p.notna().all(), f"{_name} has gaps"
            print("Basic checks passed.")
            """,
            explanation="""
            The guard in `weekly_naive_forecast` matters. With `season < horizon`
            the value you would need is itself in the future, and the "baseline"
            becomes an oracle that no model can beat. Raising is better than
            returning a spectacular number.

            Persistence does not depend on the horizon at all — the last observed
            value is the last observed value — which is exactly why its skill
            collapses as the horizon grows.
            """,
        ),
        Task(
            number="1.4",
            depends_on=("1.2", "1.3"),
            title="Linear regression and gradient boosting",
            kind="coding",
            difficulty=2,
            background="""
            Two hypothesis classes on identical features. The linear model is
            interpretable and cannot represent the V-shaped temperature response;
            the tree ensemble can, and needs no feature scaling.
            """,
            instruction="""
            Fit a linear regression and a `HistGradientBoostingRegressor` on the
            training split, then evaluate both on the test split alongside the
            baselines from task 1.3.
            """,
            requirements=(
                "Scale the features for the linear model with a `Pipeline`, so the "
                "scaler is fitted on training data only.",
                "Use `random_state=0` for the boosting model so the result is reproducible.",
                "Collect everything into one sorted results table.",
            ),
            hints=(
                "`make_pipeline(StandardScaler(), LinearRegression())` keeps the "
                "scaler inside the model, which is what makes cross-validation honest.",
                "`HistGradientBoostingRegressor(max_iter=..., learning_rate=...)` "
                "needs no scaling at all — trees compare, they do not add.",
            ),
            expected="""
            Both learned models should beat every baseline comfortably. Gradient
            boosting should beat the linear model, because the temperature
            response and the hour x weekday interaction are exactly what trees
            represent well.
            """,
            exercise_code="""
            import pandas as pd
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            set_seed()

            # TODO 1: a scaled linear model
            linear = make_pipeline(____, ____)
            linear.fit(____, ____)

            # TODO 2: gradient boosting
            boosting = HistGradientBoostingRegressor(
                max_iter=____,
                learning_rate=____,
                random_state=0,
            )
            boosting.fit(____, ____)

            predictions = dict(baseline)
            predictions["linear regression"] = ____
            predictions["gradient boosting"] = ____

            results = pd.DataFrame(
                {name: {"MAE": mae(y_test, p), "RMSE": rmse(y_test, p)}
                 for name, p in predictions.items()}
            ).T.sort_values("MAE")
            display(results.round(1))
            """,
            solution_code="""
            import pandas as pd
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            set_seed()

            linear = make_pipeline(StandardScaler(), LinearRegression())
            linear.fit(X_train, y_train)

            boosting = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                random_state=0,
            )
            boosting.fit(X_train, y_train)

            predictions = dict(baseline)
            predictions["linear regression"] = linear.predict(X_test)
            predictions["gradient boosting"] = boosting.predict(X_test)

            results = pd.DataFrame(
                {name: {"MAE": mae(y_test, p), "RMSE": rmse(y_test, p)}
                 for name, p in predictions.items()}
            ).T.sort_values("MAE")
            display(results.round(1))
            """,
            check_code="""
            assert results.index[0] in {"gradient boosting", "linear regression"}, \\
                "a learned model should top the table — check the feature alignment"
            assert results.MAE.min() < results.loc["seasonal naive (168 h)", "MAE"], \\
                "the learned models must beat the weekly naive baseline"
            print("Basic checks passed.")
            """,
            explanation="""
            The `Pipeline` is not cosmetic. Fitting a `StandardScaler` on the full
            dataset before splitting leaks the test set's mean and variance into
            training. It is a small leak, it is almost never caught in review, and
            keeping the scaler inside the estimator makes it structurally
            impossible.

            Gradient boosting needs no scaling because a decision tree only ever
            asks "is this feature above that threshold?" — a monotone rescaling
            changes nothing.
            """,
        ),
        Task(
            number="1.5",
            title="Why does a good model sometimes lose to persistence?",
            kind="reflection",
            difficulty=2,
            background="""
            In this notebook the learned models won. In published energy
            forecasting results they frequently do not, and the M4 competition
            found the same across thousands of series.
            """,
            instruction="""
            Give three distinct mechanisms by which a well-implemented machine
            learning model can fail to beat persistence on a real forecasting
            task. For each, say what you would measure to detect it.
            """,
            expected="""
            Good answers separate *problem* properties (the horizon is short, the
            series is near-random-walk) from *method* failures (distribution
            shift between train and test, a leak in the baseline's favour,
            over-tuned hyper-parameters).
            """,
            answer_template="""
            **Mechanism 1.** …

            *How I would detect it:* …

            **Mechanism 2.** …

            *How I would detect it:* …

            **Mechanism 3.** …

            *How I would detect it:* …
            """,
            answer="""
            **Mechanism 1 — the horizon is too short for anything else to matter.**
            At a one-step horizon a load series is close to a random walk: the
            best estimate of the next value really is the current one, and the
            learnable signal is a small fraction of the variance.
            *Detect it:* plot error against lead time. If the model and
            persistence converge as the horizon shrinks, this is the explanation,
            and the honest report is skill at each horizon rather than one
            aggregate number.

            **Mechanism 2 — distribution shift between training and test.**
            The model has fitted a relationship that no longer holds: new
            embedded generation, a tariff change, a pandemic, a mild winter.
            Persistence has nothing to be wrong about, so it degrades gracefully
            while the model degrades sharply.
            *Detect it:* score both on rolling windows through the test period. A
            model that wins early and loses late is drifting, not broken. Compare
            the feature distributions between the splits.

            **Mechanism 3 — the evaluation is not measuring what you think.**
            Either the baseline is handicapped (misaligned by one step, which
            makes it look worse than it is) or the model is flattered (a leaked
            feature, a scaler fitted on everything, hyper-parameters tuned on the
            test set). A model that beats a broken baseline has demonstrated
            nothing.
            *Detect it:* verify the baseline's alignment against the raw series,
            as task 1.2 does; hold out a split that is touched exactly once; and
            check that the reported gap survives cross-validation, since the
            fold-to-fold spread on this dataset is around 9% — larger than many
            published improvements.
            """,
        ),
    ),
)

# --------------------------------------------------------------------------- 02

CHAPTER_02 = Chapter(
    number=2,
    title="Neural Networks",
    tutorial="02_neural_networks.ipynb",
    intro="""
    The same forecasting problem, with the non-linearity learned instead of
    engineered. You will write a neuron and a gradient step by hand before
    touching `autograd`, because `loss.backward()` should be a convenience, not
    a mystery.
    """,
    setup_code="""
    import numpy as np
    import torch
    from torch import nn

    from ai_power_course.config import scaled, set_seed
    from ai_power_course.data import load_energy_data, make_supervised, time_split
    from ai_power_course.metrics import mae, rmse
    from ai_power_course.models.training import Standardizer, make_loader

    set_seed()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    frame = load_energy_data().frame
    HORIZON = 24
    train_raw, valid_raw, test_raw = time_split(frame)
    X_train, y_train = make_supervised(train_raw, horizon=HORIZON, exogenous=("temperature_c",))
    X_valid, y_valid = make_supervised(valid_raw, horizon=HORIZON, exogenous=("temperature_c",))
    X_test, y_test = make_supervised(test_raw, horizon=HORIZON, exogenous=("temperature_c",))

    # Networks need standardised inputs AND targets, fitted on training data only.
    feature_scaler = Standardizer().fit(X_train.to_numpy())
    target_scaler = Standardizer().fit(y_train.to_numpy().reshape(-1, 1))
    Xs_train = feature_scaler.transform(X_train.to_numpy())
    Xs_valid = feature_scaler.transform(X_valid.to_numpy())
    Xs_test = feature_scaler.transform(X_test.to_numpy())
    ys_train = target_scaler.transform(y_train.to_numpy().reshape(-1, 1))
    ys_valid = target_scaler.transform(y_valid.to_numpy().reshape(-1, 1))

    N_FEATURES = Xs_train.shape[1]
    print(f"{N_FEATURES} standardised features, {len(Xs_train):,} training rows")
    """,
    tasks=(
        Task(
            number="2.1",
            title="A neuron, a loss and one gradient step, in NumPy",
            kind="coding",
            difficulty=1,
            background="""
            A neuron is a weighted sum followed by a non-linearity:

            $$y = \\sigma(\\mathbf{w}^\\top \\mathbf{x} + b)$$

            Training it is: measure the error, work out which direction reduces
            it, take a small step that way.
            """,
            instruction="""
            Implement the forward pass of a single neuron, the mean squared
            error, and one step of gradient descent on the weights — all in
            NumPy, with the derivative written out by hand.
            """,
            requirements=(
                "`neuron(x, w, b)` applies a sigmoid and accepts `x` of shape "
                "`(n_samples, n_features)`, returning shape `(n_samples,)`.",
                "`mse(y_true, y_pred)` returns a scalar.",
                "`gradient_step` returns the updated `(w, b)`.",
                "Use NumPy only. No PyTorch in this task.",
            ),
            hints=(
                "For a linear output (no sigmoid) the MSE gradient is "
                "`dL/dw = (2/n) * X.T @ (y_pred - y_true)`.",
                "Build the sigmoid with `1 / (1 + np.exp(-np.clip(z, -500, 500)))` "
                "so a large negative `z` does not overflow.",
            ),
            expected="""
            The loss printed after the step should be lower than the loss before
            it. If it rises, the sign of the gradient is flipped or the learning
            rate is far too large.
            """,
            exercise_code="""
            def sigmoid(z):
                # TODO: numerically stable sigmoid
                raise NotImplementedError

            def neuron(x, w, b):
                \"\"\"One neuron: weighted sum, then a sigmoid.\"\"\"
                # TODO
                raise NotImplementedError

            def mse(y_true, y_pred):
                # TODO
                raise NotImplementedError

            def gradient_step(x, y_true, w, b, learning_rate=0.1):
                \"\"\"One step of gradient descent on a LINEAR neuron (no sigmoid).\"\"\"
                y_pred = x @ w + b
                n = len(y_true)
                # TODO: derivatives of the MSE with respect to w and b
                grad_w = ____
                grad_b = ____
                return w - learning_rate * grad_w, b - learning_rate * grad_b

            rng = np.random.default_rng(0)
            x_demo = rng.normal(size=(64, 3))
            y_demo = x_demo @ np.array([2.0, -1.0, 0.5]) + 0.3
            w, b = np.zeros(3), 0.0

            print(f"loss before: {mse(y_demo, x_demo @ w + b):.4f}")
            for _ in range(50):
                w, b = gradient_step(x_demo, y_demo, w, b, learning_rate=0.1)
            print(f"loss after : {mse(y_demo, x_demo @ w + b):.4f}")
            print(f"recovered weights: {w.round(3)} (true: [2.0, -1.0, 0.5])")
            """,
            solution_code="""
            def sigmoid(z):
                return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

            def neuron(x, w, b):
                \"\"\"One neuron: weighted sum, then a sigmoid.\"\"\"
                return sigmoid(x @ w + b)

            def mse(y_true, y_pred):
                return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

            def gradient_step(x, y_true, w, b, learning_rate=0.1):
                \"\"\"One step of gradient descent on a LINEAR neuron (no sigmoid).\"\"\"
                y_pred = x @ w + b
                n = len(y_true)
                residual = y_pred - y_true
                grad_w = (2.0 / n) * (x.T @ residual)
                grad_b = (2.0 / n) * residual.sum()
                return w - learning_rate * grad_w, b - learning_rate * grad_b

            rng = np.random.default_rng(0)
            x_demo = rng.normal(size=(64, 3))
            y_demo = x_demo @ np.array([2.0, -1.0, 0.5]) + 0.3
            w, b = np.zeros(3), 0.0

            print(f"loss before: {mse(y_demo, x_demo @ w + b):.4f}")
            for _ in range(50):
                w, b = gradient_step(x_demo, y_demo, w, b, learning_rate=0.1)
            print(f"loss after : {mse(y_demo, x_demo @ w + b):.4f}")
            print(f"recovered weights: {w.round(3)} (true: [2.0, -1.0, 0.5])")
            """,
            check_code="""
            _x = np.array([[1.0, 2.0]])
            _w, _b = np.array([0.5, -0.25]), 0.1
            assert np.isclose(neuron(_x, _w, _b)[0], 1 / (1 + np.exp(-0.1)), atol=1e-9)
            assert np.isclose(mse([1.0, 2.0], [1.0, 3.0]), 0.5)
            assert np.allclose(w, [2.0, -1.0, 0.5], atol=0.05), "gradient descent did not converge"
            print("Basic checks passed.")
            """,
            explanation="""
            The `np.clip` inside the sigmoid is not paranoia: `np.exp(1000)`
            overflows to `inf` and the whole array becomes `nan`, silently, on the
            first badly-scaled input.

            Note that `gradient_step` deliberately drops the sigmoid. With a
            sigmoid output and a squared-error loss the gradient carries an extra
            `sigma'(z)` factor, which is at most `0.25` and vanishes for large
            `|z|` — that is the saturation problem that made deep sigmoid networks
            untrainable and pushed the field to ReLU.
            """,
        ),
        Task(
            number="2.2",
            title="An MLP in PyTorch",
            kind="coding",
            difficulty=2,
            background="""
            Stacking neurons is only worth anything if there is a non-linearity
            between the layers. Without one, a composition of linear maps is a
            linear map, and a fifty-layer network has exactly the expressive power
            of linear regression.
            """,
            instruction="""
            Implement `LoadMLP` as an `nn.Module` with two hidden layers, ReLU
            activations and dropout, then confirm it produces the right output
            shape.
            """,
            requirements=(
                "`LoadMLP(n_features, hidden=(128, 64), dropout=0.1)`.",
                "`forward` takes `(batch, n_features)` and returns `(batch, 1)`.",
                "Use `nn.Sequential` inside the module.",
            ),
            hints=(
                "The last layer maps to 1 output and must NOT be followed by an "
                "activation — the target is an unbounded real number.",
                "Build the layer list in a loop so the hidden sizes stay configurable.",
            ),
            expected="""
            A parameter count in the low tens of thousands, and an output of shape
            `(batch, 1)`.
            """,
            exercise_code="""
            class LoadMLP(nn.Module):
                \"\"\"A multi-layer perceptron for day-ahead load forecasting.\"\"\"

                def __init__(self, n_features, hidden=(128, 64), dropout=0.1):
                    super().__init__()
                    layers = []
                    in_features = n_features
                    # TODO: for each hidden width, append Linear -> ReLU -> Dropout
                    for width in hidden:
                        layers += [____, ____, ____]
                        in_features = ____
                    # TODO: the output layer (no activation)
                    layers.append(____)
                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    # TODO
                    pass

            set_seed()
            model = LoadMLP(N_FEATURES)
            print(model)
            print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
            """,
            solution_code="""
            class LoadMLP(nn.Module):
                \"\"\"A multi-layer perceptron for day-ahead load forecasting.\"\"\"

                def __init__(self, n_features, hidden=(128, 64), dropout=0.1):
                    super().__init__()
                    layers = []
                    in_features = n_features
                    for width in hidden:
                        layers += [nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)]
                        in_features = width
                    layers.append(nn.Linear(in_features, 1))
                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            set_seed()
            model = LoadMLP(N_FEATURES)
            print(model)
            print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
            """,
            check_code="""
            _out = model(torch.randn(7, N_FEATURES))
            assert _out.shape == (7, 1), f"expected (7, 1), got {tuple(_out.shape)}"
            assert any(isinstance(m, nn.ReLU) for m in model.modules()), \\
                "without a non-linearity this is just linear regression"
            print("Basic checks passed.")
            """,
            explanation="""
            `in_features = width` inside the loop is the line people forget: each
            layer's input width is the previous layer's output width, not the
            original feature count.

            Dropout belongs after the activation, and it is active only in
            `model.train()` mode — `model.eval()` turns it off. Forgetting the
            mode switch makes evaluation noisy and is a common source of
            "my validation score is worse than training for no reason".
            """,
        ),
        Task(
            number="2.3",
            depends_on=("2.2",),
            title="Write the training loop",
            kind="coding",
            difficulty=2,
            background="""
            Four lines, in a specific order, repeated: zero the gradients,
            forward, backward, step. Getting the order wrong produces a model
            that trains but learns the wrong thing.
            """,
            instruction="""
            Complete the training loop and train the MLP for a few epochs,
            tracking training and validation loss.
            """,
            requirements=(
                "Use `nn.MSELoss()` and `torch.optim.AdamW`.",
                "Switch between `model.train()` and `model.eval()` correctly.",
                "Evaluate under `torch.no_grad()`.",
                "Return the two loss curves.",
            ),
            hints=(
                "`optimizer.zero_grad(set_to_none=True)` must come *before* "
                "`loss.backward()`, or you accumulate gradients across batches.",
                "`make_loader(X, y, batch_size=..., shuffle=True)` is already imported.",
            ),
            expected="""
            Both curves should fall. If the validation curve turns upward while
            the training curve keeps falling, that gap is overfitting — and it is
            what early stopping exists to catch.
            """,
            exercise_code="""
            def train_epoch(model, loader, loss_fn, optimizer):
                model.train()
                total, count = 0.0, 0
                for features, target in loader:
                    prediction = model(features)
                    loss = loss_fn(prediction, target)
                    # TODO: the four lines, in the right order
                    ____
                    ____
                    ____
                    total += float(loss.detach()) * len(features)
                    count += len(features)
                return total / count

            @torch.no_grad()
            def evaluate(model, loader, loss_fn):
                # TODO: eval mode, then average the loss over the loader
                raise NotImplementedError

            set_seed()
            model = LoadMLP(N_FEATURES)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            train_loader = make_loader(Xs_train, ys_train, batch_size=64, shuffle=True)
            valid_loader = make_loader(Xs_valid, ys_valid, batch_size=256)

            EPOCHS = scaled(full=12, fast=2)
            history = {"train": [], "validation": []}
            for epoch in range(EPOCHS):
                history["train"].append(train_epoch(model, train_loader, loss_fn, optimizer))
                history["validation"].append(evaluate(model, valid_loader, loss_fn))
                print(f"epoch {epoch + 1:2d}  train {history['train'][-1]:.5f}  "
                      f"validation {history['validation'][-1]:.5f}")
            """,
            solution_code="""
            def train_epoch(model, loader, loss_fn, optimizer):
                model.train()
                total, count = 0.0, 0
                for features, target in loader:
                    prediction = model(features)
                    loss = loss_fn(prediction, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(features)
                    count += len(features)
                return total / count

            @torch.no_grad()
            def evaluate(model, loader, loss_fn):
                model.eval()
                total, count = 0.0, 0
                for features, target in loader:
                    loss = loss_fn(model(features), target)
                    total += float(loss) * len(features)
                    count += len(features)
                return total / count

            set_seed()
            model = LoadMLP(N_FEATURES)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            train_loader = make_loader(Xs_train, ys_train, batch_size=64, shuffle=True)
            valid_loader = make_loader(Xs_valid, ys_valid, batch_size=256)

            EPOCHS = scaled(full=12, fast=2)
            history = {"train": [], "validation": []}
            for epoch in range(EPOCHS):
                history["train"].append(train_epoch(model, train_loader, loss_fn, optimizer))
                history["validation"].append(evaluate(model, valid_loader, loss_fn))
                print(f"epoch {epoch + 1:2d}  train {history['train'][-1]:.5f}  "
                      f"validation {history['validation'][-1]:.5f}")
            """,
            check_code="""
            assert history["train"][-1] < history["train"][0], "the training loss did not fall"
            assert len(history["validation"]) == EPOCHS
            print("Basic checks passed.")
            """,
            explanation="""
            `zero_grad` first, because PyTorch *accumulates* gradients by design
            (which is what makes gradient accumulation across micro-batches
            possible). Skip it and every batch trains on the sum of all gradients
            so far.

            `@torch.no_grad()` on the evaluation function does two things: it
            stops the autograd graph being built, which roughly halves the memory,
            and it makes it impossible to accidentally train on validation data.
            """,
        ),
        Task(
            number="2.4",
            depends_on=("2.3",),
            title="Does the network beat gradient boosting?",
            kind="analysis",
            difficulty=2,
            background="""
            Tutorial 02 found the two within a few percent of each other — a gap
            smaller than the fold-to-fold spread of cross-validation on this
            dataset.
            """,
            instruction="""
            Evaluate the trained MLP on the test set in megawatts, compare it
            against gradient boosting trained on the same features, and report
            both the accuracy gap and the training-time gap.
            """,
            requirements=(
                "Invert the target standardisation before computing MAE — a "
                "score in standardised units is not interpretable.",
                "Time both fits.",
                "Print the relative difference in MAE with an explicit direction.",
            ),
            hints=(
                "`target_scaler.inverse_transform(...)` takes the model output "
                "back to megawatts.",
                "Put the model in `eval()` mode before predicting.",
            ),
            expected="""
            A few percent between them either way, and a much larger factor in
            training time. The point of the comparison is the *ratio of effort to
            payoff*, not the winner.
            """,
            exercise_code="""
            import time

            from sklearn.ensemble import HistGradientBoostingRegressor

            # TODO 1: MLP predictions in MW
            model.eval()
            with torch.no_grad():
                mlp_scaled = model(torch.tensor(Xs_test, dtype=torch.float32)).numpy()
            mlp_prediction = ____

            # TODO 2: gradient boosting on the same features, timed
            started = time.perf_counter()
            boosting = HistGradientBoostingRegressor(max_iter=200, random_state=0)
            boosting.fit(____, ____)
            boosting_seconds = time.perf_counter() - started
            boosting_prediction = ____

            print(f"MLP               MAE {mae(y_test, mlp_prediction):8,.1f} MW")
            print(f"gradient boosting MAE {mae(y_test, boosting_prediction):8,.1f} MW"
                  f"   ({boosting_seconds:.1f}s to fit)")

            gap = mae(y_test, mlp_prediction) / mae(y_test, boosting_prediction) - 1
            print(f"\\nThe MLP's MAE is {abs(gap):.1%} "
                  f"{'higher' if gap > 0 else 'lower'} than gradient boosting's.")
            """,
            solution_code="""
            import time

            from sklearn.ensemble import HistGradientBoostingRegressor

            model.eval()
            with torch.no_grad():
                mlp_scaled = model(torch.tensor(Xs_test, dtype=torch.float32)).numpy()
            mlp_prediction = target_scaler.inverse_transform(mlp_scaled).ravel()

            started = time.perf_counter()
            boosting = HistGradientBoostingRegressor(max_iter=200, random_state=0)
            boosting.fit(X_train, y_train)
            boosting_seconds = time.perf_counter() - started
            boosting_prediction = boosting.predict(X_test)

            print(f"MLP               MAE {mae(y_test, mlp_prediction):8,.1f} MW")
            print(f"gradient boosting MAE {mae(y_test, boosting_prediction):8,.1f} MW"
                  f"   ({boosting_seconds:.1f}s to fit)")

            gap = mae(y_test, mlp_prediction) / mae(y_test, boosting_prediction) - 1
            print(f"\\nThe MLP's MAE is {abs(gap):.1%} "
                  f"{'higher' if gap > 0 else 'lower'} than gradient boosting's.")
            print("\\nThe reason to adopt neural networks is not this benchmark. It is that")
            print("a tree ensemble stores thresholds over YOUR features: there is no hidden")
            print("layer to reuse, nothing to transfer, and no way to train it without labels.")
            """,
            check_code="""
            assert mlp_prediction.shape == y_test.to_numpy().shape, \\
                "predictions and targets have different shapes"
            assert mae(y_test, mlp_prediction) < 5000, \\
                "MAE looks like standardised units — did you invert the target scaler?"
            print("Basic checks passed.")
            """,
            explanation="""
            The shape assertion earns its place: `model(...)` returns `(n, 1)` and
            `y_test` is `(n,)`. NumPy will happily broadcast those into an
            `(n, n)` matrix and hand you a meaningless average, so `.ravel()` is
            load-bearing.

            The magnitude assertion catches the other frequent slip: forgetting
            `inverse_transform` and reporting an MAE of 0.3 "megawatts".
            """,
        ),
    ),
)

# --------------------------------------------------------------------------- 03

CHAPTER_03 = Chapter(
    number=3,
    title="RNNs and LSTMs",
    tutorial="03_rnns_and_lstms.ipynb",
    intro="""
    Until now the models saw a *table* whose columns happened to be called
    `lag24`. Here they see the sequence itself. That change is what lets a model
    accept a longer history, a different site, or a different variable without
    being rebuilt.
    """,
    setup_code="""
    import numpy as np
    import torch
    from torch import nn

    from ai_power_course.config import scaled, set_seed
    from ai_power_course.data import load_energy_data, time_split
    from ai_power_course.metrics import mae, rmse
    from ai_power_course.models.training import Standardizer, TrainConfig, make_loader, train_model

    set_seed()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    frame = load_energy_data().frame
    train_raw, valid_raw, test_raw = time_split(frame)

    CONTEXT, HORIZON = 168, 24
    # Stride the training windows: consecutive windows overlap by 167 of 168
    # hours, so every fourth origin keeps full coverage at a quarter of the cost.
    TRAIN_STRIDE = scaled(full=4, fast=24)
    print(f"context {CONTEXT} h, horizon {HORIZON} h, training stride {TRAIN_STRIDE}")
    """,
    tasks=(
        Task(
            number="3.1",
            title="Slice a series into supervised windows",
            kind="coding",
            difficulty=2,
            background="""
            A sequence model consumes fixed-length windows: `context` steps of
            history in, `horizon` steps of future out. Building them is pure index
            arithmetic, and getting it wrong by one step is the most common bug in
            the whole field.
            """,
            instruction="""
            Implement `create_sequences`, returning inputs of shape
            `(n_windows, context, 1)` and targets of shape `(n_windows, horizon)`.
            """,
            requirements=(
                "Window `i` covers `series[i : i + context]`.",
                "Its target is `series[i + context : i + context + horizon]`.",
                "Support a `stride` so windows can be subsampled.",
                "Return float32 arrays.",
            ),
            hints=(
                "The last valid start index is `len(series) - context - horizon`.",
                "`np.stack([...])` over a list comprehension is clear and fast enough here.",
            ),
            expected="""
            The assertion cell should pass, and the window count should be roughly
            `(len(series) - context - horizon) / stride`.
            """,
            exercise_code="""
            def create_sequences(series, context, horizon, stride=1):
                \"\"\"Slice a 1-D series into (context, horizon) supervised windows.

                Returns
                -------
                x : (n_windows, context, 1) float32
                y : (n_windows, horizon)    float32
                \"\"\"
                values = np.asarray(series, dtype=np.float32).reshape(-1)
                # TODO 1: the last index at which a full window plus its target fits
                last_start = ____
                if last_start < 0:
                    raise ValueError("series is too short for this context and horizon")
                starts = np.arange(0, last_start + 1, stride)
                # TODO 2: build x and y
                x = ____
                y = ____
                return x, y

            x_train, y_train_seq = create_sequences(
                train_raw.load_mw, CONTEXT, HORIZON, stride=TRAIN_STRIDE)
            x_valid, y_valid_seq = create_sequences(
                valid_raw.load_mw, CONTEXT, HORIZON, stride=TRAIN_STRIDE)
            x_test, y_test_seq = create_sequences(test_raw.load_mw, CONTEXT, HORIZON, stride=1)

            print(f"x_train {x_train.shape}  y_train {y_train_seq.shape}")
            print(f"x_test  {x_test.shape}  y_test  {y_test_seq.shape}")
            """,
            solution_code="""
            def create_sequences(series, context, horizon, stride=1):
                \"\"\"Slice a 1-D series into (context, horizon) supervised windows.

                Returns
                -------
                x : (n_windows, context, 1) float32
                y : (n_windows, horizon)    float32
                \"\"\"
                values = np.asarray(series, dtype=np.float32).reshape(-1)
                last_start = len(values) - context - horizon
                if last_start < 0:
                    raise ValueError("series is too short for this context and horizon")
                starts = np.arange(0, last_start + 1, stride)
                x = np.stack([values[s : s + context] for s in starts])[:, :, None]
                y = np.stack([values[s + context : s + context + horizon] for s in starts])
                return x.astype(np.float32), y.astype(np.float32)

            x_train, y_train_seq = create_sequences(
                train_raw.load_mw, CONTEXT, HORIZON, stride=TRAIN_STRIDE)
            x_valid, y_valid_seq = create_sequences(
                valid_raw.load_mw, CONTEXT, HORIZON, stride=TRAIN_STRIDE)
            x_test, y_test_seq = create_sequences(test_raw.load_mw, CONTEXT, HORIZON, stride=1)

            print(f"x_train {x_train.shape}  y_train {y_train_seq.shape}")
            print(f"x_test  {x_test.shape}  y_test  {y_test_seq.shape}")
            """,
            check_code="""
            _v = np.arange(100, dtype=np.float32)
            _x, _y = create_sequences(_v, context=10, horizon=3)
            assert _x.shape == (88, 10, 1), f"got {_x.shape}, expected (88, 10, 1)"
            assert _y.shape == (88, 3), f"got {_y.shape}, expected (88, 3)"
            assert np.allclose(_x[5, :, 0], np.arange(5, 15)), "window contents are misaligned"
            assert np.allclose(_y[5], np.arange(15, 18)), "targets are misaligned"
            print("Basic checks passed.")
            """,
            explanation="""
            The trailing `[:, :, None]` adds the channel axis. PyTorch's recurrent
            and attention layers expect `(batch, sequence, features)`, and a
            univariate series has one feature — but the axis still has to be
            there, or `nn.LSTM` reads your sequence length as a feature count.

            `last_start = len(values) - context - horizon` is inclusive, hence the
            `+ 1` in `arange`. Off-by-one here silently drops the final window,
            which is exactly the most recent data you have.
            """,
        ),
        Task(
            number="3.2",
            title="Tensor shapes through a recurrent model",
            kind="analysis",
            difficulty=1,
            background="""
            Most time lost to sequence models is lost to shapes. Predicting them
            before running anything is a skill worth ten debugging sessions.
            """,
            instruction="""
            Fill in the expected shape at each stage of a forward pass, then check
            your answers against a real `nn.LSTM`.
            """,
            requirements=(
                "Write each expected shape as a tuple before running the check.",
                "Explain in one line why the hidden state has no sequence axis.",
            ),
            hints=(
                "`nn.LSTM(..., batch_first=True)` returns "
                "`(output, (h_n, c_n))`.",
                "`h_n` has shape `(num_layers, batch, hidden_size)` regardless of "
                "`batch_first`.",
            ),
            expected="""
            All four assertions pass. The output keeps the sequence axis; the final
            hidden state does not, because it is a summary of the whole sequence.
            """,
            exercise_code="""
            BATCH, HIDDEN = 8, 32

            # TODO: predict each shape as a tuple, then run the cell.
            expected_input_shape = ____          # what goes into the LSTM
            expected_output_shape = ____         # the per-step outputs
            expected_hidden_shape = ____         # the final hidden state h_n
            expected_prediction_shape = ____     # after a Linear(HIDDEN, HORIZON) head

            lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, num_layers=1, batch_first=True)
            head = nn.Linear(HIDDEN, HORIZON)
            sample = torch.randn(BATCH, CONTEXT, 1)

            output, (h_n, c_n) = lstm(sample)
            prediction = head(h_n[-1])

            assert tuple(sample.shape) == expected_input_shape
            assert tuple(output.shape) == expected_output_shape
            assert tuple(h_n.shape) == expected_hidden_shape
            assert tuple(prediction.shape) == expected_prediction_shape
            print("All four shapes correct.")
            """,
            solution_code="""
            BATCH, HIDDEN = 8, 32

            expected_input_shape = (BATCH, CONTEXT, 1)
            expected_output_shape = (BATCH, CONTEXT, HIDDEN)
            expected_hidden_shape = (1, BATCH, HIDDEN)      # (num_layers, batch, hidden)
            expected_prediction_shape = (BATCH, HORIZON)

            lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, num_layers=1, batch_first=True)
            head = nn.Linear(HIDDEN, HORIZON)
            sample = torch.randn(BATCH, CONTEXT, 1)

            output, (h_n, c_n) = lstm(sample)
            prediction = head(h_n[-1])

            assert tuple(sample.shape) == expected_input_shape
            assert tuple(output.shape) == expected_output_shape
            assert tuple(h_n.shape) == expected_hidden_shape
            assert tuple(prediction.shape) == expected_prediction_shape
            print("All four shapes correct.")
            print("\\nh_n has no sequence axis because it is the state AFTER the last step —")
            print("one vector summarising the whole window, which is what the head decodes.")
            """,
            explanation="""
            `batch_first=True` swaps the first two axes of the *input and output*
            but deliberately not of `h_n` and `c_n`, which keep
            `(num_layers, batch, hidden)`. That inconsistency is a genuine wart in
            the PyTorch API and it catches everyone once.

            `h_n[-1]` takes the last layer's final state. For a single-layer LSTM
            it is equivalent to `output[:, -1, :]`; for a stacked one it is not,
            and the difference matters.
            """,
        ),
        Task(
            number="3.3",
            depends_on=("3.1",),
            title="Build and train an LSTM forecaster",
            kind="coding",
            difficulty=2,
            background="""
            The LSTM's cell state is an *additive* path through time, which is why
            a gradient can survive hundreds of steps instead of decaying
            geometrically. That is the whole reason it displaced the vanilla RNN.
            """,
            instruction="""
            Implement `LoadLSTM` as a sequence-to-one model that decodes the final
            hidden state into the whole horizon, then train it on the standardised
            windows.
            """,
            requirements=(
                "`LoadLSTM(n_channels, hidden_size, horizon)`.",
                "`forward` takes `(batch, context, n_channels)` and returns "
                "`(batch, horizon)`.",
                "Train with the package's `train_model`, which handles early "
                "stopping and restores the best weights.",
            ),
            hints=(
                "Standardise using statistics from the *training* windows only.",
                "Decode `h_n[-1]` with a `nn.Linear(hidden_size, horizon)`.",
            ),
            expected="""
            A validation loss that falls and then flattens. Early stopping should
            fire before the epoch budget runs out.
            """,
            exercise_code="""
            scaler = Standardizer().fit(x_train.reshape(-1, 1))
            xs_train = scaler.transform(x_train)
            xs_valid = scaler.transform(x_valid)
            xs_test = scaler.transform(x_test)

            target_scaler = Standardizer().fit(y_train_seq)
            ys_train = target_scaler.transform(y_train_seq)
            ys_valid = target_scaler.transform(y_valid_seq)

            class LoadLSTM(nn.Module):
                def __init__(self, n_channels=1, hidden_size=64, horizon=HORIZON):
                    super().__init__()
                    # TODO: an LSTM and a linear head
                    self.lstm = ____
                    self.head = ____

                def forward(self, x):
                    # TODO: run the LSTM, take the final hidden state, decode it
                    ____
                    return ____

            set_seed()
            lstm_model = LoadLSTM()
            history = train_model(
                lstm_model,
                make_loader(xs_train, ys_train, batch_size=64, shuffle=True),
                make_loader(xs_valid, ys_valid, batch_size=256),
                loss_fn=nn.MSELoss(),
                config=TrainConfig(epochs=scaled(full=25, fast=2), learning_rate=2e-3,
                                   patience=5, verbose=False),
            )
            print(f"best epoch {history.best_epoch}, "
                  f"validation MSE {history.best_val_loss:.5f}, {history.seconds:.0f}s")
            """,
            solution_code="""
            scaler = Standardizer().fit(x_train.reshape(-1, 1))
            xs_train = scaler.transform(x_train)
            xs_valid = scaler.transform(x_valid)
            xs_test = scaler.transform(x_test)

            target_scaler = Standardizer().fit(y_train_seq)
            ys_train = target_scaler.transform(y_train_seq)
            ys_valid = target_scaler.transform(y_valid_seq)

            class LoadLSTM(nn.Module):
                def __init__(self, n_channels=1, hidden_size=64, horizon=HORIZON):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=n_channels, hidden_size=hidden_size,
                        num_layers=1, batch_first=True,
                    )
                    self.head = nn.Linear(hidden_size, horizon)

                def forward(self, x):
                    _output, (h_n, _c_n) = self.lstm(x)
                    return self.head(h_n[-1])

            set_seed()
            lstm_model = LoadLSTM()
            history = train_model(
                lstm_model,
                make_loader(xs_train, ys_train, batch_size=64, shuffle=True),
                make_loader(xs_valid, ys_valid, batch_size=256),
                loss_fn=nn.MSELoss(),
                config=TrainConfig(epochs=scaled(full=25, fast=2), learning_rate=2e-3,
                                   patience=5, verbose=False),
            )
            print(f"best epoch {history.best_epoch}, "
                  f"validation MSE {history.best_val_loss:.5f}, {history.seconds:.0f}s")
            """,
            check_code="""
            _out = lstm_model(torch.randn(3, CONTEXT, 1))
            assert _out.shape == (3, HORIZON), f"got {tuple(_out.shape)}, expected (3, {HORIZON})"
            assert history.train_loss[-1] < history.train_loss[0], "training loss did not fall"
            print("Basic checks passed.")
            """,
            explanation="""
            Decoding the *whole* horizon in one shot ("direct" multi-horizon) is a
            deliberate choice over predicting one step and feeding it back
            ("recursive"). Direct forecasting avoids compounding its own errors;
            recursive forecasting reuses one model for any horizon but drifts.
            Compare their per-lead-time curves and the difference is unmistakable.

            The two separate scalers matter: the input scaler is fitted on the
            flattened window values, the target scaler on the horizon matrix.
            Reusing one for both works here only because both are the same
            quantity — it would be wrong the moment you add covariates.
            """,
        ),
        Task(
            number="3.4",
            depends_on=("3.1", "3.3"),
            title="Where does the LSTM actually fail?",
            kind="analysis",
            difficulty=3,
            background="""
            An aggregate MAE is an average over very different regimes. Tutorial
            03 tested two hypotheses about *when* the model struggles, and one of
            them was wrong — which is worth more than the one that held.
            """,
            instruction="""
            Score the LSTM against persistence on the test windows, then test
            whether the error is predicted by (a) how volatile the recent past was
            and (b) how unusual the target hour is for its calendar slot.
            """,
            requirements=(
                "Report MAE at `t+24` for both the LSTM and persistence.",
                "Compute a correlation for each hypothesis.",
                "State which hypothesis survived, using the numbers.",
            ),
            hints=(
                "Recent volatility: the largest absolute hour-to-hour change in the "
                "last 24 hours of the context window.",
                "Unusualness: the absolute deviation of the target from the "
                "training mean for that (hour, weekday) slot.",
            ),
            expected="""
            One of the two correlations should be clearly stronger. Report the one
            that failed as well — a negative result about your own hypothesis is a
            result.
            """,
            exercise_code="""
            import pandas as pd

            lstm_model.eval()
            with torch.no_grad():
                scaled_prediction = lstm_model(torch.tensor(xs_test, dtype=torch.float32)).numpy()
            lstm_prediction = target_scaler.inverse_transform(scaled_prediction)

            truth = y_test_seq[:, -1]                  # the load at t+24
            lstm_point = lstm_prediction[:, -1]
            persistence_point = x_test[:, -1, 0]       # the load at the origin

            print(f"LSTM        MAE {mae(truth, lstm_point):8,.1f} MW")
            print(f"persistence MAE {mae(truth, persistence_point):8,.1f} MW")

            errors = np.abs(lstm_point - truth)

            # TODO 1: recent volatility — largest |change| in the last 24 h of context
            recent_volatility = ____

            # TODO 2: unusualness — |target - training mean for that (hour, weekday)|
            target_index = test_raw.index[CONTEXT + HORIZON - 1 :][: len(truth)]
            profile = train_raw.load_mw.groupby(
                [train_raw.index.hour, train_raw.index.dayofweek]).mean()
            expected_load = np.array([
                profile.get((h, d), float(train_raw.load_mw.mean()))
                for h, d in zip(target_index.hour, target_index.dayofweek, strict=True)
            ])
            unusualness = ____

            for label, driver in [("recent volatility", recent_volatility),
                                  ("unusualness of the target", unusualness)]:
                r = np.corrcoef(driver, errors)[0, 1]
                print(f"{label:28s} r = {r:+.3f}")
            """,
            solution_code="""
            import pandas as pd

            lstm_model.eval()
            with torch.no_grad():
                scaled_prediction = lstm_model(torch.tensor(xs_test, dtype=torch.float32)).numpy()
            lstm_prediction = target_scaler.inverse_transform(scaled_prediction)

            truth = y_test_seq[:, -1]                  # the load at t+24
            lstm_point = lstm_prediction[:, -1]
            persistence_point = x_test[:, -1, 0]       # the load at the origin

            print(f"LSTM        MAE {mae(truth, lstm_point):8,.1f} MW")
            print(f"persistence MAE {mae(truth, persistence_point):8,.1f} MW")

            errors = np.abs(lstm_point - truth)

            recent_volatility = np.abs(np.diff(x_test[:, -24:, 0], axis=1)).max(axis=1)

            target_index = test_raw.index[CONTEXT + HORIZON - 1 :][: len(truth)]
            profile = train_raw.load_mw.groupby(
                [train_raw.index.hour, train_raw.index.dayofweek]).mean()
            expected_load = np.array([
                profile.get((h, d), float(train_raw.load_mw.mean()))
                for h, d in zip(target_index.hour, target_index.dayofweek, strict=True)
            ])
            unusualness = np.abs(truth - expected_load)

            for label, driver in [("recent volatility", recent_volatility),
                                  ("unusualness of the target", unusualness)]:
                r = np.corrcoef(driver, errors)[0, 1]
                low = errors[driver < np.quantile(driver, 0.25)].mean()
                high = errors[driver > np.quantile(driver, 0.75)].mean()
                print(f"{label:28s} r = {r:+.3f}   "
                      f"MAE {low:6,.0f} -> {high:6,.0f} MW across quartiles")

            print("\\nThe volatility hypothesis does NOT survive: volatile and predictable")
            print("are different things, and the most volatile stretches here are winter")
            print("weekdays with a large but highly regular daily swing. Unusualness does")
            print("survive: the model is a very good climatology-plus-recent-history")
            print("machine, and it fails exactly where that description stops fitting.")
            """,
            check_code="""
            from ai_power_course.config import fast_mode

            assert recent_volatility.shape == errors.shape
            assert unusualness.shape == errors.shape
            assert np.isfinite(errors).all(), "some predictions are not finite"
            if fast_mode():
                print("Reduced configuration: skipping the accuracy check — two epochs")
                print("is not enough training to beat persistence.")
            else:
                assert mae(truth, lstm_point) < mae(truth, persistence_point), \\
                    "the LSTM should beat persistence at a 24 h horizon"
            print("Basic checks passed.")
            """,
            explanation="""
            Reporting the hypothesis that failed is the point of the task. The
            intuitive diagnostic — "the model struggles after a turbulent
            stretch" — is wrong on this data, and only measuring both told us so.

            Note the quartile comparison alongside the correlation. A correlation
            coefficient is a single number about a linear relationship; comparing
            the mean error in the top and bottom quartile shows the *size* of the
            effect, which is what decides whether it matters operationally.
            """,
        ),
    ),
)

# --------------------------------------------------------------------------- 04

CHAPTER_04 = Chapter(
    number=4,
    title="Representation Learning",
    tutorial="04_representation_learning.ipynb",
    intro="""
    The hinge of the course. Everything so far started from a task. Here you
    train a model with **no labels and no task**, then discover its internal
    representation is useful for a question it was never shown.

    That is the mechanism behind every foundation model in chapters 09 and 10.
    """,
    setup_code="""
    import numpy as np
    import torch
    from torch import nn

    from ai_power_course.config import fast_mode, scaled, set_seed
    from ai_power_course.data import load_energy_data, time_split
    from ai_power_course.metrics import mae, r2

    set_seed()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    frame = load_energy_data().frame
    train_raw, valid_raw, test_raw = time_split(frame)

    def daily_profiles(part, column="residual_load_mw"):
        \"\"\"Reshape a split into per-day profiles, normalised to shape only.\"\"\"
        series = part[column]
        complete = [d for d, g in series.groupby(series.index.date) if len(g) == 24]
        matrix = np.stack([series[series.index.date == d].to_numpy() for d in complete])
        level = matrix.mean(axis=1, keepdims=True)
        spread = matrix.std(axis=1, keepdims=True) + 1e-8
        normalised = ((matrix - level) / spread).astype(np.float32)
        import pandas as pd
        return normalised[:, :, None], pd.to_datetime(complete)

    profiles_train, dates_train = daily_profiles(train_raw)
    profiles_valid, dates_valid = daily_profiles(valid_raw)
    profiles_test, dates_test = daily_profiles(test_raw)
    print(f"pretraining profiles {profiles_train.shape} — and not one label among them")
    """,
    tasks=(
        Task(
            number="4.1",
            title="Mask part of a profile",
            kind="coding",
            difficulty=2,
            background="""
            Self-supervision manufactures a supervised problem out of the data's
            own structure: hide part of the input and predict it from the rest.

            *Which* part you hide is a modelling decision. Scattered single hours
            can be interpolated from their neighbours; a contiguous block cannot,
            and a contiguous block is what a communication outage actually looks
            like.
            """,
            instruction="""
            Implement both masking strategies and a masked reconstruction loss
            that scores **only the hidden positions**.
            """,
            requirements=(
                "`random_mask(shape, ratio)` returns a boolean tensor, `True` = hidden.",
                "`block_mask(shape, block)` hides exactly one contiguous run per sample.",
                "`masked_loss(prediction, target, mask)` ignores visible positions.",
                "All three accept a `torch.Generator` for reproducibility.",
            ),
            hints=(
                "For the block mask, draw one start per sample and compare against "
                "`torch.arange(length)` broadcast over the batch.",
                "Weight the squared error by the mask and divide by the number of "
                "masked *elements*, not by the batch size.",
            ),
            expected="""
            The block mask should hide exactly `block` positions per sample, and
            they should be consecutive. The loss on a perfect reconstruction
            should be zero even when most positions are wrong outside the mask.
            """,
            exercise_code="""
            def random_mask(shape, ratio=0.3, generator=None):
                \"\"\"Independent Bernoulli mask. True means hidden.\"\"\"
                # TODO
                raise NotImplementedError

            def block_mask(shape, block=6, generator=None):
                \"\"\"Hide exactly one contiguous run of `block` positions per sample.\"\"\"
                batch, length = shape
                # TODO 1: one random start per sample, in [0, length - block]
                starts = ____
                # TODO 2: compare a position row against the starts
                positions = torch.arange(length).unsqueeze(0)
                return ____

            def masked_loss(prediction, target, mask, eps=1e-8):
                \"\"\"MSE over the masked positions only.\"\"\"
                # TODO: weight by the mask, then normalise by the number of masked values
                raise NotImplementedError

            gen = torch.Generator().manual_seed(0)
            m_random = random_mask((100, 24), ratio=0.3, generator=gen)
            m_block = block_mask((100, 24), block=6, generator=gen)
            print(f"random mask hides {m_random.float().mean():.1%} of positions")
            print(f"block mask hides  {m_block.sum(dim=1).float().mean():.1f} positions per day")
            """,
            solution_code="""
            def random_mask(shape, ratio=0.3, generator=None):
                \"\"\"Independent Bernoulli mask. True means hidden.\"\"\"
                if not 0.0 <= ratio < 1.0:
                    raise ValueError("ratio must lie in [0, 1)")
                return torch.rand(shape, generator=generator) < ratio

            def block_mask(shape, block=6, generator=None):
                \"\"\"Hide exactly one contiguous run of `block` positions per sample.\"\"\"
                batch, length = shape
                if not 0 < block <= length:
                    raise ValueError("block must fit inside the profile")
                starts = torch.randint(0, length - block + 1, (batch,), generator=generator)
                positions = torch.arange(length).unsqueeze(0)
                starts = starts.unsqueeze(1)
                return (positions >= starts) & (positions < starts + block)

            def masked_loss(prediction, target, mask, eps=1e-8):
                \"\"\"MSE over the masked positions only.\"\"\"
                weights = mask.unsqueeze(-1).to(prediction.dtype)
                squared_error = (prediction - target) ** 2 * weights
                return squared_error.sum() / (weights.sum() * prediction.shape[-1] + eps)

            gen = torch.Generator().manual_seed(0)
            m_random = random_mask((100, 24), ratio=0.3, generator=gen)
            m_block = block_mask((100, 24), block=6, generator=gen)
            print(f"random mask hides {m_random.float().mean():.1%} of positions")
            print(f"block mask hides  {m_block.sum(dim=1).float().mean():.1f} positions per day")
            """,
            check_code="""
            _g = torch.Generator().manual_seed(1)
            _b = block_mask((50, 24), block=6, generator=_g)
            assert (_b.sum(dim=1) == 6).all(), "each sample must hide exactly `block` positions"
            for _row in _b:
                assert (torch.where(_row)[0].diff() == 1).all(), "the block must be contiguous"
            _p, _t = torch.zeros(2, 4, 1), torch.ones(2, 4, 1)
            _m = torch.tensor([[True, False, False, False], [False, False, False, True]])
            assert abs(float(masked_loss(_p, _t, _m)) - 1.0) < 1e-6, \\
                "the loss must ignore visible positions"
            print("Basic checks passed.")
            """,
            explanation="""
            Scoring only the hidden positions is what stops the model winning by
            learning the identity function on the parts it can already see. Score
            everything and the loss falls beautifully while the representation
            learns nothing.

            The division by `prediction.shape[-1]` accounts for the channel axis:
            `weights.sum()` counts masked *positions*, and each carries
            `n_channels` values. Getting that wrong scales the loss by a constant,
            which is harmless alone but silently changes the balance if you ever
            add a second loss term.
            """,
        ),
        Task(
            number="4.2",
            depends_on=("4.1",),
            title="A masked autoencoder for daily profiles",
            kind="coding",
            difficulty=2,
            background="""
            A plain autoencoder can cheat: with a wide enough bottleneck it
            approximates the identity. Masking removes that escape route — you
            cannot copy what you cannot see.
            """,
            instruction="""
            Build an encoder that compresses a 24-hour profile to a small latent
            vector and a decoder that expands it back, then pretrain them on the
            masking objective from task 4.1.
            """,
            requirements=(
                "`ProfileAutoencoder(length=24, latent_dim=8)`.",
                "`forward(x, mask)` replaces the masked positions with a *learned* "
                "mask token before encoding.",
                "`embed(x)` returns the latent code with no masking, in eval mode.",
                "Pretraining must use no labels of any kind.",
            ),
            hints=(
                "`nn.Parameter(torch.zeros(1))` gives you a learnable scalar to "
                "substitute at masked positions — BERT's `[MASK]`, one number wide.",
                "`torch.where(mask.unsqueeze(-1), token.expand_as(x), x)` does the "
                "substitution without a loop.",
            ),
            expected="""
            The reconstruction loss should fall by an order of magnitude. The
            reconstructions plotted afterwards should follow the shape of the
            hidden stretch, not just interpolate a straight line across it.
            """,
            exercise_code="""
            LATENT = 8

            class ProfileAutoencoder(nn.Module):
                def __init__(self, length=24, latent_dim=LATENT, hidden=64):
                    super().__init__()
                    self.length = length
                    # TODO 1: encoder — flatten, then down to latent_dim
                    self.encoder = nn.Sequential(
                        nn.Flatten(),
                        ____,
                        nn.GELU(),
                        ____,
                    )
                    # TODO 2: decoder — latent_dim back up to `length`
                    self.decoder = nn.Sequential(
                        ____,
                        nn.GELU(),
                        ____,
                        nn.Unflatten(1, (length, 1)),
                    )
                    # TODO 3: the learned mask token
                    self.mask_token = ____

                def forward(self, x, mask=None):
                    if mask is not None:
                        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
                    return self.decoder(self.encoder(x))

                @torch.no_grad()
                def embed(self, x):
                    self.eval()
                    return self.encoder(x)

            set_seed()
            autoencoder = ProfileAutoencoder()
            optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=2e-3)
            gen = torch.Generator().manual_seed(0)
            inputs = torch.tensor(profiles_train)

            pretrain_curve = []
            for epoch in range(scaled(full=60, fast=5)):
                autoencoder.train()
                order = torch.randperm(len(inputs), generator=gen)
                epoch_loss, seen = 0.0, 0
                for start in range(0, len(order), 64):
                    batch = inputs[order[start : start + 64]]
                    half = len(batch) // 2
                    mask = torch.cat([
                        random_mask((half, 24), ratio=0.3, generator=gen),
                        block_mask((len(batch) - half, 24), block=6, generator=gen),
                    ])
                    # TODO 4: forward, loss, backward, step
                    ____
                    ____
                    ____
                    ____
                    ____
                    epoch_loss += float(loss.detach()) * len(batch)
                    seen += len(batch)
                pretrain_curve.append(epoch_loss / seen)
            print(f"masked reconstruction loss: "
                  f"{pretrain_curve[0]:.4f} -> {pretrain_curve[-1]:.4f}")
            """,
            solution_code="""
            LATENT = 8

            class ProfileAutoencoder(nn.Module):
                def __init__(self, length=24, latent_dim=LATENT, hidden=64):
                    super().__init__()
                    self.length = length
                    self.encoder = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(length, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, latent_dim),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, length),
                        nn.Unflatten(1, (length, 1)),
                    )
                    self.mask_token = nn.Parameter(torch.zeros(1))

                def forward(self, x, mask=None):
                    if mask is not None:
                        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
                    return self.decoder(self.encoder(x))

                @torch.no_grad()
                def embed(self, x):
                    self.eval()
                    return self.encoder(x)

            set_seed()
            autoencoder = ProfileAutoencoder()
            optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=2e-3)
            gen = torch.Generator().manual_seed(0)
            inputs = torch.tensor(profiles_train)

            pretrain_curve = []
            for epoch in range(scaled(full=60, fast=5)):
                autoencoder.train()
                order = torch.randperm(len(inputs), generator=gen)
                epoch_loss, seen = 0.0, 0
                for start in range(0, len(order), 64):
                    batch = inputs[order[start : start + 64]]
                    half = len(batch) // 2
                    mask = torch.cat([
                        random_mask((half, 24), ratio=0.3, generator=gen),
                        block_mask((len(batch) - half, 24), block=6, generator=gen),
                    ])
                    prediction = autoencoder(batch, mask=mask)
                    loss = masked_loss(prediction, batch, mask)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach()) * len(batch)
                    seen += len(batch)
                pretrain_curve.append(epoch_loss / seen)
            print(f"masked reconstruction loss: "
                  f"{pretrain_curve[0]:.4f} -> {pretrain_curve[-1]:.4f}")
            """,
            check_code="""
            assert pretrain_curve[-1] < pretrain_curve[0], "the reconstruction loss did not fall"
            _z = autoencoder.embed(torch.tensor(profiles_valid[:5]))
            assert _z.shape == (5, LATENT), f"expected (5, {LATENT}), got {tuple(_z.shape)}"
            print("Basic checks passed.")
            """,
            explanation="""
            The mask token is *learned* rather than a constant zero. Zero is a
            perfectly plausible value for a standardised profile, so a constant
            zero would be indistinguishable from real data at that position; a
            learned parameter lets the network signal "this one is missing".

            Mixing the two masking strategies in each epoch is what most real
            recipes do. Random-only masking makes the task too easy (interpolate);
            block-only makes it uniformly hard and slows early learning.
            """,
        ),
        Task(
            number="4.3",
            depends_on=("4.2",),
            title="Probe what the encoder learned without labels",
            kind="analysis",
            difficulty=3,
            background="""
            The strict test of a representation is not whether a 2-D projection
            *looks* clustered — it is whether a **linear** model can read a factor
            out of it. Those two tests disagree more often than people expect.
            """,
            instruction="""
            Project the embeddings to two dimensions with PCA and colour them by a
            factor the encoder never saw, then run a linear probe for three
            different factors and report which are recoverable.
            """,
            requirements=(
                "Colour the PCA scatter by the daily PV share.",
                "Probe at least three factors with a ridge regression, reporting R².",
                "Use a random split for the probe, and say in one line why that is "
                "appropriate here when it was not in chapter 01.",
            ),
            hints=(
                "PV share for a day: that day's PV energy divided by its load energy.",
                "A negative R² means the probe is worse than predicting the mean — "
                "that factor is not linearly present.",
            ),
            expected="""
            Sun-related factors should come out clearly. At least one factor should
            *fail*, and the interesting part of the task is explaining why.
            """,
            exercise_code="""
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.linear_model import Ridge

            embeddings = autoencoder.embed(torch.tensor(profiles_train)).numpy()
            print(f"embeddings {embeddings.shape}")

            daily = frame.groupby(frame.index.date)[["pv_mw", "load_mw"]].sum()
            pv_share = np.array([
                daily.loc[d.date(), "pv_mw"] / max(daily.loc[d.date(), "load_mw"], 1e-9)
                for d in dates_train
            ])
            daily_temperature = frame.groupby(frame.index.date).temperature_c.mean()
            temperature = np.array([daily_temperature[d.date()] for d in dates_train])
            is_weekend = (dates_train.dayofweek >= 5).astype(float)

            # TODO 1: PCA to two components and a scatter coloured by pv_share
            coords = ____
            fig, ax = plt.subplots(figsize=(5.4, 4.2))
            scatter = ax.scatter(____, ____, c=____, s=9, cmap="viridis")
            plt.colorbar(scatter, ax=ax, label="daily PV share")
            ax.set_xlabel(____)
            ax.set_ylabel(____)
            ax.set_title(____)
            plt.show()

            # TODO 2: a linear probe for each factor
            def linear_probe(features, target, name, seed=0):
                rng = np.random.default_rng(seed)
                order = rng.permutation(len(features))
                cut = int(0.7 * len(order))
                model = Ridge(alpha=1.0).fit(____, ____)
                score = r2(____, ____)
                print(f"  {name:28s} R2 = {score:6.3f}")
                return score

            print("Linear readout from the 8-dimensional embedding:")
            for name, values in [("daily PV share", pv_share),
                                 ("mean temperature", temperature),
                                 ("is it a weekend?", is_weekend)]:
                linear_probe(embeddings, values, name)
            """,
            solution_code="""
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.linear_model import Ridge

            embeddings = autoencoder.embed(torch.tensor(profiles_train)).numpy()
            print(f"embeddings {embeddings.shape}")

            daily = frame.groupby(frame.index.date)[["pv_mw", "load_mw"]].sum()
            pv_share = np.array([
                daily.loc[d.date(), "pv_mw"] / max(daily.loc[d.date(), "load_mw"], 1e-9)
                for d in dates_train
            ])
            daily_temperature = frame.groupby(frame.index.date).temperature_c.mean()
            temperature = np.array([daily_temperature[d.date()] for d in dates_train])
            is_weekend = (dates_train.dayofweek >= 5).astype(float)

            coords = PCA(n_components=2).fit_transform(embeddings)
            fig, ax = plt.subplots(figsize=(5.4, 4.2))
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=pv_share, s=9, cmap="viridis")
            plt.colorbar(scatter, ax=ax, label="daily PV share")
            ax.set_xlabel("principal component 1")
            ax.set_ylabel("principal component 2")
            ax.set_title("Latent space of an encoder trained without labels")
            plt.show()

            def linear_probe(features, target, name, seed=0):
                rng = np.random.default_rng(seed)
                order = rng.permutation(len(features))
                cut = int(0.7 * len(order))
                fit, held = order[:cut], order[cut:]
                model = Ridge(alpha=1.0).fit(features[fit], target[fit])
                score = r2(target[held], model.predict(features[held]))
                print(f"  {name:28s} R2 = {score:6.3f}")
                return score

            print("Linear readout from the 8-dimensional embedding:")
            scores = {}
            for name, values in [("daily PV share", pv_share),
                                 ("mean temperature", temperature),
                                 ("is it a weekend?", is_weekend)]:
                scores[name] = linear_probe(embeddings, values, name)

            print("\\nA RANDOM split is right here and wrong in chapter 01. The question is")
            print("'is this factor linearly encoded in the representation', which is a probe,")
            print("not a forecast. A chronological split would additionally demand that the")
            print("probe extrapolate to unseen parts of the year and confound the two.")
            print("\\nSun-related factors are readable. Weekend is not, and the reason is in")
            print("the preprocessing: we normalised every day to zero mean, and a weekend")
            print("differs from a weekday mostly in LEVEL rather than in shape. The encoder")
            print("cannot encode what the preprocessing deleted.")
            """,
            check_code="""
            assert coords.shape == (len(embeddings), 2)
            assert scores["daily PV share"] > 0.3, \\
                "PV share should be clearly readable — check the pretraining loss fell"
            print("Basic checks passed.")
            """,
            explanation="""
            The weekend failure is the most valuable line in the output. It is not
            a defect of the encoder; it is a consequence of a decision made three
            cells earlier, when each day was normalised to zero mean and unit
            variance.

            **Your preprocessing and your pretraining objective jointly decide
            which downstream tasks are even possible.** Check what you deleted
            before you go looking for it.
            """,
        ),
        Task(
            number="4.4",
            depends_on=("4.2",),
            title="Reuse the frozen encoder with very few labels",
            kind="coding",
            difficulty=3,
            background="""
            Pretraining does not buy accuracy. It buys **label efficiency** — and
            whether that is worth anything depends entirely on whether labels are
            your scarce resource. In power systems they usually are: measurements
            are everywhere, annotations are not.
            """,
            instruction="""
            Freeze the encoder, fit a ridge regression on its embeddings to
            predict the daily PV share, and compare against the same ridge fitted
            on the raw 24 values — across several label budgets.
            """,
            requirements=(
                "The encoder must stay frozen; only the head is fitted.",
                "Evaluate on the test year, never on the labels used for fitting.",
                "Sweep at least three label budgets.",
                "Average over a few random draws so the small-budget numbers are "
                "not one lucky sample.",
            ),
            hints=(
                "`autoencoder.embed` is already decorated with `torch.no_grad`.",
                "Use the validation year as the labelled pool and the test year for "
                "evaluation, so neither was seen during pretraining.",
            ),
            expected="""
            The frozen encoder should help most at the smallest budget. Expect the
            advantage to be modest — 24 numbers is a small input, and the raw
            baseline is strong. Report the size of the effect honestly.
            """,
            exercise_code="""
            import pandas as pd
            from sklearn.linear_model import Ridge

            def pv_share_for(dates):
                daily = frame.groupby(frame.index.date)[["pv_mw", "load_mw"]].sum()
                return np.array([
                    daily.loc[d.date(), "pv_mw"] / max(daily.loc[d.date(), "load_mw"], 1e-9)
                    for d in dates
                ])

            y_pool = pv_share_for(dates_valid)      # labelled pool
            y_eval = pv_share_for(dates_test)       # evaluation

            # TODO 1: embeddings for the pool and the evaluation set
            z_pool = ____
            z_eval = ____
            raw_pool = profiles_valid[:, :, 0]
            raw_eval = profiles_test[:, :, 0]

            def score_at(n_labels, features_pool, features_eval, seed):
                rng = np.random.default_rng(seed)
                chosen = rng.choice(len(y_pool), size=n_labels, replace=False)
                # TODO 2: fit ridge on the chosen labels, score on the evaluation set
                model = ____
                return mae(____, ____)

            budgets = [10, 30] if fast_mode() else [10, 30, 100]
            rows = []
            for n in budgets:
                rows.append({
                    "labels": n,
                    "frozen encoder": np.mean(
                        [score_at(n, z_pool, z_eval, s) for s in range(3)]),
                    "raw 24 values": np.mean(
                        [score_at(n, raw_pool, raw_eval, s) for s in range(3)]),
                })
            efficiency = pd.DataFrame(rows).set_index("labels")
            display(efficiency.round(4))
            """,
            solution_code="""
            import pandas as pd
            from sklearn.linear_model import Ridge

            def pv_share_for(dates):
                daily = frame.groupby(frame.index.date)[["pv_mw", "load_mw"]].sum()
                return np.array([
                    daily.loc[d.date(), "pv_mw"] / max(daily.loc[d.date(), "load_mw"], 1e-9)
                    for d in dates
                ])

            y_pool = pv_share_for(dates_valid)      # labelled pool
            y_eval = pv_share_for(dates_test)       # evaluation

            z_pool = autoencoder.embed(torch.tensor(profiles_valid)).numpy()
            z_eval = autoencoder.embed(torch.tensor(profiles_test)).numpy()
            raw_pool = profiles_valid[:, :, 0]
            raw_eval = profiles_test[:, :, 0]

            def score_at(n_labels, features_pool, features_eval, seed):
                rng = np.random.default_rng(seed)
                chosen = rng.choice(len(y_pool), size=n_labels, replace=False)
                model = Ridge(alpha=1.0).fit(features_pool[chosen], y_pool[chosen])
                return mae(y_eval, model.predict(features_eval))

            budgets = [10, 30] if fast_mode() else [10, 30, 100]
            rows = []
            for n in budgets:
                rows.append({
                    "labels": n,
                    "frozen encoder": np.mean(
                        [score_at(n, z_pool, z_eval, s) for s in range(3)]),
                    "raw 24 values": np.mean(
                        [score_at(n, raw_pool, raw_eval, s) for s in range(3)]),
                })
            efficiency = pd.DataFrame(rows).set_index("labels")
            display(efficiency.round(4))

            smallest = efficiency.index.min()
            gain = 1 - (efficiency.loc[smallest, "frozen encoder"]
                        / efficiency.loc[smallest, "raw 24 values"])
            print(f"\\nAt {smallest} labels the frozen encoder's MAE is {abs(gain):.1%} "
                  f"{'lower' if gain > 0 else 'higher'} than the raw baseline.")
            print("\\nBe precise about the size of this effect. A 24-dimensional input is")
            print("easy, and a ridge on the raw values is a serious competitor. The case for")
            print("learned representations strengthens as inputs get bigger and more")
            print("structured — which is exactly the regime of chapters 09 and 10.")
            """,
            check_code="""
            assert z_pool.shape[1] == LATENT, "the probe should use the latent code"
            assert (efficiency > 0).all().all(), "MAE must be positive"
            print("Basic checks passed.")
            """,
            explanation="""
            The encoder was pretrained on the **training** years only, so both the
            labelled pool (validation year) and the evaluation set (test year) are
            unseen by it. Pretraining on everything and then probing would be a
            leak, and a subtle one — the encoder would have seen the evaluation
            profiles, just without their labels.

            Averaging over three draws matters at ten labels: a single draw can
            differ by tens of percent depending on which ten days you happen to
            get.
            """,
        ),
    ),
)

# --------------------------------------------------------------------------- 05

CHAPTER_05 = Chapter(
    number=5,
    title="Attention",
    tutorial="05_attention.ipynb",
    intro="""
    The central implementation exercise of the course. You will write scaled
    dot-product attention from the equation, in NumPy, and only compare it
    against PyTorch's version once yours works.

    Do not use `nn.MultiheadAttention` or
    `torch.nn.functional.scaled_dot_product_attention` for task 5.1. The whole
    point is that you can write it.
    """,
    setup_code="""
    import math

    import numpy as np
    import torch

    from ai_power_course.config import set_seed
    from ai_power_course.data import load_energy_data, time_split

    set_seed()
    frame = load_energy_data().frame
    train_raw, valid_raw, test_raw = time_split(frame)
    print("Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V")
    """,
    tasks=(
        Task(
            number="5.1",
            title="Scaled dot-product attention from the equation",
            kind="coding",
            difficulty=2,
            background="""
            Attention is a soft, differentiable dictionary lookup. A **query**
            asks what it is looking for, each **key** advertises what its position
            offers, and each **value** is what that position returns if chosen.
            Instead of retrieving one value, return a weighted average of all of
            them:

            $$S = \\frac{QK^\\top}{\\sqrt{d_k}}, \\qquad
              A = \\mathrm{softmax}(S), \\qquad
              Y = AV$$
            """,
            instruction="""
            Implement the three steps in NumPy: score, normalise, aggregate.
            Implement the softmax yourself too, numerically stably.
            """,
            requirements=(
                "`softmax(x, axis=-1)` must subtract the max before exponentiating.",
                "`scaled_dot_product_attention(Q, K, V, mask=None)` returns "
                "`(output, weights)`.",
                "`Q` is `(n_queries, d_k)`, `K` is `(n_keys, d_k)`, `V` is "
                "`(n_keys, d_v)`.",
                "`mask` is boolean with `True` = *may attend*; masked scores become "
                "`-inf` **before** the softmax.",
                "NumPy only. No PyTorch attention helpers.",
            ),
            hints=(
                "Each row of the weight matrix is a probability distribution over "
                "the keys, so it must sum to 1 along the last axis.",
                "Masking after the softmax would leave the surviving weights not "
                "summing to one — apply it to the scores.",
            ),
            expected="""
            Output of shape `(n_queries, d_v)`, weights of shape
            `(n_queries, n_keys)` with rows summing to 1, and agreement with
            PyTorch to floating-point precision.
            """,
            exercise_code="""
            def softmax(x, axis=-1):
                \"\"\"Numerically stable softmax.\"\"\"
                # TODO: subtract the max along `axis` before exponentiating
                raise NotImplementedError

            def scaled_dot_product_attention(Q, K, V, mask=None):
                \"\"\"Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

                Returns
                -------
                output  : (n_queries, d_v)
                weights : (n_queries, n_keys)
                \"\"\"
                d_k = Q.shape[-1]
                # TODO 1: the scaled score matrix
                scores = ____
                # TODO 2: apply the mask, if given, BEFORE the softmax
                if mask is not None:
                    scores = ____
                # TODO 3: normalise, then aggregate the values
                weights = ____
                output = ____
                return output, weights

            rng = np.random.default_rng(0)
            Q = rng.normal(size=(6, 8))
            K = rng.normal(size=(6, 8))
            V = rng.normal(size=(6, 4))
            output, weights = scaled_dot_product_attention(Q, K, V)
            print(f"output {output.shape}   weights {weights.shape}")
            print(f"every row of the weight matrix sums to 1: "
                  f"{np.allclose(weights.sum(axis=-1), 1.0)}")
            """,
            solution_code="""
            def softmax(x, axis=-1):
                \"\"\"Numerically stable softmax.\"\"\"
                shifted = x - np.max(x, axis=axis, keepdims=True)
                exponentials = np.exp(shifted)
                return exponentials / np.sum(exponentials, axis=axis, keepdims=True)

            def scaled_dot_product_attention(Q, K, V, mask=None):
                \"\"\"Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

                Returns
                -------
                output  : (n_queries, d_v)
                weights : (n_queries, n_keys)
                \"\"\"
                d_k = Q.shape[-1]
                scores = Q @ K.T / math.sqrt(d_k)
                if mask is not None:
                    scores = np.where(mask, scores, -np.inf)
                weights = softmax(scores, axis=-1)
                output = weights @ V
                return output, weights

            rng = np.random.default_rng(0)
            Q = rng.normal(size=(6, 8))
            K = rng.normal(size=(6, 8))
            V = rng.normal(size=(6, 4))
            output, weights = scaled_dot_product_attention(Q, K, V)
            print(f"output {output.shape}   weights {weights.shape}")
            print(f"every row of the weight matrix sums to 1: "
                  f"{np.allclose(weights.sum(axis=-1), 1.0)}")
            """,
            check_code="""
            def check_attention_function(fn):
                rng = np.random.default_rng(7)
                q, k, v = (rng.normal(size=(5, 8)) for _ in range(3))
                out, w = fn(q, k, v)
                assert out.shape == (5, 8), f"output should be (5, 8), got {out.shape}"
                assert w.shape == (5, 5), f"weights should be (5, 5), got {w.shape}"
                assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-6), "rows must sum to 1"
                assert (w >= 0).all(), "attention weights cannot be negative"
                reference = torch.nn.functional.scaled_dot_product_attention(
                    *(torch.tensor(a).unsqueeze(0) for a in (q, k, v))
                ).squeeze(0).numpy()
                assert np.allclose(out, reference, atol=1e-10), "disagrees with PyTorch"
                print("Basic checks passed — and it matches PyTorch to 1e-10.")

            check_attention_function(scaled_dot_product_attention)
            """,
            explanation="""
            **The softmax is applied along the key axis** because each query
            produces a probability distribution over all keys. Applying it along
            the query axis instead is a real and surprisingly common bug: the
            shapes still work, the model still trains, and it is computing
            something else entirely.

            The `-inf` before the softmax is exact: `exp(-inf) = 0`, so the masked
            positions get precisely zero weight and the remaining row still sums
            to one. Zeroing the weights after the softmax would leave the row
            summing to less than one and quietly scale down the output.
            """,
        ),
        Task(
            number="5.2",
            depends_on=("5.1",),
            title="Why divide by the square root of d_k?",
            kind="analysis",
            difficulty=2,
            background="""
            For random `q` and `k` with unit variance, the dot product
            $\\sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$. At $d_k = 256$ the
            scores have a standard deviation of 16, and a softmax over scores that
            spread is effectively an `argmax`.
            """,
            instruction="""
            Measure the effect: for several key dimensions, compare the standard
            deviation of the scores and the largest softmax weight, with and
            without the scaling.
            """,
            requirements=(
                "Sweep at least four values of `d_k` spanning two orders of magnitude.",
                "Report the score spread and the maximum attention weight for both variants.",
                "Say in one line why a saturated softmax stops the model learning.",
            ),
            hints=(
                "The scaled scores should have a standard deviation close to 1 "
                "regardless of `d_k` — that is the whole point.",
                "A maximum weight near 1.0 means one key has taken everything.",
            ),
            expected="""
            Unscaled, the maximum weight should march towards 1.0 as `d_k` grows.
            Scaled, it should stay small and roughly constant.
            """,
            exercise_code="""
            import pandas as pd

            rows = []
            for d_k in [4, 16, 64, 256, 1024]:
                rng = np.random.default_rng(1)
                q = rng.normal(size=(1, d_k))
                k = rng.normal(size=(64, d_k))
                # TODO: scores with and without the 1/sqrt(d_k) factor
                unscaled = ____
                scaled_scores = ____
                rows.append({
                    "d_k": d_k,
                    "score std (unscaled)": unscaled.std(),
                    "score std (scaled)": scaled_scores.std(),
                    "max weight (unscaled)": softmax(unscaled).max(),
                    "max weight (scaled)": softmax(scaled_scores).max(),
                })
            display(pd.DataFrame(rows).set_index("d_k").round(3))
            """,
            solution_code="""
            import pandas as pd

            rows = []
            for d_k in [4, 16, 64, 256, 1024]:
                rng = np.random.default_rng(1)
                q = rng.normal(size=(1, d_k))
                k = rng.normal(size=(64, d_k))
                unscaled = q @ k.T
                scaled_scores = unscaled / math.sqrt(d_k)
                rows.append({
                    "d_k": d_k,
                    "score std (unscaled)": unscaled.std(),
                    "score std (scaled)": scaled_scores.std(),
                    "max weight (unscaled)": softmax(unscaled).max(),
                    "max weight (scaled)": softmax(scaled_scores).max(),
                })
            table = pd.DataFrame(rows).set_index("d_k")
            display(table.round(3))

            print("A saturated softmax has a near-zero gradient: once one weight is ~1 and")
            print("the rest are ~0, changing the scores barely changes the output, so there")
            print("is almost no signal telling the model to attend somewhere else. The")
            print("division keeps the operation trainable as the model gets wider — it is")
            print("not cosmetic.")
            """,
            check_code="""
            assert table.loc[1024, "max weight (unscaled)"] > 0.9, \\
                "without scaling, a large d_k should saturate the softmax"
            assert table.loc[1024, "max weight (scaled)"] < 0.3, \\
                "with scaling, the distribution should stay usable"
            print("Basic checks passed.")
            """,
            explanation="""
            The scaled column shows a standard deviation near 1 at every `d_k`,
            which is exactly what the $1/\\sqrt{d_k}$ factor is designed to
            achieve: it normalises the variance of a sum of `d_k` independent
            products back to unity.
            """,
        ),
        Task(
            number="5.3",
            depends_on=("5.1",),
            title="Causal masking and an attention heatmap",
            kind="coding",
            difficulty=2,
            background="""
            One line separates a model that fills in blanks from a model that
            predicts the future: a lower-triangular mask, so position $i$ can
            attend only to $j \\le i$.

            That single change is the whole difference between BERT and GPT.
            """,
            instruction="""
            Build a causal mask, apply it through your attention function, and
            plot both the unmasked and masked weight matrices side by side.
            """,
            requirements=(
                "`causal_mask(n)` returns an `(n, n)` boolean array, `True` = allowed.",
                "Both heatmaps need a colourbar, axis labels and a title.",
                "Assert that the masked weights above the diagonal are exactly zero.",
            ),
            hints=(
                "`np.tril(np.ones((n, n), dtype=bool))` is the whole mask.",
                "`np.triu_indices(n, k=1)` selects the strictly upper triangle for "
                "the assertion.",
            ),
            expected="""
            The masked heatmap should be strictly lower-triangular with exact
            zeros above the diagonal, and its rows should still sum to one.
            """,
            exercise_code="""
            import matplotlib.pyplot as plt

            def causal_mask(n):
                \"\"\"Lower-triangular boolean mask: True where attention is allowed.\"\"\"
                # TODO
                raise NotImplementedError

            SEQ = 12
            rng = np.random.default_rng(3)
            Q = rng.normal(size=(SEQ, 16))
            K = rng.normal(size=(SEQ, 16))
            V = rng.normal(size=(SEQ, 16))

            _, weights_open = scaled_dot_product_attention(Q, K, V)
            # TODO: the same, with the causal mask
            _, weights_causal = ____

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, matrix, title in [
                (axes[0], weights_open, "Bidirectional: every position sees every other"),
                (axes[1], weights_causal, "Causal: position i sees only j <= i"),
            ]:
                image = ax.imshow(matrix, cmap="magma", aspect="auto")
                plt.colorbar(image, ax=ax, label="attention weight")
                ax.set_xlabel("key position (attended to)")
                ax.set_ylabel("query position")
                ax.set_title(title, fontsize=10)
            fig.tight_layout()
            plt.show()

            assert weights_causal[np.triu_indices(SEQ, k=1)].max() == 0.0
            print("No attention leaks into the future.")
            """,
            solution_code="""
            import matplotlib.pyplot as plt

            def causal_mask(n):
                \"\"\"Lower-triangular boolean mask: True where attention is allowed.\"\"\"
                return np.tril(np.ones((n, n), dtype=bool))

            SEQ = 12
            rng = np.random.default_rng(3)
            Q = rng.normal(size=(SEQ, 16))
            K = rng.normal(size=(SEQ, 16))
            V = rng.normal(size=(SEQ, 16))

            _, weights_open = scaled_dot_product_attention(Q, K, V)
            _, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask(SEQ))

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, matrix, title in [
                (axes[0], weights_open, "Bidirectional: every position sees every other"),
                (axes[1], weights_causal, "Causal: position i sees only j <= i"),
            ]:
                image = ax.imshow(matrix, cmap="magma", aspect="auto")
                plt.colorbar(image, ax=ax, label="attention weight")
                ax.set_xlabel("key position (attended to)")
                ax.set_ylabel("query position")
                ax.set_title(title, fontsize=10)
            fig.tight_layout()
            plt.show()

            assert weights_causal[np.triu_indices(SEQ, k=1)].max() == 0.0
            print("No attention leaks into the future.")
            print(f"Rows still sum to 1: {np.allclose(weights_causal.sum(axis=-1), 1.0)}")
            """,
            check_code="""
            _m = causal_mask(5)
            assert _m.dtype == bool and _m.shape == (5, 5)
            assert _m[0].sum() == 1 and _m[4].sum() == 5, "row i must allow exactly i+1 keys"
            assert not _m[0, 1], "position 0 must not see position 1"
            print("Basic checks passed.")
            """,
            explanation="""
            Row 0 of a causal mask allows exactly one key — itself — so its
            attention output is simply its own value vector. That is why the first
            token of a language model carries so little information and why models
            often prepend a dedicated start token.

            Note that the rows still sum to one after masking. That is the payoff
            for masking the scores rather than the weights.
            """,
        ),
        Task(
            number="5.4",
            title="Is a large attention weight an explanation?",
            kind="reflection",
            difficulty=3,
            background="""
            Attention heatmaps are extremely persuasive and routinely
            over-interpreted. Tutorial 05 replaced a trained model's attention
            with a uniform average and measured what actually changed.
            """,
            instruction="""
            Give three concrete reasons why a large attention weight does not, on
            its own, establish that the attended position caused the prediction.
            Then describe one experiment that would produce *evidence* rather than
            a picture.
            """,
            expected="""
            A strong answer distinguishes "the model looked here" from "this
            changed the answer", and proposes an ablation with a number attached.
            """,
            answer_template="""
            **Reason 1.** …

            **Reason 2.** …

            **Reason 3.** …

            **An experiment that would produce evidence.** …
            """,
            answer="""
            **Reason 1 — the value vector matters as much as the weight.** The
            output is $\\sum_j a_j v_j$. A weight of 0.6 on a position whose value
            vector is near zero contributes nothing, while a weight of 0.01 on a
            position with a large value vector can dominate. Task 5.1's own
            example makes this concrete: attention weights are not contributions,
            and only the product is.

            **Reason 2 — residual connections route around attention entirely.**
            In a real Transformer block the output is `x + Attention(LN(x))`. The
            representation flows down the residual stream whether or not attention
            looks at it, so a head can be almost irrelevant to the final
            prediction while still producing a beautifully structured heatmap.

            **Reason 3 — different attention distributions can give identical
            outputs.** Jain and Wallace (2019) constructed adversarial weight
            distributions that leave predictions unchanged. If two very different
            explanations produce the same answer, neither is *the* explanation.

            **An experiment that would produce evidence.** Ablate, and report a
            number. Replace the learned attention with (a) a uniform average, (b)
            the weights reversed, (c) weights permuted within each row, and (d)
            weights taken from a different, randomly initialised model. Measure
            the test error under each. If uniform attention costs you little, the
            learned pattern was not doing the work the heatmap suggests. Tutorial
            05 runs exactly (a) and finds the error rises from about 2,565 MW to
            3,745 MW — which licenses the claim that the pattern matters, without
            licensing any claim about *which* position caused *which* prediction.
            """,
        ),
    ),
)

CHAPTERS: tuple[Chapter, ...] = (
    CHAPTER_01,
    CHAPTER_02,
    CHAPTER_03,
    CHAPTER_04,
    CHAPTER_05,
)
