"""Exercise chapters 06-10: Transformers through grid foundation models.

Chapters 08 and 09 touch the network. Their setup code degrades to a clear
message under ``AI_POWER_COURSE_OFFLINE`` rather than raising, so the notebook
still executes end to end without downloads.

As in :mod:`~ai_power_course.exercises.part1`, every ``setup_code`` block
rebuilds what it needs from the package and never reads a variable a student was
asked to write.
"""

from __future__ import annotations

from . import Chapter, Task

# --------------------------------------------------------------------------- 06

CHAPTER_06 = Chapter(
    number=6,
    title="Transformers",
    tutorial="06_transformers.ipynb",
    intro="""
    Attention is a layer. A Transformer is an architecture: multi-head attention,
    a position-wise feed-forward network, residual connections, layer
    normalisation and — because attention has no notion of order — positional
    information.

    Here the tokens are **hours**, not words. That is the point of this chapter.
    """,
    setup_code="""
    import math

    import numpy as np
    import torch
    from torch import nn

    from ai_power_course.config import scaled, set_seed
    from ai_power_course.data import load_energy_data, time_split
    from ai_power_course.metrics import mae
    from ai_power_course.models.training import (
        Standardizer, TrainConfig, make_loader, train_model,
    )

    set_seed()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    frame = load_energy_data().frame
    train_raw, valid_raw, test_raw = time_split(frame)

    CONTEXT, HORIZON = 168, 24
    STRIDE = scaled(full=6, fast=36)

    def windows(part, context=CONTEXT, horizon=HORIZON, stride=1):
        values = part.load_mw.to_numpy(dtype=np.float32)
        starts = np.arange(0, len(values) - context - horizon + 1, stride)
        x = np.stack([values[s : s + context] for s in starts])[:, :, None]
        y = np.stack([values[s + context : s + context + horizon] for s in starts])
        return x.astype(np.float32), y.astype(np.float32)

    x_train, y_train = windows(train_raw, stride=STRIDE)
    x_valid, y_valid = windows(valid_raw, stride=STRIDE)
    x_test, y_test = windows(test_raw, stride=4)

    x_scaler = Standardizer().fit(x_train.reshape(-1, 1))
    y_scaler = Standardizer().fit(y_train)
    xs_train, xs_valid, xs_test = (x_scaler.transform(a) for a in (x_train, x_valid, x_test))
    ys_train, ys_valid = (y_scaler.transform(a) for a in (y_train, y_valid))
    print(f"{len(xs_train):,} training windows of {CONTEXT} hours -> {HORIZON} hours")
    """,
    tasks=(
        Task(
            number="6.1",
            title="Multi-head attention",
            kind="coding",
            difficulty=2,
            background="""
            One attention head computes one weighted average, so it can express
            one kind of relationship. Splitting `d_model` into `h` heads of width
            `d_model / h` lets the layer attend to several patterns at once — the
            same hour yesterday, the same hour last week, the recent trend — at no
            extra cost, because the head width shrinks as the count grows.
            """,
            instruction="""
            Implement multi-head self-attention in PyTorch: project to Q, K and V,
            split into heads, attend, concatenate, project out.
            """,
            requirements=(
                "`MultiHeadSelfAttention(d_model, n_heads)`, with `d_model` "
                "divisible by `n_heads`.",
                "`forward(x)` takes `(batch, length, d_model)` and returns "
                "`(output, weights)` with weights `(batch, n_heads, length, length)`.",
                "Reuse the maths from task 5.1. Do **not** call "
                "`nn.MultiheadAttention` or `F.scaled_dot_product_attention`.",
                "Assert `d_model % n_heads == 0` in `__init__`.",
            ),
            hints=(
                "One `nn.Linear(d_model, 3 * d_model)` produces Q, K and V "
                "together; `.chunk(3, dim=-1)` splits them.",
                "Splitting heads is a reshape plus a transpose: "
                "`(B, L, H, d_head).transpose(1, 2)` -> `(B, H, L, d_head)`.",
                "Merging them back is the same two operations reversed, and needs "
                "`.contiguous()` before the final `.view`.",
            ),
            expected="""
            Output shape equal to the input shape, weights summing to one along the
            last axis, and — for a single head — exact agreement with PyTorch's
            fused kernel.
            """,
            exercise_code="""
            class MultiHeadSelfAttention(nn.Module):
                def __init__(self, d_model, n_heads, dropout=0.0):
                    super().__init__()
                    # TODO 1: guard the divisibility, store d_head
                    assert ____, "d_model must be divisible by n_heads"
                    self.d_model, self.n_heads = d_model, n_heads
                    self.d_head = ____
                    self.qkv = nn.Linear(d_model, 3 * d_model)
                    self.out_proj = nn.Linear(d_model, d_model)
                    self.dropout = nn.Dropout(dropout)

                def _split_heads(self, t):
                    batch, length, _ = t.shape
                    # TODO 2: (B, L, d_model) -> (B, H, L, d_head)
                    return ____

                def _merge_heads(self, t):
                    batch, heads, length, d_head = t.shape
                    # TODO 3: (B, H, L, d_head) -> (B, L, d_model)
                    return ____

                def forward(self, x, mask=None):
                    q, k, v = (self._split_heads(p) for p in self.qkv(x).chunk(3, dim=-1))
                    # TODO 4: scores, mask, softmax, aggregate
                    scores = ____
                    if mask is not None:
                        scores = scores.masked_fill(~mask, float("-inf"))
                    weights = ____
                    attended = ____
                    return self.out_proj(self._merge_heads(attended)), weights

            set_seed()
            layer = MultiHeadSelfAttention(d_model=64, n_heads=4)
            sample = torch.randn(2, 12, 64)
            output, weights = layer(sample)
            print(f"input {tuple(sample.shape)} -> output {tuple(output.shape)}")
            print(f"weights {tuple(weights.shape)} (batch, heads, query, key)")
            """,
            solution_code="""
            class MultiHeadSelfAttention(nn.Module):
                def __init__(self, d_model, n_heads, dropout=0.0):
                    super().__init__()
                    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
                    self.d_model, self.n_heads = d_model, n_heads
                    self.d_head = d_model // n_heads
                    self.qkv = nn.Linear(d_model, 3 * d_model)
                    self.out_proj = nn.Linear(d_model, d_model)
                    self.dropout = nn.Dropout(dropout)

                def _split_heads(self, t):
                    batch, length, _ = t.shape
                    return t.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)

                def _merge_heads(self, t):
                    batch, heads, length, d_head = t.shape
                    return t.transpose(1, 2).contiguous().view(batch, length, heads * d_head)

                def forward(self, x, mask=None):
                    q, k, v = (self._split_heads(p) for p in self.qkv(x).chunk(3, dim=-1))
                    scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
                    if mask is not None:
                        scores = scores.masked_fill(~mask, float("-inf"))
                    weights = self.dropout(torch.softmax(scores, dim=-1))
                    attended = weights @ v
                    return self.out_proj(self._merge_heads(attended)), weights

            set_seed()
            layer = MultiHeadSelfAttention(d_model=64, n_heads=4)
            sample = torch.randn(2, 12, 64)
            output, weights = layer(sample)
            print(f"input {tuple(sample.shape)} -> output {tuple(output.shape)}")
            print(f"weights {tuple(weights.shape)} (batch, heads, query, key)")
            """,
            check_code="""
            assert output.shape == sample.shape, "attention must preserve the shape"
            assert weights.shape == (2, 4, 12, 12)
            assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 12), atol=1e-5)

            # A single head must reproduce PyTorch's fused kernel exactly.
            _single = MultiHeadSelfAttention(d_model=16, n_heads=1)
            _x = torch.randn(1, 5, 16)
            _q, _k, _v = _single.qkv(_x).chunk(3, dim=-1)
            _reference = torch.nn.functional.scaled_dot_product_attention(_q, _k, _v)
            _ours, _ = _single(_x)
            assert torch.allclose(_single.out_proj(_reference), _ours, atol=1e-5), \\
                "single-head attention disagrees with PyTorch"
            print("Basic checks passed — and one head matches PyTorch's kernel.")
            """,
            explanation="""
            The scaling divisor is `sqrt(d_head)`, not `sqrt(d_model)`. Each head
            takes a dot product over `d_head` dimensions, so that is the dimension
            whose variance needs correcting — task 5.2 measured exactly this.

            `.contiguous()` before `.view` in `_merge_heads` is required: after a
            `transpose` the tensor's memory layout no longer matches its shape, and
            `view` refuses to reinterpret it. `.reshape` would silently copy
            instead, which works but hides the cost.
            """,
        ),
        Task(
            number="6.2",
            depends_on=("6.1",),
            title="Positional encoding, and what happens without it",
            kind="coding",
            difficulty=2,
            background="""
            Self-attention is **permutation equivariant**: shuffle the input
            positions and the outputs shuffle identically. For a set of buses that
            is a feature. For a time series it is fatal — a load profile read
            backwards would produce the same forecast.
            """,
            instruction="""
            Implement sinusoidal positional encoding, then demonstrate the
            equivariance empirically: show that attention alone is blind to order
            and that adding the encoding fixes it.
            """,
            requirements=(
                "`positional_encoding(length, d_model)` returns `(length, d_model)`.",
                "Even indices use sine, odd indices cosine.",
                "Demonstrate the permutation property with an assertion, not prose.",
                "Plot the encoding as a heatmap.",
            ),
            hints=(
                "$PE_{(p, 2i)} = \\sin(p / 10000^{2i/d})$ and "
                "$PE_{(p, 2i+1)} = \\cos(p / 10000^{2i/d})$.",
                "Compute the divisor in log space — "
                "`torch.exp(arange(0, d, 2) * (-log(10000) / d))` — to avoid "
                "overflow at large `d_model`.",
                "To show equivariance, permute the input rows and compare against "
                "the same permutation of the output rows.",
            ),
            expected="""
            The equivariance assertion passes without positional encoding and fails
            with it. The heatmap shows smooth low-frequency stripes in the early
            dimensions and fast oscillation in the later ones.
            """,
            exercise_code="""
            import matplotlib.pyplot as plt

            def positional_encoding(length, d_model):
                \"\"\"Sinusoidal positional encoding of shape (length, d_model).\"\"\"
                position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
                # TODO 1: the geometric frequency ladder, computed in log space
                divisor = ____
                encoding = torch.zeros(length, d_model)
                # TODO 2: sine into the even columns, cosine into the odd ones
                encoding[:, 0::2] = ____
                encoding[:, 1::2] = ____
                return encoding

            pe = positional_encoding(CONTEXT, 64)
            fig, ax = plt.subplots(figsize=(8, 3.2))
            image = ax.imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            plt.colorbar(image, ax=ax, label="value")
            ax.set_xlabel("position (hour in the context window)")
            ax.set_ylabel("embedding dimension")
            ax.set_title("Sinusoidal positional encoding")
            plt.show()

            # TODO 3: show that bare attention cannot tell the order apart.
            set_seed()
            layer = MultiHeadSelfAttention(d_model=64, n_heads=4)
            x = torch.randn(1, 10, 64)
            perm = torch.randperm(10)

            out_plain, _ = layer(x)
            out_permuted, _ = layer(____)
            assert torch.allclose(out_permuted, ____, atol=1e-5), \\
                "attention should be permutation equivariant"
            print("Without positional encoding, order carries no information.")

            out_pe, _ = layer(x + pe[:10])
            out_pe_permuted, _ = layer(x[:, perm] + pe[:10])
            print(f"With positional encoding, the same test fails: "
                  f"{not torch.allclose(out_pe_permuted, out_pe[:, perm], atol=1e-5)}")
            """,
            solution_code="""
            import matplotlib.pyplot as plt

            def positional_encoding(length, d_model):
                \"\"\"Sinusoidal positional encoding of shape (length, d_model).\"\"\"
                position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
                divisor = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model)
                )
                encoding = torch.zeros(length, d_model)
                encoding[:, 0::2] = torch.sin(position * divisor)
                encoding[:, 1::2] = torch.cos(position * divisor)
                return encoding

            pe = positional_encoding(CONTEXT, 64)
            fig, ax = plt.subplots(figsize=(8, 3.2))
            image = ax.imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            plt.colorbar(image, ax=ax, label="value")
            ax.set_xlabel("position (hour in the context window)")
            ax.set_ylabel("embedding dimension")
            ax.set_title("Sinusoidal positional encoding")
            plt.show()

            set_seed()
            layer = MultiHeadSelfAttention(d_model=64, n_heads=4)
            x = torch.randn(1, 10, 64)
            perm = torch.randperm(10)

            out_plain, _ = layer(x)
            out_permuted, _ = layer(x[:, perm])
            assert torch.allclose(out_permuted, out_plain[:, perm], atol=1e-5), \\
                "attention should be permutation equivariant"
            print("Without positional encoding, order carries no information.")

            out_pe, _ = layer(x + pe[:10])
            out_pe_permuted, _ = layer(x[:, perm] + pe[:10])
            print(f"With positional encoding, the same test fails: "
                  f"{not torch.allclose(out_pe_permuted, out_pe[:, perm], atol=1e-5)}")
            print("\\nThat failure is the feature. Order now changes the answer.")
            """,
            check_code="""
            _pe = positional_encoding(50, 32)
            assert _pe.shape == (50, 32)
            assert _pe.abs().max() <= 1.0 + 1e-6, "sin and cos are bounded by 1"
            assert not torch.allclose(_pe[0], _pe[1]), "positions must be distinguishable"
            print("Basic checks passed.")
            """,
            explanation="""
            Computing the divisor with `torch.exp(... * -log(10000) / d)` rather
            than `10000 ** (-2i/d)` is numerically deliberate: the direct power
            underflows to zero for large `d_model`, which collapses the
            high-frequency dimensions to a constant.

            The equivariance demonstration is the load-bearing part of this task.
            Chapter 10 needs exactly the *opposite* property — a grid encoder must
            be permutation equivariant, because bus numbering is arbitrary — and so
            it deliberately has no positional encoding. Same layer, opposite
            requirement, decided by the domain.
            """,
        ),
        Task(
            number="6.3",
            depends_on=("6.1",),
            title="Assemble a pre-norm Transformer block",
            kind="coding",
            difficulty=2,
            background="""
            The block is `x + Attention(LN(x))` followed by `x + MLP(LN(x))`. Two
            choices deserve naming:

            - the **residual** makes the block an edit to the representation
              rather than a replacement, which is what lets dozens stack;
            - **pre-norm** (normalise *before* the sublayer) trains without a
              learning-rate warm-up, unlike the post-norm arrangement in the 2017
              paper.
            """,
            instruction="""
            Implement the block, then measure the difference the residual makes by
            comparing gradient magnitudes at the first layer of a deep stack with
            and without it.
            """,
            requirements=(
                "`TransformerBlock(d_model, n_heads, d_ff=None, dropout=0.1)`, "
                "with `d_ff` defaulting to `4 * d_model`.",
                "Pre-norm ordering.",
                "The feed-forward network uses GELU.",
                "Compare gradient norms at layer 0 of an 8-block stack, residual "
                "on versus off.",
            ),
            hints=(
                "The feed-forward network is position-wise: it maps `d_model` -> "
                "`d_ff` -> `d_model` and sees each time step independently.",
                "To disable the residual, return the sublayer output directly "
                "instead of adding `x`.",
            ),
            expected="""
            Without residuals the first block's gradient should be orders of
            magnitude smaller. That is the vanishing-gradient problem of chapter 03
            reappearing in depth rather than in time.
            """,
            exercise_code="""
            class TransformerBlock(nn.Module):
                def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, residual=True):
                    super().__init__()
                    d_ff = d_ff or 4 * d_model
                    self.residual = residual
                    self.norm1 = nn.LayerNorm(d_model)
                    self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
                    self.norm2 = nn.LayerNorm(d_model)
                    # TODO 1: the position-wise feed-forward network
                    self.feed_forward = nn.Sequential(
                        ____,
                        nn.GELU(),
                        nn.Dropout(dropout),
                        ____,
                    )
                    self.dropout = nn.Dropout(dropout)

                def forward(self, x, mask=None):
                    # TODO 2: pre-norm attention sublayer, then the MLP sublayer
                    attended, weights = self.attention(____, mask=mask)
                    x = ____ if self.residual else self.dropout(attended)
                    fed = self.dropout(self.feed_forward(____))
                    x = ____ if self.residual else fed
                    return x, weights

            def first_layer_gradient(residual, depth=8, d_model=64):
                set_seed()
                blocks = nn.ModuleList(
                    TransformerBlock(d_model, 4, dropout=0.0, residual=residual)
                    for _ in range(depth)
                )
                h = torch.randn(4, 16, d_model, requires_grad=True)
                x = h
                for block in blocks:
                    x, _ = block(x)
                x.pow(2).mean().backward()
                return float(
                    sum(p.grad.norm() for p in blocks[0].parameters() if p.grad is not None)
                )

            with_residual = first_layer_gradient(residual=True)
            without_residual = first_layer_gradient(residual=False)
            print(f"gradient norm at block 0, with residuals   : {with_residual:.3e}")
            print(f"gradient norm at block 0, without residuals: {without_residual:.3e}")
            print(f"ratio: {with_residual / max(without_residual, 1e-30):.1f}x")
            """,
            solution_code="""
            class TransformerBlock(nn.Module):
                def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, residual=True):
                    super().__init__()
                    d_ff = d_ff or 4 * d_model
                    self.residual = residual
                    self.norm1 = nn.LayerNorm(d_model)
                    self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
                    self.norm2 = nn.LayerNorm(d_model)
                    self.feed_forward = nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_ff, d_model),
                    )
                    self.dropout = nn.Dropout(dropout)

                def forward(self, x, mask=None):
                    attended, weights = self.attention(self.norm1(x), mask=mask)
                    x = x + self.dropout(attended) if self.residual else self.dropout(attended)
                    fed = self.dropout(self.feed_forward(self.norm2(x)))
                    x = x + fed if self.residual else fed
                    return x, weights

            def first_layer_gradient(residual, depth=8, d_model=64):
                set_seed()
                blocks = nn.ModuleList(
                    TransformerBlock(d_model, 4, dropout=0.0, residual=residual)
                    for _ in range(depth)
                )
                h = torch.randn(4, 16, d_model, requires_grad=True)
                x = h
                for block in blocks:
                    x, _ = block(x)
                x.pow(2).mean().backward()
                return float(
                    sum(p.grad.norm() for p in blocks[0].parameters() if p.grad is not None)
                )

            with_residual = first_layer_gradient(residual=True)
            without_residual = first_layer_gradient(residual=False)
            print(f"gradient norm at block 0, with residuals   : {with_residual:.3e}")
            print(f"gradient norm at block 0, without residuals: {without_residual:.3e}")
            print(f"ratio: {with_residual / max(without_residual, 1e-30):.1f}x")
            print("\\nThe residual gives the gradient a path to the first block that skips")
            print("every intervening transformation. Depth stopped being a barrier in 2015")
            print("(ResNet) for exactly this reason, and Transformers inherited the trick.")
            """,
            check_code="""
            _b = TransformerBlock(32, 4, dropout=0.0)
            _x = torch.randn(2, 6, 32)
            _y, _w = _b(_x)
            assert _y.shape == _x.shape, "a block must preserve its input shape"
            assert with_residual > without_residual, \\
                "residual connections should preserve gradient magnitude"
            print("Basic checks passed.")
            """,
            explanation="""
            `d_ff = 4 * d_model` is the near-universal convention, from the 2017
            paper through GPT-3 to Llama. Most of a Transformer's parameters live in
            these feed-forward layers, not in attention — a fact that surprises
            people who think of the architecture as "the attention model".

            Note that this block preserves its input shape exactly. That is what
            makes `nn.Sequential`-style stacking possible and why scaling a
            Transformer is a matter of changing three numbers.
            """,
        ),
        Task(
            number="6.4",
            title="Transformer versus LSTM on identical windows",
            kind="analysis",
            difficulty=3,
            background="""
            The claim "Transformers beat RNNs" is about *scale and
            parallelism*, not about small-data accuracy. At a few thousand training
            windows on one series, the honest expectation is a tie.
            """,
            instruction="""
            Train a small Transformer forecaster on the same windows chapter 03
            used, compare it against an LSTM under a matched optimisation budget,
            and report both accuracy and wall-clock time.
            """,
            requirements=(
                "Both models see identical data, identical scaling and the same "
                "number of epochs.",
                "Report test MAE in MW and training time for each.",
                "State a conclusion the numbers support — including if it is 'no "
                "meaningful difference'.",
            ),
            hints=(
                "Keep both small: `d_model=64`, 2 layers, 4 heads is plenty here.",
                "`train_model(model, train_loader, valid_loader, loss_fn, config)` "
                "returns a `History` with `best_val_loss` and `seconds`.",
            ),
            expected="""
            Comparable accuracy. The Transformer's advantage in this regime is
            *not* accuracy — say so plainly rather than manufacturing a winner.
            """,
            exercise_code="""
            import pandas as pd

            from ai_power_course.models.forecasters import LSTMForecaster, TransformerForecaster

            EPOCHS = scaled(full=12, fast=2)
            train_loader = make_loader(xs_train, ys_train, batch_size=64, shuffle=True)
            valid_loader = make_loader(xs_valid, ys_valid, batch_size=256)

            def fit_and_score(model, name):
                set_seed()
                history = train_model(
                    model, train_loader, valid_loader, loss_fn=nn.MSELoss(),
                    config=TrainConfig(epochs=EPOCHS, learning_rate=1e-3,
                                       patience=4, verbose=False),
                )
                model.eval()
                with torch.no_grad():
                    scaled_out = model(torch.tensor(xs_test, dtype=torch.float32)).numpy()
                prediction = y_scaler.inverse_transform(scaled_out)
                return {
                    "model": name,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "MAE [MW]": mae(y_test.ravel(), prediction.ravel()),
                    "train [s]": history.seconds,
                }

            # TODO: build both models and score them
            rows = [
                fit_and_score(____, "LSTM"),
                fit_and_score(____, "Transformer"),
            ]
            comparison = pd.DataFrame(rows).set_index("model")
            display(comparison.round(1))
            """,
            solution_code="""
            import pandas as pd

            from ai_power_course.models.forecasters import LSTMForecaster, TransformerForecaster

            EPOCHS = scaled(full=12, fast=2)
            train_loader = make_loader(xs_train, ys_train, batch_size=64, shuffle=True)
            valid_loader = make_loader(xs_valid, ys_valid, batch_size=256)

            def fit_and_score(model, name):
                set_seed()
                history = train_model(
                    model, train_loader, valid_loader, loss_fn=nn.MSELoss(),
                    config=TrainConfig(epochs=EPOCHS, learning_rate=1e-3,
                                       patience=4, verbose=False),
                )
                model.eval()
                with torch.no_grad():
                    scaled_out = model(torch.tensor(xs_test, dtype=torch.float32)).numpy()
                prediction = y_scaler.inverse_transform(scaled_out)
                return {
                    "model": name,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "MAE [MW]": mae(y_test.ravel(), prediction.ravel()),
                    "train [s]": history.seconds,
                }

            rows = [
                fit_and_score(
                    LSTMForecaster(n_channels=1, hidden_size=64, horizon=HORIZON), "LSTM"
                ),
                fit_and_score(
                    TransformerForecaster(context=CONTEXT, horizon=HORIZON, n_channels=1,
                                          d_model=64, n_heads=4, n_layers=2),
                    "Transformer",
                ),
            ]
            comparison = pd.DataFrame(rows).set_index("model")
            display(comparison.round(1))

            spread = comparison["MAE [MW]"].max() / comparison["MAE [MW]"].min() - 1
            print(f"\\nThe two differ by {spread:.1%} in MAE.")
            print("\\nAt this scale that is not a result. The Transformer's real advantages")
            print("are that it trains in parallel over the sequence instead of stepping")
            print("through it, that its path length between any two positions is 1 rather")
            print("than O(n), and that it keeps improving as data and parameters grow.")
            print("None of those are visible on a few thousand windows of one series —")
            print("which is exactly why chapters 09 and 10 change the data, not the model.")
            """,
            check_code="""
            assert set(comparison.index) == {"LSTM", "Transformer"}
            assert (comparison["MAE [MW]"] > 0).all()
            assert comparison["MAE [MW]"].max() < 20000, "check the inverse transform"
            print("Basic checks passed.")
            """,
            explanation="""
            The comparison is only meaningful because the optimisation budget is
            matched: identical epochs, identical learning rate, identical loaders,
            identical seed before each fit. Comparing a model trained for 12 epochs
            against one trained for 100 measures the budget, not the architecture.

            `set_seed()` inside `fit_and_score` rather than once at the top is
            deliberate — otherwise the second model inherits whatever random state
            the first one left behind, and the comparison is not reproducible in
            isolation.
            """,
        ),
    ),
)

# --------------------------------------------------------------------------- 07

CHAPTER_07 = Chapter(
    number=7,
    title="Language Models",
    tutorial="07_language_models.ipynb",
    intro="""
    Same architecture as chapter 06, one change: a causal mask and a
    discrete vocabulary. The training signal — predict the next token — needs no
    labels at all, which is what makes the internet a training set.

    The corpus here is a synthetic grid-operations handbook. It is far too small
    to produce a good model, and that is instructive: you will watch a model
    memorise rather than generalise, and see exactly what scale buys.
    """,
    setup_code="""
    import numpy as np
    import torch
    from torch import nn

    from ai_power_course.config import scaled, set_seed
    from ai_power_course.corpus import HANDBOOK, generate_corpus
    from ai_power_course.models.tinygpt import CharTokenizer, TinyGPT, TinyGPTConfig, generate

    set_seed()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    corpus = generate_corpus(n_logs=500, n_reports=140, n_assets=140)
    print(f"corpus: {len(corpus):,} characters, {len(corpus.split()):,} words")
    print(f"\\n{corpus[:320]}...")
    """,
    tasks=(
        Task(
            number="7.1",
            title="Tokenize, and build next-token batches",
            kind="coding",
            difficulty=2,
            background="""
            A language model sees integers. The tokenizer decides what an integer
            means, and that choice determines the sequence length, the vocabulary
            size and whether unseen words are even representable.

            Character level keeps the vocabulary near 70 symbols so a tiny model
            can learn something, and makes the step completely transparent.
            """,
            instruction="""
            Build a character tokenizer, encode the corpus, and write the batching
            function that produces `(input, target)` pairs for next-token
            prediction.
            """,
            requirements=(
                "`encode` / `decode` must round-trip exactly.",
                "`get_batch(data, batch_size, context)` returns two tensors of "
                "shape `(batch_size, context)`.",
                "The target is the input shifted by exactly one position.",
                "Hold out the last 10% of the corpus for validation.",
            ),
            hints=(
                "For a start index `i`, the input is `data[i : i+context]` and the "
                "target is `data[i+1 : i+1+context]`.",
                "The largest legal start index is `len(data) - context - 1`.",
            ),
            expected="""
            A vocabulary of roughly 70 characters, an exact round-trip, and a batch
            whose target is visibly the input shifted left by one.
            """,
            exercise_code="""
            tokenizer = CharTokenizer(corpus)
            print(f"vocabulary: {tokenizer.vocab_size} characters")
            print("".join(tokenizer.chars).replace("\\n", "\\\\n"))

            # TODO 1: encode the corpus and split off the last 10% for validation
            data = torch.tensor(____, dtype=torch.long)
            split_at = ____
            train_data, valid_data = ____, ____
            print(f"\\ntrain {len(train_data):,} tokens | validation {len(valid_data):,} tokens")

            CONTEXT = 128

            def get_batch(data, batch_size=32, context=CONTEXT, generator=None):
                \"\"\"Random windows and their one-step-shifted targets.\"\"\"
                # TODO 2: draw `batch_size` random start indices, then stack
                starts = ____
                x = ____
                y = ____
                return x, y

            gen = torch.Generator().manual_seed(0)
            xb, yb = get_batch(train_data, batch_size=4, generator=gen)
            print(f"\\ninput {tuple(xb.shape)}   target {tuple(yb.shape)}")
            print(f"\\ninput  : {tokenizer.decode(xb[0][:60])!r}")
            print(f"target : {tokenizer.decode(yb[0][:60])!r}")
            """,
            solution_code="""
            tokenizer = CharTokenizer(corpus)
            print(f"vocabulary: {tokenizer.vocab_size} characters")
            print("".join(tokenizer.chars).replace("\\n", "\\\\n"))

            data = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
            split_at = int(0.9 * len(data))
            train_data, valid_data = data[:split_at], data[split_at:]
            print(f"\\ntrain {len(train_data):,} tokens | validation {len(valid_data):,} tokens")

            CONTEXT = 128

            def get_batch(data, batch_size=32, context=CONTEXT, generator=None):
                \"\"\"Random windows and their one-step-shifted targets.\"\"\"
                starts = torch.randint(
                    0, len(data) - context - 1, (batch_size,), generator=generator
                )
                x = torch.stack([data[s : s + context] for s in starts])
                y = torch.stack([data[s + 1 : s + 1 + context] for s in starts])
                return x, y

            gen = torch.Generator().manual_seed(0)
            xb, yb = get_batch(train_data, batch_size=4, generator=gen)
            print(f"\\ninput {tuple(xb.shape)}   target {tuple(yb.shape)}")
            print(f"\\ninput  : {tokenizer.decode(xb[0][:60])!r}")
            print(f"target : {tokenizer.decode(yb[0][:60])!r}")
            """,
            check_code="""
            _text = "Bus voltage at Nordfeld fell to 0.94 per unit."
            assert tokenizer.decode(tokenizer.encode(_text)) == _text, "round-trip failed"
            assert xb.shape == yb.shape == (4, CONTEXT)
            assert torch.equal(xb[0, 1:], yb[0, :-1]), "the target must be the input shifted by 1"
            print("Basic checks passed.")
            """,
            explanation="""
            The shift by one is the entire supervision signal, and it produces
            `context` training examples per window rather than one: position 0
            predicts position 1, position 1 predicts position 2, and so on, all in
            a single forward pass. That is teacher forcing, and it is why language
            models are so data-efficient per unit of compute.

            Note the corpus split is chronological here only by accident — for text
            the concern is not temporal leakage but *duplication*. Real corpora are
            deduplicated before splitting, because a passage appearing in both
            halves turns validation perplexity into a memorisation score.
            """,
        ),
        Task(
            number="7.2",
            depends_on=("7.1",),
            title="The loss, and what perplexity means",
            kind="coding",
            difficulty=2,
            background="""
            The loss is one line: cross-entropy between the logits at each
            position and the token that actually followed.

            Perplexity is $e^{\\text{loss}}$, and it has a reading: *the effective
            number of equally likely choices the model is deciding between*. A
            uniform model over a 70-character vocabulary has perplexity 70.
            """,
            instruction="""
            Instantiate a TinyGPT, compute its loss before any training, and check
            that the perplexity matches the vocabulary size.
            """,
            requirements=(
                "Predict the untrained loss from the vocabulary size **before** "
                "running the model.",
                "Verify the prediction to within 10%.",
                "Report the parameter count.",
            ),
            hints=(
                "An untrained model is roughly uniform over the vocabulary, so the "
                "cross-entropy should be near `log(vocab_size)`.",
                "`TinyGPT.forward(idx, targets)` returns `(logits, loss)`.",
            ),
            expected="""
            A loss near `log(70) ≈ 4.25` and a perplexity near 70. If it is far off,
            the initialisation or the target alignment is wrong.
            """,
            exercise_code="""
            import math

            # TODO 1: predict the untrained loss and perplexity from first principles
            predicted_loss = ____
            predicted_perplexity = ____
            print(f"predicted untrained loss: {predicted_loss:.3f} "
                  f"(perplexity {predicted_perplexity:.1f})")

            set_seed()
            config = TinyGPTConfig(
                vocab_size=tokenizer.vocab_size, context=CONTEXT,
                d_model=128, n_heads=4, n_layers=4, dropout=0.1,
            )
            model = TinyGPT(config)
            print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

            # TODO 2: measure it
            xb, yb = get_batch(train_data, batch_size=16, generator=gen)
            with torch.no_grad():
                logits, loss = ____
            measured_perplexity = ____
            print(f"measured untrained loss : {float(loss):.3f} "
                  f"(perplexity {measured_perplexity:.1f})")
            """,
            solution_code="""
            import math

            predicted_loss = math.log(tokenizer.vocab_size)
            predicted_perplexity = float(tokenizer.vocab_size)
            print(f"predicted untrained loss: {predicted_loss:.3f} "
                  f"(perplexity {predicted_perplexity:.1f})")

            set_seed()
            config = TinyGPTConfig(
                vocab_size=tokenizer.vocab_size, context=CONTEXT,
                d_model=128, n_heads=4, n_layers=4, dropout=0.1,
            )
            model = TinyGPT(config)
            print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

            xb, yb = get_batch(train_data, batch_size=16, generator=gen)
            with torch.no_grad():
                logits, loss = model(xb, yb)
            measured_perplexity = math.exp(float(loss))
            print(f"measured untrained loss : {float(loss):.3f} "
                  f"(perplexity {measured_perplexity:.1f})")
            print(f"\\nlogits {tuple(logits.shape)} — one distribution over the vocabulary")
            print("at every position, which is why one sequence gives `context` training")
            print("signals rather than one.")
            """,
            check_code="""
            assert abs(float(loss) - predicted_loss) < 0.1 * predicted_loss, \\
                "an untrained model should be close to uniform"
            assert logits.shape == (16, CONTEXT, tokenizer.vocab_size)
            print("Basic checks passed.")
            """,
            explanation="""
            Predicting the untrained loss before measuring it is a habit worth
            keeping. It catches initialisation bugs immediately: a loss far above
            `log(V)` means the output weights are too large, and a loss far below it
            at step zero means the targets are leaking into the input.

            Perplexity is a rescaling of the loss, nothing more. It is worth
            reporting because it is interpretable, and worth distrusting across
            papers because it depends entirely on the tokenizer — a character-level
            perplexity and a subword perplexity are not comparable numbers.
            """,
        ),
        Task(
            number="7.3",
            depends_on=("7.2",),
            title="Train the model and watch it memorise",
            kind="coding",
            difficulty=2,
            background="""
            A few hundred thousand characters is not a corpus. The model will fit
            it and then start reciting it, and seeing that happen is more useful
            than any explanation of why scale matters.
            """,
            instruction="""
            Write the training loop, track the training and validation loss, and
            generate a sample from the model before and after.
            """,
            requirements=(
                "Estimate the validation loss periodically under `torch.no_grad()`.",
                "Clip gradients to a maximum norm of 1.0.",
                "Generate a sample from an untrained and a trained model, and show "
                "both.",
            ),
            hints=(
                "`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` goes "
                "between `backward()` and `step()`.",
                "`generate(model, idx, max_new_tokens=..., temperature=...)` "
                "expects `idx` of shape `(batch, length)`.",
            ),
            expected="""
            The training loss should fall well below the validation loss — that gap
            *is* the memorisation. The generated text should acquire the shape of
            the corpus (substation names, per-unit voltages, MW figures) without
            necessarily meaning anything.
            """,
            exercise_code="""
            prompt = torch.tensor([tokenizer.encode("Bus voltage at ")], dtype=torch.long)
            before = generate(model, prompt, max_new_tokens=120, temperature=0.8,
                              generator=torch.Generator().manual_seed(0))
            print("BEFORE TRAINING")
            print(tokenizer.decode(before[0]))

            STEPS = scaled(full=1200, fast=60)
            EVAL_EVERY = max(1, STEPS // 8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            gen = torch.Generator().manual_seed(1)

            @torch.no_grad()
            def estimate_loss(data, batches=8):
                model.eval()
                losses = []
                for _ in range(batches):
                    x, y = get_batch(data, batch_size=16, generator=gen)
                    losses.append(float(model(x, y)[1]))
                model.train()
                return float(np.mean(losses))

            loss_log = []
            for step in range(1, STEPS + 1):
                x, y = get_batch(train_data, batch_size=16, generator=gen)
                # TODO: forward, backward with gradient clipping, step
                _logits, loss = ____
                ____
                ____
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                ____
                if step % EVAL_EVERY == 0 or step == 1:
                    row = {"step": step, "train": estimate_loss(train_data),
                           "validation": estimate_loss(valid_data)}
                    loss_log.append(row)
                    print(f"step {step:5d}  train {row['train']:.3f}  "
                          f"validation {row['validation']:.3f}")

            after = generate(model, prompt, max_new_tokens=200, temperature=0.8,
                             generator=torch.Generator().manual_seed(0))
            print("\\nAFTER TRAINING")
            print(tokenizer.decode(after[0]))
            """,
            solution_code="""
            prompt = torch.tensor([tokenizer.encode("Bus voltage at ")], dtype=torch.long)
            before = generate(model, prompt, max_new_tokens=120, temperature=0.8,
                              generator=torch.Generator().manual_seed(0))
            print("BEFORE TRAINING")
            print(tokenizer.decode(before[0]))

            STEPS = scaled(full=1200, fast=60)
            EVAL_EVERY = max(1, STEPS // 8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            gen = torch.Generator().manual_seed(1)

            @torch.no_grad()
            def estimate_loss(data, batches=8):
                model.eval()
                losses = []
                for _ in range(batches):
                    x, y = get_batch(data, batch_size=16, generator=gen)
                    losses.append(float(model(x, y)[1]))
                model.train()
                return float(np.mean(losses))

            loss_log = []
            for step in range(1, STEPS + 1):
                x, y = get_batch(train_data, batch_size=16, generator=gen)
                _logits, loss = model(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % EVAL_EVERY == 0 or step == 1:
                    row = {"step": step, "train": estimate_loss(train_data),
                           "validation": estimate_loss(valid_data)}
                    loss_log.append(row)
                    print(f"step {step:5d}  train {row['train']:.3f}  "
                          f"validation {row['validation']:.3f}")

            after = generate(model, prompt, max_new_tokens=200, temperature=0.8,
                             generator=torch.Generator().manual_seed(0))
            print("\\nAFTER TRAINING")
            print(tokenizer.decode(after[0]))

            gap = loss_log[-1]["validation"] - loss_log[-1]["train"]
            print(f"\\nvalidation minus train: {gap:+.3f} nats — that gap is memorisation,")
            print("and it is what a corpus of this size buys you. The fix is not a better")
            print("optimiser; it is three more orders of magnitude of text.")
            """,
            check_code="""
            assert loss_log[-1]["train"] < loss_log[0]["train"], "the training loss did not fall"
            assert loss_log[-1]["train"] < math.log(tokenizer.vocab_size), \\
                "the model should beat a uniform distribution"
            print("Basic checks passed.")
            """,
            explanation="""
            Gradient clipping earns its place in language-model training
            specifically: a rare token combination can produce an enormous gradient
            that moves the weights far enough to destroy several thousand steps of
            progress in one update. Clipping the *norm* rather than the individual
            values preserves the direction and only limits the step length.

            `weight_decay=0.1` is much higher than the usual `1e-2` and is the
            standard GPT-family setting. With a small corpus it slows memorisation
            slightly; it does not prevent it.
            """,
        ),
        Task(
            number="7.4",
            depends_on=("7.3",),
            title="Temperature and top-k sampling",
            kind="coding",
            difficulty=2,
            background="""
            The model outputs a distribution. Turning it into text is a separate
            decision, and it has as much effect on what you read as the weights do.

            Temperature rescales the logits before the softmax: below 1 sharpens,
            above 1 flattens, 0 is greedy. Top-k truncates to the `k` most likely
            tokens.
            """,
            instruction="""
            Generate from the trained model at several temperatures and with top-k
            truncation, then quantify the diversity of each setting rather than
            judging by eye.
            """,
            requirements=(
                "At least four settings, including greedy (`temperature=0`).",
                "Report a numeric diversity measure per setting, not only the text.",
                "Use the same seed for every setting so the comparison is fair.",
            ),
            hints=(
                "The fraction of unique 5-character sequences is a reasonable "
                "diversity proxy.",
                "`generate(..., top_k=20)` truncates the distribution to the 20 "
                "most likely next characters at each step.",
            ),
            expected="""
            Greedy decoding should loop or repeat. High temperature should be
            diverse and less coherent. The interesting question is where the
            trade-off sits, not which end is 'right'.
            """,
            exercise_code="""
            import pandas as pd

            def diversity(text, n=5):
                \"\"\"Fraction of character n-grams that are unique.\"\"\"
                grams = [text[i : i + n] for i in range(len(text) - n)]
                return len(set(grams)) / max(len(grams), 1)

            settings = [
                {"temperature": 0.0, "top_k": None, "label": "greedy (T=0)"},
                {"temperature": 0.5, "top_k": None, "label": "T=0.5"},
                {"temperature": 1.0, "top_k": None, "label": "T=1.0"},
                {"temperature": 1.5, "top_k": None, "label": "T=1.5"},
                {"temperature": 1.0, "top_k": 10, "label": "T=1.0, top-k=10"},
            ]

            rows = []
            for setting in settings:
                # TODO: generate 220 tokens with these settings, seeded identically
                sample = ____
                text = tokenizer.decode(sample[0])
                rows.append({"setting": setting["label"], "diversity": diversity(text)})
                print(f"--- {setting['label']} ---")
                print(text[:200].replace("\\n", " | "))
                print()

            display(pd.DataFrame(rows).set_index("setting").round(3))
            """,
            solution_code="""
            import pandas as pd

            def diversity(text, n=5):
                \"\"\"Fraction of character n-grams that are unique.\"\"\"
                grams = [text[i : i + n] for i in range(len(text) - n)]
                return len(set(grams)) / max(len(grams), 1)

            settings = [
                {"temperature": 0.0, "top_k": None, "label": "greedy (T=0)"},
                {"temperature": 0.5, "top_k": None, "label": "T=0.5"},
                {"temperature": 1.0, "top_k": None, "label": "T=1.0"},
                {"temperature": 1.5, "top_k": None, "label": "T=1.5"},
                {"temperature": 1.0, "top_k": 10, "label": "T=1.0, top-k=10"},
            ]

            rows = []
            for setting in settings:
                sample = generate(
                    model, prompt, max_new_tokens=220,
                    temperature=setting["temperature"], top_k=setting["top_k"],
                    generator=torch.Generator().manual_seed(0),
                )
                text = tokenizer.decode(sample[0])
                rows.append({"setting": setting["label"], "diversity": diversity(text)})
                print(f"--- {setting['label']} ---")
                print(text[:200].replace("\\n", " | "))
                print()

            table = pd.DataFrame(rows).set_index("setting")
            display(table.round(3))
            print("\\nNone of these settings changed a single weight. Decoding is a knob on")
            print("the same model, and a paper that reports 'the model produces X' without")
            print("stating its decoding parameters has not told you enough to reproduce it.")
            """,
            check_code="""
            assert len(rows) == len(settings)
            assert table.loc["greedy (T=0)", "diversity"] <= table.loc["T=1.5", "diversity"], \\
                "greedy decoding should not be more diverse than T=1.5"
            print("Basic checks passed.")
            """,
            explanation="""
            Greedy decoding is deterministic but not optimal: taking the most
            likely token at every step does not produce the most likely
            *sequence*, and it falls into loops because a repeated phrase is
            locally high-probability at every position.

            Top-k and top-p both exist to cut the tail. The tail holds thousands of
            individually unlikely tokens whose probability mass collectively is
            large, so unrestricted sampling wanders into it regularly. Top-p adapts
            to the model's confidence; top-k does not.
            """,
        ),
        Task(
            number="7.5",
            title="Is next-token prediction 'just autocomplete'?",
            kind="reflection",
            difficulty=3,
            background="""
            The dismissal is common and it is not stupid: the training objective
            genuinely is nothing more than predicting the next token. The question
            is what that objective forces a model to represent.
            """,
            instruction="""
            Make the strongest case you can for *both* readings. Then take a
            position: for a grid operations assistant, which reading should
            determine how much you trust the system, and what would you measure to
            decide?
            """,
            expected="""
            A good answer takes the deflationary reading seriously instead of
            dismissing it, and lands on something measurable rather than on a
            definition.
            """,
            answer_template="""
            **The case for 'just autocomplete'.** …

            **The case against.** …

            **My position, and what I would measure.** …
            """,
            answer="""
            **The case for 'just autocomplete'.** The objective is exactly
            $\\max \\sum_t \\log p(x_t \\mid x_{<t})$, and nothing in it references
            truth, intent or the world. The model has no mechanism for checking a
            claim against reality; it has a mechanism for continuing text
            plausibly. That is why it produces a fluent, correctly formatted,
            entirely fabricated protection setting with the same confidence as a
            correct one — the failure is not a bug in an otherwise truth-seeking
            system, it is the system working as specified. Chapter 07's own model
            makes this vivid: it learned the *shape* of an operations log without
            learning a single fact about the grid.

            **The case against.** "Just" is doing a lot of work. Predicting the
            next token well *at scale* requires representing whatever the text
            depends on. To continue "the fault current at the 110 kV busbar was
            calculated as" accurately, something has to encode the relationship
            between short-circuit power and impedance. The objective is simple; the
            function class that minimises it is not, and we have empirical evidence
            — in-context learning, chain-of-thought, transfer to unseen tasks —
            that large models acquire structure nobody put there deliberately. The
            same argument applies to evolution: a simple objective, arbitrarily
            complex solutions.

            **My position, and what I would measure.** For a grid operations
            assistant the dispute is beside the point, because both readings agree
            on the operational fact: *fluency is uncorrelated with correctness, and
            the model gives you no signal about which you are getting*. Whether the
            model "understands" is unfalsifiable; whether it is reliable is not.

            So I would measure three things, none of which requires resolving the
            philosophy. First, **accuracy on questions with verifiable answers**
            drawn from our own documentation — and separately for questions whose
            answers appear in the pretraining corpus versus questions about our
            internal standards, because the gap between those two is the whole
            risk. Second, **calibration**: does stated or token-level confidence
            predict correctness? An uncalibrated model is unusable in an operational
            loop regardless of its average accuracy. Third, **behaviour under
            adversarial premises** — ask about a piece of equipment that does not
            exist and see whether it invents a rating.

            A system that scores well on all three is safe to deploy behind a human
            reviewer whatever we call its inner life. A system that fails the third
            is not, however impressive its prose.
            """,
        ),
    ),
)

# --------------------------------------------------------------------------- 08

CHAPTER_08 = Chapter(
    number=8,
    title="Pretrained LLMs and Adaptation",
    tutorial="08_pretrained_llms_and_adaptation.ipynb",
    intro="""
    You stop training models and start adapting one. The spectrum runs from
    zero-cost (embeddings, prompting) through cheap (linear probe, LoRA) to
    expensive (full fine-tuning), and the engineering skill is choosing the
    cheapest point that solves the problem.

    The task: classify grid operations log entries into `routine`, `overload`,
    `voltage_violation` and `outage`.

    **Requires a network connection** the first time, to download a 23 M-parameter
    encoder. Under `AI_POWER_COURSE_OFFLINE=1` these cells report and skip.
    """,
    setup_code="""
    import warnings

    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split

    from ai_power_course.config import offline_mode, scaled, set_seed
    from ai_power_course.corpus import EVENT_CLASSES, generate_event_dataset
    from ai_power_course.metrics import classification_metrics

    warnings.filterwarnings("ignore")
    set_seed()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 22.7 M params, Apache-2.0

    texts, labels = generate_event_dataset(n_per_class=scaled(full=220, fast=60))
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=0, stratify=labels
    )
    print(f"{len(texts)} log entries across {len(EVENT_CLASSES)} classes: "
          f"{', '.join(EVENT_CLASSES)}")
    print(f"train {len(train_texts)} | test {len(test_texts)}")

    tokenizer = encoder = None
    if offline_mode():
        print("\\nAI_POWER_COURSE_OFFLINE is set — the download-dependent tasks will skip.")
    else:
        import transformers
        from transformers import AutoModel, AutoTokenizer

        transformers.logging.set_verbosity_error()
        tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
        encoder = AutoModel.from_pretrained(ENCODER_NAME).eval()
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"\\n{ENCODER_NAME}: {n_params:,} parameters, "
              f"{encoder.config.num_hidden_layers} layers, "
              f"hidden {encoder.config.hidden_size}")
        print("That is the chapter-06 architecture with weights someone else paid for.")
    """,
    tasks=(
        Task(
            number="8.1",
            title="Mean pooling that respects the attention mask",
            kind="coding",
            difficulty=2,
            background="""
            An encoder returns one vector per token. To get one vector per
            sentence you average them — but a batch is padded to its longest
            member, and averaging the padding in dilutes every short sentence by a
            different amount.

            This is the single most common bug in home-made embedding code, and it
            is invisible: the shapes are right and the numbers look reasonable.
            """,
            instruction="""
            Implement mask-aware mean pooling with L2 normalisation, then
            demonstrate the bug by comparing against the naive version.
            """,
            requirements=(
                "`embed(texts)` returns `(n_texts, 384)`, L2-normalised.",
                "Run under `torch.no_grad()` — nothing is being trained.",
                "Show numerically that naive pooling changes a sentence's "
                "embedding depending on what it was batched with.",
            ),
            hints=(
                "`tokenizer(texts, padding=True, truncation=True, "
                "return_tensors='pt')` gives `input_ids` and `attention_mask`.",
                "Expand the mask to the hidden width, multiply, sum over the token "
                "axis, and divide by the mask's sum — clamped away from zero.",
            ),
            expected="""
            Correct pooling gives an identical embedding for a sentence regardless
            of its batch. Naive pooling does not, and the difference should be large
            enough to matter.
            """,
            exercise_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                @torch.no_grad()
                def embed(texts, batch_size=64, masked=True):
                    \"\"\"Mean-pooled, L2-normalised sentence embeddings.\"\"\"
                    out = []
                    for start in range(0, len(texts), batch_size):
                        chunk = list(texts[start : start + batch_size])
                        encoded = tokenizer(chunk, padding=True, truncation=True,
                                            max_length=128, return_tensors="pt")
                        hidden = encoder(**encoded).last_hidden_state
                        if masked:
                            # TODO 1: mask-aware mean over the token axis
                            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                            pooled = ____
                        else:
                            pooled = hidden.mean(dim=1)   # the bug, for comparison
                        # TODO 2: L2-normalise
                        out.append(____)
                    return torch.cat(out).numpy()

                probe = "Scheduled inspection completed on the overhead line at Nordfeld."
                padded_batch = [probe, "A" * 400]        # forces heavy padding
                alone = embed([probe])[0]
                batched_ok = embed(padded_batch, masked=True)[0]
                batched_bad = embed(padded_batch, masked=False)[0]

                alone_bad = embed([probe], masked=False)[0]
                print(f"mask-aware, alone vs batched : "
                      f"max difference {np.abs(alone - batched_ok).max():.2e}")
                print(f"naive,      alone vs batched : "
                      f"max difference {np.abs(alone_bad - batched_bad).max():.2e}")

                X_train = embed(train_texts)
                X_test = embed(test_texts)
                print(f"\\nembeddings: {X_train.shape}")
            """,
            solution_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                @torch.no_grad()
                def embed(texts, batch_size=64, masked=True):
                    \"\"\"Mean-pooled, L2-normalised sentence embeddings.\"\"\"
                    out = []
                    for start in range(0, len(texts), batch_size):
                        chunk = list(texts[start : start + batch_size])
                        encoded = tokenizer(chunk, padding=True, truncation=True,
                                            max_length=128, return_tensors="pt")
                        hidden = encoder(**encoded).last_hidden_state
                        if masked:
                            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                        else:
                            pooled = hidden.mean(dim=1)   # the bug, for comparison
                        out.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
                    return torch.cat(out).numpy()

                probe = "Scheduled inspection completed on the overhead line at Nordfeld."
                padded_batch = [probe, "A" * 400]        # forces heavy padding
                alone = embed([probe])[0]
                batched_ok = embed(padded_batch, masked=True)[0]
                batched_bad = embed(padded_batch, masked=False)[0]

                alone_bad = embed([probe], masked=False)[0]
                print(f"mask-aware, alone vs batched : "
                      f"max difference {np.abs(alone - batched_ok).max():.2e}")
                print(f"naive,      alone vs batched : "
                      f"max difference {np.abs(alone_bad - batched_bad).max():.2e}")

                X_train = embed(train_texts)
                X_test = embed(test_texts)
                print(f"\\nembeddings: {X_train.shape}")
                print("\\nThe naive version makes a sentence's representation depend on its")
                print("neighbours in the batch. Nothing errors, the accuracy just quietly")
                print("drops and never recovers.")
            """,
            check_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                assert X_train.shape[1] == encoder.config.hidden_size
                assert np.allclose(np.linalg.norm(X_train, axis=1), 1.0, atol=1e-5), \\
                    "embeddings should be L2-normalised"
                assert np.abs(alone - batched_ok).max() < 1e-5, \\
                    "mask-aware pooling must be batch-independent"
                print("Basic checks passed.")
            """,
            explanation="""
            `.clamp(min=1e-9)` on the mask sum guards against an all-padding row,
            which happens with an empty string and produces `nan` that then
            propagates through everything downstream.

            L2 normalisation makes the dot product equal to cosine similarity,
            which is what nearly all retrieval and clustering code assumes. If you
            skip it, longer texts get larger norms and dominate every similarity
            ranking for reasons that have nothing to do with meaning.
            """,
        ),
        Task(
            number="8.2",
            depends_on=("8.1",),
            title="A linear probe on frozen embeddings",
            kind="coding",
            difficulty=2,
            background="""
            The cheapest useful adaptation: freeze the 23 M-parameter encoder,
            fit a logistic regression on its output. Seconds to train, no
            gradients through the backbone, and the same embeddings serve any
            number of tasks.
            """,
            instruction="""
            Fit a linear probe on the frozen embeddings, evaluate it, and compare
            against a bag-of-words baseline on the same split.
            """,
            requirements=(
                "Use `LogisticRegression` on the embeddings from task 8.1.",
                "Include a `TfidfVectorizer` + logistic regression baseline.",
                "Report accuracy and macro-F1 for both. `classification_metrics` is "
                "binary one-vs-rest, so macro-average it yourself.",
                "Report how many parameters each approach actually fits.",
            ),
            hints=(
                "`classification_metrics(y, p, positive_label=k)` scores class `k` "
                "against all the others and returns `F1` among its keys.",
                "The probe fits `384 * 4 + 4` parameters. The backbone's 22.7 M "
                "are frozen.",
            ),
            expected="""
            Both should do well — this is an easy, templated task. Note honestly
            if TF-IDF matches the embeddings, and think about what that says.
            """,
            exercise_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                import pandas as pd
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression

                # TODO 1: the linear probe on frozen embeddings
                probe_model = ____
                probe_model.fit(____, ____)
                probe_pred = ____

                # TODO 2: a bag-of-words baseline on the same split
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
                tfidf_train = vectorizer.fit_transform(train_texts)
                tfidf_test = ____
                tfidf_model = LogisticRegression(max_iter=1000).fit(tfidf_train, train_labels)
                tfidf_pred = ____

                def multiclass_scores(y_true, y_pred):
                    \"\"\"Accuracy, and macro-F1 as the unweighted mean of per-class F1.

                    `classification_metrics` is binary one-vs-rest, so macro-averaging
                    means running it once per class and taking the plain mean — which
                    is exactly what "macro" means, and why it is the right metric when
                    the rare classes are the ones you care about.
                    \"\"\"
                    # TODO 3: run `classification_metrics` once per class, then
                    #         average the F1 scores. Accuracy is the plain match rate.
                    raise NotImplementedError

                rows = []
                for name, pred, n_fitted in [
                    ("linear probe on MiniLM", probe_pred,
                     X_train.shape[1] * len(EVENT_CLASSES) + len(EVENT_CLASSES)),
                    ("TF-IDF + logistic regression", tfidf_pred,
                     tfidf_train.shape[1] * len(EVENT_CLASSES) + len(EVENT_CLASSES)),
                ]:
                    rows.append({"method": name, **multiclass_scores(test_labels, pred),
                                 "fitted params": n_fitted})
                probe_table = pd.DataFrame(rows).set_index("method")
                display(probe_table.round(3))
            """,
            solution_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                import pandas as pd
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression

                probe_model = LogisticRegression(max_iter=1000)
                probe_model.fit(X_train, train_labels)
                probe_pred = probe_model.predict(X_test)

                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
                tfidf_train = vectorizer.fit_transform(train_texts)
                tfidf_test = vectorizer.transform(test_texts)
                tfidf_model = LogisticRegression(max_iter=1000).fit(tfidf_train, train_labels)
                tfidf_pred = tfidf_model.predict(tfidf_test)

                def multiclass_scores(y_true, y_pred):
                    \"\"\"Accuracy, and macro-F1 as the unweighted mean of per-class F1.

                    `classification_metrics` is binary one-vs-rest, so macro-averaging
                    means running it once per class and taking the plain mean — which
                    is exactly what "macro" means, and why it is the right metric when
                    the rare classes are the ones you care about.
                    \"\"\"
                    per_class = [
                        classification_metrics(y_true, y_pred, positive_label=k)
                        for k in range(len(EVENT_CLASSES))
                    ]
                    return {
                        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
                        "macro F1": float(np.mean([m["F1"] for m in per_class])),
                    }

                rows = []
                for name, pred, n_fitted in [
                    ("linear probe on MiniLM", probe_pred,
                     X_train.shape[1] * len(EVENT_CLASSES) + len(EVENT_CLASSES)),
                    ("TF-IDF + logistic regression", tfidf_pred,
                     tfidf_train.shape[1] * len(EVENT_CLASSES) + len(EVENT_CLASSES)),
                ]:
                    rows.append({"method": name, **multiclass_scores(test_labels, pred),
                                 "fitted params": n_fitted})
                probe_table = pd.DataFrame(rows).set_index("method")
                display(probe_table.round(3))

                print("\\nTF-IDF competes here, and that is a finding about the DATA, not")
                print("about embeddings. These log entries come from templates with strongly")
                print("class-specific vocabulary ('tripped', 'per unit', 'percent of its")
                print("rating'), so word counts are nearly sufficient. Real operator free")
                print("text — abbreviations, negations, paraphrase — is where the ordering")
                print("reverses. Always run the cheap baseline before claiming you needed")
                print("the expensive model.")
            """,
            check_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                assert probe_table["accuracy"].min() > 0.7, \\
                    "both methods should do well on this templated task"
                assert len(probe_pred) == len(test_labels)
                print("Basic checks passed.")
            """,
            explanation="""
            Note the parameter counts. The probe fits 1,540 parameters on top of a
            frozen 22.7 M-parameter backbone; TF-IDF fits tens of thousands because
            its feature space is the vocabulary. "Fewer parameters" and "less
            capacity" are not the same thing — the probe is small precisely because
            the representation it sits on is doing the work.

            The honest report here includes the baseline that matched. Publishing
            only the embedding result would not be false, but it would leave the
            reader unable to judge whether the encoder was needed.
            """,
        ),
        Task(
            number="8.3",
            title="LoRA: predict the parameter count, then verify it",
            kind="coding",
            difficulty=3,
            background="""
            LoRA freezes $W_0$ and learns a low-rank correction
            $\\Delta W = BA$ with $B \\in \\mathbb{R}^{d \\times r}$,
            $A \\in \\mathbb{R}^{r \\times k}$ and $r \\ll \\min(d, k)$.

            A $384 \\times 384$ attention projection has 147,456 parameters. Its
            rank-8 correction has $2 \\times 384 \\times 8 = 6{,}144$.
            """,
            instruction="""
            Work out from the architecture how many trainable parameters a LoRA
            configuration will have, then build it with `peft` and check your
            arithmetic against the real model.
            """,
            requirements=(
                "Derive the count by hand **before** building the model.",
                "Target the `query` and `value` projections with `r=8`.",
                "Assert that your formula matches the actual trainable count.",
                "Report the trainable fraction of the whole model.",
            ),
            hints=(
                "Per targeted module: `2 * hidden * r` (the `A` and `B` matrices).",
                "Multiply by the number of targeted modules per layer and by the "
                "number of layers.",
                "`sum(p.numel() for n, p in model.named_parameters() if "
                "p.requires_grad and 'lora' in n)` counts what was actually added.",
            ),
            expected="""
            Your formula should match exactly. A trainable fraction well under 1%.
            """,
            exercise_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                from peft import LoraConfig, TaskType, get_peft_model
                from transformers import AutoModelForSequenceClassification

                hidden = encoder.config.hidden_size
                n_layers = encoder.config.num_hidden_layers
                RANK = 8
                TARGETS = ["query", "value"]

                # TODO 1: predict the trainable LoRA parameter count from the architecture
                expected_lora = ____
                print(f"{n_layers} layers x {len(TARGETS)} modules x 2 matrices x "
                      f"{hidden} x {RANK} = {expected_lora:,} expected")

                set_seed()
                base = AutoModelForSequenceClassification.from_pretrained(
                    ENCODER_NAME, num_labels=len(EVENT_CLASSES)
                )
                # TODO 2: wrap it with LoRA
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=____,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=____,
                )
                lora_model = get_peft_model(base, lora_config)

                actual_lora = sum(p.numel() for n, p in lora_model.named_parameters()
                                  if p.requires_grad and "lora" in n)
                total = sum(p.numel() for p in lora_model.parameters())
                trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
                print(f"actual LoRA parameters: {actual_lora:,}")
                print(f"trainable overall     : {trainable:,} of {total:,} "
                      f"({trainable / total:.2%}) — the rest is the classifier head")
            """,
            solution_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                from peft import LoraConfig, TaskType, get_peft_model
                from transformers import AutoModelForSequenceClassification

                hidden = encoder.config.hidden_size
                n_layers = encoder.config.num_hidden_layers
                RANK = 8
                TARGETS = ["query", "value"]

                expected_lora = n_layers * len(TARGETS) * 2 * hidden * RANK
                print(f"{n_layers} layers x {len(TARGETS)} modules x 2 matrices x "
                      f"{hidden} x {RANK} = {expected_lora:,} expected")

                set_seed()
                base = AutoModelForSequenceClassification.from_pretrained(
                    ENCODER_NAME, num_labels=len(EVENT_CLASSES)
                )
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=RANK,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=TARGETS,
                )
                lora_model = get_peft_model(base, lora_config)

                actual_lora = sum(p.numel() for n, p in lora_model.named_parameters()
                                  if p.requires_grad and "lora" in n)
                total = sum(p.numel() for p in lora_model.parameters())
                trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
                print(f"actual LoRA parameters: {actual_lora:,}")
                print(f"trainable overall     : {trainable:,} of {total:,} "
                      f"({trainable / total:.2%}) — the rest is the classifier head")
                print("\\nAt inference BA folds into W0, so there is no latency penalty, and")
                print("one frozen backbone can serve many tasks by swapping small adapters.")
                print("That is the deployment argument, and it matters more than the")
                print("training-memory argument people usually quote.")
            """,
            check_code="""
            if encoder is None:
                print("Skipped: offline mode.")
            else:
                assert expected_lora == actual_lora, (
                    f"formula gives {expected_lora:,} but the model has {actual_lora:,}"
                )
                assert trainable / total < 0.05, "LoRA should train a small fraction"
                print("Basic checks passed — the formula matches the model exactly.")
            """,
            explanation="""
            The assertion is the point of the task. Reading `print_trainable_parameters()`
            and believing it teaches nothing; deriving the number and finding it
            matches means you know what LoRA actually inserted and where.

            `lora_alpha / r` is the scaling applied to the update, which is why
            `alpha=16, r=8` and `alpha=32, r=16` behave similarly. Changing `r`
            without changing `alpha` changes the effective learning rate of the
            adapter, which is a frequent and confusing source of "LoRA didn't work
            for me".
            """,
        ),
        Task(
            number="8.4",
            title="Which adaptation method, and why?",
            kind="reflection",
            difficulty=2,
            background="""
            The spectrum: embeddings → prompting → in-context learning → RAG →
            linear probe → LoRA → full fine-tuning. Cost, data requirements and
            achievable accuracy all increase along it, and so does the maintenance
            burden.
            """,
            instruction="""
            For each of the four scenarios below, pick a method and justify it in
            two or three sentences. Name the deciding constraint in each case.

            1. 300 labelled operator log entries; a classifier is needed next week.
            2. 50,000 labelled entries; accuracy is worth real money; a GPU is
               available.
            3. No labels; engineers want to search 20 years of maintenance reports
               by meaning, not keyword.
            4. The model must answer questions about internal grid codes that are
               updated quarterly.
            """,
            expected="""
            Case 4 is the one that separates answers. If your instinct is to
            fine-tune on the documents, argue against yourself before committing.
            """,
            answer_template="""
            | Scenario | Method | Deciding constraint |
            | --- | --- | --- |
            | 1. 300 labels, one week | … | … |
            | 2. 50,000 labels, GPU | … | … |
            | 3. No labels, semantic search | … | … |
            | 4. Quarterly-updated grid codes | … | … |

            **Reasoning.**

            1. …
            2. …
            3. …
            4. …
            """,
            answer="""
            | Scenario | Method | Deciding constraint |
            | --- | --- | --- |
            | 1. 300 labels, one week | Linear probe, frozen backbone | Labels and time |
            | 2. 50,000 labels, GPU | LoRA, full only if it wins | Returns vs. upkeep |
            | 3. No labels, search | Embeddings + vector index | Nothing to train on |
            | 4. Grid codes, quarterly | RAG | Knowledge outruns training |

            **Reasoning.**

            1. With 300 examples, fine-tuning 23 M parameters will overfit and you
            will spend the week fighting it. A probe fits about 1,500 parameters in
            seconds, is trivially re-runnable when more labels arrive, and — as task
            8.2 showed — you should run TF-IDF alongside it, because on templated
            text it may match and then you are done.

            2. Enough labels to justify updating the backbone, but LoRA first:
            it usually lands within a point or two of full fine-tuning at a fraction
            of the cost, and it gives you a small artefact to version instead of a
            full model copy per task. Escalate to full fine-tuning only if a
            measured gap justifies it, and if the domain is genuinely far from the
            pretraining distribution.

            3. No labels means no supervised method is even available. Off-the-shelf
            embeddings plus a vector index is the entire system, and it works on day
            one. Spend the effort on chunking and on evaluating retrieval quality
            instead — that is where semantic search actually succeeds or fails.

            4. **Fine-tuning is the wrong tool here, and it is the tempting one.**
            Weights are a bad place to put facts that change: you cannot update one
            clause without retraining, you cannot tell whether the model is quoting
            the current version or a superseded one, and you cannot cite a source.
            RAG keeps the knowledge in a retrievable store where it can be edited,
            versioned and *shown to the user*, and the model does what it is
            actually good at — reading the retrieved passage and expressing it.

            The general rule underneath all four: **fine-tune to change behaviour,
            retrieve to change knowledge.** Format, tone, a classification decision
            boundary, a domain's phrasing — those belong in weights. Facts belong in
            a database.
            """,
        ),
    ),
)

# --------------------------------------------------------------------------- 09

CHAPTER_09 = Chapter(
    number=9,
    title="Foundation Models Beyond LLMs",
    tutorial="09_foundation_models_beyond_llms.ipynb",
    intro="""
    The chapter that makes the course's argument concrete. A time-series
    foundation model has no vocabulary, no tokens in the linguistic sense and
    generates nothing you would call text — and it is unmistakably a foundation
    model: pretrained on a broad distribution, adapted to many downstream tasks
    without task-specific training.

    You will run one zero-shot against specialists trained on the target data.

    **Requires a network connection** the first time. Under
    `AI_POWER_COURSE_OFFLINE=1` the foundation-model cells report and skip, and
    the rest still runs.
    """,
    setup_code="""
    import numpy as np
    import pandas as pd
    import torch

    from ai_power_course.config import fast_mode, offline_mode, scaled, set_seed
    from ai_power_course.data import load_energy_data, time_split
    from ai_power_course.metrics import coverage, crps_from_quantiles, mae, pinball_loss

    set_seed()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    dataset = load_energy_data()
    frame = dataset.frame
    train_raw, valid_raw, test_raw = time_split(frame)

    CONTEXT, HORIZON = 512, 24
    QUANTILES = [0.1, 0.5, 0.9]

    # One forecast per day, issued at 00:00 UTC — a realistic day-ahead cadence.
    origins = test_raw.index[test_raw.index.hour == 0]
    origins = origins[origins >= frame.index[0] + pd.Timedelta(hours=CONTEXT)]
    origins = origins[origins + pd.Timedelta(hours=HORIZON) <= frame.index[-1]]
    origins = origins[: scaled(full=120, fast=12)]

    series = frame.load_mw
    positions = series.index.get_indexer(origins)
    contexts = np.stack([series.to_numpy()[p - CONTEXT + 1 : p + 1] for p in positions])
    targets = np.stack([series.to_numpy()[p + 1 : p + 1 + HORIZON] for p in positions])
    print(f"{len(origins)} daily forecast origins, "
          f"{origins[0]:%Y-%m-%d} to {origins[-1]:%Y-%m-%d}")
    print(f"contexts {contexts.shape}   targets {targets.shape}")
    print(f"\\n{dataset.describe()}")
    """,
    tasks=(
        Task(
            number="9.1",
            title="A shared evaluation harness",
            kind="coding",
            difficulty=2,
            background="""
            Before any model is loaded, fix the protocol. Every method will be
            scored on exactly the same target values, with the same metric, against
            the same reference — otherwise the comparison measures the protocol
            rather than the models.
            """,
            instruction="""
            Build the baseline forecasts and a scoring function that reports MAE
            and skill against the seasonal naive.
            """,
            requirements=(
                "A flat persistence baseline and a 168-hour seasonal naive.",
                "`score(name, prediction)` records into a shared dictionary.",
                "Skill is measured against the seasonal naive, and the reference "
                "must therefore score exactly 0.",
                "Every prediction has shape `(n_origins, HORIZON)`.",
            ),
            hints=(
                "Persistence at all 24 lead times is the value at the origin, "
                "repeated: `np.repeat(contexts[:, -1:], HORIZON, axis=1)`.",
                "Skill = `1 - MAE_model / MAE_reference`.",
            ),
            expected="""
            The seasonal naive should clearly beat flat persistence, and scoring the
            reference against itself must give exactly zero skill.
            """,
            exercise_code="""
            predictions = {}
            scores = {}

            # TODO 1: the two baselines
            predictions["Persistence (flat)"] = ____
            predictions["Seasonal naive (168 h)"] = np.stack(
                [series.to_numpy()[p + 1 - 168 : p + 1 - 168 + HORIZON] for p in positions]
            )

            REFERENCE = "Seasonal naive (168 h)"

            def score(name, prediction=None):
                \"\"\"Record MAE and skill against the reference forecast.\"\"\"
                prediction = predictions[name] if prediction is None else prediction
                if prediction.shape != targets.shape:
                    raise ValueError(f"{name}: expected {targets.shape}, got {prediction.shape}")
                # TODO 2: MAE over all origins and lead times, then skill
                model_mae = ____
                reference_mae = ____
                scores[name] = {"MAE [MW]": model_mae,
                                "skill vs naive": ____}
                return scores[name]

            for name in predictions:
                result = score(name)
                print(f"{name:26s} MAE {result['MAE [MW]']:8,.1f} MW   "
                      f"skill {result['skill vs naive']:+.3f}")
            """,
            solution_code="""
            predictions = {}
            scores = {}

            predictions["Persistence (flat)"] = np.repeat(contexts[:, -1:], HORIZON, axis=1)
            predictions["Seasonal naive (168 h)"] = np.stack(
                [series.to_numpy()[p + 1 - 168 : p + 1 - 168 + HORIZON] for p in positions]
            )

            REFERENCE = "Seasonal naive (168 h)"

            def score(name, prediction=None):
                \"\"\"Record MAE and skill against the reference forecast.\"\"\"
                prediction = predictions[name] if prediction is None else prediction
                if prediction.shape != targets.shape:
                    raise ValueError(f"{name}: expected {targets.shape}, got {prediction.shape}")
                model_mae = mae(targets.ravel(), prediction.ravel())
                reference_mae = mae(targets.ravel(), predictions[REFERENCE].ravel())
                scores[name] = {"MAE [MW]": model_mae,
                                "skill vs naive": 1.0 - model_mae / reference_mae}
                return scores[name]

            for name in predictions:
                result = score(name)
                print(f"{name:26s} MAE {result['MAE [MW]']:8,.1f} MW   "
                      f"skill {result['skill vs naive']:+.3f}")
            """,
            check_code="""
            assert abs(scores[REFERENCE]["skill vs naive"]) < 1e-12, \\
                "the reference must score exactly zero skill against itself"
            assert scores[REFERENCE]["MAE [MW]"] < scores["Persistence (flat)"]["MAE [MW]"], \\
                "the weekly naive should beat flat persistence at a 24 h horizon"
            for _name, _p in predictions.items():
                assert _p.shape == targets.shape, f"{_name} has the wrong shape"
            print("Basic checks passed.")
            """,
            explanation="""
            Fixing the harness before loading any model is not pedantry; it is what
            makes the later comparison trustworthy. The most common way a zero-shot
            claim goes wrong is that the foundation model is evaluated on a slightly
            different set of origins, or at a different lead-time aggregation, than
            the baselines it is being compared against.

            The reference-scores-zero assertion is a cheap, complete check that the
            skill formula is right.
            """,
        ),
        Task(
            number="9.2",
            depends_on=("9.1",),
            title="Run a real foundation model zero-shot",
            kind="coding",
            difficulty=3,
            background="""
            Chronos-2 (Ansari et al., 2025) is a 120 M-parameter encoder-only model
            pretrained on a broad corpus of time series. It has never seen this
            dataset — it is synthetic, generated for this course — so contamination
            is genuinely ruled out here in a way it almost never is on a public
            benchmark.

            No training. No fine-tuning. Hand it 512 numbers, ask for 24.
            """,
            instruction="""
            Load Chronos-2, produce quantile forecasts for every origin, and score
            it in the harness from task 9.1 alongside a specialist trained on this
            data.
            """,
            requirements=(
                "Guard the whole block so it degrades gracefully offline.",
                "Request the quantile levels fixed in `QUANTILES`.",
                "Report inference time per series.",
                "Fit one specialist — gradient boosting, one model per lead time — "
                "on the training split, and score it in the same harness.",
            ),
            hints=(
                "`Chronos2Pipeline.from_pretrained('amazon/chronos-2', "
                "device_map='cpu')`.",
                "`predict_quantiles(list_of_tensors, prediction_length=..., "
                "quantile_levels=...)` returns `(quantiles, mean)`; each quantile "
                "element is `(n_variates, horizon, n_quantiles)`, so index `[0]` for "
                "a univariate series.",
                "The median is `QUANTILES.index(0.5)` along the last axis.",
            ),
            expected="""
            A zero-shot model that is competitive with, and may beat, a specialist
            trained on this exact series. Whichever way it lands, report it — the
            interesting content is the comparison, not a win.
            """,
            exercise_code="""
            import time

            chronos_quantiles = None
            if offline_mode():
                print("Skipped: offline mode.")
            else:
                try:
                    from chronos import Chronos2Pipeline

                    started = time.perf_counter()
                    pipeline = Chronos2Pipeline.from_pretrained(
                        "amazon/chronos-2", device_map="cpu")
                    n_params = sum(p.numel() for p in pipeline.model.parameters())
                    print(f"loaded {n_params / 1e6:.0f} M parameters in "
                          f"{time.perf_counter() - started:.0f}s")

                    started = time.perf_counter()
                    # TODO 1: forecast every context window
                    quantile_list, _mean = pipeline.predict_quantiles(
                        ____,
                        prediction_length=____,
                        quantile_levels=____,
                        batch_size=16,
                    )
                    elapsed = time.perf_counter() - started
                    chronos_quantiles = torch.stack([q[0] for q in quantile_list]).numpy()
                    # TODO 2: the median forecast
                    predictions["Chronos-2 (zero-shot)"] = ____
                    print(f"{len(contexts)} series in {elapsed:.0f}s "
                          f"({elapsed / len(contexts) * 1000:.0f} ms each)")
                    print(score("Chronos-2 (zero-shot)"))
                except Exception as exc:
                    print(f"Chronos-2 unavailable ({type(exc).__name__}: {str(exc)[:160]})")

            # A specialist trained on THIS series, as the fair point of comparison.
            from sklearn.ensemble import HistGradientBoostingRegressor

            from ai_power_course.data import make_supervised

            specialist = np.zeros_like(targets)
            for lead in range(HORIZON):
                X_fit, y_fit = make_supervised(train_raw, horizon=lead + 1,
                                               exogenous=("temperature_c",))
                model = HistGradientBoostingRegressor(max_iter=scaled(full=120, fast=30),
                                                      random_state=0).fit(X_fit, y_fit)
                X_eval, _ = make_supervised(frame, horizon=lead + 1,
                                            exogenous=("temperature_c",))
                specialist[:, lead] = model.predict(X_eval.loc[origins])
            predictions["Gradient boosting (specialist)"] = specialist
            print(score("Gradient boosting (specialist)"))
            """,
            solution_code="""
            import time

            chronos_quantiles = None
            if offline_mode():
                print("Skipped: offline mode.")
            else:
                try:
                    from chronos import Chronos2Pipeline

                    started = time.perf_counter()
                    pipeline = Chronos2Pipeline.from_pretrained(
                        "amazon/chronos-2", device_map="cpu")
                    n_params = sum(p.numel() for p in pipeline.model.parameters())
                    print(f"loaded {n_params / 1e6:.0f} M parameters in "
                          f"{time.perf_counter() - started:.0f}s")

                    started = time.perf_counter()
                    quantile_list, _mean = pipeline.predict_quantiles(
                        [torch.tensor(row, dtype=torch.float32) for row in contexts],
                        prediction_length=HORIZON,
                        quantile_levels=QUANTILES,
                        batch_size=16,
                    )
                    elapsed = time.perf_counter() - started
                    chronos_quantiles = torch.stack([q[0] for q in quantile_list]).numpy()
                    predictions["Chronos-2 (zero-shot)"] = (
                        chronos_quantiles[:, :, QUANTILES.index(0.5)]
                    )
                    print(f"{len(contexts)} series in {elapsed:.0f}s "
                          f"({elapsed / len(contexts) * 1000:.0f} ms each)")
                    print(score("Chronos-2 (zero-shot)"))
                except Exception as exc:  # noqa: BLE001
                    print(f"Chronos-2 unavailable ({type(exc).__name__}: {str(exc)[:160]})")

            # A specialist trained on THIS series, as the fair point of comparison.
            from sklearn.ensemble import HistGradientBoostingRegressor

            from ai_power_course.data import make_supervised

            specialist = np.zeros_like(targets)
            for lead in range(HORIZON):
                X_fit, y_fit = make_supervised(train_raw, horizon=lead + 1,
                                               exogenous=("temperature_c",))
                model = HistGradientBoostingRegressor(max_iter=scaled(full=120, fast=30),
                                                      random_state=0).fit(X_fit, y_fit)
                X_eval, _ = make_supervised(frame, horizon=lead + 1,
                                            exogenous=("temperature_c",))
                specialist[:, lead] = model.predict(X_eval.loc[origins])
            predictions["Gradient boosting (specialist)"] = specialist
            print(score("Gradient boosting (specialist)"))

            display(pd.DataFrame(scores).T.sort_values("MAE [MW]").round(3))
            """,
            check_code="""
            assert "Gradient boosting (specialist)" in scores
            assert scores["Gradient boosting (specialist)"]["skill vs naive"] > 0, \\
                "a trained specialist should beat the naive baseline"
            if chronos_quantiles is not None:
                assert chronos_quantiles.shape == (len(origins), HORIZON, len(QUANTILES))
                _q = chronos_quantiles
                assert (_q[:, :, 0] <= _q[:, :, 2] + 1e-6).all(), \\
                    "the 10% quantile must not exceed the 90% quantile"
            print("Basic checks passed.")
            """,
            explanation="""
            The specialist is fitted per lead time on the *training* split, and
            evaluated at origins drawn from the test split — the feature table is
            built over the whole frame only so the origins can be indexed, and
            `make_supervised` keeps lags origin-relative, so nothing from after
            each origin enters its own features.

            The monotonicity assertion on the quantiles is worth keeping. Quantile
            crossing is a real failure mode of quantile-regression models, and it
            invalidates any interval you build from them.

            One caveat that belongs in every zero-shot claim: "zero-shot" means *no
            gradient updates on this task*. It does not mean the model has never
            seen anything like it. Chronos-2's corpus certainly contains public
            electricity data. On a public benchmark that would be contamination you
            could not rule out; here the series is synthetic, which is the only
            reason the claim is clean.
            """,
        ),
        Task(
            number="9.3",
            depends_on=("9.2",),
            title="Probabilistic evaluation: sharpness and calibration",
            kind="coding",
            difficulty=3,
            background="""
            Grid operations decisions are about risk, not point estimates. Reserve
            procurement asks "how bad could the peak be?", not "what is the
            expected peak?".

            A point metric cannot answer that, and it cannot distinguish a
            well-calibrated wide interval from an overconfident narrow one.
            """,
            instruction="""
            Score the Chronos-2 quantile forecasts with the pinball loss, CRPS,
            and empirical coverage of the 80% interval, and plot one day with the
            interval shaded.
            """,
            requirements=(
                "Pinball loss at each of the three quantile levels.",
                "CRPS approximated from the quantiles.",
                "Empirical coverage of the [0.1, 0.9] interval, compared with the "
                "nominal 80%.",
                "One plot showing the context, the truth, the median and the band.",
            ),
            hints=(
                "`pinball_loss(y_true, y_quantile, quantile)`, "
                "`crps_from_quantiles(y_true, quantiles, levels)` and "
                "`coverage(y_true, lower, upper)` are already imported.",
                "Coverage far below nominal means overconfidence — the interval is "
                "too narrow for the errors the model actually makes.",
            ),
            expected="""
            Coverage somewhere near 80%. Say which direction it misses in: below
            nominal is overconfident, above is conservative, and they have opposite
            operational consequences.
            """,
            exercise_code="""
            import matplotlib.pyplot as plt

            if chronos_quantiles is None:
                print("Skipped: no quantile forecasts available (offline mode).")
            else:
                flat_truth = targets.ravel()
                # TODO 1: pinball loss at each level
                for i, level in enumerate(QUANTILES):
                    loss = ____
                    print(f"pinball loss at q{level:.1f}: {loss:8,.1f} MW")

                # TODO 2: CRPS and coverage of the 80% interval
                crps = ____
                lower = chronos_quantiles[:, :, QUANTILES.index(0.1)].ravel()
                upper = chronos_quantiles[:, :, QUANTILES.index(0.9)].ravel()
                empirical = ____
                print(f"\\nCRPS                     : {crps:8,.1f} MW")
                print(f"80% interval coverage    : {empirical:.1%} (nominal 80.0%)")
                print(f"mean interval width      : {np.mean(upper - lower):8,.1f} MW")

                day = 0
                hours = np.arange(HORIZON)
                fig, ax = plt.subplots(figsize=(9, 3.6))
                ax.plot(np.arange(-48, 0), contexts[day, -48:], color="0.5", label="context")
                ax.plot(hours, targets[day], "o-", color="black", ms=3, label="actual")
                ax.plot(hours, chronos_quantiles[day, :, 1], "s-", ms=3,
                        label="Chronos-2 median")
                ax.fill_between(hours, chronos_quantiles[day, :, 0],
                                chronos_quantiles[day, :, 2], alpha=0.25,
                                label="80% interval")
                ax.axvline(-0.5, color="0.7", lw=1)
                ax.set_xlabel("hours from the forecast origin")
                ax.set_ylabel("load [MW]")
                ax.set_title(f"Zero-shot day-ahead forecast, {origins[day]:%Y-%m-%d}")
                ax.legend(loc="upper left", fontsize=8)
                fig.tight_layout()
                plt.show()
            """,
            solution_code="""
            import matplotlib.pyplot as plt

            if chronos_quantiles is None:
                print("Skipped: no quantile forecasts available (offline mode).")
            else:
                flat_truth = targets.ravel()
                for i, level in enumerate(QUANTILES):
                    loss = pinball_loss(flat_truth, chronos_quantiles[:, :, i].ravel(), level)
                    print(f"pinball loss at q{level:.1f}: {loss:8,.1f} MW")

                crps = crps_from_quantiles(
                    flat_truth,
                    chronos_quantiles.reshape(-1, len(QUANTILES)),
                    QUANTILES,
                )
                lower = chronos_quantiles[:, :, QUANTILES.index(0.1)].ravel()
                upper = chronos_quantiles[:, :, QUANTILES.index(0.9)].ravel()
                empirical = coverage(flat_truth, lower, upper)
                print(f"\\nCRPS                     : {crps:8,.1f} MW")
                print(f"80% interval coverage    : {empirical:.1%} (nominal 80.0%)")
                print(f"mean interval width      : {np.mean(upper - lower):8,.1f} MW")
                if empirical < 0.75:
                    verdict = "too narrow — overconfident"
                elif empirical <= 0.85:
                    verdict = "roughly calibrated"
                else:
                    verdict = "too wide — conservative"
                print(f"\\nThe interval is {verdict}.")

                day = 0
                hours = np.arange(HORIZON)
                fig, ax = plt.subplots(figsize=(9, 3.6))
                ax.plot(np.arange(-48, 0), contexts[day, -48:], color="0.5", label="context")
                ax.plot(hours, targets[day], "o-", color="black", ms=3, label="actual")
                ax.plot(hours, chronos_quantiles[day, :, 1], "s-", ms=3,
                        label="Chronos-2 median")
                ax.fill_between(hours, chronos_quantiles[day, :, 0],
                                chronos_quantiles[day, :, 2], alpha=0.25,
                                label="80% interval")
                ax.axvline(-0.5, color="0.7", lw=1)
                ax.set_xlabel("hours from the forecast origin")
                ax.set_ylabel("load [MW]")
                ax.set_title(f"Zero-shot day-ahead forecast, {origins[day]:%Y-%m-%d}")
                ax.legend(loc="upper left", fontsize=8)
                fig.tight_layout()
                plt.show()

                print("\\nSharpness and calibration trade off, and only one of them can be")
                print("gamed. An infinitely wide interval has perfect coverage and no value;")
                print("a zero-width one is maximally sharp and never right. Report both, or")
                print("report a proper scoring rule like CRPS, which penalises each.")
            """,
            check_code="""
            if chronos_quantiles is None:
                print("Skipped: offline mode.")
            else:
                assert 0.0 <= empirical <= 1.0
                assert crps > 0, "CRPS is a positive loss"
                assert (upper >= lower - 1e-6).all(), "the interval bounds are crossed"
                print("Basic checks passed.")
            """,
            explanation="""
            The pinball loss at q0.1 penalises *over*-prediction ten times more
            heavily than under-prediction, and at q0.9 the asymmetry reverses. That
            asymmetry is what makes each quantile converge to the right place, and
            averaging the pinball loss across a dense set of levels is precisely
            what approximates the CRPS.

            CRPS is a *proper* scoring rule: it is minimised only by the true
            predictive distribution, so it cannot be gamed by widening or narrowing
            the interval. Coverage alone can be — which is why calibration should
            never be reported without sharpness.
            """,
        ),
        Task(
            number="9.4",
            title="What does a foundation model actually change?",
            kind="reflection",
            difficulty=3,
            background="""
            The measured result is likely to be undramatic: a zero-shot model
            roughly level with a specialist trained on the target data. It is worth
            being precise about why that would still be significant, and about what
            it would not establish.
            """,
            instruction="""
            Answer three questions. (a) If the zero-shot model merely ties the
            specialist, what has actually changed for a practitioner? (b) Name two
            situations where you would still choose the specialist. (c) What claim
            would this experiment *not* support, even if the foundation model won
            outright?
            """,
            expected="""
            Part (c) is the one that matters. A single-series, single-horizon
            comparison on synthetic data supports a narrow claim, and naming its
            limits precisely is the skill being tested.
            """,
            answer_template="""
            **(a) What changes if it only ties.** …

            **(b) Two cases for the specialist.** …

            **(c) What this experiment does not support.** …
            """,
            answer="""
            **(a) What changes if it only ties.** The cost structure, completely.
            The specialist needed years of history for this specific series, a
            feature pipeline, a training run per lead time, and a retraining
            schedule for as long as it is in service. The foundation model needed
            512 numbers and a forward pass. At parity of accuracy that is not a tie
            — it is the same result at a small fraction of the engineering cost,
            and it is available on day one for a new substation with no history at
            all. The cold-start case is where this stops being a convenience and
            starts being a capability nobody had.

            **(b) Two cases for the specialist.** First, when the series has
            **structure the foundation model cannot see**: a large industrial
            customer whose load follows a production schedule, or a feeder whose
            behaviour depends on a curtailment signal. A specialist can take those
            as exogenous features; a univariate zero-shot forecast cannot, and no
            amount of pretraining substitutes for information that was never in the
            input. Second, when the deployment constraints bind: 120 M parameters
            per forecast is a real latency and memory cost, and for thousands of
            feeders forecast every five minutes on embedded hardware, a gradient
            boosting model that fits in a megabyte wins on grounds that have nothing
            to do with accuracy.

            **(c) What this experiment does not support.** Almost any general
            claim. It is *one* synthetic series, *one* horizon, *one* cadence, and
            the aggregate MAE hides the regimes that matter operationally — storm
            days, holidays, the winter peak. Specifically, it would not support:
            "Chronos-2 beats gradient boosting for load forecasting" (one series is
            not a population); "foundation models beat specialists" (one model
            family, one domain); or "this will work on our grid" (synthetic data
            was chosen precisely to rule out contamination, which also means it
            omits every real-world irregularity — meter faults, missing data, tariff
            changes, behaviour shifts).

            What it *does* support is narrow and still worth having: a
            general-purpose model, with no exposure to this data and no training on
            it, produced day-ahead forecasts competitive with a specialist fitted to
            it. That is evidence the pretraining transferred. Scaling that claim to
            a fleet requires a fleet-sized evaluation, and the right next step is to
            run this harness over many real series with per-regime breakdowns rather
            than to generalise from one.
            """,
        ),
    ),
)

# --------------------------------------------------------------------------- 10

CHAPTER_10 = Chapter(
    number=10,
    title="Towards Grid Foundation Models",
    tutorial="10_grid_foundation_models.ipynb",
    intro="""
    Everything converges here. A power grid is not a sequence and not an image:
    it is a graph with physics on it, a different graph for every network, and
    bus numbering that carries no meaning.

    You will build a small grid foundation model — pretrain a message-passing
    encoder on a population of networks with a masked objective, then transfer it
    to a grid it has never seen — and validate it against the physics, not only
    against the statistics.

    This is the frontier. Nothing in this chapter is settled.
    """,
    setup_code="""
    import numpy as np
    import torch
    from torch import nn

    from ai_power_course.config import fast_mode, scaled, set_seed
    from ai_power_course.grid.graphs import (
        EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, STATE_FEATURE_NAMES, net_to_graph,
    )
    from ai_power_course.grid.networks import load_network, run_power_flow
    from ai_power_course.grid.sampling import SamplingConfig, sample_operating_points
    from ai_power_course.metrics import mae
    from ai_power_course.models.gnn import GraphBatch, MaskedGridModel, collate_graphs

    set_seed()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    PRETRAIN = ("case9", "case14", "case30")
    HELDOUT = "case33bw"
    N_SAMPLES = scaled(full=80, fast=26)

    print(f"node features ({len(NODE_FEATURE_NAMES)}): {', '.join(NODE_FEATURE_NAMES)}")
    print(f"  maskable state: {', '.join(STATE_FEATURE_NAMES)}")
    print(f"edge features ({len(EDGE_FEATURE_NAMES)}): {', '.join(EDGE_FEATURE_NAMES)}")

    config = SamplingConfig(n_samples=N_SAMPLES, seed=20260101)
    pretrain_graphs, heldout_graphs = [], []
    for offset, name in enumerate(PRETRAIN):
        points, _ = sample_operating_points(
            name, SamplingConfig(**{**config.__dict__, "seed": config.seed + 100 * offset})
        )
        pretrain_graphs.extend(p.graph for p in points)
    points, _ = sample_operating_points(
        HELDOUT, SamplingConfig(**{**config.__dict__, "seed": config.seed + 9000})
    )
    heldout_graphs = [p.graph for p in points]

    print(f"\\npretraining: {len(pretrain_graphs)} states over {len(PRETRAIN)} grids "
          f"({', '.join(PRETRAIN)})")
    print(f"held out   : {len(heldout_graphs)} states of {HELDOUT}, "
          f"{heldout_graphs[0]['node_features'].shape[0]} buses — never seen in pretraining")
    """,
    tasks=(
        Task(
            number="10.1",
            title="Permutation equivariance is not optional",
            kind="coding",
            difficulty=3,
            background="""
            Bus 14 is bus 14 because someone typed it. Renumber the network and it
            is the same grid, so the model's per-bus outputs must simply come back
            in the new order.

            An MLP on a flattened state vector fails this immediately. A
            message-passing network satisfies it by construction — and it is worth
            *testing* rather than assuming, because a single indexing mistake breaks
            it silently.
            """,
            instruction="""
            Write a function that permutes a graph's node ordering consistently,
            then verify that the encoder's output permutes identically.
            """,
            requirements=(
                "`permute_graph(graph, perm)` relabels the nodes *and* the edge "
                "endpoints.",
                "Edge features must follow their edges unchanged.",
                "Verify equivariance to `1e-5`.",
                "Also show that an MLP on the flattened state is **not** equivariant.",
            ),
            hints=(
                "If `perm` gives the new position of each old node, then "
                "`inverse[perm] = arange(n)` maps old indices to new ones, and "
                "`node_features[perm]` reorders the rows.",
                "Edges are indices into the node array, so apply the inverse "
                "mapping to `edge_index`.",
            ),
            expected="""
            The message-passing encoder should match to floating-point precision.
            The MLP should not — by a wide margin.
            """,
            exercise_code="""
            def permute_graph(graph, perm):
                \"\"\"Relabel the buses of a graph. Same grid, different numbering.\"\"\"
                n = graph["node_features"].shape[0]
                inverse = torch.empty(n, dtype=torch.long)
                inverse[perm] = torch.arange(n)
                return {
                    # TODO 1: reorder the node rows
                    "node_features": ____,
                    # TODO 2: remap the edge endpoints
                    "edge_index": ____,
                    "edge_features": graph["edge_features"],
                }

            set_seed()
            model = MaskedGridModel(
                node_dim=len(NODE_FEATURE_NAMES),
                edge_dim=len(EDGE_FEATURE_NAMES),
                d_model=64, n_layers=4, n_state=len(STATE_FEATURE_NAMES),
            )

            graph = heldout_graphs[0]
            n_buses = graph["node_features"].shape[0]
            perm = torch.randperm(n_buses, generator=torch.Generator().manual_seed(0))

            original = model.embed(collate_graphs([graph]))
            permuted = model.embed(collate_graphs([permute_graph(graph, perm)]))

            # TODO 3: the permuted embedding should equal the permuted original
            gap = float((permuted - ____).abs().max())
            print(f"message passing, max |difference|: {gap:.2e}")
            assert gap < 1e-5, "the grid encoder must be permutation equivariant"

            # The contrast: an MLP on the flattened state vector.
            set_seed()
            flat_mlp = nn.Sequential(
                nn.Linear(n_buses * len(NODE_FEATURE_NAMES), 64), nn.GELU(), nn.Linear(64, 64)
            )
            flat_original = flat_mlp(graph["node_features"].reshape(1, -1))
            flat_permuted = flat_mlp(graph["node_features"][perm].reshape(1, -1))
            print(f"flat MLP,         max |difference|: "
                  f"{float((flat_permuted - flat_original).abs().max()):.2e}")
            """,
            solution_code="""
            def permute_graph(graph, perm):
                \"\"\"Relabel the buses of a graph. Same grid, different numbering.\"\"\"
                n = graph["node_features"].shape[0]
                inverse = torch.empty(n, dtype=torch.long)
                inverse[perm] = torch.arange(n)
                return {
                    "node_features": graph["node_features"][perm],
                    "edge_index": inverse[graph["edge_index"]],
                    "edge_features": graph["edge_features"],
                }

            set_seed()
            model = MaskedGridModel(
                node_dim=len(NODE_FEATURE_NAMES),
                edge_dim=len(EDGE_FEATURE_NAMES),
                d_model=64, n_layers=4, n_state=len(STATE_FEATURE_NAMES),
            )

            graph = heldout_graphs[0]
            n_buses = graph["node_features"].shape[0]
            perm = torch.randperm(n_buses, generator=torch.Generator().manual_seed(0))

            original = model.embed(collate_graphs([graph]))
            permuted = model.embed(collate_graphs([permute_graph(graph, perm)]))

            gap = float((permuted - original[perm]).abs().max())
            print(f"message passing, max |difference|: {gap:.2e}")
            assert gap < 1e-5, "the grid encoder must be permutation equivariant"

            # The contrast: an MLP on the flattened state vector.
            set_seed()
            flat_mlp = nn.Sequential(
                nn.Linear(n_buses * len(NODE_FEATURE_NAMES), 64), nn.GELU(), nn.Linear(64, 64)
            )
            flat_original = flat_mlp(graph["node_features"].reshape(1, -1))
            flat_permuted = flat_mlp(graph["node_features"][perm].reshape(1, -1))
            print(f"flat MLP,         max |difference|: "
                  f"{float((flat_permuted - flat_original).abs().max()):.2e}")
            print("\\nThe MLP would have to LEARN that bus numbering is meaningless, from")
            print("data, for every grid separately. The GNN cannot represent a violation")
            print("of it. Building a symmetry into the architecture is worth more than any")
            print("amount of data spent teaching it.")
            """,
            check_code="""
            _g = permute_graph(heldout_graphs[1], torch.arange(
                heldout_graphs[1]["node_features"].shape[0]))
            assert torch.equal(_g["edge_index"], heldout_graphs[1]["edge_index"]), \\
                "the identity permutation must leave the graph unchanged"
            assert gap < 1e-5
            print("Basic checks passed.")
            """,
            explanation="""
            The inverse permutation is the part people get wrong. `perm` says where
            each old node *goes*; `edge_index` holds old indices that must be
            rewritten as new ones, which is the inverse map. Using `perm` directly
            on `edge_index` produces a different graph that still looks plausible —
            same shapes, same degree distribution, wrong topology.

            The identity-permutation check in the self-test is a cheap way to catch
            exactly that: if you used `perm` where `inverse` belonged, the identity
            case still passes, so the real assertion is the equivariance one.
            Together they pin it down.
            """,
        ),
        Task(
            number="10.2",
            title="Message passing in ten lines",
            kind="coding",
            difficulty=3,
            background="""
            Every graph neural network is the same three steps: compute a message
            along each edge, sum the messages arriving at each node, update the node
            from its own state and that sum.

            $$m_{i \\to j} = \\phi([h_i, h_j, e_{ij}]), \\qquad
              h_j' = h_j + \\psi([h_j, \\textstyle\\sum_{i \\in N(j)} m_{i \\to j}])$$

            No library needed. `index_add_` does the aggregation.
            """,
            instruction="""
            Implement one message-passing layer from scratch and verify that the
            aggregation matches an explicit per-node loop.
            """,
            requirements=(
                "`forward(x, edge_index, edge_features)` returns updated node "
                "embeddings of the same shape.",
                "Use **sum** aggregation via `index_add_`, not mean.",
                "Include a residual connection and a `LayerNorm`.",
                "Verify the aggregation against a loop over nodes.",
            ),
            hints=(
                "`edge_index[0]` is the source, `edge_index[1]` the target. "
                "`x[source]` gathers one row per edge.",
                "`torch.zeros_like(x).index_add_(0, target, messages)` scatters "
                "messages back to their target nodes, summing collisions.",
            ),
            expected="""
            Agreement with the loop to floating-point precision, and an output that
            keeps the input shape.
            """,
            exercise_code="""
            class MyMessagePassing(nn.Module):
                def __init__(self, d_model, edge_dim, hidden=None):
                    super().__init__()
                    hidden = hidden or 2 * d_model
                    self.message = nn.Sequential(
                        nn.Linear(2 * d_model + edge_dim, hidden), nn.GELU(),
                        nn.Linear(hidden, d_model),
                    )
                    self.update = nn.Sequential(
                        nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
                    )
                    self.norm = nn.LayerNorm(d_model)

                def forward(self, x, edge_index, edge_features):
                    source, target = edge_index[0], edge_index[1]
                    # TODO 1: one message per edge, from [h_source, h_target, e]
                    messages = ____
                    # TODO 2: sum the messages arriving at each node
                    aggregated = ____
                    # TODO 3: update with a residual and normalise
                    return ____

            set_seed()
            layer = MyMessagePassing(d_model=32, edge_dim=len(EDGE_FEATURE_NAMES))
            batch = collate_graphs(heldout_graphs[:2])
            x = torch.randn(batch.n_nodes, 32)
            out = layer(x, batch.edge_index, batch.edge_features)
            print(f"nodes {batch.n_nodes}, edges {batch.edge_index.shape[1]} -> "
                  f"output {tuple(out.shape)}")

            # Verify the aggregation against an explicit loop.
            with torch.no_grad():
                src, tgt = batch.edge_index[0], batch.edge_index[1]
                msgs = layer.message(torch.cat([x[src], x[tgt], batch.edge_features], dim=-1))
                vectorised = torch.zeros_like(x).index_add_(0, tgt, msgs)
                looped = torch.zeros_like(x)
                for edge in range(batch.edge_index.shape[1]):
                    looped[tgt[edge]] += msgs[edge]
            print(f"index_add_ vs explicit loop: "
                  f"{float((vectorised - looped).abs().max()):.2e}")
            """,
            solution_code="""
            class MyMessagePassing(nn.Module):
                def __init__(self, d_model, edge_dim, hidden=None):
                    super().__init__()
                    hidden = hidden or 2 * d_model
                    self.message = nn.Sequential(
                        nn.Linear(2 * d_model + edge_dim, hidden), nn.GELU(),
                        nn.Linear(hidden, d_model),
                    )
                    self.update = nn.Sequential(
                        nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
                    )
                    self.norm = nn.LayerNorm(d_model)

                def forward(self, x, edge_index, edge_features):
                    source, target = edge_index[0], edge_index[1]
                    messages = self.message(
                        torch.cat([x[source], x[target], edge_features], dim=-1)
                    )
                    aggregated = torch.zeros_like(x).index_add_(0, target, messages)
                    return self.norm(x + self.update(torch.cat([x, aggregated], dim=-1)))

            set_seed()
            layer = MyMessagePassing(d_model=32, edge_dim=len(EDGE_FEATURE_NAMES))
            batch = collate_graphs(heldout_graphs[:2])
            x = torch.randn(batch.n_nodes, 32)
            out = layer(x, batch.edge_index, batch.edge_features)
            print(f"nodes {batch.n_nodes}, edges {batch.edge_index.shape[1]} -> "
                  f"output {tuple(out.shape)}")

            with torch.no_grad():
                src, tgt = batch.edge_index[0], batch.edge_index[1]
                msgs = layer.message(torch.cat([x[src], x[tgt], batch.edge_features], dim=-1))
                vectorised = torch.zeros_like(x).index_add_(0, tgt, msgs)
                looped = torch.zeros_like(x)
                for edge in range(batch.edge_index.shape[1]):
                    looped[tgt[edge]] += msgs[edge]
            print(f"index_add_ vs explicit loop: "
                  f"{float((vectorised - looped).abs().max()):.2e}")
            print("\\nThat is the whole of geometric deep learning on grids. Every graph")
            print("library is this, plus batching, plus CUDA kernels.")
            """,
            check_code="""
            assert out.shape == x.shape, "a layer must preserve the embedding shape"
            assert float((vectorised - looped).abs().max()) < 1e-4, \\
                "index_add_ disagrees with the explicit loop"
            print("Basic checks passed.")
            """,
            explanation="""
            **Sum, not mean.** Mean aggregation would make a bus with two incident
            lines indistinguishable from a bus with twenty carrying half the current
            each — but total injected power is a *sum* over branches. The physics is
            additive, so the aggregator should be too. This is one of the few places
            where a domain argument settles an architecture choice cleanly.

            Note that the receptive field grows by one hop per layer. Four layers
            reach four buses away: enough for a distribution feeder, deliberately
            not enough to see a whole transmission system. That is a real and
            teachable limit, not an implementation detail.
            """,
        ),
        Task(
            number="10.3",
            title="Pretrain on a population of grids",
            kind="coding",
            difficulty=3,
            background="""
            The pretraining task is BERT's, transplanted onto a power system: hide
            the state of some buses, reconstruct it from the rest of the network.
            Doing that well requires an internal model of how injections, voltages
            and topology constrain one another — which is exactly the reusable
            knowledge we want.

            The training set is several *different grids*, with different sizes and
            topologies. A fixed-width model could not even be evaluated on them.
            """,
            instruction="""
            Standardise the features across the population, write the masked
            pretraining loop, and train the encoder on all pretraining grids at once.
            """,
            requirements=(
                "Standardise node and edge features using statistics from the "
                "pretraining grids only.",
                "Mask 15% of buses per batch; score only the masked ones.",
                "Never mask the slack bus — its voltage is a reference, not an "
                "unknown.",
                "Report the loss curve.",
            ),
            hints=(
                "`MaskedGridModel(node_dim, edge_dim, d_model, n_layers, n_state)` "
                "already provides `apply_mask` and a learned `mask_token`.",
                "The slack flag is `NODE_FEATURE_NAMES.index('is_slack')` in the "
                "*unstandardised* features, so build the mask before standardising "
                "or keep the flag around.",
                "Batch with `collate_graphs`, which merges graphs instead of "
                "padding them.",
            ),
            expected="""
            A loss that falls steadily. It will not reach zero — reconstructing a
            hidden bus exactly requires solving the power flow, which four rounds of
            message passing cannot do.
            """,
            exercise_code="""
            N_STATE = len(STATE_FEATURE_NAMES)
            SLACK = NODE_FEATURE_NAMES.index("is_slack")

            # Standardise across the population, using the pretraining grids only.
            all_nodes = torch.cat([g["node_features"] for g in pretrain_graphs])
            all_edges = torch.cat([g["edge_features"] for g in pretrain_graphs])
            node_mean, node_std = all_nodes.mean(0), all_nodes.std(0).clamp(min=1e-6)
            edge_mean, edge_std = all_edges.mean(0), all_edges.std(0).clamp(min=1e-6)

            def standardise(graph):
                return {
                    "node_features": (graph["node_features"] - node_mean) / node_std,
                    "edge_index": graph["edge_index"],
                    "edge_features": (graph["edge_features"] - edge_mean) / edge_std,
                    "is_slack": graph["node_features"][:, SLACK] > 0.5,
                }

            pretrain_std = [standardise(g) for g in pretrain_graphs]
            heldout_std = [standardise(g) for g in heldout_graphs]

            set_seed()
            pretrained = MaskedGridModel(
                node_dim=len(NODE_FEATURE_NAMES), edge_dim=len(EDGE_FEATURE_NAMES),
                d_model=64, n_layers=4, n_state=N_STATE,
            )
            optimizer = torch.optim.AdamW(pretrained.parameters(), lr=3e-3)
            gen = torch.Generator().manual_seed(0)
            MASK_RATE, BATCH = 0.15, 8
            STEPS = scaled(full=500, fast=60)

            pretrain_loss = []
            pretrained.train()
            for step in range(1, STEPS + 1):
                picks = torch.randint(0, len(pretrain_std), (BATCH,), generator=gen)
                graphs = [pretrain_std[int(i)] for i in picks]
                batch = collate_graphs(graphs)
                is_slack = torch.cat([g["is_slack"] for g in graphs])

                # TODO 1: mask MASK_RATE of the buses, never the slack bus
                mask = ____
                mask = mask & ~is_slack
                if not mask.any():
                    continue

                target = batch.node_features[:, :N_STATE]
                # TODO 2: forward with the mask, then score ONLY the masked buses
                reconstruction = ____
                loss = ____

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrained.parameters(), 1.0)
                optimizer.step()
                pretrain_loss.append(float(loss.detach()))
                if step % max(1, STEPS // 6) == 0:
                    print(f"step {step:4d}  masked reconstruction loss "
                          f"{np.mean(pretrain_loss[-50:]):.4f}")
            print(f"\\nfirst 50 steps {np.mean(pretrain_loss[:50]):.4f} -> "
                  f"last 50 steps {np.mean(pretrain_loss[-50:]):.4f}")
            """,
            solution_code="""
            N_STATE = len(STATE_FEATURE_NAMES)
            SLACK = NODE_FEATURE_NAMES.index("is_slack")

            # Standardise across the population, using the pretraining grids only.
            all_nodes = torch.cat([g["node_features"] for g in pretrain_graphs])
            all_edges = torch.cat([g["edge_features"] for g in pretrain_graphs])
            node_mean, node_std = all_nodes.mean(0), all_nodes.std(0).clamp(min=1e-6)
            edge_mean, edge_std = all_edges.mean(0), all_edges.std(0).clamp(min=1e-6)

            def standardise(graph):
                return {
                    "node_features": (graph["node_features"] - node_mean) / node_std,
                    "edge_index": graph["edge_index"],
                    "edge_features": (graph["edge_features"] - edge_mean) / edge_std,
                    "is_slack": graph["node_features"][:, SLACK] > 0.5,
                }

            pretrain_std = [standardise(g) for g in pretrain_graphs]
            heldout_std = [standardise(g) for g in heldout_graphs]

            set_seed()
            pretrained = MaskedGridModel(
                node_dim=len(NODE_FEATURE_NAMES), edge_dim=len(EDGE_FEATURE_NAMES),
                d_model=64, n_layers=4, n_state=N_STATE,
            )
            optimizer = torch.optim.AdamW(pretrained.parameters(), lr=3e-3)
            gen = torch.Generator().manual_seed(0)
            MASK_RATE, BATCH = 0.15, 8
            STEPS = scaled(full=500, fast=60)

            pretrain_loss = []
            pretrained.train()
            for step in range(1, STEPS + 1):
                picks = torch.randint(0, len(pretrain_std), (BATCH,), generator=gen)
                graphs = [pretrain_std[int(i)] for i in picks]
                batch = collate_graphs(graphs)
                is_slack = torch.cat([g["is_slack"] for g in graphs])

                mask = torch.rand(batch.n_nodes, generator=gen) < MASK_RATE
                mask = mask & ~is_slack
                if not mask.any():
                    continue

                target = batch.node_features[:, :N_STATE]
                reconstruction = pretrained(batch, mask=mask)
                loss = torch.nn.functional.mse_loss(reconstruction[mask], target[mask])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrained.parameters(), 1.0)
                optimizer.step()
                pretrain_loss.append(float(loss.detach()))
                if step % max(1, STEPS // 6) == 0:
                    print(f"step {step:4d}  masked reconstruction loss "
                          f"{np.mean(pretrain_loss[-50:]):.4f}")
            print(f"\\nfirst 50 steps {np.mean(pretrain_loss[:50]):.4f} -> "
                  f"last 50 steps {np.mean(pretrain_loss[-50:]):.4f}")
            print("\\nNot one label was used. The training signal came entirely from the")
            print("structure of the operating states themselves — which is the whole")
            print("proposition, because measurements are abundant in a grid and")
            print("annotations are not.")
            """,
            check_code="""
            assert np.mean(pretrain_loss[-50:]) < np.mean(pretrain_loss[:50]), \\
                "the pretraining loss did not fall"
            assert pretrained.mask_token.shape == (N_STATE,)
            _emb = pretrained.embed(collate_graphs(heldout_std[:1]))
            assert _emb.shape[0] == heldout_std[0]["node_features"].shape[0], \\
                "the encoder must handle a grid size it never saw"
            print("Basic checks passed.")
            """,
            explanation="""
            **Standardisation across the population is what makes transfer
            possible at all.** A 9-bus transmission case and a 33-bus distribution
            feeder have per-unit quantities on entirely different scales; without
            shared statistics the encoder learns grid-specific magnitudes and the
            transferred weights are worse than random initialisation. This was
            measured during the course's development — the first version, without
            standardisation, transferred *negatively*.

            Excluding the slack bus from masking is a physics argument, not a
            heuristic. Its voltage is the angular reference of the whole system;
            asking the model to infer it is asking it to recover information that is
            definitionally fixed, and the resulting loss term is pure noise.
            """,
        ),
        Task(
            number="10.4",
            depends_on=("10.3",),
            title="Transfer to a grid the model has never seen",
            kind="coding",
            difficulty=3,
            background="""
            The claim being tested: pretraining on *other* grids produces a
            representation that makes learning on *this* grid cheaper in labels.

            The downstream task is voltage-magnitude estimation from injections and
            topology — a state-estimation-flavoured problem. The held-out grid is a
            33-bus radial distribution feeder; the pretraining grids are meshed
            transmission networks. This is a hard transfer, deliberately.
            """,
            instruction="""
            Fine-tune the pretrained encoder and an identically-sized
            randomly-initialised one on the held-out grid, across several label
            budgets, under a matched optimisation budget.
            """,
            requirements=(
                "Both conditions get identical data, steps, learning rate and seed.",
                "At least three label budgets, the smallest being tiny.",
                "Evaluate on held-out states of the same grid, never on the ones "
                "used for fitting.",
                "Report the relative improvement at each budget.",
            ),
            hints=(
                "Input: hide the voltage magnitude with the pretrained mask token, "
                "predict it. Target: the standardised `vm_pu` column.",
                "Equalise budgets by **steps**, not epochs — otherwise the "
                "small-budget condition gets far fewer updates.",
                "`MaskedGridModel(...)` fresh from `set_seed()` gives you the "
                "scratch condition with identical architecture.",
            ),
            expected="""
            The largest advantage at the smallest budget, shrinking as labels
            accumulate. If pretraining loses somewhere, report that too.
            """,
            exercise_code="""
            import copy

            import pandas as pd

            VM = STATE_FEATURE_NAMES.index("vm_pu")
            N_TEST = scaled(full=30, fast=6)
            BUDGETS = [5, 12] if fast_mode() else [5, 20, 40]
            STEPS_FT = scaled(full=150, fast=30)

            assert max(BUDGETS) + N_TEST <= len(heldout_std), (
                f"{max(BUDGETS)} training + {N_TEST} test states requested but only "
                f"{len(heldout_std)} sampled"
            )

            def voltage_task(graphs, mask_token):
                \"\"\"Hide vm_pu everywhere; the target is the value that was hidden.\"\"\"
                inputs, targets = [], []
                for g in graphs:
                    features = g["node_features"].clone()
                    targets.append(features[:, VM].clone())
                    features[:, VM] = mask_token[VM]
                    inputs.append({**g, "node_features": features})
                return inputs, targets

            def run(model, n_train, seed=0):
                torch.manual_seed(seed)
                head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
                params = list(model.encoder.parameters()) + list(head.parameters())
                optimizer = torch.optim.AdamW(params, lr=1e-3)
                inputs, targets = voltage_task(heldout_std, model.mask_token.detach())
                fit_idx, test_idx = range(n_train), range(len(inputs) - N_TEST, len(inputs))

                model.train()
                for step in range(STEPS_FT):
                    pick = [fit_idx[step % n_train]]
                    batch = collate_graphs([inputs[i] for i in pick])
                    target = torch.cat([targets[i] for i in pick])
                    prediction = head(model.encoder(batch)).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(prediction, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    batch = collate_graphs([inputs[i] for i in test_idx])
                    prediction = head(model.encoder(batch)).squeeze(-1)
                    truth = torch.cat([targets[i] for i in test_idx])
                # Back to per-unit so the number means something.
                return float((prediction - truth).abs().mean()) * float(node_std[VM])

            rows = []
            for n_train in BUDGETS:
                # TODO 1: the pretrained condition (deep-copy so budgets are independent)
                warm = copy.deepcopy(pretrained)
                warm_mae = ____

                # TODO 2: the scratch condition, identical architecture and budget
                set_seed()
                cold = ____
                cold.mask_token.data = pretrained.mask_token.data.clone()
                cold_mae = ____

                rows.append({"labels": n_train, "pretrained": warm_mae, "from scratch": cold_mae,
                             "improvement": 1 - warm_mae / cold_mae})
            transfer = pd.DataFrame(rows).set_index("labels")
            display(transfer.round(4))
            """,
            solution_code="""
            import copy

            import pandas as pd

            VM = STATE_FEATURE_NAMES.index("vm_pu")
            N_TEST = scaled(full=30, fast=6)
            BUDGETS = [5, 12] if fast_mode() else [5, 20, 40]
            STEPS_FT = scaled(full=150, fast=30)

            assert max(BUDGETS) + N_TEST <= len(heldout_std), (
                f"{max(BUDGETS)} training + {N_TEST} test states requested but only "
                f"{len(heldout_std)} sampled"
            )

            def voltage_task(graphs, mask_token):
                \"\"\"Hide vm_pu everywhere; the target is the value that was hidden.\"\"\"
                inputs, targets = [], []
                for g in graphs:
                    features = g["node_features"].clone()
                    targets.append(features[:, VM].clone())
                    features[:, VM] = mask_token[VM]
                    inputs.append({**g, "node_features": features})
                return inputs, targets

            def run(model, n_train, seed=0):
                torch.manual_seed(seed)
                head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
                params = list(model.encoder.parameters()) + list(head.parameters())
                optimizer = torch.optim.AdamW(params, lr=1e-3)
                inputs, targets = voltage_task(heldout_std, model.mask_token.detach())
                fit_idx, test_idx = range(n_train), range(len(inputs) - N_TEST, len(inputs))

                model.train()
                for step in range(STEPS_FT):
                    pick = [fit_idx[step % n_train]]
                    batch = collate_graphs([inputs[i] for i in pick])
                    target = torch.cat([targets[i] for i in pick])
                    prediction = head(model.encoder(batch)).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(prediction, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    batch = collate_graphs([inputs[i] for i in test_idx])
                    prediction = head(model.encoder(batch)).squeeze(-1)
                    truth = torch.cat([targets[i] for i in test_idx])
                # Back to per-unit so the number means something.
                return float((prediction - truth).abs().mean()) * float(node_std[VM])

            rows = []
            for n_train in BUDGETS:
                warm = copy.deepcopy(pretrained)
                warm_mae = run(warm, n_train)

                set_seed()
                cold = MaskedGridModel(
                    node_dim=len(NODE_FEATURE_NAMES), edge_dim=len(EDGE_FEATURE_NAMES),
                    d_model=64, n_layers=4, n_state=N_STATE,
                )
                cold.mask_token.data = pretrained.mask_token.data.clone()
                cold_mae = run(cold, n_train)

                rows.append({"labels": n_train, "pretrained": warm_mae, "from scratch": cold_mae,
                             "improvement": 1 - warm_mae / cold_mae})
            transfer = pd.DataFrame(rows).set_index("labels")
            display(transfer.round(4))

            best = transfer["improvement"].idxmax()
            print(f"\\nLargest gain at {best} labels: "
                  f"{transfer.loc[best, 'improvement']:.1%} lower MAE in per unit.")
            print("\\nThe encoder was pretrained on meshed transmission networks and")
            print("evaluated on a radial distribution feeder — different topology,")
            print("different R/X ratio, different voltage level. What transferred is not")
            print("'knowledge of case33bw'. It is something more general about how")
            print("injections, impedances and voltages constrain one another.")
            """,
            check_code="""
            assert set(transfer.columns) == {"pretrained", "from scratch", "improvement"}
            assert (transfer[["pretrained", "from scratch"]] > 0).all().all(), \\
                "MAE must be positive"
            assert (transfer[["pretrained", "from scratch"]] < 1.0).all().all(), \\
                "an MAE above 1.0 per unit means something is badly wrong"
            print("Basic checks passed.")
            """,
            explanation="""
            **Copying `mask_token` into the scratch model** is what makes the
            comparison fair. The masking mechanism has to be identical in both
            conditions, otherwise the pretrained model gets an advantage from a
            better-placed mask value rather than from its encoder — which would be a
            real effect but not the one being claimed.

            **Equalising by steps, not epochs**, is the other half. With five
            labels and "ten epochs", the small-budget condition gets 50 updates while
            the 60-label condition gets 600; the label-efficiency curve would then
            mostly measure the optimisation budget.

            `copy.deepcopy(pretrained)` per budget keeps the budgets independent.
            Fine-tuning in place would mean each budget starts from the previous
            one's weights, and the "improvement" would accumulate across rows for
            entirely uninteresting reasons.
            """,
        ),
        Task(
            number="10.5",
            depends_on=("10.3", "10.4"),
            title="Statistical accuracy is not physical validity",
            kind="analysis",
            difficulty=3,
            background="""
            A voltage profile can have an excellent MAE and still be a state the
            network cannot physically be in. The residual of the AC power balance,

            $$S = V \\odot \\overline{(YV)},$$

            recovers the nodal injections a predicted voltage profile implies.
            Comparing those with the injections that were actually scheduled tells
            you whether the prediction is a realisable operating point or merely a
            plausible-looking set of numbers. No statistical metric can answer
            that question.
            """,
            instruction="""
            Train the voltage head properly, confirm it is accurate in per unit,
            then compute the power-balance residual of its prediction and compare
            it against the feeder's own total load.
            """,
            requirements=(
                "Train the head first — an untrained model is bad at both "
                "statistics and physics, which would demonstrate nothing.",
                "Report the voltage MAE in per unit, and the mismatch in MW.",
                "Compare against the residual of the true solved state, which is "
                "the numerical floor.",
                "Express the mismatch as a share of the feeder's total load, "
                "because megawatts mean nothing without a denominator.",
            ),
            hints=(
                "`run_power_flow(net)` solves the base case; `internal_state(net)` "
                "returns the solved `(vm_pu, va_degree)` in the internal ordering "
                "that `get_ybus` uses.",
                "You predicted magnitudes only, so take the angles from the solved "
                "state — and note in your answer that this makes the check "
                "*optimistic*.",
                "`net.load.p_mw.sum()` is the feeder's total demand.",
            ),
            expected="""
            A voltage MAE well under 1% of nominal — by any statistical standard a
            good result — alongside a power mismatch that is a large fraction of
            everything the feeder consumes. That gap is the entire point of the
            task.
            """,
            exercise_code="""
            from ai_power_course.grid.physics import internal_state, power_balance_residual

            net = load_network(HELDOUT)
            assert run_power_flow(net), "the base case must converge"
            vm_true, va_true = internal_state(net)
            base_graph = standardise(net_to_graph(net))

            # Train the head. An untrained one would be bad at BOTH statistics and
            # physics, and would prove nothing; we want a model that looks good.
            set_seed()
            probe_model = copy.deepcopy(pretrained)
            head_check = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
            optimizer = torch.optim.AdamW(
                list(probe_model.encoder.parameters()) + list(head_check.parameters()),
                lr=1e-3,
            )
            fit_inputs, fit_targets = voltage_task(
                heldout_std[:40], pretrained.mask_token.detach())

            probe_model.train()
            for step in range(scaled(full=400, fast=120)):
                batch = collate_graphs([fit_inputs[step % len(fit_inputs)]])
                target = fit_targets[step % len(fit_targets)]
                # TODO 1: forward through encoder and head, MSE loss, optimise
                prediction = ____
                loss = ____
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            inputs, _truths = voltage_task([base_graph], pretrained.mask_token.detach())
            probe_model.eval()
            with torch.no_grad():
                predicted_std = head_check(
                    probe_model.encoder(collate_graphs(inputs))).squeeze(-1)
            vm_pred = (predicted_std * node_std[VM] + node_mean[VM]).numpy()

            # TODO 2: statistical error, in per unit
            voltage_mae = ____
            print(f"voltage MAE: {voltage_mae:.4f} per unit "
                  f"({voltage_mae * 100:.2f}% of nominal)")
            print(f"predicted range {vm_pred.min():.3f}-{vm_pred.max():.3f} pu, "
                  f"true {vm_true.min():.3f}-{vm_true.max():.3f} pu")

            # TODO 3: the residual of the PREDICTED state, and of the true one
            predicted_residual = ____
            true_residual = ____
            total_load = float(net.load.p_mw.sum())
            print(f"\\npredicted state: {predicted_residual['p_mismatch_mean_mw']:7.3f} MW "
                  f"mean, {predicted_residual['p_mismatch_max_mw']:7.3f} MW max")
            print(f"solved state   : {true_residual['p_mismatch_mean_mw']:7.3f} MW "
                  f"mean, {true_residual['p_mismatch_max_mw']:7.3f} MW max")
            print(f"\\ntotal feeder load: {total_load:.2f} MW")
            share = predicted_residual["p_mismatch_mean_mw"] / total_load
            print(f"mean mismatch as a share of it: {share:.1%}")
            """,
            solution_code="""
            from ai_power_course.grid.physics import internal_state, power_balance_residual

            net = load_network(HELDOUT)
            assert run_power_flow(net), "the base case must converge"
            vm_true, va_true = internal_state(net)
            base_graph = standardise(net_to_graph(net))

            # Train the head. An untrained one would be bad at BOTH statistics and
            # physics, and would prove nothing; we want a model that looks good.
            set_seed()
            probe_model = copy.deepcopy(pretrained)
            head_check = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
            optimizer = torch.optim.AdamW(
                list(probe_model.encoder.parameters()) + list(head_check.parameters()),
                lr=1e-3,
            )
            fit_inputs, fit_targets = voltage_task(
                heldout_std[:40], pretrained.mask_token.detach())

            probe_model.train()
            for step in range(scaled(full=400, fast=120)):
                batch = collate_graphs([fit_inputs[step % len(fit_inputs)]])
                target = fit_targets[step % len(fit_targets)]
                prediction = head_check(probe_model.encoder(batch)).squeeze(-1)
                loss = torch.nn.functional.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            inputs, _truths = voltage_task([base_graph], pretrained.mask_token.detach())
            probe_model.eval()
            with torch.no_grad():
                predicted_std = head_check(
                    probe_model.encoder(collate_graphs(inputs))).squeeze(-1)
            vm_pred = (predicted_std * node_std[VM] + node_mean[VM]).numpy()

            voltage_mae = float(np.abs(vm_pred - vm_true).mean())
            print(f"voltage MAE: {voltage_mae:.4f} per unit "
                  f"({voltage_mae * 100:.2f}% of nominal)")
            print(f"predicted range {vm_pred.min():.3f}-{vm_pred.max():.3f} pu, "
                  f"true {vm_true.min():.3f}-{vm_true.max():.3f} pu")

            # The angles come from the SOLVED state — we only predicted magnitudes.
            # That makes this check optimistic, and it still fails.
            predicted_residual = power_balance_residual(net, vm_pred, va_true)
            true_residual = power_balance_residual(net, vm_true, va_true)
            total_load = float(net.load.p_mw.sum())
            print(f"\\npredicted state: {predicted_residual['p_mismatch_mean_mw']:7.3f} MW "
                  f"mean, {predicted_residual['p_mismatch_max_mw']:7.3f} MW max")
            print(f"solved state   : {true_residual['p_mismatch_mean_mw']:7.3f} MW "
                  f"mean, {true_residual['p_mismatch_max_mw']:7.3f} MW max")
            print(f"\\ntotal feeder load: {total_load:.2f} MW")
            share = predicted_residual["p_mismatch_mean_mw"] / total_load
            print(f"mean mismatch as a share of it: {share:.1%}")

            print(f"\\nVoltage error {voltage_mae * 100:.2f}% of nominal; the injections it")
            print(f"implies are wrong by {share:.0%} of everything the feeder consumes.")
            if voltage_mae < 0.02:
                print("By any statistical standard the first number is a good state")
                print("estimator. There is no operating point in which the second is true.")
            else:
                print("(Reduced configuration: the head is undertrained, so the first")
                print("number is not yet good and the contrast is not yet the intended")
                print("one. Run the full configuration to see it.)")
            print("\\nThe solved state's residual is the numerical floor — zero, because it")
            print("is by construction a solution of these equations. Everything above that")
            print("floor is power appearing or vanishing at buses.")
            print("\\nThe model has learned the MARGINAL DISTRIBUTION of voltages — around")
            print("1.0 per unit, rarely outside 0.9 to 1.1 — without the constraints that")
            print("couple them. It is right on average and wrong as a system state, which")
            print("is the failure mode that matters operationally and the one MAE cannot")
            print("see. This is why physics validation is reported SEPARATELY and never")
            print("averaged into a statistical score.")
            """,
            check_code="""
            assert true_residual["p_mismatch_max_mw"] < 1e-3, \\
                "the solved state should satisfy its own equations"
            if fast_mode():
                print("Reduced configuration: skipping the accuracy check — 120 steps")
                print("is not enough to train the head to the point the task is making.")
            else:
                assert voltage_mae < 0.02, \\
                    "train the head further — the point needs an ACCURATE model"
            assert predicted_residual["p_mismatch_mean_mw"] > \\
                true_residual["p_mismatch_mean_mw"], \\
                "a prediction cannot beat the exact solution on its own residual"
            print("Basic checks passed.")
            """,
            explanation="""
            Training the head is what makes this task an argument rather than an
            anecdote. An untrained model is bad at the statistics *and* the
            physics, which shows nothing; the claim being demonstrated is that
            those two can come apart, so the statistics have to be good.

            Note what makes the check *optimistic*: only magnitudes were predicted,
            so the angles came from the true solved state. A full state prediction
            would have to get both right and the residual would be larger still.
            Reporting the favourable variant and still finding a large mismatch is
            stronger evidence than reporting the unfavourable one.

            Expressing the mismatch as a share of total load matters too. A
            mismatch of "1.7 MW" is meaningless in isolation — it is small for a
            transmission system and catastrophic for a 3.7 MW distribution feeder.
            Always give a physical quantity a denominator.

            Tutorial 10 runs the same comparison with a physics-informed loss term
            added. The gap narrows and does not close.
            """,
        ),
        Task(
            number="10.6",
            title="Design your own grid foundation model",
            kind="reflection",
            difficulty=3,
            background="""
            The open question this course has been building towards:

            > **What should the tokens of an electrical grid foundation model be?**

            A language model has words. An image model has patches. A grid has …
            buses? Lines? Substations? Time-steps of a bus? Feeders? Whole snapshots?

            Nobody knows. The models that exist — GridFM, PowerGraph, the
            grid-specific variants appearing since 2024 — disagree with each other,
            and the disagreement is not a detail: the tokenisation decides what can
            be masked, what transfers, and what the model can generalise across.

            This is a genuinely open research question and there is no answer key.
            """,
            instruction="""
            Design a foundation model for a power-systems problem you actually care
            about. Fill in the template below. You are being assessed on whether the
            pieces are *consistent with each other* — pretraining objective with
            token choice, evaluation with claim — not on ambition.
            """,
            requirements=(
                "Name the token explicitly and say what one token *is*.",
                "The pretraining objective must be self-supervised: no labels.",
                "Say what you would evaluate zero-shot and what needs fine-tuning.",
                "Include at least one *physical* validation, separate from the "
                "statistical one.",
                "Name the failure mode you consider most likely, and the experiment "
                "that would reveal it.",
            ),
            hints=(
                "Start from what you want to transfer ACROSS — grids? seasons? "
                "voltage levels? — because that determines what a token must be "
                "comparable across.",
                "Per-unit normalisation is what makes a 110 kV and a 20 kV "
                "measurement commensurable. Without it, your tokens are not "
                "comparable and nothing transfers.",
                "A good sanity check: would your masked-reconstruction task be "
                "trivially solvable by interpolation? If so it will teach the model "
                "nothing.",
            ),
            expected="""
            Internal consistency. The most common failure is a token choice that
            cannot support the claimed downstream task — for example tokenising
            whole snapshots and then claiming per-bus anomaly localisation.
            """,
            answer_template="""
            ### My Grid Foundation Model

            **1. The problem I care about.**
            *(Two sentences. What decision would this support?)*

            **2. The token.**
            *(What is one token? A bus at one instant? A line? A substation-day? Be
            specific about its dimensionality and units.)*

            **3. What makes tokens comparable across grids.**
            *(Per-unit? Normalised by rating? By local mean? Why does this work?)*

            **4. The pretraining corpus.**
            *(What data, how much, from where? Is it obtainable?)*

            **5. The self-supervised objective.**
            *(Masked reconstruction of what? Next-what prediction? Contrastive
            between which pairs?)*

            **6. Architecture, and the symmetry it builds in.**
            *(What is invariant or equivariant, and why does that match the domain?)*

            **7. Downstream tasks.**
            | Task | Zero-shot, probe, or fine-tune? | Why |
            | --- | --- | --- |
            | … | … | … |

            **8. Evaluation.**
            - Statistical: …
            - Physical: …
            - Baseline it must beat: …

            **9. The most likely failure mode, and the experiment that reveals it.**
            *(Be specific. "It might not work" is not a failure mode.)*

            **10. The thing I am least sure about.**
            """,
            answer="""
            ### A Worked Example — a Foundation Model for Distribution-Feeder State

            This is *one* consistent design, not the answer. Its purpose is to show
            what "consistent" looks like; a good alternative might tokenise
            completely differently and be equally defensible.

            **1. The problem I care about.**
            Medium-voltage distribution operators have measurements at the
            substation and at a handful of large connections, and almost nothing in
            between. I want a model that fills in the unmeasured state of a feeder
            well enough to flag developing voltage and thermal problems before they
            are visible at the substation.

            **2. The token.**
            One token is **one bus at one 15-minute interval**: a vector of
            `(p_pu, q_pu, vm_pu, sin va, cos va, is_measured, is_slack,
            has_generation, log_base_kv, sin/cos time-of-day, sin/cos day-of-year)`.
            A feeder snapshot is a *set* of these tokens plus the branch impedances
            connecting them; a sequence of snapshots is a set that grows along the
            time axis.

            This choice is driven by point 7: I want per-bus outputs, so the token
            must be per-bus. Tokenising whole snapshots would make the downstream
            task inexpressible.

            **3. What makes tokens comparable across grids.**
            Per-unit on a common base for the electrical quantities, and impedances
            per unit on the same base. Per unit is what makes a 110 kV and a 20 kV
            measurement the same kind of number — it is the domain's own answer to
            the normalisation problem, arrived at a century before anyone needed it
            for a neural network. The angle is represented as
            $(\\sin, \\cos)$ so that the wraparound at $\\pm\\pi$ is not a
            discontinuity, and the calendar features likewise.

            **4. The pretraining corpus.**
            Two sources. Real SCADA and smart-meter archives from as many feeders as
            can be obtained — realistic but sparse and messy. And a much larger
            synthetic corpus: public feeder models (SimBench, the IEEE European LV
            feeder, the ENWL networks) driven by sampled load and PV profiles and
            solved with a power flow, which gives complete, noise-free states in
            unlimited quantity. The synthetic part carries the physics; the real
            part carries the irregularity. Ordering matters: pretrain on synthetic,
            continue on real.

            **5. The self-supervised objective.**
            Two terms.
            *(a) Spatial masking.* Hide the state of a contiguous, randomly-sized
            **region** of the feeder and reconstruct it. Contiguous rather than
            scattered, for the reason task 4.1 established: scattered masking is
            solvable by interpolating from neighbours, which teaches nothing, and a
            contiguous region is what an actual measurement gap looks like.
            *(b) Temporal masking.* Hide a block of intervals at some buses and
            reconstruct them from the rest of the window.
            Both are label-free. Neither is solvable without an internal model of
            how injections propagate along an impedance.

            **6. Architecture, and the symmetry it builds in.**
            Message passing over the feeder graph, with attention over the time
            axis. **Permutation equivariant** over buses — bus numbering is
            arbitrary, as task 10.1 demonstrates — and **not** permutation invariant
            over time, which is what the temporal positional encoding is for. The
            same layer type, the opposite requirement on each axis. Depth is capped
            around 6–8 message-passing rounds, which is roughly the electrical
            diameter of a distribution feeder; more would over-smooth.

            **7. Downstream tasks.**

            | Task | How | Why |
            | --- | --- | --- |
            | Fill a measurement gap | Zero-shot | It *is* the pretraining objective |
            | Estimate unmeasured voltages | Zero-shot, then a probe | Same objective, new mask |
            | Flag a developing overload | Linear probe | Few labels; regimes already separate |
            | Detect a topology error | Fine-tune | Needs a comparison pretraining never asks |
            | Day-ahead feeder forecast | Fine-tune | Extrapolation is not interpolation |

            **8. Evaluation.**
            - *Statistical:* MAE and CRPS on held-out **feeders**, not held-out time
              steps of feeders that are in the training set. Broken down by how far
              a bus is from the nearest measurement, because the aggregate is
              dominated by easy buses near the substation.
            - *Physical:* the AC power-balance residual of the reconstructed state,
              in kW, reported separately and never averaged into the statistical
              score — plus the rate at which the model predicts voltages outside
              EN 50160 limits when the true state is inside them, and vice versa.
            - *Baselines it must beat:* linear interpolation along the feeder; a
              classical weighted-least-squares state estimator given the same
              measurements; and a per-feeder GNN trained from scratch on the target
              feeder's own labels. The third is the one that matters — if a scratch
              model matches it with the labels a real operator has, the foundation
              model has bought nothing.

            **9. The most likely failure mode, and the experiment that reveals it.**
            **The model learns the marginal distribution of voltages instead of the
            constraints that couple them.** Distribution voltages live in a narrow
            band; a model that always predicts something near the feeder's local
            mean will score well on MAE while being useless for exactly the
            excursions it was built to catch.

            The experiment: evaluate *conditioned on the event*. Score only the
            intervals where the true state violates a limit, and separately score
            the model's predicted injections against the scheduled ones via the
            power-balance residual. If MAE is good, event-conditional MAE is poor
            and the residual is large, the model is a climatology with extra steps.
            Task 10.5 is a miniature of exactly this diagnostic, and chapter 03's
            error analysis is the same idea one chapter earlier.

            **10. The thing I am least sure about.**
            Whether the bus-at-an-instant token is right. Its weakness is that the
            *topology* — which is what actually changes between feeders — enters only
            through the edges, as features rather than as tokens. A design that
            tokenised **branches** instead would make topology first-class and might
            transfer better across feeders with different structures, at the cost of
            making per-bus outputs indirect.

            I do not know which is better, and I do not think anyone does yet. That
            is a reason to run the comparison, not a reason to pick the familiar one.
            """,
        ),
    ),
)

CHAPTERS: tuple[Chapter, ...] = (
    CHAPTER_06,
    CHAPTER_07,
    CHAPTER_08,
    CHAPTER_09,
    CHAPTER_10,
)
