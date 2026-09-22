"""The neural forecasters of tutorials 02, 03 and 06.

All three consume exactly the same windows from
:func:`ai_power_course.data.make_windows` and emit a ``horizon``-step forecast,
so the comparison between them is about architecture and nothing else.

============  ==========================================  ====================
Model         How it sees the past                        Introduced in
============  ==========================================  ====================
``MLP``       flattened window, no notion of order        tutorial 02
``LSTM``      sequential, one hidden state carried on     tutorial 03
``Transformer`` all positions at once, learned weights    tutorial 06
============  ==========================================  ====================
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import MultiHeadSelfAttention, sinusoidal_positional_encoding

__all__ = [
    "MLPForecaster",
    "SimpleRNNCell",
    "SimpleRNNForecaster",
    "LSTMForecaster",
    "TransformerBlock",
    "TransformerForecaster",
]


class MLPForecaster(nn.Module):
    """A multi-layer perceptron over a flattened context window.

    The window is flattened, so the model has no built-in notion that position
    167 is "one hour ago" and position 0 is "a week ago" — it must learn that
    from data, and it cannot generalise to a different context length. That
    limitation is the motivation for tutorial 03.
    """

    def __init__(
        self,
        context: int,
        horizon: int,
        n_channels: int = 1,
        hidden_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.context, self.horizon, self.n_channels = context, horizon, n_channels
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = context * n_channels
        for width in hidden_sizes:
            layers += [nn.Linear(in_features, width), activation(), nn.Dropout(dropout)]
            in_features = width
        layers.append(nn.Linear(in_features, horizon))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``(batch, context, n_channels)`` -> ``(batch, horizon)``."""
        return self.net(x)


class SimpleRNNCell(nn.Module):
    r"""One vanilla recurrent step, spelled out:

    .. math:: h_t = \tanh(W_x x_t + W_h h_{t-1} + b)

    Tutorial 03 unrolls this by hand to show where the vanishing gradient comes
    from: backpropagating through ``T`` steps multiplies ``T`` copies of
    :math:`W_h^\top \operatorname{diag}(1 - h^2)`, and ``tanh'`` is at most 1.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        return torch.tanh(self.input_to_hidden(x_t) + self.hidden_to_hidden(h_prev))


class SimpleRNNForecaster(nn.Module):
    """A sequence-to-one forecaster built from :class:`SimpleRNNCell`.

    Kept in the package so tutorial 03 can compare it against ``nn.LSTM`` on
    identical data and show the gating machinery earning its keep.
    """

    def __init__(self, n_channels: int, hidden_size: int, horizon: int) -> None:
        super().__init__()
        self.cell = SimpleRNNCell(n_channels, hidden_size)
        self.head = nn.Linear(hidden_size, horizon)
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, return_states: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        batch, length, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size)
        states = []
        for t in range(length):
            h = self.cell(x[:, t, :], h)
            if return_states:
                states.append(h)
        output = self.head(h)
        if return_states:
            return output, torch.stack(states, dim=1)
        return output


class LSTMForecaster(nn.Module):
    """Sequence-to-one LSTM: the last hidden state is decoded into the horizon.

    The LSTM's cell state is an additive path through time, so a gradient can
    survive hundreds of steps instead of decaying geometrically. That is the
    whole reason LSTMs displaced vanilla RNNs (Hochreiter & Schmidhuber, 1997).
    """

    def __init__(
        self,
        n_channels: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        horizon: int = 24,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: Tensor, return_sequence: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        outputs, (hidden, _cell) = self.lstm(x)
        prediction = self.head(hidden[-1])
        if return_sequence:
            return prediction, outputs
        return prediction


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block.

    ``x + Attention(LN(x))`` then ``x + MLP(LN(x))``. Two design choices are
    worth naming:

    * **Residual connections** make the block an *edit* to the representation
      rather than a replacement, which is what lets 12 or 96 of them stack.
    * **Pre-norm** (LayerNorm before the sublayer, not after) is the modern
      default: it trains without a learning-rate warm-up, unlike the post-norm
      arrangement in the original 2017 paper.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, need_weights: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        attended, weights = self.attention(self.norm1(x), mask=mask, need_weights=need_weights)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, weights


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer for multi-step time-series forecasting.

    Each *time step* is a token: a linear layer lifts the ``n_channels``
    measurements at one hour into ``d_model`` dimensions. There is no
    vocabulary and no tokenizer — which is precisely why "Transformer" and
    "language model" are not synonyms.

    Pooling the encoder output over time and decoding the whole horizon in one
    shot (rather than autoregressively) keeps inference cheap and avoids
    compounding errors; it is the standard choice for fixed-horizon forecasting.
    """

    def __init__(
        self,
        context: int,
        horizon: int,
        n_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.context, self.horizon, self.causal = context, horizon, causal
        self.input_proj = nn.Linear(n_channels, d_model)
        self.register_buffer(
            "positional", sinusoidal_positional_encoding(context, d_model), persistent=False
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: Tensor, need_weights: bool = False):
        """``x``: ``(batch, context, n_channels)`` -> ``(batch, horizon)``."""
        h = self.input_proj(x) + self.positional[: x.shape[1]].unsqueeze(0)
        mask = None
        if self.causal:
            from .attention import causal_mask

            mask = causal_mask(x.shape[1], device=x.device)
        collected: list[Tensor] = []
        for block in self.blocks:
            h, weights = block(h, mask=mask, need_weights=need_weights)
            if need_weights and weights is not None:
                collected.append(weights)
        # Use the final position's representation: it has attended over the
        # whole context and is the step closest to the forecast origin.
        prediction = self.head(self.norm(h)[:, -1, :])
        if need_weights:
            return prediction, torch.stack(collected, dim=1)  # (batch, layer, head, q, k)
        return prediction
