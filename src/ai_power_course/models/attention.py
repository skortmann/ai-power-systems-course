"""Attention, written out rather than imported.

Tutorial 05 derives scaled dot-product attention with NumPy, checks it against
this module, and only then meets :class:`torch.nn.MultiheadAttention`. The
from-scratch versions here stay in the package so later tutorials can keep
using them and so the tests can prove they agree with PyTorch's.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "scaled_dot_product_attention_numpy",
    "softmax_numpy",
    "causal_mask",
    "sinusoidal_positional_encoding",
    "MultiHeadSelfAttention",
]


def softmax_numpy(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax: subtract the max before exponentiating.

    Without the shift, ``exp(1000)`` overflows to ``inf`` and the whole row
    becomes NaN. Attention scores grow with the key dimension, so this is not a
    theoretical concern.
    """
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def scaled_dot_product_attention_numpy(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Parameters
    ----------
    query, key:
        ``(n_queries, d_k)`` and ``(n_keys, d_k)``.
    value:
        ``(n_keys, d_v)``.
    mask:
        Boolean ``(n_queries, n_keys)``; ``True`` marks positions that may be
        attended to. Masked entries get ``-inf`` *before* the softmax, so they
        receive exactly zero weight afterwards.

    Returns ``(output, weights)`` with shapes ``(n_queries, d_v)`` and
    ``(n_queries, n_keys)``.

    The ``sqrt(d_k)`` division is not cosmetic: for random Q and K with unit
    variance, the dot product has variance ``d_k``, so without the scaling the
    softmax saturates as the model gets wider and gradients vanish.
    """
    d_k = query.shape[-1]
    scores = query @ key.T / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax_numpy(scores, axis=-1)
    return weights @ value, weights


def causal_mask(length: int, device: torch.device | str | None = None) -> Tensor:
    """Lower-triangular boolean mask: position ``i`` may attend to ``j <= i``.

    This single line is the entire difference between a BERT-style encoder
    (sees everything) and a GPT-style decoder (sees only the past), and hence
    between "fill in the blank" and "predict the next step".
    """
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))


def sinusoidal_positional_encoding(length: int, d_model: int) -> Tensor:
    """The fixed sine/cosine positional encoding of Vaswani et al. (2017).

    Attention is permutation-equivariant: shuffle the inputs and the outputs
    shuffle with them. For a *time series* that is fatal, so position has to be
    added to the representation. Sinusoids of geometrically spaced frequencies
    let the model express relative offsets as linear functions of the encoding.
    """
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(-math.log(10_000.0) * index / d_model)
    encoding = torch.zeros(length, d_model)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)[:, : encoding[:, 1::2].shape[1]]
    return encoding


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention, implemented head-by-head and visible.

    ``nn.MultiheadAttention`` does the same thing faster; this version exists so
    that the reshape into heads, the per-head attention and the output
    projection are all readable, and so that the attention weights can be
    inspected without hooks.

    Multiple heads matter because one softmax produces one weighted average.
    A load forecaster plausibly needs to look at "the last few hours" *and*
    "the same hour yesterday" at once; a single head must choose.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query_proj = nn.Linear(d_model, d_model, bias=bias)
        self.key_proj = nn.Linear(d_model, d_model, bias=bias)
        self.value_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, need_weights: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """``x`` is ``(batch, length, d_model)``; ``mask`` is ``(length, length)``.

        Returns ``(output, weights)``; ``weights`` is ``(batch, n_heads,
        length, length)`` when ``need_weights`` is set, else ``None``.

        Two code paths compute the *same* function:

        * ``need_weights=True`` runs the explicit version — scores, softmax,
          weighted sum — because that is the only way to get the attention
          matrix out, and because it is the version worth reading.
        * ``need_weights=False`` calls
          :func:`torch.nn.functional.scaled_dot_product_attention`, PyTorch's
          fused kernel. Identical maths, materially faster, and it never
          materialises the ``length x length`` matrix. That is also why it
          cannot hand the weights back: they are the thing it avoids building.

        The tests assert the two paths agree.
        """
        queries = self._split_heads(self.query_proj(x))
        keys = self._split_heads(self.key_proj(x))
        values = self._split_heads(self.value_proj(x))

        if not need_weights:
            context = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=mask if mask is None else mask.to(dtype=torch.bool),
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            weights = None
        else:
            scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.d_head)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            context = self.dropout(weights) @ values

        batch, _, length, _ = context.shape
        merged = context.transpose(1, 2).reshape(batch, length, self.d_model)
        return self.out_proj(merged), weights
