"""A small decoder-only Transformer, plus the sampling machinery around it.

Tutorial 07 trains this on a few hundred kilobytes of power-engineering text on
a laptop CPU. It is GPT's architecture at 1/100000 of the scale: the same
next-token objective, the same causal mask, the same sampling controls. What it
does *not* have is scale — and the tutorial is explicit that the gap between
this and a modern LLM is data and compute, not a different idea.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import causal_mask

__all__ = ["CharTokenizer", "TinyGPT", "TinyGPTConfig", "generate"]


class CharTokenizer:
    """Character-level tokenizer: the simplest thing that works.

    Real language models use subword (BPE / SentencePiece) vocabularies, which
    trade a larger embedding table for far shorter sequences and no
    out-of-vocabulary problem. Character level keeps the vocabulary at ~70
    symbols so a tiny model can actually learn something, and makes the
    tokenization step completely transparent. Tutorial 07 then compares it
    against a real subword tokenizer from ``transformers``.
    """

    def __init__(self, text: str) -> None:
        self.chars: list[str] = sorted(set(text))
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Unknown characters are dropped — a tiny corpus has no [UNK] budget."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int] | Tensor) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)


class TinyGPTConfig:
    """Hyper-parameters of :class:`TinyGPT`, with laptop-friendly defaults."""

    def __init__(
        self,
        vocab_size: int,
        context: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        self.vocab_size = vocab_size
        self.context = context
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout


class _CausalSelfAttention(nn.Module):
    """Self-attention that can only look backwards.

    A single fused QKV projection and a pre-registered mask buffer: the same
    two optimisations every real implementation makes, kept small enough to read.
    """

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", causal_mask(config.context), persistent=False)

    def forward(self, x: Tensor, need_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, length, d_model = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=2)
        q = q.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) * self.d_head**-0.5
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = self.attn_dropout(weights) @ v
        merged = context.transpose(1, 2).contiguous().view(batch, length, d_model)
        return self.resid_dropout(self.proj(merged)), (weights if need_weights else None)


class _Block(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = _CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, need_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        attended, weights = self.attention(self.norm1(x), need_weights=need_weights)
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return x, weights


class TinyGPT(nn.Module):
    """Decoder-only Transformer trained to predict the next token.

    The whole of language modelling, in one line of loss:
    ``cross_entropy(logits[:, :-1], tokens[:, 1:])``. Every position predicts
    its successor *simultaneously* — that is teacher forcing, and it is why a
    sequence of length ``T`` yields ``T`` training signals instead of one.
    """

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: the embedding and the output projection are transposes
        # of each other. Saves parameters and usually improves perplexity.
        self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None, need_weights: bool = False
    ):
        """``idx``: ``(batch, length)`` of token ids.

        Returns ``(logits, loss)``; ``loss`` is ``None`` without ``targets``.
        With ``need_weights`` the attention maps of every layer come back too.
        """
        _batch, length = idx.shape
        if length > self.config.context:
            raise ValueError(f"sequence length {length} exceeds context {self.config.context}")
        positions = torch.arange(length, device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions))
        collected: list[Tensor] = []
        for block in self.blocks:
            x, weights = block(x, need_weights=need_weights)
            if need_weights and weights is not None:
                collected.append(weights)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        if need_weights:
            return logits, loss, torch.stack(collected, dim=1)
        return logits, loss


def _filter_logits(logits: Tensor, top_k: int | None, top_p: float | None) -> Tensor:
    """Apply top-k and/or nucleus (top-p) truncation to a logit vector.

    Both exist for the same reason: the tail of the distribution holds
    thousands of tokens that are individually unlikely but collectively likely,
    so pure sampling drifts into nonsense. Top-k keeps a fixed count; top-p
    keeps a fixed probability mass and therefore adapts to how confident the
    model is at this particular step.
    """
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
        remove[..., 0] = False  # always keep the most likely token
        logits = logits.masked_fill(
            remove.scatter(-1, sorted_index, remove), float("-inf")
        )
    return logits


@torch.no_grad()
def generate(
    model: TinyGPT,
    idx: Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressive sampling: predict, append, repeat.

    ``temperature`` rescales the logits before the softmax. Below 1 the
    distribution sharpens (conservative, repetitive); above 1 it flattens
    (varied, error-prone); at 0 the model becomes greedy and deterministic.
    Nothing about this loop is specific to text — it is exactly how an
    autoregressive time-series model rolls out a trajectory too.
    """
    model.eval()
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    for _ in range(max_new_tokens):
        window = idx[:, -model.config.context :]
        logits, _ = model(window)
        logits = logits[:, -1, :]
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = _filter_logits(logits / temperature, top_k, top_p)
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
