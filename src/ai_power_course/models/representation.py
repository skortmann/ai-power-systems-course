"""Representation learning: autoencoders, masked modelling and contrastive loss.

This is the conceptual hinge of the course (tutorial 04). Everything before it
trains a model *for a task*. Everything after it trains a model to produce a
*representation*, and decides the task later.

The three objectives implemented here are the three that scaled:

``reconstruct``
    Compress and restore (autoencoder). Cheap, but the latent code can cheat by
    copying rather than abstracting.
``mask and predict``
    Hide part of the input, predict it from the rest. BERT for text, MAE for
    images, and the pretraining objective of most time-series foundation models.
``contrast``
    Pull two views of the same thing together, push different things apart.
    SimCLR for images; for load profiles, two augmentations of the same day.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "ProfileEncoder",
    "ProfileDecoder",
    "MaskedProfileAutoencoder",
    "random_mask",
    "block_mask",
    "nt_xent_loss",
    "masked_reconstruction_loss",
    "LinearProbe",
]


class ProfileEncoder(nn.Module):
    """Encode a fixed-length profile into a low-dimensional latent vector.

    A 1-D convolutional stem gives the encoder translation-aware local
    structure (ramps, peaks) before the global pooling, which is a better
    inductive bias for a daily profile than a plain MLP — and it makes the
    latent space depend on shape rather than on absolute position.
    """

    def __init__(
        self,
        length: int = 24,
        n_channels: int = 1,
        latent_dim: int = 16,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.length, self.n_channels, self.latent_dim = length, n_channels, latent_dim
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden_channels * length, latent_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``(batch, length, n_channels)`` -> ``(batch, latent_dim)``."""
        return self.project(self.stem(x.transpose(1, 2)))


class ProfileDecoder(nn.Module):
    """Map a latent vector back to a full profile. Discarded after pretraining."""

    def __init__(
        self, length: int = 24, n_channels: int = 1, latent_dim: int = 16, hidden: int = 128
    ) -> None:
        super().__init__()
        self.length, self.n_channels = length, n_channels
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, length * n_channels),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).view(-1, self.length, self.n_channels)


class MaskedProfileAutoencoder(nn.Module):
    """Encoder + decoder trained to reconstruct *masked* parts of a profile.

    Note what is **not** here: a label. The training signal comes entirely from
    the structure of the data itself. This is self-supervised learning, and it
    is what makes pretraining on unlabelled SCADA or smart-meter archives
    possible at all.
    """

    def __init__(
        self,
        length: int = 24,
        n_channels: int = 1,
        latent_dim: int = 16,
        hidden_channels: int = 32,
        decoder_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ProfileEncoder(length, n_channels, latent_dim, hidden_channels)
        self.decoder = ProfileDecoder(length, n_channels, latent_dim, decoder_hidden)
        #: Learned value substituted at masked positions, like BERT's [MASK].
        self.mask_token = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """``mask`` is ``(batch, length)`` with ``True`` where the input is hidden."""
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def embed(self, x: Tensor) -> Tensor:
        """Latent codes, in eval mode. This is the reusable artefact."""
        self.eval()
        return self.encoder(x)


def random_mask(
    shape: tuple[int, int], ratio: float = 0.3, generator: torch.Generator | None = None
) -> Tensor:
    """Independent Bernoulli mask over positions. ``True`` means "hidden"."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must lie in [0, 1)")
    noise = torch.rand(shape, generator=generator)
    return noise < ratio


def block_mask(
    shape: tuple[int, int], block: int = 6, generator: torch.Generator | None = None
) -> Tensor:
    """Hide one contiguous block per sample — a much harder task than scattered
    masking, because neighbouring values can no longer be interpolated.

    For power-system profiles this is the realistic case: a communication
    outage removes an hour of consecutive samples, not every third sample.
    """
    batch, length = shape
    if not 0 < block <= length:
        raise ValueError("block must be within the profile length")
    starts = torch.randint(0, length - block + 1, (batch,), generator=generator)
    positions = torch.arange(length).unsqueeze(0)
    return (positions >= starts.unsqueeze(1)) & (positions < (starts + block).unsqueeze(1))


def masked_reconstruction_loss(
    prediction: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-8
) -> Tensor:
    """MSE computed **only where the input was hidden**.

    Scoring the visible positions too would let the model win by learning the
    identity function on the parts it can already see.
    """
    weights = mask.unsqueeze(-1).to(prediction.dtype)
    squared_error = (prediction - target) ** 2 * weights
    return squared_error.sum() / (weights.sum() * prediction.shape[-1] + eps)


def nt_xent_loss(z1: Tensor, z2: Tensor, temperature: float = 0.5) -> Tensor:
    """Normalised temperature-scaled cross entropy (SimCLR's contrastive loss).

    ``z1[i]`` and ``z2[i]`` are two augmented views of sample ``i``. Every other
    sample in the batch is a negative. The loss is a ``2N``-way classification:
    "which of these is my partner?"

    Choosing the augmentations *is* the science. For daily load profiles,
    jitter and scaling preserve identity; time-reversal does not, because a
    reversed day is a physically different day.
    """
    batch = z1.shape[0]
    features = nn.functional.normalize(torch.cat([z1, z2], dim=0), dim=1)
    similarity = features @ features.T / temperature
    similarity.fill_diagonal_(float("-inf"))
    targets = torch.cat(
        [torch.arange(batch, 2 * batch), torch.arange(0, batch)]
    ).to(features.device)
    return nn.functional.cross_entropy(similarity, targets)


class LinearProbe(nn.Module):
    """A single linear layer on top of frozen features.

    The standard way to ask "how good is this representation?" without letting
    a big head do the work. If a linear probe on frozen embeddings gets close
    to full fine-tuning, the representation — not the head — holds the
    knowledge. Tutorial 04 and tutorial 10 both use it.
    """

    def __init__(self, encoder: nn.Module, latent_dim: int, n_outputs: int = 1) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent_dim, n_outputs)
        self.freeze_encoder()

    def freeze_encoder(self) -> LinearProbe:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        return self

    def unfreeze_encoder(self) -> LinearProbe:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
        return self

    def train(self, mode: bool = True) -> LinearProbe:
        """Keep the encoder in eval mode while the head trains, if it is frozen."""
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        frozen = not any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(not frozen and torch.is_grad_enabled()):
            features = self.encoder(x)
        return self.head(features)


def to_profiles(series: np.ndarray, length: int = 24) -> np.ndarray:
    """Reshape a flat hourly series into ``(n_days, length, 1)`` profiles.

    Trailing hours that do not fill a whole profile are dropped rather than
    padded — a padded final day would be a physically meaningless profile.
    """
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    n_complete = len(values) // length
    return values[: n_complete * length].reshape(n_complete, length, 1)
