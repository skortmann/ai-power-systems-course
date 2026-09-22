"""One training loop, reused by every neural network in the course.

Tutorial 02 writes a training loop by hand so the mechanics are visible. From
tutorial 03 onwards the notebooks call :func:`train_model` instead, because
re-typing the same twenty lines in eight notebooks teaches nothing and hides
the differences that actually matter between the models.

The loop is deliberately plain PyTorch: no Lightning, no callbacks framework,
no config system. A student can read it in one sitting.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "TrainConfig",
    "History",
    "Standardizer",
    "make_loader",
    "train_model",
    "predict",
    "count_parameters",
]


@dataclass
class TrainConfig:
    """Hyper-parameters of the training loop."""

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    #: Stop after this many epochs without validation improvement. ``None`` disables.
    patience: int | None = 8
    grad_clip: float | None = 1.0
    device: str = "cpu"
    verbose: bool = True
    log_every: int = 1
    shuffle: bool = True


@dataclass
class History:
    """What happened during training."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    seconds: float = 0.0

    def as_dict(self) -> dict[str, list[float]]:
        out = {"train": self.train_loss}
        if self.val_loss:
            out["validation"] = self.val_loss
        return out


class Standardizer:
    """Zero-mean, unit-variance scaling **fitted on the training split only**.

    Fitting the scaler on all the data before splitting leaks the test set's
    mean and variance into training. It is a small leak, it is almost never
    caught in review, and it is why this class exists instead of a one-line
    ``(x - x.mean()) / x.std()`` in each notebook.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.eps = eps

    def fit(self, values: np.ndarray, axis: int | tuple[int, ...] = 0) -> Standardizer:
        array = np.asarray(values, dtype=np.float32)
        self.mean_ = array.mean(axis=axis, keepdims=True)
        self.std_ = array.std(axis=axis, keepdims=True) + self.eps
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fitted before transform()")
        return ((np.asarray(values, dtype=np.float32) - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, values: np.ndarray, axis: int | tuple[int, ...] = 0) -> np.ndarray:
        return self.fit(values, axis=axis).transform(values)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fitted before inverse_transform()")
        return (np.asarray(values, dtype=np.float32) * self.std_ + self.mean_).astype(np.float32)


def make_loader(
    *arrays: np.ndarray, batch_size: int = 64, shuffle: bool = False, seed: int | None = None
) -> DataLoader:
    """Wrap NumPy arrays in a deterministic :class:`~torch.utils.data.DataLoader`."""
    tensors = [torch.as_tensor(np.asarray(a, dtype=np.float32)) for a in arrays]
    dataset = TensorDataset(*tensors)
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(20260101 if seed is None else seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters — the honest way to compare 'model size'."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    device: str,
    grad_clip: float | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = [t.to(device) for t in batch]
            inputs, targets = batch[0], batch[-1]
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += float(loss.detach()) * len(inputs)
            count += len(inputs)
    return total / max(count, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    config: TrainConfig | None = None,
) -> History:
    """Train ``model``, restoring the best-validation weights at the end.

    Early stopping and weight restoration use the *validation* split only. The
    test split is never touched here — that is the whole point of having three
    splits, and tutorial 01 shows what happens when the rule is broken.
    """
    cfg = config or TrainConfig()
    loss_fn = loss_fn or nn.MSELoss()
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    history = History()
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, cfg.device, cfg.grad_clip)
        history.train_loss.append(train_loss)

        if val_loader is not None:
            val_loss = _run_epoch(model, val_loader, loss_fn, None, cfg.device, None)
            history.val_loss.append(val_loss)
            if val_loss < history.best_val_loss - 1e-9:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs):
            message = f"epoch {epoch:3d}/{cfg.epochs}  train {train_loss:.5f}"
            if val_loader is not None:
                message += f"  val {history.val_loss[-1]:.5f}"
            print(message)
        if cfg.patience is not None and stale >= cfg.patience:
            if cfg.verbose:
                print(f"early stop at epoch {epoch} (best epoch {history.best_epoch})")
            break

    if val_loader is not None:
        model.load_state_dict(best_state)
    history.seconds = time.perf_counter() - start
    return history


@torch.no_grad()
def predict(model: nn.Module, inputs: np.ndarray, device: str = "cpu", batch_size: int = 512):
    """Batched inference returning a NumPy array. Always in ``eval`` mode."""
    model.eval().to(device)
    tensor = torch.as_tensor(np.asarray(inputs, dtype=np.float32))
    chunks = [
        model(tensor[i : i + batch_size].to(device)).cpu()
        for i in range(0, len(tensor), batch_size)
    ]
    return torch.cat(chunks).numpy()
