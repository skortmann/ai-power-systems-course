"""Model definitions used across the tutorials.

``baselines``       persistence, seasonal naive, climatology (tutorial 01+)
``training``        the shared PyTorch training loop (tutorial 02+)
``forecasters``     MLP, RNN, LSTM and Transformer forecasters (tutorials 02, 03, 06)
``attention``       attention from scratch, in NumPy and PyTorch (tutorial 05)
``representation``  autoencoders, masking, contrastive loss (tutorial 04)
``tinygpt``         a decoder-only language model (tutorial 07)
``gnn``             message passing for the Mini-GridFM (tutorial 10)
"""

from __future__ import annotations

__all__ = [
    "attention",
    "baselines",
    "forecasters",
    "gnn",
    "representation",
    "tinygpt",
    "training",
]
