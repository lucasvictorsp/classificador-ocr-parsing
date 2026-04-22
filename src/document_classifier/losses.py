"""Loss factory for classifier experiments."""

from __future__ import annotations

import torch.nn as nn


def build_loss(loss_name: str = "cross_entropy") -> nn.Module:
    """Create the configured training loss.

    Args:
        loss_name: Loss identifier. Currently supports ``cross_entropy``.

    Returns:
        PyTorch loss module.

    Raises:
        ValueError: If the loss name is unsupported.
    """
    normalized_name = loss_name.lower().strip()
    if normalized_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: {loss_name}")
