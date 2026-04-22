"""Model builders for lightweight transfer learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ModelInfo:
    """Metadata describing a built model.

    Attributes:
        model: PyTorch model.
        frozen_parameters: Number of frozen parameters.
        trainable_parameters: Number of trainable parameters.
        total_parameters: Total number of model parameters.
    """

    model: nn.Module
    frozen_parameters: int
    trainable_parameters: int
    total_parameters: int


def _count_parameters(model: nn.Module, trainable: bool | None = None) -> int:
    """Count model parameters.

    Args:
        model: PyTorch module.
        trainable: ``True`` for trainable only, ``False`` for frozen only,
            or ``None`` for all parameters.

    Returns:
        Number of scalar parameters.
    """
    parameters = model.parameters()
    if trainable is not None:
        parameters = (parameter for parameter in parameters if parameter.requires_grad is trainable)
    return sum(parameter.numel() for parameter in parameters)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set the ``requires_grad`` flag for every parameter in a module.

    Args:
        module: Module to update.
        requires_grad: Desired gradient flag.
    """
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> None:
    """Replace the final classifier layer with the requested number of outputs.

    Args:
        model: Model whose classifier is replaced in place.
        model_name: Supported architecture name.
        num_classes: Number of known document classes.

    Raises:
        ValueError: If the architecture is unsupported.
    """
    if model_name == "efficientnet_b0":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return
    if model_name == "mobilenet_v3_small":
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return
    raise ValueError(f"Unsupported model architecture: {model_name}")


def _unfreeze_last_feature_blocks(model: nn.Module, model_name: str, blocks: int) -> None:
    """Unfreeze a small number of final feature blocks for light fine-tuning.

    Args:
        model: Model to update in place.
        model_name: Supported architecture name.
        blocks: Number of final feature blocks to unfreeze.
    """
    if blocks <= 0:
        return
    if model_name in {"efficientnet_b0", "mobilenet_v3_small"}:
        for block in list(model.features.children())[-blocks:]:
            _set_requires_grad(block, True)


def build_model(
    num_classes: int,
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    train_last_blocks: int = 1,
) -> ModelInfo:
    """Build a lightweight transfer-learning model.

    Args:
        num_classes: Number of known classes. The requested project uses six.
        model_name: Architecture name. Prefer ``efficientnet_b0``.
        pretrained: Whether to load ImageNet weights through Torchvision.
        freeze_backbone: Whether to freeze the feature extractor before training.
        train_last_blocks: Number of final feature blocks to unfreeze.

    Returns:
        Model and parameter-count metadata.

    Raises:
        ValueError: If the model name is unsupported.
    """
    normalized_name = model_name.lower().strip()
    if normalized_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    elif normalized_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    if freeze_backbone:
        _set_requires_grad(model, False)

    _replace_classifier(model, normalized_name, num_classes)
    _set_requires_grad(model.classifier, True)
    if freeze_backbone:
        _unfreeze_last_feature_blocks(model, normalized_name, train_last_blocks)

    trainable_parameters = _count_parameters(model, trainable=True)
    total_parameters = _count_parameters(model)
    frozen_parameters = total_parameters - trainable_parameters
    return ModelInfo(
        model=model,
        frozen_parameters=frozen_parameters,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
    )


def load_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load a state dictionary into a model.

    Args:
        model: Model receiving the parameters.
        state_dict: Serialized PyTorch state dictionary.
    """
    model.load_state_dict(state_dict)
