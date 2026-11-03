"""Build a configured classifier and replace its classification head."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.models.registry import MODEL_REGISTRY


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        constructor, default_weights = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model {name!r}; choose from {', '.join(MODEL_REGISTRY)}") from exc
    model = constructor(weights=default_weights if pretrained else None)
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vit_b16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name == "convnext_t":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_configured_model(name: str, num_classes: int, config: dict[str, Any]) -> nn.Module:
    return build_model(name, num_classes, bool(config.get("pretrained", True)))

