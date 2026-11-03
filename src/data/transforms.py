"""Image transforms shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PIL import Image
from torch import Tensor
from torchvision import transforms


def build_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    training: bool = False,
    augmentation: Mapping[str, object] | None = None,
) -> transforms.Compose:
    """Resize, optionally augment, and normalize an RGB image."""
    ops: list[object] = [transforms.Resize((image_size, image_size))]
    if training and augmentation and augmentation.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend((transforms.ToTensor(), transforms.Normalize(mean=list(mean), std=list(std))))
    return transforms.Compose(ops)

