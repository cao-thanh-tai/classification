"""PyTorch dataset for image paths and integer labels."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


ImageTransform = Callable[[Image.Image], Tensor | Image.Image]


class FruitDataset(Dataset[tuple[Tensor | Image.Image, int]]):
    """Load images from ``root / image`` and return them with their labels."""

    def __init__(
        self,
        root: str | Path,
        images: Sequence[str | Path],
        labels: Sequence[int],
        transform: ImageTransform | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")

        self.root = Path(root)
        self.images = tuple(Path(image) for image in images)
        self.labels = tuple(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, int]:
        image_path = self.root / self.images[index]
        with Image.open(image_path) as image:
            sample = image.convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]
