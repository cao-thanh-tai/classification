"""Dataset utilities for the fruit classification project."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


CLASS_NAMES: tuple[str, ...] = (
    "apple",
    "avocado",
    "banana",
    "cherry",
    "kiwi",
    "mango",
    "orange",
    "pineapple",
    "strawberries",
    "watermelon",
)

LABEL_ALIASES = {
    "apple": "apple",
    "avocado": "avocado",
    "banana": "banana",
    "cherry": "cherry",
    "kiwi": "kiwi",
    "mango": "mango",
    "orange": "orange",
    "pinenapple": "pineapple",
    "pineapple": "pineapple",
    "stawberries": "strawberries",
    "strawberries": "strawberries",
    "watermelon": "watermelon",
}

ImageTransform = Callable[[Image.Image], Tensor | Image.Image]
Sample = tuple[Path, int]


def normalize_label(label: str) -> str:
    """Return the canonical class name for a raw directory or CSV label."""
    normalized = label.strip().lower()
    try:
        return LABEL_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown fruit label: {label!r}") from error


class FruitDataset(Dataset[tuple[Tensor | Image.Image, int]]):
    """Load fruit images from class folders or a split CSV manifest.

    A folder source must contain one subdirectory per class. A CSV source must
    contain ``path`` and ``label`` columns; paths are relative to ``root``.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        root: str | Path | None = None,
        transform: ImageTransform | None = None,
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        self.source = Path(source)
        self.root = Path(root) if root is not None else self.source.parent
        self.transform = transform
        self.class_names = tuple(class_names)
        self.class_to_index = {
            name: index for index, name in enumerate(self.class_names)
        }
        self.samples = self._load_samples()

    def _load_samples(self) -> list[Sample]:
        if self.source.is_file():
            return self._load_manifest()
        if self.source.is_dir():
            return self._load_class_folders()
        raise FileNotFoundError(f"Dataset source does not exist: {self.source}")

    def _load_manifest(self) -> list[Sample]:
        samples: list[Sample] = []
        with self.source.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {"path", "label"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("Dataset CSV must contain 'path' and 'label' columns")
            for row in reader:
                label = normalize_label(row["label"])
                samples.append((self.root / row["path"], self._label_index(label)))
        return self._validate_samples(samples)

    def _load_class_folders(self) -> list[Sample]:
        samples: list[Sample] = []
        for directory in sorted(self.source.iterdir()):
            if not directory.is_dir():
                continue
            label = normalize_label(directory.name)
            label_index = self._label_index(label)
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".gif",
                    ".webp",
                }:
                    samples.append((path, label_index))
        return self._validate_samples(samples)

    def _label_index(self, label: str) -> int:
        try:
            return self.class_to_index[label]
        except KeyError as error:
            raise ValueError(
                f"Label {label!r} is not present in class_names: {self.class_names}"
            ) from error

    @staticmethod
    def _validate_samples(samples: list[Sample]) -> list[Sample]:
        if not samples:
            raise ValueError("Dataset contains no image samples")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            sample = image.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label