"""CSV split loading and DataLoader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import FruitDataset
from src.data.transforms import build_transform


def read_split(csv_path: str | Path, classes: list[str]) -> tuple[list[str], list[int]]:
    """Read a v1 split CSV and map its string labels to stable class indices."""
    frame = pd.read_csv(csv_path)
    required = {"path", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{csv_path} must contain columns: {', '.join(sorted(required))}")
    class_to_index = {name: index for index, name in enumerate(classes)}
    unknown = sorted(set(frame["label"].dropna()) - class_to_index.keys())
    if unknown:
        raise ValueError(f"Unknown labels in {csv_path}: {unknown}")
    if frame[["path", "label"]].isna().any().any():
        raise ValueError(f"Null path or label found in {csv_path}")
    return frame["path"].astype(str).tolist(), [class_to_index[name] for name in frame["label"]]


def make_loader(
    root: str | Path,
    csv_path: str | Path,
    classes: list[str],
    train_config: dict[str, Any],
    *,
    training: bool = False,
    limit: int | None = None,
) -> DataLoader:
    paths, labels = read_split(csv_path, classes)
    if limit is not None:
        paths, labels = paths[:limit], labels[:limit]
    transform = build_transform(
        int(train_config["image_size"]), train_config["mean"], train_config["std"],
        training=training, augmentation=train_config.get("augmentation"),
    )
    dataset = FruitDataset(root, paths, labels, transform)
    return DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=training,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=bool(train_config.get("pin_memory", False)),
    )

