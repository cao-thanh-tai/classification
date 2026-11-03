"""Model evaluation and inference timing."""

from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import classification_metrics


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> tuple[dict[str, float], float]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    elapsed = 0.0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += perf_counter() - start
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())
        seen += len(labels)
    return classification_metrics(targets, predictions, num_classes), elapsed * 1000 / max(seen, 1)

