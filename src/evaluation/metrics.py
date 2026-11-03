"""Classification metrics without optional sklearn dependency."""

import torch


def classification_metrics(targets: list[int], predictions: list[int], num_classes: int) -> dict[str, float]:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    tp = matrix.diag().float()
    precision = tp / matrix.sum(dim=0).clamp_min(1)
    recall = tp / matrix.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "f1": float(f1.mean()),
        "accuracy": float(tp.sum() / matrix.sum().clamp_min(1)),
    }

