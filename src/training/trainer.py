"""Shared training loop for configured image classifiers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.evaluator import evaluate


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    num_classes: int,
    output_dir: str | Path,
    model_name: str,
    seed: int,
    epochs_requested: int,
) -> dict[str, float | int | str]:
    output_dir = Path(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        val_metrics, infer_ms = evaluate(model, val_loader, device, num_classes)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "infer_ms": infer_ms,
        }
        history.append(row)
        print(f"epoch {epoch}/{epochs} loss={row['train_loss']:.4f} val_f1={row['val_f1']:.4f} val_accuracy={row['val_accuracy']:.4f}")
        if val_metrics["f1"] > best_f1:
            best_f1, best_metrics = val_metrics["f1"], val_metrics
            torch.save(model.state_dict(), output_dir / "checkpoints" / "best.pt")

    train_min = (time.perf_counter() - started) / 60
    with (output_dir / "logs" / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1_000_000
    result: dict[str, float | int | str] = {
        "model": model_name,
        "seed": seed,
        "f1": best_metrics["f1"],
        "accuracy": best_metrics["accuracy"],
        "params_m": params_m,
        "infer_ms": history[-1]["infer_ms"],
        "train_min": train_min,
        "best_epoch": int(next(row["epoch"] for row in history if row["val_f1"] == best_f1)),
        "epochs": epochs_requested,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result

