"""Evaluate all configured best checkpoints on the untouched test split."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.comparison.collect import collect_test_metrics
from src.comparison.report import write_reports
from src.data.dataloader import make_loader
from src.evaluation.evaluator import evaluate
from src.models.builder import build_model
from src.utils.config import load_yaml


MODELS = ("resnet18", "resnet50", "vit_b16", "convnext_t")


def main() -> None:
    data_cfg = load_yaml(PROJECT_ROOT / "config" / "data.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "config" / "train.yaml")
    classes = list(data_cfg["classes"])
    split_dir = PROJECT_ROOT / data_cfg["split_dir"]
    test_csv = split_dir / "test.csv"
    test_samples = sum(1 for _ in test_csv.open(encoding="utf-8")) - 1
    train_samples = sum(1 for _ in (split_dir / "train.csv").open(encoding="utf-8")) - 1
    val_samples = sum(1 for _ in (split_dir / "val.csv").open(encoding="utf-8")) - 1
    seeds = [int(seed) for seed in train_cfg["seeds"]]

    device_value = train_cfg.get("device", "auto")
    if device_value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_value)
    loader = make_loader(
        root=PROJECT_ROOT,
        csv_path=test_csv,
        classes=classes,
        train_config=train_cfg,
        training=False,
    )

    experiments_dir = PROJECT_ROOT / "experiments"
    for model_name in MODELS:
        for seed in seeds:
            run_dir = experiments_dir / f"{model_name}_seed{seed}"
            checkpoint_path = run_dir / "checkpoints" / "best.pt"
            training_metrics_path = run_dir / "metrics.json"
            if not checkpoint_path.is_file() or not training_metrics_path.is_file():
                raise FileNotFoundError(f"Missing completed run files for {model_name}, seed {seed}: {run_dir}")
            with training_metrics_path.open(encoding="utf-8") as stream:
                training_metrics = json.load(stream)
            if (
                int(training_metrics.get("epochs", -1)) != int(train_cfg["epochs"])
                or int(training_metrics.get("train_samples", -1)) != train_samples
                or int(training_metrics.get("val_samples", -1)) != val_samples
            ):
                raise ValueError(f"{run_dir} is not a complete full-split training run")

            model = build_model(model_name, len(classes), pretrained=False).to(device)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            metrics, infer_ms = evaluate(model, loader, device, len(classes))
            result = {
                "split": "test",
                "model": model_name,
                "seed": seed,
                "test_samples": len(loader.dataset),
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "infer_ms": infer_ms,
            }
            with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
            print(f"{model_name} seed={seed}: test_f1={metrics['f1']:.4f}, test_accuracy={metrics['accuracy']:.4f}")
            del model, state_dict
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = collect_test_metrics(
        experiments_dir,
        list(MODELS),
        seeds,
        int(train_cfg["epochs"]),
        train_samples,
        val_samples,
        test_samples,
    )
    write_reports(rows, PROJECT_ROOT / "reports", basename="test_comparison", split="test")
    print(f"Wrote test comparison report to {PROJECT_ROOT / 'reports'}")


if __name__ == "__main__":
    main()

