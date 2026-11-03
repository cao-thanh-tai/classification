"""CLI entry point for the shared model training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import make_loader
from src.models.builder import build_model
from src.training.optim import build_optimizer
from src.training.trainer import train_model
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("resnet50", "vit_b16", "convnext_t"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap each split to this many rows for a smoke run")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    data_cfg = load_yaml(PROJECT_ROOT / "config" / "data.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "config" / "train.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "src" / "models" / args.model / "config.yaml")
    seed = args.seed if args.seed is not None else int(train_cfg["seed"])
    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    set_seed(seed)

    device_value = train_cfg.get("device", "auto")
    device = torch.device("cuda" if device_value == "auto" and torch.cuda.is_available() else "cpu") if device_value == "auto" else torch.device(device_value)
    classes = list(data_cfg["classes"])
    split_dir = PROJECT_ROOT / data_cfg["split_dir"]
    common = dict(root=PROJECT_ROOT, classes=classes, train_config=train_cfg, limit=args.limit)
    train_loader = make_loader(csv_path=split_dir / "train.csv", training=True, **common)
    val_loader = make_loader(csv_path=split_dir / "val.csv", training=False, **common)

    model = build_model(args.model, len(classes), bool(model_cfg.get("pretrained", train_cfg.get("pretrained", True)))).to(device)
    optimizer = build_optimizer(model, str(train_cfg["optimizer"]), float(train_cfg["learning_rate"]), float(train_cfg["weight_decay"]))
    output_dir = PROJECT_ROOT / "experiments" / f"{args.model}_seed{seed}"
    print(f"device={device}; train={len(train_loader.dataset)}; val={len(val_loader.dataset)}; output={output_dir}")
    result = train_model(model, train_loader, val_loader, optimizer, device, epochs, len(classes), output_dir, args.model, seed)
    print(f"saved metrics: {output_dir / 'metrics.json'}")
    print(result)


if __name__ == "__main__":
    main()

