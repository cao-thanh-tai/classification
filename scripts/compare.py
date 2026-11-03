"""Aggregate completed training runs and generate comparison reports."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.comparison.collect import collect_metrics
from src.comparison.report import write_reports
from src.utils.config import load_yaml


def main() -> None:
    data_cfg = load_yaml(PROJECT_ROOT / "config" / "data.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "config" / "train.yaml")
    split_dir = PROJECT_ROOT / data_cfg["split_dir"]
    train_samples = sum(1 for _ in (split_dir / "train.csv").open(encoding="utf-8")) - 1
    val_samples = sum(1 for _ in (split_dir / "val.csv").open(encoding="utf-8")) - 1
    models = ["resnet18", "resnet50", "vit_b16", "convnext_t"]
    rows = collect_metrics(
        PROJECT_ROOT / "experiments",
        models,
        [int(seed) for seed in train_cfg["seeds"]],
        int(train_cfg["epochs"]),
        train_samples,
        val_samples,
    )
    write_reports(rows, PROJECT_ROOT / "reports")
    print(f"Wrote reports for {len(models)} models to {PROJECT_ROOT / 'reports'}")


if __name__ == "__main__":
    main()

