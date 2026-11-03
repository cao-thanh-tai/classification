"""Collect complete, full split metrics and aggregate them by model."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = ("f1", "accuracy", "params_m", "infer_ms", "train_min")


def collect_metrics(
    experiments_dir: str | Path,
    models: list[str],
    seeds: list[int],
    epochs: int,
    train_samples: int,
    val_samples: int,
) -> list[dict[str, Any]]:
    """Require every requested full run and return per-model mean/std rows."""
    root = Path(experiments_dir)
    grouped: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    problems: list[str] = []
    for model in models:
        for seed in seeds:
            path = root / f"{model}_seed{seed}" / "metrics.json"
            if not path.is_file():
                problems.append(f"missing {path}")
                continue
            with path.open(encoding="utf-8") as stream:
                item = json.load(stream)
            required = {"model", "seed", "epochs", "train_samples", "val_samples", *METRICS}
            absent = required - item.keys()
            if absent:
                problems.append(f"{path} is incomplete (missing {', '.join(sorted(absent))})")
                continue
            if item["model"] != model or int(item["seed"]) != seed:
                problems.append(f"{path} has model/seed that does not match its directory")
                continue
            if int(item["epochs"]) != epochs or int(item["train_samples"]) != train_samples or int(item["val_samples"]) != val_samples:
                problems.append(f"{path} is not a full run with the configured epochs and split sizes")
                continue
            grouped[model].append(item)

    if problems:
        raise ValueError("Cannot create comparison yet:\n- " + "\n- ".join(problems))

    summary: list[dict[str, Any]] = []
    for model, runs in grouped.items():
        row: dict[str, Any] = {"model": model, "seeds": ",".join(str(seed) for seed in seeds)}
        for metric in METRICS:
            values = [float(run[metric]) for run in runs]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def collect_test_metrics(
    experiments_dir: str | Path,
    models: list[str],
    seeds: list[int],
    epochs: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> list[dict[str, Any]]:
    """Combine test scores with training-only measurements for full runs."""
    root = Path(experiments_dir)
    grouped: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    problems: list[str] = []
    for model in models:
        for seed in seeds:
            run_dir = root / f"{model}_seed{seed}"
            train_path = run_dir / "metrics.json"
            test_path = run_dir / "test_metrics.json"
            if not train_path.is_file():
                problems.append(f"missing {train_path}")
                continue
            if not test_path.is_file():
                problems.append(f"missing {test_path}; run scripts/evaluate.py first")
                continue
            with train_path.open(encoding="utf-8") as stream:
                training = json.load(stream)
            with test_path.open(encoding="utf-8") as stream:
                testing = json.load(stream)
            if int(training.get("epochs", -1)) != epochs or int(training.get("train_samples", -1)) != train_samples or int(training.get("val_samples", -1)) != val_samples:
                problems.append(f"{train_path} is not a complete configured training run")
                continue
            if testing.get("split") != "test" or testing.get("model") != model or int(testing.get("seed", -1)) != seed or int(testing.get("test_samples", -1)) != test_samples:
                problems.append(f"{test_path} does not match the requested model, seed, or test split size")
                continue
            required = ("f1", "accuracy", "infer_ms")
            if any(key not in testing for key in required) or "params_m" not in training or "train_min" not in training:
                problems.append(f"{test_path} or {train_path} is missing required metrics")
                continue
            grouped[model].append({
                "f1": testing["f1"],
                "accuracy": testing["accuracy"],
                "infer_ms": testing["infer_ms"],
                "params_m": training["params_m"],
                "train_min": training["train_min"],
            })

    if problems:
        raise ValueError("Cannot create test comparison yet:\n- " + "\n- ".join(problems))

    summary: list[dict[str, Any]] = []
    for model, runs in grouped.items():
        row: dict[str, Any] = {"model": model, "seeds": ",".join(str(seed) for seed in seeds)}
        for metric in METRICS:
            values = [float(run[metric]) for run in runs]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    return summary

