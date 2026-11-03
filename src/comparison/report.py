"""Write comparison tables and a compact metric chart."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("f1", "accuracy", "params_m", "infer_ms", "train_min")


def write_reports(
    rows: list[dict[str, Any]],
    reports_dir: str | Path,
    *,
    basename: str = "comparison",
    split: str = "validation",
) -> None:
    output = Path(reports_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    columns = ["model", "seeds"] + [column for metric in METRICS for column in (f"{metric}_mean", f"{metric}_std")]

    with (output / f"{basename}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Model comparison ({split})",
        "",
        f"Mean ± standard deviation over the configured seeds. F1 is macro F1 on the fixed {split} split.",
        "Inference time is milliseconds per image; parameter count is in millions; training time is minutes per run.",
        "",
        "| Model | Seeds | Macro F1 | Accuracy | Params (M) | Inference (ms/image) | Train (min) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cell = lambda key, digits=3: f"{row[key + '_mean']:.{digits}f} ± {row[key + '_std']:.{digits}f}"
        lines.append(
            f"| {row['model']} | {row['seeds']} | {cell('f1')} | {cell('accuracy')} | {cell('params_m', 2)} | {cell('infer_ms', 2)} | {cell('train_min', 2)} |"
        )
    (output / f"{basename}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = [row["model"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, metric, title in zip(axes, ("f1", "infer_ms"), (f"{split.title()} macro F1", "Inference time (ms/image)")):
        values = [row[f"{metric}_mean"] for row in rows]
        errors = [row[f"{metric}_std"] for row in rows]
        axis.bar(names, values, yerr=errors, capsize=4, color="#4c78a8")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / f"{basename}.png", dpi=160)
    plt.close(fig)

