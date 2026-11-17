"""Figures and reports computed from executed logs only."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sanet.io import atomic_text, read_jsonl, write_json


def report_run(run_dir, evaluation_dir=None, output_dir=None):
    run, out = Path(run_dir), Path(output_dir) if output_dir else Path(run_dir) / "report"
    out.mkdir(parents=True, exist_ok=True)
    history = read_jsonl(run / "history.jsonl")
    if not history:
        raise ValueError("No executed epochs")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epoch = [h["epoch"] + 1 for h in history]
    for split, color in [("train", "#2878b5"), ("val", "#e77635")]:
        axes[0].plot(epoch, [h[split]["total"] for h in history], label=split, color=color)
    axes[0].set(
        xlabel="Epoch", ylabel="Weighted objective", title="Executed training and validation"
    )
    axes[0].legend()
    axes[1].plot(epoch, [h["learning_rate"] for h in history], color="#339477")
    axes[1].set(xlabel="Epoch", ylabel="Learning rate", title="Recorded AdamW schedule")
    fig.savefig(out / "learning-curves.png", dpi=180)
    plt.close(fig)
    with open(out / "epochs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "learning_rate", "train_total", "val_total", "train_seconds", "val_seconds"]
        )
        for h in history:
            writer.writerow(
                [
                    h["epoch"] + 1,
                    h["learning_rate"],
                    h["train"]["total"],
                    h["val"]["total"],
                    h["train"]["seconds"],
                    h["val"]["seconds"],
                ]
            )
    text = [
        "# Executed experiment report",
        "",
        "Actual run logs; no manuscript benchmark reproduction is claimed.",
        "",
        f"Epochs: {len(history)}. Minimum validation objective: "
        f"{min(h['val']['total'] for h in history):.6f}.",
        "",
        "![Learning curves](learning-curves.png)",
        "",
    ]
    if evaluation_dir:
        metrics = json.loads((Path(evaluation_dir) / "metrics.json").read_text())
        write_json(out / "metrics.json", metrics)
        rows = [
            (name, value["precision"], value["recall"], value["f1"])
            for name, value in metrics.items()
            if isinstance(value, dict) and "f1" in value
        ]
        if rows:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            for offset, label, idx, color in [
                (-0.25, "precision", 1, "#2878b5"),
                (0, "recall", 2, "#e77635"),
                (0.25, "F1", 3, "#339477"),
            ]:
                ax.bar(
                    [v + offset for v in range(len(rows))],
                    [r[idx] for r in rows],
                    0.25,
                    label=label,
                    color=color,
                )
            ax.set_xticks(list(range(len(rows))), [r[0] for r in rows], rotation=10)
            ax.set(ylim=(0, 1), ylabel="Fraction", title="Executed holdout diagnostics")
            ax.legend()
            fig.savefig(out / "holdout-metrics.png", dpi=180)
            plt.close(fig)
        text += [
            f"Evaluated {metrics['samples']} {metrics['split']} samples on {metrics['device']}.",
            "",
            "| Protocol | Precision | Recall | F1 |",
            "|---|---:|---:|---:|",
        ]
        text += [f"| {r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} |" for r in rows]
        text += ["", "Protocols are not directly comparable.", ""]
    atomic_text(out / "report.md", "\n".join(text))
    return {"report": str(out / "report.md")}


def plot_lane_geometry(predictions, manifest, output):
    """Plot every evaluated image's original 2D lane coordinates, without redistributing RGB."""
    from sanet.io import read_jsonl

    rows = read_jsonl(predictions)
    records = {r["id"]: r for r in read_jsonl(manifest)}
    columns = min(4, len(rows))
    if not columns:
        raise ValueError("No predictions")
    fig, axes = plt.subplots(
        (len(rows) + columns - 1) // columns,
        columns,
        figsize=(4 * columns, 3 * ((len(rows) + columns - 1) // columns)),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, prediction in zip(axes.flat, rows):
        record = records[prediction["id"]]
        for lane in record["lanes_2d"]:
            axis.plot([p[0] for p in lane], [p[1] for p in lane], color="#339477", alpha=0.9)
        for lane in prediction["lanes"]:
            axis.plot(
                [p[0] for p in lane["points"]],
                [p[1] for p in lane["points"]],
                color="#d75a48",
                alpha=0.7,
                linestyle="--",
            )
        axis.invert_yaxis()
        axis.set(
            xlabel="Original image x (px)",
            ylabel="Image y (px)",
            title=f"{record['id'].split('/')[-2]}\nGT green / prediction dashed red",
        )
    for axis in list(axes.flat)[len(rows) :]:
        axis.set_visible(False)
    fig.suptitle("Executed holdout lane geometry — all evaluated frames; no success-case selection")
    fig.savefig(output, dpi=150)
    plt.close(fig)
