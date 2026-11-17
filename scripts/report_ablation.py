"""Summarize executed validation-only ablation JSON; no invented scores."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sanet.io import atomic_text, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text())["results"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(r["variant"] for r in records))
    summaries = []
    for name in variants:
        selected = [r for r in records if r["variant"] == name]
        values = [r["best_validation_loss"] for r in selected]
        summaries.append(
            {
                "variant": name,
                "seeds": [r["seed"] for r in selected],
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "observations": values,
            }
        )
    write_json(out / "summary.json", summaries)
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, summary in enumerate(summaries):
        axis.scatter(
            [i] * len(summary["observations"]), summary["observations"], color="#2878b5", zorder=3
        )
        axis.errorbar(
            i,
            summary["mean"],
            yerr=summary["sample_std"] or 0,
            color="#e77635",
            marker="s",
            capsize=5,
        )
    axis.set_xticks(range(len(variants)), variants)
    axis.set(
        ylabel="Best recorded validation objective",
        title="Executed validation ablations — mean ± sample SD",
    )
    fig.savefig(out / "validation-ablation.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Executed short image-mode ablations",
        "",
        "Per-run budgets, datasets and seeds are recorded in the accompanying configs, audits and histories; no test scoring.",
        "",
        "These runs verify the ablation workflow. Random initialization, short schedules and limited data prevent a causal or benchmark superiority claim. The baseline is this implementation's compact backbone/head, not DLNet. Loss combines terms and is not lane F1.",
        "",
        "| Variant | Seeds | Mean best validation objective | Sample SD |",
        "|---|---|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['variant']} | {summary['seeds']} | {summary['mean']:.6f} | {summary['sample_std'] if summary['sample_std'] is not None else 'unavailable'} |"
        )
    lines += ["", "![Executed validation ablations](validation-ablation.png)", ""]
    atomic_text(out / "report.md", "\n".join(lines))


if __name__ == "__main__":
    main()
