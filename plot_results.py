from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: str):
    with open(path, "r") as f:
        return json.load(f)["history"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_metrics",
        type=str,
        default="output/vit/medclip/metrics.json",
        help="Path to baseline metrics JSON",
    )
    parser.add_argument(
        "--proposed_metrics",
        type=str,
        default="output/vit/gram_med/metrics.json",
        help="Path to proposed metrics JSON",
    )
    parser.add_argument(
        "--cnn_metrics",
        type=str,
        default="output/cnn/gram_med/metrics.json",
        help="Path to CNN metrics JSON (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save the plots",
    )
    args = parser.parse_args()

    baseline = load_history(args.baseline_metrics)
    proposed = load_history(args.proposed_metrics)

    cnn = None
    if args.cnn_metrics and Path(args.cnn_metrics).exists():
        try:
            cnn = load_history(args.cnn_metrics)
        except Exception as e:
            print(f"Warning: Could not load CNN metrics from {args.cnn_metrics}: {e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard Recall Metrics (Baseline vs Proposed)
    std_keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]
    for key in std_keys:
        plt.figure(figsize=(8, 5))
        plt.plot(
            [x["epoch"] for x in baseline],
            [x[key] for x in baseline],
            label=f"ViT MedCLIP (Baseline)",
            marker='o',
            color='royalblue',
            linewidth=2
        )
        plt.plot(
            [x["epoch"] for x in proposed],
            [x[key] for x in proposed],
            label=f"ViT GRAM-Med (Proposed)",
            marker='s',
            color='darkorange',
            linewidth=2
        )
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel(key, fontsize=11)
        plt.title(f"Comparison of {key}", fontsize=13, fontweight='bold', pad=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        metric_out = output_dir / f"comparison_{key.replace('@', '')}.png"
        plt.savefig(metric_out, dpi=200)
        plt.close()

    # 2. Semantic Recall Metrics (Baseline vs Proposed vs CNN)
    sem_recall_keys = ["i2t_SemR@1", "i2t_SemR@5", "i2t_SemR@10", "t2i_SemR@1", "t2i_SemR@5", "t2i_SemR@10"]
    for key in sem_recall_keys:
        plt.figure(figsize=(8, 5))
        plt.plot(
            [x["epoch"] for x in baseline],
            [x[key] for x in baseline],
            label=f"ViT MedCLIP (Baseline)",
            marker='o',
            color='royalblue',
            linewidth=2
        )
        plt.plot(
            [x["epoch"] for x in proposed],
            [x[key] for x in proposed],
            label=f"ViT GRAM-Med (Proposed)",
            marker='s',
            color='darkorange',
            linewidth=2
        )
        if cnn is not None and len(cnn) > 0 and key in cnn[0]:
            plt.plot(
                [x["epoch"] for x in cnn],
                [x[key] for x in cnn],
                label=f"CNN GRAM-Med (Proposed)",
                marker='^',
                color='forestgreen',
                linewidth=2
            )
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel(key, fontsize=11)
        plt.title(f"Comparison of {key}", fontsize=13, fontweight='bold', pad=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        metric_out = output_dir / f"comparison_{key.replace('@', '')}.png"
        plt.savefig(metric_out, dpi=200)
        plt.close()

    # 3. Semantic Precision Metrics (Baseline vs Proposed)
    sem_prec_keys = ["i2t_SemP@1", "i2t_SemP@5", "i2t_SemP@10", "t2i_SemP@1", "t2i_SemP@5", "t2i_SemP@10"]
    for key in sem_prec_keys:
        plt.figure(figsize=(8, 5))
        plt.plot(
            [x["epoch"] for x in baseline],
            [x[key] for x in baseline],
            label=f"ViT MedCLIP (Baseline)",
            marker='o',
            color='royalblue',
            linewidth=2
        )
        plt.plot(
            [x["epoch"] for x in proposed],
            [x[key] for x in proposed],
            label=f"ViT GRAM-Med (Proposed)",
            marker='s',
            color='darkorange',
            linewidth=2
        )
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel(key, fontsize=11)
        plt.title(f"Comparison of {key}", fontsize=13, fontweight='bold', pad=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        metric_out = output_dir / f"comparison_{key.replace('@', '')}.png"
        plt.savefig(metric_out, dpi=200)
        plt.close()

    print(f"Saved all plots in {output_dir}")


if __name__ == "__main__":
    main()