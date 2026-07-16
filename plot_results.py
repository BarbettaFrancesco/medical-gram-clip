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
    parser.add_argument("--baseline_metrics", type=str, required=True)
    parser.add_argument("--proposed_metrics", type=str, required=True)
    parser.add_argument("--output", type=str, default="comparison.png")
    args = parser.parse_args()

    baseline = load_history(args.baseline_metrics)
    proposed = load_history(args.proposed_metrics)

    keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]

    for key in keys:
        plt.figure()
        plt.plot([x["epoch"] for x in baseline], [x[key] for x in baseline], label=f"MedCLIP {key}")
        plt.plot([x["epoch"] for x in proposed], [x[key] for x in proposed], label=f"GRAM-Med {key}")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.legend()
        plt.tight_layout()

        out = Path(args.output)
        metric_out = out.with_name(f"{out.stem}_{key.replace('@', '')}{out.suffix}")
        plt.savefig(metric_out, dpi=200)
        plt.close()

    print(f"Saved plots next to {args.output}")


if __name__ == "__main__":
    main()