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
    parser.add_argument("--clip_metrics", type=str, required=True)
    parser.add_argument("--gram_metrics", type=str, required=True)
    parser.add_argument("--output", type=str, default="comparison.png")
    args = parser.parse_args()

    clip = load_history(args.clip_metrics)
    gram = load_history(args.gram_metrics)

    keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]

    for key in keys:
        plt.figure()
        plt.plot([x["epoch"] for x in clip], [x[key] for x in clip], label=f"CLIP {key}")
        plt.plot([x["epoch"] for x in gram], [x[key] for x in gram], label=f"GRAM {key}")
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