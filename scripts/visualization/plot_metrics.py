#!/usr/bin/env python3
"""Plot selected metrics from one or more evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--keys", nargs="+", default=["AUROC", "AUPR"])
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/metrics.png"))
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    labels = [path.parent.name or path.stem for path in args.metrics]
    x = range(len(labels)); width = 0.8 / len(args.keys)
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    for index, key in enumerate(args.keys):
        axis.bar([value + index * width for value in x], [record[key] for record in records], width, label=key)
    axis.set_xticks([value + width * (len(args.keys) - 1) / 2 for value in x], labels, rotation=30, ha="right")
    axis.set_ylim(0, 1); axis.set_ylabel("Metric value"); axis.legend(); figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200); plt.close(figure)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

