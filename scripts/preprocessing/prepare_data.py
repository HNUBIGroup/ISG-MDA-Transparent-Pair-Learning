#!/usr/bin/env python3
"""Create deterministic train/test pair tables from user-prepared pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from isg_mda.data import read_pairs
from isg_mda.splits import audit_split, build_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-pairs", required=True, type=Path)
    parser.add_argument("--negative-pairs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-mode", choices=["random", "mirna_cold", "drug_cold", "strict_cold"], default="random")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    positive = read_pairs(args.positive_pairs, require_label=False)
    negative = read_pairs(args.negative_pairs, require_label=False)
    train, test, metadata = build_split(positive, negative, args.split_mode, args.test_ratio, args.seed)
    metadata["audit"] = audit_split(train, test, args.split_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.tsv", sep="\t", index=False)
    test.to_csv(args.output_dir / "test.tsv", sep="\t", index=False)
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

