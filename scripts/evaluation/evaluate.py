#!/usr/bin/env python3
"""Evaluate a trained ISG-MDA checkpoint on a labeled pair table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from isg_mda.config import load_config
from isg_mda.data import load_entity_features, read_pairs
from isg_mda.engine import load_checkpoint, make_loader, options_from_config, predict, set_seed, write_json
from isg_mda.metrics import binary_metrics
from isg_mda.model import ISGMDA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--drug-table", required=True, type=Path)
    parser.add_argument("--mirna-table", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(); config = load_config(args.config); device = torch.device(args.device)
    set_seed(int(config["training"]["seed"]))
    drugs, mirnas = load_entity_features(args.drug_table, args.mirna_table, int(config["model"]["drug"]["graph_max_nodes"]), int(config["model"]["mirna"]["sequence_max_length"]))
    pairs = read_pairs(args.pairs)
    loader = make_loader(pairs, drugs, mirnas, int(config["training"]["batch_size"]), int(config["training"]["dataloader_workers"]), False, int(config["training"]["seed"]))
    model = ISGMDA(options_from_config(config, args.local_files_only)).to(device)
    load_checkpoint(args.checkpoint, model, device)
    predictions = predict(model, loader, device)
    metrics = binary_metrics(predictions.label, predictions.probability, float(config["metrics"]["classification_threshold"]), int(config["metrics"]["ece_bins"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.tsv", sep="\t", index=False)
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

