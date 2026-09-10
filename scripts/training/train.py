#!/usr/bin/env python3
"""Train ISG-MDA on user-prepared data and save the fixed-final checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from isg_mda.config import load_config
from isg_mda.data import load_entity_features, read_pairs
from isg_mda.engine import make_loader, options_from_config, save_checkpoint, set_seed, train_epoch, write_json
from isg_mda.model import ISGMDA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--train-pairs", type=Path)
    parser.add_argument("--drug-table", type=Path)
    parser.add_argument("--mirna-table", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config); training = config["training"]; data_config = config["data"]
    train_path = args.train_pairs or Path(data_config["train_pairs"])
    drug_path = args.drug_table or Path(data_config["drug_table"])
    mirna_path = args.mirna_table or Path(data_config["mirna_table"])
    output = args.output_dir or Path(config["output"]["root"]) / "training"
    device_name = args.device or training["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu if appropriate")
    device = torch.device(device_name)
    seed = int(training["seed"]); set_seed(seed)
    drugs, mirnas = load_entity_features(drug_path, mirna_path, int(config["model"]["drug"]["graph_max_nodes"]), int(config["model"]["mirna"]["sequence_max_length"]))
    pairs = read_pairs(train_path)
    loader = make_loader(pairs, drugs, mirnas, int(training["batch_size"]), int(training["dataloader_workers"]), True, seed)
    model = ISGMDA(options_from_config(config, args.local_files_only)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    output.mkdir(parents=True, exist_ok=True); history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        loss = train_epoch(model, loader, optimizer, device)
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"epoch={epoch}/{training['epochs']} train_loss={loss:.8f}", flush=True)
    pd.DataFrame(history).to_csv(output / "training_history.tsv", sep="\t", index=False)
    save_checkpoint(output / "model_final_epoch.pt", model, optimizer, int(training["epochs"]), config)
    write_json(output / "run_metadata.json", {"status": "complete", "epochs": int(training["epochs"]), "seed": seed, "device": str(device), "checkpoint_policy": "fixed_final_epoch"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

