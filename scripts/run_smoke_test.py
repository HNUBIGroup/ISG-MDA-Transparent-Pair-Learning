#!/usr/bin/env python3
"""Run a synthetic CPU forward/backward smoke test without external model downloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_mda.config import load_config
from isg_mda.data import encode_sequence, smiles_to_graph
from isg_mda.metrics import binary_metrics
from isg_mda.model import ISGMDA, ModelOptions


class SyntheticSemanticEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(4, output_dim)

    def forward(self, strings):
        device = self.projection.weight.device
        rows = []
        for value in strings:
            text = str(value)
            rows.append([len(text), sum(map(ord, text)) % 97, text.count("C"), text.count("G")])
        return self.projection(torch.tensor(rows, dtype=torch.float32, device=device))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(); torch.manual_seed(args.seed)
    config = load_config(args.config)
    feature_dim = int(config["model"]["mirna"]["output_dim"])
    options = ModelOptions(
        mirna_feature_dim=feature_dim,
        drug_input_dim=int(config["model"]["drug"]["atom_input_dim"]),
        drug_hidden_dim=int(config["model"]["drug"]["hidden_dim"]),
        drug_feature_dim=int(config["model"]["drug"]["output_dim"]),
        gat_layers=int(config["model"]["drug"]["gat_layers"]),
        gat_heads=int(config["model"]["drug"]["gat_heads"]),
    )
    model = ISGMDA(options, SyntheticSemanticEncoder(feature_dim), SyntheticSemanticEncoder(feature_dim)).cpu()
    smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    sequences = ["AUGCUA", "CCGAUU", "GGGAAA", "UACGCU"]
    graphs = [smiles_to_graph(value, max_nodes=12) for value in smiles]
    encoded = torch.stack([encode_sequence(value) for value in sequences])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    model.train(); optimizer.zero_grad(set_to_none=True)
    probabilities = model(graphs, smiles, encoded, sequences)
    loss = nn.BCELoss()(probabilities, labels); loss.backward()
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    if probabilities.shape != (4,) or not torch.isfinite(probabilities).all() or not torch.isfinite(loss) or not finite_gradients:
        raise RuntimeError("Synthetic smoke test failed")
    metrics = binary_metrics(labels.numpy().astype(int), probabilities.detach().numpy())
    report = {"status": "SMOKE_PASS", "device": "cpu", "synthetic_only": True, "batch_size": 4, "output_shape": list(probabilities.shape), "pair_feature_dim": model.interaction_predictor.predictor[0].in_features, "loss_finite": True, "gradients_finite": True, "metric_keys_checked": sorted(metrics)}
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

