"""Shared training, evaluation, checkpoint, and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import PairDataset, collate_pairs
from .model import ISGMDA, ModelOptions


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def options_from_config(config: dict[str, Any], local_files_only: bool = False) -> ModelOptions:
    mirna = config["model"]["mirna"]
    drug = config["model"]["drug"]
    return ModelOptions(
        mirna_feature_dim=int(mirna["output_dim"]),
        drug_input_dim=int(drug["atom_input_dim"]),
        drug_hidden_dim=int(drug["hidden_dim"]),
        drug_feature_dim=int(drug["output_dim"]),
        gat_layers=int(drug["gat_layers"]),
        gat_heads=int(drug["gat_heads"]),
        rna_fm_model=str(mirna["semantic_backbone"]),
        chemberta_model=str(drug["semantic_backbone"]),
        local_files_only=local_files_only,
    )


def make_loader(pairs: pd.DataFrame, drugs, mirnas, batch_size: int, workers: int,
                shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        PairDataset(pairs, drugs, mirnas), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, drop_last=False, collate_fn=collate_pairs, generator=generator,
    )


def forward_batch(model: nn.Module, batch: dict, device: torch.device):
    probabilities = model(
        batch["drug_graphs"], batch["drug_smiles"], batch["mirna_encoded"], batch["mirna_sequences"]
    )
    labels = batch["labels"].to(device)
    if probabilities.shape != labels.shape or not torch.isfinite(probabilities).all():
        raise RuntimeError("Invalid model output")
    return probabilities, labels


def train_epoch(model, loader, optimizer, device) -> float:
    model.train(); criterion = nn.BCELoss(); total = 0.0; count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        probabilities, labels = forward_batch(model, batch, device)
        loss = criterion(probabilities, labels)
        loss.backward(); optimizer.step()
        total += float(loss.detach().cpu()) * len(labels); count += len(labels)
    if count == 0:
        raise RuntimeError("Training loader is empty")
    return total / count


def predict(model, loader, device) -> pd.DataFrame:
    model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            probabilities, labels = forward_batch(model, batch, device)
            for mirna, drug, label, probability in zip(
                batch["mirna_indices"].tolist(), batch["drug_indices"].tolist(),
                labels.cpu().tolist(), probabilities.cpu().tolist(),
            ):
                rows.append({"mirna_index": mirna, "drug_index": drug, "label": int(label), "probability": probability})
    return pd.DataFrame(rows)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer, epoch: int, config: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "config": config}, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

