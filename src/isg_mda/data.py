"""Public data-table validation, molecular graphs, and PyTorch datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset


PAIR_COLUMNS = ["mirna_index", "drug_index", "label"]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def read_pairs(path: str | Path, require_label: bool = True) -> pd.DataFrame:
    frame = read_table(path)
    required = PAIR_COLUMNS if require_label else PAIR_COLUMNS[:2]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing pair columns {missing}")
    frame = frame[required].copy()
    for column in PAIR_COLUMNS[:2]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
        if (frame[column] < 0).any():
            raise ValueError(f"{path}: {column} contains negative values")
    if require_label:
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{path}: labels must be binary")
    if frame.duplicated(PAIR_COLUMNS[:2]).any():
        raise ValueError(f"{path}: duplicate miRNA-drug pair")
    return frame


def encode_sequence(sequence: str, max_length: int = 24) -> torch.Tensor:
    mapping = {"A": 1, "U": 2, "C": 3, "G": 4}
    encoded = [mapping.get(base.upper(), 0) for base in str(sequence)[:max_length]]
    encoded.extend([0] * (max_length - len(encoded)))
    return torch.tensor(encoded, dtype=torch.long)


def one_hot_unknown(value, choices: Sequence) -> list[int]:
    selected = value if value in choices else choices[-1]
    return [int(selected == choice) for choice in choices]


def smiles_to_graph(smiles: str, max_nodes: int = 100):
    from torch_geometric.data import Data

    atom_types = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As",
        "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti",
        "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg",
        "Pb", "Unknown",
    ]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, "other",
    ]
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    features = []
    for atom in list(molecule.GetAtoms())[:max_nodes]:
        hybridization = atom.GetHybridization()
        features.append(
            one_hot_unknown(atom.GetSymbol(), atom_types)
            + one_hot_unknown(hybridization if hybridization in hybridizations[:-1] else "other", hybridizations)
            + one_hot_unknown(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + one_hot_unknown(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + one_hot_unknown(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + [int(atom.GetIsAromatic()), atom.GetNumRadicalElectrons(), int(atom.IsInRing())]
        )
    real_nodes = len(features)
    features.extend([[0] * 69 for _ in range(max_nodes - real_nodes)])
    edges = []
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if first < max_nodes and second < max_nodes:
            edges.extend([[first, second], [second, first]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return Data(x=torch.tensor(np.asarray(features), dtype=torch.float32), edge_index=edge_index, real_node_count=real_nodes)


def load_entity_features(drug_table: str | Path, mirna_table: str | Path,
                         max_nodes: int = 100, sequence_max_length: int = 24):
    drugs = read_table(drug_table)
    mirnas = read_table(mirna_table)
    for frame, required, path in (
        (drugs, {"drug_index", "SMILES"}, drug_table),
        (mirnas, {"mirna_index", "miRNA_Sequence"}, mirna_table),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
    if drugs.drug_index.duplicated().any() or mirnas.mirna_index.duplicated().any():
        raise ValueError("Entity indices must be unique")
    drug_features = {}
    for row in drugs.itertuples(index=False):
        index = int(row.drug_index)
        drug_features[index] = {"smiles": str(row.SMILES), "graph": smiles_to_graph(row.SMILES, max_nodes)}
    mirna_features = {}
    for row in mirnas.itertuples(index=False):
        index = int(row.mirna_index)
        sequence = str(row.miRNA_Sequence)
        if not sequence:
            raise ValueError(f"Empty miRNA sequence for index {index}")
        mirna_features[index] = {"sequence": sequence, "encoded": encode_sequence(sequence, sequence_max_length)}
    return drug_features, mirna_features


class PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, drug_features: Mapping, mirna_features: Mapping):
        self.pairs = pairs.reset_index(drop=True)
        self.drug_features = drug_features
        self.mirna_features = mirna_features
        missing_drugs = sorted(set(self.pairs.drug_index) - set(drug_features))
        missing_mirnas = sorted(set(self.pairs.mirna_index) - set(mirna_features))
        if missing_drugs or missing_mirnas:
            raise ValueError(f"Pair table references missing entities: drugs={missing_drugs[:5]}, miRNAs={missing_mirnas[:5]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        drug = self.drug_features[int(row.drug_index)]
        mirna = self.mirna_features[int(row.mirna_index)]
        return {
            "drug_smiles": drug["smiles"], "drug_graph": drug["graph"],
            "mirna_sequence": mirna["sequence"], "mirna_encoded": mirna["encoded"],
            "label": float(row.label), "drug_index": int(row.drug_index), "mirna_index": int(row.mirna_index),
        }


def collate_pairs(items):
    return {
        "drug_graphs": [item["drug_graph"] for item in items],
        "drug_smiles": [item["drug_smiles"] for item in items],
        "mirna_sequences": [item["mirna_sequence"] for item in items],
        "mirna_encoded": torch.stack([item["mirna_encoded"] for item in items]),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32),
        "drug_indices": torch.tensor([item["drug_index"] for item in items], dtype=torch.long),
        "mirna_indices": torch.tensor([item["mirna_index"] for item in items], dtype=torch.long),
    }

