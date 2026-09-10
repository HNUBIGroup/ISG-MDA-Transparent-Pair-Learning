"""Deterministic random and cold-start pair splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _test_entities(values: np.ndarray, ratio: float, rng) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) < 2:
        raise ValueError("Cold-start splitting requires at least two entities")
    count = min(max(1, int(round(len(unique) * ratio))), len(unique) - 1)
    return rng.choice(unique, size=count, replace=False)


def build_split(positive: pd.DataFrame, negative: pd.DataFrame, mode: str = "random",
                test_ratio: float = 0.2, seed: int = 0):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between zero and one")
    columns = ["mirna_index", "drug_index"]
    pos = positive[columns].to_numpy(dtype=np.int64)
    neg = negative[columns].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    pos = pos[rng.permutation(len(pos))]
    neg = neg[rng.permutation(len(neg))]
    if mode == "random":
        count = min(max(1, int(len(pos) * test_ratio)), len(pos) - 1)
        pos_train, pos_test = pos[:-count], pos[-count:]
        neg = neg[:len(pos)]
        neg_train, neg_test = neg[:-count], neg[-count:]
    else:
        if mode not in {"mirna_cold", "drug_cold", "strict_cold"}:
            raise ValueError(f"Unsupported split mode: {mode}")
        heldout_mirnas = _test_entities(pos[:, 0], test_ratio, rng) if mode in {"mirna_cold", "strict_cold"} else None
        heldout_drugs = _test_entities(pos[:, 1], test_ratio, rng) if mode in {"drug_cold", "strict_cold"} else None
        def masks(values):
            mirna_test = np.isin(values[:, 0], heldout_mirnas) if heldout_mirnas is not None else np.ones(len(values), dtype=bool)
            drug_test = np.isin(values[:, 1], heldout_drugs) if heldout_drugs is not None else np.ones(len(values), dtype=bool)
            test = mirna_test & drug_test
            train = (~mirna_test if heldout_mirnas is not None else np.ones(len(values), dtype=bool)) & (~drug_test if heldout_drugs is not None else np.ones(len(values), dtype=bool))
            return train, test
        pos_train_mask, pos_test_mask = masks(pos)
        neg_train_mask, neg_test_mask = masks(neg)
        pos_train, pos_test = pos[pos_train_mask], pos[pos_test_mask]
        neg_train, neg_test = neg[neg_train_mask], neg[neg_test_mask]
    if min(len(pos_train), len(pos_test), len(neg_train), len(neg_test)) == 0:
        raise ValueError("Split produced an empty class partition")
    neg_train = neg_train[:len(pos_train)]
    neg_test = neg_test[:len(pos_test)]
    def labeled(pos_rows, neg_rows):
        rows = np.vstack([
            np.column_stack([pos_rows, np.ones(len(pos_rows), dtype=np.int64)]),
            np.column_stack([neg_rows, np.zeros(len(neg_rows), dtype=np.int64)]),
        ])
        rows = rows[rng.permutation(len(rows))]
        return pd.DataFrame(rows, columns=["mirna_index", "drug_index", "label"])
    train, test = labeled(pos_train, neg_train), labeled(pos_test, neg_test)
    metadata = {"mode": mode, "seed": seed, "test_ratio": test_ratio, "train_rows": len(train), "test_rows": len(test)}
    return train, test, metadata


def audit_split(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> dict:
    train_pairs = set(map(tuple, train[["mirna_index", "drug_index"]].to_numpy()))
    test_pairs = set(map(tuple, test[["mirna_index", "drug_index"]].to_numpy()))
    mirna_overlap = len(set(train.mirna_index) & set(test.mirna_index))
    drug_overlap = len(set(train.drug_index) & set(test.drug_index))
    passed = not (train_pairs & test_pairs)
    if mode == "mirna_cold": passed &= mirna_overlap == 0
    if mode == "drug_cold": passed &= drug_overlap == 0
    if mode == "strict_cold": passed &= mirna_overlap == 0 and drug_overlap == 0
    report = {"pair_overlap": len(train_pairs & test_pairs), "mirna_overlap": mirna_overlap, "drug_overlap": drug_overlap, "passed": bool(passed)}
    if not passed:
        raise ValueError(f"Split leakage audit failed: {report}")
    return report

