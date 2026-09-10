"""Binary prediction metrics used by the public evaluation command."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss, matthews_corrcoef, roc_auc_score


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels); value = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & ((probabilities <= edges[index + 1]) if index == bins - 1 else (probabilities < edges[index + 1]))
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(value)


def binary_metrics(labels, probabilities, threshold: float = 0.5, ece_bins: int = 10) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or len(np.unique(labels)) != 2:
        raise ValueError("Labels and probabilities must have equal length and both classes")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0,1]")
    predictions = (probabilities > threshold).astype(np.int64)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPR": float(average_precision_score(labels, probabilities)),
        "ACC": float(accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Brier": float(brier_score_loss(labels, probabilities)),
        "ECE": expected_calibration_error(labels, probabilities, ece_bins),
        "NLL": float(log_loss(labels, clipped, labels=[0, 1])),
    }

