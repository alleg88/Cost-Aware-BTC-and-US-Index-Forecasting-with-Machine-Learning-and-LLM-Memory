"""Imbalance-aware metrics for the 3-class direction task.

Macro-F1 is the primary classification metric because the flat class dominates.
Class encoding: down=0, flat=1, up=2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["down", "flat", "up"]


def classification_scores(y_true, y_pred) -> dict:
    """Return the imbalance-aware metric bundle for one set of predictions."""
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro",
                             zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "per_class_f1": dict(zip(
            CLASS_NAMES,
            f1_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0),
        )),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS),
    }


def report_text(y_true, y_pred) -> str:
    """Human-readable per-class precision/recall/F1 table."""
    return classification_report(
        y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, zero_division=0
    )


def aggregate_confusion(matrices: list[np.ndarray]) -> np.ndarray:
    """Sum per-fold confusion matrices into one aggregate matrix."""
    return np.sum(matrices, axis=0)
