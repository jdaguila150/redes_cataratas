"""Clinical triage evaluation metrics: Sensitivity, Specificity, Precision, F2, AUC-ROC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_triage_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics tailored to medical screening."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_p >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(recall_score(y_t, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_t, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_t, y_pred, beta=2.0, zero_division=0))

    if len(np.unique(y_t)) > 1:
        auc_roc = float(roc_auc_score(y_t, y_p))
    else:
        auc_roc = 0.0

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f2": f2,
        "auc_roc": auc_roc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
