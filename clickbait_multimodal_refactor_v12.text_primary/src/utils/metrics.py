# src/utils/metrics.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, fbeta_score, balanced_accuracy_score,
    matthews_corrcoef
)

def _prob_to_1d(y_prob) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2: y_prob = y_prob[:, 1]
        else: y_prob = y_prob.squeeze()
    return y_prob.astype(np.float32)

def compute_all(y_true, y_prob, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = _prob_to_1d(y_prob)
    y_pred = (y_prob >= float(thr)).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true))>1 else float("nan"),
        "thr": float(thr),
    }
    try: out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception: out["auc"] = float("nan")
    return out

def tune_threshold(y_true, y_prob, metric: str = "f1", grid=None):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = _prob_to_1d(y_prob)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    metric = (metric or "f1").lower()
    best_thr, best_val = 0.5, -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric in ("f0.5","f05","f_0.5"):
            val = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        elif metric in ("f2","f_2"):
            val = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
        elif metric in ("acc","accuracy"):
            val = accuracy_score(y_true, y_pred)
        elif metric in ("balanced_accuracy","bacc","ba"):
            val = balanced_accuracy_score(y_true, y_pred)
        elif metric == "mcc":
            try: val = matthews_corrcoef(y_true, y_pred)
            except Exception: val = -1.0
        else:
            val = f1_score(y_true, y_pred, zero_division=0)
        if val > best_val: best_val, best_thr = float(val), float(t)
    return best_thr, best_val
