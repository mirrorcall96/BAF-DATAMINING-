"""Evaluation utilities — score extraction + metric computation.

Standard metrics (homework):
  - accuracy, recall_pos, precision_pos, f1_pos, auc
  - confusion-matrix entries: tp, fp, tn, fn

Extended metrics (extended_base sweep):
  - pr_auc                     : area under the precision-recall curve
  - tpr_at_5fpr                : recall at a strict 5% FPR — the canonical BAF metric
  - threshold_tuned_f1_pos     : best F1+ over a threshold sweep
  - best_threshold             : the threshold that achieves it
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


def predict_score(model, X: pd.DataFrame) -> np.ndarray:
    """Return a 1-D array of continuous positive-class scores for AUC.
    Falls back from `predict_proba` to `decision_function` (LinearSVC)."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise RuntimeError(f"Model {model} provides neither predict_proba nor decision_function")


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Standard homework metrics + confusion-matrix entries."""
    pred = model.predict(X_test)
    score = predict_score(model, X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    return {
        "accuracy":      float(accuracy_score(y_test, pred)),
        "recall_pos":    float(recall_score(y_test, pred, zero_division=0)),
        "precision_pos": float(precision_score(y_test, pred, zero_division=0)),
        "f1_pos":        float(f1_score(y_test, pred, zero_division=0)),
        "auc":           float(roc_auc_score(y_test, score)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def tpr_at_fpr(y_true, scores, target_fpr: float = 0.05) -> tuple[float, float]:
    """The largest TPR achievable at FPR <= target_fpr, plus the threshold."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0, float(np.max(thr))
    idx = int(np.argmax(tpr * valid.astype(float)))
    return float(tpr[idx]), float(thr[idx])


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """Sweep thresholds along the precision-recall curve and return (best_f1, threshold)."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    # F1 = 2PR/(P+R), excluding the trailing point that has no threshold
    denom = (prec[:-1] + rec[:-1])
    f1 = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / np.maximum(denom, 1e-12), 0.0)
    if len(f1) == 0:
        return 0.0, 0.5
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(thr[idx])


def evaluate_extended(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Standard metrics + PR-AUC + TPR@5%FPR + threshold-tuned F1+."""
    base = evaluate(model, X_test, y_test)
    score = predict_score(model, X_test)
    pr_auc = float(average_precision_score(y_test, score))
    tpr5, thr5 = tpr_at_fpr(y_test, score, target_fpr=0.05)
    best_f1, best_thr = best_f1_threshold(y_test, score)
    base.update({
        "pr_auc":                  pr_auc,
        "tpr_at_5fpr":             tpr5,
        "threshold_at_5fpr":       thr5,
        "threshold_tuned_f1_pos":  best_f1,
        "best_threshold":          best_thr,
    })
    return base
