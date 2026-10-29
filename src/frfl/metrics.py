"""Metrics computed on held-out examples."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred) or len(y_true) == 0: raise ValueError("y_true/y_pred must be non-empty matching 1-D arrays")
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)), "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)), "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "n_samples": int(len(y_true))}
