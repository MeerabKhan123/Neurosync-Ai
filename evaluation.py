"""
NeuroSync AI - Model Evaluation Utilities
--------------------------------------------
Shared metric computation used by every ML and DL model so the final
Model Comparison Table is generated consistently.
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)


def evaluate_model(model_name, y_true, y_pred, y_proba=None, train_time=0.0, predict_time=0.0):
    """Compute the standard metric set for one model's predictions.

    y_proba: array of shape (n_samples, n_classes) of predicted probabilities,
             required for ROC-AUC (macro, one-vs-rest) on multiclass targets.
    """
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Train_Time_sec": round(train_time, 4),
        "Predict_Time_sec": round(predict_time, 4),
    }

    if y_proba is not None:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            metrics["ROC_AUC"] = np.nan
    else:
        metrics["ROC_AUC"] = np.nan

    return metrics


def print_report(model_name, y_true, y_pred):
    print(f"\n--- {model_name} ---")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))


class Timer:
    """Small context manager to time a block and store elapsed seconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
