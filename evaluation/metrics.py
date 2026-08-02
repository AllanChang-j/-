from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    task: str,
    loss: float | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if loss is not None:
        metrics["loss"] = float(loss)
    if task == "regression":
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
        return metrics

    average = "binary" if len(np.unique(y_true)) <= 2 else "weighted"
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    if probabilities is not None:
        try:
            if probabilities.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
                metrics["pr_auc"] = float(average_precision_score(y_true, probabilities[:, 1]))
                frac_pos, mean_pred = calibration_curve(y_true, probabilities[:, 1], n_bins=10, strategy="quantile")
                metrics["calibration_curve"] = {
                    "mean_predicted_probability": mean_pred.tolist(),
                    "fraction_of_positives": frac_pos.tolist(),
                }
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted"))
                metrics["pr_auc"] = float(average_precision_score(y_true, probabilities, average="weighted"))
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    return metrics


def classification_signal(predictions: np.ndarray, probabilities: np.ndarray | None, task: str, threshold: float = 0.55) -> np.ndarray:
    if task == "regression":
        return (predictions > 0).astype(int)
    if probabilities is not None and probabilities.shape[1] == 2:
        return (probabilities[:, 1] >= threshold).astype(int)
    if probabilities is not None and probabilities.shape[1] == 3:
        return (probabilities[:, 2] > probabilities[:, 0]).astype(int)
    return (predictions > 0).astype(int)

