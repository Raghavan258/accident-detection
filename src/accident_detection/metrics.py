"""Crash-class metrics. Average precision is omitted when a class is absent."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, object]:
    """Score the crash class. Rows are not removed when a score is non-finite."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    balanced, balance_note = _balanced_accuracy(matrix)
    result: dict[str, object] = {
        "rows": int(y_true.size),
        "precision_crash": float(precision),
        "recall_crash": float(recall),
        "f1_crash": float(f1),
        "balanced_accuracy": balanced,
        "balanced_accuracy_note": balance_note,
        "confusion_matrix": {
            "labels": ["non_crash", "crash"],
            "matrix": matrix.astype(int).tolist(),
        },
        "support_non_crash": int(np.sum(y_true == 0)),
        "support_crash": int(np.sum(y_true == 1)),
    }
    if y_score is not None:
        result.update(_average_precision(y_true, np.asarray(y_score, dtype=np.float64)))
    return result


def crash_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return crash-class F1."""

    return float(classification_metrics(y_true, y_pred)["f1_crash"])


def _balanced_accuracy(matrix: np.ndarray) -> tuple[float | None, str]:
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    crash_support = true_positive + false_negative
    negative_support = true_negative + false_positive
    if crash_support == 0 or negative_support == 0:
        return None, "undefined because this slice contains only one class"
    true_positive_rate = true_positive / crash_support
    true_negative_rate = true_negative / negative_support
    return float(0.5 * (true_positive_rate + true_negative_rate)), ""


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(y_score)
    excluded = int(np.sum(~finite))
    usable_true = y_true[finite]
    both_classes = usable_true.size > 0 and len(np.unique(usable_true)) > 1
    if not both_classes:
        return {
            "average_precision_crash": None,
            "average_precision_note": "undefined because this slice does not contain both classes",
            "average_precision_excluded_nonfinite": excluded,
        }
    score = float(average_precision_score(usable_true, y_score[finite]))
    return {
        "average_precision_crash": score,
        "average_precision_note": "",
        "average_precision_excluded_nonfinite": excluded,
    }
