"""Crash metrics match a hand-counted confusion matrix."""

import numpy as np
import pytest

from accident_detection.metrics import classification_metrics


def test_two_class_counts() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1])
    metrics = classification_metrics(y_true, y_pred, y_score)
    assert metrics["precision_crash"] == 0.5
    assert metrics["recall_crash"] == 0.5
    assert metrics["f1_crash"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]
    assert metrics["average_precision_crash"] == pytest.approx(5 / 6)


def test_average_precision_follows_ranking() -> None:
    perfect = classification_metrics(np.array([1, 0]), np.array([1, 0]), np.array([0.9, 0.2]))
    reversed_rank = classification_metrics(np.array([1, 0]), np.array([0, 1]), np.array([0.2, 0.9]))
    assert perfect["average_precision_crash"] == 1.0
    assert reversed_rank["average_precision_crash"] == 0.5


def test_one_class_slice_does_not_invent_average_precision() -> None:
    metrics = classification_metrics(np.array([0, 0]), np.array([1, 0]), np.array([0.4, 0.1]))
    assert metrics["average_precision_crash"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["support_crash"] == 0
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [0, 0]]
