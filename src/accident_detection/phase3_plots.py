"""Exploratory curves. These figures are not a model-selection step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve


def save_pr_curves(curves: list[tuple[str, np.ndarray, np.ndarray]], path: Path, title: str) -> bool:
    """Save precision-recall curves. A one-class slice writes nothing."""

    drawable = [(name, *_curve(labels, scores)) for name, labels, scores in curves]
    drawable = [item for item in drawable if item[1] is not None]
    if not drawable:
        return False
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, precision, recall in drawable:
        axis.plot(recall, precision, label=name)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def save_confusion_grid(panels: list[tuple[str, list[list[int]]]], path: Path, title: str) -> bool:
    """Save confusion matrices with the raw counts visible."""

    if not panels:
        return False
    plt = _pyplot()
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5))
    if len(panels) == 1:
        axes = [axes]
    for axis, (name, matrix) in zip(axes, panels):
        _draw_matrix(axis, name, np.asarray(matrix))
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def _curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        return None, None
    precision, recall, _ = precision_recall_curve(labels, scores)
    return precision, recall


def _draw_matrix(axis, name: str, matrix: np.ndarray) -> None:
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(name)
    axis.set_xticks([0, 1], ["pred non-crash", "pred crash"], rotation=15, ha="right")
    axis.set_yticks([0, 1], ["true non-crash", "true crash"])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
