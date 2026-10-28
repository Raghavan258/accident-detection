"""Train-only descriptive summaries. These numbers are not applied to signals."""

from __future__ import annotations

import numpy as np

from accident_detection.constants import LABEL_NAMES, LABEL_ORDER
from accident_detection.models import FileScan

PERCENTILES: tuple[int, ...] = (0, 5, 25, 50, 75, 95, 100)


def percentile_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize finite values. Non-finite peaks are counted and excluded."""

    finite = values[np.isfinite(values)] if values.size else values
    summary: dict[str, float | int] = {
        "count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
    }
    if finite.size == 0:
        return summary
    summary["mean"] = float(finite.mean())
    for quantile, value in zip(PERCENTILES, np.percentile(finite, PERCENTILES)):
        summary[f"p{quantile}"] = float(value)
    return summary


def build_descriptive_stats(scans: list[FileScan] | tuple[FileScan, ...]) -> dict[str, object]:
    """Peak-acceleration summaries for training files only."""

    train = [scan for scan in scans if scan.split == "train" and scan.signals_scanned]
    peaks, horizontal, labels, label_null = _train_columns(train)
    valid = ~label_null if label_null.size else label_null
    return {
        "role": "descriptive_not_applied",
        "split": "train",
        "note": "Computed only on scanned training files. Not a scaler and not a threshold.",
        "peak_acceleration_magnitude_g": _by_label(peaks[valid], labels[valid]),
        "peak_horizontal_acceleration_g": _by_label(horizontal[valid], labels[valid]),
    }


def _train_columns(scans: list[FileScan]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _concat([scan.peak_acceleration_g for scan in scans], np.float64),
        _concat([scan.peak_horizontal_g for scan in scans], np.float64),
        _concat([scan.labels for scan in scans], np.int64),
        _concat([scan.label_null for scan in scans], np.bool_),
    )


def _concat(arrays: list[np.ndarray], dtype: type) -> np.ndarray:
    usable = [array for array in arrays if array.size]
    if not usable:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(usable)


def _by_label(peaks: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float | int]]:
    summary = {"all": percentile_summary(peaks)}
    for label_id in LABEL_ORDER:
        selected = peaks[labels == label_id] if peaks.size else peaks
        summary[LABEL_NAMES[label_id]] = percentile_summary(selected)
    return summary
