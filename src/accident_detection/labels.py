"""Map published VZCrash labels onto the crash-detection task."""

from __future__ import annotations

import numpy as np

from accident_detection.constants import LABEL_NAMES, LABEL_ORDER

CRASH_LABEL = 0
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0


def binary_crash_labels(labels: np.ndarray, is_null: np.ndarray | None = None) -> np.ndarray:
    """Return 1 for crash and 0 for near-miss and normal driving.

    Near-miss events stay negative examples, as in the VZCrash paper.
    """

    if is_null is not None and np.any(is_null):
        raise ValueError("null label cannot be mapped")
    unknown = ~np.isin(labels, np.array(LABEL_ORDER, dtype=labels.dtype))
    if np.any(unknown):
        raise ValueError("label is outside the published map")
    return np.where(labels == CRASH_LABEL, POSITIVE_CLASS, NEGATIVE_CLASS).astype(np.int8)


def crash_name(label: int) -> str:
    """Return the published name for one label id."""

    if int(label) not in LABEL_NAMES:
        raise ValueError(f"label is outside the published map: {label}")
    return LABEL_NAMES[int(label)]
