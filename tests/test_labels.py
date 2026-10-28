"""Crash-class mapping keeps near-miss events as negatives."""

import numpy as np
import pytest

from accident_detection.labels import binary_crash_labels, crash_name


def test_only_crash_is_positive() -> None:
    labels = np.array([0, 1, 2], dtype=np.int64)
    assert binary_crash_labels(labels).tolist() == [1, 0, 0]
    assert crash_name(0) == "crash"
    assert crash_name(1) == "near_miss"


def test_unknown_and_null_labels_are_rejected() -> None:
    with pytest.raises(ValueError):
        binary_crash_labels(np.array([3], dtype=np.int64))
    with pytest.raises(ValueError):
        binary_crash_labels(np.array([0], dtype=np.int64), np.array([True]))
