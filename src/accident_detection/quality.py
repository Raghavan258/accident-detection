"""Descriptive checks used to justify keeping unusual records."""

from __future__ import annotations

import numpy as np

from accident_detection.columns import ImuColumn, grouped_finite_max
from accident_detection.constants import NOMINAL_IMU_SAMPLES
from accident_detection.labels import CRASH_LABEL

GYRO_REPORT_LEVELS_DEG_S: tuple[float, ...] = (1000.0, 30000.0)


def summarize_gps(gps, labels: np.ndarray, gyro_is_hd: np.ndarray) -> list[dict[str, int]]:
    """Count GPS lengths and how often the last two samples match."""

    buckets: dict[int, dict[str, int]] = {}
    row_count = len(gps.offsets) - 1
    for index in range(row_count):
        sequence = gps.values[int(gps.offsets[index]) : int(gps.offsets[index + 1])]
        bucket = buckets.setdefault(int(sequence.size), _gps_bucket())
        bucket["rows"] += 1
        bucket["gyro_is_hd"] += int(bool(gyro_is_hd[index]))
        bucket["crash"] += int(int(labels[index]) == CRASH_LABEL)
        if _last_two_match(sequence):
            bucket["last_equals_previous"] += 1
    return [{"length": length, **buckets[length]} for length in sorted(buckets)]


def describe_gps_structure(gps) -> list[dict[str, int]]:
    """Count whether an unusual GPS length is an adjacent duplicate."""

    buckets: dict[int, dict[str, int]] = {}
    row_count = len(gps.offsets) - 1
    for index in range(row_count):
        sequence = gps.values[int(gps.offsets[index]) : int(gps.offsets[index + 1])]
        _add_gps_structure(buckets, sequence)
    return [{"length": length, **buckets[length]} for length in sorted(buckets)]


def summarize_imu(accel: ImuColumn, gyro: ImuColumn, labels: np.ndarray) -> dict[str, object]:
    """Describe IMU lengths without treating a non-1600 window as corrupt."""

    gsensor_lengths = np.diff(accel.offsets)
    gyro_lengths = np.diff(gyro.offsets)
    mismatch = np.flatnonzero(gsensor_lengths != gyro_lengths)
    unusual = (gsensor_lengths != NOMINAL_IMU_SAMPLES) | (gyro_lengths != NOMINAL_IMU_SAMPLES)
    return {
        "rows": int(gsensor_lengths.size),
        "unusual_rows": int(np.sum(unusual)),
        "mismatch_rows": int(mismatch.size),
        "max_gsensor_length": _max_or_zero(gsensor_lengths),
        "max_gyro_length": _max_or_zero(gyro_lengths),
        "mismatch_examples": [_length_example(gsensor_lengths, gyro_lengths, labels, int(i)) for i in mismatch[:5]],
    }


def summarize_gyro(gyro: ImuColumn, gyro_is_hd: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Summarize peak absolute gyroscope by the stored resolution flag."""

    peaks = _component_peaks(gyro)
    groups = [_gyro_group(peaks, gyro_is_hd, is_hd) for is_hd in (False, True)]
    above = np.flatnonzero(np.nan_to_num(peaks, nan=-1.0) > GYRO_REPORT_LEVELS_DEG_S[1])
    examples = [_extreme_example(peaks, gyro_is_hd, labels, int(index)) for index in above[:10]]
    return {"by_resolution": groups, "rows_above_30000_deg_s": examples}


def _gps_bucket() -> dict[str, int]:
    return {"rows": 0, "gyro_is_hd": 0, "crash": 0, "last_equals_previous": 0}


def _add_gps_structure(buckets: dict[int, dict[str, int]], sequence: np.ndarray) -> None:
    bucket = buckets.setdefault(int(sequence.size), _structure_bucket())
    last, first, adjacent = _structure_flags(sequence)
    bucket["rows"] += 1
    bucket["last_equals_previous"] += last
    bucket["first_equals_second"] += first
    bucket["any_adjacent_duplicate"] += adjacent
    bucket["no_adjacent_duplicate"] += 1 - adjacent


def _structure_bucket() -> dict[str, int]:
    return {
        "rows": 0,
        "last_equals_previous": 0,
        "first_equals_second": 0,
        "any_adjacent_duplicate": 0,
        "no_adjacent_duplicate": 0,
    }


def _structure_flags(sequence: np.ndarray) -> tuple[int, int, int]:
    if sequence.size < 2:
        return 0, 0, 0
    adjacent = _adjacent_equal(sequence[:-1], sequence[1:])
    return int(adjacent[-1]), int(adjacent[0]), int(np.any(adjacent))


def _adjacent_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    both = np.isfinite(left) & np.isfinite(right)
    return both & (left == right)


def _last_two_match(sequence: np.ndarray) -> bool:
    if sequence.size < 2:
        return False
    if not np.isfinite(sequence[-1]) or not np.isfinite(sequence[-2]):
        return False
    return bool(sequence[-1] == sequence[-2])


def _length_example(gsensor_lengths, gyro_lengths, labels, index: int) -> dict[str, int]:
    return {
        "gsensor_length": int(gsensor_lengths[index]),
        "gyro_length": int(gyro_lengths[index]),
        "label": int(labels[index]),
    }


def _component_peaks(gyro: ImuColumn) -> np.ndarray:
    if gyro.matrix.size == 0:
        return np.full(max(len(gyro.offsets) - 1, 0), np.nan)
    sample_peak = np.full(gyro.matrix.shape[0], np.nan, dtype=np.float64)
    finite_rows = np.isfinite(gyro.matrix).any(axis=1)
    sample_peak[finite_rows] = np.nanmax(np.abs(gyro.matrix[finite_rows]), axis=1)
    return grouped_finite_max(sample_peak, gyro.offsets)


def _gyro_group(peaks: np.ndarray, gyro_is_hd: np.ndarray, is_hd: bool) -> dict[str, object]:
    mask = np.asarray(gyro_is_hd, dtype=bool) == is_hd
    finite = peaks[mask]
    finite = finite[np.isfinite(finite)]
    summary: dict[str, object] = {"gyro_is_hd": is_hd, "rows": int(np.sum(mask))}
    if finite.size == 0:
        summary.update({"max_abs_deg_s": None, "p50_abs_deg_s": None, "rows_above_1000": 0, "rows_above_30000": 0})
        return summary
    summary.update(
        {
            "max_abs_deg_s": float(finite.max()),
            "p50_abs_deg_s": float(np.percentile(finite, 50)),
            "rows_above_1000": int(np.sum(finite > GYRO_REPORT_LEVELS_DEG_S[0])),
            "rows_above_30000": int(np.sum(finite > GYRO_REPORT_LEVELS_DEG_S[1])),
        }
    )
    return summary


def _extreme_example(peaks, gyro_is_hd, labels, index: int) -> dict[str, object]:
    return {
        "row_index": index,
        "peak_abs_deg_s": float(peaks[index]),
        "gyro_is_hd": bool(gyro_is_hd[index]),
        "label": int(labels[index]),
    }


def _max_or_zero(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    return int(values.max())
