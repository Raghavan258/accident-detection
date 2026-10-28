"""Vectorized Parquet column checks. Rows are flagged, not removed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from accident_detection.constants import IMU_AXES, LABEL_NAMES, LABEL_ORDER


@dataclass(frozen=True)
class ImuColumn:
    """Decoded accelerometer or gyroscope samples aligned to list offsets."""

    offsets: np.ndarray
    matrix: np.ndarray
    sample_null: np.ndarray
    axis_null: np.ndarray
    nonfinite: np.ndarray


@dataclass(frozen=True)
class GpsColumn:
    """Decoded GPS speeds aligned to list offsets."""

    offsets: np.ndarray
    values: np.ndarray
    sample_null: np.ndarray
    nonfinite: np.ndarray


def list_lengths(column: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """Return list lengths and a mask that is true when the list itself is null."""

    lengths = pc.list_value_length(column)
    is_null = np.asarray(pc.is_null(lengths).to_numpy(zero_copy_only=False), dtype=bool)
    filled = lengths.fill_null(0).to_numpy(zero_copy_only=False)
    return np.asarray(filled, dtype=np.int64), is_null


def histogram(lengths: np.ndarray, is_null: np.ndarray) -> dict[int, int]:
    """Count non-null list lengths."""

    observed = lengths[~is_null]
    if observed.size == 0:
        return {}
    keys, counts = np.unique(observed, return_counts=True)
    return {int(key): int(count) for key, count in zip(keys, counts)}


def grouped_finite_max(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return the finite maximum of each offset group. Empty groups are NaN."""

    group_count = len(offsets) - 1
    peaks = np.full(group_count, np.nan, dtype=np.float64)
    if values.size == 0 or group_count == 0:
        return peaks
    safe = np.where(np.isfinite(values), values, -np.inf)
    lengths = np.diff(offsets)
    starts = offsets[:-1][lengths > 0]
    if starts.size == 0:
        return peaks
    peaks[lengths > 0] = np.maximum.reduceat(safe, starts)
    peaks[peaks == -np.inf] = np.nan
    return peaks


def sum_by_group(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Sum sample-level integers into each list group."""

    group_count = len(offsets) - 1
    totals = np.zeros(group_count, dtype=np.int64)
    lengths = np.diff(offsets)
    if values.size == 0 or not np.any(lengths > 0):
        return totals
    totals[lengths > 0] = np.add.reduceat(np.asarray(values, dtype=np.int64), offsets[:-1][lengths > 0])
    return totals


def finite_extrema(values: np.ndarray) -> dict[str, float | int | None]:
    """Minimum and maximum of finite values only."""

    finite = values[np.isfinite(values)]
    result: dict[str, float | int | None] = {
        "finite_count": int(finite.size),
        "min": None,
        "max": None,
    }
    if finite.size == 0:
        return result
    result["min"] = float(finite.min())
    result["max"] = float(finite.max())
    return result


def decode_imu(column: pa.ChunkedArray) -> ImuColumn:
    """Decode a list of XYZ samples. Null axes stay NaN and are counted apart."""

    combined = column.combine_chunks()
    offsets = np.asarray(combined.offsets.to_numpy(), dtype=np.int64)
    matrix, sample_null, axis_null = _matrix_from_samples(combined.values)
    nonfinite = np.zeros(matrix.shape, dtype=bool)
    present = ~sample_null[:, None] & ~axis_null
    nonfinite[present] = ~np.isfinite(matrix[present])
    return ImuColumn(offsets, matrix, sample_null, axis_null, nonfinite)


def decode_gps(column: pa.ChunkedArray) -> GpsColumn:
    """Decode a list of scalar speeds. Null samples stay NaN and are counted apart."""

    combined = column.combine_chunks()
    offsets = np.asarray(combined.offsets.to_numpy(), dtype=np.int64)
    sample_null = _value_null_mask(combined.values)
    filled = combined.values.fill_null(float("nan")).to_numpy(zero_copy_only=False)
    values = np.asarray(filled, dtype=np.float64)
    nonfinite = ~sample_null & ~np.isfinite(values)
    return GpsColumn(offsets, values, sample_null, nonfinite)


def imu_null_and_nonfinite(decoded: ImuColumn) -> tuple[np.ndarray, np.ndarray]:
    """Return per-event counts of missing scalars and non-finite scalars."""

    per_sample_null = decoded.sample_null.astype(np.int64) * IMU_AXES
    per_sample_null = per_sample_null + decoded.axis_null.sum(axis=1)
    per_sample_nonfinite = decoded.nonfinite.sum(axis=1)
    return (
        sum_by_group(per_sample_null, decoded.offsets),
        sum_by_group(per_sample_nonfinite, decoded.offsets),
    )


def sample_magnitudes(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 3-axis and horizontal magnitudes. A missing axis makes that magnitude NaN."""

    magnitude = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    horizontal = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    complete = np.isfinite(matrix).all(axis=1)
    magnitude[complete] = np.linalg.norm(matrix[complete], axis=1)
    xy_ready = np.isfinite(matrix[:, :2]).all(axis=1)
    horizontal[xy_ready] = np.linalg.norm(matrix[xy_ready, :2], axis=1)
    return magnitude, horizontal


def count_labels(labels: np.ndarray, is_null: np.ndarray) -> dict[int, int]:
    """Count labels, keeping a zero for every published class."""

    counts = {label_id: 0 for label_id in LABEL_ORDER}
    observed = labels[~is_null] if labels.size else labels
    if observed.size == 0:
        return counts
    keys, amounts = np.unique(observed, return_counts=True)
    for key, amount in zip(keys, amounts):
        counts[int(key)] = int(amount)
    return counts


def integer_column(column: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """Return integer values with nulls filled by 0, plus the null mask."""

    is_null = np.asarray(column.is_null().to_numpy(zero_copy_only=False), dtype=bool)
    filled = column.fill_null(0).to_numpy(zero_copy_only=False)
    return np.asarray(filled, dtype=np.int64), is_null


def string_counts(values: list[object]) -> dict[str, int]:
    """Count categorical values. Nulls are recorded as the key 'null'."""

    counts: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def bool_counts(values: list[object]) -> dict[str, int]:
    """Count booleans as true, false, and null."""

    counts = {"true": 0, "false": 0, "null": 0}
    for value in values:
        if value is None:
            counts["null"] += 1
        elif value:
            counts["true"] += 1
        else:
            counts["false"] += 1
    return counts


def label_name(label: int | None) -> str:
    """Return the published class name, or an empty string when the label is missing."""

    if label is None:
        return ""
    return LABEL_NAMES.get(int(label), "unknown")


def _matrix_from_samples(samples: pa.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(samples)
    matrix = np.full((count, IMU_AXES), np.nan, dtype=np.float64)
    sample_null = _value_null_mask(samples)
    axis_null = np.zeros((count, IMU_AXES), dtype=bool)
    if count == 0 or bool(sample_null.all()):
        return matrix, sample_null, axis_null
    valid = samples.filter(pa.array(~sample_null)) if sample_null.any() else samples
    flat = pc.list_flatten(valid)
    shaped = _filled_xyz(flat)
    if shaped.shape[0] != int((~sample_null).sum()):
        raise ValueError("gyroscope/accelerometer layout does not match list offsets")
    matrix[~sample_null] = shaped
    axis_null[~sample_null] = flat.is_null().to_numpy(zero_copy_only=False).reshape(-1, IMU_AXES)
    matrix[axis_null] = np.nan
    return matrix, sample_null, axis_null


def _filled_xyz(flat: pa.Array) -> np.ndarray:
    numeric = flat.fill_null(0).to_numpy(zero_copy_only=False)
    return np.asarray(numeric, dtype=np.float64).reshape(-1, IMU_AXES)


def _value_null_mask(values: pa.Array) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(values.is_null().to_numpy(zero_copy_only=False), dtype=bool)
