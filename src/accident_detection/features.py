"""Length-invariant features. Samples are not padded, truncated, or clipped."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa

from accident_detection.columns import decode_gps, decode_imu, integer_column

FEATURE_NAMES: tuple[str, ...] = (
    "peak_horizontal_g",
    "peak_magnitude_g",
    "min_ax_g",
    "max_ax_g",
    "std_horizontal_g",
    "peak_abs_gyro_deg_s",
    "std_gyro_magnitude_deg_s",
    "imu_samples",
    "gyro_samples",
    "gps_start_kmh",
    "gps_end_kmh",
    "gps_min_kmh",
    "gps_drop_kmh",
    "gps_samples",
    "gyro_is_hd",
    "size_light",
    "size_medium",
    "size_heavy",
)
PEAK_SCORE_FEATURE = "peak_horizontal_g"
ACCEL_FEATURE_NAMES: tuple[str, ...] = (
    "peak_horizontal_g",
    "peak_magnitude_g",
    "min_ax_g",
    "max_ax_g",
    "std_horizontal_g",
)
DEFENSIBLE_FEATURE_NAMES: tuple[str, ...] = ACCEL_FEATURE_NAMES + (
    "gps_start_kmh",
    "gps_end_kmh",
    "gps_min_kmh",
    "gps_drop_kmh",
    "size_light",
    "size_medium",
    "size_heavy",
)
EXCLUDED_FEATURE_REASONS: dict[str, str] = {
    "gps_samples": (
        "GPS list length is a storage count, not a speed. Standardizing it turns a one-sample "
        "format gap into a large score when one length dominates training."
    ),
    "imu_samples": "Accelerometer list length is a storage count, not an acceleration.",
    "gyro_samples": "Gyroscope list length is a storage count, not a rotation.",
    "gyro_is_hd": (
        "This is a sensor-resolution flag. Standardization does not put high-definition and "
        "standard gyroscope samples on one scale."
    ),
    "peak_abs_gyro_deg_s": (
        "Standard gyroscope rows can saturate near the int16 full scale, while high-definition "
        "rows use a finer resolution. One standardized column does not correct that encoding."
    ),
    "std_gyro_magnitude_deg_s": (
        "Same gyroscope encoding mismatch as the peak gyroscope feature."
    ),
}
_SIZE_COLUMNS = {
    "Light": "size_light",
    "Medium": "size_medium",
    "Heavy": "size_heavy",
}


def row_features(accel, gyro, gps, *, gyro_is_hd: bool, vehiclesize: str | None) -> dict[str, float]:
    """Reduce one event to a fixed feature dict. Empty channels become NaN."""

    features = {}
    features.update(_accel_features(np.asarray(accel, dtype=np.float64)))
    features.update(_gyro_features(np.asarray(gyro, dtype=np.float64)))
    features.update(_gps_features(np.asarray(gps, dtype=np.float64)))
    features["gyro_is_hd"] = 1.0 if gyro_is_hd else 0.0
    features.update(_size_features(vehiclesize))
    return features


def select_columns(features: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Return a column subset. The names are a fixed list, not a test-set search."""

    indexes = [FEATURE_NAMES.index(name) for name in names]
    return np.asarray(features)[:, indexes]


def as_vector(features: dict[str, float]) -> np.ndarray:
    """Return features in FEATURE_NAMES order."""

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise KeyError(", ".join(missing))
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def table_matrix(table: pa.Table) -> dict[str, np.ndarray | list]:
    """Build the feature matrix for one Parquet table without altering it."""

    accel = decode_imu(table["gsensor"])
    gyro = decode_imu(table["gyro"])
    gps = decode_gps(table["gps_speed"])
    labels, label_null = integer_column(table["label"])
    event_ids, event_null = integer_column(table["event_id"])
    hd = np.asarray(table["gyro_is_hd"].to_numpy(zero_copy_only=False), dtype=bool)
    sizes = table["vehiclesize"].to_pylist()
    matrix = _stack(accel, gyro, gps, hd, sizes)
    return {
        "features": matrix,
        "labels": labels,
        "label_null": label_null,
        "event_ids": event_ids,
        "event_id_null": event_null,
        "gyro_is_hd": hd,
        "accel": accel,
        "gyro": gyro,
        "gps": gps,
        "vehiclesize": sizes,
    }


def _stack(accel, gyro, gps, hd, sizes) -> np.ndarray:
    count = len(hd)
    matrix = np.empty((count, len(FEATURE_NAMES)), dtype=np.float64)
    for index in range(count):
        features = row_features(
            _group(accel.matrix, accel.offsets, index),
            _group(gyro.matrix, gyro.offsets, index),
            _group(gps.values, gps.offsets, index),
            gyro_is_hd=bool(hd[index]),
            vehiclesize=sizes[index],
        )
        matrix[index] = as_vector(features)
    return matrix


def _group(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    return values[int(offsets[index]) : int(offsets[index + 1])]


def _accel_features(accel: np.ndarray) -> dict[str, float]:
    horizontal = _horizontal(accel)
    magnitude = _vector_norm(accel)
    longitudinal = accel[:, 0] if accel.size else np.empty(0, dtype=np.float64)
    return {
        "peak_horizontal_g": _peak(horizontal),
        "peak_magnitude_g": _peak(magnitude),
        "min_ax_g": _trough(longitudinal),
        "max_ax_g": _peak(longitudinal),
        "std_horizontal_g": _std(horizontal),
        "imu_samples": float(accel.shape[0]),
    }


def _gyro_features(gyro: np.ndarray) -> dict[str, float]:
    return {
        "peak_abs_gyro_deg_s": _peak(_peak_abs_components(gyro)),
        "std_gyro_magnitude_deg_s": _std(_vector_norm(gyro)),
        "gyro_samples": float(gyro.shape[0]),
    }


def _peak_abs_components(gyro: np.ndarray) -> np.ndarray:
    if gyro.size == 0:
        return np.empty(0, dtype=np.float64)
    absolute = np.abs(gyro)
    absolute[~np.isfinite(gyro)] = -np.inf
    peaks = absolute.max(axis=1)
    peaks[peaks == -np.inf] = np.nan
    return peaks


def _gps_features(gps: np.ndarray) -> dict[str, float]:
    finite = gps[np.isfinite(gps)]
    if finite.size == 0:
        start = end = lowest = drop = math.nan
    else:
        start = float(finite[0])
        end = float(finite[-1])
        lowest = float(finite.min())
        drop = start - lowest
    return {
        "gps_start_kmh": start,
        "gps_end_kmh": end,
        "gps_min_kmh": lowest,
        "gps_drop_kmh": drop,
        "gps_samples": float(gps.size),
    }


def _size_features(vehiclesize: str | None) -> dict[str, float]:
    flags = {name: 0.0 for name in _SIZE_COLUMNS.values()}
    column = _SIZE_COLUMNS.get(vehiclesize or "")
    if column is not None:
        flags[column] = 1.0
    return flags


def _horizontal(accel: np.ndarray) -> np.ndarray:
    if accel.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(accel[:, :2], axis=1)


def _vector_norm(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(samples, axis=1)


def _peak(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(finite.max())


def _trough(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(finite.min())


def _std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return math.nan
    return float(finite.std(ddof=0))
