"""Train-versus-shard summaries and logistic contributions. Nothing is clipped."""

from __future__ import annotations

import numpy as np

from accident_detection.features import FEATURE_NAMES

COMPARED_FEATURES: tuple[str, ...] = (
    "peak_horizontal_g",
    "peak_magnitude_g",
    "std_horizontal_g",
    "peak_abs_gyro_deg_s",
    "std_gyro_magnitude_deg_s",
    "gps_end_kmh",
    "gps_min_kmh",
    "gps_samples",
    "gyro_is_hd",
)


def z_score(value: float, mean: float, scale: float) -> float:
    """Standardize one raw value with a training mean and scale."""

    if scale == 0:
        return float("nan")
    return (float(value) - float(mean)) / float(scale)


def shard_summary(features: np.ndarray, y_crash: np.ndarray) -> dict[str, object]:
    """Describe the columns that can move the current logistic score."""

    gps_length = features[:, FEATURE_NAMES.index("gps_samples")]
    high_definition = features[:, FEATURE_NAMES.index("gyro_is_hd")]
    return {
        "rows": int(y_crash.size),
        "crash_rows": int(np.sum(y_crash == 1)),
        "gyro_is_hd_rows": int(np.sum(high_definition == 1)),
        "gps_length_16": int(np.sum(gps_length == 16)),
        "gps_length_17": int(np.sum(gps_length == 17)),
        "columns": {name: _distribution(features[:, FEATURE_NAMES.index(name)]) for name in COMPARED_FEATURES},
    }


def contribution_table(model, features: np.ndarray, names: tuple[str, ...], mask: np.ndarray) -> list[dict[str, float | str]]:
    """Mean training-scaled contribution of each coefficient on the selected rows."""

    transformed = _transformed(model, features)
    coefficients = model.named_steps["clf"].coef_.ravel()
    contribution = transformed * coefficients
    chosen = contribution[mask] if np.any(mask) else contribution
    means = chosen.mean(axis=0)
    rows = [_contribution_row(model, names, coefficients, means, index) for index in range(len(names))]
    return sorted(rows, key=lambda row: abs(float(row["mean_contribution"])), reverse=True)


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
    }


def _transformed(model, features: np.ndarray) -> np.ndarray:
    imputed = model.named_steps["imputer"].transform(features)
    return model.named_steps["scaler"].transform(imputed)


def _contribution_row(model, names, coefficients, means, index: int) -> dict[str, float | str]:
    scaler = model.named_steps["scaler"]
    return {
        "name": names[index],
        "coefficient": float(coefficients[index]),
        "train_mean": float(scaler.mean_[index]),
        "train_scale": float(scaler.scale_[index]),
        "mean_contribution": float(means[index]),
    }
