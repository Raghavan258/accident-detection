"""Score one event with the production pipeline."""

from __future__ import annotations

import time

import numpy as np

from accident_detection.baselines import logistic_scores
from accident_detection.features import row_features
from accident_detection.mlops import MODEL_NAME


def predict_from_features(bundle: dict[str, object], features: dict[str, float]) -> dict[str, object]:
    """Return a crash decision, probability, and serving metadata."""

    names = list(bundle["decision"]["feature_names"])
    missing = [name for name in names if name not in features]
    if missing:
        raise KeyError(", ".join(missing))
    vector = np.array([[float(features[name]) for name in names]], dtype=np.float64)
    return _score(bundle, vector, names)


def predict_from_window(bundle: dict[str, object], window: dict[str, object]) -> dict[str, object]:
    """Reduce a sensor window with the existing feature function, then score it."""

    features = row_features(
        window.get("gsensor") or [],
        window.get("gyro") or [],
        window.get("gps_speed") or [],
        gyro_is_hd=bool(window.get("gyro_is_hd", False)),
        vehiclesize=window.get("vehiclesize"),
    )
    return predict_from_features(bundle, features)


def _score(bundle: dict[str, object], vector: np.ndarray, names: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    probability = float(logistic_scores(bundle["model"], vector)[0])
    latency_ms = (time.perf_counter() - started) * 1000.0
    threshold = float(bundle["decision"]["threshold"])
    return {
        "prediction": "crash" if probability >= threshold else "non_crash",
        "crash": int(probability >= threshold),
        "probability": probability,
        "threshold": threshold,
        "model_name": MODEL_NAME,
        "model_version": bundle["version"],
        "feature_names": names,
        "latency_ms": latency_ms,
    }
