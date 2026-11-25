"""Log-odds shift of each frozen feature versus the model's training median."""

from __future__ import annotations

import numpy as np

from accident_detection.synthetic_v2.ranges import LABELS


def column_contributions(bundle: dict[str, object], features: dict[str, float]) -> list[dict[str, object]]:
    """Return one row per model column. An unscored bundle returns no rows."""

    model = bundle.get("model") if isinstance(bundle, dict) else None
    names = _names(bundle)
    if model is None or not names or not hasattr(model, "named_steps"):
        return []
    coef, shift = _delta(model, names, features)
    return [_row(names[index], features[names[index]], coef[index], shift[index]) for index in range(len(names))]


def top_triggers(rows: list[dict[str, object]], limit: int = 3) -> list[dict[str, object]]:
    """Columns that pushed the log-odds toward crash, largest first."""

    ordered = sorted(rows, key=lambda item: float(item["delta"]), reverse=True)
    return ordered[:limit]


def severity_tier(peak_g: float, prediction: str) -> str:
    """Name the g-force band shown on the helper banner."""

    if peak_g >= 2.2:
        return "Severe Crash"
    if peak_g >= 1.0 or prediction == "crash":
        return "Small Accident"
    return "Normal Drive"


def _names(bundle: dict[str, object]) -> list[str]:
    decision = bundle.get("decision") if isinstance(bundle, dict) else None
    if not isinstance(decision, dict):
        return []
    return [str(name) for name in decision.get("feature_names", [])]


def _delta(model: object, names: list[str], features: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    current = np.array([[float(features[name]) for name in names]], dtype=np.float64)
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    baseline = np.asarray(imputer.statistics_, dtype=np.float64).reshape(1, -1)
    shift = scaler.transform(imputer.transform(current)) - scaler.transform(imputer.transform(baseline))
    return _crash_coef(model), shift[0]


def _crash_coef(model: object) -> np.ndarray:
    coef = np.asarray(model.named_steps["clf"].coef_, dtype=np.float64).ravel()
    classes = list(model.named_steps["clf"].classes_)
    if len(classes) > 1 and classes[1] != 1:
        return -coef
    return coef


def _row(name: str, value: float, coef: float, shift: float) -> dict[str, object]:
    delta = float(coef) * float(shift)
    toward_crash = delta >= 0
    return {
        "name": name,
        "label": LABELS.get(name, name),
        "value": round(float(value), 4),
        "delta": round(delta, 4),
        "direction": "toward_crash" if toward_crash else "toward_normal",
        "sign": "+" if toward_crash else "-",
    }
