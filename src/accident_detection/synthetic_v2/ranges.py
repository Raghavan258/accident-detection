"""Named ranges for the twelve production columns. Gyro is recorded, not scored."""

from __future__ import annotations

import pandas as pd

from accident_detection.features import DEFENSIBLE_FEATURE_NAMES, EXCLUDED_FEATURE_REASONS
from accident_detection.synthetic_v2.telemetry import features_from_telemetry

LABELS = {
    "peak_horizontal_g": "Peak G-Force",
    "peak_magnitude_g": "Peak acceleration magnitude",
    "min_ax_g": "Minimum longitudinal acceleration",
    "max_ax_g": "Maximum longitudinal acceleration",
    "std_horizontal_g": "Horizontal acceleration variation",
    "gps_start_kmh": "Pre-impact speed",
    "gps_end_kmh": "Post-impact speed",
    "gps_min_kmh": "Lowest speed in the window",
    "gps_drop_kmh": "Speed drop",
    "size_light": "Light vehicle",
    "size_medium": "Medium vehicle",
    "size_heavy": "Heavy vehicle",
}
UNITS = {
    "peak_horizontal_g": "g",
    "peak_magnitude_g": "g",
    "min_ax_g": "g",
    "max_ax_g": "g",
    "std_horizontal_g": "g",
    "gps_start_kmh": "km/h",
    "gps_end_kmh": "km/h",
    "gps_min_kmh": "km/h",
    "gps_drop_kmh": "km/h",
}
HAND_TIERS = {
    "peak_horizontal_g": (
        ("normal", "No Accident", 0.0, 1.0, 0.6),
        ("elevated", "Small Accident / Near-Miss", 1.0, 2.2, 1.6),
        ("critical", "Severe Crash", 2.2, 5.0, 3.8),
    ),
    "gps_drop_kmh": (
        ("normal", "Smooth Cruise", 0.0, 5.0, 2.0),
        ("elevated", "Hard Brake / Minor Impact", 15.0, 27.0, 27.0),
        ("critical", "Sudden Stop / Crash", 35.0, 80.0, 48.0),
    ),
}
GYRO_TIERS = (
    ("normal", "Stable Lane", 0.0, 15.0, 8.0),
    ("elevated", "Sharp Swerve", 15.0, 45.0, 30.0),
    ("critical", "Spin / Rollover Jolt", 45.0, 180.0, 70.0),
)
PRESETS = (
    ("normal", "Normal Drive", 0.6, 40.0, 38.0, "Light", "normal"),
    ("near_miss", "Small Accident", 1.6, 45.0, 18.0, "Medium", "near_miss"),
    ("crash", "Severe Crash", 3.8, 48.0, 0.0, "Medium", "crash"),
)


def schema_ranges(frame: pd.DataFrame) -> dict[str, object]:
    """Return pill ranges for the frozen feature list plus the excluded gyro channel."""

    columns = [_column(name, frame) for name in DEFENSIBLE_FEATURE_NAMES]
    return {
        "columns": columns,
        "feature_names": list(DEFENSIBLE_FEATURE_NAMES),
        "gyro": _gyro(),
        "presets": [_preset(*item) for item in PRESETS],
        "synthetic": True,
    }


def preset_features(name: str) -> dict[str, float]:
    """Return the twelve-column vector for one named preset."""

    row = next(item for item in PRESETS if item[0] == name)
    return features_from_telemetry(row[2], row[3], row[4], row[5])


def _column(name: str, frame: pd.DataFrame) -> dict[str, object]:
    if name in HAND_TIERS:
        tiers = [_tier(*item) for item in HAND_TIERS[name]]
    elif name.startswith("size_"):
        tiers = _size_tiers(name)
    else:
        tiers = _quantile_tiers(frame[name] if name in frame.columns else pd.Series(dtype=float))
    return {"name": name, "label": LABELS[name], "unit": UNITS.get(name, ""), "used_by_model": True, "tiers": tiers}


def _tier(tier_id: str, title: str, low: float, high: float, value: float) -> dict[str, object]:
    return {"id": tier_id, "name": title, "min": round(float(low), 3), "max": round(float(high), 3), "value": round(float(value), 3)}


def _size_tiers(name: str) -> list[dict[str, object]]:
    selected = 1.0
    return [
        _tier("normal", "Normal Range", 0.0, 0.0, 0.0 if name != "size_light" else selected),
        _tier("elevated", "Elevated / Borderline", 0.0, 1.0, selected if name == "size_medium" else 0.0),
        _tier("critical", "Critical Impact", 1.0, 1.0, selected if name == "size_heavy" else 0.0),
    ]


def _quantile_tiers(series: pd.Series) -> list[dict[str, object]]:
    if series.empty:
        empty = _tier("normal", "Normal Range", 0.0, 0.0, 0.0)
        return [empty, _tier("elevated", "Elevated / Borderline", 0.0, 0.0, 0.0), _tier("critical", "Critical Impact", 0.0, 0.0, 0.0)]
    low, mid, high = (float(series.quantile(q)) for q in (0.33, 0.66, 1.0))
    floor = float(series.min())
    return [
        _tier("normal", "Normal Range", floor, low, _middle(series[series <= low], floor)),
        _tier("elevated", "Elevated / Borderline", low, mid, _middle(series[(series > low) & (series <= mid)], mid)),
        _tier("critical", "Critical Impact", mid, high, _middle(series[series > mid], high)),
    ]


def _middle(series: pd.Series, fallback: float) -> float:
    if series.empty:
        return fallback
    value = float(series.median())
    return fallback if value != value else value


def _gyro() -> dict[str, object]:
    return {
        "name": "peak_abs_gyro_deg_s",
        "label": "Gyroscope angular rate",
        "unit": "°/s",
        "used_by_model": False,
        "reason": EXCLUDED_FEATURE_REASONS["peak_abs_gyro_deg_s"],
        "tiers": [_tier(*item) for item in GYRO_TIERS],
    }


def _preset(preset_id: str, title: str, peak: float, start: float, end: float, size: str, scenario: str) -> dict[str, object]:
    gyro = {"normal": 8.0, "near_miss": 30.0, "crash": 70.0}[preset_id]
    return {
        "id": preset_id,
        "name": title,
        "scenario_preset": scenario,
        "vehicle_size": size,
        "peak_g": peak,
        "speed_before": start,
        "speed_after": end,
        "gyro_deg_s": gyro,
        "features": features_from_telemetry(peak, start, end, size),
    }
