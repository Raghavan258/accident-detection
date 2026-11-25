"""Map simulator telemetry onto the twelve production features.

The production logistic model is not changed. These formulas only fill the
feature vector the frozen model already expects.
"""

from __future__ import annotations

SIZES = ("Light", "Medium", "Heavy")


def features_from_telemetry(peak_g: float, speed_before: float, speed_after: float, vehicle_size: str) -> dict[str, float]:
    """Build the defensible feature dict from one telemetry sample."""

    peak = float(peak_g)
    start = float(speed_before)
    end = float(speed_after)
    features = {
        "peak_horizontal_g": peak,
        "peak_magnitude_g": peak + 0.08,
        "min_ax_g": -0.9 * peak,
        "max_ax_g": 0.28 * peak,
        "std_horizontal_g": max(0.02, 0.12 * peak),
        "gps_start_kmh": start,
        "gps_end_kmh": end,
        "gps_min_kmh": min(start, end),
        "gps_drop_kmh": start - end,
        "size_light": 0.0,
        "size_medium": 0.0,
        "size_heavy": 0.0,
    }
    features[_size_column(vehicle_size)] = 1.0
    return features


def simulation_label(prediction: str, preset: str) -> str:
    """Keep a model crash. A non-crash near-miss preset stays a near-miss."""

    if prediction == "crash":
        return "crash"
    if preset == "near_miss":
        return "near_miss"
    return "normal_driving"


def _size_column(vehicle_size: str) -> str:
    if vehicle_size not in SIZES:
        raise ValueError("vehicle_size must be Light, Medium, or Heavy")
    return f"size_{vehicle_size.lower()}"
