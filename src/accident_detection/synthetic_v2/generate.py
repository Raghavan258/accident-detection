"""Build a seeded event table inspired by the VZCrash task.

Coordinates are sampled inside declared rectangles. They are not locations
from the VZCrash files, and those files contain no latitude or longitude.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from accident_detection.constants import LABEL_NAMES
from accident_detection.synthetic_v2.localities import LOCALITIES, LocalityBox

SEED = 20261125
COUNTS = {"crash": 200, "near_miss": 150, "normal_driving": 650}
LABEL_IDS = {"crash": 0, "near_miss": 1, "normal_driving": 2}
SPLIT_FRACTIONS = (("train", 0.70), ("validation", 0.15), ("test", 0.15))
WINDOW_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
WINDOW_DAYS = 366
PEAK_CLIP_G = (0.05, 12.0)
SIZES = ("Light", "Medium", "Heavy")


def generate_events(seed: int = SEED, counts: dict[str, int] | None = None) -> pd.DataFrame:
    """Return one synthetic event per row, with a stratified split."""

    chosen = dict(COUNTS if counts is None else counts)
    rng = np.random.default_rng(seed)
    rows = [_event(rng, index, label) for label, count in chosen.items() for index in range(count)]
    frame = pd.DataFrame(rows)
    frame.insert(0, "event_id", np.arange(1, len(frame) + 1))
    return _assign_splits(frame, rng)


def _event(rng: np.random.Generator, index: int, label_name: str) -> dict[str, object]:
    label = LABEL_IDS[label_name]
    box = LOCALITIES[int(rng.integers(0, len(LOCALITIES)))]
    peak, magnitude, min_ax, max_ax, std_h = _motion(rng, label)
    start, end, minimum, drop = _speed(rng, label)
    size = str(rng.choice(SIZES))
    return {
        "source": "synthetic_v2",
        "label": label,
        "label_name": LABEL_NAMES[label],
        "latitude": _uniform(rng, box["lat_min"], box["lat_max"]),
        "longitude": _uniform(rng, box["lon_min"], box["lon_max"]),
        "locality": box["name"],
        "observed_at": _timestamp(rng, index),
        "crs": "EPSG:4326",
        "vehiclesize": size,
        "peak_horizontal_g": peak,
        "peak_magnitude_g": magnitude,
        "min_ax_g": min_ax,
        "max_ax_g": max_ax,
        "std_horizontal_g": std_h,
        "gps_start_kmh": start,
        "gps_end_kmh": end,
        "gps_min_kmh": minimum,
        "gps_drop_kmh": drop,
        "size_light": float(size == "Light"),
        "size_medium": float(size == "Medium"),
        "size_heavy": float(size == "Heavy"),
    }


def _motion(rng: np.random.Generator, label: int) -> tuple[float, float, float, float, float]:
    center = {0: 1.05, 1: 0.35, 2: -0.45}[label]
    peak = float(np.clip(rng.lognormal(center, 0.28), *PEAK_CLIP_G))
    magnitude = peak + float(rng.uniform(0.02, 0.45))
    min_ax = -float(rng.uniform(0.4, 1.0)) * peak
    max_ax = float(rng.uniform(0.05, 0.45)) * peak
    std_h = float(np.clip(rng.uniform(0.05, 0.25) * peak, 0.01, peak))
    return peak, magnitude, min_ax, max_ax, std_h


def _speed(rng: np.random.Generator, label: int) -> tuple[float, float, float, float]:
    start = float(rng.uniform(8.0, 100.0))
    if label == 0:
        end = float(rng.uniform(0.0, min(18.0, start)))
    elif label == 1:
        end = float(rng.uniform(0.0, start * 0.75))
    else:
        end = float(rng.uniform(start * 0.65, start))
    minimum = float(min(start, end))
    return start, end, minimum, start - end


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _timestamp(rng: np.random.Generator, index: int) -> str:
    del index
    offset = timedelta(seconds=int(rng.integers(0, WINDOW_DAYS * 24 * 60 * 60)))
    return (WINDOW_START + offset).strftime("%Y-%m-%dT%H:%M:%SZ")


def _assign_splits(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    split = np.empty(len(frame), dtype=object)
    for label in sorted(frame["label"].unique()):
        positions = np.flatnonzero(frame["label"].to_numpy() == label)
        rng.shuffle(positions)
        split[positions] = _bands(len(positions))
    out = frame.copy()
    out["split"] = split
    return out


def _bands(count: int) -> list[str]:
    train = int(round(count * SPLIT_FRACTIONS[0][1]))
    validation = int(round(count * SPLIT_FRACTIONS[1][1]))
    names = ["train"] * train + ["validation"] * validation
    names += ["test"] * (count - len(names))
    return names


def locality_boxes() -> tuple[LocalityBox, ...]:
    """Return the rectangles used to sample latitude and longitude."""

    return LOCALITIES
