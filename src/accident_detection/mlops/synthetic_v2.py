"""Synthetic map sample. Places and times are generated, not taken from VZCrash."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SCHEMA = "synthetic-v2"
SEED = 2
CRASHES = 500
NEAR_MISSES = 150
NORMAL = 150
SOUTH = 12.85
NORTH = 13.10
WEST = 77.45
EAST = 77.75
GRID = 4
NOTE = "Generated coordinates and times. These are not VZCrash locations or official events."


def build_catalog(seed: int = SEED) -> dict[str, object]:
    """Build a deterministic synthetic v2 catalog."""

    rng = np.random.default_rng(seed)
    events = _group(rng, 0, "crash", CRASHES)
    events.extend(_group(rng, len(events), "near_miss", NEAR_MISSES))
    events.extend(_group(rng, len(events), "normal_driving", NORMAL))
    return {
        "schema": SCHEMA,
        "synthetic": True,
        "note": NOTE,
        "bounds": {"south": SOUTH, "north": NORTH, "west": WEST, "east": EAST},
        "counts": {"crash": CRASHES, "near_miss": NEAR_MISSES, "normal_driving": NORMAL},
        "events": events,
    }


def write_catalog(path: Path, seed: int = SEED) -> dict[str, object]:
    """Write the catalog as JSON and return it."""

    catalog = build_catalog(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog


def _group(rng: np.random.Generator, start: int, scenario: str, count: int) -> list[dict[str, object]]:
    label = {"crash": 0, "near_miss": 1, "normal_driving": 2}[scenario]
    return [_event(rng, start + index, label, scenario) for index in range(count)]


def _event(rng: np.random.Generator, index: int, label: int, scenario: str) -> dict[str, object]:
    latitude, longitude, locality = _place(rng)
    return {
        "synthetic_id": f"v2-{index:04d}",
        "synthetic": True,
        "scenario": scenario,
        "label": label,
        "latitude": latitude,
        "longitude": longitude,
        "locality": locality,
        "occurred_at": _time(rng),
        "features": _features(rng, scenario),
    }


def _place(rng: np.random.Generator) -> tuple[float, float, str]:
    latitude = round(float(rng.uniform(SOUTH, NORTH)), 6)
    longitude = round(float(rng.uniform(WEST, EAST)), 6)
    row = min(GRID - 1, int((latitude - SOUTH) / (NORTH - SOUTH) * GRID))
    column = min(GRID - 1, int((longitude - WEST) / (EAST - WEST) * GRID))
    return latitude, longitude, f"cell-{row}-{column}"


def _time(rng: np.random.Generator) -> str:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    stamp = start + timedelta(minutes=int(rng.integers(0, 180 * 24 * 60)))
    return stamp.isoformat()


def _features(rng: np.random.Generator, scenario: str) -> dict[str, float]:
    peak, drop = _scale(rng, scenario)
    start = float(rng.uniform(20, 90))
    end = max(0.0, start - drop)
    size = int(rng.integers(0, 3))
    return {
        "peak_horizontal_g": peak,
        "peak_magnitude_g": peak + float(rng.uniform(0.0, 0.4)),
        "min_ax_g": -peak,
        "max_ax_g": float(rng.uniform(0.0, 1.2)),
        "std_horizontal_g": float(rng.uniform(0.05, 0.4)),
        "gps_start_kmh": start,
        "gps_end_kmh": end,
        "gps_min_kmh": min(end, start * 0.2),
        "gps_drop_kmh": start - end,
        "size_light": float(size == 0),
        "size_medium": float(size == 1),
        "size_heavy": float(size == 2),
    }


def _scale(rng: np.random.Generator, scenario: str) -> tuple[float, float]:
    if scenario == "crash":
        return float(rng.uniform(2.4, 7.5)), float(rng.uniform(20, 90))
    if scenario == "near_miss":
        return float(rng.uniform(1.2, 2.2)), float(rng.uniform(5, 40))
    return float(rng.uniform(0.2, 0.9)), float(rng.uniform(0, 15))
