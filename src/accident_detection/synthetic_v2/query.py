"""Filter the synthetic v2 table. No rows are invented here."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from accident_detection.synthetic_v2.localities import LOCALITIES

PREVIEW_LIMIT = 8


def load_events(path: Path) -> pd.DataFrame:
    """Read the synthetic event table."""

    return pd.read_parquet(path)


def filter_events(frame: pd.DataFrame, locality: str | None, label_mode: str | None) -> pd.DataFrame:
    """Keep rows for one locality and one label choice."""

    selected = frame
    if locality is not None:
        selected = selected.loc[selected["locality"].str.lower() == locality.lower()]
    if label_mode == "crash":
        selected = selected.loc[selected["label"] == 0]
    elif label_mode == "non_crash":
        selected = selected.loc[selected["label"] != 0]
    elif label_mode == "near_miss":
        selected = selected.loc[selected["label"] == 1]
    elif label_mode == "normal_driving":
        selected = selected.loc[selected["label"] == 2]
    return selected


def crash_density(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Crash count and crashes per square kilometre inside each sampling rectangle."""

    crashes = frame.loc[frame["label"] == 0]
    counts = crashes.groupby("locality").size().to_dict()
    rows = [_density_row(box, int(counts.get(box["name"], 0))) for box in LOCALITIES]
    return sorted(rows, key=lambda item: float(item["crashes_per_km2"]), reverse=True)


def preview_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return a short list of events for the chat card."""

    columns = ["event_id", "label_name", "locality", "latitude", "longitude", "observed_at", "peak_horizontal_g"]
    subset = frame.loc[:, columns].head(PREVIEW_LIMIT)
    return subset.to_dict(orient="records")


def _density_row(box: dict[str, float | str], crashes: int) -> dict[str, object]:
    area = _area_km2(box)
    return {
        "locality": box["name"],
        "crashes": crashes,
        "area_km2": round(area, 3),
        "crashes_per_km2": round(crashes / area, 3) if area else 0.0,
    }


def _area_km2(box: dict[str, float | str]) -> float:
    lat_km = (float(box["lat_max"]) - float(box["lat_min"])) * 110.574
    mid = (float(box["lat_min"]) + float(box["lat_max"])) / 2.0
    lon_km = (float(box["lon_max"]) - float(box["lon_min"])) * 111.320 * math.cos(math.radians(mid))
    return abs(lat_km * lon_km)
