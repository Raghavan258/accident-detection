"""Export the existing SQLite feature store. Raw Parquet is not copied."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from accident_detection.etl.ml_input import latest_loaded_run, load_splits
from accident_detection.features import FEATURE_NAMES

SPLIT_COLUMN = "split"


def export_feature_table(db_path: Path, destination: Path, run_id: str | None = None) -> dict[str, object]:
    """Write one processed table from a committed ETL run."""

    selected = run_id or latest_loaded_run(db_path)
    if selected is None:
        raise ValueError("no loaded ETL run is available")
    frame = frame_from_splits(load_splits(db_path, selected))
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return {"run_id": selected, "rows": int(len(frame)), "path": str(destination)}


def frame_from_splits(splits: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Stack official splits without fitting a preprocessor."""

    frames = [_split_frame(name, part) for name, part in splits.items()]
    return pd.concat(frames, ignore_index=True)


def splits_from_frame(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Rebuild the matrices `evaluate` already expects."""

    return {name: _part(frame.loc[frame[SPLIT_COLUMN] == name]) for name in ("train", "validation", "test")}


def _split_frame(name: str, part: dict[str, np.ndarray]) -> pd.DataFrame:
    columns = {feature: part["features"][:, index] for index, feature in enumerate(FEATURE_NAMES)}
    columns[SPLIT_COLUMN] = name
    columns["label"] = part["labels"]
    columns["event_id"] = part["event_ids"]
    return pd.DataFrame(columns)


def _part(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    features = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    from accident_detection.labels import binary_crash_labels

    return {
        "features": features,
        "y": binary_crash_labels(labels),
        "labels": labels,
        "event_ids": frame["event_id"].to_numpy(dtype=np.int64),
        "peak": features[:, FEATURE_NAMES.index("peak_horizontal_g")],
    }
