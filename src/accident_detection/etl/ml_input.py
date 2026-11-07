"""Read a loaded run back into the feature matrices the models already expect."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from accident_detection.etl.load import EVENT_FIELDS, connect
from accident_detection.features import FEATURE_NAMES, PEAK_SCORE_FEATURE
from accident_detection.labels import binary_crash_labels

OFFICIAL_SPLITS: tuple[str, ...] = ("train", "validation", "test")


def latest_loaded_run(db_path: Path) -> str | None:
    """Return the newest committed run id, if one exists."""

    connection = connect(db_path)
    try:
        row = connection.execute(
            "SELECT run_id FROM pipeline_runs WHERE status = 'loaded' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return str(row[0])


def load_splits(db_path: Path, run_id: str | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load official splits. Learned preprocessing is not fit here."""

    selected = run_id or latest_loaded_run(db_path)
    if selected is None:
        raise ValueError("no loaded ETL run is available")
    connection = connect(db_path)
    try:
        rows = _fetch_events(connection, selected)
    finally:
        connection.close()
    grouped = {split: [] for split in OFFICIAL_SPLITS}
    for row in rows:
        if row["split_name"] in grouped:
            grouped[row["split_name"]].append(row)
    return {split: _matrix(grouped[split]) for split in OFFICIAL_SPLITS}


def _fetch_events(connection: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    columns = ", ".join(name for name in EVENT_FIELDS if name != "run_id")
    return list(
        connection.execute(
            f"""
            SELECT {columns} FROM sensor_events
            WHERE run_id = ?
            ORDER BY source_file, row_index
            """,
            (run_id,),
        )
    )


def _matrix(rows: list[sqlite3.Row]) -> dict[str, np.ndarray]:
    features = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    labels = np.empty(len(rows), dtype=np.int64)
    event_ids = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        labels[index] = int(row["label"])
        event_ids[index] = int(row["event_id"])
        for column, name in enumerate(FEATURE_NAMES):
            value = row[name]
            features[index, column] = np.nan if value is None else float(value)
    return {
        "features": features,
        "y": binary_crash_labels(labels),
        "labels": labels,
        "event_ids": event_ids,
        "peak": features[:, FEATURE_NAMES.index(PEAK_SCORE_FEATURE)],
    }
