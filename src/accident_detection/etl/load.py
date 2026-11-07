"""SQLite storage. One transaction per run. Raw IMU payloads are not copied."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from accident_detection.features import FEATURE_NAMES

EVENT_FIELDS: tuple[str, ...] = (
    "run_id",
    "source_file",
    "row_index",
    "event_id",
    "split_name",
    "label",
    "vehiclesize",
    "hd_flag",
    "gsensor_length",
    "gyro_length",
    "gps_length",
    "gsensor_nonfinite",
    "gyro_nonfinite",
    "gps_nonfinite",
    *FEATURE_NAMES,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    input_dir TEXT NOT NULL,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS source_shards (
    run_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    split_name TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    input_rows INTEGER NOT NULL,
    accepted_rows INTEGER NOT NULL,
    rejected_rows INTEGER NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    PRIMARY KEY (run_id, file_name),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);
CREATE TABLE IF NOT EXISTS sensor_events (
    run_id TEXT NOT NULL,
    source_file TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    split_name TEXT NOT NULL,
    label INTEGER NOT NULL CHECK (label IN (0, 1, 2)),
    vehiclesize TEXT,
    hd_flag INTEGER NOT NULL,
    gsensor_length INTEGER NOT NULL,
    gyro_length INTEGER NOT NULL,
    gps_length INTEGER NOT NULL,
    gsensor_nonfinite INTEGER NOT NULL,
    gyro_nonfinite INTEGER NOT NULL,
    gps_nonfinite INTEGER NOT NULL,
    peak_horizontal_g REAL,
    peak_magnitude_g REAL,
    min_ax_g REAL,
    max_ax_g REAL,
    std_horizontal_g REAL,
    peak_abs_gyro_deg_s REAL,
    std_gyro_magnitude_deg_s REAL,
    imu_samples REAL,
    gyro_samples REAL,
    gps_start_kmh REAL,
    gps_end_kmh REAL,
    gps_min_kmh REAL,
    gps_drop_kmh REAL,
    gps_samples REAL,
    gyro_is_hd REAL,
    size_light REAL,
    size_medium REAL,
    size_heavy REAL,
    PRIMARY KEY (run_id, source_file, row_index),
    FOREIGN KEY (run_id, source_file) REFERENCES source_shards(run_id, file_name)
);
CREATE TABLE IF NOT EXISTS rejected_records (
    run_id TEXT NOT NULL,
    source_file TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    event_id INTEGER,
    reason TEXT NOT NULL,
    PRIMARY KEY (run_id, source_file, row_index, reason),
    FOREIGN KEY (run_id, source_file) REFERENCES source_shards(run_id, file_name)
);
CREATE TABLE IF NOT EXISTS data_quality_results (
    run_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    detail TEXT,
    PRIMARY KEY (run_id, scope, metric),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_events_split ON sensor_events(run_id, split_name);
CREATE INDEX IF NOT EXISTS idx_events_event_id ON sensor_events(run_id, event_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open SQLite with foreign keys enforced."""

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path: Path) -> None:
    """Create tables if they are missing. Existing runs are left in place."""

    connection = connect(db_path)
    try:
        connection.executescript(SCHEMA_SQL)
        connection.commit()
    finally:
        connection.close()


def reset_database(db_path: Path) -> None:
    """Delete the database file. This runs only when the CLI flag is set."""

    path = Path(db_path)
    if path.exists():
        path.unlink()


def commit_run(db_path: Path, run: dict[str, object], shards: list[dict[str, object]]) -> None:
    """Insert one run. A failure rolls back events, shards, and the run row."""

    connection = connect(db_path)
    try:
        connection.execute("BEGIN")
        _insert_run(connection, run)
        for shard in shards:
            _insert_shard(connection, run["run_id"], shard)
        _insert_metrics(connection, str(run["run_id"]), run["metrics"])
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def record_failure(db_path: Path, run: dict[str, object]) -> None:
    """Store a failed run row after the data transaction has rolled back."""

    connection = connect(db_path)
    try:
        _insert_run(connection, run)
        connection.commit()
    finally:
        connection.close()


def _insert_run(connection: sqlite3.Connection, run: dict[str, object]) -> None:
    connection.execute(
        """
        INSERT INTO pipeline_runs (run_id, started_at, finished_at, status, input_dir, error_message)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run["run_id"],
            run["started_at"],
            run.get("finished_at"),
            run["status"],
            run["input_dir"],
            run.get("error_message"),
        ),
    )


def _insert_shard(connection: sqlite3.Connection, run_id: object, shard: dict[str, object]) -> None:
    connection.execute(
        """
        INSERT INTO source_shards (
            run_id, file_name, split_name, byte_size, input_rows, accepted_rows,
            rejected_rows, status, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            shard["file_name"],
            shard["split_name"],
            shard["byte_size"],
            shard["input_rows"],
            shard["accepted_rows"],
            shard["rejected_rows"],
            shard["status"],
            shard.get("error_message"),
        ),
    )
    _insert_events(connection, run_id, shard["accepted"])
    _insert_rejected(connection, run_id, shard["rejected"])


def _insert_events(connection: sqlite3.Connection, run_id: object, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = ",".join(EVENT_FIELDS)
    marks = ",".join("?" for _ in EVENT_FIELDS)
    payload = [tuple(run_id if name == "run_id" else row[name] for name in EVENT_FIELDS) for row in rows]
    connection.executemany(f"INSERT INTO sensor_events ({columns}) VALUES ({marks})", payload)


def _insert_rejected(connection: sqlite3.Connection, run_id: object, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO rejected_records (run_id, source_file, row_index, event_id, reason)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(run_id, row["source_file"], row["row_index"], row["event_id"], row["reason"]) for row in rows],
    )


def _insert_metrics(connection: sqlite3.Connection, run_id: str, metrics: list[dict[str, object]]) -> None:
    connection.executemany(
        """
        INSERT INTO data_quality_results (run_id, scope, metric, value, detail)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(run_id, item["scope"], item["metric"], item["value"], item.get("detail")) for item in metrics],
    )
