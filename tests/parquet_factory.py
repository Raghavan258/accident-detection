"""Small Parquet fixtures with the VZCrash physical schema."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

IMU_TYPE = pa.list_(pa.list_(pa.float32(), 3))
GPS_TYPE = pa.list_(pa.float32())


def xyz(count: int, value: float = 0.0) -> list[list[float]]:
    """Repeat one XYZ sample."""

    return [[value, value, value] for _ in range(count)]


def write_events(path: Path, events: list[dict]) -> None:
    """Write events that already use the physical VZCrash columns."""

    table = pa.table(
        {
            "event_id": pa.array([event["event_id"] for event in events], type=pa.int64()),
            "label": pa.array([event["label"] for event in events], type=pa.int64()),
            "vehiclesize": pa.array([event["vehiclesize"] for event in events], type=pa.string()),
            "gyro_is_hd": pa.array([event["gyro_is_hd"] for event in events], type=pa.bool_()),
            "gsensor": pa.array([event["gsensor"] for event in events], type=IMU_TYPE),
            "gyro": pa.array([event["gyro"] for event in events], type=IMU_TYPE),
            "gps_speed": pa.array([event["gps_speed"] for event in events], type=GPS_TYPE),
        }
    )
    pq.write_table(table, path)


def nominal_event(event_id: int, label: int, samples: int = 2, gps: int = 2) -> dict:
    """Build one structurally valid event at a caller-chosen length."""

    return {
        "event_id": event_id,
        "label": label,
        "vehiclesize": "Light",
        "gyro_is_hd": False,
        "gsensor": xyz(samples, 1.0),
        "gyro": xyz(samples, 2.0),
        "gps_speed": [10.0] * gps,
    }
