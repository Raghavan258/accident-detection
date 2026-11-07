"""Turn one Parquet table into accepted event rows and explicit rejections."""

from __future__ import annotations

import math

import numpy as np

from accident_detection.columns import imu_null_and_nonfinite, sum_by_group
from accident_detection.constants import LABEL_ORDER
from pathlib import Path

from accident_detection.features import FEATURE_NAMES, table_matrix
from accident_detection.schema import split_name

METADATA_FIELDS: tuple[str, ...] = (
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
)


def records_from_table(table, source_name: str, row_offset: int) -> tuple[list[dict], list[dict]]:
    """Map one table. Invalid labels and null identifiers are rejected, not guessed."""

    batch = table_matrix(table)
    signal_counts = _signal_counts(batch)
    accepted: list[dict] = []
    rejected: list[dict] = []
    for index in range(table.num_rows):
        _append_row(accepted, rejected, batch, signal_counts, source_name, row_offset + index, index)
    return accepted, rejected


def _append_row(accepted, rejected, batch, signal_counts, source_name: str, row_index: int, index: int) -> None:
    reasons = _rejection_reasons(batch, index)
    if reasons:
        rejected.extend(_rejection(source_name, row_index, batch, index, reason) for reason in reasons)
        return
    accepted.append(_accepted(batch, signal_counts, source_name, row_index, index))


def _rejection_reasons(batch, index: int) -> list[str]:
    reasons: list[str] = []
    if bool(batch["event_id_null"][index]):
        reasons.append("null_event_id")
    if bool(batch["label_null"][index]):
        reasons.append("null_label")
    elif int(batch["labels"][index]) not in LABEL_ORDER:
        reasons.append("invalid_label")
    return reasons


def _rejection(source_name: str, row_index: int, batch, index: int, reason: str) -> dict[str, object]:
    event_id = None if bool(batch["event_id_null"][index]) else int(batch["event_ids"][index])
    return {
        "source_file": source_name,
        "row_index": row_index,
        "event_id": event_id,
        "reason": reason,
    }


def _accepted(batch, signal_counts, source_name: str, row_index: int, index: int) -> dict[str, object]:
    row: dict[str, object] = {
        "source_file": source_name,
        "row_index": row_index,
        "event_id": int(batch["event_ids"][index]),
        "split_name": split_name_from_file(source_name),
        "label": int(batch["labels"][index]),
        "vehiclesize": batch["vehiclesize"][index],
        "hd_flag": int(bool(batch["gyro_is_hd"][index])),
        "gsensor_length": int(signal_counts["gsensor_length"][index]),
        "gyro_length": int(signal_counts["gyro_length"][index]),
        "gps_length": int(signal_counts["gps_length"][index]),
        "gsensor_nonfinite": int(signal_counts["gsensor_nonfinite"][index]),
        "gyro_nonfinite": int(signal_counts["gyro_nonfinite"][index]),
        "gps_nonfinite": int(signal_counts["gps_nonfinite"][index]),
    }
    for name, value in zip(FEATURE_NAMES, batch["features"][index]):
        row[name] = _sql_float(float(value))
    return row


def split_name_from_file(file_name: str) -> str:
    """Use the same filename rule as the modeling code."""

    return split_name(Path(file_name))


def _signal_counts(batch) -> dict[str, np.ndarray]:
    _, gsensor_nonfinite = imu_null_and_nonfinite(batch["accel"])
    _, gyro_nonfinite = imu_null_and_nonfinite(batch["gyro"])
    gps_nonfinite = sum_by_group(batch["gps"].nonfinite.astype(np.int64), batch["gps"].offsets)
    return {
        "gsensor_length": np.diff(batch["accel"].offsets),
        "gyro_length": np.diff(batch["gyro"].offsets),
        "gps_length": np.diff(batch["gps"].offsets),
        "gsensor_nonfinite": gsensor_nonfinite,
        "gyro_nonfinite": gyro_nonfinite,
        "gps_nonfinite": gps_nonfinite,
    }


def _sql_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return value
