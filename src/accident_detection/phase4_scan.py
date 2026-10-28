"""One-pass shard audit. Records are counted and kept."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from accident_detection.columns import count_labels
from accident_detection.constants import REQUIRED_COLUMNS
from accident_detection.features import FEATURE_NAMES, PEAK_SCORE_FEATURE, table_matrix
from accident_detection.labels import binary_crash_labels
from accident_detection.quality import describe_gps_structure, summarize_gps, summarize_gyro, summarize_imu
from accident_detection.schema import schema_problems, split_name

RANGE_FEATURES: tuple[str, ...] = (
    "peak_horizontal_g",
    "gps_end_kmh",
    "gps_samples",
    "gyro_is_hd",
    "peak_abs_gyro_deg_s",
    "imu_samples",
)


def scan_shard(path: Path, *, keep_matrix: bool) -> dict[str, object]:
    """Read one official shard. The Parquet file is not rewritten."""

    schema = pq.read_schema(path)
    table = pq.read_table(path, columns=list(REQUIRED_COLUMNS))
    batch = table_matrix(table)
    labels = binary_crash_labels(batch["labels"], batch["label_null"])
    audit = _audit(path, schema, table, batch)
    part: dict[str, object] = {
        "file_name": path.name,
        "split": split_name(path),
        "audit": audit,
        "event_ids": batch["event_ids"],
        "y": labels,
    }
    if keep_matrix:
        part["features"] = batch["features"]
        part["peak"] = batch["features"][:, FEATURE_NAMES.index(PEAK_SCORE_FEATURE)]
    return part


def duplicate_report(parts: list[dict[str, object]]) -> dict[str, object]:
    """Find repeated event IDs inside and across the scanned shards."""

    within = {str(part["file_name"]): _within_file(part["event_ids"]) for part in parts}
    across = _across_files(parts)
    return {
        "within_file": {name: values for name, values in within.items() if values},
        "across_files": across,
        "vehicle_identifier": "absent",
        "vehicle_identifier_note": (
            "The schema has event_id, label, vehiclesize, gyro_is_hd, gsensor, gyro, and "
            "gps_speed. There is no vehicle id, so the paper's vehicle-disjoint split cannot "
            "be rebuilt. Official shard names are preserved instead."
        ),
    }


def _audit(path: Path, schema, table, batch) -> dict[str, object]:
    problems = schema_problems(schema)
    null_counts = {name: int(table[name].null_count) for name in REQUIRED_COLUMNS}
    return {
        "file_name": path.name,
        "split": split_name(path),
        "rows": int(table.num_rows),
        "matrix_rows": int(batch["features"].shape[0]),
        "rows_removed": 0,
        "schema_problems": problems,
        "column_types": {name: str(schema.field(name).type) for name in schema.names},
        "extra_columns": [name for name in schema.names if name not in REQUIRED_COLUMNS],
        "label_counts": {str(key): value for key, value in count_labels(batch["labels"], batch["label_null"]).items()},
        "null_counts": null_counts,
        "nonfinite_totals": _nonfinite(batch),
        "gps": summarize_gps(batch["gps"], batch["labels"], batch["gyro_is_hd"]),
        "gps_structure": describe_gps_structure(batch["gps"]),
        "imu": summarize_imu(batch["accel"], batch["gyro"], batch["labels"]),
        "gyro": summarize_gyro(batch["gyro"], batch["gyro_is_hd"], batch["labels"]),
        "feature_ranges": _ranges(batch["features"]),
        "label_nulls": int(np.sum(batch["label_null"])),
        "event_id_nulls": int(np.sum(batch["event_id_null"])),
    }


def _nonfinite(batch) -> dict[str, int]:
    return {
        "gsensor": int(batch["accel"].nonfinite.sum()),
        "gyro": int(batch["gyro"].nonfinite.sum()),
        "gps_speed": int(batch["gps"].nonfinite.sum()),
    }


def _ranges(features: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    ranges = {}
    for name in RANGE_FEATURES:
        column = features[:, FEATURE_NAMES.index(name)]
        finite = column[np.isfinite(column)]
        ranges[name] = _one_range(column, finite)
    return ranges


def _one_range(column: np.ndarray, finite: np.ndarray) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"nonfinite": int(np.sum(~np.isfinite(column)))}
    if finite.size == 0:
        summary.update({"min": None, "p50": None, "p95": None, "max": None})
        return summary
    summary.update(
        {
            "min": float(finite.min()),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(finite.max()),
        }
    )
    return summary


def _within_file(event_ids: np.ndarray) -> list[int]:
    if event_ids.size == 0:
        return []
    keys, counts = np.unique(event_ids, return_counts=True)
    return [int(key) for key, count in zip(keys, counts) if count > 1][:20]


def _across_files(parts: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: dict[int, str] = {}
    found: list[dict[str, object]] = []
    for part in parts:
        _collect_across(seen, found, str(part["file_name"]), part["event_ids"])
    return found[:50]


def _collect_across(seen: dict[int, str], found: list[dict[str, object]], file_name: str, event_ids) -> None:
    for event_id in np.asarray(event_ids).tolist():
        previous = seen.get(int(event_id))
        if previous is None:
            seen[int(event_id)] = file_name
        elif len(found) < 50:
            found.append({"event_id": int(event_id), "files": [previous, file_name]})
