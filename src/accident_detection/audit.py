"""Scan VZCrash Parquet shards and flag records without rewriting them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from accident_detection.columns import (
    bool_counts,
    count_labels,
    decode_gps,
    decode_imu,
    finite_extrema,
    grouped_finite_max,
    histogram,
    imu_null_and_nonfinite,
    integer_column,
    list_lengths,
    sample_magnitudes,
    string_counts,
    sum_by_group,
    label_name,
)
from accident_detection.constants import (
    LABEL_ORDER,
    OFFICIAL_SHARD_COUNTS,
    REQUIRED_COLUMNS,
    SIGNAL_COLUMNS,
)
from accident_detection.models import (
    DEFAULT_CONFIG,
    AuditConfig,
    AuditReport,
    ChannelStats,
    DuplicateId,
    FileScan,
    PlotExample,
)
from accident_detection.schema import schema_problems, signals_readable, split_name
from accident_detection.signals import assess_record_stats, prepare_event
from accident_detection.stats import build_descriptive_stats


def audit_directory(data_dir: Path | str, config: AuditConfig | None = None) -> AuditReport:
    """Audit every Parquet file in a directory. The directory is only read."""

    active = DEFAULT_CONFIG if config is None else config
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {root}")
    paths = sorted(path for path in root.glob("*.parquet") if path.is_file())
    scans = [scan_file(path, active) for path in paths]
    duplicates = find_duplicates(scans)
    append_duplicate_flags(scans, duplicates)
    return AuditReport(
        data_dir=str(root),
        files=tuple(scans),
        coverage_notes=coverage_notes(paths),
        duplicates=duplicates,
        examples=_merge_examples(scans),
        config=active,
        descriptive_stats=build_descriptive_stats(scans),
    )


def scan_file(path: Path, config: AuditConfig | None = None) -> FileScan:
    """Read one Parquet file and return its audit record."""

    active = DEFAULT_CONFIG if config is None else config
    try:
        metadata = pq.read_metadata(path)
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        return _unreadable(path, str(exc))
    return _scan_table(path, table, metadata.num_row_groups, metadata.num_rows, active)


def find_duplicates(scans: list[FileScan] | tuple[FileScan, ...]) -> tuple[DuplicateId, ...]:
    """Find event IDs that occur more than once across the scanned files."""

    locations: dict[int, list[str]] = {}
    for scan in scans:
        _collect_ids(scan, locations)
    duplicates = [
        DuplicateId(event_id, tuple(places))
        for event_id, places in sorted(locations.items())
        if len(places) > 1
    ]
    return tuple(duplicates)


def append_duplicate_flags(
    scans: list[FileScan] | tuple[FileScan, ...],
    duplicates: tuple[DuplicateId, ...],
) -> None:
    """Add duplicate flags onto the existing anomaly lists."""

    by_id = {item.event_id: item for item in duplicates}
    for scan in scans:
        _flag_scan_duplicates(scan, by_id)


def coverage_notes(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]:
    """Compare local shard counts with the official Hugging Face release."""

    names = [path.name for path in paths]
    notes = []
    for split, expected in OFFICIAL_SHARD_COUNTS.items():
        found = sum(name.startswith(f"{split}-") for name in names)
        if found != expected:
            notes.append(
                f"{split}: found {found} of {expected} official parquet shards in this directory"
            )
    return tuple(notes)


def _scan_table(
    path: Path,
    table: pa.Table,
    row_groups: int,
    metadata_rows: int,
    config: AuditConfig,
) -> FileScan:
    problems = list(schema_problems(table.schema))
    if table.num_rows != metadata_rows:
        problems.append("row count metadata does not match the decoded table")
    meta = _metadata_arrays(table, problems)
    signal = _signal_arrays(table, problems)
    examples = _select_examples(table, path.name, split_name(path), meta, signal, config)
    anomalies = _anomalies(path.name, split_name(path), meta, signal, config)
    return _assemble(path, table, row_groups, tuple(problems), meta, signal, examples, anomalies)


def _signal_arrays(table: pa.Table, problems: list[str]) -> dict[str, object]:
    if not signals_readable(problems):
        return _empty_signals(table.num_rows, "schema does not allow signal decoding")
    try:
        return _decode_signals(table)
    except (ValueError, pa.ArrowException) as exc:
        failed = _empty_signals(table.num_rows, str(exc))
        failed["read_error"] = str(exc)
        return failed


def _decode_signals(table: pa.Table) -> dict[str, object]:
    payload = _empty_signals(table.num_rows, None)
    payload["signals_scanned"] = True
    payload["read_error"] = None
    for name in ("gsensor", "gyro"):
        _store_imu(payload, name, decode_imu(table[name]))
    _store_gps(payload, decode_gps(table["gps_speed"]))
    for name in SIGNAL_COLUMNS:
        lengths, is_null = list_lengths(table[name])
        payload["lengths"][name] = lengths
        payload["list_null"][name] = is_null
        payload["length_counts"][name] = histogram(lengths, is_null)
    return payload


def _store_imu(payload: dict[str, object], name: str, decoded) -> None:
    nulls, nonfinite = imu_null_and_nonfinite(decoded)
    payload["null_elements"][name] = nulls
    payload["nonfinite"][name] = nonfinite
    payload["null_element_totals"][name] = int(nulls.sum())
    payload["nonfinite_totals"][name] = int(nonfinite.sum())
    payload["extrema"][name] = finite_extrema(decoded.matrix.ravel())
    magnitude, horizontal = sample_magnitudes(decoded.matrix)
    if name == "gsensor":
        payload["peak_acceleration_g"] = grouped_finite_max(magnitude, decoded.offsets)
        payload["peak_horizontal_g"] = grouped_finite_max(horizontal, decoded.offsets)


def _store_gps(payload: dict[str, object], decoded) -> None:
    nulls = sum_by_group(decoded.sample_null.astype(np.int64), decoded.offsets)
    nonfinite = sum_by_group(decoded.nonfinite.astype(np.int64), decoded.offsets)
    payload["null_elements"]["gps_speed"] = nulls
    payload["nonfinite"]["gps_speed"] = nonfinite
    payload["null_element_totals"]["gps_speed"] = int(nulls.sum())
    payload["nonfinite_totals"]["gps_speed"] = int(nonfinite.sum())
    payload["extrema"]["gps_speed"] = finite_extrema(decoded.values)


def _anomalies(file_name, split, meta, signal, config: AuditConfig) -> list[dict[str, object]]:
    if not signal["signals_scanned"]:
        return []
    rows: list[dict[str, object]] = []
    for index in range(meta["num_rows"]):
        rows.extend(_row_anomalies(file_name, split, meta, signal, config, index))
    return rows


def _row_anomalies(file_name, split, meta, signal, config, index: int) -> list[dict[str, object]]:
    flags = assess_record_stats(
        label=_optional_int(meta["labels"], meta["label_null"], index),
        vehiclesize=meta["vehiclesize"][index],
        event_id=_optional_int(meta["event_ids"], meta["event_id_null"], index),
        channels=_channels_at(signal, index),
        config=config,
    )
    label = _optional_int(meta["labels"], meta["label_null"], index)
    event_id = _optional_int(meta["event_ids"], meta["event_id_null"], index)
    return [
        _anomaly(file_name, split, index, event_id, label, flag.code, flag.detail)
        for flag in flags
    ]


def _channels_at(signal: dict[str, object], index: int) -> dict[str, ChannelStats]:
    channels = {}
    for name in SIGNAL_COLUMNS:
        missing = bool(signal["list_null"][name][index])
        channels[name] = ChannelStats(
            is_null=missing,
            length=0 if missing else int(signal["lengths"][name][index]),
            nonfinite_count=int(signal["nonfinite"][name][index]),
            null_element_count=int(signal["null_elements"][name][index]),
        )
    return channels


def _select_examples(table, file_name, split, meta, signal, config) -> dict[int, PlotExample]:
    if split != "train" or not signal["signals_scanned"]:
        return {}
    selected: dict[int, PlotExample] = {}
    for label_id in LABEL_ORDER:
        chosen = _example_index(meta, signal, config, label_id)
        if chosen is None:
            continue
        selected[label_id] = _example_at(table, file_name, label_id, chosen, config)
    return selected


def _example_index(meta, signal, config: AuditConfig, label_id: int) -> tuple[int, bool] | None:
    match = (~meta["label_null"]) & (meta["labels"] == label_id)
    nominal = match & _nominal_mask(signal, config)
    for mask, is_nominal in ((nominal, True), (match, False)):
        hits = np.flatnonzero(mask)
        if hits.size:
            return int(hits[0]), is_nominal
    return None


def _nominal_mask(signal, config: AuditConfig) -> np.ndarray:
    gps_ok = _length_ok(signal, "gps_speed", config.nominal_gps_samples)
    accel_ok = _length_ok(signal, "gsensor", config.nominal_imu_samples)
    gyro_ok = _length_ok(signal, "gyro", config.nominal_imu_samples)
    return gps_ok & accel_ok & gyro_ok


def _length_ok(signal, name: str, nominal: int) -> np.ndarray:
    return ~signal["list_null"][name] & (signal["lengths"][name] == nominal)


def _example_at(table, file_name, label_id: int, chosen: tuple[int, bool], config) -> PlotExample:
    index, nominal_timing = chosen
    row = table.slice(index, 1).to_pylist()[0]
    prepared = prepare_event(
        row["gsensor"],
        row["gyro"],
        row["gps_speed"],
        label=row["label"],
        vehiclesize=row["vehiclesize"],
        event_id=row["event_id"],
        config=config,
    )
    return PlotExample(
        label=label_id,
        event_id=int(row["event_id"]),
        source_file=file_name,
        row_index=index,
        accelerometer_g=prepared.accelerometer_g,
        nominal_timing=nominal_timing,
    )


def _identity_readable(problems: list[str]) -> bool:
    names = ("event_id", "label", "vehiclesize", "gyro_is_hd")
    for item in problems:
        for name in names:
            if item == f"missing column: {name}" or item.startswith(f"type: {name} "):
                return False
    return True


def _outer_nulls(table: pa.Table) -> dict[str, int]:
    return {
        name: table[name].null_count
        for name in REQUIRED_COLUMNS
        if name in table.column_names
    }


def _blank_meta(num_rows: int, table: pa.Table) -> dict[str, object]:
    return {
        "num_rows": num_rows,
        "event_ids": np.zeros(num_rows, dtype=np.int64),
        "event_id_null": np.ones(num_rows, dtype=bool),
        "labels": np.zeros(num_rows, dtype=np.int64),
        "label_null": np.ones(num_rows, dtype=bool),
        "vehiclesize": [None] * num_rows,
        "label_counts": {label_id: 0 for label_id in LABEL_ORDER},
        "null_counts": _outer_nulls(table),
        "vehiclesize_counts": {},
        "gyro_hd_counts": {},
    }


def _metadata_arrays(table: pa.Table, problems: list[str]) -> dict[str, object]:
    if not _identity_readable(problems):
        return _blank_meta(table.num_rows, table)
    event_ids, event_null = integer_column(table["event_id"])
    labels, label_null = integer_column(table["label"])
    return {
        "num_rows": table.num_rows,
        "event_ids": event_ids,
        "event_id_null": event_null,
        "labels": labels,
        "label_null": label_null,
        "vehiclesize": table["vehiclesize"].to_pylist(),
        "label_counts": count_labels(labels, label_null),
        "null_counts": _outer_nulls(table),
        "vehiclesize_counts": string_counts(table["vehiclesize"].to_pylist()),
        "gyro_hd_counts": bool_counts(table["gyro_is_hd"].to_pylist()),
    }


def _assemble(path, table, row_groups, problems, meta, signal, examples, anomalies) -> FileScan:
    return FileScan(
        path=str(path),
        file_name=path.name,
        split=split_name(path),
        num_rows=table.num_rows,
        num_row_groups=row_groups,
        file_size_bytes=path.stat().st_size,
        schema_problems=problems,
        read_error=signal["read_error"],
        signals_scanned=bool(signal["signals_scanned"]),
        label_counts=meta["label_counts"],
        null_counts=meta["null_counts"],
        null_element_totals=signal["null_element_totals"],
        nonfinite_totals=signal["nonfinite_totals"],
        length_counts=signal["length_counts"],
        extrema=signal["extrema"],
        vehiclesize_counts=meta["vehiclesize_counts"],
        gyro_hd_counts=meta["gyro_hd_counts"],
        anomalies=anomalies,
        event_ids=meta["event_ids"],
        event_id_null=meta["event_id_null"],
        labels=meta["labels"],
        label_null=meta["label_null"],
        peak_acceleration_g=signal["peak_acceleration_g"],
        peak_horizontal_g=signal["peak_horizontal_g"],
        examples=examples,
    )


def _empty_signals(num_rows: int, read_error: str | None) -> dict[str, object]:
    return {
        "signals_scanned": False,
        "read_error": read_error,
        "lengths": {},
        "list_null": {},
        "null_elements": {},
        "nonfinite": {},
        "null_element_totals": {},
        "nonfinite_totals": {},
        "length_counts": {},
        "extrema": {},
        "peak_acceleration_g": np.full(num_rows, np.nan, dtype=np.float64),
        "peak_horizontal_g": np.full(num_rows, np.nan, dtype=np.float64),
    }


def _unreadable(path: Path, message: str) -> FileScan:
    empty_int = np.zeros(0, dtype=np.int64)
    empty_bool = np.zeros(0, dtype=bool)
    empty_float = np.zeros(0, dtype=np.float64)
    return FileScan(
        path=str(path),
        file_name=path.name,
        split=split_name(path),
        num_rows=0,
        num_row_groups=0,
        file_size_bytes=path.stat().st_size if path.exists() else 0,
        schema_problems=(),
        read_error=message,
        signals_scanned=False,
        label_counts={label_id: 0 for label_id in LABEL_ORDER},
        null_counts={},
        null_element_totals={},
        nonfinite_totals={},
        length_counts={},
        extrema={},
        vehiclesize_counts={},
        gyro_hd_counts={},
        anomalies=[],
        event_ids=empty_int,
        event_id_null=empty_bool,
        labels=empty_int,
        label_null=empty_bool,
        peak_acceleration_g=empty_float,
        peak_horizontal_g=empty_float.copy(),
        examples={},
    )


def _collect_ids(scan: FileScan, locations: dict[int, list[str]]) -> None:
    for index, event_id in enumerate(scan.event_ids.tolist()):
        if scan.event_id_null[index]:
            continue
        locations.setdefault(int(event_id), []).append(f"{scan.file_name}:{index}")


def _flag_scan_duplicates(scan: FileScan, by_id: dict[int, DuplicateId]) -> None:
    for index, event_id in enumerate(scan.event_ids.tolist()):
        if scan.event_id_null[index] or int(event_id) not in by_id:
            continue
        match = by_id[int(event_id)]
        label = _optional_int(scan.labels, scan.label_null, index)
        detail = "locations=" + ",".join(match.locations)
        scan.anomalies.append(_anomaly(scan.file_name, scan.split, index, int(event_id), label, "duplicate_event_id", detail))


def _merge_examples(scans: list[FileScan]) -> dict[int, PlotExample]:
    merged: dict[int, PlotExample] = {}
    for scan in scans:
        for label_id, example in scan.examples.items():
            merged.setdefault(label_id, example)
    return merged


def _optional_int(values: np.ndarray, is_null: np.ndarray, index: int) -> int | None:
    if bool(is_null[index]):
        return None
    return int(values[index])


def _anomaly(file_name, split, index, event_id, label, code, detail) -> dict[str, object]:
    return {
        "source_file": file_name,
        "split": split,
        "row_index": index,
        "event_id": "" if event_id is None else int(event_id),
        "label": "" if label is None else int(label),
        "label_name": label_name(label),
        "code": code,
        "detail": detail,
    }
