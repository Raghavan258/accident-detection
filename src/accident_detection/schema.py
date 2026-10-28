"""Parquet schema checks for the VZCrash release layout."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from accident_detection.constants import (
    IMU_AXES,
    KNOWN_SPLITS,
    REQUIRED_COLUMNS,
    UNKNOWN_SPLIT,
)


def split_name(path: Path) -> str:
    """Return train, validation, test, or unknown from the file name prefix."""

    prefix = path.name.split("-", 1)[0]
    if prefix in KNOWN_SPLITS:
        return prefix
    return UNKNOWN_SPLIT


def schema_problems(schema: pa.Schema) -> list[str]:
    """List missing columns, unexpected columns, and incompatible types."""

    problems = [
        f"missing column: {name}"
        for name in REQUIRED_COLUMNS
        if name not in schema.names
    ]
    problems.extend(
        f"unexpected column: {name}"
        for name in schema.names
        if name not in REQUIRED_COLUMNS
    )
    if any(item.startswith("missing column") for item in problems):
        return problems
    problems.extend(_type_problems(schema))
    return problems


def signals_readable(problems: tuple[str, ...] | list[str]) -> bool:
    """Return whether signal arrays can be decoded from this schema."""

    return not any(
        item.startswith("missing column") or item.startswith("type:")
        for item in problems
    )


def _type_problems(schema: pa.Schema) -> list[str]:
    checks = (
        ("event_id", pa.types.is_integer),
        ("label", pa.types.is_integer),
        ("vehiclesize", _is_string),
        ("gyro_is_hd", pa.types.is_boolean),
        ("gsensor", _is_imu_list),
        ("gyro", _is_imu_list),
        ("gps_speed", _is_gps_list),
    )
    return [
        f"type: {name} is {schema.field(name).type}"
        for name, check in checks
        if not check(schema.field(name).type)
    ]


def _is_string(data_type: pa.DataType) -> bool:
    return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)


def _is_imu_list(data_type: pa.DataType) -> bool:
    value_type = getattr(data_type, "value_type", None)
    return (
        pa.types.is_list(data_type)
        and pa.types.is_fixed_size_list(value_type)
        and value_type.list_size == IMU_AXES
        and pa.types.is_floating(value_type.value_type)
    )


def _is_gps_list(data_type: pa.DataType) -> bool:
    value_type = getattr(data_type, "value_type", None)
    return pa.types.is_list(data_type) and pa.types.is_floating(value_type)
