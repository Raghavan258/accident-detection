"""Schema and sampling-spec checks."""

from pathlib import Path

import pytest

import pyarrow as pa
import pyarrow.parquet as pq

from accident_detection.constants import (
    GPS_SAMPLE_RATE_HZ,
    IMU_SAMPLE_RATE_HZ,
    NOMINAL_GPS_SAMPLES,
    NOMINAL_IMU_SAMPLES,
    WINDOW_SECONDS,
)
from accident_detection.schema import schema_problems, signals_readable, split_name

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "VZCrash" / "data"
EXPECTED_ROWS = {
    "train-00000-of-00011.parquet": 12542,
    "test-00000-of-00002.parquet": 12087,
    "test-00001-of-00002.parquet": 12087,
}


def test_nominal_lengths_follow_the_published_window() -> None:
    assert NOMINAL_IMU_SAMPLES == IMU_SAMPLE_RATE_HZ * WINDOW_SECONDS
    assert NOMINAL_GPS_SAMPLES == GPS_SAMPLE_RATE_HZ * WINDOW_SECONDS
    assert NOMINAL_IMU_SAMPLES == 1600
    assert NOMINAL_GPS_SAMPLES == 16


def test_split_name_uses_the_file_prefix() -> None:
    assert split_name(Path("train-00000-of-00011.parquet")) == "train"
    assert split_name(Path("validation-00002-of-00003.parquet")) == "validation"
    assert split_name(Path("test-00001-of-00002.parquet")) == "test"
    assert split_name(Path("notes.parquet")) == "unknown"


def test_missing_signal_column_is_not_readable() -> None:
    schema = pa.schema(
        [
            ("event_id", pa.int64()),
            ("label", pa.int64()),
            ("vehiclesize", pa.string()),
            ("gyro_is_hd", pa.bool_()),
            ("gyro", pa.list_(pa.list_(pa.float32(), 3))),
            ("gps_speed", pa.list_(pa.float32())),
        ]
    )
    problems = schema_problems(schema)
    assert "missing column: gsensor" in problems
    assert not signals_readable(problems)


def test_flat_gsensor_type_is_rejected() -> None:
    schema = pa.schema(
        [
            ("event_id", pa.int64()),
            ("label", pa.int64()),
            ("vehiclesize", pa.string()),
            ("gyro_is_hd", pa.bool_()),
            ("gsensor", pa.list_(pa.float32())),
            ("gyro", pa.list_(pa.list_(pa.float32(), 3))),
            ("gps_speed", pa.list_(pa.float32())),
        ]
    )
    problems = schema_problems(schema)
    assert any(item.startswith("type: gsensor") for item in problems)
    assert not signals_readable(problems)


def test_downloaded_shards_match_the_expected_physical_schema() -> None:
    if not DATA_DIR.is_dir():
        pytest.skip("VZCrash shards are not downloaded")
    for name, rows in EXPECTED_ROWS.items():
        path = DATA_DIR / name
        assert path.is_file(), name
        metadata = pq.read_metadata(path)
        assert metadata.num_rows == rows
        assert schema_problems(metadata.schema.to_arrow_schema()) == []
