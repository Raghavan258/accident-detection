"""ETL contracts on synthetic Parquet. The full VZCrash download is not required."""

import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from accident_detection.etl.load import commit_run, connect, init_db
from accident_detection.etl.ml_input import load_splits
from accident_detection.etl.pipeline import EtlConfig, run_etl
from accident_detection.etl.transform import records_from_table
from tests.parquet_factory import nominal_event, write_events


def test_extracts_a_valid_shard(tmp_path: Path) -> None:
    summary = _run(tmp_path, [_event(1, 0), _event(2, 2)])
    assert summary["files_processed"] == 1
    assert summary["accepted_rows"] == 2
    assert summary["rejected_rows"] == 0
    assert summary["rows_removed_without_reason"] == 0


def test_missing_input_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_etl(_config(tmp_path, tmp_path / "missing"))


def test_malformed_file_is_failed_not_loaded(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train-00000-of-00011.parquet").write_bytes(b"not parquet")
    summary = run_etl(_config(tmp_path, data))
    assert summary["files_failed"] == 1
    assert summary["accepted_rows"] == 0


def test_missing_required_column_fails_the_file(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    table = pa.table({"event_id": pa.array([1], type=pa.int64())})
    pq.write_table(table, data / "train-00000-of-00011.parquet")
    summary = run_etl(_config(tmp_path, data))
    assert summary["files_failed"] == 1
    assert summary["accepted_rows"] == 0


def test_string_label_is_a_type_failure(tmp_path: Path) -> None:
    data = _write_raw(_string_label_table(), tmp_path)
    summary = run_etl(_config(tmp_path, data))
    assert summary["files_failed"] == 1
    assert summary["accepted_rows"] == 0


def test_missing_values_are_counted_without_dropping_the_row(tmp_path: Path) -> None:
    event = _event(1, 2)
    event["gps_speed"] = [float("nan"), float("nan")]
    summary = _run(tmp_path, [event])
    assert summary["accepted_rows"] == 1
    assert summary["rejected_rows"] == 0
    assert _metric(summary, "missing_gps_start_kmh") == 1


def test_exact_duplicate_event_id_is_kept_and_counted(tmp_path: Path) -> None:
    summary = _run(tmp_path, [_event(7, 0), _event(7, 2)])
    assert summary["accepted_rows"] == 2
    assert _metric(summary, "within_file_duplicate_event_ids") == 1


def test_cross_file_duplicate_event_id_is_counted(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_events(data / "train-00000-of-00011.parquet", [_event(4, 0)])
    write_events(data / "test-00000-of-00002.parquet", [_event(4, 2)])
    summary = run_etl(_config(tmp_path, data))
    assert _metric(summary, "cross_file_duplicate_event_ids") == 1
    assert summary["accepted_rows"] == 2


def test_invalid_label_is_rejected_with_a_reason(tmp_path: Path) -> None:
    summary = _run(tmp_path, [_event(1, 9), _event(2, 0)])
    assert summary["accepted_rows"] == 1
    assert summary["rejected_rows"] == 1
    assert _metric(summary, "rejection_invalid_label") == 1


def test_transformation_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "train-00000-of-00011.parquet"
    write_events(path, [_event(1, 0), _event(2, 1)])
    table = pq.read_table(path)
    assert records_from_table(table, path.name, 0) == records_from_table(table, path.name, 0)


def test_loading_the_same_run_twice_does_not_duplicate(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_events(data / "train-00000-of-00011.parquet", [_event(1, 0), _event(2, 2)])
    config = EtlConfig(data, tmp_path / "events.db", tmp_path / "out", run_id="run-1")
    run_etl(config)
    before = _event_count(config.db_path, "run-1")
    with pytest.raises(sqlite3.IntegrityError):
        run_etl(config)
    assert _event_count(config.db_path, "run-1") == before == 2


def test_failed_insert_rolls_back_the_run(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    source = data / "train-00000-of-00011.parquet"
    write_events(source, [_event(1, 0)])
    table = pq.read_table(source)
    accepted, _rejected = records_from_table(table, source.name, 0)
    accepted[0]["label"] = 9
    db_path = tmp_path / "events.db"
    init_db(db_path)
    shard = _shard(source.name, accepted, [])
    with pytest.raises(sqlite3.IntegrityError):
        commit_run(db_path, _run_row("run-x"), [shard])
    connection = connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM sensor_events").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 0
    finally:
        connection.close()


def test_rejected_rows_match_the_report(tmp_path: Path) -> None:
    summary = _run(tmp_path, [_event(1, 9)])
    assert summary["rejected_rows"] == 1
    assert (tmp_path / "out" / "etl_quality.json").is_file()
    assert (tmp_path / "out" / "quality_metrics.csv").is_file()
    assert (tmp_path / "out" / "class_counts.png").is_file()


def test_end_to_end_fixture_preserves_split_names(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_events(data / "train-00000-of-00011.parquet", [_event(1, 0), _event(2, 2)])
    write_events(data / "validation-00000-of-00003.parquet", [_event(3, 0), _event(4, 2)])
    write_events(data / "test-00000-of-00002.parquet", [_event(5, 2)])
    summary = run_etl(_config(tmp_path, data))
    splits = load_splits(tmp_path / "events.db", str(summary["run_id"]))
    assert set(splits["train"]["event_ids"]) == {1, 2}
    assert set(splits["validation"]["event_ids"]) == {3, 4}
    assert set(splits["test"]["event_ids"]) == {5}
    assert set(splits["train"]["event_ids"]).isdisjoint(splits["test"]["event_ids"])


def test_evaluation_uses_validation_and_scores_test_once(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_events(data / "train-00000-of-00011.parquet", [_event(1, 0), _event(2, 2)])
    write_events(data / "validation-00000-of-00003.parquet", [_event(3, 0), _event(4, 2)])
    write_events(data / "test-00000-of-00002.parquet", [_event(5, 2)])
    summary = run_etl(_config(tmp_path, data))
    from accident_detection.etl.evaluate import evaluate_loaded_run

    result = evaluate_loaded_run(tmp_path / "events.db", tmp_path / "out", str(summary["run_id"]))
    assert result["status"] == "frozen"
    assert result["test"]["status"] == "scored_once"
    assert result["historical_phase4_used"] is False


def test_dry_run_does_not_create_a_database(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_events(data / "train-00000-of-00011.parquet", [_event(1, 2)])
    db_path = tmp_path / "events.db"
    summary = run_etl(EtlConfig(data, db_path, tmp_path / "out", dry_run=True))
    assert summary["database_status"] == "not_written"
    assert not db_path.exists()


def _run(tmp_path: Path, events: list[dict]) -> dict:
    data = tmp_path / "data"
    data.mkdir()
    write_events(data / "train-00000-of-00011.parquet", events)
    return run_etl(_config(tmp_path, data))


def _config(tmp_path: Path, data: Path) -> EtlConfig:
    return EtlConfig(data, tmp_path / "events.db", tmp_path / "out")


def _event(event_id: int, label: int) -> dict:
    return nominal_event(event_id, label, samples=2, gps=2)


def _metric(summary: dict, name: str) -> int:
    for item in summary["metrics"]:
        if item["scope"] == "all" and item["metric"] == name:
            return int(item["value"])
    raise KeyError(name)


def _event_count(db_path: Path, run_id: str) -> int:
    connection = connect(db_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM sensor_events WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        connection.close()
    return int(row[0])


def _write_raw(table: pa.Table, tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    pq.write_table(table, data / "train-00000-of-00011.parquet")
    return data


def _string_label_table() -> pa.Table:
    event = _event(1, 0)
    return pa.table(
        {
            "event_id": pa.array([1], type=pa.int64()),
            "label": pa.array(["crash"], type=pa.string()),
            "vehiclesize": pa.array([event["vehiclesize"]], type=pa.string()),
            "gyro_is_hd": pa.array([False], type=pa.bool_()),
            "gsensor": pa.array([event["gsensor"]], type=pa.list_(pa.list_(pa.float32(), 3))),
            "gyro": pa.array([event["gyro"]], type=pa.list_(pa.list_(pa.float32(), 3))),
            "gps_speed": pa.array([event["gps_speed"]], type=pa.list_(pa.float32())),
        }
    )


def _shard(file_name: str, accepted: list[dict], rejected: list[dict]) -> dict:
    return {
        "file_name": file_name,
        "split_name": "train",
        "byte_size": 1,
        "input_rows": len(accepted) + len(rejected),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "status": "processed",
        "error_message": None,
        "accepted": accepted,
        "rejected": rejected,
    }


def _run_row(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "status": "loaded",
        "input_dir": "data",
        "error_message": None,
        "metrics": [],
    }
