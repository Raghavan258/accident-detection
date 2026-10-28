"""Phase 4 selects on validation and does not refit from test labels."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from accident_detection.baselines import fit_logistic
from accident_detection.coverage import shard_inventory
from accident_detection.features import (
    ACCEL_FEATURE_NAMES,
    DEFENSIBLE_FEATURE_NAMES,
    EXCLUDED_FEATURE_REASONS,
    FEATURE_NAMES,
)
from accident_detection.metrics import classification_metrics
from accident_detection.phase4 import present_metrics, run_phase4
from accident_detection.protocol import TEST_SHARDS, TRAIN_SHARDS, VALIDATION_SHARDS
from accident_detection.selection import validation_choice
from tests.parquet_factory import nominal_event, write_events, xyz


def test_inventory_names_missing_official_shards(tmp_path: Path) -> None:
    (tmp_path / "train-00000-of-00011.parquet").write_bytes(b"")
    (tmp_path / "test-00000-of-00002.parquet").write_bytes(b"")
    inventory = shard_inventory(tmp_path)
    assert inventory["ready_to_select"] is False
    assert "train-00001-of-00011.parquet" in inventory["missing_train"]
    assert "validation-00000-of-00003.parquet" in inventory["missing_validation"]
    assert "test-00001-of-00002.parquet" in inventory["missing_test"]
    assert "train-00000-of-00011.parquet" not in inventory["download_plan"]["command"]


def test_defensible_features_leave_out_encoding_columns() -> None:
    assert set(DEFENSIBLE_FEATURE_NAMES).isdisjoint(EXCLUDED_FEATURE_REASONS)
    assert set(ACCEL_FEATURE_NAMES).issubset(DEFENSIBLE_FEATURE_NAMES)
    assert set(DEFENSIBLE_FEATURE_NAMES) | set(EXCLUDED_FEATURE_REASONS) == set(FEATURE_NAMES)
    assert "gps_samples" in EXCLUDED_FEATURE_REASONS
    assert "gyro_is_hd" in EXCLUDED_FEATURE_REASONS


def test_validation_choice_has_no_test_argument() -> None:
    assert "test" not in inspect.signature(validation_choice).parameters
    choice = validation_choice(
        {
            "peak_baseline": np.array([0.0, 1.0, 2.0, 3.0]),
            "accel_logistic": np.array([0.0, 0.1, 0.2, 0.3]),
        },
        np.array([0, 0, 1, 1]),
    )
    assert choice["selected_model"] == "accel_logistic"
    assert choice["threshold"] == 0.2
    assert choice["status"] == "frozen"


def test_scaler_uses_training_rows_only() -> None:
    model = fit_logistic(np.array([[0.0], [1.0], [2.0]]), np.array([0, 1, 0]))
    assert model.named_steps["scaler"].mean_[0] == 1.0


def test_metrics_are_reproducible() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.7])
    predictions = np.array([0, 1, 0, 1])
    assert classification_metrics(labels, predictions, scores) == classification_metrics(labels, predictions, scores)


def test_zero_crash_slice_does_not_invent_precision() -> None:
    view = present_metrics(np.array([0, 0]), np.array([0.2, 0.8]), 0.5, "probe")
    assert view["positive_class_metrics"] == "undefined"
    assert view["precision_crash"] is None
    assert view["average_precision_crash"] is None
    assert view["false_positives"] == 1
    assert view["rows"] == 2


def test_missing_validation_blocks_before_test_scoring(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    phase2 = tmp_path / "phase2_report.md"
    phase2.write_text("phase2 stays", encoding="utf-8")
    (data / "train-00000-of-00011.parquet").write_bytes(b"not-read")
    result = run_phase4(data, tmp_path / "phase4", "pytest")
    assert result["decision"]["status"] == "blocked"
    assert result["test_evaluation"]["status"] == "not_run"
    assert phase2.read_text(encoding="utf-8") == "phase2 stays"
    assert result["rows_removed"] == 0


def test_phase4_selects_on_validation_and_keeps_rows(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    phase3 = tmp_path / "phase3_report.md"
    phase3.write_text("phase3 stays", encoding="utf-8")
    _write_official(data)
    train_bytes = (data / "train-00000-of-00011.parquet").read_bytes()
    test_bytes = (data / "test-00001-of-00002.parquet").read_bytes()

    result = run_phase4(data, tmp_path / "phase4", "pytest")

    assert train_bytes == (data / "train-00000-of-00011.parquet").read_bytes()
    assert test_bytes == (data / "test-00001-of-00002.parquet").read_bytes()
    assert phase3.read_text(encoding="utf-8") == "phase3 stays"
    assert result["rows_removed"] == 0
    assert result["decision"]["status"] == "frozen"
    assert result["test_evaluation"]["status"] == "scored_once"
    assert "test" not in result["decision"]
    config = (tmp_path / "phase4" / "experiment_config.json").read_text(encoding="utf-8")
    assert '"test_metrics_included": false' in config
    accel = result["candidates"]["accel_logistic"]
    assert accel["scaler_mean"][0] == pytest.approx(1.55)
    assert "gps_samples" not in accel["features"]
    assert "gyro_is_hd" not in result["candidates"]["defensible_logistic"]["coefficients"]
    shard = next(item for item in result["audits"] if item["file_name"] == "train-00000-of-00011.parquet")
    assert shard["matrix_rows"] == shard["rows"] == 2
    assert _gps_rows(shard, 17) == 1
    non_hd = next(item for item in shard["gyro"]["by_resolution"] if item["gyro_is_hd"] is False)
    assert non_hd["max_abs_deg_s"] == 32762.0
    zero_crash = result["test_evaluation"]["by_file"][1]
    assert zero_crash["file_name"] == "test-00001-of-00002.parquet"
    assert zero_crash["positive_class_metrics"] == "undefined"
    assert zero_crash["false_positives"] == 0


def _write_official(data: Path) -> None:
    for index, name in enumerate(TRAIN_SHARDS):
        write_events(data / name, [_row(index * 10 + 1, 0, 3.0, index == 0), _row(index * 10 + 2, 2, 0.1, False)])
    for index, name in enumerate(VALIDATION_SHARDS):
        write_events(data / name, [_row(1000 + index * 10 + 1, 0, 2.5, False), _row(1000 + index * 10 + 2, 2, 0.2, False)])
    write_events(data / TEST_SHARDS[0], [_row(2001, 0, 3.0, False), _row(2002, 2, 0.1, False)])
    write_events(data / TEST_SHARDS[1], [_row(3001, 2, 0.05, False), _row(3002, 2, 0.05, False)])


def _row(event_id: int, label: int, accel: float, long_gps: bool) -> dict:
    event = nominal_event(event_id, label, samples=2, gps=17 if long_gps else 2)
    event["gsensor"] = [[accel, 0.0, 1.0], [0.0, 0.0, 1.0]]
    event["gyro"] = xyz(2, 32762.0 if long_gps else 4.0)
    event["gps_speed"] = [10.0] * (17 if long_gps else 2)
    return event


def _gps_rows(audit: dict, length: int) -> int:
    match = next(item for item in audit["gps"] if item["length"] == length)
    return int(match["rows"])
