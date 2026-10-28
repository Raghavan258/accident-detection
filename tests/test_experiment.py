"""The experiment fits training shards and does not consume test rows."""

from pathlib import Path

import numpy as np

from accident_detection.experiment import run_experiment
from accident_detection.features import FEATURE_NAMES
from accident_detection.splits import split_limitations
from tests.parquet_factory import nominal_event, write_events, xyz


def _event(event_id: int, label: int, accel: float, gyro: float) -> dict:
    event = nominal_event(event_id, label, samples=2, gps=2)
    event["gsensor"] = [[accel, 0.0, 1.0], [0.0, 0.0, 1.0]]
    event["gyro"] = xyz(2, gyro)
    event["gps_speed"] = [10.0, 0.0]
    return event


def test_partial_download_is_not_described_as_the_full_split() -> None:
    notes = " ".join(split_limitations(1, 0, 2, training_crashes=245))
    assert "72%" in notes
    assert "train 1" in notes
    assert "vehicle id" in notes
    assert "245" in notes
    assert "not the ITSC benchmark" in notes


def test_experiment_fits_train_and_leaves_validation_unused(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    train_path = data / "train-00000-of-00011.parquet"
    test_path = data / "test-00000-of-00002.parquet"
    validation_path = data / "validation-00000-of-00003.parquet"
    write_events(data / "train-00000-of-00011.parquet", [_event(1, 0, 3.0, 0.0), _event(2, 2, 0.1, 2.0)])
    write_events(data / "test-00000-of-00002.parquet", [_event(3, 2, 0.2, 1000.0)])
    write_events(data / "validation-00000-of-00003.parquet", [_event(4, 0, 4.0, 9000.0)])
    before_train = train_path.read_bytes()
    before_test = test_path.read_bytes()

    result = run_experiment(data, tmp_path / "out", "pytest")

    assert train_path.read_bytes() == before_train
    assert test_path.read_bytes() == before_test
    assert validation_path.is_file()
    assert result["rows_removed"] == 0
    assert result["unused_validation_files"] == ["validation-00000-of-00003.parquet"]
    column = FEATURE_NAMES.index("peak_abs_gyro_deg_s")
    assert result["feature_model"]["scaler_mean"][column] == 1.0
    assert result["test_pooled"]["rows"] == 1
    assert result["train_diagnostics"]["rows"] == 2
    assert result["peak_baseline"]["threshold_g"] == 3.0
    assert Path(result["report_path"]).is_file()
    assert "not a claim of final performance" in Path(result["report_path"]).read_text(encoding="utf-8")
