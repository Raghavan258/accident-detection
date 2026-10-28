"""Phase 3 keeps test labels out of model selection."""

from pathlib import Path

import numpy as np

from accident_detection.baselines import fit_logistic
from accident_detection.diagnosis import z_score
from accident_detection.features import ACCEL_FEATURE_NAMES, FEATURE_NAMES
from accident_detection.phase3 import run_phase3
from accident_detection.protocol import download_plan, freeze_decision
from tests.parquet_factory import nominal_event, write_events, xyz


def test_acceleration_features_leave_out_resolution_and_gps() -> None:
    assert "peak_horizontal_g" in ACCEL_FEATURE_NAMES
    assert "gyro_is_hd" not in ACCEL_FEATURE_NAMES
    assert "gps_samples" not in ACCEL_FEATURE_NAMES
    assert "peak_abs_gyro_deg_s" not in ACCEL_FEATURE_NAMES
    assert set(ACCEL_FEATURE_NAMES).issubset(FEATURE_NAMES)


def test_one_sample_length_gap_is_many_training_deviations() -> None:
    assert z_score(17.0, 16.0, 0.05) == 20.0


def test_constant_resolution_column_does_not_get_a_coefficient() -> None:
    features = np.zeros((4, 2))
    features[:, 0] = [0.0, 1.0, 0.0, 1.0]
    model = fit_logistic(features, np.array([0, 1, 0, 1]))
    assert model.named_steps["clf"].coef_.ravel()[1] == 0.0


def test_validation_absence_blocks_selection() -> None:
    decision = freeze_decision({})
    assert decision["selected_model"] is None
    assert decision["status"] == "blocked"


def test_validation_average_precision_selects_without_a_test_argument() -> None:
    decision = freeze_decision(
        {
            "peak_baseline": {"average_precision_crash": 0.4},
            "accel_logistic": {"average_precision_crash": 0.8},
        }
    )
    assert decision["selected_model"] == "accel_logistic"
    assert "test" not in decision


def test_download_plan_keeps_existing_test_shards() -> None:
    present = {
        "train-00000-of-00011.parquet",
        "test-00000-of-00002.parquet",
        "test-00001-of-00002.parquet",
    }
    plan = download_plan(present)
    assert "train-00000-of-00011.parquet" not in plan["command"]
    assert "test-00000-of-00002.parquet" not in plan["command"]
    assert "train-00001-of-00011.parquet" in plan["command"]
    assert "validation-00000-of-00003.parquet" in plan["command"]
    assert plan["missing_validation"] == [
        "validation-00000-of-00003.parquet",
        "validation-00001-of-00003.parquet",
        "validation-00002-of-00003.parquet",
    ]


def test_phase3_does_not_select_or_rewrite_an_existing_report(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    phase2 = tmp_path / "phase2_report.md"
    phase2.write_text("phase2 stays", encoding="utf-8")
    write_events(
        data / "train-00000-of-00011.parquet",
        [_synthetic(1, 0, 3.0), _synthetic(2, 2, 0.1)],
    )
    write_events(data / "test-00000-of-00002.parquet", [_synthetic(3, 2, 0.2)])
    result = run_phase3(data, tmp_path / "phase3", "pytest")
    assert phase2.read_text(encoding="utf-8") == "phase2 stays"
    assert result["selected_model"] is None
    assert result["decision"]["status"] == "blocked"
    assert "exploratory_not_for_selection" in result["exploratory_test"]["pooled"]["role"]
    assert Path(result["report_path"]).is_file()


def _synthetic(event_id: int, label: int, accel: float) -> dict:
    event = nominal_event(event_id, label, samples=2, gps=2)
    event["gsensor"] = [[accel, 0.0, 1.0], [0.0, 0.0, 1.0]]
    event["gyro"] = xyz(2, 0.0)
    return event
