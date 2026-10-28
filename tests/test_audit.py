"""Directory audit keeps official splits and does not drop rows."""

import json
from pathlib import Path

import numpy as np

from accident_detection.audit import audit_directory
from accident_detection.cli import main
from accident_detection.columns import grouped_finite_max
from accident_detection.models import AuditConfig
from accident_detection.report import write_outputs
from tests.parquet_factory import nominal_event, write_events, xyz

SHORT = AuditConfig(nominal_imu_samples=2, nominal_gps_samples=2)


def test_grouped_finite_max_skips_empty_and_nonfinite_groups() -> None:
    values = np.array([1.0, np.nan, 5.0])
    offsets = np.array([0, 2, 2, 3])
    peaks = grouped_finite_max(values, offsets)
    assert peaks[0] == 1.0
    assert np.isnan(peaks[1])
    assert peaks[2] == 5.0


def test_file_counts_stay_separate_and_rows_are_kept(tmp_path: Path) -> None:
    train = nominal_event(1, 0)
    train["gps_speed"] = [1.0, 2.0, 3.0]
    write_events(tmp_path / "train-00000-of-00011.parquet", [train, nominal_event(2, 2)])
    write_events(tmp_path / "test-00000-of-00002.parquet", [nominal_event(3, 2), nominal_event(4, 2)])
    before = (tmp_path / "train-00000-of-00011.parquet").read_bytes()

    report = audit_directory(tmp_path, SHORT)

    assert (tmp_path / "train-00000-of-00011.parquet").read_bytes() == before
    by_name = {scan.file_name: scan for scan in report.files}
    assert by_name["train-00000-of-00011.parquet"].num_rows == 2
    assert by_name["train-00000-of-00011.parquet"].label_counts[0] == 1
    assert by_name["test-00000-of-00002.parquet"].label_counts[0] == 0
    assert by_name["test-00000-of-00002.parquet"].label_counts[2] == 2
    flagged = by_name["train-00000-of-00011.parquet"].anomalies
    assert any(row["code"] == "length_not_nominal:gps_speed" for row in flagged)
    assert report.examples[0].event_id == 1
    assert 0 not in report.examples or report.examples[0].source_file.startswith("train")


def test_duplicate_ids_are_flagged_across_splits(tmp_path: Path) -> None:
    write_events(tmp_path / "train-00000-of-00011.parquet", [nominal_event(9, 1)])
    write_events(tmp_path / "test-00000-of-00002.parquet", [nominal_event(9, 2)])
    report = audit_directory(tmp_path, SHORT)
    assert [item.event_id for item in report.duplicates] == [9]
    codes = [row["code"] for scan in report.files for row in scan.anomalies]
    assert codes.count("duplicate_event_id") == 2
    assert report.examples[1].source_file.startswith("train")


def test_null_list_and_mismatch_do_not_reduce_the_row_count(tmp_path: Path) -> None:
    broken = nominal_event(5, 0)
    broken["gsensor"] = None
    broken["gyro"] = xyz(1, 1.0)
    broken["gps_speed"] = [1.0, None]
    write_events(tmp_path / "train-00000-of-00011.parquet", [broken])
    report = audit_directory(tmp_path, SHORT)
    scan = report.files[0]
    assert scan.num_rows == 1
    codes = {row["code"] for row in scan.anomalies}
    assert "null_signal:gsensor" in codes
    assert "null_elements:gps_speed" in codes
    assert "imu_length_mismatch" not in codes


def test_report_states_that_no_rows_were_removed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    reports = tmp_path / "reports"
    figures = tmp_path / "figures"
    data.mkdir()
    write_events(data / "train-00000-of-00011.parquet", [nominal_event(1, 0), nominal_event(2, 1), nominal_event(3, 2)])
    report = audit_directory(data, SHORT)
    write_outputs(report, reports, figures, command="pytest")
    summary = json.loads((reports / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["rows_removed"] == 0
    assert summary["files"][0]["label_counts"]["0"] == 1
    text = (reports / "audit_report.md").read_text(encoding="utf-8")
    assert "0 crash" in text
    assert "No rows were removed" in text
    assert (figures / "accel_magnitude_by_class.png").is_file()
    assert (reports / "anomalies.csv").read_text(encoding="utf-8").splitlines()[0].startswith("source_file")


def test_cli_writes_the_audit_report(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_events(data / "train-00000-of-00011.parquet", [nominal_event(1, 2)])
    code = main(
        [
            "--data-dir",
            str(data),
            "--report-dir",
            str(tmp_path / "reports"),
            "--figure-dir",
            str(tmp_path / "figures"),
        ]
    )
    assert code == 0
    assert (tmp_path / "reports" / "audit_report.md").is_file()


def test_empty_directory_has_coverage_notes(tmp_path: Path) -> None:
    report = audit_directory(tmp_path, SHORT)
    assert report.files == ()
    assert any(note.startswith("train: found 0 of 11") for note in report.coverage_notes)
