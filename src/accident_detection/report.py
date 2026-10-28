"""Write the audit report from an in-memory AuditReport."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from accident_detection.constants import LABEL_NAMES, LABEL_ORDER, SIGNAL_COLUMNS
from accident_detection.models import AuditReport, FileScan
from accident_detection.plots import save_audit_figures

_ANOMALY_FIELDS = (
    "source_file",
    "split",
    "row_index",
    "event_id",
    "label",
    "label_name",
    "code",
    "detail",
)


def write_outputs(
    report: AuditReport,
    report_dir: Path,
    figure_dir: Path,
    *,
    generated_at: datetime | None = None,
    command: str = "python -m accident_detection.cli",
) -> tuple[Path, ...]:
    """Write the markdown report, JSON summary, anomaly table, and figures."""

    report_dir.mkdir(parents=True, exist_ok=True)
    moment = generated_at or datetime.now(timezone.utc)
    summary = summary_dict(report, generated_at=moment, command=command)
    json_path = report_dir / "audit_summary.json"
    markdown_path = report_dir / "audit_report.md"
    anomaly_path = report_dir / "anomalies.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report, summary), encoding="utf-8")
    _write_anomalies(anomaly_path, report)
    figures = save_audit_figures(
        report.files,
        report.examples,
        figure_dir,
        nominal_gps_samples=report.config.nominal_gps_samples,
        imu_hz=report.config.imu_hz,
    )
    return (markdown_path, json_path, anomaly_path, *figures)


def summary_dict(
    report: AuditReport,
    *,
    generated_at: datetime,
    command: str,
) -> dict[str, object]:
    """Machine-readable audit summary. Row-level flags stay in the CSV."""

    return {
        "data_dir": report.data_dir,
        "generated_at": generated_at.isoformat(),
        "command": command,
        "rows_removed": 0,
        "nominal_imu_samples": report.config.nominal_imu_samples,
        "nominal_gps_samples": report.config.nominal_gps_samples,
        "imu_hz": report.config.imu_hz,
        "gps_hz": report.config.gps_hz,
        "coverage_notes": list(report.coverage_notes),
        "files": [_file_dict(scan) for scan in report.files],
        "split_totals": _split_totals(report.files),
        "duplicate_event_ids": [
            {"event_id": item.event_id, "locations": list(item.locations)}
            for item in report.duplicates
        ],
        "descriptive_stats": report.descriptive_stats,
        "examples": [_example_dict(example) for example in report.examples.values()],
        "anomaly_policy": _policy_lines(report),
    }


def _file_dict(scan: FileScan) -> dict[str, object]:
    return {
        "file_name": scan.file_name,
        "split": scan.split,
        "num_rows": scan.num_rows,
        "num_row_groups": scan.num_row_groups,
        "file_size_bytes": scan.file_size_bytes,
        "schema_problems": list(scan.schema_problems),
        "read_error": scan.read_error,
        "signals_scanned": scan.signals_scanned,
        "label_counts": {str(key): value for key, value in sorted(scan.label_counts.items())},
        "label_percentages": _label_percentages(scan.label_counts, scan.num_rows),
        "null_counts": scan.null_counts,
        "null_element_totals": scan.null_element_totals,
        "nonfinite_totals": scan.nonfinite_totals,
        "length_counts": {
            name: {str(length): count for length, count in sorted(counts.items())}
            for name, counts in scan.length_counts.items()
        },
        "extrema": scan.extrema,
        "vehiclesize_counts": scan.vehiclesize_counts,
        "gyro_is_hd_counts": scan.gyro_hd_counts,
        "anomaly_counts": dict(_flag_counter(scan.anomalies)),
    }


def _markdown(report: AuditReport, summary: dict[str, object]) -> str:
    parts = [
        _header(summary),
        _section("File inventory", _inventory_lines(report)),
        _section("Class distribution", _class_intro() + _label_lines(report)),
        _section("Split totals", _split_intro() + _split_lines(summary["split_totals"])),
        _section("Missingness", _null_lines(report)),
        _section("Duplicates", _duplicate_lines(report)),
        _section("Signal integrity", _signal_lines(report)),
        _section("Train-only descriptive statistics", _stats_lines(report)),
        _section("Anomaly policy", [f"- {line}" for line in summary["anomaly_policy"]]),
        _section("Representative examples", _example_lines(report)),
        _section("Limitations", _limitation_lines(report)),
    ]
    return "\n".join(parts)


def _header(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# VZCrash data audit",
            "",
            f"Generated at {summary['generated_at']}.",
            f"Command: `{summary['command']}`.",
            "",
            "Raw Parquet files were read and not modified. No rows were removed.",
            "No model was trained, and no cleaning threshold was fit on the test split.",
            "",
        ]
    )


def _section(title: str, lines: list[str]) -> str:
    return "\n".join([f"## {title}", "", *lines, ""])


def _class_intro() -> list[str]:
    return ["Counts below are per file. A shard is not assumed to match the full split.", ""]


def _split_intro() -> list[str]:
    return ["These sums hide the per-file differences shown above.", ""]


def _inventory_lines(report: AuditReport) -> list[str]:
    if not report.files:
        return ["No Parquet files were found."]
    lines = ["| File | Split | Rows | Row groups | Bytes | Schema problems |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for scan in report.files:
        problem = "; ".join(scan.schema_problems) or "none"
        if scan.read_error:
            problem = f"{problem}; read error: {scan.read_error}"
        lines.append(
            f"| {scan.file_name} | {scan.split} | {scan.num_rows} | {scan.num_row_groups} | {scan.file_size_bytes} | {problem} |"
        )
    return lines


def _label_lines(report: AuditReport) -> list[str]:
    lines = ["| File | Split | Label | Count | Percent of file |", "| --- | --- | --- | ---: | ---: |"]
    for scan in report.files:
        lines.extend(_label_rows(scan.file_name, scan.split, scan.label_counts, scan.num_rows))
    return lines


def _label_rows(file_name: str, split: str, counts: dict[int, int], total: int) -> list[str]:
    rows = []
    seen = set(LABEL_ORDER)
    for label_id in list(LABEL_ORDER) + sorted(set(counts) - seen):
        name = LABEL_NAMES.get(label_id, "unknown")
        count = counts.get(label_id, 0)
        rows.append(f"| {file_name} | {split} | {label_id} {name} | {count} | {_pct(count, total)} |")
    return rows


def _split_lines(totals: dict[str, object]) -> list[str]:
    lines = ["| Split | Label | Count | Percent of split |", "| --- | --- | ---: | ---: |"]
    for split, payload in totals.items():
        rows = int(payload["rows"])
        for label_id, count in payload["labels"].items():
            name = LABEL_NAMES.get(int(label_id), "unknown")
            lines.append(f"| {split} | {label_id} {name} | {count} | {_pct(int(count), rows)} |")
    return lines


def _null_lines(report: AuditReport) -> list[str]:
    if not report.files:
        return ["No files to inspect."]
    lines = ["Outer nulls are null column values. Null elements are missing samples inside a list.", ""]
    for scan in report.files:
        outer = ", ".join(f"{name}={count}" for name, count in scan.null_counts.items())
        inner = ", ".join(f"{name}={count}" for name, count in scan.null_element_totals.items()) or "not scanned"
        nonfinite = ", ".join(f"{name}={count}" for name, count in scan.nonfinite_totals.items()) or "not scanned"
        lines.append(f"- `{scan.file_name}` outer nulls: {outer}.")
        lines.append(f"  Null elements: {inner}. Non-finite scalars: {nonfinite}.")
    return lines


def _duplicate_lines(report: AuditReport) -> list[str]:
    if not report.duplicates:
        return ["No duplicate `event_id` values were found within or across the audited files."]
    lines = [f"{len(report.duplicates)} duplicated event ids. Locations are `file:row`.", ""]
    for item in report.duplicates[:20]:
        lines.append(f"- `{item.event_id}`: {', '.join(item.locations)}")
    if len(report.duplicates) > 20:
        lines.append("- Further duplicates are listed as `duplicate_event_id` in `anomalies.csv`.")
    return lines


def _signal_lines(report: AuditReport) -> list[str]:
    lines = []
    for scan in report.files:
        lines.append(f"### {scan.file_name}")
        lines.append("")
        if not scan.signals_scanned:
            lines.append("Signals were not decoded.")
            lines.append("")
            continue
        for name in SIGNAL_COLUMNS:
            counts = scan.length_counts.get(name, {})
            rendered = ", ".join(f"{length}:{count}" for length, count in sorted(counts.items()))
            lines.append(f"- `{name}` lengths: {rendered}.")
            extrema = scan.extrema.get(name, {})
            lines.append(f"  Finite range: min={extrema.get('min')} max={extrema.get('max')} count={extrema.get('finite_count')}.")
        lines.append("")
    return lines


def _stats_lines(report: AuditReport) -> list[str]:
    stats = report.descriptive_stats
    lines = [str(stats.get("note", ""))]
    peaks = stats.get("peak_acceleration_magnitude_g", {})
    if not isinstance(peaks, dict) or not peaks:
        return lines + ["No scanned training rows were available."]
    lines.append("")
    lines.append("| Class | Peaks | Mean g | p50 g | p95 g |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, summary in peaks.items():
        lines.append(
            f"| {name} | {summary.get('count', 0)} | {_num(summary.get('mean'))} | {_num(summary.get('p50'))} | {_num(summary.get('p95'))} |"
        )
    return lines


def _example_lines(report: AuditReport) -> list[str]:
    if not report.examples:
        return ["No training examples were available to plot."]
    lines = ["Plots use the first nominal-length training row of each class. Test rows are not plotted.", ""]
    for label_id in LABEL_ORDER:
        example = report.examples.get(label_id)
        if example is None:
            lines.append(f"- {LABEL_NAMES[label_id]}: no training example.")
            continue
        timing = "nominal" if example.nominal_timing else "non-nominal"
        lines.append(
            f"- {LABEL_NAMES[label_id]}: `{example.source_file}` row {example.row_index}, event {example.event_id}, {timing}."
        )
    return lines


def _limitation_lines(report: AuditReport) -> list[str]:
    lines = [f"- {note}." if not note.endswith(".") else f"- {note}" for note in report.coverage_notes]
    lines.extend(
        [
            "- This phase does not train a model and does not standardize signals.",
            "- Nominal IMU length is 100 Hz times 16 seconds. Nominal GPS length is 1 Hz times 16 seconds.",
            "- Those nominal lengths were not replaced by the most common length in a test shard.",
            "- Class-conditional peak summaries and example plots use training rows only.",
            "- Test and validation files are still counted, checked for integrity, and kept separate.",
            "- Large gyroscope values are reported in the finite range and are not clipped.",
            "- The release license is CC BY-NC 4.0. This audit does not add video or location fields.",
        ]
    )
    return lines


def _policy_lines(report: AuditReport) -> list[str]:
    return [
        "Every audited row is kept. `rows_removed` is 0.",
        f"IMU lists that are not {report.config.nominal_imu_samples} samples are flagged `length_not_nominal`.",
        f"GPS lists that are not {report.config.nominal_gps_samples} samples are flagged `length_not_nominal` and kept at their original length.",
        "Accelerometer and gyroscope length disagreements are flagged `imu_length_mismatch`.",
        "Nulls, empty lists, non-finite values, unknown labels, and duplicate event IDs are flagged.",
        "Flags are listed in `anomalies.csv`. They are not a filter.",
    ]


def _split_totals(scans: tuple[FileScan, ...]) -> dict[str, dict[str, object]]:
    totals: dict[str, dict[str, object]] = {}
    for scan in scans:
        bucket = totals.setdefault(scan.split, {"rows": 0, "labels": {}})
        bucket["rows"] = int(bucket["rows"]) + scan.num_rows
        labels = bucket["labels"]
        for label_id, count in scan.label_counts.items():
            labels[str(label_id)] = int(labels.get(str(label_id), 0)) + int(count)
    for payload in totals.values():
        for label_id in LABEL_ORDER:
            payload["labels"].setdefault(str(label_id), 0)
    return totals


def _label_percentages(counts: dict[int, int], total: int) -> dict[str, float]:
    percentages = {}
    for label_id in list(LABEL_ORDER) + sorted(set(counts) - set(LABEL_ORDER)):
        name = LABEL_NAMES.get(label_id, f"unknown_{label_id}")
        count = counts.get(label_id, 0)
        percentages[name] = 0.0 if total == 0 else round(100.0 * count / total, 4)
    return percentages


def _flag_counter(anomalies: list[dict[str, object]]) -> Counter[str]:
    return Counter(str(row["code"]) for row in anomalies)


def _write_anomalies(path: Path, report: AuditReport) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ANOMALY_FIELDS)
        writer.writeheader()
        for scan in report.files:
            for row in scan.anomalies:
                writer.writerow({field: row.get(field, "") for field in _ANOMALY_FIELDS})


def _example_dict(example) -> dict[str, object]:
    return {
        "label": example.label,
        "label_name": LABEL_NAMES.get(example.label, "unknown"),
        "event_id": example.event_id,
        "source_file": example.source_file,
        "row_index": example.row_index,
        "nominal_timing": example.nominal_timing,
    }


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0.00"
    return f"{(100.0 * count / total):.2f}"


def _num(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return ""
