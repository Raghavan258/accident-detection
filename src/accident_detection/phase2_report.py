"""Write the phase-2 findings and the single baseline run."""

from __future__ import annotations

import json
from pathlib import Path


def write_phase2(result: dict[str, object], output_dir: Path) -> None:
    """Write the markdown report and the JSON result for one experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "experiment_results.json"
    report_path = output_dir / "phase2_report.md"
    result["json_path"] = str(json_path)
    result["report_path"] = str(report_path)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_markdown(result), encoding="utf-8")


def _markdown(result: dict[str, object]) -> str:
    parts = [
        _header(result),
        _section("Split limits", [f"- {note}" for note in result["limitations"]]),
        _section("GPS length", _gps_lines(result)),
        _section("IMU length", _imu_lines(result)),
        _section("Gyroscope extremes", _gyro_lines(result)),
        _section("Decisions", [f"- {rule}" for rule in result["rules"]]),
        _section("Baselines", _baseline_lines(result)),
        _section("Command", [f"`{result['command']}`"]),
        _section("This run", _run_lines(result)),
    ]
    return "\n".join(parts)


def _header(result: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Phase 2 preparation",
            "",
            "Raw Parquet files were not modified. "
            f"Rows removed: {result['rows_removed']}.",
            "No hyperparameter was chosen from the test shards.",
            "",
        ]
    )


def _section(title: str, lines: list[str]) -> str:
    return "\n".join([f"## {title}", "", *lines, ""])


def _gps_lines(result: dict[str, object]) -> list[str]:
    lines = [
        "A 16-second, 1 Hz speed trace is 16 samples. Length 17 on every row would match an inclusive clock. It does not.",
        "The training shard is almost entirely length 16. Length 17 is not a duplicated last sample often enough to treat it as padding.",
        "Test shards have a different length mix. That mix was not used to pick a target length.",
        "",
        "| File | Length | Rows | HD gyro | Crashes | Last sample equals previous |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for quality in result["quality"]:
        lines.extend(_gps_rows(quality))
    return lines


def _gps_rows(quality: dict[str, object]) -> list[str]:
    return [
        "| {file} | {length} | {rows} | {hd} | {crash} | {match} |".format(
            file=quality["file_name"],
            length=bucket["length"],
            rows=bucket["rows"],
            hd=bucket["gyro_is_hd"],
            crash=bucket["crash"],
            match=bucket["last_equals_previous"],
        )
        for bucket in quality["gps"]
    ]


def _imu_lines(result: dict[str, object]) -> list[str]:
    lines = [
        "Windows whose length is not 1600 usually still match across accelerometer and gyroscope. That is a shorter or longer recording, not a broken axis.",
        "Mismatched channels stay in the matrix. Each sensor is reduced on its own samples. Nothing is resampled to 1600.",
        "",
        "| File | Rows | Not 1600 | Mismatched | Max accelerometer | Max gyroscope |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for quality in result["quality"]:
        imu = quality["imu"]
        lines.append(
            f"| {quality['file_name']} | {imu['rows']} | {imu['unusual_rows']} | {imu['mismatch_rows']} | {imu['max_gsensor_length']} | {imu['max_gyro_length']} |"
        )
        for example in imu["mismatch_examples"]:
            lines.append(
                f"- Mismatch in `{quality['file_name']}`: accelerometer {example['gsensor_length']}, gyroscope {example['gyro_length']}, label {example['label']}."
            )
    return lines


def _gyro_lines(result: dict[str, object]) -> list[str]:
    lines = [
        "Non-HD gyroscopes are stored at 1 deg/s. HD gyroscopes use 0.001 deg/s. 32767 deg/s is the int16 full scale.",
        "The 30000 deg/s row in the table is a reporting level, not a clip. No gyroscope sample is replaced.",
        "A training shard with no HD rows cannot teach the resolution difference. That is a limit of this download, not a reason to drop HD test rows.",
        "",
        "| File | HD | Rows | Median peak | Max peak | Rows > 1000 | Rows > 30000 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for quality in result["quality"]:
        for group in quality["gyro"]["by_resolution"]:
            lines.append(_gyro_row(quality["file_name"], group))
        for example in quality["gyro"]["rows_above_30000_deg_s"]:
            lines.append(
                f"- `{quality['file_name']}` row {example['row_index']}: peak {example['peak_abs_deg_s']:.1f} deg/s, HD {example['gyro_is_hd']}, label {example['label']}."
            )
    return lines


def _gyro_row(file_name: str, group: dict[str, object]) -> str:
    return (
        f"| {file_name} | {group['gyro_is_hd']} | {group['rows']} | "
        f"{_num(group['p50_abs_deg_s'])} | {_num(group['max_abs_deg_s'])} | "
        f"{group['rows_above_1000']} | {group['rows_above_30000']} |"
    )


def _baseline_lines(result: dict[str, object]) -> list[str]:
    peak = result["peak_baseline"]
    model = result["feature_model"]
    return [
        f"Peak baseline score: {peak['score']}.",
        f"Its threshold is {peak['threshold_g']:.6f} g, chosen by {peak['threshold_selection']}.",
        (
            f"Logistic regression uses C={model['logistic_C']}, class_weight={model['class_weight']}, "
            f"and crash probability >= {model['probability_threshold']}."
        ),
        "The imputer and scaler means come from the training rows only.",
        "Unknown vehicle size is the reference category. Light, Medium, and Heavy are indicators.",
    ]


def _run_lines(result: dict[str, object]) -> list[str]:
    lines = [
        "Training metrics are fit diagnostics. Test metrics are one pass on the local test shards.",
        "They are not a claim of final performance.",
        "",
    ]
    lines.extend(_metric_block("Training diagnostics", result["train_diagnostics"]))
    for evaluation in result["test_files"]:
        lines.extend(_metric_block(evaluation["name"], evaluation))
    if result["test_pooled"] is not None:
        lines.extend(_metric_block("Pooled test shards", result["test_pooled"]))
    return lines


def _metric_block(title: str, evaluation: dict[str, object]) -> list[str]:
    lines = [f"### {title}", "", f"Role: `{evaluation['role']}`. Rows: {evaluation['rows']}.", ""]
    lines.append("| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in ("peak_baseline", "logistic_regression"):
        lines.append(_metric_row(name, evaluation[name]))
    for name in ("peak_baseline", "logistic_regression"):
        matrix = evaluation[name]["confusion_matrix"]["matrix"]
        lines.append(f"- {name} confusion [[tn, fp], [fn, tp]] = `{matrix}`.")
    lines.append("")
    return lines


def _metric_row(name: str, metrics: dict[str, object]) -> str:
    return (
        f"| {name} | {_num(metrics['precision_crash'])} | {_num(metrics['recall_crash'])} | "
        f"{_num(metrics['f1_crash'])} | {_num(metrics['balanced_accuracy'])} | "
        f"{_num(metrics['average_precision_crash'])} |"
    )


def _num(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
