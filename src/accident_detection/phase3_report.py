"""Write the phase-3 diagnosis. Test figures stay labeled exploratory."""

from __future__ import annotations

import json
from pathlib import Path


def write_phase3(result: dict[str, object], output_dir: Path) -> None:
    """Write the diagnosis report and machine-readable result."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diagnosis.json"
    report_path = output_dir / "phase3_report.md"
    result["json_path"] = str(json_path)
    result["report_path"] = str(report_path)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_markdown(result), encoding="utf-8")


def _markdown(result: dict[str, object]) -> str:
    parts = [
        _header(result),
        _section("Why the full logistic model floods the second test shard", _cause_lines(result)),
        _section("Feature distributions", _distribution_lines(result)),
        _section("Resolution features", _resolution_lines(result)),
        _section("Acceleration-only baseline", _accel_lines(result)),
        _section("Exploratory test results", _exploratory_lines(result)),
        _section("Why one shard is not a generalization claim", [f"- {line}" for line in result["insufficiency"]]),
        _section("Download plan", _download_lines(result)),
        _section("Decision", _decision_lines(result)),
    ]
    return "\n".join(parts)


def _header(result: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Phase 3 diagnosis",
            "",
            "Raw files and the official test shards were not modified. "
            f"Rows removed: {result['rows_removed']}.",
            f"Selected model: `{result['selected_model']}`.",
            "",
        ]
    )


def _section(title: str, lines: list[str]) -> str:
    return "\n".join([f"## {title}", "", *lines, ""])


def _cause_lines(result: dict[str, object]) -> list[str]:
    encoding = result["encoding"]
    lines = [
        "The full model median-imputes, standardizes with the training mean and scale, then fits balanced logistic regression at probability 0.5.",
        (
            f"A GPS length of 17 is {encoding['gps_samples_z_at_17']:.2f} training standard deviations "
            f"above the training mean, adding {encoding['gps_samples_logit_at_17']:.2f} to the logit."
        ),
        f"`gyro_is_hd` has coefficient {encoding['gyro_is_hd_coefficient']} because every training row is 0. It does not move the score.",
        "Mean contribution on rows the model calls crashes in the second test shard:",
        "",
        "| Feature | Coefficient | Train scale | Mean contribution |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in result["shard2_contributions"][:8]:
        lines.append(
            f"| {row['name']} | {row['coefficient']:.4f} | {row['train_scale']:.6f} | {row['mean_contribution']:.4f} |"
        )
    return lines


def _distribution_lines(result: dict[str, object]) -> list[str]:
    lines = [
        "| File | Rows | Crashes | HD gyro | GPS length 17 | Peak horizontal p50 | GPS end p50 | Gyro peak p50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for shard in result["distributions"]:
        columns = shard["columns"]
        lines.append(
            "| {file} | {rows} | {crashes} | {hd} | {gps17} | {peak} | {end} | {gyro} |".format(
                file=shard["file_name"],
                rows=shard["rows"],
                crashes=shard["crash_rows"],
                hd=shard["gyro_is_hd_rows"],
                gps17=shard["gps_length_17"],
                peak=_num(columns["peak_horizontal_g"]["p50"]),
                end=_num(columns["gps_end_kmh"]["p50"]),
                gyro=_num(columns["peak_abs_gyro_deg_s"]["p50"]),
            )
        )
    return lines


def _resolution_lines(result: dict[str, object]) -> list[str]:
    return [
        "High-definition and standard gyroscopes use different stored resolutions. This training shard contains no HD rows, so the fit never observes that encoding.",
        "The HD indicator is constant, and standardizing it does not create a usable resolution correction.",
        "Gyroscope magnitude is not clipped. Its coefficient stays in the full model, but an acceleration-only model leaves the column out rather than rescaling it.",
        f"Full-model features: {', '.join(result['full_feature_names'])}.",
        f"Acceleration-only features: {', '.join(result['accel_feature_names'])}.",
    ]


def _accel_lines(result: dict[str, object]) -> list[str]:
    return [
        "The acceleration-only logistic model uses the five acceleration summaries already defined for the full model.",
        "Its imputer and scaler are fit on those training columns only. The probability threshold stays 0.5.",
        "This set was specified because the paper's physical baseline uses acceleration and because length and resolution columns are not acceleration. It was not chosen by searching the test set.",
        "No validation shard is available, so this run does not promote it over the peak baseline.",
    ]


def _exploratory_lines(result: dict[str, object]) -> list[str]:
    exploratory = result["exploratory_test"]
    if exploratory is None:
        return ["Test curves were not written because a validation decision was frozen first."]
    lines = [
        "These numbers describe the frozen training fits. They were not used to pick features or thresholds.",
        "",
    ]
    lines.extend(_metric_block(exploratory["pooled"]))
    for evaluation in exploratory["per_file"]:
        lines.extend(_metric_block(evaluation))
    lines.append(f"Figures: `{exploratory['figures']}`.")
    return lines


def _metric_block(evaluation: dict[str, object]) -> list[str]:
    lines = [
        f"### {evaluation['name']}",
        "",
        f"Role: `{evaluation['role']}`. Rows: {evaluation['rows']}.",
        "",
        "| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in evaluation["models"].items():
        lines.append(
            f"| {name} | {_num(metrics['precision_crash'])} | {_num(metrics['recall_crash'])} | "
            f"{_num(metrics['f1_crash'])} | {_num(metrics['balanced_accuracy'])} | "
            f"{_num(metrics['average_precision_crash'])} |"
        )
        lines.append(f"- {name} confusion `{metrics['confusion_matrix']['matrix']}`.")
    lines.append("")
    return lines


def _download_lines(result: dict[str, object]) -> list[str]:
    plan = result["download_plan"]
    lines = [
        f"Missing training shards: {len(plan['missing_train'])}. Missing validation shards: {len(plan['missing_validation'])}.",
        f"Approximate download size: {plan['approximate_gib']} GiB.",
        "",
        "```",
        plan["command"],
        "```",
        "",
    ]
    lines.extend(f"- {rule}" for rule in plan["rules"])
    return lines


def _decision_lines(result: dict[str, object]) -> list[str]:
    decision = result["decision"]
    return [
        f"Status: `{decision['status']}`.",
        decision["reason"],
        "When the validation shards are present, rerun the same command. Selection uses validation average precision only, and the official test split is scored after that choice is frozen.",
        f"Command: `{result['command']}`.",
    ]


def _num(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
