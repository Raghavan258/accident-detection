"""Write the phase 4 selection report. Test metrics are displayed, not used to refit."""

from __future__ import annotations

import json
from pathlib import Path


def write_phase4(result: dict[str, object], output_dir: Path) -> dict[str, object]:
    """Write JSON and Markdown. Raw Parquet files are not touched."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase4_results.json"
    report_path = output_dir / "phase4_report.md"
    result["report_path"] = str(report_path)
    result["json_path"] = str(json_path)
    report_path.write_text("\n".join(_lines(result)) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _lines(result: dict[str, object]) -> list[str]:
    decision = result["decision"]
    lines = [
        "# Phase 4 validation selection",
        "",
        f"Status: `{decision['status']}`. Selected model: `{decision.get('selected_model')}`.",
        f"Rows removed: {result['rows_removed']}.",
        "",
        _reason(decision),
        "",
        "## Coverage",
        "",
    ]
    lines.extend(_coverage(result["coverage"]))
    lines.extend(_audit_section(result.get("audits") or []))
    lines.extend(_gps_section(result.get("gps_conclusion")))
    lines.extend(_exclusion_section(result.get("excluded_features") or {}))
    lines.extend(_vehicle_section(result.get("duplicates")))
    lines.extend(_validation_section(result.get("validation_comparison") or {}, decision))
    lines.extend(_test_section(result.get("test_evaluation") or {}))
    lines.extend(_limits())
    lines.extend(["", f"Command: `{result['command']}`."])
    return lines


def _reason(decision: dict[str, object]) -> str:
    if decision.get("use_case_note"):
        return str(decision["use_case_note"])
    return str(decision.get("reason", ""))


def _coverage(coverage: dict[str, object]) -> list[str]:
    lines = [
        f"Missing training shards: {len(coverage['missing_train'])}.",
        f"Missing validation shards: {len(coverage['missing_validation'])}.",
        f"Missing test shards: {len(coverage['missing_test'])}.",
        "",
        "| File | Split | Bytes | Hard links |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in coverage["files"]:
        lines.append(
            f"| {item['file_name']} | {item['split']} | {item['bytes']} | {item['hardlink_count']} |"
        )
    if coverage["missing_train"] or coverage["missing_validation"]:
        lines.extend(["", "Download command:", "", "```", coverage["download_plan"]["command"], "```"])
    return lines


def _audit_section(audits: list[dict[str, object]]) -> list[str]:
    if not audits:
        return ["", "## Shard audit", "", "Not run, because selection was blocked before the shards were read."]
    lines = [
        "",
        "## Shard audit",
        "",
        "Counts below include the official test shards as a consistency check. They are not model-selection inputs.",
        "",
        "| File | Rows | Crash | Near miss | Normal | HD gyro | GPS 17 | Nulls | Non-finite | IMU mismatches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for audit in audits:
        lines.append(_audit_row(audit))
    schema = sorted({problem for audit in audits for problem in audit["schema_problems"]})
    extra = sorted({name for audit in audits for name in audit["extra_columns"]})
    lines.append("")
    lines.append(f"Schema problems: {schema or 'none'}.")
    lines.append(f"Extra columns: {extra or 'none'}.")
    return lines


def _audit_row(audit: dict[str, object]) -> str:
    labels = audit["label_counts"]
    nulls = sum(audit["null_counts"].values())
    nonfinite = sum(audit["nonfinite_totals"].values())
    gps_17 = _gps_rows(audit, 17)
    hd = _hd_rows(audit)
    return (
        f"| {audit['file_name']} | {audit['rows']} | {labels.get('0', 0)} | {labels.get('1', 0)} | "
        f"{labels.get('2', 0)} | {hd} | {gps_17} | {nulls} | {nonfinite} | {audit['imu']['mismatch_rows']} |"
    )


def _gps_rows(audit: dict[str, object], length: int) -> int:
    match = next((item for item in audit["gps"] if item["length"] == length), None)
    if match is None:
        return 0
    return int(match["rows"])


def _hd_rows(audit: dict[str, object]) -> int:
    groups = audit["gyro"]["by_resolution"]
    hd = next((item for item in groups if item["gyro_is_hd"]), None)
    if hd is None:
        return 0
    return int(hd["rows"])


def _gps_section(conclusion: dict[str, object] | None) -> list[str]:
    if not conclusion:
        return []
    stats = conclusion["length_17"]
    lines = ["", "## GPS length 17", "", str(conclusion["interpretation"]), ""]
    lines.append(
        f"Length 17 totals: rows {stats['rows']}, no adjacent duplicate {stats['no_adjacent_duplicate']}, "
        f"last sample repeated {stats['last_equals_previous']}, first sample repeated {stats['first_equals_second']}."
    )
    return lines


def _exclusion_section(reasons: dict[str, str]) -> list[str]:
    if not reasons:
        return []
    lines = [
        "",
        "## Excluded features",
        "",
        "These columns stay out of every candidate. The exclusion is part of the experiment definition, not a test-set search.",
        "",
    ]
    for name, reason in reasons.items():
        lines.append(f"- `{name}`: {reason}")
    lines.append("")
    lines.append(
        "The acceleration-only model uses the five acceleration summaries. The defensible logistic model adds "
        "GPS start, end, minimum, and drop, plus Light, Medium, and Heavy vehicle-size indicators. Unknown size is the reference."
    )
    return lines


def _vehicle_section(duplicates: dict[str, object] | None) -> list[str]:
    if not duplicates:
        return []
    across = duplicates["across_files"]
    within = duplicates["within_file"]
    return [
        "",
        "## Identifiers",
        "",
        str(duplicates["vehicle_identifier_note"]),
        f"Within-file duplicate event IDs: {within or 'none'}.",
        f"Cross-file duplicate event IDs recorded: {len(across)}.",
    ]


def _validation_section(comparison: dict[str, object], decision: dict[str, object]) -> list[str]:
    if not comparison:
        return ["", "## Validation", "", "No validation comparison was run."]
    lines = [
        "",
        "## Validation comparison",
        "",
        f"Model rule: {decision.get('model_criterion')}.",
        f"Threshold rule: {decision.get('threshold_criterion')}.",
        f"Frozen threshold: `{decision.get('threshold')}`.",
        "",
        "Operating points below maximize crash F1 on validation for display. The frozen model is the one with the highest validation average precision.",
        "",
        "| Model | Average precision | Precision | Recall | F1 | Balanced accuracy | FP | FN | Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in comparison.items():
        lines.append(_metric_row(name, payload["max_f1_threshold"]))
        if "probability_0_5" in payload:
            lines.append(_metric_row(f"{name} at 0.5", payload["probability_0_5"]))
    lines.extend(["", "Validation curves: `reports/phase4/figures/pr_curves_validation.png`."])
    return lines


def _test_section(evaluation: dict[str, object]) -> list[str]:
    lines = ["", "## Frozen test evaluation", ""]
    if evaluation.get("status") != "scored_once":
        lines.append(str(evaluation.get("reason", "Test split was not scored.")))
        return lines
    lines.append(
        f"Model `{evaluation['model']}` was scored once at threshold `{evaluation['threshold']}` after the config was written."
    )
    lines.append("These numbers were not used to change features or the threshold.")
    lines.extend(["", "### Pooled official test", ""])
    lines.append(_metric_line(evaluation["pooled"]))
    for shard in evaluation["by_file"]:
        lines.extend(["", f"### {shard['file_name']}", ""])
        lines.append(_metric_line(shard))
        if shard.get("positive_class_note"):
            lines.append(str(shard["positive_class_note"]))
            lines.append(f"False positives on this zero-crash shard: {shard['false_positives']}.")
    return lines


def _metric_row(name: str, metrics: dict[str, object]) -> str:
    return (
        f"| {name} | {_num(metrics.get('average_precision_crash'))} | {_num(metrics.get('precision_crash'))} | "
        f"{_num(metrics.get('recall_crash'))} | {_num(metrics.get('f1_crash'))} | "
        f"{_num(metrics.get('balanced_accuracy'))} | {metrics['false_positives']} | "
        f"{metrics['false_negatives']} | {_num(metrics.get('threshold'))} |"
    )


def _metric_line(metrics: dict[str, object]) -> str:
    matrix = metrics["confusion_matrix"]["matrix"]
    return (
        f"Precision {_num(metrics.get('precision_crash'))}, recall {_num(metrics.get('recall_crash'))}, "
        f"F1 {_num(metrics.get('f1_crash'))}, balanced accuracy {_num(metrics.get('balanced_accuracy'))}, "
        f"average precision {_num(metrics.get('average_precision_crash'))}. "
        f"False positives {metrics['false_positives']}, false negatives {metrics['false_negatives']}. "
        f"Confusion `{matrix}`."
    )


def _limits() -> list[str]:
    return [
        "",
        "## Limitations",
        "",
        "- Official shard names are the split. There is no vehicle id, so this run does not reconstruct the paper's vehicle-disjoint assignment.",
        "- Near-miss events stay in the negative class.",
        "- No missed-crash versus false-alarm cost was supplied, so the threshold is validation crash F1 rather than a cost-weighted point.",
        "- A change motivated by the test result needs a new validation selection and a fresh evaluation. This test pass is not reused for tuning.",
        "- Phase 2 and phase 3 reports are separate outputs and are not rewritten by this command.",
    ]


def _num(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
