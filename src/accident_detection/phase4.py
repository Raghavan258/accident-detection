"""Fit on every official training shard and select on the validation split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from accident_detection.baselines import (
    LOGISTIC_C,
    LOGISTIC_MAX_ITER,
    LOGISTIC_RANDOM_STATE,
    PROBABILITY_THRESHOLD,
    fit_logistic,
    fit_peak_baseline,
    logistic_scores,
    predict_at_threshold,
)
from accident_detection.coverage import shard_inventory
from accident_detection.experiment import _concat
from accident_detection.features import (
    ACCEL_FEATURE_NAMES,
    DEFENSIBLE_FEATURE_NAMES,
    EXCLUDED_FEATURE_REASONS,
    PEAK_SCORE_FEATURE,
    select_columns,
)
from accident_detection.metrics import classification_metrics
from accident_detection.phase3_plots import save_confusion_grid, save_pr_curves
from accident_detection.phase4_report import write_phase4
from accident_detection.phase4_scan import duplicate_report, scan_shard
from accident_detection.selection import EXPERIMENT_VERSION, validation_choice

logger = logging.getLogger(__name__)
PEAK_MODEL = "peak_baseline"
ACCEL_MODEL = "accel_logistic"
DEFINED_MODEL = "defensible_logistic"
CANDIDATE_NAMES = (PEAK_MODEL, ACCEL_MODEL, DEFINED_MODEL)


def run_phase4(data_dir: Path | str, output_dir: Path | str, command: str) -> dict[str, object]:
    """Audit, fit, select on validation, then score test once."""

    inventory = shard_inventory(data_dir)
    output = Path(output_dir)
    if not inventory["ready_to_select"]:
        result = _blocked(inventory, command, "training or validation shards are missing")
        _write_config(output, _blocked_config(result["decision"]))
        write_phase4(result, output)
        return result
    scanned = _scan_all(inventory)
    train = _concat(_split(scanned, "train"))
    validation = _concat(_split(scanned, "validation"))
    specs = _fit_candidates(train)
    decision = _choose(validation, specs)
    _write_config(output, _config(decision, specs))
    _validation_curves(validation, specs, output / "figures")
    result = _result(command, inventory, scanned, specs, validation, decision)
    result["test_evaluation"] = _maybe_test(scanned, specs, decision, output / "figures")
    write_phase4(result, output)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the validation-based experiment."""

    parser = argparse.ArgumentParser(description="Select a VZCrash model on the official validation split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/VZCrash/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.data_dir.is_dir():
        logger.error("Data directory does not exist: %s", args.data_dir)
        return 2
    recorded = sys.argv[1:] if argv is None else argv
    command = "python -m accident_detection.phase4 " + " ".join(recorded)
    result = run_phase4(args.data_dir, args.output_dir, command)
    logger.info("Wrote %s", result["report_path"])
    logger.info("Status: %s", result["decision"]["status"])
    return 0 if result["decision"]["status"] == "frozen" else 1


def _scan_all(inventory: dict[str, object]) -> list[dict[str, object]]:
    parts = []
    for split in ("train", "validation", "test"):
        for path in inventory["paths"][split]:
            logger.info("Scanning %s", path.name)
            parts.append(scan_shard(path, keep_matrix=True))
    return parts


def _choose(validation: dict[str, np.ndarray], specs: dict[str, dict[str, object]]) -> dict[str, object]:
    scores = {name: _scores(validation, specs[name]) for name in CANDIDATE_NAMES}
    return validation_choice(scores, validation["y"])


def _maybe_test(parts, specs, decision, figure_dir: Path) -> dict[str, object]:
    if decision["selected_model"] is None:
        return {"status": "not_run", "reason": "selection is blocked, so the test split was not scored"}
    test_parts = _split(parts, "test")
    if not test_parts:
        return {"status": "blocked", "reason": "official test shards are missing"}
    return _frozen_test(test_parts, specs, decision, figure_dir)


def _result(command, inventory, parts, specs, validation, decision) -> dict[str, object]:
    public_inventory = {key: value for key, value in inventory.items() if key != "paths"}
    return {
        "command": command,
        "rows_removed": 0,
        "coverage": public_inventory,
        "duplicates": duplicate_report(parts),
        "audits": [part["audit"] for part in parts],
        "gps_conclusion": _gps_conclusion(parts),
        "excluded_features": EXCLUDED_FEATURE_REASONS,
        "candidates": {name: _public_spec(specs[name]) for name in CANDIDATE_NAMES},
        "validation_comparison": _comparison(validation, specs),
        "decision": decision,
    }


def _fit_candidates(train: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    labels = train["y"]
    accel = fit_logistic(select_columns(train["features"], ACCEL_FEATURE_NAMES), labels)
    defined = fit_logistic(select_columns(train["features"], DEFENSIBLE_FEATURE_NAMES), labels)
    return {
        PEAK_MODEL: {"kind": "peak", "features": (PEAK_SCORE_FEATURE,)},
        ACCEL_MODEL: {"kind": "logistic", "model": accel, "features": ACCEL_FEATURE_NAMES},
        DEFINED_MODEL: {"kind": "logistic", "model": defined, "features": DEFENSIBLE_FEATURE_NAMES},
    }


def _comparison(validation: dict[str, np.ndarray], specs: dict[str, dict[str, object]]) -> dict[str, object]:
    compared = {}
    for name in CANDIDATE_NAMES:
        scores = _scores(validation, specs[name])
        threshold = fit_peak_baseline(scores, validation["y"]).threshold_g
        compared[name] = {
            "max_f1_threshold": present_metrics(validation["y"], scores, threshold, "validation_operating_point"),
        }
        if specs[name]["kind"] == "logistic":
            compared[name]["probability_0_5"] = present_metrics(
                validation["y"], scores, PROBABILITY_THRESHOLD, "validation_reference_not_selected"
            )
    return compared


def _frozen_test(test_parts, specs, decision, figure_dir: Path) -> dict[str, object]:
    selected = str(decision["selected_model"])
    threshold = float(decision["threshold"])
    pooled = _concat(test_parts)
    pooled_scores = _scores(pooled, specs[selected])
    per_file = [_frozen_slice(part, specs[selected], threshold) for part in test_parts]
    pooled_metrics = present_metrics(pooled["y"], pooled_scores, threshold, "frozen_test_scored_once")
    _test_figures(test_parts, pooled, specs[selected], selected, threshold, figure_dir)
    return {
        "status": "scored_once",
        "model": selected,
        "threshold": threshold,
        "pooled": pooled_metrics,
        "by_file": per_file,
        "figures": str(figure_dir),
    }


def _frozen_slice(part, spec, threshold: float) -> dict[str, object]:
    metrics = present_metrics(part["y"], _scores(part, spec), threshold, "frozen_test_shard_diagnostic")
    metrics["file_name"] = part["file_name"]
    return metrics


def _scores(part: dict[str, np.ndarray], spec: dict[str, object]) -> np.ndarray:
    if spec["kind"] == "peak":
        return part["peak"]
    return logistic_scores(spec["model"], select_columns(part["features"], spec["features"]))


def _split(parts: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    return [part for part in parts if part["split"] == split]


def present_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, role: str) -> dict[str, object]:
    """Attach false-positive counts and mark an empty positive class as undefined."""

    prediction = predict_at_threshold(scores, threshold)
    metrics = classification_metrics(labels, prediction, scores)
    matrix = metrics["confusion_matrix"]["matrix"]
    view: dict[str, object] = {
        "role": role,
        "threshold": float(threshold),
        "false_positives": int(matrix[0][1]),
        "false_negatives": int(matrix[1][0]),
        **metrics,
    }
    if int(metrics["support_crash"]) == 0:
        _mark_undefined(view)
    return view


def _mark_undefined(view: dict[str, object]) -> None:
    view["positive_class_metrics"] = "undefined"
    view["precision_crash"] = None
    view["recall_crash"] = None
    view["f1_crash"] = None
    view["positive_class_note"] = (
        "This slice has zero crash labels, so precision, recall, F1, and average precision are undefined."
    )


def _public_spec(spec: dict[str, object]) -> dict[str, object]:
    if spec["kind"] == "peak":
        return {"kind": "peak", "features": list(spec["features"]), "score": "max over time of hypot(accel_x, accel_y)"}
    return _logistic_public(spec["model"], spec["features"])


def _logistic_public(model, names: tuple[str, ...]) -> dict[str, object]:
    classifier = model.named_steps["clf"]
    return {
        "kind": "logistic",
        "features": list(names),
        "imputer": "median fit on training rows",
        "scaler": "standard scaler fit on training rows",
        "scaler_mean": [float(value) for value in model.named_steps["scaler"].mean_],
        "scaler_scale": [float(value) for value in model.named_steps["scaler"].scale_],
        "logistic_C": LOGISTIC_C,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": LOGISTIC_RANDOM_STATE,
        "max_iter": LOGISTIC_MAX_ITER,
        "intercept": float(classifier.intercept_[0]),
        "coefficients": {name: float(value) for name, value in zip(names, classifier.coef_.ravel())},
    }


def _config(decision: dict[str, object], specs: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "status": decision["status"],
        "selected_model": decision["selected_model"],
        "threshold": decision["threshold"],
        "model_criterion": decision["model_criterion"],
        "threshold_criterion": decision["threshold_criterion"],
        "use_case_note": decision["use_case_note"],
        "random_state": LOGISTIC_RANDOM_STATE,
        "candidates": {name: _public_spec(specs[name]) for name in CANDIDATE_NAMES},
        "excluded_features": EXCLUDED_FEATURE_REASONS,
        "test_metrics_included": False,
    }


def _write_config(output: Path, config: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _blocked(inventory: dict[str, object], command: str, reason: str) -> dict[str, object]:
    public_inventory = {key: value for key, value in inventory.items() if key != "paths"}
    return {
        "command": command,
        "rows_removed": 0,
        "coverage": public_inventory,
        "decision": {"selected_model": None, "status": "blocked", "reason": reason, "threshold": None},
        "test_evaluation": {"status": "not_run", "reason": "selection is blocked, so the test split was not scored"},
        "audits": [],
        "validation_comparison": {},
    }


def _gps_conclusion(parts: list[dict[str, object]]) -> dict[str, object]:
    length_17 = _sum_length(parts, 17)
    return {
        "length_16": _sum_length(parts, 16),
        "length_17": length_17,
        "interpretation": _gps_text(length_17),
    }


def _sum_length(parts: list[dict[str, object]], length: int) -> dict[str, int]:
    totals = {"rows": 0, "last_equals_previous": 0, "first_equals_second": 0, "no_adjacent_duplicate": 0}
    for part in parts:
        match = next((item for item in part["audit"]["gps_structure"] if item["length"] == length), None)
        if match is None:
            continue
        for key in totals:
            totals[key] += int(match[key])
    return totals


def _gps_text(stats: dict[str, int]) -> str:
    if stats["rows"] == 0:
        return "No GPS sequences of length 17 were stored."
    return (
        f"{stats['rows']} rows store 17 GPS samples. {stats['last_equals_previous']} repeat the final "
        f"sample and {stats['first_equals_second']} repeat the first sample. Those counts can overlap. "
        f"{stats['no_adjacent_duplicate']} rows have no adjacent duplicate, so length 17 is not one "
        "parse error. Rows stay at their stored length."
    )


def _test_figures(test_parts, pooled, spec, model_name: str, threshold: float, figure_dir: Path) -> None:
    scores = _scores(pooled, spec)
    save_pr_curves(
        [(model_name, pooled["y"], scores)],
        figure_dir / "pr_curve_frozen_test.png",
        "Frozen test — scored once, not used for selection",
    )
    panels = [("pooled test", present_metrics(pooled["y"], scores, threshold, "figure")["confusion_matrix"]["matrix"])]
    for part in test_parts:
        part_scores = _scores(part, spec)
        matrix = present_metrics(part["y"], part_scores, threshold, "figure")["confusion_matrix"]["matrix"]
        panels.append((part["file_name"], matrix))
    save_confusion_grid(panels, figure_dir / "confusion_frozen_test.png", "Frozen test confusion — scored once")


def _blocked_config(decision: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "status": decision["status"],
        "selected_model": None,
        "threshold": None,
        "reason": decision["reason"],
        "test_metrics_included": False,
    }


def _validation_curves(validation, specs, figure_dir: Path) -> None:
    curves = [(name, validation["y"], _scores(validation, specs[name])) for name in CANDIDATE_NAMES]
    save_pr_curves(curves, figure_dir / "pr_curves_validation.png", "Validation precision-recall — used for selection")
    panels = []
    for name in CANDIDATE_NAMES:
        scores = _scores(validation, specs[name])
        threshold = fit_peak_baseline(scores, validation["y"]).threshold_g
        matrix = present_metrics(validation["y"], scores, threshold, "figure")["confusion_matrix"]["matrix"]
        panels.append((name, matrix))
    save_confusion_grid(panels, figure_dir / "confusion_validation.png", "Validation confusion at each max-F1 threshold")


if __name__ == "__main__":
    raise SystemExit(main())
