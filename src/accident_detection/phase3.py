"""Diagnose the logistic baseline and fit an acceleration-only comparison."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from accident_detection.baselines import (
    fit_logistic,
    fit_peak_baseline,
    logistic_scores,
    predict_logistic,
    predict_peak,
)
from accident_detection.diagnosis import contribution_table, shard_summary, z_score
from accident_detection.experiment import _concat, _group_paths, _load_shard, _model_summary
from accident_detection.features import ACCEL_FEATURE_NAMES, FEATURE_NAMES, select_columns
from accident_detection.metrics import classification_metrics
from accident_detection.phase3_plots import save_confusion_grid, save_pr_curves
from accident_detection.phase3_report import write_phase3
from accident_detection.protocol import download_plan, freeze_decision

logger = logging.getLogger(__name__)
FULL_MODEL = "full_logistic"
ACCEL_MODEL = "accel_logistic"
PEAK_MODEL = "peak_baseline"


def run_phase3(data_dir: Path | str, output_dir: Path | str, command: str) -> dict[str, object]:
    """Fit both logistics on train. Do not select a winner without validation."""

    groups = _phase3_groups(Path(data_dir))
    if not groups["train"]:
        raise FileNotFoundError(f"no training parquet files in {data_dir}")
    train_parts = [_load_shard(path) for path in groups["train"]]
    test_parts = [_load_shard(path) for path in groups["test"]]
    validation_parts = [_load_shard(path) for path in groups["validation"]]
    train = _concat(train_parts)
    models = _fit_models(train)
    decision = _validation_decision(validation_parts, models)
    result = {
        "command": command,
        "rows_removed": 0,
        "selected_model": decision["selected_model"],
        "decision": decision,
        "full_feature_names": list(FEATURE_NAMES),
        "accel_feature_names": list(ACCEL_FEATURE_NAMES),
        "download_plan": download_plan({path.name for path in Path(data_dir).glob("*.parquet")}),
        "distributions": _distributions(train_parts, test_parts),
        "models": {
            name: _model_summary(model, names)
            for name, (model, names) in models.items()
            if name != PEAK_MODEL
        },
        "train_diagnostics": _slice_metrics("train", train, models, "fit_diagnostic"),
        "exploratory_test": _exploratory_test(test_parts, models, decision, Path(output_dir) / "figures"),
        "insufficiency": _insufficiency(train, train_parts),
    }
    result["encoding"] = _encoding_shift(result["models"][FULL_MODEL])
    result["shard2_contributions"] = _shard2_contributions(test_parts, models[FULL_MODEL][0])
    write_phase3(result, Path(output_dir))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the phase-3 diagnosis."""

    parser = argparse.ArgumentParser(description="Diagnose generalization without selecting on test.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/VZCrash/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.data_dir.is_dir():
        logger.error("Data directory does not exist: %s", args.data_dir)
        return 2
    recorded = sys.argv[1:] if argv is None else argv
    command = "python -m accident_detection.phase3 " + " ".join(recorded)
    result = run_phase3(args.data_dir, args.output_dir, command)
    logger.info("Wrote %s", result["report_path"])
    logger.info("Selected model: %s", result["selected_model"])
    return 0


def _phase3_groups(data_dir: Path) -> dict[str, list[Path]]:
    """Keep the one-shard diagnosis stable after the remaining shards arrive."""

    groups = _group_paths(data_dir)
    groups["train"] = [path for path in groups["train"] if path.name == "train-00000-of-00011.parquet"]
    groups["validation"] = []
    return groups


def _fit_models(train: dict[str, np.ndarray]) -> dict[str, tuple[object, tuple[str, ...]]]:
    peak = fit_peak_baseline(train["peak"], train["y"])
    full = fit_logistic(train["features"], train["y"])
    accel_matrix = select_columns(train["features"], ACCEL_FEATURE_NAMES)
    accel = fit_logistic(accel_matrix, train["y"])
    return {PEAK_MODEL: (peak, ()), FULL_MODEL: (full, FEATURE_NAMES), ACCEL_MODEL: (accel, ACCEL_FEATURE_NAMES)}


def _validation_decision(parts: list[dict[str, object]], models) -> dict[str, object]:
    if not parts:
        return freeze_decision({})
    validation = _concat(parts)
    scores = {name: _metrics_for(validation, models, name) for name in models}
    return freeze_decision(scores)


def _distributions(train_parts, test_parts) -> list[dict[str, object]]:
    summaries = []
    for part in train_parts + test_parts:
        summary = shard_summary(part["features"], part["y"])
        summary["file_name"] = part["file_name"]
        summaries.append(summary)
    return summaries


def _exploratory_test(test_parts, models, decision, figure_dir: Path) -> dict[str, object] | None:
    if decision["selected_model"] is not None or not test_parts:
        return None
    pooled = _concat(test_parts)
    per_file = [_slice_metrics(part["file_name"], part, models, "exploratory_not_for_selection") for part in test_parts]
    pooled_metrics = _slice_metrics("pooled_test", pooled, models, "exploratory_not_for_selection")
    _save_figures(test_parts, pooled, models, figure_dir)
    return {"per_file": per_file, "pooled": pooled_metrics, "figures": str(figure_dir)}


def _slice_metrics(name, part, models, role: str) -> dict[str, object]:
    return {
        "name": name,
        "role": role,
        "rows": int(part["y"].size),
        "models": {model_name: _metrics_for(part, models, model_name) for model_name in models},
    }


def _metrics_for(part, models, model_name: str) -> dict[str, object]:
    scores, predictions = _scores(part, models, model_name)
    return classification_metrics(part["y"], predictions, scores)


def _scores(part, models, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    model, names = models[model_name]
    if model_name == PEAK_MODEL:
        return part["peak"], predict_peak(model, part["peak"])
    matrix = part["features"] if names == FEATURE_NAMES else select_columns(part["features"], names)
    return logistic_scores(model, matrix), predict_logistic(model, matrix)


def _encoding_shift(summary: dict[str, object]) -> dict[str, object]:
    coefficients = summary["coefficients"]
    scales = dict(zip(summary["coefficients"], summary["scaler_scale"]))
    means = dict(zip(summary["coefficients"], summary["scaler_mean"]))
    return {
        "gps_samples_z_at_17": z_score(17.0, means["gps_samples"], scales["gps_samples"]),
        "gps_samples_logit_at_17": z_score(17.0, means["gps_samples"], scales["gps_samples"]) * coefficients["gps_samples"],
        "gyro_is_hd_coefficient": coefficients["gyro_is_hd"],
        "gyro_is_hd_train_scale": scales["gyro_is_hd"],
    }


def _shard2_contributions(test_parts, full_model) -> list[dict[str, float | str]]:
    shard = next((part for part in test_parts if part["file_name"].startswith("test-00001")), None)
    if shard is None:
        return []
    scores, predictions = _scores(shard, {FULL_MODEL: (full_model, FEATURE_NAMES)}, FULL_MODEL)
    del scores
    return contribution_table(full_model, shard["features"], FEATURE_NAMES, predictions == 1)


def _insufficiency(train, train_parts) -> list[str]:
    high_definition = int(sum(np.sum(part["features"][:, FEATURE_NAMES.index("gyro_is_hd")] == 1) for part in train_parts))
    crashes = int(np.sum(train["y"] == 1))
    return [
        f"The training matrix has {crashes} crashes and {high_definition} high-definition gyroscope rows.",
        "One shard is 1 of 11 official training shards, and this schema has no vehicle id.",
        "No validation shard is present, so a model or threshold cannot be frozen without using test labels.",
        "The second official test shard has no crashes, so its precision is a stress check rather than a tuning target.",
        "Train-only standardization of a near-constant length or resolution column does not transfer to a shard with a different encoding.",
    ]


def _save_figures(test_parts, pooled, models, figure_dir: Path) -> None:
    pooled_curves = [(name, pooled["y"], _scores(pooled, models, name)[0]) for name in models]
    save_pr_curves(pooled_curves, figure_dir / "pr_curves_exploratory_test.png", "Exploratory pooled test — not used for selection")
    panels = [(name, _metrics_for(pooled, models, name)["confusion_matrix"]["matrix"]) for name in models]
    save_confusion_grid(panels, figure_dir / "confusion_pooled_test.png", "Exploratory pooled test confusion")
    shard = next((part for part in test_parts if part["file_name"].startswith("test-00001")), None)
    if shard is not None:
        shard_panels = [(name, _metrics_for(shard, models, name)["confusion_matrix"]["matrix"]) for name in models]
        save_confusion_grid(shard_panels, figure_dir / "confusion_test_shard2.png", "Exploratory second test shard — zero crashes")


if __name__ == "__main__":
    raise SystemExit(main())
