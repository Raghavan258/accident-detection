"""Fit the first baselines on training shards and evaluate test shards once."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from accident_detection.baselines import (
    LOGISTIC_C,
    PROBABILITY_THRESHOLD,
    fit_logistic,
    fit_peak_baseline,
    logistic_scores,
    predict_logistic,
    predict_peak,
)
from accident_detection.constants import REQUIRED_COLUMNS
from accident_detection.features import FEATURE_NAMES, PEAK_SCORE_FEATURE, table_matrix
from accident_detection.labels import binary_crash_labels
from accident_detection.metrics import classification_metrics
from accident_detection.phase2_report import write_phase2
from accident_detection.quality import summarize_gps, summarize_gyro, summarize_imu
from accident_detection.schema import split_name
from accident_detection.splits import split_limitations

logger = logging.getLogger(__name__)
PHASE2_TRAIN_SHARD = "train-00000-of-00011.parquet"


def run_experiment(data_dir: Path | str, output_dir: Path | str, command: str) -> dict[str, object]:
    """Fit on train shards only, then score the official test shards once."""

    groups = _group_paths(Path(data_dir))
    train_paths = [path for path in groups["train"] if path.name == PHASE2_TRAIN_SHARD]
    if not train_paths:
        raise FileNotFoundError(f"no training parquet files in {data_dir}")
    train_parts = [_load_shard(path) for path in train_paths]
    groups = {**groups, "train": train_paths}
    test_parts = [_load_shard(path) for path in groups["test"]]
    train = _concat(train_parts)
    peak_baseline = fit_peak_baseline(train["peak"], train["y"])
    model = fit_logistic(train["features"], train["y"])
    result = _result(groups, train_parts, test_parts, train, peak_baseline, model, command)
    write_phase2(result, Path(output_dir))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the first training experiment."""

    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.data_dir.is_dir():
        logger.error("Data directory does not exist: %s", args.data_dir)
        return 2
    recorded = sys.argv[1:] if argv is None else argv
    command = "python -m accident_detection.experiment " + " ".join(recorded)
    result = run_experiment(args.data_dir, args.output_dir, command)
    logger.info("Wrote %s", result["report_path"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit VZCrash baselines on the training split only.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/VZCrash/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase2"))
    return parser


def _group_paths(data_dir: Path) -> dict[str, list[Path]]:
    groups = {"train": [], "validation": [], "test": [], "other": []}
    for path in sorted(data_dir.glob("*.parquet")):
        split = split_name(path)
        groups.get(split, groups["other"]).append(path)
    return groups


def _load_shard(path: Path) -> dict[str, object]:
    table = pq.read_table(path, columns=list(REQUIRED_COLUMNS))
    batch = table_matrix(table)
    labels = binary_crash_labels(batch["labels"], batch["label_null"])
    peak_index = FEATURE_NAMES.index(PEAK_SCORE_FEATURE)
    quality = {
        "file_name": path.name,
        "split": split_name(path),
        "rows": int(table.num_rows),
        "gps": summarize_gps(batch["gps"], batch["labels"], batch["gyro_is_hd"]),
        "imu": summarize_imu(batch["accel"], batch["gyro"], batch["labels"]),
        "gyro": summarize_gyro(batch["gyro"], batch["gyro_is_hd"], batch["labels"]),
    }
    return {
        "file_name": path.name,
        "features": batch["features"],
        "y": labels,
        "peak": batch["features"][:, peak_index],
        "quality": quality,
    }


def _concat(parts: list[dict[str, object]]) -> dict[str, np.ndarray]:
    return {
        "features": np.vstack([part["features"] for part in parts]),
        "y": np.concatenate([part["y"] for part in parts]),
        "peak": np.concatenate([part["peak"] for part in parts]),
    }


def _result(groups, train_parts, test_parts, train, peak_baseline, model, command) -> dict[str, object]:
    crashes = int(np.sum(train["y"] == 1))
    limitations = split_limitations(
        len(groups["train"]),
        len(groups["validation"]),
        len(groups["test"]),
        crashes,
    )
    return {
        "command": command,
        "rows_removed": 0,
        "feature_names": list(FEATURE_NAMES),
        "limitations": limitations,
        "rules": _rules(),
        "quality": [part["quality"] for part in train_parts + test_parts],
        "unused_validation_files": [path.name for path in groups["validation"]],
        "ignored_non_split_files": [path.name for path in groups["other"]],
        "peak_baseline": {
            "score": "max over time of hypot(accel_x, accel_y)",
            "threshold_g": peak_baseline.threshold_g,
            "threshold_selection": "maximum crash F1 on the training rows; lowest threshold wins ties",
            "train_f1_at_selection": peak_baseline.train_f1_crash,
        },
        "feature_model": _model_summary(model),
        "train_diagnostics": _evaluate("train", train, model, peak_baseline, "fit_diagnostic"),
        "test_files": [_evaluate(part["file_name"], part, model, peak_baseline, "single_preliminary_evaluation") for part in test_parts],
        "test_pooled": _evaluate("test", _concat(test_parts), model, peak_baseline, "single_preliminary_evaluation") if test_parts else None,
    }


def _evaluate(name, part, model, peak_baseline, role: str) -> dict[str, object]:
    peak_prediction = predict_peak(peak_baseline, part["peak"])
    probability = logistic_scores(model, part["features"])
    class_prediction = predict_logistic(model, part["features"])
    return {
        "name": name,
        "role": role,
        "rows": int(part["y"].size),
        "peak_baseline": classification_metrics(part["y"], peak_prediction, part["peak"]),
        "logistic_regression": classification_metrics(part["y"], class_prediction, probability),
    }


def _model_summary(model, feature_names: tuple[str, ...] = FEATURE_NAMES) -> dict[str, object]:
    classifier = model.named_steps["clf"]
    coefficients = classifier.coef_.ravel()
    return {
        "imputer": "median fit on training rows",
        "scaler": "standard scaler fit on training rows",
        "scaler_mean": model.named_steps["scaler"].mean_.tolist(),
        "scaler_scale": model.named_steps["scaler"].scale_.tolist(),
        "logistic_C": LOGISTIC_C,
        "class_weight": "balanced",
        "probability_threshold": PROBABILITY_THRESHOLD,
        "intercept": float(classifier.intercept_[0]),
        "coefficients": {name: float(value) for name, value in zip(feature_names, coefficients)},
        "iterations": int(classifier.n_iter_.ravel()[0]),
    }


def _rules() -> list[str]:
    return [
        "Crash is the positive class. Near-miss and normal driving stay negative. No class is resampled.",
        "GPS lists stay at their stored length. They are not truncated to 16 or padded to 17.",
        "IMU windows stay at their stored length. They are not resampled to 1600 samples.",
        "Accelerometer and gyroscope features are reduced separately when their lengths differ.",
        "Gyroscope samples are not clipped, including values near the int16 full scale.",
        "Median imputation and standardization are fit on training rows only.",
        "The peak threshold is chosen on training rows only. Logistic regression uses probability 0.5.",
        "Test rows are scored once and are not used to change features, the scaler, or the threshold.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
