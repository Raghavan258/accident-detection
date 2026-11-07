"""Dataset checks used before training. Deepchecks is optional."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from accident_detection.features import DEFENSIBLE_FEATURE_NAMES
from accident_detection.mlops import MIN_VALIDATION_AP, MIN_VALIDATION_RECALL
from accident_detection.mlops.dataset import SPLIT_COLUMN

OUTLIER_MAD_Z = 8.0


def validate_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Check schema, labels, missing values, and train-fit outliers."""

    problems = _schema_problems(frame)
    missing = _missing(frame)
    outliers = _outlier_share(frame)
    deepchecks = _deepchecks(frame)
    return {
        "passed": not problems,
        "problems": problems,
        "missing_counts": missing,
        "outlier_share_by_feature": outliers,
        "deepchecks": deepchecks,
        "rows": int(len(frame)),
    }


def promotion_gate(validation: dict[str, object]) -> dict[str, object]:
    """Promote only from validation average precision and recall."""

    average_precision = validation.get("average_precision_crash")
    recall = validation.get("recall_crash")
    passed = _meets(average_precision, MIN_VALIDATION_AP) and _meets(recall, MIN_VALIDATION_RECALL)
    return {
        "passed": passed,
        "min_validation_average_precision": MIN_VALIDATION_AP,
        "min_validation_recall": MIN_VALIDATION_RECALL,
        "validation_average_precision": average_precision,
        "validation_recall": recall,
        "used_test_labels": False,
    }


def write_validation(path: Path, report: dict[str, object]) -> None:
    """Store a validation report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _schema_problems(frame: pd.DataFrame) -> list[str]:
    problems = []
    if SPLIT_COLUMN not in frame or set(frame[SPLIT_COLUMN].unique()) < {"train", "validation"}:
        problems.append("train and validation splits are required")
    if "label" not in frame or not set(frame["label"].dropna().unique()).issubset({0, 1, 2}):
        problems.append("labels must be 0, 1, or 2")
    missing_columns = [name for name in DEFENSIBLE_FEATURE_NAMES if name not in frame]
    if missing_columns:
        problems.append("missing features: " + ", ".join(missing_columns))
    return problems


def _missing(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.loc[:, list(DEFENSIBLE_FEATURE_NAMES)].isna().sum()
    return {name: int(counts[name]) for name in counts.index if int(counts[name]) > 0}


def _outlier_share(frame: pd.DataFrame) -> dict[str, float]:
    train = frame.loc[frame[SPLIT_COLUMN] == "train"]
    shares = {}
    for name in DEFENSIBLE_FEATURE_NAMES:
        shares[name] = _share(train[name].to_numpy(dtype=float), frame[name].to_numpy(dtype=float))
    return shares


def _share(reference: np.ndarray, current: np.ndarray) -> float:
    finite = reference[np.isfinite(reference)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) or 1e-9
    current_finite = current[np.isfinite(current)]
    if current_finite.size == 0:
        return 0.0
    z = np.abs(current_finite - median) / (1.4826 * mad)
    return float(np.mean(z > OUTLIER_MAD_Z))


def _meets(value: object, minimum: float) -> bool:
    return isinstance(value, (int, float)) and value is not None and float(value) >= minimum


def _deepchecks(frame: pd.DataFrame) -> dict[str, object]:
    try:
        from accident_detection.mlops.deepchecks_adapter import run_deepchecks

        return run_deepchecks(frame)
    except Exception as error:
        return {"available": False, "passed": False, "reason": f"{type(error).__name__}: {error}"}
