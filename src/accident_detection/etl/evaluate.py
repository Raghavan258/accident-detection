"""Score the existing baselines on one loaded ETL run. Test labels are not used to choose."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from accident_detection.baselines import fit_logistic, fit_peak_baseline, logistic_scores
from accident_detection.etl.ml_input import load_splits
from accident_detection.features import ACCEL_FEATURE_NAMES, DEFENSIBLE_FEATURE_NAMES, select_columns
from accident_detection.phase4 import present_metrics
from accident_detection.selection import validation_choice

PEAK = "peak_baseline"
ACCEL = "accel_logistic"
DEFINED = "defensible_logistic"


def evaluate_loaded_run(db_path: Path, output_dir: Path, run_id: str) -> dict[str, object]:
    """Fit on train, select on validation, then score test once."""

    splits = load_splits(db_path, run_id)
    result = _blocked_or_fit(splits, run_id)
    _write(output_dir, result)
    return result


def _blocked_or_fit(splits: dict[str, dict[str, np.ndarray]], run_id: str) -> dict[str, object]:
    result: dict[str, object] = {"run_id": run_id, "rows": _sizes(splits), "historical_phase4_used": False}
    if min(splits["train"]["y"].size, splits["validation"]["y"].size) == 0:
        result["status"] = "blocked"
        result["reason"] = "train or validation rows are missing"
        return result
    models = _fit(splits["train"])
    scores = {name: _scores(splits["validation"], models[name]) for name in models}
    decision = validation_choice(scores, splits["validation"]["y"])
    result["decision"] = decision
    result["validation"] = _operating_points(splits["validation"], models)
    result["test"] = _test_once(splits["test"], models, decision)
    result["status"] = decision["status"]
    return result


def _fit(train: dict[str, np.ndarray]) -> dict[str, tuple[str, object]]:
    labels = train["y"]
    return {
        PEAK: ("peak", None),
        ACCEL: ("logistic", fit_logistic(select_columns(train["features"], ACCEL_FEATURE_NAMES), labels)),
        DEFINED: ("logistic", fit_logistic(select_columns(train["features"], DEFENSIBLE_FEATURE_NAMES), labels)),
    }


def _scores(part: dict[str, np.ndarray], spec: tuple[str, object]) -> np.ndarray:
    kind, model = spec
    if kind == "peak":
        return part["peak"]
    names = _feature_names(model)
    return logistic_scores(model, select_columns(part["features"], names))


def _feature_names(model) -> tuple[str, ...]:
    width = int(model.named_steps["clf"].coef_.shape[1])
    if width == len(ACCEL_FEATURE_NAMES):
        return ACCEL_FEATURE_NAMES
    return DEFENSIBLE_FEATURE_NAMES


def _operating_points(part, models) -> dict[str, object]:
    points = {}
    for name, spec in models.items():
        scores = _scores(part, spec)
        threshold = fit_peak_baseline(scores, part["y"]).threshold_g
        points[name] = present_metrics(part["y"], scores, threshold, "validation_operating_point")
    return points


def _test_once(part, models, decision: dict[str, object]) -> dict[str, object]:
    if decision.get("selected_model") is None or part["y"].size == 0:
        return {"status": "not_run"}
    spec = models[str(decision["selected_model"])]
    scores = _scores(part, spec)
    metrics = present_metrics(part["y"], scores, float(decision["threshold"]), "etl_test_scored_once")
    return {"status": "scored_once", "metrics": metrics}


def _sizes(splits: dict[str, dict[str, np.ndarray]]) -> dict[str, int]:
    return {name: int(part["y"].size) for name, part in splits.items()}


def _write(output_dir: Path, result: dict[str, object]) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "etl_evaluation.json").write_text(json.dumps(result, indent=2, default=_json), encoding="utf-8")


def _json(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)
