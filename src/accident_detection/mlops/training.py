"""Train the existing logistic candidates and record one MLflow run."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from accident_detection.baselines import LOGISTIC_C, LOGISTIC_MAX_ITER, LOGISTIC_RANDOM_STATE, fit_logistic, logistic_scores
from accident_detection.etl.evaluate import ACCEL, DEFINED, PEAK
from accident_detection.features import ACCEL_FEATURE_NAMES, DEFENSIBLE_FEATURE_NAMES, select_columns
from accident_detection.mlops import EXPERIMENT_NAME, MODEL_NAME
from accident_detection.mlops.dataset import splits_from_frame
from accident_detection.mlops.quality import promotion_gate
from accident_detection.phase4 import present_metrics
from accident_detection.selection import THRESHOLD_CRITERION, validation_choice


def train_frame(frame: pd.DataFrame, tracking_uri: str, run_name: str = "defensible-retrain") -> dict[str, object]:
    """Fit on train, choose on validation, then score test once."""

    splits = splits_from_frame(frame)
    models = _fit(splits["train"])
    scores = {name: _scores(splits["validation"], models[name]) for name in models}
    decision = validation_choice(scores, splits["validation"]["y"])
    selected = str(decision["selected_model"])
    threshold = float(decision["threshold"])
    validation = present_metrics(splits["validation"]["y"], scores[selected], threshold, "validation_operating_point")
    gate = promotion_gate(validation)
    test = _test_once(splits["test"], models[selected], threshold)
    run_id, model_uri = _log(tracking_uri, run_name, models[selected][1], decision, validation, gate, test)
    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "selected_model": selected,
        "threshold": threshold,
        "validation": validation,
        "gate": gate,
        "test": test,
        "model": models[selected][1],
        "feature_names": _names(models[selected][1]),
    }


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
    return logistic_scores(model, select_columns(part["features"], _names(model)))


def _names(model) -> tuple[str, ...]:
    if model is None:
        return ("peak_horizontal_g",)
    width = int(model.named_steps["clf"].coef_.shape[1])
    if width == len(ACCEL_FEATURE_NAMES):
        return ACCEL_FEATURE_NAMES
    return DEFENSIBLE_FEATURE_NAMES


def _test_once(part, spec, threshold: float) -> dict[str, object]:
    if part["y"].size == 0:
        return {"status": "not_run"}
    metrics = present_metrics(part["y"], _scores(part, spec), threshold, "scored_once_after_freeze")
    return {"status": "scored_once", "metrics": metrics}


def _log(uri, run_name, model, decision, validation, gate, test) -> tuple[str, str]:
    _prepare_experiment(uri)
    with mlflow.start_run(run_name=run_name) as run:
        _params(decision, gate)
        _metrics("validation", validation)
        if test.get("status") == "scored_once":
            _metrics("test", test["metrics"])
        mlflow.log_dict(_public(decision, gate, validation, test), "decision.json")
        model_uri = _store_model(model)
        mlflow.set_tags({"model_name": MODEL_NAME, "gate_passed": str(bool(gate["passed"])).lower()})
        return run.info.run_id, model_uri


def _prepare_experiment(uri: str) -> None:
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(uri)
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=_artifact_root(uri).as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def _artifact_root(uri: str) -> Path:
    if uri.startswith("sqlite:///"):
        database = Path(uri.removeprefix("sqlite:///"))
        if not database.is_absolute():
            database = Path.cwd() / database
        return (database.parent / "artifacts").resolve()
    return (Path("mlops") / "artifacts").resolve()


def _store_model(model) -> str:
    if model is None:
        raise ValueError("the selected candidate has no pipeline to register")
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
    )
    return info.model_uri


def _params(decision: dict[str, object], gate: dict[str, object]) -> None:
    mlflow.log_params(
        {
            "C": LOGISTIC_C,
            "max_iter": LOGISTIC_MAX_ITER,
            "random_state": LOGISTIC_RANDOM_STATE,
            "class_weight": "balanced",
            "selected_model": decision["selected_model"],
            "threshold": decision["threshold"],
            "threshold_criterion": THRESHOLD_CRITERION,
            "min_validation_average_precision": gate["min_validation_average_precision"],
            "min_validation_recall": gate["min_validation_recall"],
        }
    )


def _metrics(prefix: str, metrics: dict[str, object]) -> None:
    numeric = {}
    for key in ("precision_crash", "recall_crash", "f1_crash", "average_precision_crash", "false_positives", "false_negatives"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            numeric[f"{prefix}_{key}"] = float(value)
    mlflow.log_metrics(numeric)
    mlflow.log_dict(metrics["confusion_matrix"], f"{prefix}_confusion_matrix.json")


def _public(decision, gate, validation, test) -> dict[str, object]:
    return {
        "selected_model": decision["selected_model"],
        "threshold": decision["threshold"],
        "feature_names": list(DEFENSIBLE_FEATURE_NAMES if decision["selected_model"] == DEFINED else ACCEL_FEATURE_NAMES),
        "gate": gate,
        "validation_confusion_matrix": validation["confusion_matrix"],
        "test_status": test.get("status"),
        "test_metrics_role": "scored after the validation decision; not used for promotion",
    }
