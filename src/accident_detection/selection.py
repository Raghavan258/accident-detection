"""Choose one model and one threshold from validation scores only."""

from __future__ import annotations

import numpy as np

from accident_detection.baselines import fit_peak_baseline
from accident_detection.metrics import classification_metrics
from accident_detection.protocol import freeze_decision

EXPERIMENT_VERSION = "phase4-v1"
MODEL_CRITERION = "highest validation average precision; model name breaks ties"
THRESHOLD_CRITERION = (
    "maximum crash F1 on the selected model's validation scores; the lowest score wins ties"
)
USE_CASE_NOTE = (
    "No missed-crash versus false-alarm cost ratio was specified. Average precision ranks "
    "crash scores without an operating point. Crash F1 is the documented balanced threshold. "
    "Precision, recall, false positives, and false negatives are reported beside that choice."
)


def validation_choice(
    scores_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, object]:
    """Freeze a model and threshold. This function does not accept test scores."""

    average_precision = {name: _average_precision(labels, scores) for name, scores in scores_by_model.items()}
    decision = freeze_decision(
        {name: {"average_precision_crash": value} for name, value in average_precision.items()}
    )
    threshold = _threshold(decision, scores_by_model, labels)
    return {
        **decision,
        "threshold": threshold,
        "model_criterion": MODEL_CRITERION,
        "threshold_criterion": THRESHOLD_CRITERION,
        "use_case_note": USE_CASE_NOTE,
        "experiment_version": EXPERIMENT_VERSION,
    }


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    unused = np.zeros(len(labels), dtype=np.int8)
    value = classification_metrics(labels, unused, scores)["average_precision_crash"]
    if value is None:
        return None
    return float(value)


def _threshold(decision: dict[str, object], scores_by_model: dict[str, np.ndarray], labels: np.ndarray) -> float | None:
    selected = decision["selected_model"]
    if selected is None:
        return None
    return fit_peak_baseline(scores_by_model[str(selected)], labels).threshold_g
