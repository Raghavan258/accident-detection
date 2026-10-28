"""Train-only baselines. Thresholds and scalers are not fit on test rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000
LOGISTIC_RANDOM_STATE = 0
PROBABILITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class PeakBaseline:
    """Horizontal-acceleration peak with a threshold chosen on training rows."""

    threshold_g: float
    train_f1_crash: float


def fit_peak_baseline(scores: np.ndarray, y_crash: np.ndarray) -> PeakBaseline:
    """Pick the lowest training score whose crash F1 is best.

    Ties keep the lower threshold. Non-finite training scores are ignored here
    and remain in the evaluation set.
    """

    usable_scores, labels = _finite_pairs(scores, y_crash)
    thresholds, f1_scores = _f1_by_threshold(usable_scores, labels)
    best = int(np.argmax(f1_scores))
    return PeakBaseline(float(thresholds[best]), float(f1_scores[best]))


def _finite_pairs(scores: np.ndarray, y_crash: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise ValueError("peak baseline has no finite training scores")
    return np.asarray(scores)[finite], np.asarray(y_crash)[finite]


def _f1_by_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    crash = labels[order] == 1
    suffix_crash = np.cumsum(crash[::-1])[::-1]
    suffix_count = np.arange(len(labels), 0, -1)
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    true_positive = suffix_crash[starts]
    false_positive = suffix_count[starts] - true_positive
    false_negative = int(np.sum(crash)) - true_positive
    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, true_positive + false_negative)
    return sorted_scores[starts], _divide(2 * precision * recall, precision + recall)


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros(len(numerator), dtype=float), where=denominator > 0)


def predict_peak(baseline: PeakBaseline, scores: np.ndarray) -> np.ndarray:
    """Predict crash when the finite peak reaches the frozen training threshold."""

    prediction = np.zeros(len(scores), dtype=np.int8)
    finite = np.isfinite(scores)
    prediction[finite] = (scores[finite] >= baseline.threshold_g).astype(np.int8)
    return prediction


def fit_logistic(features: np.ndarray, y_crash: np.ndarray) -> Pipeline:
    """Fit median imputation, standardization, and balanced logistic regression."""

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=LOGISTIC_C,
                    class_weight="balanced",
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=LOGISTIC_RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(features, np.asarray(y_crash))
    return pipeline


def logistic_scores(pipeline: Pipeline, features: np.ndarray) -> np.ndarray:
    """Return the predicted probability of the crash class."""

    classes = list(pipeline.named_steps["clf"].classes_)
    positive = classes.index(1)
    return pipeline.predict_proba(features)[:, positive]


def predict_logistic(pipeline: Pipeline, features: np.ndarray) -> np.ndarray:
    """Apply the fixed 0.5 probability threshold."""

    return predict_at_threshold(logistic_scores(pipeline, features), PROBABILITY_THRESHOLD)


def predict_at_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Predict crash when a finite score reaches a frozen threshold."""

    prediction = np.zeros(len(scores), dtype=np.int8)
    finite = np.isfinite(scores)
    prediction[finite] = (np.asarray(scores)[finite] >= threshold).astype(np.int8)
    return prediction
