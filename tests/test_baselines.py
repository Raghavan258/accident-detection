"""The peak threshold is selected from training scores only."""

import numpy as np

from accident_detection.baselines import fit_logistic, fit_peak_baseline, predict_peak
from accident_detection.features import FEATURE_NAMES
from accident_detection.metrics import crash_f1


def test_lowest_threshold_with_best_train_f1_is_kept() -> None:
    scores = np.array([np.nan, 0.0, 1.0, 2.0, 3.0])
    labels = np.array([1, 0, 0, 1, 1])
    baseline = fit_peak_baseline(scores, labels)
    assert baseline.threshold_g == 2.0
    assert baseline.train_f1_crash == 1.0
    prediction = predict_peak(baseline, scores)
    assert prediction.tolist() == [0, 0, 0, 1, 1]
    assert crash_f1(labels[np.isfinite(scores)], prediction[np.isfinite(scores)]) == 1.0


def test_scaler_mean_uses_the_fitted_rows_only() -> None:
    column = FEATURE_NAMES.index("peak_abs_gyro_deg_s")
    train = np.zeros((2, len(FEATURE_NAMES)))
    train[:, column] = [0.0, 2.0]
    model = fit_logistic(train, np.array([0, 1]))
    assert model.named_steps["scaler"].mean_[column] == 1.0
