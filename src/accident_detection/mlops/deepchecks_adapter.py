"""Run Deepchecks when the package imports. Callers treat failure as unavailable."""

from __future__ import annotations

import pandas as pd

from accident_detection.features import DEFENSIBLE_FEATURE_NAMES
from accident_detection.mlops.dataset import SPLIT_COLUMN


def run_deepchecks(frame: pd.DataFrame) -> dict[str, object]:
    """Validate train integrity and the train/test feature schema."""

    _register_removed_scorer()
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity, train_test_validation

    train, test = _sets(frame)
    integrity = data_integrity().run(train)
    comparison = train_test_validation().run(train, test)
    return {
        "available": True,
        "passed": _passed(integrity) and _passed(comparison),
        "integrity_passed": _passed(integrity),
        "train_test_passed": _passed(comparison),
        "integrity_not_passed": _names(integrity),
        "train_test_not_passed": _names(comparison),
    }


def _sets(frame: pd.DataFrame):
    from deepchecks.tabular import Dataset

    columns = list(DEFENSIBLE_FEATURE_NAMES)
    train = frame.loc[frame[SPLIT_COLUMN] == "train", columns + ["label"]]
    held = frame.loc[frame[SPLIT_COLUMN] != "train", columns + ["label"]]
    return Dataset(train, label="label", cat_features=[]), Dataset(held, label="label", cat_features=[])


def _register_removed_scorer() -> None:
    """Restore names Deepchecks 0.19 still imports on NumPy 2 and scikit-learn 1.9."""

    import numpy as np

    np.Inf = np.inf
    np.NaN = np.nan
    _register_max_error()


def _register_max_error() -> None:
    """scikit-learn 1.9 dropped the max_error scorer name that Deepchecks 0.19 imports."""

    from sklearn.metrics import make_scorer, max_error
    from sklearn.metrics._scorer import _SCORERS

    if "max_error" not in _SCORERS:
        _SCORERS["max_error"] = make_scorer(max_error, greater_is_better=False)


def _names(result) -> list[str]:
    if not hasattr(result, "get_not_passed_checks"):
        return []
    names = []
    for item in result.get_not_passed_checks():
        check = getattr(item, "check", None)
        names.append(check.name() if hasattr(check, "name") else type(check).__name__)
    return names


def _passed(result) -> bool:
    if not hasattr(result, "passed"):
        return True
    return bool(result.passed())
