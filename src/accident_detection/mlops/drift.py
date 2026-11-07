"""Compare a reference split with a current split and write an HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from accident_detection.features import DEFENSIBLE_FEATURE_NAMES
from accident_detection.mlops import DRIFT_SHARE_THRESHOLD
from accident_detection.mlops.dataset import SPLIT_COLUMN


def monitor_drift(
    frame: pd.DataFrame,
    output_html: Path,
    *,
    reference_split: str = "train",
    current_split: str = "validation",
    share_threshold: float = DRIFT_SHARE_THRESHOLD,
) -> dict[str, object]:
    """Flag features whose mean shift exceeds the configured threshold."""

    reference = _columns(frame, reference_split)
    current = _columns(frame, current_split)
    shifts = _shifts(reference, current)
    drifted = [name for name, item in shifts.items() if item["share"] >= share_threshold]
    report = {
        "reference_split": reference_split,
        "current_split": current_split,
        "share_threshold": share_threshold,
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
        "features": shifts,
        "evidently": _evidently(reference, current, output_html),
    }
    output_html.parent.mkdir(parents=True, exist_ok=True)
    if not report["evidently"]["wrote_html"]:
        output_html.write_text(_fallback_html(report), encoding="utf-8")
        report["html"] = str(output_html)
    else:
        report["html"] = str(output_html)
    output_html.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _columns(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = frame.loc[frame[SPLIT_COLUMN] == split, list(DEFENSIBLE_FEATURE_NAMES)]
    if rows.empty:
        raise ValueError(f"split {split} has no rows")
    return rows.reset_index(drop=True)


def _shifts(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, dict[str, float]]:
    shifts = {}
    for name in DEFENSIBLE_FEATURE_NAMES:
        left = reference[name].to_numpy(dtype=float)
        right = current[name].to_numpy(dtype=float)
        scale = float(np.nanstd(left)) or 1.0
        share = abs(float(np.nanmean(right) - np.nanmean(left))) / scale
        shifts[name] = {"mean_shift_in_reference_std": share, "share": share}
    return shifts


def _evidently(reference: pd.DataFrame, current: pd.DataFrame, output_html: Path) -> dict[str, object]:
    try:
        _patch_pandas_dtype_helper()
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_html))
        return {"available": True, "wrote_html": True}
    except Exception as error:
        return {"available": False, "wrote_html": False, "reason": f"{type(error).__name__}: {error}"}


def _patch_pandas_dtype_helper() -> None:
    """Evidently 0.7.23 imports a helper pandas 3.1 removed. Drift does not call it."""

    import pandas.core.dtypes.common as common

    if hasattr(common, "infer_dtype_from_object"):
        return

    def infer_dtype_from_object(dtype):
        if isinstance(dtype, np.dtype):
            return dtype.type
        if isinstance(dtype, type):
            return dtype
        return np.dtype(dtype).type

    common.infer_dtype_from_object = infer_dtype_from_object


def _fallback_html(report: dict[str, object]) -> str:
    rows = "".join(
        f"<tr><td>{name}</td><td>{item['share']:.4f}</td></tr>"
        for name, item in report["features"].items()
    )
    return (
        "<html><body><h1>Drift report</h1>"
        f"<p>Evidently was not used. Threshold {report['share_threshold']}.</p>"
        f"<table><tr><th>Feature</th><th>Shift</th></tr>{rows}</table></body></html>"
    )
