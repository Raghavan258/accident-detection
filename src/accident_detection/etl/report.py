"""Write JSON, CSV, and plots from a finished ETL summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from accident_detection.constants import LABEL_NAMES


def write_quality_report(output_dir: Path, summary: dict[str, object]) -> None:
    """Write the run summary. Plot data comes only from this summary."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "etl_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_tables(root, summary)
    _write_plots(root, summary)


def _write_tables(root: Path, summary: dict[str, object]) -> None:
    pd.DataFrame(summary["shards"]).to_csv(root / "shards.csv", index=False)
    pd.DataFrame(summary["metrics"]).to_csv(root / "quality_metrics.csv", index=False)


def _write_plots(root: Path, summary: dict[str, object]) -> None:
    plt = _pyplot()
    _plot_shard_rows(plt, root, summary["shards"])
    _plot_labels(plt, root, summary["metrics"])
    _plot_missing(plt, root, summary["metrics"])


def _plot_shard_rows(plt, root: Path, shards: list[dict[str, object]]) -> None:
    if not shards:
        return
    figure, axis = plt.subplots(figsize=(10, 4))
    names = [str(shard["file_name"]).replace(".parquet", "") for shard in shards]
    accepted = [int(shard["accepted_rows"]) for shard in shards]
    rejected = [int(shard["rejected_rows"]) for shard in shards]
    axis.bar(names, accepted, label="accepted")
    axis.bar(names, rejected, bottom=accepted, label="rejected")
    axis.tick_params(axis="x", labelrotation=75, labelsize=7)
    axis.set_ylabel("Rows")
    axis.set_title("Accepted and rejected rows by shard")
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "shard_rows.png", dpi=120)
    plt.close(figure)


def _plot_labels(plt, root: Path, metrics: list[dict[str, object]]) -> None:
    counts = [_metric_value(metrics, "all", f"label_{name}") for name in LABEL_NAMES.values()]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(list(LABEL_NAMES.values()), counts)
    axis.set_ylabel("Accepted rows")
    axis.set_title("Class counts in accepted rows")
    figure.tight_layout()
    figure.savefig(root / "class_counts.png", dpi=120)
    plt.close(figure)


def _plot_missing(plt, root: Path, metrics: list[dict[str, object]]) -> None:
    missing = [item for item in metrics if str(item["metric"]).startswith("missing_") and item["scope"] == "all"]
    if not missing:
        return
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar([str(item["metric"]).removeprefix("missing_") for item in missing], [int(item["value"]) for item in missing])
    axis.tick_params(axis="x", labelrotation=75, labelsize=7)
    axis.set_ylabel("Null accepted values")
    axis.set_title("Missing values among accepted rows")
    figure.tight_layout()
    figure.savefig(root / "missing_values.png", dpi=120)
    plt.close(figure)


def _metric_value(metrics: list[dict[str, object]], scope: str, name: str) -> int:
    for item in metrics:
        if item["scope"] == scope and item["metric"] == name:
            return int(item["value"])
    return 0


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
