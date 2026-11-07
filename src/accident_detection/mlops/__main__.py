"""Command line for the local IVERAS lifecycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from accident_detection.mlops.dataset import export_feature_table
from accident_detection.mlops.lifecycle import run_lifecycle


def main() -> None:
    """Export processed features when asked, then run the lifecycle."""

    args = _parser().parse_args()
    frame = _frame(args)
    result = run_lifecycle(frame, args.tracking_uri, Path(args.reports), args.run_name)
    print(json.dumps({key: result[key] for key in result if key != "drift"}, indent=2, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and register the existing crash detector.")
    parser.add_argument("--db", type=Path, default=Path("data/processed/accident_detection.db"))
    parser.add_argument("--features", type=Path, default=Path("datasets/processed/features.parquet"))
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--tracking-uri", default="sqlite:///mlops/mlflow.db")
    parser.add_argument("--reports", default="reports/mlops")
    parser.add_argument("--run-name", default="defensible-retrain")
    return parser


def _frame(args) -> pd.DataFrame:
    if args.export or not args.features.exists():
        export_feature_table(args.db, args.features)
    return pd.read_parquet(args.features)


if __name__ == "__main__":
    main()
