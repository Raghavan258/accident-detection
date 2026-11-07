"""Command line for the VZCrash ETL pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from accident_detection.etl.evaluate import evaluate_loaded_run
from accident_detection.etl.pipeline import EtlConfig, run_etl

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run extraction, validation, SQLite load, and the quality report."""

    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.input.is_dir():
        logger.error("Input directory does not exist: %s", args.input)
        return 2
    config = EtlConfig(args.input, args.db, args.output, args.dry_run, args.reset_db)
    try:
        summary = run_etl(config)
    except Exception:
        logger.exception("ETL run failed")
        return 1
    if args.evaluate and not args.dry_run and summary["status"] == "loaded":
        evaluate_loaded_run(args.db, args.output, str(summary["run_id"]))
    logger.info("Run %s status %s", summary["run_id"], summary["status"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load VZCrash Parquet shards into SQLite.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-db", action="store_true", help="Delete the SQLite file before this run.")
    parser.add_argument("--evaluate", action="store_true", help="Fit and score the existing baselines on the loaded run.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
