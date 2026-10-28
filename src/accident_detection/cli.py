"""Command-line entry point for the VZCrash audit."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from accident_detection.audit import audit_directory
from accident_detection.report import write_outputs
from accident_detection.schema import signals_readable

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return the audit command parser."""

    parser = argparse.ArgumentParser(description="Audit VZCrash parquet shards without training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/VZCrash/data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/data_audit"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the audit and return 0 when every file could be decoded."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.data_dir.is_dir():
        logger.error("Data directory does not exist: %s", args.data_dir)
        return 2
    report = audit_directory(args.data_dir)
    recorded = sys.argv[1:] if argv is None else argv
    command = "python -m accident_detection.cli " + " ".join(recorded)
    written = write_outputs(report, args.report_dir, args.figure_dir, command=command)
    for path in written:
        logger.info("Wrote %s", path)
    if not report.files:
        logger.error("No parquet files found in %s", args.data_dir)
        return 1
    broken = [
        scan.file_name
        for scan in report.files
        if scan.read_error or not signals_readable(scan.schema_problems)
    ]
    if broken:
        logger.error("Unreadable or invalid schema: %s", ", ".join(broken))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
