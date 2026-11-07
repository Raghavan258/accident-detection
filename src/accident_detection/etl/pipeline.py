"""Orchestrate shard discovery, validation, loading, and the quality report."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from accident_detection.constants import REQUIRED_COLUMNS
from accident_detection.etl.extract import discover_shards
from accident_detection.etl.load import commit_run, init_db, record_failure, reset_database
from accident_detection.etl.profile import (
    DUPLICATE_POLICY,
    MISSING_POLICY,
    REJECTION_POLICY,
    build_metrics,
    warnings_for,
)
from accident_detection.etl.report import write_quality_report
from accident_detection.etl.transform import records_from_table
from accident_detection.etl.validate import file_schema_errors
from accident_detection.schema import split_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EtlConfig:
    """Paths and flags for one pipeline invocation."""

    input_dir: Path
    db_path: Path
    output_dir: Path
    dry_run: bool = False
    reset_db: bool = False
    run_id: str | None = None


def run_etl(config: EtlConfig) -> dict[str, object]:
    """Read every Parquet shard, then commit one SQLite run or stop on dry-run."""

    started = time.perf_counter()
    stamp = _now()
    files = discover_shards(config.input_dir)
    logger.info("Discovered %s parquet shards", len(files))
    shards = [_process_file(path) for path in files]
    metrics = build_metrics(shards)
    summary = _summary(config, shards, metrics, stamp)
    try:
        if config.dry_run:
            summary["database_status"] = "not_written"
            summary["status"] = "dry_run"
        else:
            _store(config, summary, shards)
    except Exception:
        _finish(summary, started)
        write_quality_report(config.output_dir, summary)
        raise
    _finish(summary, started)
    write_quality_report(config.output_dir, summary)
    return summary


def _process_file(path: Path) -> dict[str, object]:
    logger.info("Reading %s", path.name)
    shard = _empty_shard(path)
    errors = file_schema_errors(path)
    if errors:
        return _fail(shard, "; ".join(errors))
    try:
        accepted, rejected, input_rows = _read_row_groups(path)
    except (OSError, pa.ArrowException) as exc:
        return _fail(shard, type(exc).__name__)
    shard["accepted"] = accepted
    shard["rejected"] = rejected
    shard["input_rows"] = input_rows
    shard["accepted_rows"] = len(accepted)
    shard["rejected_rows"] = len(rejected)
    shard["status"] = "processed"
    return shard


def _read_row_groups(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    parquet = pq.ParquetFile(path)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    offset = 0
    for index in range(parquet.num_row_groups):
        table = parquet.read_row_group(index, columns=list(REQUIRED_COLUMNS))
        part_accepted, part_rejected = records_from_table(table, path.name, offset)
        accepted.extend(part_accepted)
        rejected.extend(part_rejected)
        offset += table.num_rows
    return accepted, rejected, offset


def _store(config: EtlConfig, summary: dict[str, object], shards: list[dict[str, object]]) -> None:
    if config.reset_db:
        reset_database(config.db_path)
    init_db(config.db_path)
    run = _run_row(summary, "loaded", None)
    try:
        commit_run(config.db_path, run, shards)
    except Exception as exc:
        record_failure(config.db_path, _run_row(summary, "failed", type(exc).__name__))
        summary["database_status"] = "rolled_back"
        summary["status"] = "failed"
        raise
    summary["database_status"] = "committed"


def _summary(config: EtlConfig, shards, metrics, stamp: str) -> dict[str, object]:
    return {
        "run_id": config.run_id or str(uuid.uuid4()),
        "started_at": stamp,
        "finished_at": _now(),
        "status": "loaded" if not config.dry_run else "dry_run",
        "input_dir": str(config.input_dir),
        "duration_seconds": 0.0,
        "files_discovered": len(shards),
        "files_processed": sum(shard["status"] == "processed" for shard in shards),
        "files_failed": sum(shard["status"] == "failed" for shard in shards),
        "input_rows": sum(int(shard["input_rows"]) for shard in shards),
        "accepted_rows": sum(int(shard["accepted_rows"]) for shard in shards),
        "rejected_rows": sum(int(shard["rejected_rows"]) for shard in shards),
        "shards": [_public_shard(shard) for shard in shards],
        "metrics": metrics,
        "warnings": warnings_for(shards),
        "duplicate_policy": DUPLICATE_POLICY,
        "missing_value_policy": MISSING_POLICY,
        "rejection_policy": REJECTION_POLICY,
        "rows_removed_without_reason": 0,
    }


def _finish(summary: dict[str, object], started: float) -> None:
    summary["finished_at"] = _now()
    summary["duration_seconds"] = round(time.perf_counter() - started, 3)


def _public_shard(shard: dict[str, object]) -> dict[str, object]:
    return {
        "file_name": shard["file_name"],
        "split_name": shard["split_name"],
        "byte_size": shard["byte_size"],
        "input_rows": shard["input_rows"],
        "accepted_rows": shard["accepted_rows"],
        "rejected_rows": shard["rejected_rows"],
        "status": shard["status"],
        "error_message": shard.get("error_message"),
    }


def _empty_shard(path: Path) -> dict[str, object]:
    return {
        "file_name": path.name,
        "split_name": split_name(path),
        "byte_size": path.stat().st_size,
        "input_rows": 0,
        "accepted_rows": 0,
        "rejected_rows": 0,
        "status": "processed",
        "error_message": None,
        "accepted": [],
        "rejected": [],
    }


def _fail(shard: dict[str, object], message: str) -> dict[str, object]:
    shard["status"] = "failed"
    shard["error_message"] = message
    return shard


def _run_row(summary: dict[str, object], status: str, error_message: str | None) -> dict[str, object]:
    return {
        "run_id": summary["run_id"],
        "started_at": summary["started_at"],
        "finished_at": summary["finished_at"],
        "status": status,
        "input_dir": summary["input_dir"],
        "error_message": error_message,
        "metrics": summary["metrics"],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
