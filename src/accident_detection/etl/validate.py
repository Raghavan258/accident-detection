"""Schema checks for one shard. Row rules live in the transformer."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from accident_detection.schema import schema_problems


def file_schema_errors(path: Path) -> list[str]:
    """Return schema problems, or one unreadable error. An empty list is valid."""

    try:
        schema = pq.read_schema(path)
    except (OSError, pa.ArrowException) as exc:
        return [f"unreadable: {type(exc).__name__}"]
    return schema_problems(schema)
