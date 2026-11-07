"""Discover official Parquet shards. CSV is not a VZCrash source format."""

from __future__ import annotations

from pathlib import Path


def discover_shards(input_dir: Path) -> list[Path]:
    """Return Parquet files in name order. The directory is only read."""

    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    return sorted(path for path in root.glob("*.parquet") if path.is_file())
