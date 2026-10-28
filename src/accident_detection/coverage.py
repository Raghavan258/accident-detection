"""Official shard inventory. Missing files are reported, not invented."""

from __future__ import annotations

from pathlib import Path

from accident_detection.protocol import TEST_SHARDS, TRAIN_SHARDS, VALIDATION_SHARDS, download_plan
from accident_detection.schema import split_name


def shard_inventory(data_dir: Path | str) -> dict[str, object]:
    """Compare the directory with the official train, validation, and test names."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {root}")
    files = [path for path in sorted(root.glob("*.parquet")) if path.is_file()]
    present = {path.name for path in files}
    missing_train = [name for name in TRAIN_SHARDS if name not in present]
    missing_validation = [name for name in VALIDATION_SHARDS if name not in present]
    missing_test = [name for name in TEST_SHARDS if name not in present]
    return {
        "files": [_file_record(path) for path in files],
        "missing_train": missing_train,
        "missing_validation": missing_validation,
        "missing_test": missing_test,
        "ready_to_select": not missing_train and not missing_validation,
        "ready_to_test": not missing_test,
        "download_plan": download_plan(present),
        "paths": {
            "train": [path for path in files if split_name(path) == "train" and path.name in TRAIN_SHARDS],
            "validation": [path for path in files if split_name(path) == "validation" and path.name in VALIDATION_SHARDS],
            "test": [path for path in files if split_name(path) == "test" and path.name in TEST_SHARDS],
        },
    }


def _file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "file_name": path.name,
        "split": split_name(path),
        "bytes": int(stat.st_size),
        "hardlink_count": int(stat.st_nlink),
    }
