"""Validation-only model selection and the remaining official downloads."""

from __future__ import annotations

TRAIN_SHARDS: tuple[str, ...] = tuple(f"train-{index:05d}-of-00011.parquet" for index in range(11))
VALIDATION_SHARDS: tuple[str, ...] = tuple(
    f"validation-{index:05d}-of-00003.parquet" for index in range(3)
)
TEST_SHARDS: tuple[str, ...] = tuple(f"test-{index:05d}-of-00002.parquet" for index in range(2))
TRAIN_SHARD_MIB = 513
VALIDATION_SHARD_MIB = 370


def freeze_decision(validation_scores: dict[str, dict[str, object]]) -> dict[str, object]:
    """Choose a model from validation average precision only.

    The argument must not contain test metrics. Ties break by model name.
    """

    usable = {
        name: float(metrics["average_precision_crash"])
        for name, metrics in validation_scores.items()
        if metrics.get("average_precision_crash") is not None
    }
    if not usable:
        return {
            "selected_model": None,
            "status": "blocked",
            "reason": "validation split is absent or has no defined average precision",
        }
    selected = sorted(usable, key=lambda name: (-usable[name], name))[0]
    return {
        "selected_model": selected,
        "status": "frozen",
        "reason": "highest validation average precision; name breaks ties",
        "validation_average_precision": usable,
    }


def download_plan(present_names: set[str]) -> dict[str, object]:
    """List missing train and validation shards without repeating the test download."""

    missing_train = [name for name in TRAIN_SHARDS if name not in present_names]
    missing_validation = [name for name in VALIDATION_SHARDS if name not in present_names]
    to_download = missing_train + missing_validation
    command = _download_command(to_download)
    return {
        "missing_train": missing_train,
        "missing_validation": missing_validation,
        "already_present": sorted(present_names),
        "command": command,
        "approximate_gib": _approximate_gib(len(missing_train), len(missing_validation)),
        "rules": _download_rules(),
    }


def _download_command(names: list[str]) -> str:
    if not names:
        return ""
    files = " ".join(f"data/{name}" for name in names)
    return (
        "hf download vzc-research-chapter/VZCrash "
        f"{files} --repo-type dataset --local-dir data/VZCrash"
    )


def _approximate_gib(train_count: int, validation_count: int) -> float:
    mebibytes = train_count * TRAIN_SHARD_MIB + validation_count * VALIDATION_SHARD_MIB
    return round(mebibytes / 1024, 1)


def _download_rules() -> list[str]:
    return [
        "Leave every file name unchanged so train, validation, and test stay in their official shards.",
        "Do not copy a training row into validation or test.",
        "Do not download the test shards again.",
        "Fit on the training shards, select the model and threshold on validation, then score test once.",
    ]
