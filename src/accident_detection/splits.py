"""Official VZCrash split rules and the limits of a partial download."""

from __future__ import annotations

from accident_detection.constants import OFFICIAL_SHARD_COUNTS

OFFICIAL_SPLIT_FRACTIONS: dict[str, float] = {
    "train": 0.72,
    "validation": 0.14,
    "test": 0.14,
}
PAPER_REPORTED_CRASHES = 31090


def split_limitations(
    train_files: int,
    validation_files: int,
    test_files: int,
    training_crashes: int | None = None,
) -> list[str]:
    """Describe why this checkout cannot reproduce the paper benchmark."""

    notes = [
        (
            "The paper uses vehicle-disjoint shares of 72% train, 14% validation, "
            "and 14% test, stratified by crash and vehicle size."
        ),
        (
            "Official parquet shard counts are "
            f"train {OFFICIAL_SHARD_COUNTS['train']}, "
            f"validation {OFFICIAL_SHARD_COUNTS['validation']}, "
            f"test {OFFICIAL_SHARD_COUNTS['test']}."
        ),
        (
            f"This run found train {train_files}, validation {validation_files}, "
            f"test {test_files}."
        ),
        "The released columns do not include a vehicle id, so the vehicle split cannot be rebuilt inside one shard.",
        "Validation files are left unused. They are not fit and they are not added to the test set.",
        (
            "The paper reports 31,090 verified crashes in the full release. "
            "A single training shard is not one eleventh of that crash count."
        ),
    ]
    if training_crashes is not None:
        notes.append(f"The training files in this run contain {training_crashes} crash events.")
    notes.append("Metrics below are a first pass, not the ITSC benchmark and not a tuned final model.")
    return notes
