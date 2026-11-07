"""Quality counts for one ETL run. Nothing is resampled or dropped here."""

from __future__ import annotations

from collections import Counter

from accident_detection.constants import ALLOWED_VEHICLE_SIZES, LABEL_NAMES
from accident_detection.features import FEATURE_NAMES

DUPLICATE_POLICY = (
    "A duplicate is a repeated event_id. Within-file and cross-file repeats are counted "
    "and kept. Equal sensor summaries with different event_ids are not treated as duplicates."
)
MISSING_POLICY = (
    "Null labels and null event ids are rejected. Other missing values stay null. "
    "The training imputer is not applied in this pipeline."
)
REJECTION_POLICY = (
    "Rejected rows are stored with a reason and are excluded from sensor_events. "
    "Valid rows, including extreme gyroscope values and unusual lengths, are kept."
)


def build_metrics(shards: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate accepted rows, rejections, and event-id duplicates."""

    accepted = [row for shard in shards for row in shard["accepted"]]
    rejected = [row for shard in shards for row in shard["rejected"]]
    metrics = [_metric("all", "input_rows", _inputs(shards), None)]
    metrics.append(_metric("all", "accepted_rows", len(accepted), None))
    metrics.append(_metric("all", "rejected_rows", len(rejected), None))
    metrics.extend(_label_metrics(shards, accepted))
    metrics.extend(_missing_metrics(accepted))
    metrics.extend(_duplicate_metrics(accepted))
    metrics.extend(_reason_metrics(rejected))
    metrics.append(_metric("all", "nonfinite_feature_values", _nonfinite_features(accepted), None))
    metrics.append(_metric("all", "unexpected_vehiclesize", _unexpected_sizes(accepted), None))
    return metrics


def warnings_for(shards: list[dict[str, object]]) -> list[str]:
    """Describe failed files and shards whose accepted rows contain one class."""

    notes = [f"{shard['file_name']}: {shard['error_message']}" for shard in shards if shard["status"] == "failed"]
    for shard in shards:
        labels = {int(row["label"]) for row in shard["accepted"]}
        if shard["accepted"] and labels == {0}:
            notes.append(f"{shard['file_name']} accepted rows are all crashes")
        if shard["accepted"] and 0 not in labels:
            notes.append(f"{shard['file_name']} accepted rows contain no crashes")
    return notes


def _label_metrics(shards: list[dict[str, object]], accepted: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = []
    for label, name in LABEL_NAMES.items():
        metrics.append(_metric("all", f"label_{name}", _label_count(accepted, label), None))
        for shard in shards:
            count = _label_count(shard["accepted"], label)
            metrics.append(_metric(str(shard["file_name"]), f"label_{name}", count, shard["split_name"]))
    return metrics


def _missing_metrics(accepted: list[dict[str, object]]) -> list[dict[str, object]]:
    total = max(len(accepted), 1)
    metrics = []
    missing_size = sum(row["vehiclesize"] is None for row in accepted)
    metrics.append(_metric("all", "missing_vehiclesize", missing_size, f"{missing_size / total:.6f}"))
    for name in FEATURE_NAMES:
        missing = sum(row[name] is None for row in accepted)
        metrics.append(_metric("all", f"missing_{name}", missing, f"{missing / total:.6f}"))
    return metrics


def _duplicate_metrics(accepted: list[dict[str, object]]) -> list[dict[str, object]]:
    within = Counter((row["source_file"], row["event_id"]) for row in accepted)
    files_for_id: dict[int, set[str]] = {}
    for row in accepted:
        files_for_id.setdefault(int(row["event_id"]), set()).add(str(row["source_file"]))
    within_ids = sum(1 for count in within.values() if count > 1)
    cross_ids = sum(1 for names in files_for_id.values() if len(names) > 1)
    return [
        _metric("all", "within_file_duplicate_event_ids", within_ids, DUPLICATE_POLICY),
        _metric("all", "cross_file_duplicate_event_ids", cross_ids, DUPLICATE_POLICY),
    ]


def _reason_metrics(rejected: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter(str(row["reason"]) for row in rejected)
    if not counts:
        return [_metric("all", "rejection_reason_none", 0, None)]
    return [_metric("all", f"rejection_{reason}", count, None) for reason, count in sorted(counts.items())]


def _nonfinite_features(accepted: list[dict[str, object]]) -> int:
    return sum(row[name] is None for row in accepted for name in FEATURE_NAMES)


def _unexpected_sizes(accepted: list[dict[str, object]]) -> int:
    return sum(row["vehiclesize"] not in ALLOWED_VEHICLE_SIZES and row["vehiclesize"] is not None for row in accepted)


def _label_count(rows: list[dict[str, object]], label: int) -> int:
    return sum(int(row["label"]) == label for row in rows)


def _inputs(shards: list[dict[str, object]]) -> int:
    return sum(int(shard["input_rows"]) for shard in shards)


def _metric(scope: str, name: str, value: int, detail: str | None) -> dict[str, object]:
    return {"scope": scope, "metric": name, "value": value, "detail": detail}
