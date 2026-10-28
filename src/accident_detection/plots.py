"""Audit figures. Example traces are taken from training rows only."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from accident_detection.constants import LABEL_NAMES, LABEL_ORDER
from accident_detection.models import FileScan, PlotExample
from accident_detection.signals import acceleration_magnitude

_CLASS_COLORS = ("#b00020", "#e07a00", "#4c78a8")


def save_audit_figures(
    scans: tuple[FileScan, ...] | list[FileScan],
    examples: dict[int, PlotExample],
    figure_dir: Path,
    *,
    nominal_gps_samples: int,
    imu_hz: int,
) -> tuple[Path, ...]:
    """Write the audit figures and return the paths that were created."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    label_path = figure_dir / "label_counts_by_file.png"
    if _save_label_chart(scans, label_path):
        written.append(label_path)
    gps_path = figure_dir / "gps_length_by_file.png"
    if _save_gps_chart(scans, nominal_gps_samples, gps_path):
        written.append(gps_path)
    accel_path = figure_dir / "accel_magnitude_by_class.png"
    if _save_accel_examples(examples, imu_hz, accel_path):
        written.append(accel_path)
    return tuple(written)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_label_chart(scans, path: Path) -> bool:
    if not scans:
        return False
    plt = _pyplot()
    positions = np.arange(len(scans))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 4))
    for offset, label_id in enumerate(LABEL_ORDER):
        counts = [scan.label_counts.get(label_id, 0) for scan in scans]
        axis.bar(positions + (offset - 1) * width, counts, width, label=LABEL_NAMES[label_id], color=_CLASS_COLORS[offset])
    axis.set_xticks(positions, [scan.file_name for scan in scans], rotation=15, ha="right")
    axis.set_ylabel("Events")
    axis.set_title("Label counts by file")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def _save_gps_chart(scans, nominal: int, path: Path) -> bool:
    if not scans:
        return False
    plt = _pyplot()
    mode = _most_common_other_length(scans, nominal)
    positions = np.arange(len(scans))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 4))
    nominal_counts, mode_counts, other_counts = _gps_buckets(scans, nominal, mode)
    axis.bar(positions - width, nominal_counts, width, label=f"length {nominal}")
    if mode is not None:
        axis.bar(positions, mode_counts, width, label=f"length {mode}")
    axis.bar(positions + width, other_counts, width, label="other lengths")
    axis.set_xticks(positions, [scan.file_name for scan in scans], rotation=15, ha="right")
    axis.set_ylabel("Events")
    axis.set_title("GPS speed list length by file")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def _save_accel_examples(examples: dict[int, PlotExample], imu_hz: int, path: Path) -> bool:
    if not examples:
        return False
    plt = _pyplot()
    figure, axes = plt.subplots(len(LABEL_ORDER), 1, figsize=(10, 8), sharex=False)
    for axis, label_id in zip(axes, LABEL_ORDER):
        _draw_class_panel(axis, examples.get(label_id), label_id, imu_hz)
    axes[-1].set_xlabel(f"Time (s), sample index / {imu_hz} Hz")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def _draw_class_panel(axis, example: PlotExample | None, label_id: int, imu_hz: int) -> None:
    name = LABEL_NAMES[label_id]
    if example is None or example.accelerometer_g.size == 0:
        axis.set_title(f"{name}: no training example in the audited files")
        axis.axis("off")
        return
    time_s = np.arange(example.accelerometer_g.shape[0]) / imu_hz
    for column, channel in enumerate(("X", "Y", "Z")):
        axis.plot(time_s, example.accelerometer_g[:, column], label=channel, linewidth=0.8)
    axis.plot(time_s, acceleration_magnitude(example.accelerometer_g), label="magnitude", color="black", linewidth=1.2)
    timing = "nominal length" if example.nominal_timing else "non-nominal length"
    axis.set_title(f"{name}  {example.source_file}  event {example.event_id}  {timing}")
    axis.set_ylabel("g")
    axis.legend(loc="upper right", ncol=4)


def _most_common_other_length(scans, nominal: int) -> int | None:
    combined: Counter[int] = Counter()
    for scan in scans:
        for length, count in scan.length_counts.get("gps_speed", {}).items():
            if length != nominal:
                combined[length] += count
    if not combined:
        return None
    return sorted(combined.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _gps_buckets(scans, nominal: int, mode: int | None) -> tuple[list[int], list[int], list[int]]:
    nominal_counts, mode_counts, other_counts = [], [], []
    for scan in scans:
        counts = scan.length_counts.get("gps_speed", {})
        nominal_counts.append(counts.get(nominal, 0))
        mode_counts.append(0 if mode is None else counts.get(mode, 0))
        other_counts.append(_remaining_gps(counts, nominal, mode))
    return nominal_counts, mode_counts, other_counts


def _remaining_gps(counts: dict[int, int], nominal: int, mode: int | None) -> int:
    excluded = {nominal}
    if mode is not None:
        excluded.add(mode)
    return sum(count for length, count in counts.items() if length not in excluded)
