"""Parquet-to-SQLite quality pipeline for VZCrash shards.

The official files are Parquet, not CSV. Raw IMU lists stay in those files.
SQLite stores one row per event: identifiers, labels, lengths, and the same
window summaries the model already uses. Rows are not deleted for extreme
sensor values. Invalid labels and null identifiers are rejected and counted.
"""
