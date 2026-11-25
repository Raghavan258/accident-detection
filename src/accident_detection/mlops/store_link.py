"""Point recorded MLflow file URIs at the mounted store.

Training records absolute artifact paths. A container mount does not move
those paths. This links a missing host ``mlops`` directory to the mounted
store. It does not copy or rewrite the tracking database.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse


def link_recorded_store(database: Path, mount: Path) -> list[Path]:
    """Link missing recorded ``mlops`` directories to ``mount``.

    Existing directories are left unchanged. The database bytes are not edited.
    """

    target = mount.resolve()
    linked: list[Path] = []
    for root in _missing_roots(database):
        if root.exists():
            continue
        root.parent.mkdir(parents=True, exist_ok=True)
        _link_directory(root, target)
        linked.append(root)
    return linked


def _link_directory(root: Path, target: Path) -> None:
    try:
        root.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(root), str(target)], check=True)


def _missing_roots(database: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for uri in _artifact_uris(database):
        root = _mlops_root(_path_from_uri(uri))
        if root is None or root.exists():
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _artifact_uris(database: Path) -> list[str]:
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        found = _column(connection, "SELECT artifact_location FROM experiments")
        found.extend(_column(connection, "SELECT artifact_uri FROM runs"))
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    return [value for value in found if value]


def _column(connection: sqlite3.Connection, query: str) -> list[str]:
    return [str(row[0]) for row in connection.execute(query) if row[0]]


def _path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    raw = unquote(parsed.path)
    if os.name == "nt" and _windows_drive(raw):
        raw = raw[1:]
    if not raw:
        return None
    return Path(raw)


def _windows_drive(raw: str) -> bool:
    return len(raw) > 2 and raw[0] == "/" and raw[2] == ":"


def _mlops_root(path: Path | None) -> Path | None:
    if path is None or "mlops" not in path.parts:
        return None
    index = path.parts.index("mlops")
    return Path(*path.parts[: index + 1])
