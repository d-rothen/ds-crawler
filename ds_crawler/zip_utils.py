"""Utilities for reading and writing dataset files inside ZIP archives."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def is_zip_path(path: str | Path) -> bool:
    """Return True if *path* points to an existing ``.zip`` file."""
    p = Path(path)
    return p.suffix.lower() == ".zip" and p.is_file()


def _detect_root_prefix(namelist: list[str]) -> str:
    """Detect a single common root directory prefix from a zip namelist.

    Many zip tools (e.g. macOS Compress) wrap all entries under a
    top-level directory named after the original folder.  This helper
    detects that pattern and returns the prefix (e.g. ``"test_kitti/"``)
    so callers can account for it.

    Returns the prefix string including the trailing ``/``, or ``""``
    if there is no common single-directory prefix.

    Runs in a single O(n) pass with early exit — negligible overhead
    even for archives with hundreds of thousands of entries.
    """
    candidate: str | None = None
    has_nested = False

    for entry in namelist:
        if entry.startswith("__MACOSX"):
            continue
        first, _, rest = entry.partition("/")
        if candidate is None:
            candidate = first
        elif first != candidate:
            # Two different top-level components → no common prefix.
            return ""
        if rest:
            has_nested = True

    if candidate is None or not has_nested:
        return ""
    return candidate + "/"


def get_zip_root_prefix(zip_path: Path) -> str:
    """Return the common root directory prefix inside a zip, or ``""``."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return _detect_root_prefix(zf.namelist())


def _matches_zip_stem(prefix: str, zip_path: Path) -> bool:
    """Return True if *prefix* matches the zip filename stem.

    This is the heuristic used to distinguish a "wrapper" directory
    (artifact of zipping a folder) from legitimate dataset structure.
    For example ``test_kitti.zip`` whose entries all start with
    ``test_kitti/`` has a wrapper prefix, whereas
    ``depth_predictions.zip`` whose entries start with ``Scene01/``
    does not.
    """
    return prefix.rstrip("/") == zip_path.stem


def read_json_from_zip(zip_path: Path, entry_name: str) -> dict | None:
    """Read and parse a JSON entry from inside a ZIP archive.

    Handles archives that wrap all entries under a single root directory
    (e.g. ``test_kitti/ds-crawler.json`` instead of ``ds-crawler.json``).

    Returns ``None`` if *entry_name* does not exist in the archive.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Try exact match first
        if entry_name in names:
            with zf.open(entry_name) as f:
                return json.load(f)

        # Try with detected root prefix.  We use the zip filename stem
        # as a guard to avoid false positives (e.g. interpreting a
        # legitimate subdirectory like ``Scene01/`` as a wrapper prefix).
        prefix = _detect_root_prefix(names)
        if prefix and _matches_zip_stem(prefix, zip_path):
            prefixed_name = prefix + entry_name
            if prefixed_name in names:
                with zf.open(prefixed_name) as f:
                    return json.load(f)

        return None


def write_json_to_zip(
    zip_path: Path, entry_name: str, data: Any
) -> None:
    """Write (or replace) a JSON entry inside a ZIP archive.

    If *zip_path* does not exist yet the archive is created.  When the
    archive already exists and has a single root directory prefix that
    matches the zip filename, the entry is written under that prefix so
    the convention is preserved.  When replacing an existing entry the
    archive is rebuilt via a temporary file so that no duplicate entries
    are left behind.
    """
    json_bytes = json.dumps(data, indent=2).encode("utf-8")

    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(entry_name, json_bytes)
        return

    # Detect root prefix and adjust the entry name to match existing
    # convention inside the archive.  Only apply when the prefix matches
    # the zip filename stem to avoid misinterpreting dataset structure.
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = _detect_root_prefix(zf.namelist())
    if prefix and _matches_zip_stem(prefix, zip_path):
        entry_name = prefix + entry_name

    # Rebuild: copy every entry except the one we are replacing, then
    # add the new version.
    fd, tmp_name = tempfile.mkstemp(suffix=".zip")
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(zip_path, "r") as src, zipfile.ZipFile(
            tmp_path, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for item in src.infolist():
                if item.filename != entry_name:
                    dst.writestr(item, src.read(item.filename))
            dst.writestr(entry_name, json_bytes)
        shutil.move(str(tmp_path), str(zip_path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        # mkstemp returns an open fd; close it if the file still exists
        import os

        try:
            os.close(fd)
        except OSError:
            pass
