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


def read_json_from_zip(zip_path: Path, entry_name: str) -> dict | None:
    """Read and parse a JSON entry from inside a ZIP archive.

    Returns ``None`` if *entry_name* does not exist in the archive.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        if entry_name not in zf.namelist():
            return None
        with zf.open(entry_name) as f:
            return json.load(f)


def write_json_to_zip(
    zip_path: Path, entry_name: str, data: Any
) -> None:
    """Write (or replace) a JSON entry inside a ZIP archive.

    If *zip_path* does not exist yet the archive is created.  When
    replacing an existing entry the archive is rebuilt via a temporary
    file so that no duplicate entries are left behind.
    """
    json_bytes = json.dumps(data, indent=2).encode("utf-8")

    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(entry_name, json_bytes)
        return

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
