"""Utilities for reading and writing dataset files inside ZIP archives."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


METADATA_DIR = ".ds_crawler"
DATASET_HEAD_FILENAME = "dataset-head.json"
INDEX_FILENAME = "index.json"
OUTPUT_FILENAME = INDEX_FILENAME
SPLIT_FILENAME_PREFIX = "split_"
_SPLIT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# File extensions whose contents are already compressed.  Writing these
# with ZIP_STORED instead of ZIP_DEFLATED avoids a costly recompression
# pass that yields virtually no size reduction.
COMPRESSED_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".exr", ".webp",
})


def validate_split_name(split_name: str) -> str:
    """Validate and normalize a split name used for metadata filenames."""
    if not isinstance(split_name, str):
        raise ValueError("split_name must be a string")
    normalized = split_name.strip()
    if not normalized:
        raise ValueError("split_name must be a non-empty string")
    if not _SPLIT_NAME_PATTERN.match(normalized):
        raise ValueError(
            "split_name may only contain letters, digits, '.', '-', or '_'"
        )
    return normalized


def get_split_filename(split_name: str) -> str:
    """Return the metadata filename for a named split."""
    normalized = validate_split_name(split_name)
    return f"{SPLIT_FILENAME_PREFIX}{normalized}.json"


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


def write_json_entries_to_zip(
    zip_path: Path,
    entries: dict[str, Any],
) -> None:
    """Write or replace multiple JSON entries inside a ZIP archive in one pass."""
    if not entries:
        return

    resolved_entries = dict(entries)
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            prefix = _detect_root_prefix(zf.namelist())
        if prefix and _matches_zip_stem(prefix, zip_path):
            resolved_entries = {
                prefix + entry_name: data
                for entry_name, data in entries.items()
            }

    json_bytes = {
        entry_name: json.dumps(data, indent=2).encode("utf-8")
        for entry_name, data in resolved_entries.items()
    }

    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry_name, payload in json_bytes.items():
                zf.writestr(entry_name, payload)
        return

    replace_names = set(json_bytes)
    fd, tmp_name = tempfile.mkstemp(suffix=".zip")
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(zip_path, "r") as src, zipfile.ZipFile(
            tmp_path, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for item in src.infolist():
                if item.filename not in replace_names:
                    dst.writestr(item, src.read(item.filename))
            for entry_name, payload in json_bytes.items():
                dst.writestr(entry_name, payload)
        shutil.move(str(tmp_path), str(zip_path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        import os

        try:
            os.close(fd)
        except OSError:
            pass


def read_metadata_json(
    dataset_path: Path, filename: str
) -> dict | None:
    """Read a metadata JSON file from a dataset's ``.ds_crawler/`` directory.

    For both ZIP and filesystem datasets, checks
    ``.ds_crawler/{filename}`` first, then falls back to ``{filename}``
    at the dataset root for backwards compatibility.

    Args:
        dataset_path: Root directory or ``.zip`` archive of the dataset.
        filename: The metadata filename (e.g. ``"index.json"``).

    Returns:
        Parsed JSON dict, or ``None`` if the file is not found in
        either location.
    """
    new_name = f"{METADATA_DIR}/{filename}"

    if is_zip_path(dataset_path):
        result = read_json_from_zip(dataset_path, new_name)
        if result is not None:
            return result
        return read_json_from_zip(dataset_path, filename)

    new_path = dataset_path / METADATA_DIR / filename
    if new_path.is_file():
        with open(new_path) as f:
            return json.load(f)

    old_path = dataset_path / filename
    if old_path.is_file():
        with open(old_path) as f:
            return json.load(f)

    return None


def list_metadata_json_filenames(dataset_path: Path) -> list[str]:
    """List JSON metadata filenames stored under ``.ds_crawler/``."""
    if is_zip_path(dataset_path):
        with zipfile.ZipFile(dataset_path, "r") as zf:
            names = zf.namelist()
        prefix = _detect_root_prefix(names)
        if prefix and _matches_zip_stem(prefix, dataset_path):
            names = [
                name[len(prefix):] if name.startswith(prefix) else name
                for name in names
            ]
        metadata_prefix = f"{METADATA_DIR}/"
        filenames = {
            name[len(metadata_prefix):]
            for name in names
            if name.startswith(metadata_prefix) and name.endswith(".json")
        }
        return sorted(filenames)

    metadata_dir = dataset_path / METADATA_DIR
    if not metadata_dir.is_dir():
        return []
    return sorted(
        path.name
        for path in metadata_dir.glob("*.json")
        if path.is_file()
    )


def list_split_names(dataset_path: Path) -> list[str]:
    """List split names available under ``.ds_crawler/``."""
    suffix = ".json"
    result: list[str] = []
    for filename in list_metadata_json_filenames(dataset_path):
        if (
            filename.startswith(SPLIT_FILENAME_PREFIX)
            and filename.endswith(suffix)
        ):
            split_name = filename[len(SPLIT_FILENAME_PREFIX):-len(suffix)]
            if split_name:
                result.append(split_name)
    return sorted(result)


def write_metadata_json(
    dataset_path: Path, filename: str, data: Any
) -> Path:
    """Write a metadata JSON file to a dataset's ``.ds_crawler/`` directory.

    Always writes to ``.ds_crawler/{filename}``.  For filesystem
    datasets the directory is created if it does not exist.  For ZIP
    datasets the entry is written as ``.ds_crawler/{filename}`` inside
    the archive.

    Args:
        dataset_path: Root directory or ``.zip`` archive of the dataset.
        filename: The metadata filename (e.g. ``"index.json"``).
        data: JSON-serialisable data to write.

    Returns:
        The path that was written to.  For filesystem datasets this is
        the actual file path; for ZIP datasets this is the ZIP path
        itself.
    """
    new_name = f"{METADATA_DIR}/{filename}"

    if is_zip_path(dataset_path):
        write_json_to_zip(dataset_path, new_name, data)
        return dataset_path

    output_dir = dataset_path / METADATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path


def write_metadata_json_batch(
    dataset_path: Path,
    entries: dict[str, Any],
) -> Path:
    """Write multiple metadata JSON files under ``.ds_crawler/``."""
    if not entries:
        return dataset_path

    metadata_entries = {
        f"{METADATA_DIR}/{filename}": data
        for filename, data in entries.items()
    }

    if is_zip_path(dataset_path):
        write_json_entries_to_zip(dataset_path, metadata_entries)
        return dataset_path

    output_dir = dataset_path / METADATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in entries.items():
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    return output_dir
