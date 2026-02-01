"""Handler for datasets stored inside ZIP archives."""

import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterator

from .base import BaseHandler
from ..zip_utils import _detect_root_prefix, _matches_zip_stem


class ZipHandler(BaseHandler):
    """Dataset handler that discovers files inside a ZIP archive.

    Yields synthetic ``Path`` objects of the form
    ``<zip_file_path>/<entry_name>`` so that
    ``file_path.relative_to(base_path)`` and ``file_path.name`` produce
    the correct relative path and filename – identical to what
    ``GenericHandler`` returns for a real directory tree.

    When the archive contains a single root directory prefix whose name
    matches the zip filename (e.g. ``test_kitti.zip`` with entries under
    ``test_kitti/``), that prefix is stripped so the yielded paths match
    what the user expects.
    """

    def get_files(self) -> Iterator[Path]:
        """Yield all files with valid extensions inside the ZIP archive."""
        if not self.base_path.exists() or not self.base_path.is_file():
            return

        extensions = self.config.get_file_extensions()

        with zipfile.ZipFile(self.base_path, "r") as zf:
            namelist = zf.namelist()
            prefix = _detect_root_prefix(namelist)
            # Only strip prefix when it matches the zip filename stem.
            if not _matches_zip_stem(prefix, self.base_path):
                prefix = ""
            prefix_len = len(prefix)

            for entry in namelist:
                # Skip __MACOSX resource-fork entries
                if entry.startswith("__MACOSX"):
                    continue
                # Skip directory entries
                if entry.endswith("/"):
                    continue
                # Strip root prefix if present
                stripped = entry[prefix_len:] if prefix else entry
                if not stripped:
                    continue
                if extensions is None or PurePosixPath(stripped).suffix.lower() in extensions:
                    yield self.base_path / stripped
