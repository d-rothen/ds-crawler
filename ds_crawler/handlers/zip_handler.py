"""Handler for datasets stored inside ZIP archives."""

import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterator

from .base import BaseHandler


class ZipHandler(BaseHandler):
    """Dataset handler that discovers files inside a ZIP archive.

    Yields synthetic ``Path`` objects of the form
    ``<zip_file_path>/<entry_name>`` so that
    ``file_path.relative_to(base_path)`` and ``file_path.name`` produce
    the correct relative path and filename – identical to what
    ``GenericHandler`` returns for a real directory tree.
    """

    def get_files(self) -> Iterator[Path]:
        """Yield all files with valid extensions inside the ZIP archive."""
        if not self.base_path.exists() or not self.base_path.is_file():
            return

        extensions = self.config.get_file_extensions()

        with zipfile.ZipFile(self.base_path, "r") as zf:
            for entry in zf.namelist():
                # Skip directory entries
                if entry.endswith("/"):
                    continue
                if PurePosixPath(entry).suffix.lower() in extensions:
                    yield self.base_path / entry
