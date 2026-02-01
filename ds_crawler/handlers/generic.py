"""Generic handler for filesystem-based datasets."""

import os
from pathlib import Path
from typing import Iterator

from .base import BaseHandler


class GenericHandler(BaseHandler):
    """Generic dataset handler for standard filesystem layouts.

    Recursively finds all files with matching extensions under the
    dataset path. This is the default handler for all datasets that
    don't have a registered custom handler.
    """

    def get_files(self) -> Iterator[Path]:
        """Yield all files (optionally filtered by extension) under the dataset path."""
        if not self.base_path.exists():
            return

        extensions = self.config.get_file_extensions()

        for root, _dirs, files in os.walk(self.base_path):
            for filename in files:
                if extensions is None or Path(filename).suffix.lower() in extensions:
                    yield Path(root) / filename
