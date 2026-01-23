"""Handler for VKITTI2 dataset."""

from pathlib import Path
from typing import Iterator

from .base import BaseHandler


class VKITTI2Handler(BaseHandler):
    """Handler for Virtual KITTI 2 dataset.

    VKITTI2 typically has structure like:
        Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg
        Scene01/clone/frames/depth/Camera_0/depth_00001.png

    This handler recursively finds all files with valid extensions
    for the configured data type.
    """

    def get_files(self) -> Iterator[Path]:
        """Yield all relevant files in the VKITTI2 dataset."""
        if not self.base_path.exists():
            return

        extensions = self.get_file_extensions()

        for file_path in self.base_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                yield file_path
