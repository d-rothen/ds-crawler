"""Base handler for dataset parsing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from ..config import DatasetConfig


class BaseHandler(ABC):
    """Base class for dataset handlers."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize handler with dataset config."""
        self.config = config
        self.base_path = Path(config.path)

    @abstractmethod
    def get_files(self) -> Iterator[Path]:
        """Yield all relevant files in the dataset.

        Subclasses should implement this to define which files
        are considered part of the dataset.
        """
        pass

    def get_file_extensions(self) -> set[str]:
        """Return valid file extensions for this dataset type.

        Can be overridden by subclasses for custom behavior.
        """
        type_extensions = {
            "rgb": {".png", ".jpg", ".jpeg"},
            "depth": {".png", ".exr", ".npy", ".pfm"},
            "segmentation": {".png"},
        }
        return type_extensions.get(self.config.type, {".png"})
