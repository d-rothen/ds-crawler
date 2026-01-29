"""Base handler for dataset parsing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from ..config import DatasetConfig


class BaseHandler(ABC):
    """Base class for dataset handlers.

    Subclass this for datasets that require custom file discovery logic
    (e.g., reading from archives, databases, or non-standard layouts).
    For standard filesystem datasets, GenericHandler is used automatically.
    """

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
