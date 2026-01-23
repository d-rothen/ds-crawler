"""Configuration loading and validation."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DatasetType = Literal["depth", "rgb", "segmentation"]


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    path: str
    type: DatasetType
    gt: bool
    basename_regex: str
    path_regex: str | None = None

    def __post_init__(self) -> None:
        """Validate the configuration."""
        self._validate_type()
        self._validate_basename_regex()
        if self.path_regex:
            self._validate_path_regex()

    def _validate_type(self) -> None:
        """Validate dataset type."""
        valid_types = {"depth", "rgb", "segmentation"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid type '{self.type}'. Must be one of: {valid_types}")

    def _validate_basename_regex(self) -> None:
        """Validate basename_regex has required 'id' capture group."""
        try:
            pattern = re.compile(self.basename_regex)
        except re.error as e:
            raise ValueError(f"Invalid basename_regex: {e}")

        if "id" not in pattern.groupindex:
            raise ValueError(
                f"basename_regex must contain named capture group 'id'. "
                f"Found groups: {list(pattern.groupindex.keys())}"
            )

    def _validate_path_regex(self) -> None:
        """Validate path_regex is a valid regex."""
        try:
            re.compile(self.path_regex)
        except re.error as e:
            raise ValueError(f"Invalid path_regex: {e}")

    @property
    def compiled_basename_regex(self) -> re.Pattern:
        """Return compiled basename regex."""
        return re.compile(self.basename_regex)

    @property
    def compiled_path_regex(self) -> re.Pattern | None:
        """Return compiled path regex or None."""
        if self.path_regex:
            return re.compile(self.path_regex)
        return None


@dataclass
class Config:
    """Main configuration containing multiple datasets."""

    datasets: list[DatasetConfig]

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        if "datasets" not in data:
            raise ValueError("Config must contain 'datasets' key")

        datasets = []
        for i, ds_data in enumerate(data["datasets"]):
            try:
                ds_config = DatasetConfig(
                    name=ds_data["name"],
                    path=ds_data["path"],
                    type=ds_data["type"],
                    gt=ds_data["gt"],
                    basename_regex=ds_data["basename_regex"],
                    path_regex=ds_data.get("path_regex"),
                )
                datasets.append(ds_config)
            except KeyError as e:
                raise ValueError(f"Dataset {i} missing required field: {e}")
            except ValueError as e:
                raise ValueError(f"Dataset {i} ({ds_data.get('name', 'unknown')}): {e}")

        return cls(datasets=datasets)
