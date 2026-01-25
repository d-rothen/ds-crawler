"""Configuration loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


DatasetType = Literal["depth", "rgb", "segmentation"]


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    path: str
    type: DatasetType  # Used internally for file extension filtering
    basename_regex: str
    id_regex: str
    path_regex: str | None = None
    hierarchy_regex: str | None = None
    named_capture_group_value_separator: str | None = None
    intrinsics_regex: str | None = None
    extrinsics_regex: str | None = None
    flat_ids_unique: bool = False
    properties: dict[str, Any] = field(default_factory=dict)
    output_json: str | None = None

    def __post_init__(self) -> None:
        """Validate the configuration."""
        self._validate_type()
        self._validate_basename_regex()
        self._validate_id_regex()
        if self.path_regex:
            self._validate_path_regex()
        if self.hierarchy_regex:
            self._validate_hierarchy_regex()
        if self.intrinsics_regex:
            self._validate_intrinsics_regex()
        if self.extrinsics_regex:
            self._validate_extrinsics_regex()

    def _validate_type(self) -> None:
        """Validate dataset type."""
        valid_types = {"depth", "rgb", "segmentation"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid type '{self.type}'. Must be one of: {valid_types}")

    def _validate_basename_regex(self) -> None:
        """Validate basename_regex is a valid regex."""
        try:
            re.compile(self.basename_regex)
        except re.error as e:
            raise ValueError(f"Invalid basename_regex: {e}")

    def _validate_id_regex(self) -> None:
        """Validate id_regex has at least one capture group."""
        try:
            pattern = re.compile(self.id_regex)
        except re.error as e:
            raise ValueError(f"Invalid id_regex: {e}")

        if pattern.groups == 0:
            raise ValueError(
                "id_regex must contain at least one capture group."
            )

    def _validate_path_regex(self) -> None:
        """Validate path_regex is a valid regex."""
        try:
            re.compile(self.path_regex)
        except re.error as e:
            raise ValueError(f"Invalid path_regex: {e}")

    def _validate_hierarchy_regex(self) -> None:
        """Validate hierarchy_regex has at least one capture group."""
        try:
            pattern = re.compile(self.hierarchy_regex)
        except re.error as e:
            raise ValueError(f"Invalid hierarchy_regex: {e}")

        if pattern.groups == 0:
            raise ValueError(
                "hierarchy_regex must contain at least one capture group."
            )

        # Check if named capture groups require separator
        if pattern.groupindex and not self.named_capture_group_value_separator:
            raise ValueError(
                "hierarchy_regex has named capture groups but "
                "named_capture_group_value_separator is not defined."
            )

    def _validate_intrinsics_regex(self) -> None:
        """Validate intrinsics_regex is a valid regex with capture groups."""
        try:
            pattern = re.compile(self.intrinsics_regex)
        except re.error as e:
            raise ValueError(f"Invalid intrinsics_regex: {e}")

        if pattern.groups == 0:
            raise ValueError(
                "intrinsics_regex must contain at least one capture group."
            )

    def _validate_extrinsics_regex(self) -> None:
        """Validate extrinsics_regex is a valid regex with capture groups."""
        try:
            pattern = re.compile(self.extrinsics_regex)
        except re.error as e:
            raise ValueError(f"Invalid extrinsics_regex: {e}")

        if pattern.groups == 0:
            raise ValueError(
                "extrinsics_regex must contain at least one capture group."
            )

    @property
    def compiled_basename_regex(self) -> re.Pattern:
        """Return compiled basename regex."""
        return re.compile(self.basename_regex)

    @property
    def compiled_id_regex(self) -> re.Pattern:
        """Return compiled id regex."""
        return re.compile(self.id_regex)

    @property
    def compiled_path_regex(self) -> re.Pattern | None:
        """Return compiled path regex or None."""
        if self.path_regex:
            return re.compile(self.path_regex)
        return None

    @property
    def compiled_hierarchy_regex(self) -> re.Pattern | None:
        """Return compiled hierarchy regex or None."""
        if self.hierarchy_regex:
            return re.compile(self.hierarchy_regex)
        return None

    @property
    def compiled_intrinsics_regex(self) -> re.Pattern | None:
        """Return compiled intrinsics regex or None."""
        if self.intrinsics_regex:
            return re.compile(self.intrinsics_regex)
        return None

    @property
    def compiled_extrinsics_regex(self) -> re.Pattern | None:
        """Return compiled extrinsics regex or None."""
        if self.extrinsics_regex:
            return re.compile(self.extrinsics_regex)
        return None


@dataclass
class Config:
    """Main configuration containing multiple datasets."""

    datasets: list[DatasetConfig]

    @classmethod
    def from_file(cls, path: str | Path, workdir: str | Path | None = None) -> "Config":
        """Load configuration from a JSON file.

        Args:
            path: Path to the configuration JSON file.
            workdir: Optional working directory. If provided, dataset paths
                are treated as relative to this directory.
        """
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
                # If workdir is provided, join it with the dataset path
                ds_path = ds_data["path"]
                if workdir is not None:
                    ds_path = str(Path(workdir) / ds_path)

                ds_config = DatasetConfig(
                    name=ds_data["name"],
                    path=ds_path,
                    type=ds_data["type"],
                    basename_regex=ds_data["basename_regex"],
                    id_regex=ds_data["id_regex"],
                    path_regex=ds_data.get("path_regex"),
                    hierarchy_regex=ds_data.get("hierarchy_regex"),
                    named_capture_group_value_separator=ds_data.get(
                        "named_capture_group_value_separator"
                    ),
                    intrinsics_regex=ds_data.get("intrinsics_regex"),
                    extrinsics_regex=ds_data.get("extrinsics_regex"),
                    flat_ids_unique=ds_data.get("flat_ids_unique", False),
                    properties=ds_data.get("properties", {}),
                    output_json=ds_data.get("output_json"),
                )
                datasets.append(ds_config)
            except KeyError as e:
                raise ValueError(f"Dataset {i} missing required field: {e}")
            except ValueError as e:
                raise ValueError(f"Dataset {i} ({ds_data.get('name', 'unknown')}): {e}")

        return cls(datasets=datasets)
