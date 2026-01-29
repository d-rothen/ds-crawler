"""Configuration loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .schema import DatasetDescriptor


DatasetType = Literal["depth", "rgb", "segmentation"]
CONFIG_FILENAME = "ds-crawler.json"

DEFAULT_TYPE_EXTENSIONS: dict[str, set[str]] = {
    "rgb": {".png", ".jpg", ".jpeg"},
    "depth": {".png", ".exr", ".npy", ".pfm"},
    "segmentation": {".png"},
}


@dataclass(kw_only=True)
class DatasetConfig(DatasetDescriptor):
    """Configuration for a single dataset."""

    basename_regex: str
    id_regex: str
    path_regex: str | None = None
    hierarchy_regex: str | None = None
    named_capture_group_value_separator: str | None = None
    intrinsics_regex: str | None = None
    extrinsics_regex: str | None = None
    flat_ids_unique: bool = False
    id_regex_join_char: str = "+"
    output_json: str | None = None
    file_extensions: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration and compile regex patterns."""
        self._validate_type()
        self._normalize_file_extensions()
        self._compile_and_validate_regexes()

    def _validate_type(self) -> None:
        valid_types = {"depth", "rgb", "segmentation"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid type '{self.type}'. Must be one of: {valid_types}")

    def _normalize_file_extensions(self) -> None:
        """Ensure file extensions start with a dot."""
        if self.file_extensions is not None:
            self.file_extensions = [
                ext if ext.startswith(".") else f".{ext}"
                for ext in self.file_extensions
            ]

    def get_file_extensions(self) -> set[str]:
        """Return the effective file extensions for this dataset.

        Uses file_extensions from config if provided, otherwise
        falls back to defaults based on dataset type.
        """
        if self.file_extensions is not None:
            return set(self.file_extensions)
        return DEFAULT_TYPE_EXTENSIONS.get(self.type, {".png"})

    @classmethod
    def from_dict(cls, data: dict[str, Any], workdir: str | Path | None = None) -> "DatasetConfig":
        """Create a DatasetConfig from a dict (single dataset entry).

        Args:
            data: Dict with the same keys as a config.json dataset entry.
            workdir: Optional working directory prepended to relative paths.
        """
        ds_path = data["path"]
        if workdir is not None:
            ds_path = str(Path(workdir) / ds_path)

        return cls(
            name=data["name"],
            path=ds_path,
            type=data["type"],
            basename_regex=data["basename_regex"],
            id_regex=data["id_regex"],
            path_regex=data.get("path_regex"),
            hierarchy_regex=data.get("hierarchy_regex"),
            named_capture_group_value_separator=data.get(
                "named_capture_group_value_separator"
            ),
            intrinsics_regex=data.get("intrinsics_regex"),
            extrinsics_regex=data.get("extrinsics_regex"),
            flat_ids_unique=data.get("flat_ids_unique", False),
            id_regex_join_char=data.get("id_regex_join_char", "+"),
            properties=data.get("properties", {}),
            output_json=data.get("output_json"),
            file_extensions=data.get("file_extensions"),
        )

    def _compile_and_validate_regexes(self) -> None:
        """Compile all regex patterns once, validating as we go."""
        # basename_regex (required)
        try:
            self.compiled_basename_regex: re.Pattern = re.compile(self.basename_regex)
        except re.error as e:
            raise ValueError(f"Invalid basename_regex: {e}")

        # id_regex (required, needs capture groups)
        try:
            self.compiled_id_regex: re.Pattern = re.compile(self.id_regex)
        except re.error as e:
            raise ValueError(f"Invalid id_regex: {e}")
        if self.compiled_id_regex.groups == 0:
            raise ValueError("id_regex must contain at least one capture group.")

        # path_regex (optional)
        self.compiled_path_regex: re.Pattern | None = None
        if self.path_regex:
            try:
                self.compiled_path_regex = re.compile(self.path_regex)
            except re.error as e:
                raise ValueError(f"Invalid path_regex: {e}")

        # hierarchy_regex (optional, needs capture groups)
        self.compiled_hierarchy_regex: re.Pattern | None = None
        if self.hierarchy_regex:
            try:
                self.compiled_hierarchy_regex = re.compile(self.hierarchy_regex)
            except re.error as e:
                raise ValueError(f"Invalid hierarchy_regex: {e}")
            if self.compiled_hierarchy_regex.groups == 0:
                raise ValueError(
                    "hierarchy_regex must contain at least one capture group."
                )
            if self.compiled_hierarchy_regex.groupindex and not self.named_capture_group_value_separator:
                raise ValueError(
                    "hierarchy_regex has named capture groups but "
                    "named_capture_group_value_separator is not defined."
                )

        # intrinsics_regex (optional, needs capture groups)
        self.compiled_intrinsics_regex: re.Pattern | None = None
        if self.intrinsics_regex:
            try:
                self.compiled_intrinsics_regex = re.compile(self.intrinsics_regex)
            except re.error as e:
                raise ValueError(f"Invalid intrinsics_regex: {e}")
            if self.compiled_intrinsics_regex.groups == 0:
                raise ValueError(
                    "intrinsics_regex must contain at least one capture group."
                )

        # extrinsics_regex (optional, needs capture groups)
        self.compiled_extrinsics_regex: re.Pattern | None = None
        if self.extrinsics_regex:
            try:
                self.compiled_extrinsics_regex = re.compile(self.extrinsics_regex)
            except re.error as e:
                raise ValueError(f"Invalid extrinsics_regex: {e}")
            if self.compiled_extrinsics_regex.groups == 0:
                raise ValueError(
                    "extrinsics_regex must contain at least one capture group."
                )


def load_dataset_config(
    data: dict[str, Any], workdir: str | Path | None = None
) -> DatasetConfig:
    """Load a DatasetConfig, resolving from a ``ds-crawler.config`` file if needed.

    If *data* contains all required fields (e.g. ``basename_regex``), it is
    used directly.  Otherwise the function looks for a ``ds-crawler.config``
    JSON file inside the dataset ``path`` and merges the two dicts (explicit
    *data* keys take precedence).

    Args:
        data: Dataset entry dict — either a full config or just ``{"path": "..."}``.
        workdir: Optional working directory prepended to relative paths.
    """
    resolved = data
    if "basename_regex" not in data:
        # Path-only entry — resolve the rest from ds-crawler.config
        ds_path = data["path"]
        if workdir is not None:
            ds_path = str(Path(workdir) / ds_path)
        config_file = Path(ds_path) / CONFIG_FILENAME
        if not config_file.exists():
            raise FileNotFoundError(
                f"Dataset entry has no inline config and no {CONFIG_FILENAME} "
                f"found at: {config_file}"
            )
        with open(config_file) as f:
            file_config = json.load(f)
        # Caller-supplied keys override file values
        resolved = {**file_config, **data}
    return DatasetConfig.from_dict(resolved, workdir=workdir)


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
                ds_config = load_dataset_config(ds_data, workdir=workdir)
                datasets.append(ds_config)
            except FileNotFoundError:
                raise
            except KeyError as e:
                raise ValueError(f"Dataset {i} missing required field: {e}")
            except ValueError as e:
                raise ValueError(f"Dataset {i} ({ds_data.get('name', 'unknown')}): {e}")

        return cls(datasets=datasets)
