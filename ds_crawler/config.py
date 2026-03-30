"""Configuration loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._dataset_contract import (
    DATASET_CONTRACT_VERSION,
    MODALITY_META_SCHEMAS as _MODALITY_META_SCHEMAS,
    PROPERTY_NAMESPACE_KEYS as _PROPERTY_NAMESPACE_KEYS,
    normalize_euler_train as _normalize_euler_train_contract,
    normalize_meta_dict as _normalize_meta_dict,
    validate_contract_version,
)
from .schema import DatasetDescriptor
from .path_filters import PathFilters


CONFIG_FILENAME = "ds-crawler.json"
_validate_meta_dict = _normalize_meta_dict

_RESERVED_TOP_LEVEL_PROPERTIES: frozenset[str] = frozenset({
    "name",
    "path",
    "type",
    "id_regex",
    "id_regex_join_char",
    "hierarchy_regex",
    "named_capture_group_value_separator",
    "sampled",
    "id_override",
    "path_filters",
    "dataset_contract_version",
})

@dataclass
class DatasetConfig(DatasetDescriptor):
    """Configuration for a single dataset."""

    basename_regex: str | None = None
    id_regex: str = ""
    path_regex: str | None = None
    hierarchy_regex: str | None = None
    named_capture_group_value_separator: str | None = None
    intrinsics_regex: str | None = None
    extrinsics_regex: str | None = None
    flat_ids_unique: bool = False
    id_regex_join_char: str = "+"
    id_override: str | None = None
    path_filters: dict[str, Any] | None = None
    output_json: str | None = None
    file_extensions: list[str] | None = None
    dataset_contract_version: str = DATASET_CONTRACT_VERSION
    euler_train: dict[str, Any] = field(init=False)
    compiled_path_filters: PathFilters = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and compile regex patterns."""
        if not self.id_regex:
            raise ValueError("id_regex is required")
        self._normalize_file_extensions()
        self._normalize_path_filters()
        self._compile_and_validate_regexes()
        self._validate_properties()
        validate_contract_version(self.dataset_contract_version)
        self.euler_train = self._normalize_euler_train()
        self._normalize_modality_meta()

    def _validate_properties(self) -> None:
        if not isinstance(self.properties, dict):
            raise ValueError("properties must be a dict when provided")

        reserved = sorted(
            key for key in self.properties if key in _RESERVED_TOP_LEVEL_PROPERTIES
        )
        if reserved:
            joined = ", ".join(reserved)
            raise ValueError(
                f"properties contains reserved top-level key(s): {joined}. "
                "Use dedicated top-level config fields instead."
            )

    def _normalize_euler_train(self) -> dict[str, Any]:
        if "runlog" in self.properties:
            raise ValueError(
                "properties.runlog has been renamed to properties.euler_train"
            )

        raw = self.properties.get("euler_train")
        if raw is None:
            raise ValueError(
                "properties.euler_train is required and must define "
                "'used_as' and 'modality_type'"
            )
        return _normalize_euler_train_contract(
            raw,
            dataset_name=self.name,
            inferred_hierarchy_scope=self._infer_hierarchy_scope(),
            context="properties.euler_train",
        )

    def _infer_hierarchy_scope(self) -> str:
        if self.compiled_hierarchy_regex is None:
            return "root"

        if self.compiled_hierarchy_regex.groupindex:
            ordered_names = [
                name
                for name, _ in sorted(
                    self.compiled_hierarchy_regex.groupindex.items(),
                    key=lambda item: item[1],
                )
            ]
            if ordered_names:
                return "_".join(ordered_names)

        groups = self.compiled_hierarchy_regex.groups
        if groups <= 0:
            return "root"
        return f"level_{groups}"

    def _normalize_modality_meta(self) -> None:
        """Normalize ``properties.meta`` against shared modality contracts."""
        modality_type = self.euler_train["modality_type"]
        normalized = _normalize_meta_dict(
            self.properties.get("meta"),
            modality_type,
            "properties.meta",
        )
        if normalized is None:
            self.properties.pop("meta", None)
        else:
            self.properties["meta"] = normalized

    def _normalize_file_extensions(self) -> None:
        """Ensure file extensions start with a dot."""
        if self.file_extensions is not None:
            self.file_extensions = [
                ext if ext.startswith(".") else f".{ext}"
                for ext in self.file_extensions
            ]

    def _normalize_path_filters(self) -> None:
        """Validate and normalize optional path filter rules."""
        self.compiled_path_filters = PathFilters.from_raw(self.path_filters)
        normalized = self.compiled_path_filters.to_dict()
        self.path_filters = normalized or None

    def matches_path_filters(self, path: str) -> bool:
        """Return whether *path* should be kept by path filter rules."""
        return self.compiled_path_filters.matches(path)

    def get_file_extensions(self) -> set[str] | None:
        """Return the file extensions to filter by, or ``None`` to skip filtering.

        Returns ``None`` when no ``file_extensions`` are explicitly configured,
        meaning the handlers should yield all files and let the regex chain
        handle inclusion/exclusion.
        """
        if self.file_extensions is not None:
            return set(self.file_extensions)
        return None

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

        # Fold top-level namespace keys into properties so that both
        # layouts are supported:
        #   {"properties": {"euler_train": {...}}}   (nested)
        #   {"euler_train": {...}}                    (top-level shorthand)
        # Explicit nested keys take precedence over top-level ones.
        props = dict(data.get("properties", {}))
        for ns in _PROPERTY_NAMESPACE_KEYS:
            if ns in data and ns not in props:
                props[ns] = data[ns]

        return cls(
            name=data["name"],
            path=ds_path,
            type=data["type"],
            basename_regex=data.get("basename_regex"),
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
            id_override=data.get("id_override"),
            path_filters=data.get("path_filters"),
            properties=props,
            output_json=data.get("output_json"),
            file_extensions=data.get("file_extensions"),
            dataset_contract_version=data.get(
                "dataset_contract_version", DATASET_CONTRACT_VERSION
            ),
        )

    def _compile_and_validate_regexes(self) -> None:
        """Compile all regex patterns once, validating as we go."""
        # basename_regex (optional)
        self.compiled_basename_regex: re.Pattern | None = None
        if self.basename_regex:
            try:
                self.compiled_basename_regex = re.compile(self.basename_regex)
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
    """Load a DatasetConfig, resolving from a ``ds-crawler.json`` file if needed.

    If *data* contains all required fields (e.g. ``basename_regex``), it is
    used directly.  Otherwise the function looks for a ``ds-crawler.json``
    file inside the dataset ``path`` (or inside a ``.zip`` archive at that
    path) and merges the two dicts (explicit *data* keys take precedence).

    Args:
        data: Dataset entry dict — either a full config or just ``{"path": "..."}``.
        workdir: Optional working directory prepended to relative paths.
    """
    resolved = data
    if "id_regex" not in data:
        # Path-only entry — resolve the rest from ds-crawler.json
        ds_path = data["path"]
        if workdir is not None:
            ds_path = str(Path(workdir) / ds_path)

        ds_path_obj = Path(ds_path)
        from .zip_utils import read_metadata_json

        file_config = read_metadata_json(ds_path_obj, CONFIG_FILENAME)
        if file_config is None:
            raise FileNotFoundError(
                f"Dataset entry has no inline config and no {CONFIG_FILENAME} "
                f"found at: {ds_path_obj}"
            )
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
