"""Configuration loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import DatasetDescriptor


CONFIG_FILENAME = "ds-crawler.json"
EULER_TRAIN_ALLOWED_USED_AS: frozenset[str] = frozenset({
    "input", "target", "condition",
})
_EULER_TRAIN_ALLOWED_KEYS: frozenset[str] = frozenset({
    "used_as",
    "slot",
    "modality_type",
    "hierarchy_scope",
    "applies_to",
    "task",
})
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
})
_SLOT_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+){2,}$")
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")

# Modality-specific required fields in ``properties.meta``.
# Maps modality_type -> {field_name: (accepted_types, human_label)}.
_MODALITY_META_SCHEMAS: dict[str, dict[str, tuple[type | tuple[type, ...], str]]] = {
    "depth": {
        "radial_depth": (bool, "a bool"),
        "scale_to_meters": ((int, float), "a number"),
    },
}


def _as_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "dataset"

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
    output_json: str | None = None
    file_extensions: list[str] | None = None
    euler_train: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        """Validate configuration and compile regex patterns."""
        if not self.id_regex:
            raise ValueError("id_regex is required")
        self._normalize_file_extensions()
        self._compile_and_validate_regexes()
        self._validate_properties()
        self.euler_train = self._normalize_euler_train()
        self._validate_modality_meta()

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
        if not isinstance(raw, dict):
            raise ValueError("properties.euler_train must be an object")

        unknown = sorted(set(raw.keys()) - _EULER_TRAIN_ALLOWED_KEYS)
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(f"Unknown properties.euler_train key(s): {joined}")

        used_as = _as_non_empty_str(raw.get("used_as"))
        if used_as is None:
            raise ValueError("properties.euler_train.used_as is required")
        if used_as not in EULER_TRAIN_ALLOWED_USED_AS:
            allowed = ", ".join(sorted(EULER_TRAIN_ALLOWED_USED_AS))
            raise ValueError(
                f"properties.euler_train.used_as must be one of {{{allowed}}}, "
                f"got {used_as!r}"
            )

        modality_type = _as_non_empty_str(raw.get("modality_type"))
        if modality_type is None:
            raise ValueError("properties.euler_train.modality_type is required")
        self._validate_token(
            modality_type,
            "properties.euler_train.modality_type",
        )

        slot = _as_non_empty_str(raw.get("slot"))
        task = _as_non_empty_str(raw.get("task"))
        if slot is None:
            scope = task or _slugify(self.name)
            slot = f"{scope}.{used_as}.{modality_type}"
        self._validate_slot(slot)

        result: dict[str, Any] = {
            "used_as": used_as,
            "slot": slot,
            "modality_type": modality_type,
        }

        raw_hierarchy_scope = _as_non_empty_str(raw.get("hierarchy_scope"))
        raw_applies_to = raw.get("applies_to")

        if used_as == "condition":
            hierarchy_scope = raw_hierarchy_scope or self._infer_hierarchy_scope()
            self._validate_token(
                hierarchy_scope,
                "properties.euler_train.hierarchy_scope",
            )
            applies_to = self._normalize_applies_to(raw_applies_to)
            if applies_to is None:
                applies_to = ["*"]
            if not applies_to:
                raise ValueError("properties.euler_train.applies_to cannot be empty")
            result["hierarchy_scope"] = hierarchy_scope
            result["applies_to"] = applies_to
        else:
            if raw_hierarchy_scope is not None or raw_applies_to is not None:
                raise ValueError(
                    "properties.euler_train.hierarchy_scope and applies_to are only "
                    "allowed when used_as is 'condition'"
                )

        return result

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

    def _normalize_applies_to(self, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(
                "properties.euler_train.applies_to must be a list of strings"
            )

        result: list[str] = []
        for item in value:
            token = _as_non_empty_str(item)
            if token is None:
                raise ValueError(
                    "properties.euler_train.applies_to entries must be non-empty strings"
                )
            if token != "*":
                self._validate_token(token, "properties.euler_train.applies_to")
            result.append(token)
        return result

    def _validate_modality_meta(self) -> None:
        """Validate ``properties.meta`` against modality-specific schemas."""
        modality_type = self.euler_train["modality_type"]
        schema = _MODALITY_META_SCHEMAS.get(modality_type)
        if schema is None:
            return

        meta = self.properties.get("meta")
        if meta is None or not isinstance(meta, dict):
            required_keys = ", ".join(sorted(schema))
            raise ValueError(
                f"properties.meta is required for modality_type={modality_type!r} "
                f"and must contain: {required_keys}"
            )

        for key, (expected_type, type_label) in schema.items():
            if key not in meta:
                raise ValueError(
                    f"properties.meta.{key} is required for "
                    f"modality_type={modality_type!r}"
                )
            if not isinstance(meta[key], expected_type):
                raise ValueError(
                    f"properties.meta.{key} must be {type_label}"
                )

    def _validate_slot(self, value: str) -> None:
        if not _SLOT_PATTERN.match(value):
            raise ValueError(
                "properties.euler_train.slot must match "
                "'segment.segment.segment' (alphanumeric/underscore only)"
            )

    def _validate_token(self, value: str, label: str) -> None:
        if not _TOKEN_PATTERN.match(value):
            raise ValueError(
                f"{label} must contain only letters, digits, or underscores"
            )

    def _normalize_file_extensions(self) -> None:
        """Ensure file extensions start with a dot."""
        if self.file_extensions is not None:
            self.file_extensions = [
                ext if ext.startswith(".") else f".{ext}"
                for ext in self.file_extensions
            ]

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
            properties=data.get("properties", {}),
            output_json=data.get("output_json"),
            file_extensions=data.get("file_extensions"),
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
