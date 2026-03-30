"""Configuration loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._dataset_contract import (
    DATASET_CONTRACT_VERSION,
    DatasetHeadContract,
    normalize_meta_dict as _normalize_meta_dict,
    parse_dataset_head,
    validate_contract_version,
)
from .path_filters import PathFilters
from .zip_utils import DATASET_HEAD_FILENAME, read_metadata_json


CONFIG_FILENAME = "ds-crawler.json"
CRAWLER_CONFIG_KIND = "ds_crawler_config"
CRAWLER_CONFIG_VERSION = "2.0"


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _require_non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _resolve_path(
    value: str | Path,
    *,
    base_path: str | Path | None = None,
) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if base_path is None:
        return str(path)
    return str(Path(base_path) / path)


def _normalize_config_contract(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    contract = _require_mapping(value.get("contract"), f"{context}.contract")
    kind = contract.get("kind")
    if kind != CRAWLER_CONFIG_KIND:
        raise ValueError(
            f"{context}.contract.kind must be {CRAWLER_CONFIG_KIND!r}, got {kind!r}"
        )
    version = contract.get("version", CRAWLER_CONFIG_VERSION)
    validate_contract_version(version, f"{context}.contract.version")


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    dataset_root: str
    path: str
    dataset_head: DatasetHeadContract
    head_file: str = DATASET_HEAD_FILENAME
    basename_regex: str | None = None
    id_regex: str = ""
    path_regex: str | None = None
    hierarchy_regex: str | None = None
    named_capture_group_value_separator: str | None = None
    flat_ids_unique: bool = False
    id_regex_join_char: str = "+"
    id_override: str | None = None
    path_filters: dict[str, Any] | None = None
    prebuilt_index_file: str | None = None
    file_extensions: list[str] | None = None
    compiled_path_filters: PathFilters = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.id_regex:
            raise ValueError("indexing.id.regex is required")
        self._normalize_file_extensions()
        self._normalize_path_filters()
        self._compile_and_validate_regexes()

    @property
    def name(self) -> str:
        return self.dataset_head.dataset_name

    @property
    def type(self) -> str:
        return self.dataset_head.modality_key

    @property
    def properties(self) -> dict[str, Any]:
        return self.dataset_head.to_properties_dict()

    @property
    def dataset_id(self) -> str:
        return self.dataset_head.dataset_id

    def head_mapping(self) -> dict[str, Any]:
        return self.dataset_head.to_mapping()

    def to_indexing_dict(self) -> dict[str, Any]:
        indexing: dict[str, Any] = {
            "files": {},
            "id": {
                "regex": self.id_regex,
                "join_char": self.id_regex_join_char,
            },
            "properties": {},
            "constraints": {
                "flat_ids_unique": self.flat_ids_unique,
            },
        }
        if self.file_extensions is not None:
            indexing["files"]["extensions"] = list(self.file_extensions)
        if self.path_filters is not None:
            indexing["files"]["path_filters"] = dict(self.path_filters)
        if self.id_override is not None:
            indexing["id"]["override"] = self.id_override
        if self.hierarchy_regex is not None:
            indexing["hierarchy"] = {
                "regex": self.hierarchy_regex,
            }
            if self.named_capture_group_value_separator is not None:
                indexing["hierarchy"]["separator"] = (
                    self.named_capture_group_value_separator
                )
        if self.path_regex is not None:
            indexing["properties"]["path"] = {"regex": self.path_regex}
        if self.basename_regex is not None:
            indexing["properties"]["basename"] = {"regex": self.basename_regex}
        if not indexing["files"]:
            indexing.pop("files")
        if not indexing["properties"]:
            indexing.pop("properties")
        if not indexing["constraints"]:
            indexing.pop("constraints")
        return indexing

    def to_crawler_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "contract": {
                "kind": CRAWLER_CONFIG_KIND,
                "version": CRAWLER_CONFIG_VERSION,
            },
            "head_file": self.head_file,
            "source": {
                "path": ".",
            },
            "indexing": self.to_indexing_dict(),
        }
        if self.prebuilt_index_file is not None:
            result["source"]["prebuilt_index_file"] = self.prebuilt_index_file
        return result

    def _normalize_file_extensions(self) -> None:
        if self.file_extensions is not None:
            self.file_extensions = [
                ext if ext.startswith(".") else f".{ext}"
                for ext in self.file_extensions
            ]

    def _normalize_path_filters(self) -> None:
        self.compiled_path_filters = PathFilters.from_raw(self.path_filters)
        normalized = self.compiled_path_filters.to_dict()
        self.path_filters = normalized or None

    def matches_path_filters(self, path: str) -> bool:
        return self.compiled_path_filters.matches(path)

    def get_file_extensions(self) -> set[str] | None:
        if self.file_extensions is not None:
            return set(self.file_extensions)
        return None

    def _compile_and_validate_regexes(self) -> None:
        self.compiled_basename_regex: re.Pattern | None = None
        if self.basename_regex:
            try:
                self.compiled_basename_regex = re.compile(self.basename_regex)
            except re.error as e:
                raise ValueError(f"Invalid indexing.properties.basename.regex: {e}")

        try:
            self.compiled_id_regex: re.Pattern = re.compile(self.id_regex)
        except re.error as e:
            raise ValueError(f"Invalid indexing.id.regex: {e}")
        if self.compiled_id_regex.groups == 0:
            raise ValueError("indexing.id.regex must contain at least one capture group.")

        self.compiled_path_regex: re.Pattern | None = None
        if self.path_regex:
            try:
                self.compiled_path_regex = re.compile(self.path_regex)
            except re.error as e:
                raise ValueError(f"Invalid indexing.properties.path.regex: {e}")

        self.compiled_hierarchy_regex: re.Pattern | None = None
        if self.hierarchy_regex:
            try:
                self.compiled_hierarchy_regex = re.compile(self.hierarchy_regex)
            except re.error as e:
                raise ValueError(f"Invalid indexing.hierarchy.regex: {e}")
            if self.compiled_hierarchy_regex.groups == 0:
                raise ValueError(
                    "indexing.hierarchy.regex must contain at least one capture group."
                )
            if (
                self.compiled_hierarchy_regex.groupindex
                and not self.named_capture_group_value_separator
            ):
                raise ValueError(
                    "indexing.hierarchy.separator is required when "
                    "indexing.hierarchy.regex has named capture groups."
                )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        workdir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        dataset_head: DatasetHeadContract | dict[str, Any] | None = None,
    ) -> "DatasetConfig":
        if not isinstance(data, dict):
            raise ValueError("Crawler config must be a JSON object")

        _normalize_config_contract(data, "config")

        source = _require_mapping(data.get("source", {}), "config.source")
        raw_source_path = source.get("path", ".")
        if not isinstance(raw_source_path, str) or not raw_source_path:
            raise ValueError("config.source.path must be a non-empty string")

        dataset_root_path = Path(dataset_root) if dataset_root is not None else None
        resolved_source = _resolve_path(
            raw_source_path,
            base_path=dataset_root_path or workdir,
        )

        if dataset_root_path is None:
            dataset_root_path = Path(resolved_source)

        head_file = data.get("head_file", DATASET_HEAD_FILENAME)
        if not isinstance(head_file, str) or not head_file:
            raise ValueError("config.head_file must be a non-empty string")

        if dataset_head is None:
            embedded_head = data.get("head")
            if embedded_head is not None:
                dataset_head = embedded_head
            else:
                raw_head = read_metadata_json(dataset_root_path, head_file)
                if raw_head is None:
                    raise FileNotFoundError(
                        f"No {head_file} found at: {dataset_root_path}"
                    )
                dataset_head = raw_head

        if isinstance(dataset_head, DatasetHeadContract):
            parsed_head = dataset_head
        else:
            parsed_head = parse_dataset_head(dataset_head, context="config.head")

        indexing = _require_mapping(data.get("indexing"), "config.indexing")
        files_cfg = _require_mapping(indexing.get("files", {}), "config.indexing.files")
        id_cfg = _require_mapping(indexing.get("id"), "config.indexing.id")
        props_cfg = _require_mapping(
            indexing.get("properties", {}),
            "config.indexing.properties",
        )
        basename_cfg = _require_mapping(
            props_cfg.get("basename", {}),
            "config.indexing.properties.basename",
        )
        path_cfg = _require_mapping(
            props_cfg.get("path", {}),
            "config.indexing.properties.path",
        )
        hierarchy_cfg = _require_mapping(
            indexing.get("hierarchy", {}),
            "config.indexing.hierarchy",
        )
        constraints_cfg = _require_mapping(
            indexing.get("constraints", {}),
            "config.indexing.constraints",
        )

        prebuilt_index_file = source.get("prebuilt_index_file")
        if prebuilt_index_file is not None:
            prebuilt_index_file = _resolve_path(
                _require_non_empty_string(
                    prebuilt_index_file,
                    "config.source.prebuilt_index_file",
                ),
                base_path=dataset_root_path,
            )

        file_extensions = files_cfg.get("extensions")
        if file_extensions is not None:
            if not isinstance(file_extensions, list) or any(
                not isinstance(item, str) or not item
                for item in file_extensions
            ):
                raise ValueError(
                    "config.indexing.files.extensions must be a list of strings"
                )

        normalized_meta = _normalize_meta_dict(
            parsed_head.modality_meta,
            parsed_head.modality_key,
            "config.head.modality.meta",
        )
        if normalized_meta is not None and normalized_meta != parsed_head.modality_meta:
            parsed_head = parse_dataset_head(
                {
                    **parsed_head.to_mapping(),
                    "modality": {
                        **parsed_head.to_mapping()["modality"],
                        "meta": normalized_meta,
                    },
                },
                context="config.head",
            )

        join_char = id_cfg.get("join_char", "+")
        if not isinstance(join_char, str) or not join_char:
            raise ValueError("config.indexing.id.join_char must be a non-empty string")

        return cls(
            dataset_root=str(dataset_root_path),
            path=resolved_source,
            dataset_head=parsed_head,
            head_file=head_file,
            basename_regex=basename_cfg.get("regex"),
            id_regex=_require_non_empty_string(
                id_cfg.get("regex"),
                "config.indexing.id.regex",
            ),
            path_regex=path_cfg.get("regex"),
            hierarchy_regex=hierarchy_cfg.get("regex"),
            named_capture_group_value_separator=hierarchy_cfg.get("separator"),
            flat_ids_unique=bool(constraints_cfg.get("flat_ids_unique", False)),
            id_regex_join_char=join_char,
            id_override=id_cfg.get("override"),
            path_filters=files_cfg.get("path_filters"),
            prebuilt_index_file=prebuilt_index_file,
            file_extensions=file_extensions,
        )


def load_dataset_config(
    data: dict[str, Any],
    workdir: str | Path | None = None,
) -> DatasetConfig:
    """Load a DatasetConfig, resolving from ``ds-crawler.json`` when needed."""
    if not isinstance(data, dict):
        raise ValueError("Crawler config must be a JSON object")

    if "indexing" in data or "source" in data:
        return DatasetConfig.from_dict(data, workdir=workdir)

    if "path" not in data:
        raise ValueError("Crawler config shorthand requires a 'path' field")

    ds_root = Path(_resolve_path(data["path"], base_path=workdir))
    file_config = read_metadata_json(ds_root, CONFIG_FILENAME)
    if file_config is None:
        raise FileNotFoundError(
            f"No {CONFIG_FILENAME} found at: {ds_root}"
        )
    return DatasetConfig.from_dict(file_config, dataset_root=ds_root)


@dataclass
class Config:
    """Main configuration containing multiple datasets."""

    datasets: list[DatasetConfig]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        workdir: str | Path | None = None,
    ) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        base_workdir = Path(workdir) if workdir is not None else path.parent

        if "datasets" not in data:
            dataset = load_dataset_config(data, workdir=base_workdir)
            return cls(datasets=[dataset])

        datasets: list[DatasetConfig] = []
        for i, ds_data in enumerate(data["datasets"]):
            try:
                ds_config = load_dataset_config(ds_data, workdir=base_workdir)
                datasets.append(ds_config)
            except FileNotFoundError:
                raise
            except ValueError as e:
                raise ValueError(
                    f"Dataset {i} ({ds_data.get('path', 'unknown')}): {e}"
                )

        return cls(datasets=datasets)
