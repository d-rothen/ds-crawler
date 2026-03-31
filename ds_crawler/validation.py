"""Validation helpers for crawler config and output objects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ._dataset_contract import (
    parse_dataset_head,
    validate_contract_kind,
    validate_contract_version,
)
from .artifacts import (
    DATASET_SPLIT_KIND,
    DATASET_SPLIT_VERSION,
    load_saved_output,
)
from .config import (
    CONFIG_FILENAME,
    CRAWLER_CONFIG_KIND,
    CRAWLER_CONFIG_VERSION,
    DatasetConfig,
)
from .path_filters import PathFilters
from .zip_utils import DATASET_HEAD_FILENAME, OUTPUT_FILENAME, read_metadata_json


DATASET_INDEX_KIND = "dataset_index"
DATASET_INDEX_VERSION = "1.0"


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _validate_contract(
    value: Any,
    *,
    expected_kind: str,
    expected_version: str,
    context: str,
) -> None:
    contract = _require_mapping(value, context)
    validate_contract_kind(
        contract.get("kind") if expected_kind == "dataset_head" else contract.get("kind"),
        f"{context}.kind",
    )
    if expected_kind != "dataset_head":
        kind = contract.get("kind")
        if kind != expected_kind:
            raise ValueError(f"{context}.kind must be {expected_kind!r}, got {kind!r}")
    version = contract.get("version", expected_version)
    validate_contract_version(version, f"{context}.version")


def validate_crawler_config(
    config: dict[str, Any],
    workdir: str | Path | None = None,
) -> DatasetConfig:
    if not isinstance(config, dict):
        raise ValueError("Crawler config must be a JSON object")

    contract = _require_mapping(config.get("contract"), "config.contract")
    kind = contract.get("kind")
    if kind != CRAWLER_CONFIG_KIND:
        raise ValueError(
            f"config.contract.kind must be {CRAWLER_CONFIG_KIND!r}, got {kind!r}"
        )
    version = contract.get("version", CRAWLER_CONFIG_VERSION)
    validate_contract_version(version, "config.contract.version")

    try:
        return DatasetConfig.from_dict(config, workdir=workdir)
    except KeyError as exc:
        raise ValueError(f"Crawler config missing required field: {exc}") from exc


def _validate_string_dict(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        if not isinstance(item, str):
            raise ValueError(f"{label}[{key!r}] must be a string")


def _validate_file_entry(entry: Any, context: str) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{context} must be an object")

    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{context}.path must be a non-empty string")

    file_id = entry.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise ValueError(f"{context}.id must be a non-empty string")

    path_props = entry.get("path_properties")
    if path_props is not None:
        _validate_string_dict(path_props, f"{context}.path_properties")

    basename_props = entry.get("basename_properties")
    if basename_props is not None:
        _validate_string_dict(basename_props, f"{context}.basename_properties")


def _validate_index_node(node: Any, context: str) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"{context} must be an object")

    files = node.get("files")
    if files is not None:
        if not isinstance(files, list):
            raise ValueError(f"{context}.files must be a list")
        for i, entry in enumerate(files):
            _validate_file_entry(entry, f"{context}.files[{i}]")

    children = node.get("children")
    if children is not None:
        if not isinstance(children, dict):
            raise ValueError(f"{context}.children must be an object")
        for key, child in children.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context}.children keys must be non-empty strings")
            _validate_index_node(child, f"{context}.children[{key!r}]")


def _validate_optional_regex(value: Any, context: str, *, require_groups: bool = False) -> re.Pattern | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    try:
        compiled = re.compile(value)
    except re.error as exc:
        raise ValueError(f"{context} is not a valid regex: {exc}") from exc
    if require_groups and compiled.groups == 0:
        raise ValueError(f"{context} must contain at least one capture group")
    return compiled


def _validate_output_indexing(value: Any, context: str) -> None:
    indexing = _require_mapping(value, context)

    files = indexing.get("files")
    if files is not None:
        files = _require_mapping(files, f"{context}.files")
        extensions = files.get("extensions")
        if extensions is not None:
            if not isinstance(extensions, list) or any(
                not isinstance(item, str) or not item for item in extensions
            ):
                raise ValueError(
                    f"{context}.files.extensions must be a list of strings"
                )
        path_filters = files.get("path_filters")
        if path_filters is not None:
            PathFilters.from_raw(path_filters, context=f"{context}.files.path_filters")

    id_cfg = indexing.get("id")
    if id_cfg is not None:
        id_cfg = _require_mapping(id_cfg, f"{context}.id")
        _validate_optional_regex(
            id_cfg.get("regex"),
            f"{context}.id.regex",
            require_groups=True,
        )
        join_char = id_cfg.get("join_char")
        if join_char is not None and (not isinstance(join_char, str) or not join_char):
            raise ValueError(f"{context}.id.join_char must be a non-empty string")
        override = id_cfg.get("override")
        if override is not None and (not isinstance(override, str) or not override):
            raise ValueError(f"{context}.id.override must be a non-empty string")

    properties = indexing.get("properties")
    if properties is not None:
        properties = _require_mapping(properties, f"{context}.properties")
        path_cfg = properties.get("path")
        if path_cfg is not None:
            path_cfg = _require_mapping(path_cfg, f"{context}.properties.path")
            _validate_optional_regex(
                path_cfg.get("regex"),
                f"{context}.properties.path.regex",
            )
        basename_cfg = properties.get("basename")
        if basename_cfg is not None:
            basename_cfg = _require_mapping(
                basename_cfg,
                f"{context}.properties.basename",
            )
            _validate_optional_regex(
                basename_cfg.get("regex"),
                f"{context}.properties.basename.regex",
            )

    hierarchy = indexing.get("hierarchy")
    if hierarchy is not None:
        hierarchy = _require_mapping(hierarchy, f"{context}.hierarchy")
        compiled = _validate_optional_regex(
            hierarchy.get("regex"),
            f"{context}.hierarchy.regex",
            require_groups=True,
        )
        separator = hierarchy.get("separator")
        if separator is not None and (not isinstance(separator, str) or not separator):
            raise ValueError(f"{context}.hierarchy.separator must be a non-empty string")
        if compiled is not None and compiled.groupindex and not separator:
            raise ValueError(
                f"{context}.hierarchy.separator is required when "
                "hierarchy.regex has named groups"
            )

    constraints = indexing.get("constraints")
    if constraints is not None:
        constraints = _require_mapping(constraints, f"{context}.constraints")
        flat_ids_unique = constraints.get("flat_ids_unique")
        if flat_ids_unique is not None and not isinstance(flat_ids_unique, bool):
            raise ValueError(f"{context}.constraints.flat_ids_unique must be a bool")


def _validate_output_dataset(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")

    contract = _require_mapping(value.get("contract"), f"{context}.contract")
    kind = contract.get("kind")
    if kind != DATASET_INDEX_KIND:
        raise ValueError(
            f"{context}.contract.kind must be {DATASET_INDEX_KIND!r}, got {kind!r}"
        )
    version = contract.get("version", DATASET_INDEX_VERSION)
    validate_contract_version(version, f"{context}.contract.version")

    head_file = value.get("head_file")
    if not isinstance(head_file, str) or not head_file:
        raise ValueError(f"{context}.head_file must be a non-empty string")

    if "head" not in value:
        raise ValueError(f"{context}.head is required")
    parse_dataset_head(value["head"], context=f"{context}.head")

    indexing = value.get("indexing")
    if indexing is not None:
        _validate_output_indexing(indexing, f"{context}.indexing")

    execution = value.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError(f"{context}.execution must be an object")
    sampled = execution.get("sampled")
    if sampled is not None and (not isinstance(sampled, int) or sampled <= 0):
        raise ValueError(f"{context}.execution.sampled must be a positive integer")

    if "index" not in value:
        raise ValueError(f"{context}.index is required")
    _validate_index_node(value["index"], f"{context}.index")


def _validate_index_artifact(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")

    contract = _require_mapping(value.get("contract"), f"{context}.contract")
    kind = contract.get("kind")
    if kind != DATASET_INDEX_KIND:
        raise ValueError(
            f"{context}.contract.kind must be {DATASET_INDEX_KIND!r}, got {kind!r}"
        )
    version = contract.get("version", DATASET_INDEX_VERSION)
    validate_contract_version(version, f"{context}.contract.version")

    generator = value.get("generator")
    if generator is not None:
        _require_mapping(generator, f"{context}.generator")

    execution = value.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError(f"{context}.execution must be an object")
    sampled = execution.get("sampled")
    if sampled is not None and (not isinstance(sampled, int) or sampled <= 0):
        raise ValueError(f"{context}.execution.sampled must be a positive integer")

    if "index" not in value:
        raise ValueError(f"{context}.index is required")
    _validate_index_node(value["index"], f"{context}.index")


def validate_split_artifact(value: Any, context: str = "split") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")

    contract = _require_mapping(value.get("contract"), f"{context}.contract")
    kind = contract.get("kind")
    if kind != DATASET_SPLIT_KIND:
        raise ValueError(
            f"{context}.contract.kind must be {DATASET_SPLIT_KIND!r}, got {kind!r}"
        )
    version = contract.get("version", DATASET_SPLIT_VERSION)
    validate_contract_version(version, f"{context}.contract.version")

    split = _require_mapping(value.get("split"), f"{context}.split")
    name = split.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{context}.split.name must be a non-empty string")
    source_index_file = split.get("source_index_file")
    if not isinstance(source_index_file, str) or not source_index_file:
        raise ValueError(
            f"{context}.split.source_index_file must be a non-empty string"
        )

    generator = value.get("generator")
    if generator is not None:
        generator = _require_mapping(generator, f"{context}.generator")
        name = generator.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context}.generator.name must be a non-empty string")
        version = generator.get("version")
        if version is not None and (not isinstance(version, str) or not version):
            raise ValueError(
                f"{context}.generator.version must be a non-empty string"
            )

    execution = value.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError(f"{context}.execution must be an object")
    ratio = execution.get("ratio")
    if ratio is not None and not isinstance(ratio, (int, float)):
        raise ValueError(f"{context}.execution.ratio must be a number")
    sampled = execution.get("sampled")
    if sampled is not None and (not isinstance(sampled, int) or sampled <= 0):
        raise ValueError(f"{context}.execution.sampled must be a positive integer")
    seed = execution.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ValueError(f"{context}.execution.seed must be an integer")

    if "index" not in value:
        raise ValueError(f"{context}.index is required")
    _validate_index_node(value["index"], f"{context}.index")
    return value


def validate_output(
    output: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(output, list):
        for i, entry in enumerate(output):
            _validate_output_dataset(entry, f"output[{i}]")
        return output

    _validate_output_dataset(output, "output")
    return output


def validate_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    config_data = read_metadata_json(dataset_path, CONFIG_FILENAME)
    head_data = read_metadata_json(dataset_path, DATASET_HEAD_FILENAME)
    index_data = read_metadata_json(dataset_path, OUTPUT_FILENAME)

    if config_data is None and head_data is None and index_data is None:
        raise FileNotFoundError(
            f"No {CONFIG_FILENAME}, {DATASET_HEAD_FILENAME}, or {OUTPUT_FILENAME} "
            f"found at: {dataset_path}"
        )

    validated_config: DatasetConfig | None = None
    if config_data is not None:
        validated_config = validate_crawler_config(config_data)

    if head_data is not None:
        parse_dataset_head(head_data, context="dataset_head")

    hydrated_output = None
    if index_data is not None:
        _validate_index_artifact(index_data, "index")
        if config_data is None or head_data is None:
            raise FileNotFoundError(
                f"{CONFIG_FILENAME} and {DATASET_HEAD_FILENAME} are required when "
                f"{OUTPUT_FILENAME} is present at: {dataset_path}"
            )
        hydrated_output = load_saved_output(dataset_path)

    return {
        "path": str(dataset_path),
        "has_config": config_data is not None,
        "has_head": head_data is not None,
        "has_output": index_data is not None,
        "has_index": index_data is not None,
        "config": validated_config,
        "head": head_data,
        "index": index_data,
        "output": hydrated_output,
    }


__all__ = [
    "validate_crawler_config",
    "validate_split_artifact",
    "validate_output",
    "validate_dataset",
]
