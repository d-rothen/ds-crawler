"""Validation helpers for crawler config and output objects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import _MODALITY_META_SCHEMAS, CONFIG_FILENAME, DatasetConfig
from .zip_utils import OUTPUT_FILENAME, read_metadata_json

_EULER_TRAIN_ALLOWED_USED_AS: frozenset[str] = frozenset({
    "input",
    "target",
    "condition",
})
_EULER_TRAIN_ALLOWED_KEYS: frozenset[str] = frozenset({
    "used_as",
    "slot",
    "modality_type",
    "hierarchy_scope",
    "applies_to",
})
_SLOT_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+){2,}$")
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def validate_crawler_config(
    config: dict[str, Any], workdir: str | Path | None = None
) -> DatasetConfig:
    """Validate a single ``ds-crawler.json`` object.

    Args:
        config: Dataset config object (same shape as embedded ``ds-crawler.json``).
        workdir: Optional working directory prepended to relative dataset paths.

    Returns:
        Parsed ``DatasetConfig`` when validation succeeds.

    Raises:
        ValueError: If schema/content is invalid.
    """
    if not isinstance(config, dict):
        raise ValueError("Crawler config must be a JSON object")

    try:
        return DatasetConfig.from_dict(config, workdir=workdir)
    except KeyError as exc:
        raise ValueError(f"Crawler config missing required field: {exc}") from exc


def _validate_token(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if not _TOKEN_PATTERN.match(value):
        raise ValueError(
            f"{label} must contain only letters, digits, or underscores"
        )


def _validate_euler_train(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")

    unknown = sorted(set(value.keys()) - _EULER_TRAIN_ALLOWED_KEYS)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {context} key(s): {joined}")

    used_as = value.get("used_as")
    if not isinstance(used_as, str) or not used_as:
        raise ValueError(f"{context}.used_as is required")
    if used_as not in _EULER_TRAIN_ALLOWED_USED_AS:
        allowed = ", ".join(sorted(_EULER_TRAIN_ALLOWED_USED_AS))
        raise ValueError(
            f"{context}.used_as must be one of {{{allowed}}}, got {used_as!r}"
        )

    modality_type = value.get("modality_type")
    _validate_token(modality_type, f"{context}.modality_type")

    slot = value.get("slot")
    if slot is not None:
        if not isinstance(slot, str) or not slot:
            raise ValueError(f"{context}.slot must be a non-empty string")
        if not _SLOT_PATTERN.match(slot):
            raise ValueError(
                f"{context}.slot must match "
                "'segment.segment.segment' (alphanumeric/underscore only)"
            )

    hierarchy_scope = value.get("hierarchy_scope")
    applies_to = value.get("applies_to")
    if used_as == "condition":
        if hierarchy_scope is not None:
            _validate_token(hierarchy_scope, f"{context}.hierarchy_scope")
        if applies_to is not None:
            if not isinstance(applies_to, list):
                raise ValueError(f"{context}.applies_to must be a list of strings")
            if not applies_to:
                raise ValueError(f"{context}.applies_to cannot be empty")
            for i, token in enumerate(applies_to):
                if token == "*":
                    continue
                _validate_token(token, f"{context}.applies_to[{i}]")
    elif hierarchy_scope is not None or applies_to is not None:
        raise ValueError(
            f"{context}.hierarchy_scope and {context}.applies_to are only allowed "
            "when used_as is 'condition'"
        )


def _validate_modality_meta(
    value: dict[str, Any], modality_type: str, context: str,
) -> None:
    schema = _MODALITY_META_SCHEMAS.get(modality_type)
    if schema is None:
        return

    meta = value.get("meta")
    if meta is None or not isinstance(meta, dict):
        required_keys = ", ".join(sorted(schema))
        raise ValueError(
            f"{context}.meta is required for modality_type={modality_type!r} "
            f"and must contain: {required_keys}"
        )

    for key, (expected_type, type_label) in schema.items():
        if key not in meta:
            raise ValueError(
                f"{context}.meta.{key} is required for "
                f"modality_type={modality_type!r}"
            )
        if not isinstance(meta[key], expected_type):
            raise ValueError(f"{context}.meta.{key} must be {type_label}")


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


def _validate_dataset_node(node: Any, context: str) -> None:
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
            _validate_dataset_node(child, f"{context}.children[{key!r}]")

    for camera_key in ("camera_intrinsics", "camera_extrinsics"):
        camera_path = node.get(camera_key)
        if camera_path is None:
            continue
        if not isinstance(camera_path, str) or not camera_path:
            raise ValueError(f"{context}.{camera_key} must be a non-empty string")


def _validate_output_dataset(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")

    for key in ("name", "type", "id_regex", "id_regex_join_char"):
        key_value = value.get(key)
        if not isinstance(key_value, str) or not key_value:
            raise ValueError(f"{context}.{key} must be a non-empty string")

    sampled = value.get("sampled")
    if sampled is not None and (not isinstance(sampled, int) or sampled <= 0):
        raise ValueError(f"{context}.sampled must be a positive integer")

    id_override = value.get("id_override")
    if id_override is not None:
        if not isinstance(id_override, str) or not id_override:
            raise ValueError(f"{context}.id_override must be a non-empty string")

    id_regex = value["id_regex"]
    try:
        compiled_id_regex = re.compile(id_regex)
    except re.error as exc:
        raise ValueError(f"{context}.id_regex is not a valid regex: {exc}") from exc
    if compiled_id_regex.groups == 0:
        raise ValueError(f"{context}.id_regex must contain at least one capture group")

    hierarchy_regex = value.get("hierarchy_regex")
    separator = value.get("named_capture_group_value_separator")
    if separator is not None and not isinstance(separator, str):
        raise ValueError(
            f"{context}.named_capture_group_value_separator must be a string"
        )
    if hierarchy_regex is not None:
        if not isinstance(hierarchy_regex, str) or not hierarchy_regex:
            raise ValueError(f"{context}.hierarchy_regex must be a non-empty string")
        try:
            compiled_hierarchy_regex = re.compile(hierarchy_regex)
        except re.error as exc:
            raise ValueError(
                f"{context}.hierarchy_regex is not a valid regex: {exc}"
            ) from exc
        if compiled_hierarchy_regex.groups == 0:
            raise ValueError(
                f"{context}.hierarchy_regex must contain at least one capture group"
            )
        if compiled_hierarchy_regex.groupindex and not separator:
            raise ValueError(
                f"{context}.named_capture_group_value_separator is required when "
                "hierarchy_regex has named groups"
            )

    if "euler_train" not in value:
        raise ValueError(f"{context}.euler_train is required")
    _validate_euler_train(value["euler_train"], f"{context}.euler_train")

    modality_type = value["euler_train"].get("modality_type", "")
    _validate_modality_meta(value, modality_type, context)

    if "dataset" not in value:
        raise ValueError(f"{context}.dataset is required")
    _validate_dataset_node(value["dataset"], f"{context}.dataset")


def validate_output(output: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    """Validate an ``output.json`` object.

    Accepts either a single dataset output object (the per-dataset format)
    or a list of dataset outputs.

    Args:
        output: Parsed JSON object from ``output.json``.

    Returns:
        The original ``output`` object when validation succeeds.

    Raises:
        ValueError: If the object does not match the expected schema.
    """
    if isinstance(output, list):
        for i, entry in enumerate(output):
            _validate_output_dataset(entry, f"output[{i}]")
        return output

    _validate_output_dataset(output, "output")
    return output


def validate_dataset(path: str | Path) -> dict[str, Any]:
    """Validate metadata files found for a dataset path.

    Looks for ``ds-crawler.json`` and ``output.json`` using the same
    lookup order as crawler internals:
    1) ``.ds_crawler/{filename}``
    2) ``{filename}`` at dataset root

    Works for both directories and ``.zip`` datasets.

    Args:
        path: Dataset root directory or ``.zip`` archive path.

    Returns:
        A dict containing what was found and validated:
        ``{"path", "has_config", "has_output", "config", "output"}``.

    Raises:
        FileNotFoundError: If neither metadata file exists.
        ValueError: If a found metadata object is invalid.
    """
    dataset_path = Path(path)
    config_data = read_metadata_json(dataset_path, CONFIG_FILENAME)
    output_data = read_metadata_json(dataset_path, OUTPUT_FILENAME)

    if config_data is None and output_data is None:
        raise FileNotFoundError(
            f"No {CONFIG_FILENAME} or {OUTPUT_FILENAME} found at: {dataset_path}"
        )

    validated_config: DatasetConfig | None = None
    if config_data is not None:
        validated_config = validate_crawler_config(config_data)

    if output_data is not None:
        validate_output(output_data)

    return {
        "path": str(dataset_path),
        "has_config": config_data is not None,
        "has_output": output_data is not None,
        "config": validated_config,
        "output": output_data,
    }


__all__ = [
    "validate_crawler_config",
    "validate_output",
    "validate_dataset",
]
