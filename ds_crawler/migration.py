"""Metadata migration helpers for legacy ds-crawler datasets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ._dataset_contract import DATASET_CONTRACT_VERSION, parse_dataset_head
from .config import (
    CONFIG_FILENAME,
    CRAWLER_CONFIG_KIND,
    CRAWLER_CONFIG_VERSION,
    DatasetConfig,
)
from .validation import DATASET_INDEX_KIND, DATASET_INDEX_VERSION, validate_output
from .zip_utils import (
    DATASET_HEAD_FILENAME,
    OUTPUT_FILENAME,
    list_metadata_json_filenames,
    read_metadata_json,
    write_metadata_json,
)


_LEGACY_STRUCTURAL_KEYS = frozenset({
    "dataset_contract_version",
    "name",
    "path",
    "type",
    "basename_regex",
    "id_regex",
    "path_regex",
    "id_regex_join_char",
    "hierarchy_regex",
    "named_capture_group_value_separator",
    "flat_ids_unique",
    "id_override",
    "path_filters",
    "output_json",
    "file_extensions",
    "sampled",
    "dataset",
})


def _slugify(value: str) -> str:
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "_"
        for ch in value.strip()
    )
    cleaned = "_".join(token for token in cleaned.split("_") if token)
    return cleaned or "dataset"


def _unwrap_single_output(value: Any, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{context} must contain exactly one dataset entry")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _is_new_config(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("contract"), dict)
        and value["contract"].get("kind") == CRAWLER_CONFIG_KIND
    )


def _is_new_output(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("contract"), dict)
        and value["contract"].get("kind") == DATASET_INDEX_KIND
    )


def _legacy_properties(value: dict[str, Any]) -> dict[str, Any]:
    raw_properties = value.get("properties", {})
    if raw_properties is None:
        raw_properties = {}
    if not isinstance(raw_properties, dict):
        raise ValueError("legacy properties must be an object")

    properties = dict(raw_properties)
    for key in ("meta", "euler_train", "euler_loading"):
        if key in value and key not in properties:
            properties[key] = value[key]

    for key, item in value.items():
        if key in _LEGACY_STRUCTURAL_KEYS or key in properties:
            continue
        properties[key] = item
    return properties


def _normalize_legacy_addons(
    properties: dict[str, Any],
    *,
    modality_key: str,
) -> dict[str, dict[str, Any]]:
    addons: dict[str, dict[str, Any]] = {}
    for key in sorted(list(properties.keys())):
        if key not in ("euler_train", "euler_loading") and not key.startswith("euler_"):
            continue
        value = properties.pop(key)
        if not isinstance(value, dict):
            continue
        payload = dict(value)
        payload.setdefault("version", DATASET_CONTRACT_VERSION)
        legacy_modality_type = payload.pop("modality_type", None)
        if legacy_modality_type is not None and legacy_modality_type != modality_key:
            raise ValueError(
                f"Cannot migrate {key}.modality_type={legacy_modality_type!r} "
                f"for modality.key={modality_key!r}"
            )
        addons[key] = payload
    return addons


def _build_head_from_legacy(
    dataset_path: Path,
    *,
    legacy_config: dict[str, Any] | None,
    legacy_output: dict[str, Any] | None,
) -> dict[str, Any]:
    source = legacy_output or legacy_config
    if source is None:
        raise ValueError("Cannot migrate dataset head without legacy metadata")

    properties = _legacy_properties(source)
    name = source.get("name") or dataset_path.stem
    if not isinstance(name, str) or not name:
        raise ValueError("Legacy metadata is missing a usable name")

    modality_key = source.get("type")
    if not isinstance(modality_key, str) or not modality_key:
        euler_train = properties.get("euler_train")
        if isinstance(euler_train, dict):
            maybe_type = euler_train.get("modality_type")
            if isinstance(maybe_type, str) and maybe_type:
                modality_key = maybe_type
    if not isinstance(modality_key, str) or not modality_key:
        raise ValueError("Legacy metadata is missing a usable modality type")

    meta = properties.pop("meta", None)

    dataset_attributes: dict[str, Any] = {}
    explicit_attributes = properties.pop("dataset", None)
    if explicit_attributes is not None:
        if not isinstance(explicit_attributes, dict):
            raise ValueError("Legacy properties.dataset must be an object when present")
        dataset_attributes.update(explicit_attributes)
    dataset_attributes.update(properties)

    addons = _normalize_legacy_addons(dataset_attributes, modality_key=modality_key)

    return parse_dataset_head(
        {
            "contract": {
                "kind": "dataset_head",
                "version": DATASET_CONTRACT_VERSION,
            },
            "dataset": {
                "id": _slugify(name),
                "name": name,
                "attributes": dataset_attributes,
            },
            "modality": {
                "key": modality_key,
                "meta": meta,
            },
            "addons": addons,
        },
        context="dataset_head",
    ).to_mapping()


def _relativize_source_path(dataset_path: Path, value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "."
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        rel = path.relative_to(dataset_path)
    except ValueError:
        return value
    rel_str = str(rel)
    return rel_str or "."


def _build_crawler_config_from_legacy(
    dataset_path: Path,
    *,
    head_file: str,
    legacy_config: dict[str, Any] | None,
    legacy_output: dict[str, Any] | None,
) -> dict[str, Any]:
    source = legacy_config or legacy_output
    if source is None:
        raise ValueError("Cannot migrate crawler config without legacy metadata")

    indexing: dict[str, Any] = {
        "id": {
            "regex": source.get("id_regex"),
            "join_char": source.get("id_regex_join_char", "+"),
        },
        "constraints": {
            "flat_ids_unique": bool(source.get("flat_ids_unique", False)),
        },
    }

    hierarchy_regex = source.get("hierarchy_regex")
    if isinstance(hierarchy_regex, str) and hierarchy_regex:
        indexing["hierarchy"] = {"regex": hierarchy_regex}
        separator = source.get("named_capture_group_value_separator")
        if isinstance(separator, str) and separator:
            indexing["hierarchy"]["separator"] = separator

    properties_cfg: dict[str, Any] = {}
    path_regex = source.get("path_regex")
    if isinstance(path_regex, str) and path_regex:
        properties_cfg["path"] = {"regex": path_regex}
    basename_regex = source.get("basename_regex")
    if isinstance(basename_regex, str) and basename_regex:
        properties_cfg["basename"] = {"regex": basename_regex}
    if properties_cfg:
        indexing["properties"] = properties_cfg

    files_cfg: dict[str, Any] = {}
    file_extensions = source.get("file_extensions")
    if isinstance(file_extensions, list) and file_extensions:
        files_cfg["extensions"] = list(file_extensions)
    path_filters = source.get("path_filters")
    if isinstance(path_filters, dict) and path_filters:
        files_cfg["path_filters"] = deepcopy(path_filters)
    if files_cfg:
        indexing["files"] = files_cfg

    id_override = source.get("id_override")
    if isinstance(id_override, str) and id_override:
        indexing["id"]["override"] = id_override

    source_cfg: dict[str, Any] = {
        "path": _relativize_source_path(dataset_path, source.get("path")),
    }
    legacy_output_json = legacy_config.get("output_json") if legacy_config else None
    if isinstance(legacy_output_json, str) and legacy_output_json:
        source_cfg["prebuilt_index_file"] = legacy_output_json

    config = {
        "contract": {
            "kind": CRAWLER_CONFIG_KIND,
            "version": CRAWLER_CONFIG_VERSION,
        },
        "head_file": head_file,
        "source": source_cfg,
        "indexing": indexing,
    }
    return config


def _strip_legacy_camera_fields(node: Any) -> Any:
    if not isinstance(node, dict):
        return node

    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in {"camera_intrinsics", "camera_extrinsics"}:
            continue
        if key == "children" and isinstance(value, dict):
            result[key] = {
                child_key: _strip_legacy_camera_fields(child_value)
                for child_key, child_value in value.items()
            }
        else:
            result[key] = deepcopy(value)
    return result


def _build_output_from_legacy(
    *,
    head: dict[str, Any],
    legacy_config: dict[str, Any] | None,
    legacy_output: dict[str, Any],
) -> dict[str, Any]:
    if "dataset" not in legacy_output or not isinstance(legacy_output["dataset"], dict):
        raise ValueError("Legacy output.json must contain a dataset object")

    output = {
        "contract": {
            "kind": DATASET_INDEX_KIND,
            "version": DATASET_INDEX_VERSION,
        },
        "head_file": DATASET_HEAD_FILENAME,
        "head": deepcopy(head),
        "generator": {
            "name": "ds_crawler",
            "version": "migrated",
        },
        "indexing": _build_crawler_config_from_legacy(
            Path("."),
            head_file=DATASET_HEAD_FILENAME,
            legacy_config=legacy_config,
            legacy_output=legacy_output,
        )["indexing"],
        "execution": {},
        "index": _strip_legacy_camera_fields(legacy_output["dataset"]),
    }
    sampled = legacy_output.get("sampled")
    if isinstance(sampled, int) and sampled > 0:
        output["execution"]["sampled"] = sampled
    if not output["execution"]:
        output["execution"] = {}
    return output


def migrate_dataset_metadata(
    dataset_path: str | Path,
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    """Rewrite one legacy dataset's metadata into the new schema."""
    dataset_root = Path(dataset_path)
    raw_config = _unwrap_single_output(
        read_metadata_json(dataset_root, CONFIG_FILENAME),
        CONFIG_FILENAME,
    )
    raw_output = _unwrap_single_output(
        read_metadata_json(dataset_root, OUTPUT_FILENAME),
        OUTPUT_FILENAME,
    )

    legacy_config = None if _is_new_config(raw_config) else raw_config
    legacy_output = None if _is_new_output(raw_output) else raw_output

    if legacy_config is None and legacy_output is None:
        raise FileNotFoundError(
            f"No legacy {CONFIG_FILENAME} or {OUTPUT_FILENAME} found at {dataset_root}"
        )

    head = _build_head_from_legacy(
        dataset_root,
        legacy_config=legacy_config,
        legacy_output=legacy_output,
    )
    config = _build_crawler_config_from_legacy(
        dataset_root,
        head_file=DATASET_HEAD_FILENAME,
        legacy_config=legacy_config,
        legacy_output=legacy_output,
    )
    DatasetConfig.from_dict(
        config,
        dataset_root=dataset_root,
        dataset_head=head,
    )

    write_metadata_json(dataset_root, DATASET_HEAD_FILENAME, head)
    write_metadata_json(dataset_root, CONFIG_FILENAME, config)

    output_written = False
    if legacy_output is not None and write_output:
        output = _build_output_from_legacy(
            head=head,
            legacy_config=legacy_config,
            legacy_output=legacy_output,
        )
        validate_output(output)
        write_metadata_json(dataset_root, OUTPUT_FILENAME, output)
        output_written = True

    migrated_splits: list[str] = []
    for filename in list_metadata_json_filenames(dataset_root):
        if not (
            filename.startswith("split_")
            and filename.endswith(".json")
        ):
            continue
        node = read_metadata_json(dataset_root, filename)
        if node is None:
            continue
        migrated_node = _strip_legacy_camera_fields(node)
        write_metadata_json(dataset_root, filename, migrated_node)
        migrated_splits.append(filename)

    return {
        "path": str(dataset_root),
        "wrote_head": True,
        "wrote_config": True,
        "wrote_output": output_written,
        "migrated_splits": migrated_splits,
    }
