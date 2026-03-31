"""Public helpers for building canonical dataset artifacts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from ._dataset_contract import (
    DATASET_CONTRACT_VERSION,
    build_default_meta,
    parse_dataset_head,
)
from .artifacts import build_index_artifact
from .config import CONFIG_FILENAME, CRAWLER_CONFIG_KIND, CRAWLER_CONFIG_VERSION, DatasetConfig
from .parser import index_dataset_from_files
from .traversal import get_files
from .zip_utils import DATASET_HEAD_FILENAME, OUTPUT_FILENAME


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _slugify(value: str) -> str:
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "_"
        for ch in value.strip()
    )
    cleaned = "_".join(token for token in cleaned.split("_") if token)
    return cleaned or "dataset"


def _reject_unknown_keys(
    value: dict[str, Any],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {context} key(s): {joined}")


def _merge_meta_defaults(
    modality_key: str,
    raw_meta: Any,
) -> dict[str, Any] | None:
    defaults = build_default_meta(modality_key)
    if raw_meta is None:
        return defaults
    if not isinstance(raw_meta, dict):
        raise ValueError("modality.meta must be an object")
    if defaults is None:
        return dict(raw_meta)
    merged = dict(defaults)
    merged.update(raw_meta)
    return merged


def build_dataset_head(
    *,
    dataset: dict[str, Any],
    modality: dict[str, Any],
    addons: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a canonical ``dataset-head.json`` mapping."""
    dataset = _require_mapping(dataset, "dataset")
    modality = _require_mapping(modality, "modality")
    addons = {} if addons is None else _require_mapping(addons, "addons")

    _reject_unknown_keys(dataset, allowed={"id", "name", "attributes"}, context="dataset")
    _reject_unknown_keys(modality, allowed={"key", "meta"}, context="modality")

    dataset_name = dataset.get("name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset.name must be a non-empty string")

    dataset_id = dataset.get("id")
    if dataset_id is None:
        dataset_id = _slugify(dataset_name)
    elif not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset.id must be a non-empty string when provided")

    attributes = dataset.get("attributes", {})
    if attributes is None:
        attributes = {}
    if not isinstance(attributes, dict):
        raise ValueError("dataset.attributes must be an object")

    modality_key = modality.get("key")
    if not isinstance(modality_key, str) or not modality_key:
        raise ValueError("modality.key must be a non-empty string")

    head = {
        "contract": {
            "kind": "dataset_head",
            "version": DATASET_CONTRACT_VERSION,
        },
        "dataset": {
            "id": dataset_id,
            "name": dataset_name,
        },
        "modality": {
            "key": modality_key,
            "meta": _merge_meta_defaults(modality_key, modality.get("meta")),
        },
        "addons": deepcopy(addons),
    }
    if attributes:
        head["dataset"]["attributes"] = deepcopy(attributes)
    if not head["addons"]:
        head.pop("addons")

    return parse_dataset_head(head, context="dataset_head").to_mapping()


def build_crawler_config(
    *,
    head: dict[str, Any],
    indexing: dict[str, Any],
    source_path: str = ".",
    head_file: str = DATASET_HEAD_FILENAME,
    prebuilt_index_file: str | None = None,
) -> dict[str, Any]:
    """Build and validate a canonical ``ds-crawler.json`` mapping."""
    indexing = _require_mapping(indexing, "indexing")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("source_path must be a non-empty string")
    if not isinstance(head_file, str) or not head_file:
        raise ValueError("head_file must be a non-empty string")
    if prebuilt_index_file is not None and (
        not isinstance(prebuilt_index_file, str) or not prebuilt_index_file
    ):
        raise ValueError("prebuilt_index_file must be a non-empty string when provided")

    config: dict[str, Any] = {
        "contract": {
            "kind": CRAWLER_CONFIG_KIND,
            "version": CRAWLER_CONFIG_VERSION,
        },
        "head_file": head_file,
        "source": {
            "path": source_path,
        },
        "indexing": deepcopy(indexing),
    }
    if prebuilt_index_file is not None:
        config["source"]["prebuilt_index_file"] = prebuilt_index_file

    runtime_config = dict(config)
    runtime_config["head"] = deepcopy(head)
    dataset_config = DatasetConfig.from_dict(
        runtime_config,
        dataset_root=Path(source_path),
        dataset_head=head,
    )
    return dataset_config.to_crawler_mapping()


def build_dataset_artifacts_from_files(
    *,
    dataset: dict[str, Any],
    modality: dict[str, Any],
    indexing: dict[str, Any],
    files: Iterable[str | Path],
    addons: dict[str, Any] | None = None,
    base_path: str | Path | None = None,
    strict: bool = False,
    sample: int | None = None,
    match_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical dataset artifacts from a parameter set and file list."""
    head = build_dataset_head(dataset=dataset, modality=modality, addons=addons)
    config = build_crawler_config(head=head, indexing=indexing)

    runtime_config = dict(config)
    runtime_config["head"] = deepcopy(head)
    output = index_dataset_from_files(
        runtime_config,
        files,
        base_path=base_path,
        strict=strict,
        sample=sample,
        match_index=match_index,
    )

    artifacts = {
        DATASET_HEAD_FILENAME: deepcopy(output["head"]),
        CONFIG_FILENAME: config,
        OUTPUT_FILENAME: build_index_artifact(output),
    }
    return {
        "artifacts": artifacts,
        "summary": {
            "dataset_id": output["head"]["dataset"]["id"],
            "dataset_name": output["head"]["dataset"]["name"],
            "modality_key": output["head"]["modality"]["key"],
            "file_count": len(get_files(output)),
        },
    }


__all__ = [
    "build_crawler_config",
    "build_dataset_artifacts_from_files",
    "build_dataset_head",
]
