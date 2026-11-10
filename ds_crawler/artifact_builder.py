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
from .artifacts import build_index_artifact, save_output_artifacts
from .config import (
    CONFIG_FILENAME,
    CRAWLER_CONFIG_KIND,
    CRAWLER_CONFIG_VERSION,
    DatasetConfig,
)
from .parser import index_dataset_from_files
from .traversal import get_files
from .validation import DATASET_INDEX_KIND, DATASET_INDEX_VERSION, validate_output
from .zip_utils import (
    DATASET_HEAD_FILENAME,
    OUTPUT_FILENAME,
    SCOPES_FILENAME,
    _scopes_manifest_with,
    get_metadata_filename,
    validate_metadata_scope,
)


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
    metadata_scope: str | None = None,
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
    if metadata_scope is not None:
        config["metadata_scope"] = validate_metadata_scope(metadata_scope)

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
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Build canonical dataset artifacts from a parameter set and file list."""
    normalized_scope = (
        validate_metadata_scope(metadata_scope)
        if metadata_scope is not None
        else None
    )
    head = build_dataset_head(dataset=dataset, modality=modality, addons=addons)
    config = build_crawler_config(
        head=head,
        indexing=indexing,
        metadata_scope=normalized_scope,
    )

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
    if normalized_scope is not None:
        unscoped_artifact_names = list(artifacts)
        artifacts = {
            get_metadata_filename(
                filename,
                metadata_scope=normalized_scope,
            ): payload
            for filename, payload in artifacts.items()
        }
        artifacts[SCOPES_FILENAME] = _scopes_manifest_with(
            None,
            metadata_scope=normalized_scope,
            filenames=unscoped_artifact_names,
        )

    summary = {
        "dataset_id": output["head"]["dataset"]["id"],
        "dataset_name": output["head"]["dataset"]["name"],
        "modality_key": output["head"]["modality"]["key"],
        "file_count": len(get_files(output)),
    }
    if normalized_scope is not None:
        summary["metadata_scope"] = normalized_scope

    return {
        "artifacts": artifacts,
        "summary": summary,
    }


def build_index_output(
    *,
    head: dict[str, Any],
    index: dict[str, Any],
    indexing: dict[str, Any] | None = None,
    generator: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a validated index output from a dataset head and an index tree.

    Use this when the caller, not a regex crawl, decides IDs and hierarchy, for
    example from a dataset's metadata tables. *index* is a node of the shape
    ``index.json`` stores: ``{"files": [...], "children": {key: node}}``, whose
    file entries carry ``path`` (relative to the dataset root), ``id``,
    ``path_properties``, ``basename_properties`` and optional ``attributes``.

    Without *indexing* the dataset is *prebuilt*: :func:`save_dataset_index`
    records ``source.prebuilt_index_file`` instead of regexes, and loading reads
    the saved index back. *generator* (``{"name", "version"}``) and *execution*
    record provenance in the index artifact.

    Raises ``ValueError`` if the head or the index is invalid.
    """
    output: dict[str, Any] = {
        "contract": {"kind": DATASET_INDEX_KIND, "version": DATASET_INDEX_VERSION},
        "head_file": DATASET_HEAD_FILENAME,
        "head": deepcopy(head),
        "indexing": deepcopy(indexing or {}),
        "index": deepcopy(index),
    }
    if generator is not None:
        output["generator"] = deepcopy(generator)
    if execution is not None:
        output["execution"] = deepcopy(execution)
    validate_output(output)
    validate_index_provenance(output)
    return output


def validate_index_provenance(output: dict[str, Any]) -> None:
    """Check the optional ``generator`` block the index artifact will carry."""
    generator = output.get("generator")
    if generator is None:
        return
    if not isinstance(generator, dict):
        raise ValueError("output.generator must be an object")
    for key in ("name", "version"):
        value = generator.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"output.generator.{key} must be a non-empty string")


def save_dataset_index(
    dataset_path: str | Path,
    output: dict[str, Any],
    *,
    metadata_scope: str | None = None,
    filename: str = OUTPUT_FILENAME,
) -> Path:
    """Validate *output* and write its artifact set below ``.ds_crawler/``.

    Writes ``dataset-head.json``, ``ds-crawler.json`` and the index, under
    ``.ds_crawler/<metadata_scope>/`` when a scope is given (``scopes.json`` is
    kept up to date). *output* is a hydrated index output: one built by
    :func:`build_index_output`, or one returned by ``index_dataset_from_files``
    or ``index_dataset``, whose regexes are then kept in ``ds-crawler.json``.
    The files the index names must already exist below *dataset_path*; this
    writes only metadata. Nothing is written if validation fails.

    Returns the path of the written index (the archive path for a ZIP root).
    """
    validate_output(output)
    validate_index_provenance(output)
    normalized_scope = (
        validate_metadata_scope(metadata_scope) if metadata_scope is not None else None
    )
    return save_output_artifacts(
        Path(dataset_path),
        output,
        filename=filename,
        metadata_scope=normalized_scope,
    )


__all__ = [
    "build_crawler_config",
    "build_dataset_artifacts_from_files",
    "build_dataset_head",
    "build_index_output",
    "save_dataset_index",
]
