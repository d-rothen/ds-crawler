"""Shared dataset descriptor and dataset-head helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._euler_modalities import fold_property_namespaces
from .zip_utils import OUTPUT_FILENAME, read_metadata_json

# Keys written by the crawler that are structural, not dataset properties.
_DATASET_STRUCTURAL_KEYS = frozenset({
    "name",
    "path",
    "type",
    "basename_regex",
    "id_regex",
    "path_regex",
    "id_regex_join_char",
    "hierarchy_regex",
    "named_capture_group_value_separator",
    "intrinsics_regex",
    "extrinsics_regex",
    "flat_ids_unique",
    "id_override",
    "output_json",
    "file_extensions",
    "properties",
    "dataset_contract_version",
    "sampled",
    "dataset",
    "path_filters",
})

_PROPERTY_NAMESPACES = frozenset({
    "dataset",
    "euler_loading",
    "euler_train",
    "meta",
})


def extract_dataset_properties(data: dict[str, Any]) -> dict[str, Any]:
    """Return normalized dataset properties from a config or output dict.

    Supports both forms used in this repo:

    - config-style objects with nested ``properties`` and optional
      top-level shorthand namespaces such as ``meta`` / ``euler_train``
    - output-style objects with dataset properties written directly at the
      top level next to structural keys like ``id_regex`` and ``dataset``
    """
    properties = fold_property_namespaces(data, context="dataset")
    for key, value in data.items():
        if key in _DATASET_STRUCTURAL_KEYS or key in _PROPERTY_NAMESPACES:
            continue
        if key not in properties:
            properties[key] = value
    return properties


def get_dataset_properties(
    source: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """Resolve dataset properties from a path, output head, or config dict."""
    if isinstance(source, dict):
        return extract_dataset_properties(source)

    dataset_path = Path(source)
    output_data = read_metadata_json(dataset_path, OUTPUT_FILENAME)
    if output_data is not None:
        if isinstance(output_data, list):
            if len(output_data) != 1:
                raise ValueError(
                    "Expected a single dataset output object, got a list"
                )
            output_data = output_data[0]
        if not isinstance(output_data, dict):
            raise ValueError("output.json must contain a dataset object")
        return extract_dataset_properties(output_data)

    config_data = read_metadata_json(dataset_path, "ds-crawler.json")
    if config_data is not None:
        if not isinstance(config_data, dict):
            raise ValueError("ds-crawler.json must contain a dataset object")
        return extract_dataset_properties(config_data)

    raise FileNotFoundError(
        f"No ds-crawler.json or {OUTPUT_FILENAME} found at: {dataset_path}"
    )


def infer_dataset_file_types(dataset_node: dict[str, Any]) -> list[str]:
    """Collect observed data file types from a crawler dataset tree."""
    file_types: set[str] = set()

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return

        files = node.get("files")
        if isinstance(files, list):
            for entry in files:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                suffix = ""
                if isinstance(path, str):
                    suffix = Path(path).suffix
                token = suffix.lower().lstrip(".")
                if token:
                    file_types.add(token)

        children = node.get("children")
        if isinstance(children, dict):
            for child in children.values():
                _walk(child)

    _walk(dataset_node)
    return sorted(file_types)


@dataclass
class DatasetDescriptor:
    """Minimal description of a dataset.

    Attributes:
        name: Human-readable dataset name.
        path: Root directory (or file) path for the dataset.
        type: Semantic label for the data modality (e.g. "rgb", "depth").
        properties: Arbitrary key-value metadata.
    """

    name: str
    path: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_output(cls, data: dict[str, Any], path: str) -> DatasetDescriptor:
        """Create a descriptor from a single entry in an ``output.json``.

        The crawler output contains structural keys (``id_regex``,
        ``dataset``, etc.) alongside user-defined properties.  This method
        extracts ``name`` and ``type``, treats everything else that is not a
        known structural key as a property, and pairs it with the caller-
        supplied *path* (which is not stored in the output).

        Args:
            data: One element of the list stored in an ``output.json`` file.
            path: Root directory of the dataset (not present in the output).
        """
        return cls(
            name=data["name"],
            path=path,
            type=data.get("type", ""),
            properties=extract_dataset_properties(data),
        )

    @classmethod
    def from_output_file(
        cls, path: str | Path, dataset_root: str,
    ) -> list[DatasetDescriptor]:
        """Load all descriptors from an ``output.json`` file.

        Args:
            path: Path to the ``output.json`` file.
            dataset_root: Root directory to assign to every descriptor.
        """
        with open(path) as f:
            entries = json.load(f)
        return [cls.from_output(entry, path=dataset_root) for entry in entries]
