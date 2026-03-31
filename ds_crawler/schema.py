"""Shared dataset descriptor and dataset-head helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._dataset_contract import DATASET_HEAD_KIND, DatasetHeadContract, parse_dataset_head
from .config import CONFIG_FILENAME
from .zip_utils import DATASET_HEAD_FILENAME, read_metadata_json


def _extract_head_mapping(source: dict[str, Any]) -> dict[str, Any]:
    if "head" in source:
        head = source["head"]
        if not isinstance(head, dict):
            raise ValueError("output.head must be an object")
        return head

    contract = source.get("contract")
    if isinstance(contract, dict) and contract.get("kind") == DATASET_HEAD_KIND:
        return source

    raise ValueError("mapping does not contain a dataset head")


def extract_dataset_properties(data: dict[str, Any]) -> dict[str, Any]:
    """Return normalized dataset properties from a head or output dict."""
    return get_dataset_contract(data).to_properties_dict()


def _read_single_dataset_head(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return _extract_head_mapping(source)

    dataset_path = Path(source)
    head_data = read_metadata_json(dataset_path, DATASET_HEAD_FILENAME)
    if head_data is not None:
        if not isinstance(head_data, dict):
            raise ValueError(f"{DATASET_HEAD_FILENAME} must contain a JSON object")
        return head_data

    config_data = read_metadata_json(dataset_path, CONFIG_FILENAME)
    if isinstance(config_data, dict):
        head_file = config_data.get("head_file")
        if isinstance(head_file, str) and head_file and head_file != DATASET_HEAD_FILENAME:
            custom_head = read_metadata_json(dataset_path, head_file)
            if custom_head is not None:
                if not isinstance(custom_head, dict):
                    raise ValueError(f"{head_file} must contain a JSON object")
                return custom_head

    raise FileNotFoundError(
        f"No {DATASET_HEAD_FILENAME} found at: {dataset_path}"
    )


def get_dataset_contract(
    source: str | Path | dict[str, Any],
) -> DatasetHeadContract:
    """Resolve a normalized dataset-head contract from a path or mapping."""
    data = _read_single_dataset_head(source)
    return parse_dataset_head(data, context="dataset_head")


def get_dataset_properties(
    source: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """Resolve dataset properties from a path, head, or output dict."""
    return get_dataset_contract(source).to_properties_dict()


def infer_dataset_file_types(index_node: dict[str, Any]) -> list[str]:
    """Collect observed data file types from an index tree."""
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

    _walk(index_node)
    return sorted(file_types)


@dataclass
class DatasetDescriptor:
    """Minimal description of a dataset."""

    name: str
    path: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_output(cls, data: dict[str, Any], path: str) -> "DatasetDescriptor":
        contract = get_dataset_contract(data)
        return cls(
            name=contract.name,
            path=path,
            type=contract.type,
            properties=contract.to_properties_dict(),
        )

    @classmethod
    def from_output_file(
        cls,
        path: str | Path,
        dataset_root: str,
    ) -> list["DatasetDescriptor"]:
        with open(path) as f:
            entries = json.load(f)
        return [cls.from_output(entry, path=dataset_root) for entry in entries]
