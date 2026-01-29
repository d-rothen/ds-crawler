"""Shared dataset descriptor schema.

This module defines the base ``DatasetDescriptor`` dataclass that captures
the identity of a dataset (name, path, type, and arbitrary properties).
Other packages (e.g. evaluation, dataloading) can import and reuse this
descriptor without depending on crawler-specific configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keys written by the crawler's _build_output that are not user properties.
_OUTPUT_KEYS = frozenset({
    "name",
    "type",
    "id_regex",
    "id_regex_join_char",
    "dataset",
    "hierarchy_regex",
    "named_capture_group_value_separator",
})


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
            properties={k: v for k, v in data.items() if k not in _OUTPUT_KEYS},
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
