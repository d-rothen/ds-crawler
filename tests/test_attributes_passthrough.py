"""Round-trip tests for the per-file ``attributes`` field.

The crawler operations (align/copy/split/extract) treat file entries as
opaque dicts.  These tests pin that behaviour so future refactors that
rebuild entries from scratch get caught.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ds_crawler import (
    DatasetWriter,
    align_datasets,
    copy_dataset,
    create_dataset_splits,
    extract_datasets,
    index_dataset_from_path,
    load_dataset_split,
    split_dataset,
)
from ds_crawler.traversal import filter_index_by_qualified_ids

from .current_helpers import sample_head


def _build_writer_with_attributes(root: Path) -> dict[str, Any]:
    """Create an indexed dataset whose files carry per-file attributes."""
    writer = DatasetWriter(root, head=sample_head(name="Aug", modality="rgb"))
    writer.get_path(
        "/scene:Scene01/0001", "0001.png",
        attributes={"weight": 0.42, "src": "blender", "tags": ["aug"]},
    ).write_bytes(b"data")
    writer.get_path(
        "/scene:Scene01/0002", "0002.png",
        attributes={"weight": 0.91, "src": "houdini"},
    ).write_bytes(b"data")
    return writer.save_index()


def _files_in(node: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in node.get("files", []):
        out.append(f)
    for child in node.get("children", {}).values():
        out.extend(_files_in(child))
    return out


def test_attributes_survive_align_datasets(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    _build_writer_with_attributes(rgb_root)

    aligned = align_datasets({"modality": "rgb", "source": rgb_root})

    assert "0001" in aligned
    assert aligned["0001"]["rgb"]["attributes"] == {
        "weight": 0.42, "src": "blender", "tags": ["aug"],
    }


def test_attributes_survive_copy_dataset_to_directory(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    out_root = tmp_path / "rgb_copy"
    _build_writer_with_attributes(rgb_root)

    copy_dataset(rgb_root, out_root)
    loaded = index_dataset_from_path(out_root)

    files = _files_in(loaded["index"])
    by_id = {f["id"]: f for f in files}
    assert by_id["0001"]["attributes"]["src"] == "blender"
    assert by_id["0002"]["attributes"]["weight"] == 0.91


def test_attributes_survive_copy_dataset_to_zip(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    out_zip = tmp_path / "rgb_copy.zip"
    _build_writer_with_attributes(rgb_root)

    copy_dataset(rgb_root, out_zip)
    loaded = index_dataset_from_path(out_zip)

    files = _files_in(loaded["index"])
    by_id = {f["id"]: f for f in files}
    assert by_id["0001"]["attributes"] == {
        "weight": 0.42, "src": "blender", "tags": ["aug"],
    }


def test_attributes_survive_create_and_load_split(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    _build_writer_with_attributes(rgb_root)

    create_dataset_splits(rgb_root, ["train"], [1.0], seed=0)
    loaded = load_dataset_split(rgb_root, "train")

    files = _files_in(loaded["index"])
    by_id = {f["id"]: f for f in files}
    assert by_id["0001"]["attributes"]["src"] == "blender"


def test_attributes_survive_filter_by_qualified_ids(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    out = _build_writer_with_attributes(rgb_root)
    output = index_dataset_from_path(rgb_root)

    filtered = filter_index_by_qualified_ids(
        output, {("scene:Scene01", "0001")},
    )
    files = _files_in(filtered["index"])
    assert len(files) == 1
    assert files[0]["attributes"]["src"] == "blender"
