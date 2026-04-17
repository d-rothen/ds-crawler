from __future__ import annotations

from pathlib import Path

import pytest

from ds_crawler import (
    DatasetWriter,
    copy_dataset_splits,
    create_aligned_dataset_splits,
    create_dataset_splits,
    list_dataset_splits,
    load_dataset_split,
)
from ds_crawler.traversal import collect_qualified_ids
from ds_crawler.zip_utils import read_metadata_json

from .current_helpers import sample_head


def _make_dataset(root: Path, ids: list[str], *, modality: str = "rgb") -> Path:
    writer = DatasetWriter(root, head=sample_head(name=root.name, modality=modality))
    for file_id in ids:
        writer.get_path(f"/scene/{file_id}", f"{file_id}.png").write_bytes(b"data")
    writer.save_index()
    return root


def test_create_dataset_splits_lists_and_loads_split(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="RGB"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.get_path("/scene/0002", "0002.png").write_bytes(b"data")
    writer.get_path("/scene/0003", "0003.png").write_bytes(b"data")
    writer.save_index()

    create_dataset_splits(root, ["train", "val"], [50, 50], seed=1)
    loaded = load_dataset_split(root, "train")

    assert list_dataset_splits(root) == ["train", "val"]
    assert loaded["split"]["name"] == "train"
    assert loaded["split"]["source_index_file"] == "index.json"


def test_create_aligned_dataset_splits_reuses_same_partition(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root, modality in ((root_a, "rgb"), (root_b, "depth")):
        writer = DatasetWriter(root, head=sample_head(name=root.name, modality=modality))
        writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
        writer.get_path("/scene/0002", "0002.png").write_bytes(b"data")
        writer.save_index()

    result = create_aligned_dataset_splits(
        [root_a, root_b],
        ["train", "val"],
        [50, 50],
        seed=2,
    )

    loaded_a = load_dataset_split(root_a, "train")
    loaded_b = load_dataset_split(root_b, "train")

    assert collect_qualified_ids(loaded_a) == result["qualified_id_splits"][0]
    assert collect_qualified_ids(loaded_b) == result["qualified_id_splits"][0]


def test_copy_dataset_splits_replicates_partition(tmp_path: Path) -> None:
    ids = ["0001", "0002", "0003", "0004"]
    source = _make_dataset(tmp_path / "rgb", ids, modality="rgb")
    target = _make_dataset(tmp_path / "depth", ids, modality="depth")

    create_dataset_splits(source, ["train", "val"], [50, 50], seed=7)

    result = copy_dataset_splits(source, target)

    assert {entry["split"] for entry in result["splits"]} == {"train", "val"}
    assert list_dataset_splits(target) == ["train", "val"]

    source_train = load_dataset_split(source, "train")
    target_train = load_dataset_split(target, "train")
    assert collect_qualified_ids(source_train) == collect_qualified_ids(target_train)

    source_val = load_dataset_split(source, "val")
    target_val = load_dataset_split(target, "val")
    assert collect_qualified_ids(source_val) == collect_qualified_ids(target_val)


def test_copy_dataset_splits_records_provenance(tmp_path: Path) -> None:
    ids = ["0001", "0002"]
    source = _make_dataset(tmp_path / "rgb", ids, modality="rgb")
    target = _make_dataset(tmp_path / "depth", ids, modality="depth")

    create_dataset_splits(source, ["test"], [100], seed=1)
    copy_dataset_splits(source, target)

    written = read_metadata_json(target, "split_test.json")
    assert written is not None
    copied_from = written["execution"]["copied_from"]
    assert copied_from["source"] == str(source)
    assert copied_from["split"] == "test"


def test_copy_dataset_splits_subset_by_name(tmp_path: Path) -> None:
    ids = ["0001", "0002", "0003", "0004"]
    source = _make_dataset(tmp_path / "rgb", ids)
    target = _make_dataset(tmp_path / "depth", ids, modality="depth")

    create_dataset_splits(source, ["train", "val", "test"], [50, 25, 25], seed=3)

    copy_dataset_splits(source, target, split_names=["test"])

    assert list_dataset_splits(target) == ["test"]


def test_copy_dataset_splits_fails_on_missing_ids(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "rgb", ["0001", "0002", "0003"])
    # Target is missing id "0003".
    target = _make_dataset(tmp_path / "depth", ["0001", "0002"], modality="depth")

    create_dataset_splits(source, ["train"], [100], seed=0)

    with pytest.raises(ValueError, match="have no match on the target"):
        copy_dataset_splits(source, target)

    # Nothing was written.
    assert list_dataset_splits(target) == []


def test_copy_dataset_splits_blocks_existing_without_override(tmp_path: Path) -> None:
    ids = ["0001", "0002"]
    source = _make_dataset(tmp_path / "rgb", ids)
    target = _make_dataset(tmp_path / "depth", ids, modality="depth")

    create_dataset_splits(source, ["train"], [100], seed=0)
    create_dataset_splits(target, ["train"], [100], seed=99)

    with pytest.raises(ValueError, match="already has splits"):
        copy_dataset_splits(source, target)


def test_copy_dataset_splits_overrides_when_allowed(tmp_path: Path) -> None:
    ids = ["0001", "0002", "0003", "0004"]
    source = _make_dataset(tmp_path / "rgb", ids)
    target = _make_dataset(tmp_path / "depth", ids, modality="depth")

    create_dataset_splits(source, ["train"], [50], seed=1)
    create_dataset_splits(target, ["train"], [50], seed=99)

    source_train_ids = collect_qualified_ids(load_dataset_split(source, "train"))
    target_train_ids_before = collect_qualified_ids(load_dataset_split(target, "train"))
    # Different seed → different partition on the target before override.
    assert source_train_ids != target_train_ids_before

    result = copy_dataset_splits(source, target, override=True)

    assert result["splits"][0]["overridden"] is True
    target_train_ids_after = collect_qualified_ids(load_dataset_split(target, "train"))
    assert target_train_ids_after == source_train_ids


def test_copy_dataset_splits_errors_on_unknown_source_split(tmp_path: Path) -> None:
    ids = ["0001", "0002"]
    source = _make_dataset(tmp_path / "rgb", ids)
    target = _make_dataset(tmp_path / "depth", ids, modality="depth")

    create_dataset_splits(source, ["train"], [100], seed=0)

    with pytest.raises(FileNotFoundError, match="no splits named"):
        copy_dataset_splits(source, target, split_names=["test"])


def test_copy_dataset_splits_errors_when_source_has_no_splits(tmp_path: Path) -> None:
    ids = ["0001"]
    source = _make_dataset(tmp_path / "rgb", ids)
    target = _make_dataset(tmp_path / "depth", ids, modality="depth")

    with pytest.raises(FileNotFoundError, match="No inline splits"):
        copy_dataset_splits(source, target)
